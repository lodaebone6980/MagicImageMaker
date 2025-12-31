import streamlit as st
import requests
import json
import time
import os
import re
import shutil
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from google import genai
from google.genai import types

# [NEW] 오디오 처리를 위한 라이브러리 추가
from pydub import AudioSegment
from pydub.silence import detect_silence

# [NEW] 동영상 생성을 위한 라이브러리 및 효과 추가
try:
    # VideoFileClip, concatenate_videoclips 추가 (영상 병합용)
    from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, VideoFileClip, concatenate_videoclips
    import numpy as np 
except ImportError:
    st.error("⚠️ 'moviepy' 라이브러리가 없습니다. 터미널에 'pip install moviepy numpy'를 입력하세요.")
    st.stop()

# ==========================================
# [설정] 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="이미지 생성기", layout="wide", page_icon="🎨")

# 파일 저장 경로 설정
BASE_PATH = "./web_result_files"
IMAGE_OUTPUT_DIR = os.path.join(BASE_PATH, "output_images")
AUDIO_OUTPUT_DIR = os.path.join(BASE_PATH, "output_audio")
VIDEO_OUTPUT_DIR = os.path.join(BASE_PATH, "output_video") 

# 텍스트 모델 설정
GEMINI_TEXT_MODEL_NAME = "gemini-2.5-pro" 

# [기본값] 문서에 명시된 정확한 호스트 주소
DEFAULT_SUPERTONE_URL = "https://supertoneapi.com"
DEFAULT_VOICE_ID = "ff700760946618e1dcf7bd" 

# ==========================================
# [함수] 0. TTS용 텍스트 정규화 (숫자/기호 -> 한글)
# ==========================================
def num_to_kor(num_str):
    """숫자 문자열을 한글 발음으로 변환 (예: 1,500 -> 천오백)"""
    try:
        num_str = num_str.replace(',', '')
        if not num_str.isdigit(): return num_str
        
        num = int(num_str)
        if num == 0: return "영"
        
        units = ['', '십', '백', '천']
        big_units = ['', '만', '억', '조', '경']
        num_chars = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
        
        result = []
        big_idx = 0
        
        while num > 0:
            small_part = num % 10000
            if small_part > 0:
                small_res = []
                small_idx = 0
                while small_part > 0:
                    digit = small_part % 10
                    if digit > 0:
                        unit = units[small_idx]
                        char = num_chars[digit]
                        if digit == 1 and small_idx > 0:
                            char = ""
                        small_res.append(char + unit)
                    small_part //= 10
                    small_idx += 1
                result.append("".join(reversed(small_res)) + big_units[big_idx])
            num //= 10000
            big_idx += 1
            
        return "".join(reversed(result))
    except:
        return num_str

def normalize_text_for_tts(text):
    """TTS 발음을 위해 특수문자와 숫자를 한글로 변환"""
    text = text.replace("%", " 퍼센트")
    
    def replace_decimal(match):
        return f"{match.group(1)} 점 {match.group(2)}"
    text = re.sub(r'(\d+)\.(\d+)', replace_decimal, text)

    def replace_number(match):
        return num_to_kor(match.group())
    
    text = re.sub(r'\d+(?:,\d+)*', replace_number, text)
    
    return text

# ==========================================
# [함수] 1. 대본 구조화 로직
# ==========================================
def generate_structure(client, full_script):
    """Gemini를 이용해 대본 구조화"""
    prompt = f"""
    [Role]
    You are a professional YouTube Content Editor and Scriptwriter.

    [Task]
    Analyze the provided transcript (script).
    Restructure the content into a highly detailed, list-style format suitable for a blog post or a new video plan.
      
    [Output Format]
    1. **Video Theme/Title**: (Extract or suggest a catchy title based on the whole script)
    2. **Intro**: (Hook and background, no music) Approve specific channel names, The intro hooks the overall topic (안녕하십니까 같은 인사 금지)
    3. **Chapter 1** to **Chapter 8**: (Divide the main content into logical sections. Use detailed bullet points for each chapter.)
    4. **Epilogue**: (Conclusion and Subscribe Like Comments that make you anticipate the next specific content)

    [Constraint]
    - Analyze the entire context deeply.
    - Write the output in **Korean**.
    - Make the content rich and detailed.
    - If the original script has a channel name, remove it.

    [Transcript]
    {full_script}
    """
    
    try:
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# [함수] 2. 섹션별 대본 생성 (수정 버전)
# ==========================================
def generate_section(client, section_title, full_structure, duration_type="fixed", custom_instruction=""):
    # 1. 분량에 따른 글자수 및 지침 설정
    if duration_type == "2min":
        target_chars = "약 1,000자 (공백 포함)"
        detail_level = "핵심 내용 위주로 명확하게 전달하되, 너무 짧지 않게 서술하십시오."
    elif duration_type == "3min":
        target_chars = "약 1,500자 (공백 포함)"
        detail_level = "충분한 예시와 설명을 곁들여 상세하게 서술하십시오."
    elif duration_type == "4min":
        target_chars = "약 2,000자 이상 (공백 포함)"
        detail_level = "현미경으로 들여다보듯 아주 깊이 있고 디테일하게 묘사하십시오. 절대 요약하지 마십시오."
    else: # Intro / Epilogue (Fixed)
        target_chars = "약 400단어 (약 1,400자)"
        detail_level = "시청자를 사로잡는 강력한 후킹과 여운을 주는 마무리로 작성하십시오. 안녕 인사는 하지 않는다"

    user_guide_prompt = ""
    if custom_instruction:
        user_guide_prompt = f"""
    [User's Special Direction]
    The user has provided specific instructions for the tone/style. You MUST follow this:
    👉 "{custom_instruction}"
        """

    prompt = f"""
    [Role]
    당신은 대한민국 최고의 유튜브 다큐멘터리 작가입니다.

    [Task]
    전체 대본 구조 중 오직 **"{section_title}"** 부분만 작성하십시오.
      
    [Context (Overall Structure)]
    {full_structure}
    {user_guide_prompt}

    [Target Section]
    **{section_title}**

    [Length Constraints]
    - **목표 분량: {target_chars}** - **작성 지침:** {detail_level}
      
    [Style Guidelines - 매우 중요]
    1. '습니다' 체를 사용하고, 다큐멘터리 특유의 진지하고 몰입감 있는 어조를 유지하세요.
    2. 앞뒤 문맥(이전 챕터, 다음 챕터)을 고려하되, 이 파트의 내용에만 집중하세요.
    3. (지문), (효과음) 같은 연출 지시어는 제외하고 **오직 나레이션 대사만** 출력하세요.
    4. 서두에 "네, 알겠습니다" 같은 잡담을 하지 말고 바로 대본 내용을 시작하세요.
    5. 영문 병기(괄호)는 하지 마세요. 깔끔하게 한글만.
    6. 쉼표와 접속어 등을 사용하여, 리듬이 있지만 너무 끊기지 않는 흐름을 만들 것.
    
    # [수정됨] 흐름 끊김 방지 핵심 지침 --------------------------
    7. **[금지사항]** 글의 마지막에 "다음 장에서는...", "이어서...", "이제 ~를 알아보겠습니다" 같은 **예고성 멘트를 절대 쓰지 마십시오.**
    8. **[금지사항]** "지금까지 ~를 알아보았습니다" 같은 **중간 정리 멘트도 절대 쓰지 마십시오.**
    9. 이 텍스트들은 나중에 하나로 합쳐질 것입니다. 따라서 **그냥 설명하다가 자연스럽게 문장(마침표)으로 끝내십시오.** 뒷내용은 다음 텍스트가 자연스럽게 이어받습니다.
    10. 챕터 번호나 소제목을 본문에 다시 적지 마십시오. 오직 내용만 서술하십시오.
    ---------------------------------------------------------

    [Output]
    (지금 바로 {section_title}의 원고를 작성 시작하세요)
    """
    
    try:
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL_NAME, 
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.75 
            )
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# [함수] 3. 이미지 생성 관련 로직
# ==========================================

def init_folders():
    for path in [IMAGE_OUTPUT_DIR, AUDIO_OUTPUT_DIR, VIDEO_OUTPUT_DIR]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

def split_script_by_time(script, chars_per_chunk=100):
    temp_sentences = script.replace(".", ".|").replace("?", "?|").replace("!", "!|").split("|")
    chunks = []
    current_chunk = ""
    for sentence in temp_sentences:
        sentence = sentence.strip()
        if not sentence: continue
        if len(current_chunk) + len(sentence) < chars_per_chunk:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def parse_numbered_script(script):
    """
    번호(1. 2. 3.)로 분할된 대본을 파싱하여 씬 리스트로 반환.
    줄바꿈을 제거하고 문장을 조화롭게 연결합니다.
    """
    import re

    scenes = []
    lines = script.strip().split('\n')
    current_scene_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 새로운 씬 시작인지 확인 (숫자 + 점으로 시작)
        match = re.match(r'^(\d+)\.(.*)$', line)
        if match:
            # 이전 씬이 있으면 저장
            if current_scene_lines:
                scene_text = ' '.join(current_scene_lines)
                # 연속 공백 제거
                scene_text = re.sub(r'\s+', ' ', scene_text).strip()
                if scene_text:
                    scenes.append(scene_text)

            # 새 씬 시작 (번호 뒤의 내용)
            rest = match.group(2).strip()
            current_scene_lines = [rest] if rest else []
        else:
            # 현재 씬에 라인 추가
            current_scene_lines.append(line)

    # 마지막 씬 저장
    if current_scene_lines:
        scene_text = ' '.join(current_scene_lines)
        scene_text = re.sub(r'\s+', ' ', scene_text).strip()
        if scene_text:
            scenes.append(scene_text)

    return scenes

def make_filename(scene_num, text_chunk):
    clean_line = text_chunk.replace("\n", " ").strip()
    clean_line = re.sub(r'[\\/:*?"<>|]', "", clean_line)
    words = clean_line.split()
    
    if len(words) <= 6:
        summary = " ".join(words)
    else:
        start_part = " ".join(words[:3])
        end_part = " ".join(words[-3:])
        summary = f"{start_part}...{end_part}"
    
    filename = f"S{scene_num:03d}_{summary}.png"
    return filename

# ==========================================
# [수정됨] 함수: 프롬프트 생성 (컨셉 기반 통합 버전)
# ==========================================
def generate_prompt(api_key, index, text_chunk, style_instruction, video_title, target_language="Korean"):
    scene_num = index + 1
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TEXT_MODEL_NAME}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}

    # [언어 설정 로직] 선택된 언어에 따라 지침 자동 변경
    if target_language == "Korean":
        lang_guide = "화면 속 글씨는 무조건 한글(Korean)로 표기하십시오. (다른 언어 절대 금지)"
        lang_example = "(예: 'New York' -> '뉴욕', 'Tokyo' -> '도쿄')"
    elif target_language == "English":
        lang_guide = "화면 속 글씨는 무조건 영어(English)로 표기하십시오."
        lang_example = "(예: '서울' -> 'Seoul', '독도' -> 'Dokdo')"
    elif target_language == "Japanese":
        lang_guide = "화면 속 글씨는 무조건 일본어(Japanese)로 표기하십시오."
        lang_example = "(예: '서울' -> 'ソウル', 'New York' -> 'ニューヨーク')"
    else:
        lang_guide = f"화면 속 글씨는 무조건 '{target_language}'로 표기하십시오."
        lang_example = ""

    full_instruction = f"""
[역할]
당신은 '2D 캐릭터 애니메이션 전문 프롬프트 디렉터'입니다.

[전체 영상 주제]
"{video_title}"

[스타일 가이드 - 최우선 준수]
{style_instruction}

[프롬프트 작성 공식]
[카메라 연출(앵글)] + [캐릭터 개성(직업별 의상/컬러) + 구체적 행동] + [장소/시간을 분위기로 녹여냄 + 다채로운 조명] + [고정 문장]

[필수 연출 지침]
1. 무조건 2D 애니메이션 스타일로 그린다

2. 캐릭터 (일관된 스타일 유지):
   - 얼굴 형태: 심플한 동그란 하얀색 원형 (이모티콘처럼 단순한 원형)
   - 얼굴 요소: 점 눈, 작은 입으로 심플하게 표현
   - 표정 스타일: 귀여운 이모티콘 스타일 (^_^, >_<, :D 같은 느낌)
   - 피부: 순수한 하얀색
   - 몸: 슬림한 체형에 하얀색 피부
   - 팔다리: 가늘고 부드러운 하얀색 (일반적인 사람 비율)
   - 의상: 직업/역할에 맞는 컬러풀한 의상

3. 배경: 2D로 디테일하게 몰입감 있게, 다양한 장소와 상황 연출
4. 색감: 에메랄드, 보라, 주황, 핑크, 골드 등 다채로운 색상
5. 하나의 통일된 장면으로 연출
6. 텍스트: 핵심 키워드 2~3개만 자연스럽게 배치
7. 텍스트 언어: {lang_guide} {lang_example}

[임무]
제공된 대본 조각을 바탕으로, 이미지 생성 AI가 그릴 수 있는 구체적인 묘사 프롬프트를 작성하십시오.

[작성 요구사항]
- 분량: 최소 5문장 이상으로 상세하게 묘사
- 포함 요소:
  - 카메라 앵글 지정
  - 캐릭터의 의상, 머리카락, 소품 구체적 묘사
  - 캐릭터의 행동과 감정 표현
  - 대본에 맞는 배경과 조명 분위기
  - 시각적 은유 (추상적 내용일 경우)
  - 핵심 한국어 키워드 2~3개

[출력 형식]
- 순수 텍스트만 출력 (마크다운 강조 **금지**)
- 무조건 한국어로만 작성
- 부가 설명 없이 오직 프롬프트 텍스트만 출력
"""

    # 공통 실행 로직
    payload = {
        "contents": [{"parts": [{"text": f"지시사항(Instruction):\n{full_instruction}\n\n대본 내용(Script Segment):\n\"{text_chunk}\"\n\n이미지 프롬프트 결과:"}]}]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            try:
                prompt = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            except:
                prompt = text_chunk
            return (scene_num, prompt)
        elif response.status_code == 429:
            time.sleep(2)
            return (scene_num, f"일러스트 묘사: {text_chunk}")
        else:
            return (scene_num, f"Error generating prompt: {response.status_code}")
    except Exception as e:
        return (scene_num, f"Error: {e}")

# ==========================================
# [수정됨] generate_image: API 제한(429) 완벽 대응 + 재시도 강화
# ==========================================
def generate_image(client, prompt, filename, output_dir, selected_model_name):
    full_path = os.path.join(output_dir, filename)
    
    # [수정 1] 재시도 횟수를 10회로 늘려서 절대 포기하지 않게 함
    max_retries = 10
    
    # [수정 2] 안전 필터 (기존 유지)
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_ONLY_HIGH"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_ONLY_HIGH"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_ONLY_HIGH"
        ),
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=selected_model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(aspect_ratio="16:9"),
                    safety_settings=safety_settings 
                )
            )
            
            if response.parts:
                for part in response.parts:
                    if part.inline_data:
                        img_data = part.inline_data.data
                        image = Image.open(BytesIO(img_data))
                        image.save(full_path)
                        return full_path
            
            # 응답은 왔으나 이미지가 없는 경우 (필터링 등)
            print(f"⚠️ [시도 {attempt}/{max_retries}] 이미지 데이터 없음. 재시도... ({filename})")
            time.sleep(2)
            
        except Exception as e:
            error_msg = str(e)
            # [상세 에러 로깅 추가]
            print(f"=" * 50)
            print(f"🔴 [에러 상세] {filename}")
            print(f"   시도: {attempt}/{max_retries}")
            print(f"   에러 타입: {type(e).__name__}")
            print(f"   에러 메시지: {error_msg}")
            print(f"=" * 50)

            # [핵심 수정] 429 (Too Many Requests) 또는 429 Resource Exhausted 에러 발생 시
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                wait_time = 30  # 30초 동안 멈췄다가 다시 시도 (분당 제한 초기화 대기)
                print(f"🛑 [API 제한 감지] {filename} - {wait_time}초 대기 후 재시도합니다...")
                time.sleep(wait_time)
            elif "400" in error_msg or "InvalidArgument" in error_msg or "SAFETY" in error_msg.upper():
                # 400 에러 또는 안전 필터 - 상세 로깅
                print(f"🚫 [컨텐츠 거부] {filename} - 프롬프트가 거부됨. 5초 대기 후 재시도...")
                time.sleep(5)
            else:
                # 일반 에러는 5초 대기
                print(f"⚠️ [기타 에러] {filename} - 5초 대기")
                time.sleep(5)
            
    # [최종 실패]
    print(f"❌ [최종 실패] {filename} - 모든 재시도 실패.")
    return None

def create_zip_buffer(source_dir):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.basename(file_path))
    buffer.seek(0)
    return buffer

# ==========================================
# [함수] 4. Supertone TTS 및 오디오 후처리 (Noise Cut - Micro Fade)
# ==========================================
def check_connection_and_get_voices(api_key, base_url):
    """연결 테스트 및 목소리 목록 가져오기"""
    base_url = base_url.rstrip('/')
    url = f"{base_url}/v1/voices"
    headers = {"x-sup-api-key": api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            voices = []
            if isinstance(data, dict) and "items" in data:
                voices = data["items"]
            elif isinstance(data, list):
                voices = data
            else:
                return False, [], f"응답 구조가 예상과 다릅니다. (items 키 없음: {list(data.keys())})"
            
            return True, voices, "✅ 연결 성공!"
            
        elif response.status_code == 401:
            return False, [], "❌ API Key가 틀렸습니다 (401)"
        elif response.status_code == 404:
            return False, [], f"❌ 주소(URL) 오류 (404). {base_url} 이 맞는지 확인하세요."
        else:
            return False, [], f"❌ 서버 오류 ({response.status_code}): {response.text}"
            
    except Exception as e:
        return False, [], f"❌ 네트워크 연결 오류: {str(e)}"

def generate_supertone_tts(api_key, voice_id, text, scene_num, base_url, speed=1.0, pitch=0):
    """Supertone API를 사용해 TTS 오디오 생성 및 저장"""
    
    # 1. 텍스트 정규화 (숫자 -> 한글)
    normalized_text = normalize_text_for_tts(text)

    # 2. 마침표가 있든 없든 무조건 마침표 하나 더 추가하여 확실한 끝맺음 유도
    normalized_text = normalized_text.strip() + "."

    base_url = base_url.rstrip('/')
    url = f"{base_url}/v1/text-to-speech/{voice_id}"
    
    headers = {
        "x-sup-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    safe_text = normalized_text[:500] 
    
    payload = {
        "text": safe_text,
        "language": "ko",
        "model": "sona_speech_1",
        "voice_settings": {
            "speed": float(speed),
            "pitch_shift": int(pitch),
            "pitch_variance": 1
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        
        if response.status_code == 200:
            filename = f"S{scene_num:03d}_audio.wav"
            full_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
            
            # 파일 저장
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            # [KEY FIX] 마이크로 페이드 (Micro-Fade): 끝 30ms만 살짝 줄여서 튀는 소리 제거
            try:
                saved_audio = AudioSegment.from_wav(full_path)
                
                if len(saved_audio) > 100:
                    # 0.03초(30ms)만 페이드 아웃 -> 말끝은 살리고, 기계음 '틱' 소리만 제거
                    saved_audio = saved_audio.fade_out(30)
                    saved_audio.export(full_path, format="wav")
            except Exception as e:
                print(f"Audio tail fix error: {e}")

            return full_path
        elif response.status_code == 404:
            return "VOICE_NOT_FOUND"
        else:
            return f"Error ({response.status_code}): {response.text}"
            
    except Exception as e:
        return f"System Error: {e}"

def smart_shorten_silence(file_path, max_allowed_silence_ms=300, min_silence_len=100, silence_thresh=-40):
    try:
        audio = AudioSegment.from_wav(file_path)
        silence_ranges = detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        if not silence_ranges:
            return True, "무음 구간 없음"

        output_audio = AudioSegment.empty()
        last_pos = 0

        for start, end in silence_ranges:
            output_audio += audio[last_pos:start]
            silence_duration = end - start
            if silence_duration > max_allowed_silence_ms:
                output_audio += AudioSegment.silent(duration=max_allowed_silence_ms)
            else:
                output_audio += audio[start:end]
            last_pos = end

        output_audio += audio[last_pos:]
        output_audio.export(file_path, format="wav")
        return True, "성공"

    except Exception as e:
        return False, str(e)

def process_single_tts_task(api_key, voice_id, text, scene_num, base_url, speed, pitch, apply_silence_trim):
    audio_res = generate_supertone_tts(
        api_key, voice_id, text, scene_num, base_url, speed, pitch
    )
    if "Error" not in str(audio_res) and "VOICE_NOT_FOUND" not in str(audio_res):
        if apply_silence_trim:
            smart_shorten_silence(audio_res, max_allowed_silence_ms=300)
    return audio_res

# ==========================================
# [함수] 5. 비디오 생성 (MoviePy - 고화질 설정)
# ==========================================
def create_video_with_zoom(image_path, audio_path, output_dir, scene_num, is_zoom_in=True):
    """
    이미지와 오디오를 합쳐서 자연스러운 줌 효과 비디오 생성.
    [고화질 설정 적용 완료]
    """
    try:
        output_filename = f"S{scene_num:03d}_video_zoom.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        original_pil = Image.open(image_path).convert("RGB")
        W, H = original_pil.size
        
        if W % 2 != 0: W -= 1
        if H % 2 != 0: H -= 1
        if original_pil.size != (W, H):
            original_pil = original_pil.resize((W, H), Image.LANCZOS)

        max_crop_ratio = 0.85 
        
        def effect(get_frame, t):
            progress = t / duration
            if is_zoom_in:
                current_ratio = 1.0 - (1.0 - max_crop_ratio) * progress
            else:
                current_ratio = max_crop_ratio + (1.0 - max_crop_ratio) * progress
            
            roi_w = W * current_ratio
            roi_h = H * current_ratio
            
            x0 = (W - roi_w) / 2
            y0 = (H - roi_h) / 2
            x1 = x0 + roi_w
            y1 = y0 + roi_h
            
            transformed_img = original_pil.transform(
                (W, H), 
                Image.EXTENT, 
                (x0, y0, x1, y1), 
                Image.BICUBIC 
            )
            return np.array(transformed_img)

        video = ImageClip(np.array(original_pil)).set_duration(duration).set_fps(30)
        video = video.fl(effect)
        video = video.set_audio(audio_clip)
        
        # [KEY FIX] 고화질 렌더링 설정 (비트레이트 8000k, 오디오 192k, preset=slow)
        video.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac", 
            bitrate="8000k",        # 영상 화질 대폭 향상
            audio_bitrate="192k",   # 오디오 음질 향상
            preset="slow",          # 인코딩 품질 향상 (속도는 조금 느려짐)
            logger=None
        )
        
        return output_path
        
    except Exception as e:
        return f"Error: {e}"

def process_single_video_task(item, output_dir, is_zoom_in):
    if item.get('audio_path') and os.path.exists(item['audio_path']):
        return create_video_with_zoom(
            item['path'], 
            item['audio_path'], 
            output_dir, 
            item['scene'], 
            is_zoom_in=is_zoom_in
        )
    return None

def merge_all_videos(video_paths, output_dir):
    try:
        clips = []
        for path in video_paths:
            if path and os.path.exists(path):
                clips.append(VideoFileClip(path))
        
        if not clips:
            return "No clips to merge"

        final_clip = concatenate_videoclips(clips, method="compose")
        final_output_path = os.path.join(output_dir, "FINAL_FULL_VIDEO.mp4")
        
        # [KEY FIX] 병합 시에도 고화질 유지
        final_clip.write_videofile(
            final_output_path, 
            codec="libx264", 
            audio_codec="aac", 
            bitrate="8000k",
            audio_bitrate="192k",
            preset="slow",
            logger=None
        )
        return final_output_path
    except Exception as e:
        return f"Merge Error: {e}"

# ==========================================
# [UI] 사이드바 (자동 로그인 + 장르 선택 적용)
# ==========================================
with st.sidebar:
    st.header("⚙️ 환경 설정")
    
    # 1. Google API Key 자동 로드 (secrets.toml 활용)
    # .streamlit/secrets.toml 파일에 [general] google_api_key = "..." 가 있으면 자동 로드
    if "general" in st.secrets and "google_api_key" in st.secrets["general"]:
        api_key = st.secrets["general"]["google_api_key"]
        st.success("🔑 Google API Key가 로드되었습니다.")
    else:
        api_key = st.text_input("🔑 Google API Key", type="password", help="secrets.toml이 없으면 직접 입력하세요.")

    st.markdown("---")
    
    st.subheader("🖼️ 이미지 모델 선택")
    model_choice = st.radio("사용할 AI 모델:", ("Premium (Gemini 3 Pro)", "Fast (Gemini-2.5-pro)"), index=0)
    
    if "Gemini 3 Pro" in model_choice:
        SELECTED_IMAGE_MODEL = "gemini-3-pro-image-preview" 
    else:
        SELECTED_IMAGE_MODEL = "gemini-2.5-pro"

    st.info(f"✅ 선택 모델: `{SELECTED_IMAGE_MODEL}`")
    
    st.markdown("---")
    st.subheader("📝 대본 분할 안내")
    st.info("대본을 번호(1. 2. 3.)로 분할해서 입력하세요. 각 번호가 하나의 씬이 됩니다.") 
    
    st.markdown("---")

    # [NEW] 이미지 내 텍스트 언어 선택
    st.subheader("🌐 이미지 텍스트 언어")
    target_language = st.selectbox(
        "이미지 속에 들어갈 글자 언어:",
        ("Korean", "English", "Japanese"),
        index=0,
        help="이미지에 텍스트가 연출될 때 어떤 언어로 적을지 선택합니다."
    )

    st.markdown("---")

    # ==========================================
    # [NEW] 컨셉별 스타일 프리셋 시스템
    # ==========================================
    STYLE_PRESETS = {
        "경제학": """
무조건 2D 애니메이션 스타일. 심플한 동그란 하얀색 얼굴 캐릭터로 설명이 잘되는 화면 자료 느낌.

[캐릭터 - 일관된 스타일]
- 얼굴 형태: 심플한 동그란 하얀색 원 (이모티콘처럼 단순)
- 얼굴 요소: 점 눈, 작은 입 (심플하게)
- 표정: 귀여운 이모티콘 스타일 표정
- 몸: 슬림한 체형, 하얀색 피부
- 팔다리: 가늘고 부드러운 하얀색 (일반 비율)
- 의상: 직업에 맞는 컬러풀한 의상

[경제/비즈니스 의상]
CEO: 네이비 정장, 빨간 넥타이 / 직장인: 와이셔츠, 블라우스 / 자영업자: 컬러풀한 앞치마 / 학생: 교복

[배경]
2D로 디테일하게 몰입감 있게, 다양한 장소와 상황 연출. 하나의 통일된 장면.

[텍스트]
핵심 키워드 2~3개만 자연스럽게 배치

[색감]
에메랄드, 보라, 주황, 핑크, 민트, 골드 등 화려한 색상

[고정 스타일]
2D animation, simple white round circle face (like emoticon), dot eyes and small mouth, cute emoticon-style expressions, slim white body, slender limbs, colorful costumes, detailed 2D background, Korean keywords (2-3).
""",
        "역사": """
무조건 2D 애니메이션 스타일. 심플한 동그란 하얀색 얼굴 캐릭터로 역사 설명.

[캐릭터 - 일관된 스타일]
- 얼굴 형태: 심플한 동그란 하얀색 원 (이모티콘처럼 단순)
- 얼굴 요소: 점 눈, 작은 입 (심플하게)
- 표정: 귀여운 이모티콘 스타일 표정
- 몸: 슬림한 체형, 하얀색 피부
- 팔다리: 가늘고 부드러운 하얀색 (일반 비율)
- 의상: 시대에 맞는 역사적 의상

[역사 의상]
조선: 한복, 갓 / 로마: 토가, 갑옷 / 중세: 갑옷, 왕관, 드레스 / 왕족: 금색 장식

[배경]
2D로 디테일하게, 시대적 분위기와 역사적 장소 연출. 하나의 통일된 장면.

[텍스트]
핵심 키워드 2~3개만 자연스럽게 배치

[색감]
왕실: 골드, 퍼플 / 전쟁: 딥 레드, 실버 / 평화: 에메랄드, 스카이블루

[고정 스타일]
2D animation, simple white round circle face (like emoticon), dot eyes and small mouth, cute emoticon-style expressions, slim white body, slender limbs, era-appropriate costumes, detailed 2D background, Korean keywords (2-3).
""",
        "과학": """
무조건 2D 애니메이션 스타일. 과학/기술 상황을 심플한 동그란 하얀색 얼굴 캐릭터로 설명.

[캐릭터 - 일관된 스타일]
- 얼굴 형태: 심플한 동그란 하얀색 원 (이모티콘처럼 단순)
- 얼굴 요소: 점 눈, 작은 입 (심플하게)
- 표정: 귀여운 이모티콘 스타일 표정
- 몸: 슬림한 체형, 하얀색 피부
- 팔다리: 가늘고 부드러운 하얀색 (일반 비율)
- 의상: 과학/기술 분야에 맞는 의상

[과학 의상]
과학자: 흰 가운, 보안경 / 의사: 수술복, 청진기 / 우주비행사: 우주복 / 엔지니어: 작업복, 안전모

[배경]
2D로 디테일하게, 과학적 장소와 상황 연출. 하나의 통일된 장면.

[텍스트]
핵심 키워드 2~3개만 자연스럽게 배치

[색감]
실험실: 민트, 스카이블루 / 우주: 딥 퍼플, 네온 블루 / 디지털: 사이버 블루, 네온 핑크

[고정 스타일]
2D animation, simple white round circle face (like emoticon), dot eyes and small mouth, cute emoticon-style expressions, slim white body, slender limbs, science/tech costumes, detailed 2D background, Korean keywords (2-3).
""",
        "커스텀 (직접 입력)": """
무조건 2D 애니메이션 스타일. 심플한 동그란 하얀색 얼굴 캐릭터로 설명이 잘되는 화면 자료 느낌.

[캐릭터 - 일관된 스타일]
- 얼굴 형태: 심플한 동그란 하얀색 원 (이모티콘처럼 단순)
- 얼굴 요소: 점 눈, 작은 입 (심플하게)
- 표정: 귀여운 이모티콘 스타일 표정
- 몸: 슬림한 체형, 하얀색 피부
- 팔다리: 가늘고 부드러운 하얀색 (일반 비율)
- 의상: 직업/역할에 맞는 컬러풀한 의상

[배경]
2D로 디테일하게 몰입감 있게, 대본에 맞는 상황 연출. 하나의 통일된 장면.

[텍스트]
핵심 키워드 2~3개만 자연스럽게 배치

[고정 스타일]
2D animation, simple white round circle face (like emoticon), dot eyes and small mouth, cute emoticon-style expressions, slim white body, slender limbs, colorful costumes, detailed 2D background, Korean keywords (2-3).
"""
    }

    st.subheader("🎨 컨셉 선택")

    # 컨셉 선택 드롭박스
    selected_style_version = st.selectbox(
        "컨셉 프리셋",
        list(STYLE_PRESETS.keys()),
        index=0,
        help="컨셉별로 최적화된 스타일이 적용됩니다."
    )

    # 선택된 프리셋 가져오기
    default_style = STYLE_PRESETS[selected_style_version]

    style_instruction = st.text_area("AI에게 지시할 그림 스타일", value=default_style.strip(), height=150)
    st.markdown("---")
    
    # [NEW] Supertone TTS 설정 (secrets.toml 적용)
    st.subheader("🎙️ Supertone TTS 설정")
    
    # Base URL은 보통 안 바뀌므로 기본값 유지
    supertone_base_url = st.text_input("API 주소 (Base URL)", value=DEFAULT_SUPERTONE_URL)
    
    # API Key 자동 로드
    if "general" in st.secrets and "supertone_api_key" in st.secrets["general"]:
        supertone_api_key = st.secrets["general"]["supertone_api_key"]
        st.success("🔑 Supertone API Key가 로드되었습니다.")
    else:
        supertone_api_key = st.text_input("🔑 Supertone API Key", type="password")
    
    # 목소리 목록 관리
    if 'supertone_voices' not in st.session_state:
        st.session_state['supertone_voices'] = []
    
    # 연결 테스트 버튼
    if supertone_api_key:
        if st.button("🔌 연결 테스트 및 목소리 갱신"):
            with st.spinner("목소리 리스트 조회 중..."):
                success, voices, msg = check_connection_and_get_voices(supertone_api_key, supertone_base_url)
                if success:
                    st.session_state['supertone_voices'] = voices
                    st.success(f"{msg} ({len(voices)}개)")
                else:
                    st.error(msg)
                    st.session_state['supertone_voices'] = []
    
    # 목소리 선택 UI
    available_voices = []
    selected_voice_id = ""
    
    if st.session_state['supertone_voices']:
        raw_list = st.session_state['supertone_voices']
        valid_voices = [v for v in raw_list if isinstance(v, dict) and 'name' in v and 'voice_id' in v]
        if valid_voices:
            voice_options = {f"{v['name']} ({v['voice_id']})": v['voice_id'] for v in valid_voices}
            selected_voice_label = st.selectbox("목소리 선택", list(voice_options.keys()))
            selected_voice_id = voice_options[selected_voice_label]
            
            # 썸네일 표시
            current_voice = next((v for v in valid_voices if v['voice_id'] == selected_voice_id), None)
            if current_voice and current_voice.get('thumbnail_image_url'):
                st.image(current_voice['thumbnail_image_url'], width=100)
    else:
        # 연결 안 되었을 때 수동 입력창
        selected_voice_id = st.text_input("Voice ID 직접 입력", value=DEFAULT_VOICE_ID)
    
    st.caption("TTS 옵션")
    tts_speed = st.slider("말하기 속도", 0.5, 2.0, 1.0, 0.1)
    tts_pitch = st.slider("피치 조절", -12, 12, 0, 1)

    st.markdown("---")
    max_workers = st.slider("작업 속도(병렬 수)", 1, 10, 5)

# ==========================================
# [UI] 메인 화면 1: 대본 구조화 및 생성
# ==========================================
st.title("🎨 이미지 생성기")
st.caption("번호로 분할된 대본(1. 2. 3.)을 입력하면 씬별 이미지를 생성합니다.")

# 세션 초기화
if 'structured_content' not in st.session_state:
    st.session_state['structured_content'] = None
if 'section_scripts' not in st.session_state:
    st.session_state['section_scripts'] = {}
if 'video_title' not in st.session_state:
    st.session_state['video_title'] = ""
if 'user_initial_title' not in st.session_state:
    st.session_state['user_initial_title'] = ""

# 1. 구조 분석 섹션
with st.container(border=True):
    user_title_input = st.text_input(
        "📌 영상 제목 (선택사항)", 
        placeholder="이 제목을 입력하면 나중에 이미지 생성 단계에서 이와 유사한 제목들을 추천받을 수 있습니다.",
        help="비워두면 대본 내용을 바탕으로 AI가 알아서 제목을 추천합니다."
    )

    raw_script = st.text_area("✍️ 분석할 원고(대본)를 여기에 붙여넣으세요:", height=200, placeholder="안녕하세요, 오늘은...")
    analyze_btn = st.button("🔍 구조 분석 실행", width="stretch", type="primary")

    if analyze_btn:
        if not api_key:
            st.error("⚠️ 사이드바에서 Google API Key를 먼저 입력해주세요.")
        elif not raw_script:
            st.warning("⚠️ 분석할 대본 내용이 없습니다.")
        else:
            st.session_state['user_initial_title'] = user_title_input

            client = genai.Client(api_key=api_key)
            with st.status("대본 내용 분석 중...", expanded=True) as status:
                status.write(f"🧠 Gemini가 내용을 읽고 구조를 잡고 있습니다...")
                result_text = generate_structure(client, raw_script)
                
                st.session_state['structured_content'] = result_text
                st.session_state['section_scripts'] = {} 

                import re
                match = re.search(r'^\s*1\.\s*\*\*(.*?)\*\*:\s*(.*)', result_text, re.MULTILINE)
                if match:
                    extracted = match.group(2).strip() if match.group(2).strip() else match.group(1).strip()
                    st.session_state['video_title'] = re.sub(r'\(.*?\)', '', extracted).strip()
                else:
                    st.session_state['video_title'] = user_title_input if user_title_input else "제목을 찾을 수 없음"

                status.update(label="✅ 분석 완료! 제목이 추출되었습니다.", state="complete", expanded=False)

# 2. 대본 생성 섹션
if st.session_state['structured_content']:
    st.divider()
    st.subheader("📑 대본 구조화 결과")
    st.markdown(st.session_state['structured_content'])
    
    st.info(f"📌 **추출된 영상 제목:** {st.session_state['video_title']} (이미지 생성 단계에서 수정 가능)")

    st.divider()
    st.subheader("⚡ 롱폼 대본 전체 일괄 생성 (병렬 처리)")
    st.caption("🚀 버튼 한번으로 모든 챕터를 동시에 작성합니다. (15분/20분/25분 옵션)")

    lines = st.session_state['structured_content'].split('\n')
    chapter_titles = ["Intro (도입부)"]
    found_chapters = re.findall(r'(?:Chapter|챕터)\s*\d+.*', st.session_state['structured_content'])
    seen = set()
    for ch in found_chapters:
        clean_ch = ch.replace('*', '').strip()
        if clean_ch not in seen:
            chapter_titles.append(clean_ch)
            seen.add(clean_ch)
    chapter_titles.append("Epilogue (결론)")
    
    for title in chapter_titles:
        if title not in st.session_state['section_scripts']:
            st.session_state['section_scripts'][title] = ""

    with st.container(border=True):
        batch_instruction = st.text_area(
            "📢 전체 대본 작성 지침 (선택 사항)", 
            placeholder="예: 아주 비판적인 어조로 써줘 / 초등학생도 이해하기 쉽게 비유를 많이 들어줘 / 반말(평어)로 작성해줘 등",
            height=70
        )

        col_batch1, col_batch2 = st.columns([1, 1])
        with col_batch1:
            target_time = st.radio(
                "🎬 총 영상 목표 시간 (텍스트 분량)",
                ("15분 (약 7,000자)", "20분 (약 10,000자)", "25분 (약 13,000자)"),
                index=1
            )
            if "15분" in target_time: batch_duration_type = "2min" 
            elif "20분" in target_time: batch_duration_type = "3min" 
            else: batch_duration_type = "4min"

        with col_batch2:
            st.write("")
            st.write("") 
            st.write("") 
            batch_btn = st.button("🚀 전체 대본 동시 생성 시작", type="primary", use_container_width=True)

    if batch_btn:
        if not api_key:
            st.error("⚠️ Google API Key가 필요합니다.")
        else:
            client = genai.Client(api_key=api_key)
            status_box = st.status("🚀 AI가 지침을 반영하여 모든 챕터를 작성 중입니다...", expanded=True)
            progress_bar = status_box.progress(0)
            
            total_tasks = len(chapter_titles)
            completed_tasks = 0
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_title = {}
                for title in chapter_titles:
                    is_fixed = any(x in title for x in ["Intro", "Epilogue", "도입부", "결론"])
                    current_duration = "fixed" if is_fixed else batch_duration_type
                    
                    future = executor.submit(
                        generate_section, 
                        client, 
                        title, 
                        st.session_state['structured_content'], 
                        current_duration, 
                        batch_instruction
                    )
                    future_to_title[future] = title
                
                for future in as_completed(future_to_title):
                    title = future_to_title[future]
                    try:
                        result_text = future.result()
                        st.session_state['section_scripts'][title] = result_text
                        st.session_state[f"txt_{title}"] = result_text 
                        completed_tasks += 1
                        progress_bar.progress(completed_tasks / total_tasks)
                        status_box.write(f"✅ 완료: {title}")
                    except Exception as e:
                        status_box.error(f"❌ 실패 ({title}): {e}")
            
            status_box.update(label="✨ 전체 생성 완료! 아래에서 확인하세요.", state="complete", expanded=False)
            time.sleep(1)
            st.rerun()

    st.subheader("📝 섹션별 확인 및 수정")
    full_combined_script = ""
    
    for title in chapter_titles:
        with st.expander(f"📌 {title}", expanded=False):
            is_intro_epilogue = any(x in title for x in ["Intro", "Epilogue", "도입부", "결론"])
            
            if is_intro_epilogue:
                if st.button(f"🔄 {title} 다시 생성", key=f"r_fix_{title}"):
                    client = genai.Client(api_key=api_key)
                    with st.spinner("재생성 중..."):
                        result = generate_section(client, title, st.session_state['structured_content'], "fixed")
                        st.session_state['section_scripts'][title] = result
                        st.session_state[f"txt_{title}"] = result 
                        st.rerun()
            else:
                c_cols = st.columns(3)
                def regen(dur):
                    client = genai.Client(api_key=api_key)
                    with st.spinner(f"{dur} 모드로 재생성 중..."):
                        dur_code = "2min" if "2분" in dur else "3min" if "3분" in dur else "4min"
                        result = generate_section(client, title, st.session_state['structured_content'], dur_code)
                        st.session_state['section_scripts'][title] = result
                        st.session_state[f"txt_{title}"] = result
                        st.rerun()

                if c_cols[0].button("🔄 다시 생성 (2분)", key=f"r2_{title}"): regen("2분")
                if c_cols[1].button("🔄 다시 생성 (3분)", key=f"r3_{title}"): regen("3분")
                if c_cols[2].button("🔄 다시 생성 (4분)", key=f"r4_{title}"): regen("4분")

            if f"txt_{title}" not in st.session_state:
                st.session_state[f"txt_{title}"] = st.session_state['section_scripts'].get(title, "")

            new_text = st.text_area(label="📜 대본 내용 (수정 가능)", height=300, key=f"txt_{title}")
            st.session_state['section_scripts'][title] = new_text
        
        if st.session_state['section_scripts'].get(title):
            full_combined_script += st.session_state['section_scripts'][title] + "\n\n"

    if full_combined_script:
        st.divider()
        st.subheader("📦 최종 완성 대본")
        col_info, col_down = st.columns([3, 1])
        with col_info:
            st.caption(f"📝 총 글자 수: {len(full_combined_script)}자 (공백 포함)")
        with col_down:
            st.download_button(label="💾 대본 다운로드 (.txt)", data=full_combined_script, file_name="final_script.txt", mime="text/plain", use_container_width=True)
        st.text_area("아래 내용을 복사하거나 위 버튼을 눌러 저장하세요", value=full_combined_script, height=500)

# ==========================================
# [수정된 UI] 메인 화면 3: 이미지 생성
# ==========================================
st.divider()
st.title("🖼️ 씬별 이미지 생성")
st.caption(f"번호(1. 2. 3.)로 분할된 대본을 넣으면 각 씬에 맞는 이미지를 생성합니다. | 🎨 Model: {SELECTED_IMAGE_MODEL}")

st.subheader("📌 전체 영상 테마(제목) 설정")
st.caption("이미지 생성 시 이 제목이 '전체적인 분위기 기준'이 됩니다.")

if 'video_title' not in st.session_state:
    st.session_state['video_title'] = ""
if 'title_candidates' not in st.session_state:
    st.session_state['title_candidates'] = []

col_title_input, col_title_btn = st.columns([4, 1])

# [수정됨] 버튼 로직: 구조 분석이 없어도 제목 입력이 있으면 작동하도록 변경
with col_title_btn:
    st.write("") 
    st.write("") 
    if st.button("💡 제목 5개 추천", help="입력한 키워드나 대본을 바탕으로 제목을 추천합니다.", use_container_width=True):
        # 현재 입력된 제목(주제) 가져오기
        current_user_title = st.session_state.get('video_title', "").strip()
        has_structure = st.session_state.get('structured_content')

        if not api_key:
            st.error("API Key 필요")
        # [핵심 수정] 구조 분석도 안 했고, 제목 입력도 없으면 경고
        elif not has_structure and not current_user_title:
            st.warning("⚠️ '구조 분석'을 먼저 하거나, 왼쪽에 '주제'를 입력해주세요.")
        else:
            client = genai.Client(api_key=api_key)
            with st.spinner("AI가 최적의 제목을 고민 중입니다..."):
                
                # 1. 구조 분석 데이터는 없지만, 사용자가 입력한 주제는 있는 경우
                if current_user_title and not has_structure:
                    prompt_instruction = f"""
                    [Target Topic]
                    "{current_user_title}"
                    [Task]
                    Generate 5 click-bait YouTube video titles based on the Target Topic above.
                    사용자가 입력한거랑 최대한 비슷한 제목으로 추천, '몰락'이 들어간 경우 맨 뒤에 몰락으로 끝나게 한다.
                    """
                    context_data = "No script provided. Base it solely on the topic."

                # 2. 구조 분석 데이터가 있는 경우 (입력한 제목이 있으면 그것도 반영)
                else:
                    if current_user_title:
                        prompt_instruction = f"""
                        [Target Context]
                        "{current_user_title}"
                        [Task]
                        Generate 5 variations of this title suitable for YouTube, considering the script below.
                        '몰락'이 들어간 경우 맨 뒤에 몰락으로 끝나게 한다.
                        """
                    else:
                        prompt_instruction = f"""
                        [Task]
                        Read the provided script structure and generate 5 catchy YouTube video titles in Korean.
                        """
                    context_data = st.session_state['structured_content']

                title_prompt = f"""
                [Role] You are a YouTube viral marketing expert.
                {prompt_instruction}
                
                [Script Context]
                {context_data}
                
                [Output Format]
                - Output ONLY the list of 5 titles.
                - No numbering (1., 2.), just 5 lines of text.
                - Language: Korean
                """
                
                try:
                    resp = client.models.generate_content(
                        model=GEMINI_TEXT_MODEL_NAME, 
                        contents=title_prompt
                    )
                    candidates = [line.strip() for line in resp.text.split('\n') if line.strip()]
                    clean_candidates = []
                    import re
                    for c in candidates:
                        clean = re.sub(r'^\d+\.\s*', '', c).replace('*', '').replace('"', '').strip()
                        if clean: clean_candidates.append(clean)
                    
                    st.session_state['title_candidates'] = clean_candidates[:5]
                except Exception as e:
                    st.error(f"오류 발생: {e}")

with col_title_input:
    st.text_input(
        "영상 제목 (직접 입력하거나 우측 버튼으로 추천받으세요)",
        key="video_title", 
        placeholder="제목 혹은 만들고 싶은 주제를 입력하세요 (예: 부자들의 습관)"
    )

if st.session_state['title_candidates']:
    st.info("👇 AI가 추천한 제목입니다. 클릭하면 적용됩니다.")

    def apply_title(new_title):
        st.session_state['video_title'] = new_title
        st.session_state['title_candidates'] = [] 

    for idx, title in enumerate(st.session_state['title_candidates']):
        col_c1, col_c2 = st.columns([4, 1])
        with col_c1:
            st.markdown(f"**{idx+1}. {title}**")
        with col_c2:
            st.button(
                "✅ 선택", 
                key=f"sel_title_{idx}", 
                on_click=apply_title, 
                args=(title,), 
                use_container_width=True
            )
    
    if st.button("❌ 목록 닫기"):
        st.session_state['title_candidates'] = []

if 'section_scripts' in st.session_state and st.session_state['section_scripts']:
    intro_text_acc = ""
    main_text_acc = ""
    for title_key, text in st.session_state['section_scripts'].items():
        if "Intro" in title_key or "도입부" in title_key:
            intro_text_acc += text + "\n\n"
        else:
            main_text_acc += text + "\n\n"
            
    st.write("👇 **생성된 대본 가져오기 (클릭 시 아래 입력창에 채워집니다)**")
    
    col_get1, col_get2 = st.columns(2)
    if "image_gen_input" not in st.session_state:
        st.session_state["image_gen_input"] = ""

    with col_get1:
        if st.button("📥 인트로(Intro)만 가져오기", use_container_width=True):
            st.session_state["image_gen_input"] = intro_text_acc.strip()
            st.rerun()
    with col_get2:
        if st.button("📥 본론(Chapters) + 결론(Epilogue) 가져오기", use_container_width=True):
            st.session_state["image_gen_input"] = main_text_acc.strip()
            st.rerun()

script_input = st.text_area(
    "📜 번호로 분할된 대본 입력 (1. 2. 3. 형태)",
    height=300,
    placeholder="""예시:
1.하지만 영원할 것 같았던
이 거대한 제국은 어느 순간부터
거리에 하나둘씩 간판을 내리기 시작하더니
마치 신기루처럼 사라져버렸습니다

2.오백 개가 넘는 매장이
순식간에 증발해 버린 진짜 이유는,
외부의 적이 아닌 내부의 가족,
바로 부부의 전쟁 때문이었습니다

3.한때 대한민국 요식업 프랜차이즈의 신화였으나
지금은 오너 리스크의 가장 끔찍한 교과서로 남게 된
비운의 브랜드...""",
    key="image_gen_input"
)

if 'generated_results' not in st.session_state:
    st.session_state['generated_results'] = []
if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False

# [KEY FIX] 버튼 클릭 시 결과물 초기화 함수 추가
def clear_generated_results():
    st.session_state['generated_results'] = []

start_btn = st.button("🚀 이미지 생성 시작", type="primary", width="stretch", on_click=clear_generated_results)

if start_btn:
    if not api_key:
        st.error("⚠️ Google API Key를 입력해주세요.")
    elif not script_input:
        st.warning("⚠️ 대본을 입력해주세요.")
    else:
        # [FIX] 기존 결과 확실히 날리기
        st.session_state['generated_results'] = [] 
        st.session_state['is_processing'] = True
        
        # [FIX] 기존 이미지 파일들 물리적으로 삭제 (찌꺼기 제거)
        if os.path.exists(IMAGE_OUTPUT_DIR):
            shutil.rmtree(IMAGE_OUTPUT_DIR) # 폴더 통째로 삭제
        init_folders() # 다시 깨끗한 폴더 생성
        
        client = genai.Client(api_key=api_key)
        
        status_box = st.status("작업 진행 중...", expanded=True)
        progress_bar = st.progress(0)
        
        # 1. 대본 분할 (번호 기반)
        status_box.write(f"✂️ 번호(1. 2. 3.)로 분할된 대본 파싱 중...")
        chunks = parse_numbered_script(script_input)
        total_scenes = len(chunks)

        if total_scenes == 0:
            status_box.update(label="⚠️ 번호로 분할된 씬을 찾을 수 없습니다. (예: 1.내용 2.내용)", state="error")
            st.stop()

        status_box.write(f"✅ {total_scenes}개 씬으로 파싱 완료.")
        
        current_video_title = st.session_state.get('video_title', "").strip()
        if not current_video_title:
            current_video_title = "전반적인 대본 분위기에 어울리는 배경 (Context based on the script)"

        # 2. 프롬프트 생성 (병렬)
        status_box.write(f"📝 프롬프트 작성 중 ({GEMINI_TEXT_MODEL_NAME})...")
        prompts = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for i, chunk in enumerate(chunks):
                futures.append(executor.submit(
                    generate_prompt,
                    api_key,
                    i,
                    chunk,
                    style_instruction,
                    current_video_title,
                    target_language
                ))
            
            for i, future in enumerate(as_completed(futures)):
                prompts.append(future.result())
                progress_bar.progress((i + 1) / (total_scenes * 2))
        
        prompts.sort(key=lambda x: x[0])
        
        # ... (이전 코드: 프롬프트 생성 부분은 그대로 유지) ...

        # 3. 이미지 생성 (병렬 처리 + 속도 조절)
        status_box.write(f"🎨 이미지 생성 중 ({SELECTED_IMAGE_MODEL})... (API 보호를 위해 천천히 진행됩니다)")
        results = []
        
        # [중요] API 제한을 피하기 위해 worker 수를 강제로 조절하거나, 제출 간격을 둡니다.
        # 사용자가 설정한 max_workers를 쓰되, 요청 간격을 벌립니다.
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {}
            for s_num, prompt_text in prompts:
                idx = s_num - 1
                orig_text = chunks[idx]
                fname = make_filename(s_num, orig_text)
                
                # [핵심 수정] 요청을 한꺼번에 쏘지 않고 3초씩 쉬면서 제출합니다.
                # 이렇게 하면 분당 20회 제한 안쪽으로 자연스럽게 들어옵니다.
                time.sleep(3) 
                
                future = executor.submit(generate_image, client, prompt_text, fname, IMAGE_OUTPUT_DIR, SELECTED_IMAGE_MODEL)
                future_to_meta[future] = (s_num, fname, orig_text, prompt_text)
            
            # 결과 수집
            completed_cnt = 0
            for future in as_completed(future_to_meta):
                s_num, fname, orig_text, p_text = future_to_meta[future]
                path = future.result()
                
                # [핵심] 실패(None)하더라도 결과 리스트에는 넣어서 순서가 밀리지 않게 함 (원한다면 에러 이미지 처리 가능)
                if path:
                    results.append({
                        "scene": s_num,
                        "path": path,
                        "filename": fname,
                        "script": orig_text,
                        "prompt": p_text,
                        "audio_path": None,
                        "video_path": None 
                    })
                else:
                    # 실패 시 로그만 남기고 넘어가거나, 더미 데이터를 넣을 수도 있음
                    st.error(f"Scene {s_num} 이미지 생성 최종 실패.")

                completed_cnt += 1
                progress_bar.progress(0.5 + (completed_cnt / total_scenes * 0.5))
        
        results.sort(key=lambda x: x['scene'])
        st.session_state['generated_results'] = results
        
        status_box.update(label="✅ 완료되었습니다!", state="complete", expanded=False)
        st.session_state['is_processing'] = False
        
# ==========================================
# [수정됨] 결과창 및 개별 재생성 기능 추가
# ==========================================
if st.session_state['generated_results']:
    st.divider()
    st.header(f"📸 결과물 ({len(st.session_state['generated_results'])}장)")
    
    # ------------------------------------------------
    # 1. 일괄 작업 버튼 영역
    # ------------------------------------------------
    st.write("---")
    st.subheader("⚡ 원클릭 일괄 생성 작업")
    
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
    
    with c_btn1:
        zip_data = create_zip_buffer(IMAGE_OUTPUT_DIR)
        st.download_button("📦 전체 이미지 ZIP 다운로드", data=zip_data, file_name="all_images.zip", mime="application/zip", use_container_width=True)

    # TTS 전체 생성
    with c_btn2:
        tts_batch_mode = st.selectbox("TTS 생성 모드", ["원본 음성 생성", "무음 조절 음성 (최대 0.3초)"], help="무음 조절 선택 시 공백 자동 축소")
        if st.button("🔊 TTS 일괄 생성", use_container_width=True):
            if not supertone_api_key or not selected_voice_id:
                st.error("사이드바에서 API Key와 목소리를 설정해주세요.")
            else:
                # 오디오 변경 시 통합본 삭제
                final_merged_file = os.path.join(VIDEO_OUTPUT_DIR, "FINAL_FULL_VIDEO.mp4")
                if os.path.exists(final_merged_file):
                    try: os.remove(final_merged_file)
                    except: pass

                status_box = st.status("🎙️ TTS 일괄 생성 중...", expanded=True)
                progress_bar = status_box.progress(0)
                
                apply_trim = (tts_batch_mode == "무음 조절 음성 (최대 0.3초)")
                total_files = len(st.session_state['generated_results'])
                completed_cnt = 0
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {}
                    for i, item in enumerate(st.session_state['generated_results']):
                        future = executor.submit(
                            process_single_tts_task, supertone_api_key, selected_voice_id, 
                            item['script'], item['scene'], supertone_base_url, 
                            tts_speed, tts_pitch, apply_trim
                        )
                        future_to_idx[future] = i
                    
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result_path = future.result()
                            if "Error" not in str(result_path) and "VOICE_NOT_FOUND" not in str(result_path):
                                st.session_state['generated_results'][idx]['audio_path'] = result_path
                                st.session_state['generated_results'][idx]['video_path'] = None # 비디오 리셋
                            else:
                                st.write(f"⚠️ Scene {idx+1} 오류: {result_path}")
                        except Exception as e:
                            st.write(f"⚠️ Scene {idx+1} 시스템 오류: {e}")
                        
                        completed_cnt += 1
                        progress_bar.progress(completed_cnt / total_files)
                
                status_box.update(label="✅ TTS 생성 완료!", state="complete", expanded=False)
                time.sleep(1)
                st.rerun()

    # 비디오 전체 생성
    with c_btn3:
        has_audio = any(item.get('audio_path') for item in st.session_state['generated_results'])
        if st.button("🎬 비디오 전체 일괄 생성", disabled=not has_audio, use_container_width=True):
            final_merged_file = os.path.join(VIDEO_OUTPUT_DIR, "FINAL_FULL_VIDEO.mp4")
            if os.path.exists(final_merged_file):
                try: os.remove(final_merged_file)
                except: pass
            
            status_box = st.status("🎬 비디오 렌더링 중...", expanded=True)
            progress_bar = status_box.progress(0)
            
            total_files = len(st.session_state['generated_results'])
            completed_cnt = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                for i, item in enumerate(st.session_state['generated_results']):
                    is_zoom_in = (i % 2 == 0)
                    future = executor.submit(process_single_video_task, item, VIDEO_OUTPUT_DIR, is_zoom_in)
                    future_to_idx[future] = i
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        vid_path = future.result()
                        if vid_path and "Error" not in vid_path:
                            st.session_state['generated_results'][idx]['video_path'] = vid_path
                        elif vid_path:
                            st.write(f"⚠️ Scene {idx+1} 렌더링 오류: {vid_path}")
                    except Exception as e:
                        st.write(f"⚠️ Scene {idx+1} 시스템 오류: {e}")
                    
                    completed_cnt += 1
                    progress_bar.progress(completed_cnt / total_files)
            
            status_box.update(label="✅ 비디오 생성 완료!", state="complete", expanded=False)
            time.sleep(1)
            st.rerun()

    # 전체 병합
    with c_btn4:
        video_paths = [item.get('video_path') for item in st.session_state['generated_results'] if item.get('video_path')]
        final_path = os.path.join(VIDEO_OUTPUT_DIR, "FINAL_FULL_VIDEO.mp4")
        
        if video_paths:
            if st.button("🎞️ 전체 영상 합치기 (새로고침)", use_container_width=True):
                with st.spinner("모든 비디오를 하나로 합치는 중..."):
                    if os.path.exists(final_path):
                        try: os.remove(final_path)
                        except: pass
                        
                    merged_result = merge_all_videos(video_paths, VIDEO_OUTPUT_DIR)
                    if "Error" in merged_result:
                        st.error(merged_result)
                    else:
                        st.success("병합 완료!")
                        st.rerun()

            if os.path.exists(final_path):
                 with open(final_path, "rb") as f:
                    st.download_button("💾 전체 영상 다운로드 (MP4)", data=f, file_name="final_video.mp4", mime="video/mp4", use_container_width=True)
        else:
            st.button("🎞️ 전체 영상 합치기", disabled=True, use_container_width=True)

    if not supertone_api_key or not selected_voice_id:
        st.warning("🎙️ Supertone TTS 사용을 위해 API 설정을 확인하세요.")

    # ------------------------------------------------
    # 2. 개별 리스트 및 [재생성] 기능
    # ------------------------------------------------
    for index, item in enumerate(st.session_state['generated_results']):
        with st.container(border=True):
            cols = st.columns([1, 2])
            
            # [왼쪽] 이미지 및 재생성 버튼
            with cols[0]:
                try: st.image(item['path'], use_container_width=True)
                except: st.error("이미지 없음")
                
                # [NEW] 이미지 개별 재생성 버튼
                if st.button(f"🔄 이 장면만 이미지 다시 생성", key=f"regen_img_{index}", use_container_width=True):
                    if not api_key:
                        st.error("API Key가 필요합니다.")
                    else:
                        with st.spinner(f"Scene {item['scene']} 다시 그리는 중..."):
                            client = genai.Client(api_key=api_key)
                            
                            # 1. 프롬프트 다시 생성 (현재 대본과 스타일 반영)
                            current_title = st.session_state.get('video_title', '')
                            # 대본이 수정되었을 수도 있으므로 item['script'] 사용
                            _, new_prompt = generate_prompt(
                                api_key, index, item['script'], style_instruction,
                                current_title, target_language
                            )
                            
                            # 2. 이미지 생성
                            new_path = generate_image(
                                client, new_prompt, item['filename'], 
                                IMAGE_OUTPUT_DIR, SELECTED_IMAGE_MODEL
                            )
                            
                            if new_path:
                                # 3. 결과 업데이트
                                st.session_state['generated_results'][index]['path'] = new_path
                                st.session_state['generated_results'][index]['prompt'] = new_prompt
                                # 이미지가 바뀌었으므로 기존 비디오는 무효화
                                st.session_state['generated_results'][index]['video_path'] = None
                                st.success("이미지가 변경되었습니다!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("이미지 생성에 실패했습니다.")

            # [오른쪽] 정보 및 오디오/비디오 컨트롤
            with cols[1]:
                st.subheader(f"Scene {item['scene']:02d}")
                st.caption(f"파일명: {item['filename']}")
                
                # 대본 수정 가능하게 할지? (현재는 display만)
                st.write(f"**대본:** {item['script']}")
                
                st.markdown("---")
                audio_col1, audio_col2 = st.columns([1, 3])
                
                # 오디오 로직
                if item.get('audio_path') and os.path.exists(item['audio_path']):
                    with audio_col1:
                        st.audio(item['audio_path'])
                        if st.button("🔄 오디오 재생성", key=f"re_tts_{item['scene']}"):
                             item['audio_path'] = None
                             item['video_path'] = None 
                             st.rerun()
                    
                    with audio_col2:
                        if item.get('video_path') and os.path.exists(item['video_path']):
                            st.video(item['video_path'])
                            with open(item['video_path'], "rb") as vf:
                                st.download_button("⬇️ 비디오 저장", data=vf, file_name=f"scene_{item['scene']}.mp4", mime="video/mp4", key=f"down_vid_{item['scene']}")
                        else:
                            is_zoom_in_mode = (index % 2 == 0)
                            button_label = f"🎬 비디오 생성 ({'줌인' if is_zoom_in_mode else '줌아웃'})"

                            if st.button(button_label, key=f"gen_vid_{item['scene']}"):
                                with st.spinner("렌더링 중..."):
                                    vid_path = create_video_with_zoom(
                                        item['path'], item['audio_path'], VIDEO_OUTPUT_DIR, 
                                        item['scene'], is_zoom_in=is_zoom_in_mode
                                    )
                                    if "Error" in vid_path:
                                        st.error(vid_path)
                                    else:
                                        st.session_state['generated_results'][index]['video_path'] = vid_path
                                        st.rerun()
                else:
                    with audio_col1:
                        if st.button("🔊 TTS 생성", key=f"gen_tts_{item['scene']}"):
                            if not supertone_api_key or not selected_voice_id:
                                st.error("설정 필요")
                            else:
                                with st.spinner("오디오 생성 중..."):
                                    audio_result = generate_supertone_tts(
                                        supertone_api_key, selected_voice_id, 
                                        item['script'], item['scene'], supertone_base_url, 
                                        speed=tts_speed, pitch=tts_pitch
                                    )
                                    if "Error" not in str(audio_result) and "VOICE_NOT_FOUND" != audio_result:
                                        st.session_state['generated_results'][index]['audio_path'] = audio_result
                                        st.rerun()
                                    else:
                                        st.error(audio_result)

                with st.expander("프롬프트 확인"):
                    st.text(item['prompt'])
                try:
                    with open(item['path'], "rb") as file:
                        st.download_button("⬇️ 이미지 저장", data=file, file_name=item['filename'], mime="image/png", key=f"btn_down_{item['scene']}")
                except: pass



