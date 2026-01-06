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

# ==========================================
# [UPGRADE] 함수: AI 기반 대본 맥락 분할 (글자수 제한 엄격화)
# ==========================================
def split_text_automatically(client, full_text, target_chars=370):
    """
    Gemini를 이용해 문맥(Context)을 파악하고,
    시각적 장면 전환이 필요한 지점마다 대본을 분할합니다.
    각 씬은 반드시 target_chars 이하로 제한됩니다.
    """
    prompt = f"""
    [Role] Video Storyboard Editor
    [Task] Split the [Script] into multiple "Scenes".
    [Rules]
    1. **STRICT LIMIT:** Each scene MUST be under {target_chars} characters.
    2. **NEVER MERGE:** Even if only 100 characters remain at the end, make it a SEPARATE scene. NEVER create a chunk longer than {target_chars} characters.
    3. **Context:** Split where the visual topic changes, but prioritize the length limit.
    4. **Output:** Return ONLY a raw JSON list of strings.

    [Script]
    {full_text}
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        scenes = json.loads(response.text)
        if isinstance(scenes, list):
            return [s.strip() for s in scenes if s.strip()]
        else:
            return split_script_by_time(full_text, chars_per_chunk=target_chars)
    except Exception as e:
        print(f"AI Split Error: {e}")
        return split_script_by_time(full_text, chars_per_chunk=target_chars)


# [NEW] 규칙 기반 분할 함수 (370자 기준 엄격 분할)
def split_script_by_time(script, chars_per_chunk=370):
    """370자 기준 엄격 분할: 마지막 남은 글자도 무조건 별도 씬으로 분할"""
    sentences = re.split(r'(?<=[.?!])\s+', script.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence.strip():
            continue

        # 현재 문장이 제한을 넘는 경우 (강제 절단)
        if len(sentence) > chars_per_chunk:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            temp = sentence
            while len(temp) > chars_per_chunk:
                chunks.append(temp[:chars_per_chunk].strip())
                temp = temp[chars_per_chunk:]
            current_chunk = temp
            continue

        # 글자수 체크 (합쳤을 때 제한을 넘으면 현재까지를 씬으로 확정)
        if len(current_chunk) + len(sentence) + 1 > chars_per_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

    # [핵심] 마지막 남은 문장이 500자든 10자든 무조건 별개 씬으로 추가
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ==========================================
# [NEW] 도입부 분할 함수 (24~48자, 의미 기준 분할)
# ==========================================
def split_intro_by_meaning(client, intro_text, min_chars=24, max_chars=48):
    """
    도입부를 의미 단위로 분할합니다.
    각 씬은 4~8초(약 24~48자) 범위로 제한됩니다.
    """
    if not intro_text or not intro_text.strip():
        return []

    prompt = f"""
[Role] Short-form Video Editor (숏폼 영상 편집자)

[Task]
아래 [도입부 대본]을 **의미 단위**로 나누어 짧은 씬들로 분할하세요.
각 씬은 4~8초로 읽을 수 있는 분량이어야 합니다.

[Rules]
1. **글자수 범위:** 각 씬은 **{min_chars}자 ~ {max_chars}자** 범위가 이상적입니다.
2. **의미 단위 분할:** 문장을 억지로 자르지 말고, 의미가 완결되는 지점에서 나누세요.
3. **짧은 임팩트:** "바로 삼성입니다" 같은 짧은 문장(24자 미만)도 그 자체로 하나의 씬이 됩니다.
4. **자연스러운 끊김:** 쉼표(,)나 문장 구조를 활용해 자연스럽게 분할하세요.
5. **너무 길면 분할:** {max_chars}자를 크게 초과하면 의미 단위로 나누세요.

[예시]
입력: "한 기업이 삼십년 만에 자동차 시장에 다시 발을 들였습니다. 바로 삼성입니다."
출력: ["한 기업이 30년 만에 자동차 시장에 다시 발을 들였습니다", "바로 삼성입니다"]

[도입부 대본]
{intro_text}

[Output]
- JSON 배열 형식으로만 출력하세요. 설명 없이 배열만 반환하세요.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        scenes = json.loads(response.text)
        if isinstance(scenes, list):
            return [s.strip() for s in scenes if s.strip()]
        else:
            return split_intro_fallback(intro_text, max_chars)
    except Exception as e:
        print(f"Intro Split Error: {e}")
        return split_intro_fallback(intro_text, max_chars)


def split_intro_fallback(intro_text, max_chars=48):
    """도입부 분할 폴백: 문장 부호 기준으로 분할 (4~8초 기준)"""
    # 마침표, 쉼표, 물음표, 느낌표 기준으로 분할
    parts = re.split(r'(?<=[.,?!])\s*', intro_text.strip())
    chunks = []
    current = ""

    for part in parts:
        if not part.strip():
            continue
        if len(part) <= max_chars:
            chunks.append(part.strip())
        elif len(current) + len(part) + 1 <= max_chars:
            current = (current + " " + part).strip() if current else part
        else:
            if current:
                chunks.append(current.strip())
            # 긴 문장은 강제 분할
            while len(part) > max_chars:
                chunks.append(part[:max_chars].strip())
                part = part[max_chars:]
            current = part

    if current:
        chunks.append(current.strip())

    return chunks


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
# [마스터 업그레이드] 함수: 프롬프트 생성 (범용 월드빌딩 + 환경적 일치 + 스토리 전개)
# ==========================================
def generate_prompt(api_key, index, text_chunk, style_instruction, video_title, target_language="Korean"):
    """[최종 마스터 버전] 범용 맥락 인지형 월드빌딩 + 환경적 일치 + 캐릭터 행동 연출"""
    scene_num = index + 1
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TEXT_MODEL_NAME}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}

    lang_guide = f"화면 속 텍스트는 반드시 '{target_language}'(으)로만 정확히 표기하십시오."

    # [핵심] 스타일 지침: 3D 제거, 상하의 착장, 캐릭터 눈, 환경 통합
    style_suffix = (
        ", **High-quality 2D flat animation style with subtle cell shading**. "
        "Strictly NO 3D, NO rendering. "
        "The white stickman MUST be fully dressed (vibrant colorful top and trousers). "
        "Character's eyes: large white sclera with small black pupils (cartoon expressive eyes). "
        "**ONE SINGLE UNIFIED SCENE. NO split screens. NO generic square boxes or TV screens.** "
        "**ENVIRONMENTAL INTEGRATION**: Text and data must be part of the world (e.g., as neon signs, massive stone monuments, or physical landscapes). "
        "Text must have a thick high-pixel black outline."
    )

    full_instruction = f"""
[Role] 2D 애니메이션 월드빌딩 디렉터 (World-Building Director)

[Style Guide]
{style_instruction}

[Overall Context]
- 전체 영상 제목: "{video_title}"
- 현재 장면 순서: {scene_num}번째 장면
- 전체적인 흐름을 고려하여, 단순히 정보를 나열하는 게 아니라 대본의 상황이 '물리적인 세계'에서 벌어지는 것처럼 연출하십시오.

[Visual Task: 대본과 환경의 일치]
1. **박스형 연출 금지:** 텍스트나 데이터를 보여주기 위해 네모난 스크린, TV, 칠판을 배경에 두지 마십시오. 대신 그 정보들이 배경의 '일부'가 되게 하십시오.
   - 예: 중요한 숫자/금액 -> 거대한 네온사인이나 하늘에 떠 있는 홀로그램처럼 표현
   - 예: 도전/성장 -> 캐릭터가 가파른 산을 오르거나 넓은 바다를 항해하는 모습
   - 예: 위기/변화 -> 건물이 변형되거나 풍경이 극적으로 바뀌는 모습
2. **배경의 구체화:** 사무실을 벗어나십시오. 활기찬 도시, 광활한 자연, 미래적인 공간, 역동적인 현장 등 대본의 '감정과 주제'에 맞는 장소를 선택하십시오.
3. **캐릭터의 연기:** 스틱맨은 단순히 서 있지 않고, 배경의 상황에 직접 반응해야 합니다. (달리기, 점프, 놀람, 환호, 고민 등 상황에 맞는 동작)
4. **전개성:** 이전 장면들과 이어지는 하나의 '이야기'처럼 느껴지도록 풍경의 디테일을 설정하십시오.

[캐릭터 및 텍스트 기술 지침]
- **눈:** 하얀 눈자위 안에 작은 검은색 눈동자 (놀람, 슬픔, 기쁨 등 감정 표현).
- **의상:** 상하의를 갖춰 입은 컬러풀한 의상.
- **텍스트:** {lang_guide} (사물 위나 말풍선에 배치하되 두꺼운 외곽선 사용).
- **단일 장면:** 화면 분할이나 번호 매기기 절대 금지.
- **밝은 톤:** 어두운 주제라도 밝은 조명을 유지하고 시각적 상징물로 표현.

[현재 대본 조각]
"{text_chunk}"

[Output]
- 번호 없이, 캐릭터의 행동과 배경의 상호작용이 돋보이는 한국어 프롬프트 문단 하나만 출력하십시오.
"""

    payload = {"contents": [{"parts": [{"text": full_instruction}]}]}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            generated_text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            # 안전장치: SCENE 번호나 분할 단어 제거
            generated_text = re.sub(r'#+\s*SCENE\s*\d+|장면\s*\d+|씬\s*\d+', '', generated_text, flags=re.IGNORECASE)

            final_prompt = f"{generated_text.rstrip('.')}{style_suffix}"
            return (scene_num, final_prompt)
        else:
            return (scene_num, f"Error: {response.status_code}")
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

    # 1. Google API Key 설정 (멀티 API 지원)
    st.subheader("🔑 API 키 설정")

    # API 키 개수 선택 드롭박스
    num_api_keys = st.selectbox(
        "API 키 개수",
        options=[1, 2, 3, 4],
        index=0,
        help="여러 API 키를 사용하면 병렬 처리로 더 빠르게 생성할 수 있습니다. (키당 분당 20개)"
    )

    # API 키 입력 필드들
    api_keys = []

    # secrets.toml에서 자동 로드 시도
    for i in range(num_api_keys):
        secret_key = f"google_api_key_{i+1}" if i > 0 else "google_api_key"

        if "general" in st.secrets and secret_key in st.secrets["general"]:
            key = st.secrets["general"][secret_key]
            st.success(f"🔑 API Key {i+1} 로드됨")
            api_keys.append(key)
        else:
            key = st.text_input(
                f"🔑 Google API Key {i+1}" if num_api_keys > 1 else "🔑 Google API Key",
                type="password",
                key=f"api_key_{i}",
                help=f"API 키 {i+1}번을 입력하세요."
            )
            if key:
                api_keys.append(key)

    # 호환성을 위해 첫 번째 키를 api_key로도 저장
    api_key = api_keys[0] if api_keys else ""

    if num_api_keys > 1 and len(api_keys) == num_api_keys:
        st.info(f"✅ {num_api_keys}개 API 키 설정됨 → 분당 최대 {num_api_keys * 20}개 생성 가능")

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
    # [수정됨] Gems 스타일 + Context-Aware Visual Guide
    STYLE_PRESETS = {
        "경제학": """
[Core Identity]
- Character: "White circle-faced stickman" with PURE WHITE body and limbs.
- Face Style: Minimalist white round head with eyes and mouth.
- **Critical Rule:** Character MUST have EYES and MOUTH.

[Lighting & Background Guide - BRIGHT & CLEAR]
- **General Tone:** ALWAYS use **Bright, Clear, and Visible** lighting.
- **Avoid:** Pitch black darkness, heavy shadows, or muddy colors.
- **Scenario Lighting:**
    - *Success/Business:* "Bright White Studio", "Sunny Day", "Warm Golden Light".
    - *Crisis/Sadness:* "Grey Cloudy Daylight", "Dim Indoor Light" (NOT pitch black), "Cold Blue Tint" (keep it bright enough to see).
    - *News:* "Bright TV Studio Lighting".

[Context-Aware Visual Guide (Crucial)]
1. **Scenario: Business/Partnership (e.g., M&A, Deals)**
   - **Background:** Bright conference room, stage with handshake, modern office.
   - **Extras:** A few other stickmen (reporters, investors) in the background.
   - **Text Integration:** Place text on **podiums**, **company flags**, **shirt labels**, or **large presentation screens**.

2. **Scenario: Economic Crisis/Failure (e.g., Collapse, Despair)**
   - **Background:** Grey cloudy city, dim office, rainy street (NOT pitch black).
   - **Extras:** Usually solo, or with a few sad figures in the distance.
   - **Text Integration:** Place text on **broken neon signs**, **cracked walls**, **graffiti**, or **scattered papers**.

3. **Scenario: Market/Public Reaction (e.g., Trends, Opinions)**
   - **Background:** Public spaces like streets, stock market floors, or online communities.
   - **Extras:** **Crowd of anonymous stickmen** showing reactions (angry, confused, cheering).
   - **Text Integration:** Text on **protest signs**, **thought bubbles above crowds**, **stock ticker boards**.

4. **Scenario: News/Announcement**
   - **Background:** Cozy living room with TV, or a bright news studio desk.
   - **Extras:** None (focus on TV) or a news anchor.
   - **Text Integration:** Text inside a **"Breaking News" banner on a TV screen**.

[Text Object Integration (Limit 2-3 Keywords)]
- **QUANTITY RULE:** Strictly limit text to **2-3 CORE KEYWORDS** per scene. Do not use long sentences or too many tags.
- **Readability:** Text must have a **CLEAN OUTLINE**.
- **Placement:** Text on TV Screen, Presentation Slide, Labels on Shirt, Neon Sign with clean border.

[Costume & Role]
- CEO: 네이비 정장, 빨간 넥타이 / 가난한 사람: 낡은 회색 가디건
- 직장인: 와이셔츠, 블라우스 / 부자: 금색 액세서리

[Face Expression Guide]
- Use simple cartoon eyes and mouths to clearly show emotions.
""",
        "역사": """
[Core Identity]
- Character: "White circle-faced stickman" with PURE WHITE body and limbs.
- Face Style: Minimalist white round head with eyes and mouth.
- **Critical Rule:** Character MUST have EYES and MOUTH.

[Lighting & Background Guide - BRIGHT & CLEAR]
- **General Tone:** ALWAYS use **Bright, Clear, and Visible** lighting.
- **Avoid:** Pitch black darkness, heavy shadows, or muddy colors.
- **Scenario Lighting:**
    - *Victory/Coronation:* "Bright Golden Sunlight", "Warm Throne Room Light".
    - *Battle/War:* "Smoky but Visible Daylight", "Orange Firelight" (NOT pitch black).
    - *Tragedy:* "Grey Cloudy Sky", "Dim Candlelight" (keep it bright enough to see).

[Context-Aware Visual Guide (Crucial)]
1. **Scenario: War/Battle**
   - **Background:** Smoky battlefield with visible sky, siege walls, army camps.
   - **Extras:** **Army of stickmen soldiers** in the background, fallen warriors.
   - **Text Integration:** Text on **war flags**, **shield emblems**, **banners**.

2. **Scenario: Royal/Palace**
   - **Background:** Bright throne room with golden decorations, sunny royal garden.
   - **Extras:** Servants, guards, nobles in the background.
   - **Text Integration:** Text on **royal seals**, **scrolls**, **throne inscriptions**.

3. **Scenario: Revolution/Uprising**
   - **Background:** Town square at dusk (visible), palace gates with torchlight.
   - **Extras:** **Angry crowd of stickmen** with torches and pitchforks.
   - **Text Integration:** Text on **protest banners**, **wanted posters**, **graffiti on walls**.

4. **Scenario: Historical Event/Moment**
   - **Background:** Iconic historical setting with clear lighting (signing ceremony, coronation).
   - **Extras:** Witnesses, historians, important figures.
   - **Text Integration:** Text on **documents**, **stone tablets**, **flags**.

[Text Object Integration (Limit 2-3 Keywords)]
- **QUANTITY RULE:** Strictly limit text to **2-3 CORE KEYWORDS** per scene. Do not use long sentences or too many tags.
- **Readability:** Text must have a **CLEAN OUTLINE**.
- **Placement:** Text on Banners, Royal Seals, Stone Tablets, War Flags.

[Costume & Role - 역사 의상]
- 조선: 한복, 갓 / 로마: 토가, 갑옷 / 중세: 갑옷, 왕관, 드레스
- 왕족: 금색 장식, 왕관 / 농민: 소박한 옷 / 전사: 무기와 갑옷

[Face Expression Guide]
- Use simple cartoon eyes and mouths to clearly show emotions.
""",
        "과학": """
[Core Identity]
- Character: "White circle-faced stickman" with PURE WHITE body and limbs.
- Face Style: Minimalist white round head with eyes and mouth.
- **Critical Rule:** Character MUST have EYES and MOUTH.

[Lighting & Background Guide - BRIGHT & CLEAR]
- **General Tone:** ALWAYS use **Bright, Clear, and Visible** lighting.
- **Avoid:** Pitch black darkness, heavy shadows, or muddy colors.
- **Scenario Lighting:**
    - *Lab/Research:* "Bright White Lab Lighting", "Clean Fluorescent Light".
    - *Space:* "Starry but Visible Space", "Bright Spaceship Interior" (NOT pitch black void).
    - *Disaster:* "Red Warning Lights with Visible Background", "Smoky but Lit Lab".

[Context-Aware Visual Guide (Crucial)]
1. **Scenario: Discovery/Breakthrough**
   - **Background:** Bright clean laboratory, research facility, eureka moment setting.
   - **Extras:** Research team stickmen celebrating or observing.
   - **Text Integration:** Text on **computer monitors**, **hologram displays**, **scientific charts**.

2. **Scenario: Space/Exploration**
   - **Background:** Starry cosmos with visible elements, bright spaceship interior, alien planet with clear sky.
   - **Extras:** Astronaut crew, mission control stickmen on screens.
   - **Text Integration:** Text on **spaceship consoles**, **mission patches**, **floating HUD**.

3. **Scenario: Disaster/Failure**
   - **Background:** Lab with warning lights (visible), malfunctioning equipment, smoky but lit scene.
   - **Extras:** Panicking scientists, evacuation scenes.
   - **Text Integration:** Text on **warning signs**, **error screens**, **scattered papers**.

4. **Scenario: Future/Technology**
   - **Background:** Bright futuristic city, glowing cyber world, well-lit high-tech facility.
   - **Extras:** Robots, AI interfaces, holographic beings.
   - **Text Integration:** Text as **hologram UI**, **laser projections**, **digital billboards**.

[Text Object Integration (Limit 2-3 Keywords)]
- **QUANTITY RULE:** Strictly limit text to **2-3 CORE KEYWORDS** per scene. Do not use long sentences or too many tags.
- **Readability:** Text must have a **CLEAN OUTLINE**.
- **Placement:** Text on Computer Monitors, Hologram UI, Digital Billboards, Warning Signs.

[Costume & Role - 과학 의상]
- 과학자: 흰 가운, 보안경 / 의사: 수술복, 청진기
- 우주비행사: 우주복 / 엔지니어: 작업복, 안전모

[Face Expression Guide]
- Use simple cartoon eyes and mouths to clearly show emotions.
""",
        "커스텀 (직접 입력)": """
[Core Identity]
- Character: "White circle-faced stickman" with PURE WHITE body and limbs.
- Face Style: Minimalist white round head with eyes and mouth.
- **Critical Rule:** Character MUST have EYES and MOUTH.

[Lighting & Background Guide - BRIGHT & CLEAR]
- **General Tone:** ALWAYS use **Bright, Clear, and Visible** lighting.
- **Avoid:** Pitch black darkness, heavy shadows, or muddy colors.
- **Scenario Lighting:**
    - *Positive/Success:* "Bright White Studio", "Sunny Day", "Warm Golden Light".
    - *Negative/Sad:* "Grey Cloudy Daylight", "Dim Indoor Light" (NOT pitch black).
    - *News:* "Bright TV Studio Lighting".

[Context-Aware Visual Guide (Crucial)]
1. **Scenario: Business/Partnership**
   - **Background:** Bright conference room, stage, modern office.
   - **Extras:** Other stickmen (reporters, investors) in the background.
   - **Text Integration:** Text on **podiums**, **flags**, **shirt labels**, **screens**.

2. **Scenario: Crisis/Failure**
   - **Background:** Grey cloudy scene, dim office, rainy street (NOT pitch black).
   - **Extras:** Solo, or with sad figures in the distance.
   - **Text Integration:** Text on **broken signs**, **cracked walls**, **graffiti**.

3. **Scenario: Public Reaction**
   - **Background:** Bright public spaces, streets, gathering places.
   - **Extras:** **Crowd of stickmen** showing reactions.
   - **Text Integration:** Text on **signs**, **thought bubbles**, **ticker boards**.

4. **Scenario: News/Announcement**
   - **Background:** Bright living room with TV, or well-lit news studio.
   - **Extras:** None or news anchor.
   - **Text Integration:** Text on **TV screen banner**.

[Text Object Integration (Limit 2-3 Keywords)]
- **QUANTITY RULE:** Strictly limit text to **2-3 CORE KEYWORDS** per scene. Do not use long sentences or too many tags.
- **Readability:** Text must have a **CLEAN OUTLINE**.
- **Placement:** Text on TV Screen, Signs, Banners, Shirt Labels.

[Costume & Role]
- 각 캐릭터의 직업/역할에 맞는 컬러풀하고 특징적인 의상

[Face Expression Guide]
- Use simple cartoon eyes and mouths to clearly show emotions.
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

    max_workers = st.slider("작업 속도(병렬 수)", 1, 10, 5)

    # [TTS 비활성화] Supertone TTS 설정 제거됨 - 기본값으로 비활성화
    supertone_api_key = ""
    supertone_base_url = DEFAULT_SUPERTONE_URL
    selected_voice_id = ""
    tts_speed = 1.0
    tts_pitch = 0

# ==========================================
# [수정된 UI] 메인 화면: 이미지 생성
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

# [수정됨] 버튼 로직: 제목 입력 기반으로 추천
with col_title_btn:
    st.write("")
    st.write("")
    if st.button("💡 제목 5개 추천", help="입력한 키워드를 바탕으로 제목을 추천합니다.", use_container_width=True):
        current_user_title = st.session_state.get('video_title', "").strip()

        if not api_key:
            st.error("API Key 필요")
        elif not current_user_title:
            st.warning("⚠️ 먼저 '주제'를 입력해주세요.")
        else:
            client = genai.Client(api_key=api_key)
            with st.spinner("AI가 최적의 제목을 고민 중입니다..."):
                title_prompt = f"""
                [Role] You are a YouTube viral marketing expert.
                [Target Topic]
                "{current_user_title}"
                [Task]
                Generate 5 click-bait YouTube video titles based on the Target Topic above.
                사용자가 입력한거랑 최대한 비슷한 제목으로 추천, '몰락'이 들어간 경우 맨 뒤에 몰락으로 끝나게 한다.

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

# ==========================================
# [UI] 메인 화면: 대본 입력 (도입부 + 본문 분리)
# ==========================================
st.divider()
st.subheader("📜 대본 입력 (도입부 / 본문 분리)")

# 도입부 입력
st.markdown("### 🎬 도입부 (인트로)")
st.caption("도입부는 **4~8초(24~48자)** 단위로 의미 기준 분할됩니다. 짧고 임팩트 있는 문장들로 구성하세요.")
intro_input = st.text_area(
    "도입부 대본",
    height=120,
    placeholder="예시:\n한 기업이 삼십년 만에 자동차 시장에 다시 발을 들였습니다. 바로 삼성입니다. 하지만 이번엔 SM5 같은 차를 만들겠다는 게 아닙니다.",
    key="intro_input"
)

st.markdown("---")

# 본문 입력
st.markdown("### 📝 본문")
col_input_opt, col_input_txt = st.columns([1, 3])

with col_input_opt:
    st.info("⏱️ 본문 씬 분할 설정")
    scene_duration = st.slider(
        "한 씬당 목표 글자수",
        min_value=100,
        max_value=500,
        value=370,
        step=10,
        help="본문의 각 씬은 이 글자수 이하로 분할됩니다."
    )
    st.caption(f"약 {scene_duration}자 ≈ {scene_duration // 6}초 분량")

with col_input_txt:
    script_input = st.text_area(
        "본문 대본",
        height=250,
        placeholder="본문 대본을 붙여넣으세요. AI가 문맥을 파악해 자동으로 씬을 나눕니다.\n\n예시:\n삼성이 노리는 건 자동차의 껍데기가 아니라 영혼입니다. 2조 6천억 원, 웬만한 대기업 시가총액에 맞먹는 돈을 베팅했습니다...",
        key="image_gen_input"
    )

# 세션 상태 초기화
if 'generated_results' not in st.session_state:
    st.session_state['generated_results'] = []
if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False
if 'split_scenes' not in st.session_state:
    st.session_state['split_scenes'] = []

# ==========================================
# [NEW] 씬 분할 미리보기 (이미지 생성 전 확인)
# ==========================================
col_split_btn, col_gen_btn = st.columns(2)

with col_split_btn:
    split_btn = st.button("✂️ 씬 분할 미리보기", type="secondary", use_container_width=True)

with col_gen_btn:
    def clear_generated_results():
        st.session_state['generated_results'] = []
    start_btn = st.button("🚀 이미지 생성 시작", type="primary", use_container_width=True, on_click=clear_generated_results)

# [씬 분할 미리보기 처리] - 도입부 + 본문 통합
if split_btn:
    if not api_key:
        st.error("⚠️ Google API Key를 입력해주세요.")
    elif not intro_input and not script_input:
        st.warning("⚠️ 도입부 또는 본문을 입력해주세요.")
    else:
        with st.spinner("🧠 AI가 대본을 분석하여 씬을 나누는 중..."):
            preview_client = genai.Client(api_key=api_key)
            all_scenes = []

            # 1. 도입부 분할 (24~48자, 의미 기준)
            if intro_input and intro_input.strip():
                intro_scenes = split_intro_by_meaning(preview_client, intro_input)
                all_scenes.extend(intro_scenes)
                st.info(f"🎬 도입부: {len(intro_scenes)}개 씬 (4~8초)")

            # 2. 본문 분할 (370자 기준)
            if script_input and script_input.strip():
                main_scenes = split_text_automatically(preview_client, script_input, target_chars=scene_duration)
                all_scenes.extend(main_scenes)
                st.info(f"📝 본문: {len(main_scenes)}개 씬")

            st.session_state['split_scenes'] = all_scenes
        st.success(f"✅ 총 {len(st.session_state['split_scenes'])}개 씬으로 분할되었습니다. (도입부 + 본문)")

# [분할된 씬 표시] - 단일 드롭박스 안에 개별 박스로 표시 (이미지 생성 전에만 표시)
if st.session_state.get('split_scenes') and not st.session_state.get('generated_results'):
    with st.expander(f"🎬 씬 분할 결과 ({len(st.session_state['split_scenes'])}개)", expanded=True):
        for idx, scene_text in enumerate(st.session_state['split_scenes']):
            st.markdown(f"""
            <div style="background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #4CAF50;">
                <strong style="color: #1f77b4;">씬 {idx + 1}</strong> <span style="color: #666;">({len(scene_text)}자)</span>
                <p style="margin-top: 8px; color: #333; white-space: pre-wrap;">{scene_text}</p>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# [이미지 생성 처리] - 도입부 + 본문 통합
if start_btn:
    if not api_key:
        st.error("⚠️ Google API Key를 입력해주세요.")
    elif not intro_input and not script_input:
        st.warning("⚠️ 도입부 또는 본문을 입력해주세요.")
    else:
        # 기존 결과 초기화
        st.session_state['generated_results'] = []
        st.session_state['is_processing'] = True

        # 기존 이미지 파일들 물리적으로 삭제
        if os.path.exists(IMAGE_OUTPUT_DIR):
            shutil.rmtree(IMAGE_OUTPUT_DIR)
        init_folders()

        # [멀티 API 지원] 여러 클라이언트 생성
        clients = []
        for key in api_keys:
            clients.append(genai.Client(api_key=key))

        # 메인 클라이언트 (AI 분할용)
        client = clients[0] if clients else genai.Client(api_key=api_key)

        status_box = st.status("작업 진행 중...", expanded=True)
        progress_bar = st.progress(0)

        # -------------------------------------------------------
        # [핵심] 도입부 + 본문 분할
        # -------------------------------------------------------
        chunks = []

        # 1. 도입부 분할 (24~48자, 의미 기준)
        if intro_input and intro_input.strip():
            status_box.write("🎬 도입부를 의미 단위로 분할하는 중... (4~8초)")
            intro_chunks = split_intro_by_meaning(client, intro_input)
            chunks.extend(intro_chunks)
            status_box.write(f"✅ 도입부: {len(intro_chunks)}개 씬")

        # 2. 본문 분할 (370자 기준)
        if script_input and script_input.strip():
            status_box.write(f"📝 본문을 분할하는 중... (기준: 약 {scene_duration}자)")
            main_chunks = split_text_automatically(client, script_input, target_chars=scene_duration)
            chunks.extend(main_chunks)
            status_box.write(f"✅ 본문: {len(main_chunks)}개 씬")

        total_scenes = len(chunks)

        if total_scenes == 0:
            status_box.update(label="⚠️ 분할 실패.", state="error")
            st.stop()

        status_box.write(f"✅ AI 분석 완료: 총 {total_scenes}개의 장면으로 구성되었습니다.")

        # [중요] 분할된 씬을 session_state에 저장 (결과물 위에 표시하기 위해)
        st.session_state['split_scenes'] = chunks

        # [맥락 주입] 영상 제목이 없다면 첫 문장으로 대체
        current_video_title = st.session_state.get('video_title', "").strip()
        if not current_video_title:
            # 도입부 또는 본문에서 맥락 추출
            context_text = intro_input if intro_input else script_input
            current_video_title = f"Context: {context_text[:200]}..."

        # -------------------------------------------------------
        # 2. 프롬프트 생성 (병렬) - 기존 로직 유지
        # -------------------------------------------------------
        status_box.write(f"📝 씬별 프롬프트 작성 중 ({GEMINI_TEXT_MODEL_NAME})...")
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

        # 3. 이미지 생성 (멀티 API 병렬 처리)
        num_clients = len(clients)
        total_rate_limit = num_clients * 20  # 분당 최대 요청 수 (키당 20개)

        if num_clients > 1:
            status_box.write(f"🎨 이미지 생성 중 ({SELECTED_IMAGE_MODEL}) - {num_clients}개 API 병렬 처리 (분당 최대 {total_rate_limit}개)")
        else:
            status_box.write(f"🎨 이미지 생성 중 ({SELECTED_IMAGE_MODEL})... (API 보호를 위해 천천히 진행됩니다)")

        results = []

        # [멀티 API] API 키 개수에 따라 worker 수와 대기 시간 조절 (안정성 강화)
        # 키 1개: 4초 간격 (분당 15개, 안전 마진)
        # 키 2개: 3초 간격 (분당 20개 x 2 = 40개 가능하지만 안전하게)
        # 키 3개: 2초 간격
        # 키 4개: 1.5초 간격
        sleep_interval = max(4.0 / num_clients, 1.5)  # 최소 1.5초 보장
        adjusted_workers = min(max_workers, num_clients * 3)  # API 키당 3개 worker로 축소

        with ThreadPoolExecutor(max_workers=adjusted_workers) as executor:
            future_to_meta = {}
            request_count = 0

            for s_num, prompt_text in prompts:
                idx = s_num - 1
                orig_text = chunks[idx]
                fname = make_filename(s_num, orig_text)

                # [라운드 로빈] 클라이언트 순환 배정
                current_client = clients[request_count % num_clients]

                # [속도 조절] API 키 개수에 맞춰 대기
                time.sleep(sleep_interval)

                # [80개 제한] 멀티 API 사용 시 80개마다 1분 대기
                if num_clients > 1 and request_count > 0 and request_count % total_rate_limit == 0:
                    status_box.write(f"⏳ API 제한 보호: {request_count}개 완료, 60초 대기 중...")
                    time.sleep(60)

                future = executor.submit(generate_image, current_client, prompt_text, fname, IMAGE_OUTPUT_DIR, SELECTED_IMAGE_MODEL)
                future_to_meta[future] = (s_num, fname, orig_text, prompt_text)
                request_count += 1
            
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

    # [NEW] 결과물 위에 씬 분할 결과 표시 - 단일 드롭박스 안에 개별 박스
    if st.session_state.get('split_scenes'):
        with st.expander(f"📋 씬 분할 결과 ({len(st.session_state['split_scenes'])}개)", expanded=False):
            for idx, scene_text in enumerate(st.session_state['split_scenes']):
                st.markdown(f"""
                <div style="background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #4CAF50;">
                    <strong style="color: #1f77b4;">씬 {idx + 1}</strong> <span style="color: #666;">({len(scene_text)}자)</span>
                    <p style="margin-top: 8px; color: #333; white-space: pre-wrap;">{scene_text}</p>
                </div>
                """, unsafe_allow_html=True)
        st.divider()

    st.header(f"📸 결과물 ({len(st.session_state['generated_results'])}장)")
    
    # ------------------------------------------------
    # 1. 일괄 작업 버튼 영역
    # ------------------------------------------------
    st.write("---")
    zip_data = create_zip_buffer(IMAGE_OUTPUT_DIR)
    st.download_button("📦 전체 이미지 ZIP 다운로드", data=zip_data, file_name="all_images.zip", mime="application/zip", use_container_width=True)

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

            # [오른쪽] 정보
            with cols[1]:
                st.subheader(f"Scene {item['scene']:02d}")
                st.caption(f"파일명: {item['filename']}")
                st.write(f"**대본:** {item['script']}")

                with st.expander("프롬프트 확인"):
                    st.text(item['prompt'])
                try:
                    with open(item['path'], "rb") as file:
                        st.download_button("⬇️ 이미지 저장", data=file, file_name=item['filename'], mime="image/png", key=f"btn_down_{item['scene']}")
                except: pass



