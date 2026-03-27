import mlx_whisper
import yt_dlp
import os
from huggingface_hub import login
from deep_translator import GoogleTranslator

# 1. 본인의 허깅페이스 토큰을 여기에 넣으세요 (hf_... 형식)
YOUR_HF_TOKEN = "여기에_토큰을_입력하세요"

def translate_text(text, target_lang='ko'):
    print(f"🌐 한글로 번역 중 (Google Translator)...")
    translator = GoogleTranslator(source='auto', target=target_lang)
    # 글자 수 제한 해결을 위해 4000자씩 분할 번역
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    return " ".join(translated_chunks)

def save_result(title, content, suffix):
    clean_title = "".join([c for c in title if c.isalnum() or c in (' ', '_')]).strip()
    filename = f"{clean_title[:20]}_{suffix}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

def process_youtube(url):
    # 로그인 수행
    if "hf_" in YOUR_HF_TOKEN:
        login(token=YOUR_HF_TOKEN)

    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("\n🌐 유튜브 서버 연결 중...")
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'video_script')
            audio_file = "temp_audio.m4a"

        print(f"🎙️ 음성 인식 및 스크립트 추출 중 (M4 Pro GPU)...")
        # 모델은 성능이 입증된 large-v3-turbo를 사용합니다.
        result = mlx_whisper.transcribe(
            audio_file, 
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
        )
        en_script = result['text']
        
        # 1. 영어 스크립트 저장
        en_file = save_result(video_title, en_script, "EN_Full")
        print(f"📄 영어 원문 저장 완료: {en_file}")
        
        # 2. 한글 번역 진행
        ko_script = translate_text(en_script)
        ko_file = save_result(video_title, ko_script, "KO_Full")
        print(f"📄 한글 번역본 저장 완료: {ko_file}")

        print(f"\n✨ 모든 작업이 성공적으로 완료되었습니다!")
        return en_script

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        if os.path.exists("temp_audio.m4a"):
            os.remove("temp_audio.m4a")

if __name__ == "__main__":
    target_url = input("🔗 분석할 유튜브 URL을 입력하세요: ").strip()
    if target_url:
        process_youtube(target_url)