import mlx_whisper
import yt_dlp
import os
from huggingface_hub import login

# Set HF_TOKEN env var or paste your token below
login(token=os.environ.get("HF_TOKEN", ""))

def get_audio_and_transcribe(url):
    print("\n🌐 유튜브 서버 연결 중...")
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
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'transcript')
            audio_file = "temp_audio.m4a"

        print(f"🚀 M4 Pro 가동: [ {video_title} ] 변환 중...")

        # 모델 경로를 최신 버전으로 변경했습니다.
        result = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo", 
            language="ko"
        )

        output_file = f"{video_title[:20]}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result['text'])

        print(f"\n✨ 완료! 파일명: {output_file}")

    except Exception as e:
        print(f"❌ 오류: {e}")
    finally:
        if os.path.exists("temp_audio.m4a"):
            os.remove("temp_audio.m4a")

if __name__ == "__main__":
    target_url = input("유튜브 URL: ")
    get_audio_and_transcribe(target_url)