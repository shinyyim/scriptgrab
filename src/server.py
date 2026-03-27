#!/usr/bin/env python3
"""Flask server wrapping yt_transcribe for the web frontend."""

import os
import uuid
import subprocess
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import mlx_whisper

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__))))
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max upload
CORS(app)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
UPLOAD_DIR = os.path.join(PROJECT_DIR, "uploads")
DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo"
    "/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


def download_youtube_audio(url, temp_audio, start=None, end=None):
    import yt_dlp
    from yt_dlp.utils import download_range_func

    opts = {
        "format": "m4a/bestaudio/best",
        "outtmpl": temp_audio.replace(".m4a", ".%(ext)s"),
        "quiet": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}
        ],
    }
    if start or end:
        s = parse_time(start) if start else None
        e = parse_time(end) if end else None
        ranges = [(s, e)]
        opts["download_ranges"] = download_range_func(chapters=None, ranges=ranges)
        opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return info.get("title", "transcript")


def parse_time(t):
    """Parse time string like '1:30', '01:02:30', or '90' into seconds."""
    if not t:
        return 0
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def trim_audio(audio_path, output_path, start=None, end=None):
    """Trim an audio file using ffmpeg."""
    cmd = ["ffmpeg", "-i", audio_path]
    if start:
        cmd += ["-ss", str(parse_time(start))]
    if end:
        cmd += ["-to", str(parse_time(end))]
    cmd += ["-y", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Trim failed: {result.stderr[-300:]}")
    return output_path


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


def extract_audio(filepath, temp_audio):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in AUDIO_EXTS:
        # Audio files can be transcribed directly, no ffmpeg needed
        return filepath
    # Check if file has an audio stream
    probe = subprocess.run(
        ["ffprobe", "-i", filepath, "-show_streams", "-select_streams", "a", "-loglevel", "quiet"],
        capture_output=True, text=True,
    )
    if not probe.stdout.strip():
        raise RuntimeError("This file has no audio track. Cannot transcribe video-only files.")

    # Video files: extract audio as wav (universally supported)
    temp_wav = temp_audio.rsplit(".", 1)[0] + ".wav"
    result = subprocess.run(
        ["ffmpeg", "-i", filepath, "-vn", "-ar", "16000", "-ac", "1", "-y", temp_wav],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-500:]}")
    return temp_wav


def format_ts(seconds):
    """Convert seconds to MM:SS or HH:MM:SS."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def transcribe_audio(audio_path, language=None):
    kwargs = {"path_or_hf_repo": DEFAULT_MODEL}
    if language:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(audio_path, **kwargs)
    segments = result.get("segments", [])
    # Build timestamped text with line breaks
    lines = []
    for seg in segments:
        ts = format_ts(seg["start"])
        text = seg["text"].strip()
        lines.append(f"[{ts}] {text}")
    timestamped = "\n\n".join(lines)
    plain = result.get("text", "")
    return {"timestamped": timestamped, "plain": plain}


def translate_text(text, target_lang="ko"):
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target=target_lang)
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    return " ".join(translator.translate(chunk) for chunk in chunks)


def safe_filename(title, max_len=40):
    return "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip()[:max_len]


def load_api_key():
    """Load Anthropic API key from .env file."""
    env_path = os.path.join(PROJECT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.strip().split("=", 1)[1]
    return os.environ.get("ANTHROPIC_API_KEY", "")


def summarize_with_claude(text):
    """Summarize transcript using Claude API."""
    import requests
    api_key = load_api_key()
    if not api_key:
        return "(Summary unavailable: no API key)"

    # Truncate to ~6000 words
    words = text.split()
    if len(words) > 6000:
        text = " ".join(words[:6000])

    try:
        resp = requests.post("https://api.anthropic.com/v1/messages", headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }, json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": f"""이 트랜스크립트를 요약해줘. 포인트를 잡아서 논리 흐름이 있게 정리해.

규칙:
- 핵심 메시지를 먼저 한 문장으로
- 주제별로 3-6개 섹션으로 나눠서 각각 2-3줄로 요약
- 마지막에 한 줄 요약
- Notable Quotes 3-5개 (영어 원문)
- 한국어로 작성, 마크다운 포맷

TRANSCRIPT:
{text}"""}],
        }, timeout=60)
        data = resp.json()
        return data["content"][0]["text"]
    except Exception as e:
        return f"(Summary unavailable: {e})"


def make_output_name(title):
    """Generate filename like 20260326_Some_English_Title"""
    from deep_translator import GoogleTranslator

    date_str = datetime.now().strftime("%Y%m%d")
    # Translate title to English if it contains non-ASCII
    en_title = title
    if any(ord(c) > 127 for c in title):
        try:
            en_title = GoogleTranslator(source="auto", target="en").translate(title)
        except Exception:
            en_title = title
    clean = safe_filename(en_title, max_len=50).replace(" ", "_")
    return f"{date_str}_{clean}"


REF_DIR = os.path.join(PROJECT_DIR, "obsidian", "references")


def extract_key_points(text):
    """Extract key points from transcript, grouped by topic chunks."""
    import re
    # Split into sentences
    raw = text.replace("\n", " ").strip()
    sentences = re.split(r'(?<=[.!?])\s+', raw)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return "- (No content to summarize)\n"

    # Group sentences into chunks of ~5 sentences (rough topic segments)
    chunk_size = max(5, len(sentences) // 6)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunks.append(sentences[i:i + chunk_size])

    # From each chunk, pick the longest sentence (usually the most informative)
    # and the first sentence (usually introduces the topic)
    result = ""
    for idx, chunk in enumerate(chunks[:6]):  # max 6 sections
        if not chunk:
            continue
        # First sentence = topic intro
        intro = chunk[0]
        # Longest sentence in chunk = key claim
        key = max(chunk, key=len)

        result += f"- {intro}\n"
        if key != intro and len(key) > 30:
            result += f"- {key}\n"
        result += "\n"

    return result.strip()


def save_to_reference(title, en_file, text, url=None, ko_file=None):
    """Save transcript as a reference note in Obsidian."""
    os.makedirs(REF_DIR, exist_ok=True)
    today = datetime.now()
    date_str = today.strftime("%Y-%m-%d")
    date_prefix = today.strftime("%Y%m%d")

    # English title for filename
    from deep_translator import GoogleTranslator
    en_title = title
    if any(ord(c) > 127 for c in title):
        try:
            en_title = GoogleTranslator(source="auto", target="en").translate(title)
        except Exception:
            en_title = title
    clean_title = safe_filename(en_title, max_len=50).replace(" ", "_")
    ref_filename = f"{date_prefix}_{clean_title}.md"
    ref_path = os.path.join(REF_DIR, ref_filename)

    word_count = len(text.split())
    summary = summarize_with_claude(text)

    content = f"""---
created: "{date_str}"
categories:
  - reference
  - transcript
tags:
  - youtube
author: ""
source: "{url or ''}"
---

# {title}

## Info
- **Source**: {url or 'Local file'}
- **Words**: {word_count:,}
- **Files**: [[{en_file}]]{'  |  [[' + ko_file + ']]' if ko_file else ''}

{summary}

## Notes
-

## Connections
- Related: [[]]
"""
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write(content)


@app.route("/")
def index():
    return send_from_directory(PROJECT_DIR, "index.html")


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    job_id = uuid.uuid4().hex[:8]
    temp_audio = os.path.join(UPLOAD_DIR, f"{job_id}.m4a")
    uploaded_file = None
    audio_to_transcribe = None

    try:
        url = request.form.get("url", "").strip()
        file = request.files.get("file")
        language = request.form.get("language", "").strip() or None
        translate_to = request.form.get("translate", "").strip() or None
        start = request.form.get("start", "").strip() or None
        end = request.form.get("end", "").strip() or None

        if file and file.filename:
            ext = os.path.splitext(file.filename)[1]
            uploaded_file = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
            file.save(uploaded_file)
            title = os.path.splitext(file.filename)[0]
            result_path = extract_audio(uploaded_file, temp_audio)
            audio_to_transcribe = result_path if result_path != uploaded_file else uploaded_file
            # Trim local file if time range specified
            if start or end:
                trimmed = os.path.join(UPLOAD_DIR, f"{job_id}_trim.wav")
                trim_audio(audio_to_transcribe, trimmed, start, end)
                audio_to_transcribe = trimmed
        elif url:
            title = download_youtube_audio(url, temp_audio, start, end)
            audio_to_transcribe = temp_audio
        else:
            return jsonify({"error": "No URL or file provided"}), 400

        transcript = transcribe_audio(audio_to_transcribe, language=language)
        timestamped = transcript["timestamped"]
        plain_text = transcript["plain"]
        source_url = url or ""

        base = make_output_name(title)

        # Build file header
        header = f"# {title}\n"
        if source_url:
            header += f"# Source: {source_url}\n"
        header += f"# Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"

        # Save EN with timestamps
        en_file = f"{base}_EN.txt"
        out_path = os.path.join(OUTPUT_DIR, en_file)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header + timestamped)

        result = {"title": title, "text": timestamped, "file": en_file}

        if translate_to:
            translated = translate_text(plain_text, target_lang=translate_to)
            # Add line breaks to translated text (~200 chars per paragraph)
            sentences = translated.replace(". ", ".\n\n").replace("? ", "?\n\n").replace("! ", "!\n\n")
            ko_file = f"{base}_KO.txt"
            ko_path = os.path.join(OUTPUT_DIR, ko_file)
            with open(ko_path, "w", encoding="utf-8") as f:
                f.write(header + sentences)
            result["translated"] = sentences
            result["translated_file"] = ko_file

        # Summarize with Ollama
        summary = summarize_with_claude(plain_text)
        result["summary"] = summary

        # Save to Obsidian references
        save_to_reference(title, en_file, plain_text, url=source_url or None, ko_file=result.get("translated_file"))

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        temp_wav = temp_audio.rsplit(".", 1)[0] + ".wav"
        trimmed = os.path.join(UPLOAD_DIR, f"{job_id}_trim.wav")
        for f in [temp_audio, temp_wav, trimmed, uploaded_file]:
            if f and os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    print(f"ScriptGrab server running at http://localhost:5001")
    print(f"Output dir: {OUTPUT_DIR}")
    app.run(host="0.0.0.0", port=5001, debug=True)
