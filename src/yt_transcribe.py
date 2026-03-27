#!/usr/bin/env python3
"""CLI tool to transcribe YouTube videos using MLX Whisper and optionally translate."""

import argparse
import os
import sys

import mlx_whisper
import yt_dlp


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "output")
TEMP_AUDIO = "temp_audio.m4a"
DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo"
    "/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86"
)


def is_url(source):
    return source.startswith(("http://", "https://", "www."))


def download_audio(url):
    print("\nConnecting to YouTube...")
    opts = {
        "format": "m4a/bestaudio/best",
        "outtmpl": "temp_audio.%(ext)s",
        "quiet": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}
        ],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return info.get("title", "transcript")


def extract_audio(filepath):
    import subprocess

    print(f"\nExtracting audio from {os.path.basename(filepath)}...")
    subprocess.run(
        ["ffmpeg", "-i", filepath, "-vn", "-acodec", "aac", "-y", TEMP_AUDIO],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return os.path.splitext(os.path.basename(filepath))[0]


def transcribe(model, language=None):
    print(f"Transcribing with {os.path.basename(model)}...")
    kwargs = {"path_or_hf_repo": model}
    if language:
        kwargs["language"] = language
    return mlx_whisper.transcribe(TEMP_AUDIO, **kwargs)["text"]


def translate_text(text, target_lang="ko"):
    from deep_translator import GoogleTranslator

    print(f"Translating to [{target_lang}]...")
    translator = GoogleTranslator(source="auto", target=target_lang)
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    return " ".join(translator.translate(chunk) for chunk in chunks)


def safe_filename(title, max_len=40):
    return "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip()[:max_len]


def save(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos using MLX Whisper")
    parser.add_argument("source", help="YouTube URL or local video/audio file path")
    parser.add_argument("-l", "--language", default=None, help="Source language code (e.g. ko, en, ja). Auto-detect if omitted.")
    parser.add_argument("-t", "--translate", metavar="LANG", nargs="?", const="ko", default=None,
                        help="Translate transcript to LANG (default: ko)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Local model path (default: whisper-large-v3-turbo)")
    parser.add_argument("-o", "--output", default=None, help="Output filename (without extension)")
    args = parser.parse_args()

    is_local = not is_url(args.source)

    try:
        if is_local:
            if not os.path.exists(args.source):
                print(f"File not found: {args.source}", file=sys.stderr)
                sys.exit(1)
            title = extract_audio(args.source)
        else:
            title = download_audio(args.source)

        text = transcribe(args.model, language=args.language)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = args.output or safe_filename(title)

        save(text, os.path.join(OUTPUT_DIR, f"{base}.txt"))

        if args.translate:
            translated = translate_text(text, target_lang=args.translate)
            save(translated, os.path.join(OUTPUT_DIR, f"{base}_{args.translate}.txt"))

        print("\nDone!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(TEMP_AUDIO):
            os.remove(TEMP_AUDIO)


if __name__ == "__main__":
    main()
