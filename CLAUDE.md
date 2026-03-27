# YouTube Script Transcriber

## Project Structure
```
05.youtube_script/
├── src/              # Python scripts
├── output/           # Transcription output files
├── docs/             # Documentation
├── obsidian/         # → symlink to Obsidian vault
└── CLAUDE.md
```

## Main Tool
`src/yt_transcribe.py` — CLI tool for transcribing YouTube videos or local video/audio files using MLX Whisper (Apple Silicon optimized).

## Usage
```bash
python src/yt_transcribe.py <url-or-file> [-l language] [-t [lang]] [-m model] [-o output]
```

## Dependencies
- mlx_whisper
- yt_dlp
- deep_translator (for translation)
- ffmpeg (for local video/audio extraction)

## Model
Local: `~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo`
