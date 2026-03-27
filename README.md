# ScriptGrab

Video-to-text transcription tool powered by MLX Whisper on Apple Silicon.

Paste a YouTube URL or upload a local video/audio file — get a clean transcript with optional translation in seconds.

## Features

- **YouTube & local files** — supports mp4, mkv, mov, mp3, wav, m4a
- **Time range clipping** — transcribe only the part you need
- **15+ languages** — auto-detect source, translate to Korean, Japanese, Chinese, Spanish, etc.
- **Apple Silicon native** — MLX Whisper large-v3 turbo runs on your M-series GPU
- **Timestamped output** — `[M:SS]` markers per segment
- **Obsidian integration** — auto-generates reference notes in your vault
- **Fully local** — nothing leaves your machine

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ffmpeg
- Conda environment with: `mlx_whisper`, `yt-dlp`, `flask`, `flask-cors`, `deep_translator`

## Setup

```bash
# Clone
git clone https://github.com/shinyyim/scriptgrab.git
cd scriptgrab

# Install dependencies (conda)
conda create -n scriptgrab python=3.11
conda activate scriptgrab
pip install mlx-whisper yt-dlp flask flask-cors deep_translator

# Download the model (first run only)
python -c "import mlx_whisper; mlx_whisper.transcribe('/dev/null', path_or_hf_repo='mlx-community/whisper-large-v3-turbo')" 2>/dev/null || true
```

## Usage

### Web UI

```bash
conda activate scriptgrab
python src/server.py
```

Open `http://localhost:5001`

### CLI

```bash
# YouTube URL
python src/yt_transcribe.py "https://youtube.com/watch?v=..." -t ko

# Local file
python src/yt_transcribe.py ./video.mp4 -l en -t ko

# Time range (CLI uses yt-dlp ranges)
python src/yt_transcribe.py "URL" -t ja
```

| Flag | Description |
|------|-------------|
| `-l` | Source language (e.g. `ko`, `en`, `ja`). Auto-detect if omitted |
| `-t` | Translate to language (default: `ko`) |
| `-m` | Local model path |
| `-o` | Custom output filename |

## Output

Transcripts are saved to `output/` with the format:

```
20260326_Video_Title_EN.txt   # Original with timestamps
20260326_Video_Title_KO.txt   # Translated
```

## Project Structure

```
scriptgrab/
├── index.html          # Web UI
├── src/
│   ├── server.py       # Flask backend
│   ├── yt_transcribe.py # CLI tool
│   ├── whisper_yt.py    # Original script (Korean)
│   └── whisper_translate.py # Original script (EN+KO)
├── output/             # Transcription files
└── obsidian/           # Obsidian vault (symlink)
    └── references/     # Auto-generated reference notes
```

## License

MIT
