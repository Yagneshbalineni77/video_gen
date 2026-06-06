# Faceless YouTube Pipeline

Fully autonomous end-to-end pipeline that finds trending topics, writes scripts,
generates media, assembles a video, and publishes to YouTube — daily, on autopilot.

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
Scraper   Writer    Assets    Editor    Publisher
YouTube   Claude    ElevenLabs FFmpeg   YouTube
Data API  Sonnet    + Luma AI  + Whisper Data API
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <your-repo>
cd faceless
pip install -r requirements.txt
```

FFmpeg must be installed and on your PATH:
- **Windows**: [ffmpeg.org/download.html](https://ffmpeg.org/download.html) → add to PATH
- **Ubuntu/CI**: `sudo apt-get install -y ffmpeg`

### 2. Configure

```bash
cp .env.example .env
# Fill in all API keys in .env
```

Required API keys:
| Key | Where to get |
|-----|-------------|
| `YOUTUBE_API_KEY` | [Google Cloud Console](https://console.cloud.google.com) → YouTube Data API v3 |
| `YOUTUBE_CLIENT_SECRETS_PATH` | Google Cloud Console → OAuth 2.0 → Desktop app → download JSON |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `ELEVENLABS_API_KEY` | [elevenlabs.io](https://elevenlabs.io) |
| `LUMAAI_API_KEY` | [lumalabs.ai](https://lumalabs.ai) (or set `VISUAL_PROVIDER=fal` / `none`) |

### 3. First run (OAuth consent)

```bash
python main.py --dry-run
```

This opens a browser for YouTube OAuth on first run. A token is saved to
`credentials/youtube_token.json` — keep it secret.

### 4. Full run

```bash
python main.py
```

### 5. Run individual phases (for testing)

```bash
python main.py --phase 1   # trend scraper only
python main.py --phase 2   # script writer (needs output/trends.json)
python main.py --phase 3   # asset factory (needs output/script.json)
python main.py --phase 4   # video editor  (needs output/audio/ + output/video/)
python main.py --phase 5   # publisher     (needs output/renders/final_render.mp4)
```

---

## GitHub Actions (Automated Daily Runs)

Add these **Secrets** to your repository (`Settings → Secrets and variables → Actions`):

| Secret | Value |
|--------|-------|
| `YOUTUBE_API_KEY` | Your YouTube Data API key |
| `YOUTUBE_TOKEN_JSON` | Contents of `credentials/youtube_token.json` |
| `YOUTUBE_CLIENT_SECRETS_JSON` | Contents of `credentials/client_secrets.json` |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ELEVENLABS_API_KEY` | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID |
| `LUMAAI_API_KEY` | Luma AI key |

Optional **Variables** (`Settings → Secrets and variables → Variables`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NICHE_KEYWORDS` | `dark history,untold history` | Comma-separated topic niches |
| `VISUAL_PROVIDER` | `luma` | `luma`, `fal`, or `none` |

The workflow runs daily at 08:00 UTC. Trigger manually via **Actions → Run workflow**.

---

## Project Structure

```
faceless/
├── pipeline/
│   ├── phase1_scraper.py    # YouTube trend scraping
│   ├── phase2_writer.py     # Claude script generation
│   ├── phase3_assets.py     # ElevenLabs audio + visual gen
│   ├── phase4_editor.py     # FFmpeg assembly + Whisper subs
│   ├── phase5_publisher.py  # YouTube upload
│   └── orchestrator.py      # Runs all phases in sequence
├── schemas/models.py         # Pydantic data models
├── config/settings.py        # All configuration
├── utils/
│   ├── ffmpeg_helpers.py     # FFmpeg subprocess wrappers
│   └── whisper_captions.py  # faster-whisper → ASS subtitles
├── output/                   # Generated assets (gitignored)
├── credentials/              # OAuth tokens (gitignored)
├── assets/bgm/               # Put royalty-free .mp3 tracks here
└── .github/workflows/        # Daily cron job
```

---

## Output Files

After each run:

| File | Description |
|------|-------------|
| `output/trends.json` | Top trending candidates |
| `output/script.json` | Full structured video script |
| `output/audio/scene_N.mp3` | ElevenLabs narration per scene |
| `output/video/scene_N_raw.mp4` | Raw generated video clips |
| `output/renders/final_render.mp4` | Final assembled video |
| `output/thumbnails/thumbnail.jpg` | YouTube thumbnail |
| `output/pipeline_state.json` | Full run state (for debugging) |

---

## Visual Provider Options

| `VISUAL_PROVIDER` | Quality | Cost | Notes |
|-------------------|---------|------|-------|
| `luma` | Excellent | ~$0.05/clip | Async, ~30s per clip |
| `fal` | Good | ~$0.02/clip | Fast, Kling v1.6 model |
| `none` | N/A | Free | Static black frames (dev/testing) |

---

## Adding Background Music

Drop royalty-free `.mp3` files into `assets/bgm/`. The pipeline randomly picks
one per run. Good sources: [pixabay.com/music](https://pixabay.com/music),
[freemusicarchive.org](https://freemusicarchive.org).
