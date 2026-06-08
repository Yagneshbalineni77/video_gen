# Hosting — Faceless Video Studio

The whole app (FastAPI backend + web UI + the full video pipeline) ships in one
Docker image. The only thing you must supply is a **Gemini API key**.

## Prerequisites
- Docker (and optionally Docker Compose v2)
- A Gemini API key → https://aistudio.google.com/app/apikey

## Quick start (Docker Compose — recommended)
```bash
# 1. provide your key (either export it or put it in a .env file next to docker-compose.yml)
echo "GEMINI_API_KEY=YOUR_KEY_HERE" > .env

# 2. build + run
docker compose up --build

# 3. open the UI
#    http://localhost:8000
```
Generated videos persist in `./output` on the host (mounted volume), and the job
history survives restarts.

## Quick start (plain Docker)
```bash
docker build -t faceless-studio .

docker run -p 8000:8000 \
  -e GEMINI_API_KEY=YOUR_KEY_HERE \
  -e MAX_CONCURRENT_JOBS=10 \
  -v "$(pwd)/output:/app/output" \
  faceless-studio
# → http://localhost:8000
```

## Configuration (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — (**required**) | Gemini key for script, TTS, Imagen, Veo |
| `MAX_CONCURRENT_JOBS` | `10` | Videos rendered in parallel; the rest queue. Total concurrent Veo calls ≈ this × 2 (auto-tuned). |
| `TARGET_TOTAL_VEO` | `20` | Target total concurrent Veo/Imagen calls used to derive per-job concurrency. |

Raising `MAX_CONCURRENT_JOBS` much past 10 on a standard AI-Studio key will hit
API rate limits (the pipeline degrades gracefully, but throughput drops). For a
genuine higher scale, use a Vertex AI project with provisioned Veo/Imagen quota.

## What's inside the image (so it "just runs")
- **ffmpeg** — video/audio assembly, loudness mastering, subtitle burn
- **Chromium** — the `code_animator` scene path (GSAP HTML → frames), run with `--no-sandbox`
- **Noto Sans Devanagari + Liberation fonts** — correct Hindi + Latin subtitles
- **Whisper "base" model** — pre-cached at build, so the first render doesn't stall downloading

## Notes
- Nothing secret is baked into the image — the API key is passed at runtime only.
- First container start is instant; each video takes a few minutes (mostly Veo render time).
- Health/usage endpoints: `GET /api/stats`, `GET /api/jobs`.
