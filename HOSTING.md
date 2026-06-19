# Hosting — Dextora Creator

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
| `SIGNUP_INVITE_CODE` | `dextora-invite` | Shared code required to register. Change it; share with your team. |
| `ADMIN_EMAILS` | — | Comma-separated emails granted admin (see all jobs) on signup. |
| `JWT_SECRET` | auto | Login token signing key; auto-generated + persisted if unset. |
| `WATERMARK_TEXT` | `Dextora` | Brand watermark burned top-right; `""` disables. |
| `GEMINI_API_KEYS` | — | Key pool: `key1,key2,…` → round-robins API calls to dodge per-key rate limits. Keys must be from **separate projects** to multiply quota. Falls back to `GEMINI_API_KEY`. |
| `SKIP_VEO` | `false` | Fast mode: skip Veo motion, build from Imagen keyframes + Ken Burns (~3× faster, no motion). |
| `GEMINI_IMAGE_MODEL` | `imagen-4.0-ultra-generate-001` | Keyframe model; set to `imagen-4.0-generate-001` for faster/cheaper. |
| `PORTAL_API_KEY` | — | If set, the portal must send it as `X-API-Key` on `/api/v1/videos`. Blank = open. |
| `PORTAL_API_URL` | dextora.org curriculum endpoint | Where the curriculum bridge pulls authored scripts from. |

## Curriculum bridge (education-portal integration)
The portal authors scripts; this app generates the videos and serves them keyed by
the portal's own script `id`. Endpoints the portal calls are documented in
**`PORTAL_API.md`** (with an importable `dextora_portal_api.postman_collection.json`):
- `GET /api/v1/videos?ids=…` / `?class_id&subject_id&chapter_id` — finished video URLs + status
- `GET /api/v1/videos/{id}/file` — the MP4
- `POST /api/v1/bridge/run` (admin) — generate a whole chapter (EN + HI supported)

The store (`output/portal_videos.db`) persists with the mounted `output/` volume.

## Accounts / login
The web UI requires login. Users **sign up with the shared `SIGNUP_INVITE_CODE`**
(email + password), then log in — JWT sessions, bcrypt-hashed passwords, stored in
a SQLite file under `output/` (persists with the mounted volume). Each user sees only
their own jobs; `ADMIN_EMAILS` accounts see all. No per-user generation cap.

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
