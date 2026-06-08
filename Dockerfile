# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# Faceless Video Studio — self-contained image.
# Bundles every system dependency the pipeline needs so it "just runs":
#   ffmpeg (assembly+mastering), Chromium (code_animator), Noto Devanagari +
#   Liberation fonts (subtitles), and a pre-cached Whisper model.
#
# Build:  docker build -t faceless-studio .
# Run:    docker run -p 8000:8000 -e GEMINI_API_KEY=YOUR_KEY \
#                 -v "$(pwd)/output:/app/output" faceless-studio
# Open:   http://localhost:8000
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium \
    MAX_CONCURRENT_JOBS=10

# ── system dependencies ───────────────────────────────────────────────────────
#   ffmpeg            → all video/audio assembly, mastering, subtitle burn
#   chromium          → code_animator (renders GSAP HTML → frames)
#   fonts-noto-core   → Noto Sans Devanagari (Hindi subtitles) + wide coverage
#   fonts-liberation  → Arial-metric substitute for Latin subtitles
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        chromium \
        fonts-noto-core \
        fonts-liberation \
        ca-certificates \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── python dependencies (own layer so code edits don't reinstall) ─────────────
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── pre-cache the Whisper "base" model so the first render never stalls ───────
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# ── application source ────────────────────────────────────────────────────────
COPY . .

# Generated videos + the job registry live here — mount a volume to persist them.
VOLUME ["/app/output"]

EXPOSE 8000

# Host 0.0.0.0 is correct inside a container; only the published port is exposed.
# GEMINI_API_KEY must be supplied at runtime (never baked into the image).
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
