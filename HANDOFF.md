# HANDOFF — pick up exactly where we left off

This is the running context for the project so any new machine / fresh session can
continue without re-discovering everything. (Pairs with `git log` and `HOSTING.md`.)

## What this is
Faceless educational video generator: **prompt + style + language → narrated,
subtitled MP4**. Pipeline = Phase 2 (Gemini script) → Phase 3 (Gemini TTS +
Imagen keyframe + Veo clip) → Phase 4 (ffmpeg assemble + subtitles). Web product
in `server.py` (FastAPI) + `frontend/` (React). Parallel batches via `produce.py`.

## How to run (on the new server)
```bash
git clone https://github.com/Yagneshbalineni77/video_gen.git && cd video_gen
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# system deps: ffmpeg, chromium, fonts-noto-core (Devanagari). Or just use Docker.
echo "GEMINI_API_KEY=YOUR_KEY" > .env        # required; not in git
uvicorn server:app --host 127.0.0.1 --port 8000   # → http://localhost:8000
```
Docker (preferred for the colleague): `docker compose up --build` (see HOSTING.md).

## Key knobs (env)
- `GEMINI_API_KEY` (required), `GEMINI_IMAGE_MODEL` (default Imagen 4 **Ultra** — best text),
  `GEMINI_SCRIPT_MODEL` (flash; set `gemini-2.5-pro` for higher script quality),
  `MAX_CONCURRENT_JOBS` (default 10; set ≈ cores÷2), `FFMPEG_PRESET` (veryfast),
  `NARRATION_LANGUAGE`, `VISUAL_STYLE`.

## What's DONE (v2 — this commit)
- Resilient pipeline: every scene path has a graceful fallback; no scene crashes/blanks a video.
- 10-at-a-time validated: API side handles 10 parallel with **0 rate-limits**.
- Job queue + persistence (survives restart), live progress, Single + Batch UI.
- 5 narration languages (Indian/British/American English, Hindi, Hinglish) — Hindi
  uses known-text subtitles + Noto Devanagari font.
- TTS: per-style voice + directed delivery + loudness mastering.
- Subtitles built from KNOWN script text (no ASR errors).
- Imagen 4 Ultra for reliable in-frame text/formulas.
- Docker packaging complete.

## CRITICAL hardware lesson (why the last 10-batch felt "stuck")
The pipeline is **two-tier**:
- Phase 2/3 (script/voice/Veo) run on **Google's servers** — local CPU idle, 10× is fine.
- Phase 4 (ffmpeg assembly) runs on **YOUR CPU** — CPU-bound.
The old sandbox had only **8 cores**, so 10 simultaneous assemblies thrashed (load ~70).
**On the 32-core box this disappears.** Rule: `MAX_CONCURRENT_JOBS ≈ cores ÷ 2`.

## OPEN / NEXT (where we stopped)
1. DONE — pushed to GitHub (Yagneshbalineni77/video_gen, branch video_gen).
2. DONE — **Formula-overlay**: Phase 2 emits `overlay_text` per scene; Phase 4
   burns it as a pixel-perfect graphic (`_render_overlay` + `overlay_image_on_video`).
   Formulas (F=ma, CO₂) are now model-independent and always correct.
3. Optional: assembly-slot throttle so Phase 4 auto-limits to cores regardless of MAX_CONCURRENT_JOBS.
4. Optional: Vertex AI provisioned quota for true >10× (AI-Studio key caps Imagen RPM).
5. TODO (user): mirror to dextoraai-dev/dextora_micro_services branch video_gen_V2
   (run locally — sandbox safety guard blocks pushing to a different org).

## Gotchas
- Secrets (`.env`, `credentials/`) are gitignored — never commit them; set via env on each host.
- The leaked Gemini key (`AIza…`) is dead; use a fresh AI-Studio key (`AQ.…` format worked).
- YouTube key in old `.env` was also leaked — rotate before using Phase 1/5 (upload).
