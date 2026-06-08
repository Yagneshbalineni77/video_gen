"""
Product API server — backend for the web UI.

Exposes the single-video pipeline (prompt + style → downloadable mp4) over HTTP
with live progress, and serves the React frontend.

Run:
  uvicorn server:app --reload --port 8000
  # then open http://localhost:8000

Endpoints:
  GET  /                      → React frontend
  GET  /api/styles            → available visual styles
  POST /api/generate          → {prompt, style} → {job_id}
  GET  /api/jobs/{id}         → live status/progress
  GET  /api/jobs/{id}/video   → the finished mp4 (inline/stream)
  GET  /api/jobs/{id}/download→ the finished mp4 (attachment)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
RUNS = ROOT / "output" / "runs"
FRONTEND = ROOT / "frontend"
PYTHON = sys.executable

# Friendly styles surfaced to the UI (key must match main.py STYLE_MAP)
STYLES = [
    {"key": "educational",    "label": "Educational",    "desc": "Clean 3D textbook diagrams", "emoji": "🔬"},
    {"key": "ghibli",         "label": "Ghibli",         "desc": "Hand-painted Studio-Ghibli anime", "emoji": "🌸"},
    {"key": "realistic",      "label": "Realistic",      "desc": "Photoreal, shallow depth of field", "emoji": "🎬"},
    {"key": "documentary",    "label": "Documentary",    "desc": "Warm 35mm film look", "emoji": "📽️"},
    {"key": "dark_cinematic", "label": "Dark Cinematic", "desc": "Moody teal-orange film", "emoji": "🌑"},
    {"key": "thriller",       "label": "Thriller",       "desc": "Cold, clinical, tense", "emoji": "🕵️"},
]
VALID_STYLE_KEYS = {s["key"] for s in STYLES}

# Narration language/accent options surfaced to the UI (keys match main.py --lang)
LANGUAGES = [
    {"key": "indian_english",   "label": "Indian English",   "emoji": "🇮🇳"},
    {"key": "hinglish",         "label": "Hinglish",         "emoji": "🗣️"},
    {"key": "hindi",            "label": "Hindi",            "emoji": "🪔"},
    {"key": "british_english",  "label": "British English",  "emoji": "🇬🇧"},
    {"key": "american_english", "label": "American English", "emoji": "🇺🇸"},
]
VALID_LANG_KEYS = {l["key"] for l in LANGUAGES}

app = FastAPI(title="Faceless Video Studio")

# ── concurrency tuning ───────────────────────────────────────────────────────
# MAX_CONCURRENT caps how many pipelines run at once. Default 10 — the validated
# clean ceiling on an AI-Studio key (the box has ample headroom; the API quota is
# the limit). The queue holds any overflow.
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))
SMOKE_TEST = os.getenv("SMOKE_TEST", "").lower() in ("1", "true", "yes")

# Total concurrent Veo/Imagen calls across ALL jobs ≈ MAX_CONCURRENT × per-job
# VEO_CONCURRENCY. We keep that total near a known-clean target (~20 was 0-rate-limit
# in testing) by auto-deriving per-job concurrency from the job cap. Smaller job-cap
# → more workers each; larger cap → fewer, so the API is never flooded.
TARGET_TOTAL_VEO = int(os.getenv("TARGET_TOTAL_VEO", "20"))
PER_JOB_VEO = max(1, min(4, TARGET_TOTAL_VEO // max(1, MAX_CONCURRENT)))

# ── in-memory job registry + bounded worker pool ─────────────────────────────
import queue as _queuemod  # noqa: E402

_jobs: dict[str, dict] = {}
_order: list[str] = []                 # submission order, for queue-position display
_lock = threading.Lock()
_pending: "_queuemod.Queue[str]" = _queuemod.Queue()

# Persisted registry so a server restart doesn't lose job history / in-flight state.
_STATE_FILE = RUNS / "_jobs.json"
_PERSIST_KEYS = ("id", "status", "prompt", "style", "lang", "created", "run_dir", "video", "error")


def _save_jobs() -> None:
    """Persist the registry to disk (best-effort). Called on every status change."""
    try:
        RUNS.mkdir(parents=True, exist_ok=True)
        with _lock:
            data = {"order": list(_order),
                    "jobs": {jid: {k: j.get(k) for k in _PERSIST_KEYS} for jid, j in _jobs.items()}}
        tmp = _STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(_STATE_FILE)
    except Exception:
        pass  # persistence is best-effort; never break a request over it


def _load_jobs() -> None:
    """Reload registry on startup and reconcile interrupted jobs.
    A job that was running/queued when the server stopped is marked 'done' if its
    video exists, otherwise 'interrupted' (its subprocess is gone)."""
    if not _STATE_FILE.exists():
        return
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    for jid, j in data.get("jobs", {}).items():
        status = j.get("status")
        if status in ("running", "queued"):
            final = Path(j["run_dir"]) / "renders" / "final_render.mp4" if j.get("run_dir") else None
            if final and final.exists():
                j["status"], j["video"] = "done", str(final)
            else:
                j["status"], j["error"] = "interrupted", "server restarted while this job was in progress"
        _jobs[jid] = j
    _order.extend(data.get("order", []))


class GenerateReq(BaseModel):
    prompt: str
    style: str = "educational"
    lang: str = "indian_english"


def _slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40] or "video"


def _execute(job_id: str, prompt: str, style: str, lang: str) -> None:
    """Actually run one pipeline (or a cheap fake in SMOKE_TEST mode)."""
    job = _jobs[job_id]
    run_dir = RUNS / f"web_{job_id[:8]}_{_slug(prompt)}"
    (run_dir / "renders").mkdir(parents=True, exist_ok=True)
    with _lock:
        job["run_dir"] = str(run_dir)
        job["status"] = "running"
    _save_jobs()
    log_path = run_dir / "run.log"
    # Auto-tuned per-job Veo/Imagen concurrency keeps the total API pressure
    # (across all concurrent jobs) near the known-clean target.
    env = {**os.environ, "PYTHONUTF8": "1", "VEO_CONCURRENCY": str(PER_JOB_VEO)}

    if SMOKE_TEST:
        # Fake a job: sleep to simulate work, emit a tiny valid mp4. Lets us load-test
        # the queue/cap/status machinery WITHOUT spending Veo quota.
        final = run_dir / "renders" / "final_render.mp4"
        cmd = ["bash", "-c",
               f"echo 'SMOKE: Phase 3 – generating assets for 4 scenes' > '{log_path}'; "
               f"sleep 6; "
               f"ffmpeg -y -loglevel error -f lavfi -i testsrc=duration=2:size=640x360:rate=12 "
               f"-pix_fmt yuv420p '{final}'; echo 'VIDEO_READY' >> '{log_path}'"]
    else:
        cmd = [PYTHON, str(ROOT / "main.py"),
               "--prompt", prompt, "--style", style, "--lang", lang,
               "--output-dir", str(run_dir.relative_to(ROOT))]

    try:
        with log_path.open("a", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                    stdout=fh, stderr=subprocess.STDOUT)
        with _lock:
            job["pid"] = proc.pid
        proc.wait()
        final = run_dir / "renders" / "final_render.mp4"
        with _lock:
            if proc.returncode == 0 and final.exists():
                job["status"] = "done"
                job["video"] = str(final)
            else:
                job["status"] = "error"
                job["error"] = f"pipeline exited {proc.returncode}"
    except Exception as exc:  # pragma: no cover
        with _lock:
            job["status"] = "error"
            job["error"] = str(exc)
    _save_jobs()


def _worker() -> None:
    """One pool worker: pull queued jobs and run them one at a time."""
    while True:
        job_id = _pending.get()
        try:
            j = _jobs.get(job_id)
            if j and j["status"] == "queued":
                _execute(job_id, j["prompt"], j["style"], j.get("lang", "indian_english"))
        finally:
            _pending.task_done()


# Reload any persisted jobs (reconciling interrupted ones) before serving.
_load_jobs()

# Start the fixed worker pool — this is the concurrency cap.
for _ in range(max(1, MAX_CONCURRENT)):
    threading.Thread(target=_worker, daemon=True).start()


def _queue_position(job_id: str) -> int:
    """1-based position among jobs still waiting (0 if not queued)."""
    with _lock:
        waiting = [jid for jid in _order if _jobs[jid]["status"] == "queued"]
    return waiting.index(job_id) + 1 if job_id in waiting else 0


def _progress_from_log(run_dir: str) -> dict:
    """Derive phase + scene progress + percent from the subprocess log."""
    log = Path(run_dir) / "run.log"
    if not log.exists():
        return {"phase": "starting", "percent": 2, "scenes_done": 0, "scenes_total": 0, "tail": ""}
    text = log.read_text(encoding="utf-8", errors="replace")

    phase, percent = "starting", 3
    scenes_total = scenes_done = 0

    m = re.search(r"generating assets for (\d+) scenes", text)
    if m:
        scenes_total = int(m.group(1))
    scenes_done = len(re.findall(r"\[veo3\] scene \d+ clip 0 done", text)) \
        + len(re.findall(r"\[animator\] scene \d+ ->", text))

    if "Phase 2 – generating script" in text:
        phase, percent = "Writing script", 8
    if re.search(r"Phase 2 done", text):
        phase, percent = "Script ready", 12
    if "Phase 3 – generating assets" in text:
        phase = "Generating visuals"
        if scenes_total:
            percent = 15 + int(65 * scenes_done / max(scenes_total, 1))
        else:
            percent = 15
    if "Phase 4 – assembling" in text:
        # Finer Phase 4 granularity so the bar keeps moving through the (CPU-heavy)
        # assembly instead of sitting at one number.
        prepared = len(re.findall(r"\[edit\] scene \d+ prepared", text))
        phase, percent = "Assembling clips", 80
        if scenes_total and prepared:
            percent = 80 + int(8 * min(prepared, scenes_total) / max(scenes_total, 1))  # 80→88
        if "xfade concat done" in text:
            phase, percent = "Stitching scenes", 90
        if "narration merged" in text:
            phase, percent = "Mixing audio", 92
        if "transcribing" in text or "built from script" in text:
            phase, percent = "Building subtitles", 94
        if "subtitles burned" in text:
            phase, percent = "Burning subtitles", 97
    if "Phase 4 done" in text or "VIDEO_READY" in text:
        phase, percent = "Finalizing", 99

    tail = "\n".join(
        ln for ln in text.splitlines()[-12:]
        if "AFC is enabled" not in ln
    )
    return {"phase": phase, "percent": percent, "scenes_done": scenes_done,
            "scenes_total": scenes_total, "tail": tail}


# ── API ──────────────────────────────────────────────────────────────────────

@app.get("/api/styles")
def styles() -> JSONResponse:
    return JSONResponse(STYLES)


@app.get("/api/languages")
def languages() -> JSONResponse:
    return JSONResponse(LANGUAGES)


@app.get("/api/stats")
def stats() -> JSONResponse:
    with _lock:
        statuses = [j["status"] for j in _jobs.values()]
    return JSONResponse({
        "max_concurrent": MAX_CONCURRENT,
        "per_job_veo": PER_JOB_VEO,
        "smoke_test": SMOKE_TEST,
        "running": statuses.count("running"),
        "queued": statuses.count("queued"),
        "done": statuses.count("done"),
        "error": statuses.count("error"),
        "interrupted": statuses.count("interrupted"),
        "total": len(statuses),
    })


@app.get("/api/jobs")
def jobs_list(limit: int = 30) -> JSONResponse:
    """Recent jobs (newest first) so the UI can repopulate after reload/restart."""
    with _lock:
        recent = [_jobs[jid] for jid in reversed(_order) if jid in _jobs][:limit]
        out = [{"id": j["id"], "prompt": j["prompt"], "style": j["style"],
                "lang": j.get("lang"), "status": j["status"],
                "has_video": bool(j.get("video"))} for j in recent]
    return JSONResponse(out)


@app.post("/api/generate")
def generate(req: GenerateReq) -> JSONResponse:
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")
    if req.style not in VALID_STYLE_KEYS:
        raise HTTPException(400, f"unknown style '{req.style}'")
    if req.lang not in VALID_LANG_KEYS:
        raise HTTPException(400, f"unknown language '{req.lang}'")

    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "id": job_id, "status": "queued", "prompt": prompt, "style": req.style, "lang": req.lang,
            "created": datetime.now(timezone.utc).isoformat(),
            "run_dir": "", "video": None, "error": None, "pid": None,
        }
        _order.append(job_id)
    _save_jobs()
    _pending.put(job_id)   # picked up by the bounded worker pool
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    out = {k: job[k] for k in ("id", "status", "prompt", "style", "error")}
    out["has_video"] = bool(job.get("video"))
    if job["status"] == "queued":
        pos = _queue_position(job_id)
        out.update({"phase": f"Queued (#{pos} in line)", "percent": 0,
                    "scenes_done": 0, "scenes_total": 0,
                    "tail": f"Waiting for a free slot ({MAX_CONCURRENT} run at a time)…",
                    "queue_position": pos})
        return JSONResponse(out)
    out.update(_progress_from_log(job.get("run_dir", "")) if job["status"] == "running"
               else {"phase": job["status"], "percent": 100 if job["status"] == "done" else 0})
    return JSONResponse(out)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = _jobs.get(job_id)
    if not job or not job.get("video"):
        raise HTTPException(404, "video not ready")
    return FileResponse(job["video"], media_type="video/mp4")


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str) -> FileResponse:
    job = _jobs.get(job_id)
    if not job or not job.get("video"):
        raise HTTPException(404, "video not ready")
    name = f"{_slug(job['prompt'])}_{job['style']}.mp4"
    return FileResponse(job["video"], media_type="video/mp4", filename=name)


# ── frontend (mounted last so /api/* wins) ───────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    idx = FRONTEND / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>frontend/index.html missing</h1>", status_code=500)
    return HTMLResponse(idx.read_text(encoding="utf-8"))


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
