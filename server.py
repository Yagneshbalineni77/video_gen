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

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth

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

app = FastAPI(title="Dextora Creator")
auth.init_db()

# ── authentication ───────────────────────────────────────────────────────────
_bearer = HTTPBearer(auto_error=True)


def current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """Resolve the logged-in user from the Bearer JWT, or 401."""
    payload = auth.decode_token(creds.credentials)
    if not payload:
        raise HTTPException(401, "invalid or expired token")
    user = auth.get_user_by_id(payload.get("sub"))
    if not user:
        raise HTTPException(401, "user no longer exists")
    return user


class RegisterReq(BaseModel):
    email: str
    password: str
    invite_code: str


class LoginReq(BaseModel):
    email: str
    password: str


@app.post("/api/auth/register")
def register(req: RegisterReq) -> JSONResponse:
    if req.invite_code.strip() != auth.INVITE_CODE:
        raise HTTPException(403, "invalid invite code")
    try:
        user = auth.create_user(req.email, req.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return JSONResponse({"token": auth.make_token(user), "user": {"email": user["email"], "role": user["role"]}})


@app.post("/api/auth/login")
def login(req: LoginReq) -> JSONResponse:
    user = auth.authenticate(req.email, req.password)
    if not user:
        raise HTTPException(401, "wrong email or password")
    return JSONResponse({"token": auth.make_token(user), "user": {"email": user["email"], "role": user["role"]}})


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)) -> JSONResponse:
    return JSONResponse({"email": user["email"], "role": user["role"]})

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
_PERSIST_KEYS = ("id", "status", "prompt", "style", "lang", "created", "run_dir", "video", "video_en", "video_hi", "bilingual", "error", "user")


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
    bilingual: bool = False


def _slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40] or "video"


def _execute(job_id: str, prompt: str, style: str, lang: str, bilingual: bool = False) -> None:
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
        if bilingual:
            cmd.append("--bilingual")

    try:
        with log_path.open("a", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                    stdout=fh, stderr=subprocess.STDOUT)
        with _lock:
            job["pid"] = proc.pid
        proc.wait()
        with _lock:
            if bilingual:
                final_en = run_dir / "renders" / "final_render_en.mp4"
                final_hi = run_dir / "renders" / "final_render_hi.mp4"
                if proc.returncode == 0 and final_en.exists():
                    job["status"] = "done"
                    job["video"] = str(final_en)  # backward compat
                    job["video_en"] = str(final_en)
                    if final_hi.exists():
                        job["video_hi"] = str(final_hi)
                else:
                    job["status"] = "error"
                    job["error"] = f"pipeline exited {proc.returncode}"
            else:
                final = run_dir / "renders" / "final_render.mp4"
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
                _execute(job_id, j["prompt"], j["style"], j.get("lang", "indian_english"),
                         bilingual=j.get("bilingual", False))
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
    keyframes_done = len(re.findall(r"\[visual/dextora\] scene \d+ ->", text))
    veo_done = len(re.findall(r"\[dextora\] scene \d+ clip 0 done", text))
    anim_done = len(re.findall(r"\[animator\] scene \d+ ->", text))
    scenes_done = max(keyframes_done, veo_done) + anim_done
    fast_mode = "FAST MODE (SKIP_VEO)" in text

    if "Phase 2 – generating script" in text:
        phase, percent = "Writing script", 8
    if re.search(r"Phase 2 done", text):
        phase, percent = "Script ready", 12
    if "Phase 3 – generating assets" in text:
        phase = "Generating visuals"
        if not scenes_total:
            percent = 15
        elif fast_mode:
            # keyframes ARE the whole visual stage → advance 15→80 as they complete
            done = min(keyframes_done + anim_done, scenes_total)
            percent = 15 + int(65 * done / scenes_total)
        else:
            # full Veo: Pass-1 keyframes → 15..45, Pass-2 clips → 45..80 (so it never
            # sits frozen at 15 through the keyframe pass)
            kf = min(keyframes_done, scenes_total)
            vd = min(veo_done + anim_done, scenes_total)
            percent = 15 + int(30 * kf / scenes_total) + int(35 * vd / scenes_total)
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


def _can_see(job: dict, user: dict) -> bool:
    return user["role"] == "admin" or job.get("user") == user["email"]


@app.get("/api/jobs")
def jobs_list(limit: int = 30, user: dict = Depends(current_user)) -> JSONResponse:
    """The caller's own recent jobs (admins see everyone's) so the UI repopulates."""
    with _lock:
        recent = [_jobs[jid] for jid in reversed(_order)
                  if jid in _jobs and _can_see(_jobs[jid], user)][:limit]
        out = [{"id": j["id"], "prompt": j["prompt"], "style": j["style"],
                "lang": j.get("lang"), "status": j["status"],
                "has_video": bool(j.get("video"))} for j in recent]
    return JSONResponse(out)


@app.post("/api/generate")
def generate(req: GenerateReq, user: dict = Depends(current_user)) -> JSONResponse:
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
            "bilingual": req.bilingual,
            "created": datetime.now(timezone.utc).isoformat(),
            "run_dir": "", "video": None, "video_en": None, "video_hi": None, "error": None, "pid": None,
            "user": user["email"],
        }
        _order.append(job_id)
    _save_jobs()
    _pending.put(job_id)   # picked up by the bounded worker pool
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, user: dict = Depends(current_user)) -> JSONResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if not _can_see(job, user):
        raise HTTPException(403, "not your job")
    out = {k: job[k] for k in ("id", "status", "prompt", "style", "error")}
    out["has_video"] = bool(job.get("video"))
    out["bilingual"] = job.get("bilingual", False)
    out["has_hi"] = bool(job.get("video_hi"))
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


def _user_from_query_token(token: str) -> dict:
    """Auth for media URLs: <video src> / download links can't send an Authorization
    header, so the token rides as a ?token= query param instead."""
    payload = auth.decode_token(token or "")
    u = auth.get_user_by_id(payload.get("sub")) if payload else None
    if not u:
        raise HTTPException(401, "invalid or expired token")
    return u


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str, token: str = "", lang: str = "") -> FileResponse:
    user = _user_from_query_token(token)
    job = _jobs.get(job_id)
    if not job or not job.get("video"):
        raise HTTPException(404, "video not ready")
    if not _can_see(job, user):
        raise HTTPException(403, "not your job")
    if lang == "hi" and job.get("video_hi"):
        video_path = job["video_hi"]
    elif lang == "en" and job.get("video_en"):
        video_path = job["video_en"]
    else:
        video_path = job["video"]
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str, token: str = "") -> FileResponse:
    user = _user_from_query_token(token)
    job = _jobs.get(job_id)
    if not job or not job.get("video"):
        raise HTTPException(404, "video not ready")
    if not _can_see(job, user):
        raise HTTPException(403, "not your job")
    name = f"{_slug(job['prompt'])}_{job['style']}.mp4"
    return FileResponse(job["video"], media_type="video/mp4", filename=name)


# ── curriculum bridge: the education portal fetches finished videos here ──────
import portal_bridge  # noqa: E402


def _key_qs() -> str:
    """`?key=...` suffix appended to media URLs when PORTAL_API_KEY is set, so the
    portal can embed them directly in <video>/<img> tags (browsers can't send the
    X-API-Key header on those). Empty when no key is configured."""
    k = os.getenv("PORTAL_API_KEY", "")
    return f"?key={k}" if k else ""


def _portal_video_url(request: Request, script_id: int) -> str:
    return f"{str(request.base_url).rstrip('/')}/api/v1/videos/{script_id}/file{_key_qs()}"


def _portal_audio_url(request: Request, script_id: int) -> str:
    return f"{str(request.base_url).rstrip('/')}/api/v1/videos/{script_id}/audio{_key_qs()}"


def _portal_thumb_url(request: Request, script_id: int) -> str:
    return f"{str(request.base_url).rstrip('/')}/api/v1/videos/{script_id}/thumbnail{_key_qs()}"


def _portal_key_ok(request: Request) -> bool:
    """Optional auth: if PORTAL_API_KEY is set, require it — either in the X-API-Key
    header (for API calls) OR a ?key= query param (so embedded media URLs work)."""
    want = os.getenv("PORTAL_API_KEY", "")
    if not want:
        return True
    return request.headers.get("X-API-Key") == want or request.query_params.get("key") == want


@app.get("/api/v1/videos")
def portal_videos(request: Request, ids: str = "", class_id: int | None = None,
                  subject_id: int | None = None, chapter_id: int | None = None) -> JSONResponse:
    """Portal-facing: finished video URLs keyed by the portal's own script id.
    Query by ?ids=881,882  OR  ?class_id=11&subject_id=49&chapter_id=509."""
    if not _portal_key_ok(request):
        raise HTTPException(401, "invalid or missing X-API-Key")
    id_list = [int(i) for i in ids.split(",") if i.strip().isdigit()] or None
    rows = portal_bridge.get_videos(ids=id_list, class_id=class_id,
                                    subject_id=subject_id, chapter_id=chapter_id)
    videos = []
    for r in rows:
        ready = r["status"] == "done" and r.get("video_path")
        videos.append({
            "id": r["script_id"], "atom_id": r["atom_id"], "chapter_id": r["chapter_id"],
            "subject_id": r["subject_id"], "language": r["language"], "title": r["title"],
            "status": r["status"], "duration": r["duration"],
            "video_url": _portal_video_url(request, r["script_id"]) if ready else None,
            "audio_url": _portal_audio_url(request, r["script_id"]) if ready else None,
            "thumbnail_url": _portal_thumb_url(request, r["script_id"]) if ready else None,
        })
    return JSONResponse({"count": len(videos), "videos": videos})


@app.get("/api/v1/videos/{script_id}/file")
def portal_video_file(script_id: int, request: Request) -> FileResponse:
    if not _portal_key_ok(request):
        raise HTTPException(401, "invalid or missing X-API-Key")
    rows = portal_bridge.get_videos(ids=[script_id])
    if not rows or rows[0]["status"] != "done" or not rows[0].get("video_path"):
        raise HTTPException(404, "video not ready")
    path = rows[0]["video_path"]
    if not os.path.exists(path):
        raise HTTPException(404, "video file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"script_{script_id}.mp4")


@app.get("/api/v1/videos/{script_id}/audio")
def portal_video_audio(script_id: int, request: Request) -> FileResponse:
    """Narration audio (MP3) for one script — extracted from the final video on first
    request and cached. For the portal's 'Listen' feature."""
    if not _portal_key_ok(request):
        raise HTTPException(401, "invalid or missing X-API-Key")
    rows = portal_bridge.get_videos(ids=[script_id])
    if not rows or rows[0]["status"] != "done" or not rows[0].get("video_path"):
        raise HTTPException(404, "audio not ready")
    video_path = Path(rows[0]["video_path"])
    if not video_path.exists():
        raise HTTPException(404, "video file missing")
    audio_path = video_path.parent / f"{video_path.stem}.mp3"
    if not audio_path.exists():
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
                 "-vn", "-acodec", "libmp3lame", "-q:a", "4", str(audio_path)],
                check=True, timeout=180,
            )
        except Exception as exc:
            raise HTTPException(500, f"audio extraction failed: {exc}")
    return FileResponse(str(audio_path), media_type="audio/mpeg", filename=f"script_{script_id}.mp3")


@app.get("/api/v1/videos/{script_id}/thumbnail")
def portal_video_thumbnail(script_id: int, request: Request) -> FileResponse:
    """Thumbnail (JPG) for one script — the clean scene-1 keyframe if available, else a
    frame grabbed a few seconds into the video. Generated on first request and cached."""
    if not _portal_key_ok(request):
        raise HTTPException(401, "invalid or missing X-API-Key")
    rows = portal_bridge.get_videos(ids=[script_id])
    if not rows or rows[0]["status"] != "done" or not rows[0].get("video_path"):
        raise HTTPException(404, "thumbnail not ready")
    video_path = Path(rows[0]["video_path"])
    if not video_path.exists():
        raise HTTPException(404, "video file missing")
    thumb_path = video_path.parent / f"{video_path.stem}_thumb.jpg"
    if not thumb_path.exists():
        keyframe = video_path.parent.parent / "video" / "scene_1_keyframe.png"
        try:
            if keyframe.exists():
                src = ["-i", str(keyframe)]
            else:
                src = ["-ss", "3", "-i", str(video_path)]  # grab a frame ~3s in
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", *src, "-vframes", "1",
                 "-vf", "scale=1280:-2", "-q:v", "3", str(thumb_path)],
                check=True, timeout=60,
            )
        except Exception as exc:
            raise HTTPException(500, f"thumbnail generation failed: {exc}")
    return FileResponse(str(thumb_path), media_type="image/jpeg", filename=f"script_{script_id}.jpg")


@app.get("/api/v1/videos/{script_id}/log")
def portal_video_log(script_id: int, request: Request, lines: int = 80) -> JSONResponse:
    """Live backend log + progress for one script's render — per-run visibility
    without host shell access. Returns phase, percent, scene counts, and the log tail."""
    if not _portal_key_ok(request):
        raise HTTPException(401, "invalid or missing X-API-Key")
    run_dir = portal_bridge.RUNS_DIR / f"script_{script_id}"
    log_path = run_dir / "run.log"
    if not log_path.exists():
        raise HTTPException(404, "no log for this script yet")
    prog = _progress_from_log(str(run_dir))
    n = max(1, min(lines, 1000))
    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    return JSONResponse({
        "script_id": script_id,
        "phase": prog.get("phase"),
        "percent": prog.get("percent"),
        "scenes_done": prog.get("scenes_done"),
        "scenes_total": prog.get("scenes_total"),
        "log_tail": tail,
    })


# ── push-generate: the portal sends us a script, we render + store by its id ──
from concurrent.futures import ThreadPoolExecutor as _ThreadPool  # noqa: E402
_portal_gen_pool = _ThreadPool(max_workers=int(os.getenv("PORTAL_GEN_WORKERS", "2")))


class GenerateScriptReq(BaseModel):
    id: int                              # the portal's script id — results keyed by this
    audio_script: str                    # narration text (required)
    title: str = ""
    visual_directions: str = ""
    language: str = "English"            # English | Hindi
    subject: str = ""                    # subject name → visual style
    duration_seconds: int | None = None
    class_id: int | None = None
    subject_id: int | None = None
    chapter_id: int | None = None
    script_type: str = "VIDEO"


@app.post("/api/v1/generate")
def portal_generate(req: GenerateScriptReq, request: Request) -> JSONResponse:
    """Push-generate: the portal sends a full script (narration + visual directions),
    we render the video and store it keyed by the portal's own `id`. Then poll
    GET /api/v1/videos?ids=<id> for status + video/audio/thumbnail URLs."""
    if not _portal_key_ok(request):
        raise HTTPException(401, "invalid or missing X-API-Key")
    if not (req.audio_script or "").strip():
        raise HTTPException(400, "audio_script is required")
    item = {
        "id": req.id, "atom_id": None, "chapter_id": req.chapter_id,
        "script_type": req.script_type, "language": req.language,
        "title": req.title or f"script {req.id}",
        "audio_script": req.audio_script, "visual_directions": req.visual_directions,
        "duration_seconds": req.duration_seconds, "metadata": {},
    }
    portal_bridge.upsert(req.id, atom_id=None, chapter_id=req.chapter_id,
                         subject_id=req.subject_id, class_id=req.class_id,
                         script_type=req.script_type, language=req.language,
                         title=item["title"], status="queued", video_path=None,
                         duration=req.duration_seconds, error=None)
    _portal_gen_pool.submit(portal_bridge.generate_one, item, req.subject, req.class_id, req.subject_id)
    return JSONResponse({"id": req.id, "status": "queued"})


class BridgeRunReq(BaseModel):
    class_id: int
    subject_id: int
    chapter_id: int
    subject: str = ""
    languages: list[str] = ["English"]
    types: list[str] = ["VIDEO"]
    limit: int | None = None
    workers: int = 2


@app.post("/api/v1/bridge/run")
def bridge_run(req: BridgeRunReq, user: dict = Depends(current_user)) -> JSONResponse:
    """Admin-triggered mass generation for one chapter (runs in the background)."""
    if user["role"] != "admin":
        raise HTTPException(403, "admin only")
    threading.Thread(
        target=portal_bridge.run_chapter,
        args=(req.class_id, req.subject_id, req.chapter_id, req.subject,
              tuple(req.languages), tuple(req.types), req.limit, req.workers),
        daemon=True,
    ).start()
    return JSONResponse({"started": True, **req.model_dump()})


# ── frontend (mounted last so /api/* wins) ───────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    idx = FRONTEND / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>frontend/index.html missing</h1>", status_code=500)
    return HTMLResponse(idx.read_text(encoding="utf-8"))


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
