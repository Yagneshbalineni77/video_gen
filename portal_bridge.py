"""
Curriculum bridge — Dextora education portal ↔ video pipeline.

Flow:
  1. fetch_scripts(class, subject, chapter)  → pull authored scripts from the portal
  2. generate_one(item)                      → main.py --from-script (Phase 3/4, no Phase 2)
  3. store {script_id → video_path, status}  → SQLite (output/portal_videos.db)
The portal then fetches finished URLs from our API (see /api/v1/videos in server.py).

The store is the single source of truth keyed by the portal's own script `id`.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from config import settings

log = logging.getLogger(__name__)

PORTAL_API = os.getenv("PORTAL_API_URL", "https://dextora.org/api/v1/curriculum/video-reel-scripts")
ROOT = Path(__file__).resolve().parent
DB_PATH = settings.OUTPUT_DIR / "portal_videos.db"
RUNS_DIR = settings.OUTPUT_DIR / "portal_runs"

# portal language → our narration profile key
LANG_MAP = {"english": "indian_english", "hindi": "hindi"}
# subject → visual style key (main.py --style)
SUBJECT_STYLE = {
    "chemistry": "educational", "physics": "educational", "biology": "educational",
    "mathematics": "educational", "computer science": "educational", "informatics practices": "educational",
    "history": "documentary", "geography": "documentary", "political science": "documentary",
    "sociology": "documentary", "psychology": "documentary",
    "english": "ghibli", "hindi": "ghibli",
    "economics": "educational", "accountancy": "educational", "business studies": "educational",
}

_db_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS portal_videos(
                script_id INTEGER PRIMARY KEY, atom_id INTEGER, chapter_id INTEGER,
                subject_id INTEGER, class_id INTEGER, script_type TEXT, language TEXT,
                title TEXT, status TEXT, video_path TEXT, duration INTEGER,
                error TEXT, updated TEXT)"""
        )


_FIELDS = ("atom_id", "chapter_id", "subject_id", "class_id", "script_type",
           "language", "title", "status", "video_path", "duration", "error")


def upsert(script_id: int, **fields) -> None:
    init_db()
    fields = {k: v for k, v in fields.items() if k in _FIELDS}
    fields["updated"] = _now()
    with _db_lock, _conn() as c:
        exists = c.execute("SELECT 1 FROM portal_videos WHERE script_id=?", (script_id,)).fetchone()
        if exists:
            sets = ", ".join(f"{k}=?" for k in fields)
            c.execute(f"UPDATE portal_videos SET {sets} WHERE script_id=?", (*fields.values(), script_id))
        else:
            keys = ["script_id", *fields.keys()]
            c.execute(
                f"INSERT INTO portal_videos({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})",
                (script_id, *fields.values()),
            )


def get_videos(ids=None, class_id=None, subject_id=None, chapter_id=None) -> list[dict]:
    init_db()
    q, args = "SELECT * FROM portal_videos WHERE 1=1", []
    if ids:
        q += f" AND script_id IN ({', '.join('?' * len(ids))})"
        args += [int(i) for i in ids]
    for col, val in (("class_id", class_id), ("subject_id", subject_id), ("chapter_id", chapter_id)):
        if val is not None:
            q += f" AND {col}=?"
            args.append(int(val))
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


# ── portal API ───────────────────────────────────────────────────────────────

def fetch_scripts(class_id: int, subject_id, chapter_id) -> tuple[list, list]:
    """Pull authored scripts for one chapter → (videos, reels)."""
    payload = {"class": class_id, "subject_id": subject_id, "chapter_id": str(chapter_id)}
    req = urllib.request.Request(
        PORTAL_API, data=json.dumps(payload).encode(),
        headers={"Accept": "application/json", "Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    data = d.get("data", {})
    return data.get("videos", []), data.get("reels", [])


# ── generation ───────────────────────────────────────────────────────────────

def generate_one(item: dict, subject_name: str = "", class_id=None, subject_id=None) -> bool:
    """Generate one video from a portal script item, updating the store. Returns success."""
    sid = item["id"]
    lang = LANG_MAP.get((item.get("language") or "").strip().lower(), "indian_english")
    style = SUBJECT_STYLE.get((subject_name or "").strip().lower(), "educational")
    run_dir = RUNS_DIR / f"script_{sid}"
    run_dir.mkdir(parents=True, exist_ok=True)

    upsert(sid, atom_id=item.get("atom_id"), chapter_id=item.get("chapter_id"),
           subject_id=subject_id, class_id=class_id, script_type=item.get("script_type"),
           language=item.get("language"), title=item.get("title"), status="generating",
           video_path=None, duration=item.get("duration_seconds"), error=None)

    item_path = run_dir / "script.json"
    item_path.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "main.py"), "--from-script", str(item_path),
           "--subject", subject_name, "--style", style, "--lang", lang,
           "--output-dir", str(run_dir.relative_to(ROOT))]
    log_path = run_dir / "run.log"
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONUTF8": "1"})
        final = run_dir / "renders" / "final_render.mp4"
        if proc.returncode == 0 and final.exists():
            upsert(sid, status="done", video_path=str(final))
            log.info("[bridge] script %s done -> %s", sid, final)
            return True
        upsert(sid, status="error", error=f"pipeline exit {proc.returncode}")
        log.warning("[bridge] script %s failed (exit %s)", sid, proc.returncode)
        return False
    except Exception as exc:  # never let one script kill a batch
        upsert(sid, status="error", error=str(exc))
        log.warning("[bridge] script %s crashed: %s", sid, exc)
        return False


def run_chapter(class_id: int, subject_id, chapter_id, subject_name: str = "",
                languages=("English",), types=("VIDEO",), limit=None, workers=2) -> dict:
    """Fetch a chapter's scripts and generate the selected ones. Returns a summary."""
    videos, reels = fetch_scripts(class_id, subject_id, chapter_id)
    pool: list[dict] = []
    if "VIDEO" in {t.upper() for t in types}:
        pool += videos
    if "REEL" in {t.upper() for t in types}:
        pool += reels
    langset = {l.lower() for l in languages}
    pool = [it for it in pool if (it.get("language") or "").lower() in langset]
    if limit:
        pool = pool[:limit]
    log.info("[bridge] chapter %s: generating %d scripts (langs=%s types=%s workers=%d)",
             chapter_id, len(pool), list(languages), list(types), workers)

    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(generate_one, it, subject_name, class_id, subject_id) for it in pool]
        for f in futs:
            if f.result():
                ok += 1
    summary = {"chapter_id": chapter_id, "total": len(pool), "done": ok, "failed": len(pool) - ok}
    log.info("[bridge] chapter %s complete: %s", chapter_id, summary)
    return summary


if __name__ == "__main__":
    # CLI: python portal_bridge.py <class> <subject_id> <chapter_id> [subject_name] [limit]
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Dextora curriculum bridge — generate videos for a chapter")
    ap.add_argument("class_id", type=int)
    ap.add_argument("subject_id", type=int)
    ap.add_argument("chapter_id", type=int)
    ap.add_argument("--subject", default="", help="subject name (style brief), e.g. Chemistry")
    ap.add_argument("--langs", default="English", help="comma list: English,Hindi")
    ap.add_argument("--types", default="VIDEO", help="comma list: VIDEO,REEL")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    print(run_chapter(a.class_id, a.subject_id, a.chapter_id, a.subject,
                      tuple(a.langs.split(",")), tuple(a.types.split(",")), a.limit, a.workers))
