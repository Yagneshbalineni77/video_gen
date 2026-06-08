"""
Parallel video producer — run many (prompt, style) jobs at once, each fully
isolated, resumable, and crash-safe. This is the scale layer for large runs
(e.g. thousands of videos); the unit of work is `main.py --prompt --style`.

Each job runs as its OWN process with its OWN --output-dir, so jobs never share
files and one crash never touches the others.

Jobs file
---------
JSON  : [{"prompt": "...", "style": "educational"}, ...]
JSONL : one {"prompt","style"} object per line
TXT   : one job per line as  "prompt | style"  (style optional → 'educational')

Usage
-----
  python produce.py --jobs jobs.json --workers 10
  python produce.py --jobs jobs.txt  --workers 10 --veo-concurrency 1
  python produce.py --jobs jobs.json --plan            # show plan, generate nothing
  python produce.py --jobs jobs.json --limit 5

Rate-limit math
---------------
Simultaneous Veo jobs ≈ workers × veo-concurrency. Keep that under your API
limit. For 10 parallel videos on an AI-Studio key, use --veo-concurrency 1
(≈10 concurrent Veo). With Vertex provisioned quota you can raise both.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "output" / "runs"
PYTHON = sys.executable
VALID_STYLES = {"educational", "ghibli", "realistic", "documentary", "dark_cinematic", "thriller"}


def _slug(prompt: str, idx: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:50]
    return f"{idx:04d}_{s or 'video'}"


def load_jobs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Jobs file not found: {path}")
    text = path.read_text(encoding="utf-8")

    def _norm(job: dict) -> dict:
        style = (job.get("style") or "educational").strip()
        if style not in VALID_STYLES:
            raise ValueError(f"Unknown style '{style}' (valid: {sorted(VALID_STYLES)})")
        return {"prompt": job["prompt"].strip(), "style": style}

    if path.suffix.lower() == ".json":
        return [_norm(j) for j in json.loads(text)]
    if path.suffix.lower() == ".jsonl":
        return [_norm(json.loads(ln)) for ln in text.splitlines() if ln.strip()]
    # .txt  →  "prompt | style"
    jobs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompt, _, style = line.partition("|")
        jobs.append(_norm({"prompt": prompt, "style": style or "educational"}))
    return jobs


def produce_one(job: dict, idx: int, veo_concurrency: int) -> dict:
    slug = _slug(job["prompt"], idx)
    run_dir = RUNS_DIR / slug
    final = run_dir / "renders" / "final_render.mp4"
    started = time.time()

    if final.exists():  # resume
        return {"slug": slug, "status": "skipped", "prompt": job["prompt"],
                "style": job["style"], "seconds": 0, "output": str(final)}

    run_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "VEO_CONCURRENCY": str(veo_concurrency),
    }
    log_path = run_dir / "run.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_fh:
            proc = subprocess.run(
                [PYTHON, str(ROOT / "main.py"),
                 "--prompt", job["prompt"],
                 "--style", job["style"],
                 "--output-dir", str(run_dir.relative_to(ROOT))],
                cwd=str(ROOT), env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0 or not final.exists():
            return {"slug": slug, "status": "failed", "prompt": job["prompt"],
                    "style": job["style"], "seconds": round(time.time() - started, 1),
                    "error": f"exit {proc.returncode}", "log": str(log_path)}
        return {"slug": slug, "status": "ok", "prompt": job["prompt"], "style": job["style"],
                "seconds": round(time.time() - started, 1), "output": str(final)}
    except Exception as exc:
        return {"slug": slug, "status": "failed", "prompt": job["prompt"],
                "style": job["style"], "seconds": round(time.time() - started, 1),
                "error": str(exc), "log": str(log_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel video producer")
    ap.add_argument("--jobs", required=True, help=".json / .jsonl / .txt of (prompt, style)")
    ap.add_argument("--workers", type=int, default=3, help="videos to produce in parallel")
    ap.add_argument("--veo-concurrency", type=int, default=2,
                    help="Veo workers PER video (workers × this ≈ total Veo jobs)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N jobs")
    ap.add_argument("--plan", action="store_true", help="print plan and exit (no generation)")
    args = ap.parse_args()

    jobs = load_jobs(Path(args.jobs))
    if args.limit:
        jobs = jobs[: args.limit]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    total_veo = args.workers * args.veo_concurrency
    print(f"Producer: {len(jobs)} job(s) | {args.workers} parallel × "
          f"{args.veo_concurrency} Veo each ≈ {total_veo} concurrent Veo jobs\n")
    if total_veo > 16:
        print(f"  ⚠ {total_veo} concurrent Veo jobs may hit AI-Studio rate limits; "
              f"lower --workers or --veo-concurrency, or use Vertex quota.\n")

    pending = []
    for i, j in enumerate(jobs, 1):
        slug = _slug(j["prompt"], i)
        done = (RUNS_DIR / slug / "renders" / "final_render.mp4").exists()
        print(f"  [{i:04d}] {'✓ done (skip)' if done else '· pending'}  "
              f"[{j['style']}] {j['prompt'][:60]}")
        if not done:
            pending.append((j, i))

    print(f"\n{len(pending)} to produce, {len(jobs) - len(pending)} already done.")
    if args.plan:
        print("\n--plan set; exiting without generating.")
        return
    if not pending:
        print("Nothing to do.")
        return

    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(produce_one, j, i, args.veo_concurrency): i for j, i in pending}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            icon = {"ok": "✓", "skipped": "•", "failed": "✗"}.get(r["status"], "?")
            print(f"  {icon} [{r['status']}] [{r['style']}] {r['prompt'][:45]} "
                  f"({r['seconds']}s) {r.get('error', r.get('output', ''))}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(jobs),
        "produced": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "results": sorted(results, key=lambda r: r["slug"]),
    }
    (RUNS_DIR / "batch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone in {report['elapsed_seconds']}s — {report['produced']} ok, "
          f"{report['failed']} failed. Report: {RUNS_DIR / 'batch_report.json'}")
    if report["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
