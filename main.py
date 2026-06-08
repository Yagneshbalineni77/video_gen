"""
Entry point – run with:
  python main.py                                   # full pipeline (trend-scraped)
  python main.py --dry-run                         # skip YouTube upload
  python main.py --phase 1                         # run only Phase 1 (trend scraper)

  # Product flow — prompt + style → one downloadable video (no scraping, no upload):
  python main.py --prompt "How mRNA vaccines work" --style educational
  python main.py --prompt "The myth of Icarus" --style ghibli --output-dir output/runs/icarus
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Friendly style names exposed to users → VISUAL_STYLE_PRESETS keys in config/settings.py
STYLE_MAP: dict[str, str] = {
    "educational":    "academic_science",
    "ghibli":         "ghibli_anime",
    "realistic":      "cinematic_portrait",
    "documentary":    "warm_documentary",
    "dark_cinematic": "dark_cinematic",
    "thriller":       "cold_thriller",
}


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress noisy library loggers
    for lib in ("httpx", "httpcore", "urllib3", "googleapiclient"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Faceless YouTube pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Skip YouTube upload")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run only a specific phase (for testing)",
    )
    parser.add_argument("--prompt", help="Topic/prompt for a single styled video (skips scraping & upload)")
    parser.add_argument(
        "--style",
        choices=sorted(STYLE_MAP.keys()),
        default="educational",
        help="Visual theme for --prompt mode",
    )
    parser.add_argument(
        "--lang",
        default="indian_english",
        help="Narration profile: indian_english, british_english, american_english, hindi, hinglish",
    )
    parser.add_argument("--output-dir", help="Per-video output dir (enables isolated parallel runs)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    # Set overrides BEFORE importing settings so they're picked up at module load.
    if args.output_dir:
        os.environ["OUTPUT_DIR"] = args.output_dir
    if args.prompt:
        os.environ["VISUAL_STYLE"] = STYLE_MAP[args.style]
        os.environ["NARRATION_LANGUAGE"] = args.lang

    # Import after env is set so settings.py picks up overrides
    from config import settings

    _setup_logging(args.log_level or settings.LOG_LEVEL)
    log = logging.getLogger(__name__)

    if args.prompt:
        _run_generate(args.prompt, args.style)
        return

    if args.phase:
        _run_single_phase(args.phase)
        return

    from pipeline.orchestrator import Pipeline

    log.info("Starting full pipeline…")
    results = Pipeline().run()
    failed = [r for r in results if not r.completed]
    if failed:
        log.error("%d run(s) failed", len(failed))
        sys.exit(1)
    log.info("All runs completed successfully.")


def _run_generate(prompt: str, style: str) -> None:
    """Product flow: user prompt + style → one downloadable video.
    Phase 2 (script) → Phase 3 (assets) → Phase 4 (assemble). No scraping, no upload."""
    from datetime import datetime, timezone

    from config import settings
    from pipeline.phase2_writer import ScriptWriter
    from pipeline.phase3_assets import AssetFactory
    from pipeline.phase4_editor import VideoEditor
    from schemas.models import TrendCandidate

    log = logging.getLogger(__name__)
    log.info("Generating video — style=%s (%s)", style, settings.VISUAL_STYLE)
    log.info("Prompt: %s", prompt)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wrap the user's prompt as a manual trend so Phase 2 can run without scraping.
    now = datetime.now(timezone.utc)
    trend = TrendCandidate(
        video_id="prompt",
        title=prompt,
        channel_title="User",
        views=500_000,
        published_at=now,
        days_since_upload=2.0,
        view_velocity=250_000.0,
        niche_keyword=settings.NICHE_KEYWORDS[0] if settings.NICHE_KEYWORDS else "educational",
        velocity_score=1.0,
    )

    script = ScriptWriter().run(trend)
    bundle = AssetFactory().run(script)
    result = VideoEditor().run(bundle)

    log.info("✅ Done — downloadable video: %s (%.1fs)",
             result.final_video_path, result.duration_seconds)
    print(f"\nVIDEO_READY: {result.final_video_path}")


def _run_single_phase(phase: int) -> None:
    log = logging.getLogger(__name__)
    log.info("Running Phase %d only…", phase)

    if phase == 1:
        from pipeline.phase1_scraper import TrendScraper
        TrendScraper().run()

    elif phase == 2:
        from config import settings
        from schemas.models import TrendCandidate
        import json
        from datetime import datetime, timezone

        trend_file = settings.OUTPUT_DIR / "trends.json"
        if not trend_file.exists():
            log.error("Run Phase 1 first to generate trends.json")
            sys.exit(1)

        data = json.loads(trend_file.read_text(encoding="utf-8"))
        trend = TrendCandidate(**data["top_concepts"][0])

        from pipeline.phase2_writer import ScriptWriter
        ScriptWriter().run(trend)

    elif phase == 3:
        from config import settings
        from schemas.models import VideoScript
        import json

        script_file = settings.OUTPUT_DIR / "script.json"
        if not script_file.exists():
            log.error("Run Phase 2 first to generate script.json")
            sys.exit(1)

        script = VideoScript.model_validate_json(script_file.read_text(encoding="utf-8"))
        from pipeline.phase3_assets import AssetFactory
        AssetFactory().run(script)

    elif phase == 4:
        from config import settings
        from schemas.models import AssetBundle
        import json

        # Reconstruct bundle from disk state
        script_file = settings.OUTPUT_DIR / "script.json"
        state_file = settings.OUTPUT_DIR / "pipeline_state.json"
        if not script_file.exists():
            log.error("Run Phases 1-3 first")
            sys.exit(1)

        from schemas.models import VideoScript, AssetBundle, AudioAsset, VisualAsset
        from pathlib import Path

        script = VideoScript.model_validate_json(script_file.read_text(encoding="utf-8"))
        audio_dir = settings.OUTPUT_DIR / "audio"
        video_dir = settings.OUTPUT_DIR / "video"

        import wave as _wave

        def _wav_dur(p: Path, fallback: int = 0) -> int:
            try:
                with _wave.open(str(p), "rb") as w:
                    return int(w.getnframes() / w.getframerate() * 1000)
            except Exception:
                return fallback

        audio_assets = [
            AudioAsset(
                scene_id=s.scene_id,
                path=audio_dir / f"scene_{s.scene_id}.wav",
                duration_ms=_wav_dur(audio_dir / f"scene_{s.scene_id}.wav", s.duration_ms or 0),
            )
            for s in script.scenes
        ]
        visual_assets = [
            VisualAsset(
                scene_id=s.scene_id,
                raw_video_path=video_dir / f"scene_{s.scene_id}_veo.mp4",
                keyframe_image_path=video_dir / f"scene_{s.scene_id}_keyframe.png",
            )
            for s in script.scenes
        ]

        bgm = None
        if settings.BGM_DIR.exists():
            tracks = list(settings.BGM_DIR.glob("*.mp3"))
            if tracks:
                bgm = tracks[0]

        bundle = AssetBundle(
            script=script,
            audio_assets=audio_assets,
            visual_assets=visual_assets,
            bgm_path=bgm,
        )
        from pipeline.phase4_editor import VideoEditor
        VideoEditor().run(bundle)

    elif phase == 5:
        from config import settings
        from schemas.models import VideoScript, RenderResult
        from pathlib import Path

        script_file = settings.OUTPUT_DIR / "script.json"
        render_file = settings.OUTPUT_DIR / "renders" / "final_render.mp4"

        if not script_file.exists() or not render_file.exists():
            log.error("Run Phases 1-4 first")
            sys.exit(1)

        script = VideoScript.model_validate_json(script_file.read_text(encoding="utf-8"))
        render = RenderResult(final_video_path=render_file, duration_seconds=0)

        from pipeline.phase5_publisher import YouTubePublisher
        YouTubePublisher().run(render, script)


if __name__ == "__main__":
    main()
