import logging
from pathlib import Path

from config import settings
from pipeline.phase3_animator import AnimatorFactory
from schemas.models import VideoScript

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger(__name__)


def evaluate_html(html: str) -> bool:
    if "<svg" not in html and "div" not in html:
        return False
    if "gsap.timeline" not in html and "gsap.to" not in html:
        return False
    if len(html) < 1500:
        return False
    if "stick" in html.lower():
        return False
    return True


def main():
    script_file = settings.OUTPUT_DIR / "script.json"
    script = VideoScript.model_validate_json(script_file.read_text())
    scene = script.scenes[0]

    duration_secs = (scene.duration_ms or 8000) / 1000.0

    factory = AnimatorFactory()

    for attempt in range(1, 11):
        log.info("--- Iteration %d ---", attempt)
        html_code = factory._generate_html_code(scene, duration_secs)
        log.info("Generated HTML size: %d bytes", len(html_code))

        if evaluate_html(html_code):
            log.info("Perfect sophisticated HTML found! Saving...")
            dest_html = factory._html_dir / f"scene_{scene.scene_id}.html"
            dest_html.write_text(html_code, encoding="utf-8")

            dest_mp4 = factory._video_dir / f"scene_{scene.scene_id}_anim.mp4"
            dest_mp4.unlink(missing_ok=True)
            log.info("Recording MP4...")
            factory._record_html_to_mp4(dest_html, scene.duration_ms or 8000, dest_mp4)
            log.info("Done recording MP4!")
            break
        else:
            log.warning("Output was too simple or contained stick figures. Retrying...")


if __name__ == "__main__":
    main()
