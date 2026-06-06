"""
Pipeline test runner — validates each phase independently.
Usage:
  .venv\\Scripts\\python test_pipeline.py --phase 2     Gemini script
  .venv\\Scripts\\python test_pipeline.py --phase 3a    TTS audio
  .venv\\Scripts\\python test_pipeline.py --phase 3b    Veo 3.0 video (~2 min)
  .venv\\Scripts\\python test_pipeline.py --phase 4     FFmpeg check
  .venv\\Scripts\\python test_pipeline.py --all         Run 2 + 3a + 4
"""
from __future__ import annotations

import argparse
import logging
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("test")

MOCK_TREND = {
    "video_id": "test123",
    "title": "The Secret History of the Roman Empire Nobody Talks About",
    "channel_title": "Dark History",
    "views": 850_000,
    "published_at": datetime.now(timezone.utc).isoformat(),
    "days_since_upload": 3.0,
    "view_velocity": 283_333.0,
    "niche_keyword": "dark history",
    "velocity_score": 1.0,
}


def test_phase2():
    log.info("=" * 55)
    log.info("PHASE 2 - Gemini Script Writer")
    log.info("=" * 55)
    from schemas.models import TrendCandidate
    from pipeline.phase2_writer import ScriptWriter

    trend = TrendCandidate(**MOCK_TREND)
    script = ScriptWriter().run(trend)

    log.info("OK - Script generated")
    log.info("  Title   : %s", script.metadata.title)
    log.info("  Scenes  : %d", len(script.scenes))
    log.info("  Tags    : %s", ", ".join(script.metadata.tags[:5]))
    log.info("  Scene 1 : %s...", script.scenes[0].narration[:80])
    return script


def test_phase3a(script=None):
    log.info("=" * 55)
    log.info("PHASE 3A - Gemini TTS (scene 1)")
    log.info("=" * 55)

    if script is None:
        from schemas.models import VideoScript
        f = Path("output/script.json")
        if not f.exists():
            log.error("Run --phase 2 first")
            sys.exit(1)
        script = VideoScript.model_validate_json(f.read_text())

    from google import genai
    from google.genai import types
    from config import settings

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    scene = script.scenes[0]
    log.info("Narration: '%s...'", scene.narration[:60])

    response = client.models.generate_content(
        model=settings.GEMINI_TTS_MODEL,
        contents=scene.narration,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=settings.GEMINI_TTS_VOICE,
                    )
                )
            ),
        ),
    )

    audio_data = response.candidates[0].content.parts[0].inline_data.data
    out = Path("output/audio/test_scene_1.wav")
    out.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)

    with wave.open(str(out), "rb") as wf:
        dur = round(wf.getnframes() / wf.getframerate(), 2)

    log.info("OK - TTS saved: %s (%.2fs, voice=%s)", out.name, dur, settings.GEMINI_TTS_VOICE)
    return out


def test_phase3b(script=None):
    log.info("=" * 55)
    log.info("PHASE 3B - Veo 3.0 (scene 1, ~2 min)")
    log.info("=" * 55)

    if script is None:
        from schemas.models import VideoScript
        f = Path("output/script.json")
        if not f.exists():
            log.error("Run --phase 2 first")
            sys.exit(1)
        script = VideoScript.model_validate_json(f.read_text())

    from pipeline.phase3_assets import AssetFactory
    factory = AssetFactory()
    scene = script.scenes[0]
    log.info("Prompt: '%s...'", scene.video_prompt[:70])
    asset = factory._gen_veo(scene)
    log.info("OK - Veo video saved: %s", asset.raw_video_path)
    return asset


def test_phase4():
    log.info("=" * 55)
    log.info("PHASE 4 - FFmpeg check")
    log.info("=" * 55)
    import subprocess
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if r.returncode != 0:
        log.error("FAIL - FFmpeg not in PATH")
        return False
    log.info("OK - %s", r.stdout.split("\n")[0])
    r2 = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
    log.info("OK - ffprobe available")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["2", "3a", "3b", "4", "all"], default="all")
    args = parser.parse_args()

    if args.phase == "all":
        script = test_phase2()
        test_phase3a(script)
        test_phase4()
        log.info("")
        log.info("All fast tests passed!")
        log.info("Run --phase 3b to test Veo video generation (~2 min)")
    elif args.phase == "2":
        test_phase2()
    elif args.phase == "3a":
        test_phase3a()
    elif args.phase == "3b":
        test_phase3b()
    elif args.phase == "4":
        test_phase4()


if __name__ == "__main__":
    main()
