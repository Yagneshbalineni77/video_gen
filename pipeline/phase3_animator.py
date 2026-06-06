"""
Phase 3 – Code-Driven Animation Engine
1. Gemini TTS  → per-scene narration .wav files
2. mutagen     → exact duration measurement → written back into script JSON
3. Gemini Script → writes GSAP HTML5 animation for each scene
4. Playwright  → loads HTML, screen-records to exact duration, saves MP4
"""
from __future__ import annotations

import logging
import time
import wave
import shutil
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from playwright.sync_api import sync_playwright

from config import settings
from schemas.models import AssetBundle, AudioAsset, SceneScript, VideoScript, VisualAsset

log = logging.getLogger(__name__)

_TTS_VOICE = settings.GEMINI_TTS_VOICE

_ANIMATOR_SYSTEM = """You are a world-class motion graphics designer creating premium educational animations for YouTube.
Create a single self-contained HTML file with inline CSS and JS that produces a stunning, broadcast-quality 2D animation.

VISUAL QUALITY REQUIREMENTS:
- Output ONLY raw HTML code. No markdown formatting, no backticks, no explanations.
- Canvas: exactly 1920x1080 (body { margin: 0; overflow: hidden; width: 1920px; height: 1080px; })
- Include GSAP: <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
- Use a dark, premium background (deep navy #0a0e27, charcoal #1a1a2e, or rich dark #0f0f23) with subtle gradient.
- Use SVG for ALL illustrations — clean vector art, NOT stick figures. Think: polished infographic style.
- Typography: Use Google Fonts (Inter, Poppins, or Outfit). Import via @import in a <style> block.
- Color scheme: vibrant accent colors (electric blue #4fc3f7, warm amber #ffb74d, emerald #66bb6a, coral #ef5350) against the dark background.
- Add subtle glow effects (box-shadow, filter: drop-shadow) on key elements.
- Smooth easing on ALL animations: use GSAP's "power2.out", "elastic.out", "back.out" easings.

ANIMATION PRINCIPLES:
- Elements must animate IN (fade+slide or scale) — never just appear instantly.
- Stagger related items (GSAP stagger: 0.15s between siblings).
- Use GSAP timeline for precise sequencing. The total animation MUST last EXACTLY the specified duration.
- Add subtle continuous motion (floating particles, gentle pulse on key elements, breathing scale).
- Text should animate word-by-word or line-by-line, never as a single block.
- Use morphing, scaling, and rotation for transitions between concepts.
- Include connecting lines/arrows that draw themselves (SVG stroke-dashoffset animation).

LAYOUT RULES:
- Center key content with generous padding (min 100px from edges).
- Use visual hierarchy: large title (48-64px), medium labels (28-36px), small annotations (18-24px).
- Icons and diagrams should be at least 200x200px — never tiny.
- When showing models/diagrams, make them the hero of the frame (500-800px).

WHAT TO AVOID:
- No stick figures. No crude drawings. No plain white backgrounds.
- No Comic Sans or default serif fonts.
- No static frames — everything must have motion.
- No text walls — use icons, shapes, and visuals to convey meaning."""


class AnimatorFactory:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._audio_dir = settings.OUTPUT_DIR / "audio"
        self._video_dir = settings.OUTPUT_DIR / "video"
        self._html_dir = settings.OUTPUT_DIR / "html"
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._html_dir.mkdir(parents=True, exist_ok=True)

    def run(self, script: VideoScript) -> AssetBundle:
        log.info("Phase 3 (Animator) – generating assets for %d scenes", len(script.scenes))

        audio_assets = self._generate_audio(script)
        script = self._annotate_durations(script, audio_assets)
        self._save_script(script)

        visual_assets = self._generate_visuals(script)
        bgm = self._pick_bgm()

        bundle = AssetBundle(
            script=script,
            audio_assets=audio_assets,
            visual_assets=visual_assets,
            bgm_path=bgm,
        )
        log.info("Phase 3 done – %d audio, %d visual assets", len(audio_assets), len(visual_assets))
        return bundle

    @staticmethod
    def _save_script(script: VideoScript) -> None:
        out = settings.OUTPUT_DIR / "script.json"
        out.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        log.info("Script (with durations) saved -> %s", out)

    # ── audio via Gemini TTS ─────────────────────────────────────────────────

    def _generate_audio(self, script: VideoScript) -> list[AudioAsset]:
        assets: list[AudioAsset] = []
        for scene in script.scenes:
            wav_path = self._audio_dir / f"scene_{scene.scene_id}.wav"
            if wav_path.exists():
                log.info("  [audio] scene %d cached", scene.scene_id)
            else:
                self._tts(scene.narration, wav_path)
                log.info("  [audio] scene %d -> %s", scene.scene_id, wav_path.name)

            duration_ms = self._wav_duration_ms(wav_path)
            assets.append(AudioAsset(scene_id=scene.scene_id, path=wav_path, duration_ms=duration_ms))
        return assets

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
    def _tts(self, text: str, dest: Path) -> None:
        response = self._client.models.generate_content(
            model=settings.GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=_TTS_VOICE)
                    )
                ),
            ),
        )
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        with wave.open(str(dest), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_data)

    @staticmethod
    def _wav_duration_ms(path: Path) -> int:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return int(frames / rate * 1000)

    @staticmethod
    def _annotate_durations(script: VideoScript, audio_assets: list[AudioAsset]) -> VideoScript:
        dur_map = {a.scene_id: a.duration_ms for a in audio_assets}
        for scene in script.scenes:
            scene.duration_ms = dur_map.get(scene.scene_id)
        return script

    # ── HTML animation ───────────────────────────────────────────────────────

    def _generate_visuals(self, script: VideoScript) -> list[VisualAsset]:
        assets: list[VisualAsset] = []
        for scene in script.scenes:
            asset = self._gen_animation(scene)
            assets.append(asset)
        return assets

    def _gen_animation(self, scene: SceneScript) -> VisualAsset:
        dest_mp4 = self._video_dir / f"scene_{scene.scene_id}_raw.mp4"
        dest_html = self._html_dir / f"scene_{scene.scene_id}.html"

        if dest_mp4.exists():
            log.info("  [animator] scene %d cached", scene.scene_id)
            return VisualAsset(scene_id=scene.scene_id, raw_video_path=dest_mp4)

        duration_secs = (scene.duration_ms or 8000) / 1000.0

        if not dest_html.exists():
            log.info("  [animator] generating HTML for scene %d (%.1fs)", scene.scene_id, duration_secs)
            html_code = self._generate_html_code(scene, duration_secs)
            dest_html.write_text(html_code, encoding="utf-8")

        log.info("  [animator] recording MP4 for scene %d via Playwright", scene.scene_id)
        self._record_html_to_mp4(dest_html, scene.duration_ms or 8000, dest_mp4)
        
        log.info("  [animator] scene %d -> %s", scene.scene_id, dest_mp4.name)
        return VisualAsset(scene_id=scene.scene_id, raw_video_path=dest_mp4)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_html_code(self, scene: SceneScript, duration_secs: float) -> str:
        prompt = (
            f"DURATION: {duration_secs} seconds\n\n"
            f"NARRATION:\n\"{scene.narration}\"\n\n"
            f"VISUAL PROMPT:\n\"{scene.video_prompt}\"\n\n"
            f"Write the HTML. It must exactly match the {duration_secs}s duration using GSAP timelines."
        )

        response = self._client.models.generate_content(
            model=settings.GEMINI_SCRIPT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_ANIMATOR_SYSTEM,
                temperature=0.7,
            ),
        )
        
        html = response.text.strip()
        # Fallback strip markdown if model hallucinates it despite instructions
        if html.startswith("```html"):
            html = html[7:]
        elif html.startswith("```"):
            html = html[3:]
        if html.endswith("```"):
            html = html[:-3]
            
        return html.strip()

    def _record_html_to_mp4(self, html_path: Path, duration_ms: int, dest_mp4: Path) -> None:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                record_video_dir=str(self._video_dir),
                record_video_size={"width": 1920, "height": 1080}
            )
            page = context.new_page()
            page.set_viewport_size({"width": 1920, "height": 1080})
            
            # Use file URL
            file_url = f"file:///{html_path.resolve().as_posix()}"
            page.goto(file_url)
            
            # Wait for animation to finish playing
            page.wait_for_timeout(duration_ms)
            
            video_path = page.video.path()
            context.close()
            browser.close()
            
            shutil.move(video_path, dest_mp4)

    def _pick_bgm(self) -> Path | None:
        bgm_dir = settings.BGM_DIR
        if not bgm_dir.exists():
            return None
        tracks = list(bgm_dir.glob("*.mp3")) + list(bgm_dir.glob("*.wav"))
        if not tracks:
            return None
        import random
        return random.choice(tracks)
