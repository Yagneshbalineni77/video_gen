"""
Phase 3 – Asset Factory (Google-only stack)
1. Gemini TTS  → per-scene narration .wav files
2. mutagen     → exact duration measurement → written back into script JSON
3. Prompt Enrichment → Gemini enhances each video_prompt for Veo
4. Veo 3.0     → per-scene video clips (via AI Studio API key)

v2: Adds prompt enrichment step — uses Gemini to transform Phase 2's video
    prompts into hyper-detailed Veo-optimized cinematic descriptions before
    generating each clip.
"""
from __future__ import annotations

import logging
import math
import time
import wave
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

from google import genai
from google.genai import types
from google.cloud import texttospeech
from google.oauth2 import service_account
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from schemas.models import AssetBundle, AudioAsset, CharacterRef, SceneScript, Storyboard, VideoScript, VisualAsset

log = logging.getLogger(__name__)

# Deep dramatic voices available in Gemini TTS
# Charon = deep dramatic male, Fenrir = intense male, Kore = calm female
_TTS_VOICE = settings.GEMINI_TTS_VOICE

_ENRICHMENT_SYSTEM = """You are a veteran cinematographer writing prompts for Veo 3.0, 
Google's state-of-the-art AI video generator. Your job is to transform draft video 
descriptions into hyper-detailed, production-ready prompts that produce cinematic masterpieces.

RULES:
- Output ONLY the rewritten prompt (60-100 words). No explanations, no labels.
- Start with the EXACT camera movement and shot type
- Describe ONLY visible, tangible physical objects and environments
- Focus heavily on the human element, capturing highly authentic micro-expressions and raw emotion on the character's face
- Frame the character dynamically (e.g., extreme close-up on eyes, over-the-shoulder)
- Name the lighting technique (chiaroscuro, volumetric, rim light, etc.)
- Include lens/depth-of-field (shallow DOF, anamorphic, wide-angle, macro)
- State the color palette (teal-orange, cold desaturated, warm amber)
- End with an ambient audio cue for Veo's native audio generation
- Every element must be SPECIFIC to the narration context — not generic atmosphere
- Mandate perfectly smooth, highly dynamic cinematic camera movements to ensure the video does not look static.
- IMPORTANT SAFETY RULE: AVOID overly graphic, violent, or intensely terrifying descriptions (e.g. no "raw terror", "screaming", "panic"). Express emotion through profound but safe human expressions (tears, subtle smiles, longing).
- Use professional cinematography terminology throughout"""

class AssetFactory:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._audio_dir = settings.OUTPUT_DIR / "audio"
        self._video_dir = settings.OUTPUT_DIR / "video"
        self._char_dir = settings.OUTPUT_DIR / "characters"
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._char_dir.mkdir(parents=True, exist_ok=True)

    # ── public ──────────────────────────────────────────────────────────────

    def run(self, script: VideoScript) -> AssetBundle:
        log.info("Phase 3 – generating assets for %d scenes", len(script.scenes))

        audio_assets = self._generate_audio(script)
        script = self._annotate_durations(script, audio_assets)
        self._save_script(script)          # persist duration_ms so Phase 4 can read it

        # Generate character reference portraits for visual consistency
        if script.storyboard and script.storyboard.characters:
            self._generate_character_references(script.storyboard)

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
        creds = service_account.Credentials.from_service_account_file(settings.VEO_SERVICE_ACCOUNT_PATH)
        client = texttospeech.TextToSpeechClient(credentials=creds)
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="hi-IN",
            name="hi-IN-Neural2-B"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000
        )
        
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        dest.write_bytes(response.audio_content)

    @staticmethod
    def _wav_duration_ms(path: Path) -> int:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return int(frames / rate * 1000)

    # ── script annotation ────────────────────────────────────────────────────

    @staticmethod
    def _annotate_durations(script: VideoScript, audio_assets: list[AudioAsset]) -> VideoScript:
        dur_map = {a.scene_id: a.duration_ms for a in audio_assets}
        for scene in script.scenes:
            scene.duration_ms = dur_map.get(scene.scene_id)
        return script

    # ── character reference portraits ────────────────────────────────────────

    def _generate_character_references(self, storyboard: Storyboard) -> None:
        """Generate a reference portrait for each character in the cast.
        These anchors ensure visual consistency across all scene keyframes."""
        for char in storyboard.characters:
            ref_path = self._char_dir / f"{char.name.lower().replace(' ', '_')}_ref.png"
            if ref_path.exists():
                log.info("  [character] '%s' reference cached", char.name)
                char.reference_image_path = ref_path
                continue

            log.info("  [character] generating reference portrait for '%s'", char.name)
            prompt = (
                f"Professional character reference portrait, photorealistic, neutral gray background, "
                f"soft studio lighting, 16:9 aspect ratio.\n\n"
                f"CHARACTER: {char.physical_description}\n\n"
                f"Show the character in a neutral standing or seated pose, facing slightly left, "
                f"with clear visibility of face, hair, clothing, and build. "
                f"No text, no labels, no props. Clean background."
            )

            try:
                response = self._client.models.generate_images(
                    model='imagen-4.0-generate-001',
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9",
                        output_mime_type="image/png",
                    ),
                )
                image_bytes = response.generated_images[0].image.image_bytes
                ref_path.write_bytes(image_bytes)
                char.reference_image_path = ref_path
                log.info("  [character] '%s' reference -> %s", char.name, ref_path.name)
            except Exception as exc:
                log.warning("  [character] failed to generate reference for '%s': %s", char.name, exc)
                # Non-fatal — we'll still use text descriptions as fallback

    def _build_character_anchor(self, script: VideoScript, scene: SceneScript) -> str:
        """Build a text block describing exactly which characters appear in this scene
        with their locked physical descriptions. This is prepended to image/video prompts
        to enforce visual consistency."""
        if not script.storyboard or not script.storyboard.characters:
            return ""

        # Find which characters are in this scene
        comp = next(
            (sc for sc in script.storyboard.scene_compositions if sc.scene_id == scene.scene_id),
            None,
        )
        if not comp or not comp.characters_present:
            return ""

        char_map = {c.name: c for c in script.storyboard.characters}
        lines = ["CHARACTERS IN THIS SCENE (use these EXACT descriptions):"]
        for name in comp.characters_present:
            char = char_map.get(name)
            if char:
                lines.append(f"  • {char.name}: {char.physical_description}")

        return "\n".join(lines)

    # ── prompt enrichment ────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _enrich_video_prompt(self, scene: SceneScript, style_brief: str, character_anchor: str = "") -> str:
        """Use Gemini to transform a Phase 2 video prompt into a Veo-optimized
        cinematic masterpiece prompt with full directorial detail."""

        style = settings.VISUAL_STYLE_PRESETS.get(
            settings.VISUAL_STYLE,
            settings.VISUAL_STYLE_PRESETS["dark_cinematic"],
        )

        char_block = ""
        if character_anchor:
            char_block = (
                f"\n{character_anchor}\n\n"
                f"CRITICAL: You MUST use the EXACT physical descriptions above for any character "
                f"in this scene. Do NOT change their appearance, clothing, age, or features.\n"
            )

        prompt = (
            f"VISUAL STYLE DNA: {style_brief}\n"
            f"Color palette: {style['color_palette']}\n"
            f"Lighting style: {style['lighting']}\n"
            f"Lens: {style['lens']}\n"
            f"Textures: {style['texture']}\n"
            f"{char_block}\n"
            f"NARRATION (for context — the visual must illustrate this):\n"
            f'"{scene.narration}"\n\n'
            f"DRAFT VIDEO PROMPT to enhance:\n"
            f'"{scene.video_prompt}"\n\n'
            f"(60-100 words) optimized for Veo 3.0. Produce highly literal, textbook-style, hyper-realistic 3D scientific visualizations or diagrams. Do NOT include human characters or faces unless strictly necessary. Focus entirely on the academic subject matter. "
            f"CRITICAL: Do NOT use real names of public figures or historical people to avoid safety filters. "
            f"Only the rewritten prompt, nothing else."
        )

        response = self._client.models.generate_content(
            model=settings.GEMINI_SCRIPT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_ENRICHMENT_SYSTEM,
                temperature=0.7,
            ),
        )
        enriched = response.text.strip().strip('"').strip("'")
        log.info(
            "  [enrich] scene %d: %d->%d words",
            scene.scene_id,
            len(scene.video_prompt.split()),
            len(enriched.split()),
        )
        return enriched

    # ── visuals via Imagen 3.0 & Veo 3.0 ─────────────────────────────────────

    def _generate_visuals(self, script: VideoScript) -> list[VisualAsset]:
        assets: list[VisualAsset] = []
        for scene in script.scenes:
            # Build character anchor for this specific scene
            character_anchor = self._build_character_anchor(script, scene)
            # 1. Enrich prompt (now with character descriptions injected)
            enriched_prompt = self._enrich_video_prompt(scene, script.visual_style_brief, character_anchor)
            # 2. Generate perfect composition image via Imagen 3
            keyframe_path = self._gen_keyframe_image(scene, enriched_prompt, character_anchor)
            # 3. Pass image to Veo to animate
            veo_path = self._gen_veo(scene, enriched_prompt, keyframe_path)
            
            assets.append(VisualAsset(
                scene_id=scene.scene_id,
                raw_video_path=veo_path,
                keyframe_image_path=keyframe_path
            ))
        return assets

    @retry(stop=stop_after_attempt(50), wait=wait_exponential(multiplier=3, min=30, max=300))
    def _gen_keyframe_image(self, scene: SceneScript, enriched_prompt: str, character_anchor: str = "") -> Path:
        dest = self._video_dir / f"scene_{scene.scene_id}_keyframe.png"
        if dest.exists():
            log.info("  [visual/imagen3] scene %d keyframe cached", scene.scene_id)
            return dest

        # Prepend character descriptions to anchor the image to the same people
        full_prompt = enriched_prompt
        if character_anchor:
            full_prompt = (
                f"{character_anchor}\n\n"
                f"SCENE VISUAL:\n{enriched_prompt}"
            )

        log.info("  [visual/imagen3] scene %d generating keyframe...", scene.scene_id)
        response = self._client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=full_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
                output_mime_type="image/png"
            )
        )
        if not response.generated_images:
            raise RuntimeError(f"Imagen returned no images for scene {scene.scene_id} (likely safety filter block). Prompt was: {full_prompt}")
            
        image_bytes = response.generated_images[0].image.image_bytes
        dest.write_bytes(image_bytes)
        return dest

    def _gen_veo(self, scene: SceneScript, enriched_prompt: str, keyframe_path: Path) -> Path:
        dest = self._video_dir / f"scene_{scene.scene_id}_veo.mp4"
        if dest.exists():
            log.info("  [visual/veo3] scene %d cached", scene.scene_id)
            return dest

        log.info(
            "  [visual/veo3] scene %d enriched prompt:\n    %s",
            scene.scene_id,
            enriched_prompt[:200],
        )

        # How many 8s clips do we need to cover the narration without looping?
        audio_secs = (scene.duration_ms or 8000) / 1000.0
        n_clips = max(1, math.ceil(audio_secs / 8))
        n_request = min(n_clips, 4)  # Veo supports up to 4 per call

        log.info(
            "  [visual/veo3] scene %d – requesting %d clip(s) for %.1fs narration",
            scene.scene_id, n_request, audio_secs,
        )

        clips = [self._fetch_veo_clip(scene, enriched_prompt, keyframe_path, i) for i in range(n_request)]

        if len(clips) == 1:
            clips[0].rename(dest)
        else:
            self._concat_clips(clips, dest)
            for c in clips:
                c.unlink(missing_ok=True)

        log.info("  [visual/veo3] scene %d -> %s (%.1fs covered)", scene.scene_id, dest.name, n_request * 8)
        return dest

    @retry(stop=stop_after_attempt(50), wait=wait_exponential(multiplier=3, min=30, max=300))
    def _fetch_veo_clip(self, scene: SceneScript, prompt: str, keyframe_path: Path, clip_idx: int) -> Path:
        dest = self._video_dir / f"scene_{scene.scene_id}_clip{clip_idx}.mp4"
        if dest.exists():
            return dest

        # Build Veo config
        veo_config = types.GenerateVideosConfig(
            aspect_ratio="16:9",
            duration_seconds=8,
            number_of_videos=1,
        )
        
        # In Veo image-to-video, we must pass the image in the source
        image_obj = types.Image(
            image_bytes=keyframe_path.read_bytes(),
            mime_type="image/png"
        )
        source = types.GenerateVideosSource(
            image=image_obj,
            prompt="Cinematically pan and animate this scene with perfect smooth and highly dynamic cinematic camera movements. Add subtle environmental movement like smoke, dust, or gentle camera drift. " + prompt
        )

        operation = self._client.models.generate_videos(
            model=settings.VEO_MODEL,
            source=source,
            config=veo_config,
        )

        for attempt in range(90):
            operation = self._client.operations.get(operation)
            if operation.done:
                break
            log.debug("  [veo3] scene %d clip %d polling (%ds)", scene.scene_id, clip_idx, (attempt + 1) * 5)
            time.sleep(5)
        else:
            raise TimeoutError(f"Veo timed out for scene {scene.scene_id} clip {clip_idx}")

        if operation.error:
            raise RuntimeError(f"Veo error scene {scene.scene_id} clip {clip_idx}: {operation.error}")
        if not operation.response:
            raise RuntimeError(f"Veo returned no response for scene {scene.scene_id} clip {clip_idx}")

        videos = operation.response.generated_videos
        if not videos:
            raise RuntimeError(f"Veo returned empty video list for scene {scene.scene_id} clip {clip_idx}")
        generated = videos[0]
        video_bytes = self._client.files.download(file=generated)
        dest.write_bytes(video_bytes)
        return dest

    def _concat_clips(self, clips: list[Path], dest: Path) -> None:
        from utils.ffmpeg_helpers import run_ffmpeg
        list_file = dest.parent / f"_concat_{dest.stem}.txt"
        list_file.write_text(
            "\n".join(f"file '{p.resolve().as_posix()}'" for p in clips),
            encoding="utf-8",
        )
        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c:v", "copy", "-an", str(dest)],
            desc=f"concat {len(clips)} clips for scene",
        )
        list_file.unlink(missing_ok=True)

    # ── BGM ──────────────────────────────────────────────────────────────────

    def _pick_bgm(self) -> Path | None:
        bgm_dir = settings.BGM_DIR
        if not bgm_dir.exists():
            log.warning("No BGM directory at %s – skipping background music", bgm_dir)
            return None
        tracks = list(bgm_dir.glob("*.mp3")) + list(bgm_dir.glob("*.wav"))
        if not tracks:
            return None
        import random
        track = random.choice(tracks)
        log.info("BGM: %s", track.name)
        return track


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_wav(dest: Path, pcm_data: bytes, sample_rate: int, channels: int, sampwidth: int) -> None:
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
