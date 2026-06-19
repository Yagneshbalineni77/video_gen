"""
Phase 3 – Asset Factory (Google-only stack)
1. Dextora TTS  → per-scene narration .wav files
2. mutagen     → exact duration measurement → written back into script JSON
3. Prompt Enrichment → Gemini enhances each video_prompt for Veo
4. Dextora     → per-scene video clips (via AI Studio API key)

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
import threading
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from config import settings
from schemas.models import AssetBundle, AudioAsset, CharacterRef, SceneScript, Storyboard, VideoScript, VisualAsset

log = logging.getLogger(__name__)


class VeoBlockedError(RuntimeError):
    """Veo completed but returned no video — a safety/RAI filter block.
    Re-submitting the same prompt is pointless, so this is NOT retried."""


class ImagenBlockedError(RuntimeError):
    """Imagen returned no images — a safety filter block. Deterministic for a
    given prompt, so this is NOT retried (avoids a multi-hour retry hang)."""

# Deep dramatic voices available in Dextora TTS
# Charon = deep dramatic male, Fenrir = intense male, Kore = calm female
_TTS_VOICE = settings.GEMINI_TTS_VOICE

def _build_enrichment_system(style: dict) -> str:
    """Build the enrichment system prompt for the ACTIVE visual style.

    The render medium is the single biggest lever on style consistency: a
    hardcoded photoreal prompt drags Ghibli/3D-textbook scenes back toward
    live-action. So we make the medium mandatory and adapt the subject rules
    to whether the style is photoreal or illustrated/rendered."""
    medium = style.get("medium", "photorealistic cinematic film")
    photoreal = style.get("photoreal", True)

    if photoreal:
        subject_rule = (
            "- When a person is present, capture authentic emotion via micro-expressions and faces\n"
            "- Use real-camera language: lens, depth-of-field, film grain, bokeh"
        )
    else:
        subject_rule = (
            f"- Render EVERY element strictly in this medium: {medium}\n"
            "- Do NOT describe photorealistic skin, real-camera lenses, film grain, or live-action realism\n"
            "- Keep the illustrated/rendered look identical in every described element"
        )

    return f"""You are a veteran art director writing prompts for Dextora, Google's
state-of-the-art AI video generator. Transform draft descriptions into hyper-detailed,
production-ready prompts whose every frame matches ONE consistent visual medium.

RENDER MEDIUM (MANDATORY — every frame MUST look exactly like this):
{medium}

RULES:
- Output ONLY the rewritten prompt (60-100 words). No explanations, no labels.
- The prompt MUST OPEN by naming the render medium above, then the camera move.
- Describe ONLY visible, tangible physical objects and environments.
{subject_rule}
- Name the lighting technique (chiaroscuro, volumetric, rim light, dappled, etc.).
- State the dominant color palette for the shot.
- End with a brief ambient audio cue.
- Every element must be SPECIFIC to the narration context — not generic atmosphere.
- Mandate smooth, dynamic camera movement so the shot is never static.
- SAFETY: avoid graphic/violent/terrifying wording (no "raw terror", "screaming",
  "panic"); express emotion safely (tears, subtle smiles, longing).
- NEVER depict a periodic table of elements — AI image generators cannot render
  accurate chemical symbols (they hallucinate fake ones). Instead show: 3D molecular
  models, glowing atomic orbitals, chemistry lab glassware, molecular lattice
  structures, or coloured electron-cloud diagrams. Replace any "periodic table"
  reference with one of these visually accurate alternatives."""


class AssetFactory:
    def __init__(self) -> None:
        # Key pool: one client per API key. Calls round-robin across them to spread
        # load past per-key rate limits (see settings.GEMINI_API_KEYS).
        self._clients = [genai.Client(api_key=k) for k in settings.GEMINI_API_KEYS]
        self._client = self._clients[0]          # primary (default / non-rotated)
        self._key_i = 0
        self._key_lock = threading.Lock()
        if len(self._clients) > 1:
            log.info("  [keys] API key pool active: %d keys (round-robin)", len(self._clients))
        self._audio_dir = settings.OUTPUT_DIR / "audio"
        self._video_dir = settings.OUTPUT_DIR / "video"
        self._char_dir = settings.OUTPUT_DIR / "characters"
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._char_dir.mkdir(parents=True, exist_ok=True)

    def _next_client(self):
        """Thread-safe round-robin over the key pool. One-shot calls (TTS, Imagen,
        enrichment) rotate per call; a Veo clip grabs one client for its whole
        generate→poll→download lifecycle (the operation handle is key-bound)."""
        if len(self._clients) == 1:
            return self._clients[0]
        with self._key_lock:
            c = self._clients[self._key_i % len(self._clients)]
            self._key_i += 1
        return c

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

    # ── audio via Dextora TTS ─────────────────────────────────────────────────

    @staticmethod
    def _tts_style() -> tuple[str, str]:
        """Resolve (voice, delivery-tone) for the active visual style."""
        preset = settings.TTS_STYLE_PRESETS.get(settings.VISUAL_STYLE)
        if preset:
            return preset["voice"], preset["tone"]
        # No style-specific entry → fall back to env voice + the default tone.
        return settings.GEMINI_TTS_VOICE, settings.TTS_DEFAULT["tone"]

    def _generate_audio(self, script: VideoScript) -> list[AudioAsset]:
        voice, tone = self._tts_style()
        log.info("  [audio] voice=%s, directed delivery for style '%s'", voice, settings.VISUAL_STYLE)
        assets: list[AudioAsset] = []
        for scene in script.scenes:
            wav_path = self._audio_dir / f"scene_{scene.scene_id}.wav"
            if wav_path.exists():
                log.info("  [audio] scene %d cached", scene.scene_id)
            else:
                try:
                    self._tts(scene.narration, wav_path, voice, tone)
                    log.info("  [audio] scene %d -> %s", scene.scene_id, wav_path.name)
                except Exception as exc:  # blocked/empty TTS — never kill the whole video
                    log.warning("  [audio] scene %d TTS failed (%s); retrying as plain read",
                                scene.scene_id, exc)
                    try:
                        self._tts(scene.narration, wav_path, voice, plain=True)
                        log.info("  [audio] scene %d -> %s (plain)", scene.scene_id, wav_path.name)
                    except Exception as exc2:
                        secs = max(2.0, len(scene.narration.split()) / 2.6)
                        log.warning("  [audio] scene %d TTS unrecoverable (%s); writing %.1fs "
                                    "silence so the video still completes (subtitle still shows)",
                                    scene.scene_id, exc2, secs)
                        self._write_silence(wav_path, secs)

            duration_ms = self._wav_duration_ms(wav_path)
            assets.append(AudioAsset(scene_id=scene.scene_id, path=wav_path, duration_ms=duration_ms))
        return assets

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
    def _tts(self, text: str, dest: Path, voice: str | None = None,
             tone: str | None = None, plain: bool = False) -> None:
        voice = voice or settings.GEMINI_TTS_VOICE
        if plain:
            # Fallback path: speak the raw text with no delivery directive — fewer
            # tokens for a safety filter to trip on if the directive was the problem.
            contents = text
        else:
            # Style prompt: Dextora TTS interprets a leading directive as DELIVERY guidance
            # (it speaks only the text after it). This turns flat read-aloud into a
            # directed performance — the biggest single TTS quality lever, and free.
            # The active narration profile adds the accent/language (e.g. Indian English).
            directive = tone or settings.TTS_DEFAULT["tone"]
            profile = settings.narration_profile()
            if profile.get("tts_directive"):
                directive = f"{directive}, {profile['tts_directive']}"
            contents = f"{directive}:\n\n{text}"
        response = self._next_client().models.generate_content(
            model=settings.GEMINI_TTS_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
            ),
        )
        audio_data = self._extract_audio(response)
        with wave.open(str(dest), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_data)

    @staticmethod
    def _extract_audio(response) -> bytes:
        """Pull PCM bytes out of a TTS response, defensively.

        A blocked/empty candidate has ``content is None`` (the old code did
        ``response.candidates[0].content.parts[0]`` and threw AttributeError,
        killing the whole job). Here we surface *why* and raise a retryable
        error instead, so transient empties retry and the caller can fall back."""
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            for part in (parts or []):
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline else None
                if data:
                    return data
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(
            f"TTS returned no audio (finish_reason={finish}, prompt_feedback={feedback})"
        )

    @staticmethod
    def _write_silence(dest: Path, seconds: float) -> None:
        """Write a mono 24 kHz 16-bit silent WAV so a failed scene still has a
        valid, correctly-timed audio track and the render completes."""
        rate = 24000
        frames = int(rate * max(0.5, seconds))
        with wave.open(str(dest), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(b"\x00\x00" * frames)

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
                response = self._next_client().models.generate_images(
                    model=settings.GEMINI_IMAGE_MODEL,
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
    def _enrich_video_prompt(
        self,
        scene: SceneScript,
        style_brief: str,
        character_anchor: str = "",
        prev_narration: str = "",
        next_narration: str = "",
    ) -> str:
        """Transform a Phase 2 prompt into a Veo-optimized cinematic prompt.

        Accepts adjacent scene narrations to produce visuals that flow
        naturally from the previous scene and bridge into the next."""

        style = settings.VISUAL_STYLE_PRESETS.get(
            settings.VISUAL_STYLE,
            settings.VISUAL_STYLE_PRESETS["dark_cinematic"],
        )

        char_block = ""
        if character_anchor:
            char_block = (
                f"\n{character_anchor}\n\n"
                f"CRITICAL: Use EXACT physical descriptions above for any character. "
                f"Do NOT alter appearance, clothing, age, or features.\n"
            )

        continuity_block = ""
        if prev_narration:
            continuity_block += (
                f"\nPREVIOUS SCENE (for visual continuity — start visually near where that ended):\n"
                f'"{prev_narration}"\n'
            )
        if next_narration:
            continuity_block += (
                f"\nNEXT SCENE (visually lead INTO this — end on a frame that transitions naturally):\n"
                f'"{next_narration}"\n'
            )

        medium = style.get("medium", "photorealistic cinematic film")
        prompt = (
            f"RENDER MEDIUM (every frame must look like this): {medium}\n"
            f"VISUAL STYLE DNA: {style_brief}\n"
            f"Color palette: {style['color_palette']}\n"
            f"Lighting: {style['lighting']}\n"
            f"Lens/medium: {style['lens']}\n"
            f"Textures: {style['texture']}\n"
            f"{char_block}"
            f"{continuity_block}\n"
            f"CURRENT NARRATION (visual must illustrate this PRECISELY):\n"
            f'"{scene.narration}"\n\n'
            f"DRAFT PROMPT to enhance:\n"
            f'"{scene.video_prompt}"\n\n'
            f"Rewrite as a 60-100 word Dextora prompt. Requirements:\n"
            f"- OPEN by naming the render medium above, so the style stays consistent\n"
            f"- ZERO vague mood words — describe ONLY what the camera sees\n"
            f"- Start the action with an explicit camera move; end with an ambient audio cue\n"
            f"- No real names of public figures (safety filter risk)\n"
            f"- Visual must PRECISELY match the current narration\n"
            f"- Camera movement must naturally BRIDGE from previous scene and LEAD INTO next\n"
            f"Output ONLY the rewritten prompt, nothing else."
        )

        response = self._next_client().models.generate_content(
            model=settings.GEMINI_SCRIPT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_build_enrichment_system(style),
                temperature=0.7,
                # Disable "thinking" so the model can't leak its reasoning preamble
                # ("THINKING PROCESS: ...") into the prompt text fed to Veo.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        enriched = self._sanitize_enriched(response.text, scene)
        log.info(
            "  [enrich] scene %d: %d->%d words",
            scene.scene_id,
            len(scene.video_prompt.split()),
            len(enriched.split()),
        )
        return enriched

    @staticmethod
    def _sanitize_enriched(text: str | None, scene: SceneScript) -> str:
        """Defend against a polluted enrichment response.

        Even with thinking disabled, the model can occasionally prepend a
        reasoning preamble or run far over length. A valid Veo prompt is ~60-100
        words; if the output is clearly broken we drop the preamble, and as a last
        resort fall back to the clean Phase 2 video_prompt."""
        import re

        t = (text or "").strip().strip('"').strip("'")

        # Drop a leaked reasoning preamble — keep the final paragraph (the real prompt).
        if re.match(r"(?i)^\s*(think|thinking process|reasoning|here'?s|okay|the user)\b", t):
            paras = [p.strip() for p in t.split("\n\n") if p.strip()]
            if paras:
                t = paras[-1]

        # Hard length guard: anything wildly over a prompt's length is polluted.
        if not t or len(t.split()) > 160:
            log.warning(
                "  [enrich] scene %d enriched prompt unusable (%d words) — using Phase 2 prompt",
                scene.scene_id, len(t.split()),
            )
            t = scene.video_prompt

        return t

    # ── visuals via Dextora & Dextora ───────────────────────────────────────

    def _generate_visuals(self, script: VideoScript) -> list[VisualAsset]:
        """Concurrent two-pass generation:
        Pass 1 — enrich prompt + generate Imagen keyframe for every veo scene (parallel).
        Pass 2 — generate Veo videos (parallel); blocked/failed scenes fall back to the
                  keyframe (Ken Burns in Phase 4) instead of crashing the run.

        Veo polling is I/O-bound, so a thread pool of VEO_CONCURRENCY workers turns a
        serial N×(render time) wait into roughly one clip's wall-time."""
        from concurrent.futures import ThreadPoolExecutor

        from pipeline.phase3_animator import AnimatorFactory

        workers = max(1, settings.VEO_CONCURRENCY)
        veo_scenes = [s for s in script.scenes if s.engine_type == "veo_cinematic"]
        veo_scene_ids = [s.scene_id for s in veo_scenes]
        scene_map = {s.scene_id: s for s in script.scenes}

        enriched_map: dict[int, str] = {}
        keyframe_map: dict[int, Path | None] = {}   # None for direct (text-to-video) scenes

        # ── Pass 1: enrich (always) + keyframe (skip for direct scenes) ──────
        def _prep(scene: SceneScript) -> tuple[int, str, Path | None]:
            idx = veo_scene_ids.index(scene.scene_id)
            prev_n = scene_map[veo_scene_ids[idx - 1]].narration if idx > 0 else ""
            next_n = scene_map[veo_scene_ids[idx + 1]].narration if idx + 1 < len(veo_scene_ids) else ""
            anchor = self._build_character_anchor(script, scene)
            try:
                enriched = self._enrich_video_prompt(scene, script.visual_style_brief, anchor, prev_n, next_n)
            except Exception as exc:
                # Enrichment (Gemini text) rate-limited/failed — fall back to the clean
                # Phase 2 video_prompt instead of crashing the whole job.
                log.warning("  [enrich] scene %d failed (%s) — using Phase 2 prompt", scene.scene_id, exc)
                enriched = scene.video_prompt
            if scene.veo_mode == "direct" and not settings.SKIP_VEO:
                return scene.scene_id, enriched, None   # text-to-video → no keyframe needed
            try:
                kf = self._gen_keyframe_image(scene, enriched, anchor)
            except Exception as exc:
                # Imagen blocked/failed — render a fallback slide so the scene is still
                # covered and the video completes (never crash the whole run on one scene).
                log.warning("  [visual/dextora] scene %d keyframe failed (%s) — fallback slide", scene.scene_id, exc)
                kf = self._make_fallback_slide(scene)
            return scene.scene_id, enriched, kf

        log.info("  [visual] Pass 1 – enrich+keyframe for %d scenes (%d workers)", len(veo_scenes), workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for sid, enriched, kf in ex.map(_prep, veo_scenes):
                enriched_map[sid] = enriched
                keyframe_map[sid] = kf

        # ── FAST MODE: skip Veo, every scene is its keyframe + Ken Burns ─────
        if settings.SKIP_VEO:
            log.info("  [visual] FAST MODE (SKIP_VEO) – %d scenes use keyframe + Ken Burns, no Dextora render", len(veo_scenes))
            veo_assets: dict[int, VisualAsset] = {}
            for scene in veo_scenes:
                sid = scene.scene_id
                kf = keyframe_map.get(sid) or self._make_fallback_slide(scene)
                # nonexistent path → Phase 4 Ken-Burns animates the keyframe
                veo_path = self._video_dir / f"scene_{sid}_veo.mp4"
                veo_assets[sid] = VisualAsset(scene_id=sid, raw_video_path=veo_path, keyframe_image_path=kf)
            anim_assets = {}
            anim_scenes = [s for s in script.scenes if s.engine_type == "code_animator"]
            for scene in anim_scenes:
                kf = self._make_fallback_slide(scene)
                veo_path = self._video_dir / f"scene_{scene.scene_id}_veo.mp4"
                anim_assets[scene.scene_id] = VisualAsset(
                    scene_id=scene.scene_id, raw_video_path=veo_path, keyframe_image_path=kf)
            assets: list[VisualAsset] = []
            for scene in script.scenes:
                assets.append(veo_assets.get(scene.scene_id) or anim_assets[scene.scene_id])
            return assets

        # ── Pass 2: Veo clips (parallel, with graceful fallback) ─────────────
        def _veo(scene: SceneScript) -> VisualAsset:
            sid = scene.scene_id
            idx = veo_scene_ids.index(sid)
            next_kf = keyframe_map.get(veo_scene_ids[idx + 1]) if idx + 1 < len(veo_scene_ids) else None
            kf = keyframe_map.get(sid)   # None → Veo text-to-video (direct mode)
            try:
                veo_path = self._gen_veo(scene, enriched_map[sid], kf, next_kf)
            except Exception as exc:
                log.warning(
                    "  [visual/dextora] scene %d (%s) failed (%s) — falling back to keyframe card",
                    sid, scene.veo_mode, exc,
                )
                # Direct scenes have no keyframe to Ken-Burns; make a text card so the
                # scene is still covered. Keyframe-mode scenes already have their slide.
                if kf is None:
                    kf = self._make_fallback_slide(scene)
                veo_path = self._video_dir / f"scene_{sid}_veo.mp4"  # nonexistent → Phase 4 fallback
            return VisualAsset(scene_id=sid, raw_video_path=veo_path, keyframe_image_path=kf)

        log.info("  [visual] Pass 2 – Dextora render for %d scenes (%d workers)", len(veo_scenes), workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            veo_assets = {a.scene_id: a for a in ex.map(_veo, veo_scenes)}

        # ── code_animator scenes (sequential — Playwright is not thread-safe) ─
        anim_assets: dict[int, VisualAsset] = {}
        anim_scenes = [s for s in script.scenes if s.engine_type == "code_animator"]
        if anim_scenes:
            animator = AnimatorFactory()
            for scene in anim_scenes:
                try:
                    anim_assets[scene.scene_id] = animator._gen_animation(scene)
                except Exception as exc:
                    # Animator failed (e.g. headless browser issue) — fall back to the
                    # Veo path so the scene still gets an on-topic visual, never a crash.
                    log.warning(
                        "  [animator] scene %d failed (%s) — falling back to Veo path",
                        scene.scene_id, exc,
                    )
                    anim_assets[scene.scene_id] = self._veo_fallback_asset(scene, script)

        # ── reassemble in original scene order ───────────────────────────────
        assets: list[VisualAsset] = []
        for scene in script.scenes:
            if scene.scene_id in veo_assets:
                assets.append(veo_assets[scene.scene_id])
            else:
                assets.append(anim_assets[scene.scene_id])
        return assets

    @staticmethod
    def _active_style() -> dict:
        """The active VISUAL_STYLE preset (falls back to dark_cinematic)."""
        return settings.VISUAL_STYLE_PRESETS.get(
            settings.VISUAL_STYLE, settings.VISUAL_STYLE_PRESETS["dark_cinematic"]
        )

    def _veo_fallback_asset(self, scene: SceneScript, script: VideoScript) -> VisualAsset:
        """Run one scene through the full Veo path (enrich → keyframe → Veo) with all
        the usual fallbacks. Used when the code_animator path fails for a scene."""
        anchor = self._build_character_anchor(script, scene)
        try:
            enriched = self._enrich_video_prompt(scene, script.visual_style_brief, anchor)
        except Exception as exc:
            log.warning("  [animator→veo] scene %d enrich failed (%s) — using Phase 2 prompt", scene.scene_id, exc)
            enriched = scene.video_prompt
        try:
            keyframe = self._gen_keyframe_image(scene, enriched, anchor)
        except Exception as exc:
            log.warning("  [animator→veo] scene %d keyframe failed (%s) — fallback slide", scene.scene_id, exc)
            keyframe = self._make_fallback_slide(scene)
        try:
            veo_path = self._gen_veo(scene, enriched, keyframe, None)
        except Exception as exc:
            log.warning("  [animator→veo] scene %d Veo failed (%s) — Ken Burns keyframe", scene.scene_id, exc)
            veo_path = self._video_dir / f"scene_{scene.scene_id}_veo.mp4"  # nonexistent → Phase 4 fallback
        return VisualAsset(scene_id=scene.scene_id, raw_video_path=veo_path, keyframe_image_path=keyframe)

    @retry(
        # Many keyframes fire at once across parallel jobs → Imagen rate-limits (429
        # ClientError). Retry hard with JITTERED backoff so the herd spreads out and
        # transient limits recover, instead of mass-falling-back to slides.
        # (A real safety block raises ImagenBlockedError, which is NOT retried.)
        stop=stop_after_attempt(8),
        wait=wait_random_exponential(multiplier=2, min=4, max=120),
        retry=retry_if_not_exception_type(ImagenBlockedError),
    )
    def _gen_keyframe_image(self, scene: SceneScript, enriched_prompt: str, character_anchor: str = "") -> Path:
        dest = self._video_dir / f"scene_{scene.scene_id}_keyframe.png"
        if dest.exists():
            log.info("  [visual/dextora] scene %d keyframe cached", scene.scene_id)
            return dest

        # Lead with the render medium so Imagen locks the style for this keyframe;
        # the keyframe is Veo's start image, so getting the style right here anchors
        # the whole clip.
        medium = self._active_style().get("medium", "")
        medium_line = f"RENDER STYLE: {medium}\n\n" if medium else ""
        body = (
            f"{character_anchor}\n\nSCENE VISUAL:\n{enriched_prompt}"
            if character_anchor else enriched_prompt
        )
        full_prompt = f"{medium_line}{body}"

        log.info("  [visual/dextora] scene %d generating keyframe…", scene.scene_id)
        response = self._next_client().models.generate_images(
            model=settings.GEMINI_IMAGE_MODEL,
            prompt=full_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
                output_mime_type="image/png",
            ),
        )
        if not response.generated_images:
            # Safety filter — deterministic, do not retry; caller makes a fallback slide.
            raise ImagenBlockedError(
                f"Imagen returned no images for scene {scene.scene_id} "
                f"(safety filter). Prompt: {full_prompt[:200]}"
            )
        dest.write_bytes(response.generated_images[0].image.image_bytes)
        return dest

    @staticmethod
    def _load_font(size: int, devanagari: bool = False):
        """Best-available TrueType font (Devanagari-capable when needed)."""
        from PIL import ImageFont
        latin = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        deva = [
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        ]
        for path in (deva + latin) if devanagari else latin:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _make_fallback_slide(self, scene: SceneScript) -> Path:
        """When Imagen blocks/fails, render the scene's NARRATION as a clean text card
        (not a blank gradient) so the scene always has meaningful, on-topic visual that
        matches the audio. Guarantees the video never shows 'audio but nothing on screen'."""
        from PIL import Image, ImageDraw

        dest = self._video_dir / f"scene_{scene.scene_id}_keyframe.png"
        W, H = 1920, 1080
        img = Image.new("RGB", (W, H), (10, 14, 28))
        try:
            draw = ImageDraw.Draw(img)
            for y in range(H):  # subtle vertical gradient
                shade = int(10 + 22 * (y / H))
                draw.line([(0, y), (W, y)], fill=(shade, shade + 4, shade + 14))
            draw.rectangle([60, 60, W - 60, H - 60], outline=(70, 90, 140), width=3)

            text = (scene.narration or "").strip()
            is_deva = any("ऀ" <= ch <= "ॿ" for ch in text)
            font = self._load_font(64, devanagari=is_deva)

            # word-wrap to fit within the inner margin
            max_w = W - 360
            words, lines, cur = text.split(), [], ""
            for w in words:
                trial = f"{cur} {w}".strip()
                if draw.textlength(trial, font=font) <= max_w:
                    cur = trial
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            lines = lines[:6] or [" "]

            line_h = int(font.size * 1.4)
            total_h = line_h * len(lines)
            y = (H - total_h) // 2
            for ln in lines:
                w = draw.textlength(ln, font=font)
                x = (W - w) // 2
                draw.text((x + 2, y + 2), ln, font=font, fill=(0, 0, 0))      # shadow
                draw.text((x, y), ln, font=font, fill=(235, 233, 245))         # text
                y += line_h
        except Exception as exc:
            # never let the last-resort slide raise — fall back to a plain solid card
            log.warning("  [visual] fallback slide render degraded (%s) — plain card", exc)
            img = Image.new("RGB", (W, H), (10, 14, 28))

        img.save(str(dest))
        log.info("  [visual/dextora] scene %d -> text-card fallback", scene.scene_id)
        return dest

    def _gen_veo(
        self,
        scene: SceneScript,
        enriched_prompt: str,
        keyframe_path: Path | None,
        next_keyframe_path: Path | None = None,
    ) -> Path:
        # keyframe_path is None for veo_mode="direct" scenes → Veo text-to-video.
        dest = self._video_dir / f"scene_{scene.scene_id}_veo.mp4"
        if dest.exists():
            log.info("  [visual/dextora] scene %d cached", scene.scene_id)
            return dest

        audio_secs = (scene.duration_ms or 8000) / 1000.0
        n_clips = max(1, math.ceil(audio_secs / 8))

        log.info(
            "  [visual/dextora] scene %d – %d clip(s) for %.1fs, prompt: %s…",
            scene.scene_id, n_clips, audio_secs, enriched_prompt[:120],
        )

        clips: list[Path] = []
        for i in range(n_clips):
            # Only the final clip of the scene uses last_frame to bridge into next scene
            use_last_frame = (i == n_clips - 1)
            clip = self._fetch_veo_clip(
                scene, enriched_prompt, keyframe_path, i,
                next_keyframe_path if use_last_frame else None,
            )
            clips.append(clip)

        if len(clips) == 1:
            clips[0].rename(dest)
        else:
            self._concat_clips(clips, dest)
            for c in clips:
                c.unlink(missing_ok=True)

        log.info("  [visual/dextora] scene %d -> %s", scene.scene_id, dest.name)
        return dest

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=10, max=60),
        retry=retry_if_not_exception_type(VeoBlockedError),
    )
    def _fetch_veo_clip(
        self,
        scene: SceneScript,
        prompt: str,
        keyframe_path: Path | None,
        clip_idx: int,
        next_keyframe_path: Path | None = None,
    ) -> Path:
        dest = self._video_dir / f"scene_{scene.scene_id}_clip{clip_idx}.mp4"
        if dest.exists():
            return dest

        # One key for this clip's whole lifecycle (generate → poll → download):
        # the operation handle is bound to the key/project that created it.
        client = self._next_client()

        # keyframe_path None → text-to-video (direct mode); else image-to-video.
        start_image = (
            types.Image(image_bytes=keyframe_path.read_bytes(), mime_type="image/png")
            if keyframe_path else None
        )

        # last_frame (first+last-frame interpolation) gives perfect scene-to-scene
        # handoff, BUT it is Veo's slow/expensive path (10x+ render time). It is
        # OFF by default; the xfade dissolve in Phase 4 + prompt-level continuity
        # already produce smooth transitions. Enable via VEO_LAST_FRAME=true only
        # if you need frame-exact handoffs and accept the render-time cost.
        last_frame = None
        if settings.VEO_LAST_FRAME and next_keyframe_path and next_keyframe_path.exists():
            last_frame = types.Image(
                image_bytes=next_keyframe_path.read_bytes(), mime_type="image/png"
            )

        # Style-aware negatives: each style declares exactly what to push Veo AWAY
        # from (e.g. Ghibli negates 3D/photoreal; academic 3D negates photoreal but
        # NOT "3D render"). Keeps the animation from drifting off the keyframe style.
        style = self._active_style()
        medium = style.get("medium", "")
        negatives = (
            "text overlay, watermark, logo, blurry, low quality, pixelated, "
            "compression artifacts, static frame, no motion, handheld shake, "
            "jump cut, duplicate frames, distorted proportions"
        )
        if style.get("negative_extra"):
            negatives += ", " + style["negative_extra"]

        # NOTE: generate_audio is NOT supported on AI Studio (Developer API) keys —
        # only on Vertex/Enterprise. We omit it; Veo's audio track is discarded in
        # Phase 4 (normalize_clip uses -an) since we supply our own narration + BGM.
        veo_config = types.GenerateVideosConfig(
            aspect_ratio="16:9",
            duration_seconds=8,
            number_of_videos=1,
            negative_prompt=negatives,
            last_frame=last_frame,  # None unless VEO_LAST_FRAME enabled
        )

        # Lead the Veo prompt with the render medium so the style holds.
        medium_prefix = f"{medium}. " if medium else ""
        if start_image is not None:
            # image-to-video: animate the keyframe, preserve its exact style
            veo_prompt = (
                f"{medium_prefix}Smooth camera movement animating this scene while keeping "
                f"the exact visual style of the source image. "
                f"Add subtle environmental motion (particles, haze, gentle drift). "
                + prompt
            )
        else:
            # direct text-to-video: no source frame — describe the living moment directly
            veo_prompt = (
                f"{medium_prefix}Cinematic footage with natural, dynamic motion true to the moment. "
                + prompt
            )

        t_submit = time.time()
        gen_kwargs = {"model": settings.VEO_MODEL, "prompt": veo_prompt, "config": veo_config}
        if start_image is not None:
            gen_kwargs["image"] = start_image   # omit entirely for text-to-video
        operation = client.models.generate_videos(**gen_kwargs)

        # Poll up to ~10 min, logging progress at INFO so a slow job is visible.
        max_polls = 120
        for attempt in range(max_polls):
            operation = client.operations.get(operation)
            if operation.done:
                break
            if attempt and attempt % 6 == 0:  # every ~30s
                log.info(
                    "  [dextora] scene %d clip %d still rendering (%.0fs elapsed)",
                    scene.scene_id, clip_idx, time.time() - t_submit,
                )
            time.sleep(5)
        else:
            raise TimeoutError(
                f"Veo timed out for scene {scene.scene_id} clip {clip_idx} "
                f"after {time.time() - t_submit:.0f}s"
            )

        if operation.error:
            raise RuntimeError(f"Veo error scene {scene.scene_id} clip {clip_idx}: {operation.error}")
        if not (operation.response and operation.response.generated_videos):
            resp = operation.response
            reasons = getattr(resp, "rai_media_filtered_reasons", None) if resp else None
            count = getattr(resp, "rai_media_filtered_count", None) if resp else None
            raise VeoBlockedError(
                f"Veo returned no video for scene {scene.scene_id} clip {clip_idx} "
                f"(safety filter). filtered_count={count} reasons={reasons}"
            )

        video_bytes = client.files.download(file=operation.response.generated_videos[0])
        dest.write_bytes(video_bytes)
        log.info(
            "  [dextora] scene %d clip %d done in %.0fs (%d bytes)",
            scene.scene_id, clip_idx, time.time() - t_submit, len(video_bytes),
        )
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
