"""
Custom content runner — Gemini TTS + Imagen 4 + Veo 3 pipeline.
Bypasses Phase 1+2 and feeds pre-written content directly into
the AI generation pipeline → video editor.

Usage:
  python run_custom.py               # full run: TTS + Imagen + Veo + edit
  python run_custom.py --phase 4     # run only Phase 4 (editor) from cached assets
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from schemas.models import (
    AssetBundle,
    AudioAsset,
    SceneScript,
    VideoMetadata,
    VideoScript,
    VisualAsset,
)

log = logging.getLogger(__name__)


# ── Thomson's Plum Pudding Model — pre-written script ────────────────────────

SCENES = [
    SceneScript(
        scene_id=1,
        narration=(
            "Ever wondered if an atom could be like a watermelon? It sounds strange, "
            "but that's exactly how one of the first models of the atom was described! "
            "So, what did a 19th-century physicist have in common with your favourite summer fruit?"
        ),
        image_prompt=(
            "Cinematic close-up of a ripe watermelon splitting open, revealing a glowing, "
            "ethereal atomic structure inside with tiny luminous particles suspended in "
            "translucent red flesh, dramatic volumetric lighting, shallow depth of field, "
            "teal-orange color grading, 16:9 photorealistic"
        ),
        video_prompt=(
            "Slow-motion dolly push-in on a watermelon sitting on a dark wooden table. "
            "The watermelon splits open cinematically, revealing a glowing atomic model inside — "
            "tiny blue electrons suspended in warm red translucent flesh. "
            "Volumetric god rays illuminate the interior. Shallow depth of field, "
            "anamorphic lens flare. Teal-orange palette. Ambient: crisp splitting sound, low hum."
        ),
    ),
    SceneScript(
        scene_id=2,
        narration=(
            "After discovering the electron in 1897, physicist J. J. Thomson had a big question to answer: "
            "If atoms contain tiny negative particles, but are neutral overall, what do they actually look like? "
            "In this video, we'll break down his revolutionary 1898 proposal – the very first model to look inside the atom."
        ),
        image_prompt=(
            "Dramatic portrait of a distinguished Victorian-era scientist in a dark laboratory, "
            "warm candlelight illuminating his face, surrounded by cathode ray tubes and glass apparatus, "
            "a faint glow of electrons visible, Rembrandt lighting, photorealistic, cinematic 16:9"
        ),
        video_prompt=(
            "Slow crane descending shot into a Victorian physics laboratory. A distinguished elderly scientist "
            "with a grey moustache and round spectacles studies a glowing cathode ray tube. "
            "Warm candlelight casts Rembrandt lighting on his contemplative face. "
            "The camera slowly pushes in on a tiny blue particle — an electron — floating from the tube. "
            "Shallow DOF, warm amber-teal palette. Ambient: electrical hum, crackling tube."
        ),
    ),
    SceneScript(
        scene_id=3,
        narration=(
            "Alright, let's dive into Thomson's model. He proposed three main things. "
            "First, the shape. Thomson imagined the atom as a perfect sphere, incredibly tiny, "
            "with a radius of about 10 to the power of minus 10 meters. Think of it as a uniform, solid ball."
        ),
        image_prompt=(
            "A perfectly smooth luminous sphere hovering in dark space, emitting soft golden light, "
            "surface has subtle translucent glow, tiny scale markers visible, "
            "macro photography style with extreme shallow DOF, cinematic 16:9"
        ),
        video_prompt=(
            "Steadicam orbit around a perfectly smooth, luminous golden sphere hovering in a dark void. "
            "The sphere glows with soft warm light from within, surface slightly translucent. "
            "Tiny scale markers fade in beside it showing its impossibly small size. "
            "Camera slowly orbits 90 degrees. Volumetric fog surrounds the sphere. "
            "Shallow DOF, anamorphic. Gold-navy palette. Ambient: deep resonant hum."
        ),
    ),
    SceneScript(
        scene_id=4,
        narration=(
            "Second, the positive charge. Unlike today's model with a central nucleus, "
            "Thomson suggested that this entire sphere was made of a positively charged substance. "
            "Imagine a cloud or a gel of uniform positive charge, filling up the entire atom."
        ),
        image_prompt=(
            "A translucent sphere filling with warm orange-red luminous substance from inside out, "
            "tiny plus symbols floating within, soft volumetric glow, dark background, "
            "macro lens with beautiful bokeh, cinematic 16:9"
        ),
        video_prompt=(
            "Slow dolly push-in on the golden sphere. A warm red-orange luminous substance begins "
            "filling it from the centre outward like a sunrise inside glass. "
            "Tiny glowing plus symbols drift gently within the substance. "
            "The sphere transforms from hollow to a solid, uniform, positively-charged gel. "
            "Rim lighting on edges, volumetric interior glow. Warm amber palette. "
            "Ambient: gentle rising tone, electrical crackle."
        ),
    ),
    SceneScript(
        scene_id=5,
        narration=(
            "Third, and this is the key part, the electrons. Thomson proposed that the negatively charged "
            "electrons he discovered were embedded within this sphere of positive charge. "
            "They weren't orbiting or flying around; they were stuck inside, like plums in a pudding "
            "or seeds in a watermelon."
        ),
        image_prompt=(
            "Close-up of a translucent red-orange sphere with small bright blue electron particles "
            "embedded inside at fixed positions, each with a subtle minus glow, "
            "macro photography with shallow DOF and lens flare, cinematic 16:9"
        ),
        video_prompt=(
            "Tracking shot following small bright blue electron particles as they drift toward "
            "the red-orange sphere and embed themselves one by one with subtle flash effects. "
            "Each electron settles into a fixed position, their blue glow contrasting the warm sphere. "
            "Camera pulls back to reveal the complete model — electrons suspended like seeds. "
            "Shallow DOF, anamorphic bokeh. Blue-orange palette. Ambient: soft chime on each embed."
        ),
    ),
    SceneScript(
        scene_id=6,
        narration=(
            "Why this arrangement? It was designed to create the most stable electrostatic setup. "
            "The repulsion between the negative electrons is balanced by their attraction to the positive sphere. "
            "Most importantly, the total negative charge of the electrons perfectly balanced the total "
            "positive charge of the sphere, making the atom electrically neutral as a whole."
        ),
        image_prompt=(
            "Artistic visualization of electrostatic balance — blue and red energy fields merging "
            "into perfect equilibrium, streams of light connecting particles, "
            "dark background with volumetric fog, cinematic 16:9"
        ),
        video_prompt=(
            "Slow parallax shot of the Thomson atom model rotating. "
            "Thin blue repulsion lines animate between electron pairs, pulsing outward. "
            "Warm orange attraction lines draw from electrons toward the positive sphere centre. "
            "The lines gradually reach equilibrium — equal intensity, perfect balance. "
            "Camera slowly tilts up as a serene glow envelops the balanced atom. "
            "Cold blue + warm orange dual palette. Ambient: harmonic drone resolving to calm."
        ),
    ),
    SceneScript(
        scene_id=7,
        narration=(
            "This is why the model got its famous nicknames: the Plum Pudding Model, "
            "the Raisin Pudding Model, or the Watermelon Model. "
            "The pudding or the red part of the watermelon represents the uniform positive charge, "
            "and the plums, raisins, or seeds represent the embedded electrons."
        ),
        image_prompt=(
            "Split frame: left side shows a traditional English plum pudding with visible plums, "
            "right side shows a Thomson atom model with electrons — visual parallel, "
            "warm cinematic lighting, shallow DOF, food photography meets science, 16:9"
        ),
        video_prompt=(
            "Slider move from left to right. Left frame: a richly textured English plum pudding "
            "in warm candlelight, dark plums visible inside golden-brown dough. "
            "Right frame dissolves to reveal a Thomson atom — same composition, "
            "blue electrons in warm red sphere. The camera slides between both, "
            "drawing the visual parallel. Then a watermelon slice fades in briefly as alternate metaphor. "
            "Warm golden lighting. Ambient: cozy hearth crackling, gentle string note."
        ),
    ),
    SceneScript(
        scene_id=8,
        narration=(
            "Now, for your exams. This is a classic topic in your Class 11 syllabus. "
            "You can expect a 2 or 3-mark question asking you to explain Thomson's model. "
            "Remember to mention the three key postulates: the positive sphere, "
            "the embedded electrons, and the overall neutrality."
        ),
        image_prompt=(
            "Close-up of hands writing on an exam paper under warm desk lamp light, "
            "the page shows atomic model diagrams and bullet points, "
            "shallow depth of field, warm amber tones, cinematic 16:9"
        ),
        video_prompt=(
            "Slow dolly push-in on a student's hands writing on an exam paper under a warm desk lamp. "
            "The paper shows neat diagrams of Thomson's model with labeled bullet points. "
            "Camera focuses on key words being underlined: 'positive sphere', 'embedded electrons', "
            "'neutral'. The pen moves confidently. Warm amber lamplight, shallow DOF with bokeh. "
            "Ambient: soft pen scratching, quiet room tone."
        ),
    ),
    SceneScript(
        scene_id=9,
        narration=(
            "So, let's recap Thomson's model in 30 seconds. "
            "One: An atom is a sphere of uniformly distributed positive charge. "
            "Two: Negatively charged electrons are embedded within this sphere. "
            "Three: The atom is electrically neutral as the positive and negative charges balance out. "
            "And four: Its popular nicknames are the Plum Pudding or Watermelon model."
        ),
        image_prompt=(
            "Four miniature glowing atomic models arranged in a row, each highlighting a different aspect "
            "— sphere, electrons, balance, watermelon metaphor — "
            "floating above a dark reflective surface, soft volumetric lighting, cinematic 16:9"
        ),
        video_prompt=(
            "Drone aerial push-in over a dark reflective surface where four miniature Thomson models "
            "are arranged in a line, each glowing. Camera descends and tracks across them one by one: "
            "first shows the golden sphere, second shows embedded blue electrons, "
            "third shows balanced charge equilibrium glow, fourth morphs into a tiny watermelon slice. "
            "Each model pulses as the narration mentions it. "
            "Dark navy background, rim lighting. Ambient: gentle chime per model."
        ),
    ),
    SceneScript(
        scene_id=10,
        narration=(
            "If this video helped you understand the plum pudding model, smash that like button! "
            "Have any questions? Ask away in the comments below. "
            "And make sure you subscribe for the next video on Rutherford's model, "
            "which completely changed the game!"
        ),
        image_prompt=(
            "Artistic end screen with glowing Thomson atom model fading into Rutherford's model teaser, "
            "subscribe button glowing, dark premium background with particle effects, cinematic 16:9"
        ),
        video_prompt=(
            "Static locked-off shot of the Thomson atom model slowly rotating in the centre of frame. "
            "It begins to destabilize — electrons trembling — teasing Rutherford's experiment. "
            "The model fades to a soft glow. A 'Subscribe' text with gentle pulse appears below. "
            "Floating particle effects drift across frame. "
            "Dark premium background, volumetric fog. Ambient: anticipatory low drone."
        ),
    ),
]

METADATA = VideoMetadata(
    title="Thomson's Plum Pudding Model – Class 11 Chemistry",
    description=(
        "Understanding J.J. Thomson's atomic model — the Plum Pudding Model explained "
        "with stunning cinematic visuals. Perfect for Class 11 Chemistry exam prep.\n\n"
        "[TIMESTAMPS]\n"
        "0:00 - Hook\n0:15 - Introduction\n0:30 - Thomson's 3 Postulates\n"
        "1:45 - The Plum Pudding Analogy\n2:30 - Exam Connect\n3:00 - Summary\n3:30 - Next Video\n\n"
        "#ThomsonModel #PlumPuddingModel #AtomicStructure #Class11Chemistry #JEE #NEET"
    ),
    tags=[
        "Thomson model", "plum pudding model", "atomic structure",
        "class 11 chemistry", "JEE prep", "NEET prep",
        "atom model", "electron discovery", "physics", "chemistry",
        "watermelon model", "Rutherford", "education", "science", "exam prep",
    ],
    thumbnail_concept=(
        "Split watermelon revealing glowing atomic structure inside, "
        "dramatic volumetric lighting, text overlay 'ATOM = WATERMELON?'"
    ),
)

SCRIPT = VideoScript(
    metadata=METADATA,
    scenes=SCENES,
    visual_style_brief=(
        "Cinematic dark-background visuals with warm amber-teal color grading, "
        "volumetric lighting, shallow depth of field, anamorphic lens aesthetic, "
        "photorealistic textures with subtle sci-fi glow on atomic elements."
    ),
)


# ── Gemini + Imagen + Veo Pipeline ───────────────────────────────────────────

class CustomPipeline:
    """Self-contained pipeline: Gemini TTS → Imagen 4 → Veo 3 → Editor."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._audio_dir = settings.OUTPUT_DIR / "audio"
        self._video_dir = settings.OUTPUT_DIR / "video"
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._video_dir.mkdir(parents=True, exist_ok=True)

    def run(self, script: VideoScript) -> AssetBundle:
        log.info("Custom pipeline — %d scenes", len(script.scenes))

        # 1. Gemini TTS → narration audio
        audio_assets = self._generate_audio(script)

        # 2. Annotate durations on script
        dur_map = {a.scene_id: a.duration_ms for a in audio_assets}
        for scene in script.scenes:
            scene.duration_ms = dur_map.get(scene.scene_id)

        # 3. Imagen 4 keyframes → Veo 3 animation
        visual_assets = self._generate_visuals(script)

        # 4. BGM
        bgm = self._pick_bgm()

        return AssetBundle(
            script=script,
            audio_assets=audio_assets,
            visual_assets=visual_assets,
            bgm_path=bgm,
        )

    # ── Audio: Gemini TTS ────────────────────────────────────────────────────

    def _generate_audio(self, script: VideoScript) -> list[AudioAsset]:
        assets: list[AudioAsset] = []
        for scene in script.scenes:
            wav_path = self._audio_dir / f"scene_{scene.scene_id}.wav"
            if wav_path.exists():
                log.info("  [audio] scene %d cached", scene.scene_id)
            else:
                self._tts(scene.narration, wav_path)
                log.info("  [audio] scene %d → %s", scene.scene_id, wav_path.name)

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
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=settings.GEMINI_TTS_VOICE
                        )
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
            return int(w.getnframes() / w.getframerate() * 1000)

    # ── Visuals: Imagen 4 keyframe → Veo 3 animation ────────────────────────

    def _generate_visuals(self, script: VideoScript) -> list[VisualAsset]:
        assets: list[VisualAsset] = []
        for scene in script.scenes:
            # 1. Enrich the prompt via Gemini
            enriched = self._enrich_prompt(scene, script.visual_style_brief)
            # 2. Imagen 4 → keyframe image
            keyframe = self._gen_keyframe(scene, enriched)
            # 3. Veo 3 → animate the keyframe
            video = self._gen_veo(scene, enriched, keyframe)

            assets.append(VisualAsset(
                scene_id=scene.scene_id,
                raw_video_path=video,
                keyframe_image_path=keyframe,
            ))
        return assets

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _enrich_prompt(self, scene: SceneScript, style_brief: str) -> str:
        """Gemini enriches the video prompt into a Veo-optimized cinematic description."""
        style = settings.VISUAL_STYLE_PRESETS.get(
            settings.VISUAL_STYLE,
            settings.VISUAL_STYLE_PRESETS["dark_cinematic"],
        )
        prompt = (
            f"VISUAL STYLE DNA: {style_brief}\n"
            f"Color palette: {style['color_palette']}\n"
            f"Lighting: {style['lighting']}\n"
            f"Lens: {style['lens']}\n"
            f"Textures: {style['texture']}\n\n"
            f"NARRATION:\n\"{scene.narration}\"\n\n"
            f"DRAFT VIDEO PROMPT:\n\"{scene.video_prompt}\"\n\n"
            f"Rewrite this into a 60-100 word hyper-detailed Veo 3.0 prompt. "
            f"Start with camera movement, describe tangible physical subjects, "
            f"name the lighting technique, specify lens/DOF, state color palette, "
            f"end with ambient audio cue. Output ONLY the rewritten prompt."
        )
        response = self._client.models.generate_content(
            model=settings.GEMINI_SCRIPT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7),
        )
        enriched = response.text.strip().strip('"').strip("'")
        log.info("  [enrich] scene %d: %d→%d words",
                 scene.scene_id, len(scene.video_prompt.split()), len(enriched.split()))
        return enriched

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=60))
    def _gen_keyframe(self, scene: SceneScript, enriched_prompt: str) -> Path:
        """Imagen 4 → photorealistic keyframe image."""
        dest = self._video_dir / f"scene_{scene.scene_id}_keyframe.png"
        if dest.exists():
            log.info("  [imagen] scene %d keyframe cached", scene.scene_id)
            return dest

        log.info("  [imagen] scene %d generating keyframe…", scene.scene_id)
        response = self._client.models.generate_images(
            model="imagen-4.0-generate-001",
            prompt=enriched_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
                output_mime_type="image/png",
            ),
        )
        dest.write_bytes(response.generated_images[0].image.image_bytes)
        log.info("  [imagen] scene %d → %s", scene.scene_id, dest.name)
        return dest

    def _gen_veo(self, scene: SceneScript, enriched_prompt: str, keyframe: Path) -> Path:
        """Veo 3 → animate the keyframe into a video clip."""
        dest = self._video_dir / f"scene_{scene.scene_id}_veo.mp4"
        if dest.exists():
            log.info("  [veo] scene %d cached", scene.scene_id)
            return dest

        audio_secs = (scene.duration_ms or 8000) / 1000.0
        n_clips = max(1, math.ceil(audio_secs / 8))
        n_request = min(n_clips, 4)

        log.info("  [veo] scene %d — %d clip(s) for %.1fs", scene.scene_id, n_request, audio_secs)

        clips = [self._fetch_veo_clip(scene, enriched_prompt, keyframe, i) for i in range(n_request)]

        if len(clips) == 1:
            clips[0].rename(dest)
        else:
            self._concat_clips(clips, dest)
            for c in clips:
                c.unlink(missing_ok=True)

        return dest

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=60))
    def _fetch_veo_clip(self, scene: SceneScript, prompt: str, keyframe: Path, idx: int) -> Path:
        dest = self._video_dir / f"scene_{scene.scene_id}_clip{idx}.mp4"
        if dest.exists():
            return dest

        image_obj = types.Image(image_bytes=keyframe.read_bytes(), mime_type="image/png")
        source = types.GenerateVideosSource(
            image=image_obj,
            prompt=(
                "Cinematically animate this scene with perfectly smooth, highly dynamic "
                "camera movements. Add subtle environmental movement. " + prompt
            ),
        )
        operation = self._client.models.generate_videos(
            model=settings.VEO_MODEL,
            source=source,
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                duration_seconds=8,
                number_of_videos=1,
            ),
        )

        for attempt in range(90):
            operation = self._client.operations.get(operation)
            if operation.done:
                break
            log.debug("  [veo] scene %d clip %d polling (%ds)", scene.scene_id, idx, (attempt + 1) * 5)
            time.sleep(5)
        else:
            raise TimeoutError(f"Veo timed out for scene {scene.scene_id} clip {idx}")

        if operation.error:
            raise RuntimeError(f"Veo error scene {scene.scene_id}: {operation.error}")
        if not operation.response or not operation.response.generated_videos:
            raise RuntimeError(f"Veo returned empty for scene {scene.scene_id}")

        video_bytes = self._client.files.download(file=operation.response.generated_videos[0])
        dest.write_bytes(video_bytes)
        log.info("  [veo] scene %d clip %d → %s", scene.scene_id, idx, dest.name)
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
            desc=f"concat {len(clips)} clips",
        )
        list_file.unlink(missing_ok=True)

    def _pick_bgm(self) -> Path | None:
        if not settings.BGM_DIR.exists():
            return None
        tracks = list(settings.BGM_DIR.glob("*.mp3")) + list(settings.BGM_DIR.glob("*.wav"))
        if not tracks:
            return None
        import random
        return random.choice(tracks)


# ── Runner ───────────────────────────────────────────────────────────────────

def run(skip_phase3: bool = False) -> None:
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save script
    script_out = settings.OUTPUT_DIR / "script.json"
    script_out.write_text(SCRIPT.model_dump_json(indent=2), encoding="utf-8")
    log.info("Script saved → %s", script_out)

    if not skip_phase3:
        pipeline = CustomPipeline()
        bundle = pipeline.run(SCRIPT)
    else:
        # Reconstruct from cache
        audio_dir = settings.OUTPUT_DIR / "audio"
        video_dir = settings.OUTPUT_DIR / "video"

        audio_assets = []
        for s in SCRIPT.scenes:
            p = audio_dir / f"scene_{s.scene_id}.wav"
            dur = 8000
            if p.exists():
                with wave.open(str(p), "rb") as w:
                    dur = int(w.getnframes() / w.getframerate() * 1000)
            audio_assets.append(AudioAsset(scene_id=s.scene_id, path=p, duration_ms=dur))

        visual_assets = [
            VisualAsset(
                scene_id=s.scene_id,
                raw_video_path=video_dir / f"scene_{s.scene_id}_veo.mp4",
                keyframe_image_path=video_dir / f"scene_{s.scene_id}_keyframe.png",
            )
            for s in SCRIPT.scenes
        ]

        bgm = None
        if settings.BGM_DIR.exists():
            tracks = list(settings.BGM_DIR.glob("*.mp3")) + list(settings.BGM_DIR.glob("*.wav"))
            if tracks:
                import random
                bgm = random.choice(tracks)

        bundle = AssetBundle(
            script=SCRIPT,
            audio_assets=audio_assets,
            visual_assets=visual_assets,
            bgm_path=bgm,
        )

    # Phase 4 — Video Editor
    from pipeline.phase4_editor import VideoEditor
    result = VideoEditor().run(bundle)
    log.info("✅ Final render: %s (%.1fs)", result.final_video_path, result.duration_seconds)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for lib in ("httpx", "httpcore", "urllib3", "googleapiclient"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Custom content → Gemini + Imagen + Veo pipeline")
    parser.add_argument("--phase", type=int, choices=[4], help="Run only Phase 4 from cached assets")
    args = parser.parse_args()

    run(skip_phase3=(args.phase == 4))


if __name__ == "__main__":
    main()
