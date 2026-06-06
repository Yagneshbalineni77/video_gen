"""
Phase 4 – Programmatic Video Editor
Produces a flawless final_render.mp4 with zero hard cuts between scenes.

Assembly order (matches reference channel style):
  1. Normalise all raw clips to 1920x1080 @ 25fps (silent)
  2. Trim each clip to exact narration duration
  3. Concatenate with 0.7s xfade dissolve transitions  → no gaps
  4. Concatenate all narration audio into one clean track
  5. Mux video + narration
  6. Sidechain-duck background music under narration
  7. Transcribe narration with Whisper → burn ASS subtitles
  8. Output: output/renders/final_render.mp4
"""
from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from schemas.models import AssetBundle, RenderResult
from utils.ffmpeg_helpers import (
    burn_subtitles,
    concat_audio,
    concat_video_xfade,
    image_to_video,
    merge_video_audio,
    mix_bgm,
    normalize_clip,
    probe_duration,
    trim_video_to_duration,
)
from utils.whisper_captions import transcribe_to_ass

log = logging.getLogger(__name__)


class VideoEditor:
    def __init__(self) -> None:
        self._render_dir = settings.OUTPUT_DIR / "renders"
        self._tmp = settings.OUTPUT_DIR / "renders" / "_tmp"
        self._render_dir.mkdir(parents=True, exist_ok=True)
        self._tmp.mkdir(parents=True, exist_ok=True)

    def run(self, bundle: AssetBundle) -> RenderResult:
        log.info("Phase 4 – assembling %d scenes", len(bundle.script.scenes))

        # Step 1 & 2: prepare one silent normalised clip per scene
        silent_clips = self._prepare_scene_clips(bundle)

        # Step 3: xfade concatenate all clips (no hard cuts)
        concat_silent = self._tmp / "concat_silent.mp4"
        concat_video_xfade(silent_clips, concat_silent, fade_dur=0.7)
        log.info("  [edit] xfade concat done")

        # Step 4: single continuous narration track
        narration_wav = self._tmp / "narration_full.wav"
        concat_audio([a.path for a in bundle.audio_assets], narration_wav)

        # Step 5: mux video + narration
        with_narration = self._tmp / "with_narration.mp4"
        merge_video_audio(concat_silent, narration_wav, with_narration)
        log.info("  [edit] narration merged")

        # Step 6: add background music
        if bundle.bgm_path:
            with_bgm = self._tmp / "with_bgm.mp4"
            mix_bgm(with_narration, bundle.bgm_path, with_bgm)
            log.info("  [edit] BGM mixed")
            pre_sub = with_bgm
        else:
            pre_sub = with_narration

        # Step 7: Whisper subtitles
        sub_path = self._tmp / "subtitles.ass"
        transcribe_to_ass(narration_wav, sub_path)

        # Step 8: burn subtitles → final render
        final = self._render_dir / "final_render.mp4"
        burn_subtitles(pre_sub, sub_path, final)
        log.info("  [edit] subtitles burned")

        duration = probe_duration(final)
        log.info("Phase 4 done - %.1fs -> %s", duration, final.name)
        return RenderResult(final_video_path=final, duration_seconds=duration, subtitle_path=sub_path)

    # ── private ──────────────────────────────────────────────────────────────

    def _prepare_scene_clips(self, bundle: AssetBundle) -> list[Path]:
        audio_map = {a.scene_id: a for a in bundle.audio_assets}
        visual_map = {v.scene_id: v for v in bundle.visual_assets}
        clips: list[Path] = []

        for scene in bundle.script.scenes:
            sid = scene.scene_id
            audio = audio_map[sid]
            visual = visual_map[sid]

            # Raw source → silent normalised MP4
            raw_norm = self._tmp / f"scene_{sid}_norm.mp4"
            trimmed = self._tmp / f"scene_{sid}_clip.mp4"

            if visual.raw_video_path and visual.raw_video_path.exists():
                normalize_clip(visual.raw_video_path, raw_norm)
                trim_video_to_duration(raw_norm, audio.duration_ms, trimmed)
            elif visual.keyframe_image_path and visual.keyframe_image_path.exists():
                image_to_video(visual.keyframe_image_path, audio.duration_ms, trimmed)
            else:
                raise FileNotFoundError(f"No visual asset for scene {sid}")

            log.info("  [edit] scene %d prepared (%.2fs)", sid, audio.duration_ms / 1000)
            clips.append(trimmed)

        return clips
