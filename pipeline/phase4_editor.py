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
    master_narration,
    merge_video_audio,
    mix_bgm,
    normalize_clip,
    overlay_image_on_video,
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

    FADE_DUR = 0.7  # xfade dissolve overlap between scenes (seconds)

    def run(self, bundle: AssetBundle) -> RenderResult:
        log.info("Phase 4 – assembling %d scenes", len(bundle.script.scenes))

        # Step 1 & 2: prepare one silent normalised clip per scene.
        # Each non-final clip is padded by FADE_DUR so the xfade overlap is absorbed
        # by the padding instead of shortening the narration-aligned timeline. This
        # keeps audio/video in sync across all scenes (otherwise N transitions drop
        # N×FADE_DUR seconds of video and -shortest truncates the narration tail).
        silent_clips = self._prepare_scene_clips(bundle, fade_dur=self.FADE_DUR)

        # Step 3: xfade concatenate all clips (no hard cuts)
        concat_silent = self._tmp / "concat_silent.mp4"
        concat_video_xfade(silent_clips, concat_silent, fade_dur=self.FADE_DUR)
        log.info("  [edit] xfade concat done")

        # Step 4: single continuous narration track, then master it to broadcast
        # loudness/clarity (also evens out level differences between scenes).
        narration_raw = self._tmp / "narration_raw.wav"
        concat_audio([a.path for a in bundle.audio_assets], narration_raw)
        narration_wav = self._tmp / "narration_full.wav"
        try:
            master_narration(narration_raw, narration_wav)
            log.info("  [edit] narration mastered (-16 LUFS, compressed)")
        except Exception as exc:
            log.warning("  [edit] mastering failed (%s) — using raw narration", exc)
            narration_wav = narration_raw

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

        # Step 7: subtitles. English → Whisper word-level (karaoke). Hindi/Hinglish →
        # build from the KNOWN script text (ASR is wrong-script/unreliable there).
        sub_path = self._tmp / "subtitles.ass"
        profile = settings.narration_profile()
        subs_ready = False
        try:
            if profile.get("use_asr", True):
                transcribe_to_ass(narration_wav, sub_path, language=profile["whisper_lang"])
            else:
                from utils.whisper_captions import subtitles_from_scenes
                dur_map = {a.scene_id: a.duration_ms for a in bundle.audio_assets}
                items = [(s.narration, dur_map.get(s.scene_id, s.duration_ms or 4000))
                         for s in bundle.script.scenes]
                subtitles_from_scenes(items, sub_path, font=profile.get("sub_font", "Arial"))
            subs_ready = sub_path.exists()
        except Exception as exc:
            log.warning("  [edit] subtitle generation failed (%s) — rendering without subtitles", exc)

        # Step 8: burn subtitles (+ brand watermark) → final render. Neither a subtitle
        # failure nor a burn failure may lose the finished video.
        burned = self._tmp / "burned.mp4"
        try:
            burn_subtitles(pre_sub, sub_path if subs_ready else None, burned,
                           watermark=settings.WATERMARK_TEXT)
            log.info("  [edit] %s burned",
                     "subtitles + watermark" if subs_ready else "watermark (subtitles skipped)")
        except Exception as exc:
            log.warning("  [edit] burn step failed (%s) — saving un-burned video so the render still completes", exc)
            import shutil
            shutil.copy2(pre_sub, burned)

        # Step 9: hard-trim final output to exact narration audio length.
        # This eliminates any video overrun that BGM mixing or xfade padding can introduce
        # (e.g. a 13-min video with audio content at only 4:33 gets trimmed to 4:33).
        final = self._render_dir / "final_render.mp4"
        narration_dur_s = probe_duration(narration_wav)
        burned_dur_s = probe_duration(burned)
        if burned_dur_s > narration_dur_s + 1.0:
            log.info("  [edit] trimming final from %.1fs to narration length %.1fs",
                     burned_dur_s, narration_dur_s)
            trim_video_to_duration(burned, int(narration_dur_s * 1000), final)
        else:
            import shutil
            shutil.move(str(burned), str(final))

        duration = probe_duration(final)
        log.info("Phase 4 done - %.1fs -> %s", duration, final.name)
        return RenderResult(final_video_path=final, duration_seconds=duration, subtitle_path=sub_path)

    # ── private ──────────────────────────────────────────────────────────────

    def _prepare_scene_clips(self, bundle: AssetBundle, fade_dur: float = 0.7) -> list[Path]:
        audio_map = {a.scene_id: a for a in bundle.audio_assets}
        visual_map = {v.scene_id: v for v in bundle.visual_assets}
        clips: list[Path] = []

        scenes = bundle.script.scenes
        fade_ms = int(fade_dur * 1000)

        for i, scene in enumerate(scenes):
            sid = scene.scene_id
            audio = audio_map.get(sid)
            visual = visual_map.get(sid)

            # Pad every clip except the last by the fade overlap so the xfade does
            # not consume narration-aligned time (keeps A/V in sync — see run()).
            is_last = (i == len(scenes) - 1)
            dur_ms = audio.duration_ms if audio else (scene.duration_ms or 4000)
            target_ms = dur_ms + (0 if is_last else fade_ms)

            # Raw source → silent normalised MP4. ANY failure here (corrupt clip,
            # missing asset, ffmpeg error) drops to an emergency text card so a single
            # bad scene can never crash the whole render.
            raw_norm = self._tmp / f"scene_{sid}_norm.mp4"
            trimmed = self._tmp / f"scene_{sid}_clip.mp4"
            try:
                if visual and visual.raw_video_path and visual.raw_video_path.exists():
                    normalize_clip(visual.raw_video_path, raw_norm)
                    trim_video_to_duration(raw_norm, target_ms, trimmed)
                elif visual and visual.keyframe_image_path and visual.keyframe_image_path.exists():
                    image_to_video(visual.keyframe_image_path, target_ms, trimmed)
                else:
                    raise FileNotFoundError(f"no visual asset for scene {sid}")
            except Exception as exc:
                log.warning("  [edit] scene %d clip prep failed (%s) — emergency text card", sid, exc)
                trimmed = self._emergency_clip(scene, target_ms)

            # Pin a guaranteed-accurate formula/term graphic over this scene if the
            # script declared one (model-independent → never garbled like F=ma->f=a).
            # An overlay failure must never drop the scene — skip it and keep the clip.
            if getattr(scene, "overlay_text", None):
                try:
                    ov_png = self._render_overlay(scene.overlay_text)
                    ov_clip = self._tmp / f"scene_{sid}_overlay.mp4"
                    overlay_image_on_video(trimmed, ov_png, ov_clip)
                    trimmed = ov_clip
                    log.info("  [edit] scene %d overlay: %s", sid, scene.overlay_text)
                except Exception as exc:
                    log.warning("  [edit] scene %d overlay failed (%s) — skipping overlay", sid, exc)

            log.info("  [edit] scene %d prepared (%.2fs)", sid, target_ms / 1000)
            clips.append(trimmed)

        return clips

    def _emergency_clip(self, scene, target_ms: int) -> Path:
        """Last-resort scene visual: the narration rendered as a dark text card, turned
        into a clip. Guarantees every scene yields a valid clip for the concat even if
        its real visual is missing or ffmpeg choked on the source."""
        from PIL import Image, ImageDraw

        png = self._tmp / f"scene_{scene.scene_id}_emergency.png"
        dest = self._tmp / f"scene_{scene.scene_id}_emergency.mp4"
        try:
            W, H = 1920, 1080
            img = Image.new("RGB", (W, H), (10, 14, 28))
            draw = ImageDraw.Draw(img)
            font = self._overlay_font(58)
            text = (getattr(scene, "narration", "") or "").strip()
            words, lines, cur = text.split(), [], ""
            for w in words:
                trial = f"{cur} {w}".strip()
                if draw.textlength(trial, font=font) <= W - 360:
                    cur = trial
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            lines = lines[:6] or [" "]
            lh = int(font.size * 1.4)
            y = (H - lh * len(lines)) // 2
            for ln in lines:
                x = (W - draw.textlength(ln, font=font)) // 2
                draw.text((x, y), ln, font=font, fill=(235, 233, 245))
                y += lh
            img.save(str(png))
        except Exception:
            # even text rendering failed → plain solid frame, still a valid visual
            Image.new("RGB", (1920, 1080), (10, 14, 28)).save(str(png))
        image_to_video(png, target_ms, dest)
        return dest

    @staticmethod
    def _overlay_font(size: int):
        from PIL import ImageFont
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _render_overlay(self, text: str) -> Path:
        """Render the exact formula/term as a crisp top-center graphic (RGBA PNG).
        Full Unicode (subscripts ₂, arrows →) via a symbol-capable font — always
        pixel-perfect because it's drawn from the script text, not AI-generated."""
        from PIL import Image, ImageDraw

        W, H = 1920, 1080
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # shrink font until the text fits comfortably across the frame
        size = 72
        font = self._overlay_font(size)
        while draw.textlength(text, font=font) > W - 520 and size > 30:
            size -= 4
            font = self._overlay_font(size)

        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad_x, pad_y = 56, 34
        panel_w, panel_h = tw + 2 * pad_x, th + 2 * pad_y
        x0 = (W - panel_w) // 2
        y0 = 70  # top band, clear of bottom subtitles

        # rounded translucent panel + accent border
        draw.rounded_rectangle([x0, y0, x0 + panel_w, y0 + panel_h], radius=22,
                               fill=(8, 12, 28, 205), outline=(79, 195, 247, 235), width=3)
        tx = x0 + pad_x - bb[0]
        ty = y0 + pad_y - bb[1]
        draw.text((tx + 2, ty + 2), text, font=font, fill=(0, 0, 0, 180))   # shadow
        draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))     # text

        dest = self._tmp / "_overlay.png"
        img.save(str(dest))
        return dest
