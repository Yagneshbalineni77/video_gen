"""
FFmpeg helpers – all subprocess wrappers.
Key upgrade over v1: concat_video_xfade uses xfade dissolve transitions
so there are ZERO hard cuts between scenes (matches reference channel style).
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# ── core runner ───────────────────────────────────────────────────────────────

def run_ffmpeg(args: list[str], desc: str = "", cwd: str | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    log.debug("[ffmpeg] %s", " ".join(str(a) for a in args[:6]))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed ({desc}):\n{result.stderr.strip()}")


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_entries", "format=duration", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return float(json.loads(result.stdout)["format"]["duration"])


# ── per-clip preparation ──────────────────────────────────────────────────────

def normalize_clip(src: Path, dest: Path, width: int = 1920, height: int = 1080, fps: int = 25) -> None:
    """Force every clip to the same resolution/fps before concatenation."""
    run_ffmpeg(
        [
            "-i", str(src),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",          # no audio – we keep narration separate
            str(dest),
        ],
        desc=f"normalize {src.name}",
    )


def trim_video_to_duration(src: Path, duration_ms: int, dest: Path) -> None:
    """Trim or loop a clip to an exact duration.

    Uses re-encode on loop so timestamps are monotonic (avoids the visible
    hard-cut every 8s that -c:v copy produces with stream_loop).
    """
    secs = duration_ms / 1000.0
    src_dur = probe_duration(src)
    if src_dur >= secs:
        # Simple trim – copy is fine, no looping involved
        run_ffmpeg(
            ["-i", str(src), "-t", f"{secs:.3f}", "-c:v", "copy", "-an", str(dest)],
            desc=f"trim {src.name}",
        )
    else:
        # Loop with re-encode so each cycle gets correct monotonic timestamps
        run_ffmpeg(
            [
                "-stream_loop", "-1",
                "-i", str(src),
                "-t", f"{secs:.3f}",
                "-vf", "fps=25",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-an",
                str(dest),
            ],
            desc=f"loop {src.name}",
        )


def image_to_video(image_path: Path, duration_ms: int, dest: Path) -> None:
    """Animate a static image with a slow Ken Burns zoom-in."""
    secs = duration_ms / 1000.0
    frames = int(secs * 25)
    vf = (
        f"zoompan=z='min(zoom+0.0006,1.15)':d={frames}:s=1920x1080,"
        "fps=25,scale=1920:1080"
    )
    run_ffmpeg(
        [
            "-loop", "1",
            "-i", str(image_path),
            "-vf", vf,
            "-t", f"{secs:.3f}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            str(dest),
        ],
        desc=f"image→video {dest.name}",
    )


# ── composite ────────────────────────────────────────────────────────────────

def composite_chromakey(bg: Path, fg: Path, dest: Path) -> None:
    """Composite a green-screen foreground video over a background video."""
    run_ffmpeg(
        [
            "-i", str(bg),
            "-i", str(fg),
            "-filter_complex", "[1:v]colorkey=0x00FF00:0.2:0.1[ckout];[0:v][ckout]overlay=shortest=1[outv]",
            "-map", "[outv]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            str(dest),
        ],
        desc=f"chromakey {fg.name} over {bg.name}",
    )

# ── seamless concatenation with xfade ────────────────────────────────────────

def concat_video_xfade(clips: list[Path], dest: Path, fade_dur: float = 0.7) -> None:
    """
    Concatenate silent video clips with smooth dissolve crossfades (no hard cuts).
    Each clip must already be normalised to the same resolution/fps.
    fade_dur: overlap duration in seconds (0.5–1.0 works well).
    """
    if len(clips) == 1:
        shutil.copy(clips[0], dest)
        return

    durations = [probe_duration(p) for p in clips]

    inputs: list[str] = []
    for p in clips:
        inputs += ["-i", str(p)]

    # Build chained xfade filter
    filters: list[str] = []
    offset = 0.0
    current = "[0:v]"

    for i in range(1, len(clips)):
        offset += durations[i - 1] - fade_dur
        out = f"[xf{i}]" if i < len(clips) - 1 else "[vout]"
        filters.append(
            f"{current}[{i}:v]xfade=transition=fade:duration={fade_dur:.2f}:offset={offset:.3f}{out}"
        )
        current = out

    # Cinematic color grade on the final composite:
    # slight contrast boost, mild desaturate, warm shadows (teal-orange look)
    grade_filter = (
        "eq=contrast=1.12:brightness=-0.03:saturation=0.88,"
        "curves=r='0/0 0.5/0.48 1/1':g='0/0 0.5/0.50 1/0.97':b='0/0 0.5/0.54 1/1',"
        "vignette=PI/5"
    )
    filters.append(f"[vout]{grade_filter}[vfinal]")

    run_ffmpeg(
        inputs + [
            "-filter_complex", ";".join(filters),
            "-map", "[vfinal]",
            "-c:v", "libx264", "-crf", "17", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(dest),
        ],
        desc="xfade concat",
    )


# ── audio helpers ─────────────────────────────────────────────────────────────

def concat_audio(audio_paths: list[Path], dest: Path) -> None:
    """Concatenate audio files into one continuous track (no fades – clean narration)."""
    list_file = dest.parent / "_audio_list.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in audio_paths),
        encoding="utf-8",
    )
    run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(list_file),
         "-ar", "24000", "-ac", "1", str(dest)],
        desc="concat narration",
    )
    list_file.unlink(missing_ok=True)


def merge_video_audio(video_path: Path, audio_path: Path, dest: Path) -> None:
    """Mux silent video with narration audio track, trim to shortest."""
    run_ffmpeg(
        [
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(dest),
        ],
        desc="merge video+audio",
    )


def mix_bgm(video_path: Path, bgm_path: Path, dest: Path, bgm_db: float = -18.0) -> None:
    """
    Mix background music at bgm_db volume under the narration.
    Uses sidechain compression so BGM ducks automatically when voice is loud.
    """
    run_ffmpeg(
        [
            "-i", str(video_path),
            "-stream_loop", "-1",
            "-i", str(bgm_path),
            "-filter_complex",
            (
                f"[1:a]volume={bgm_db}dB[bgm];"
                "[0:a][bgm]sidechaincompress=threshold=0.02:ratio=6:attack=10:release=300[aout]"
            ),
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(dest),
        ],
        desc="mix BGM",
    )


# ── subtitles ─────────────────────────────────────────────────────────────────

def burn_subtitles(video_path: Path, subtitle_path: Path, dest: Path) -> None:
    """Burn ASS/SRT subtitles into the video (center-screen, bold white text).

    Uses a relative sub path + cwd to avoid Windows drive-letter colon being
    misread as an FFmpeg filter option separator (e.g. ass='C:/...' → error).
    """
    sub_name = subtitle_path.name
    sub_cwd  = str(subtitle_path.parent.resolve())

    if subtitle_path.suffix.lower() == ".ass":
        vf = f"ass={sub_name}"
    else:
        vf = (
            f"subtitles={sub_name}"
            ":force_style='FontName=Arial,FontSize=22,Bold=1,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "Outline=3,Shadow=1,Alignment=2'"
        )
    run_ffmpeg(
        [
            "-i", str(video_path.resolve()),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "17", "-preset", "fast",
            "-c:a", "copy",
            str(dest.resolve()),
        ],
        desc="burn subtitles",
        cwd=sub_cwd,
    )


# ── thumbnail ─────────────────────────────────────────────────────────────────

def generate_thumbnail(concept: str, title: str, dest: Path) -> Path:
    """Generate a cinematic dark thumbnail with bold title text."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    img = Image.new("RGB", (1280, 720), color=(12, 12, 18))

    # Dark gradient from bottom
    overlay = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    for y in range(720):
        alpha = int(180 * (y / 720) ** 1.5)
        draw_ov.line([(0, y), (1280, y)], fill=(0, 0, 0, alpha))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))

    draw = ImageDraw.Draw(img)

    # Word-wrap title at 24 chars per line
    words = title.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= 24:
            cur = f"{cur} {w}".strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    try:
        font_lg = ImageFont.truetype("arialbd.ttf", 80)
        font_sm = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = font_lg

    total_h = len(lines) * 95
    y = (720 - total_h) // 2 + 60

    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font_lg)
        w = bb[2] - bb[0]
        x = (1280 - w) // 2
        # Shadow
        draw.text((x + 4, y + 4), line, font=font_lg, fill=(0, 0, 0, 180))
        # Yellow title
        draw.text((x, y), line, font=font_lg, fill=(255, 215, 0))
        y += 95

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dest), quality=95)
    log.info("Thumbnail → %s", dest.name)
    return dest
