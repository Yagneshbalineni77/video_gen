"""
Whisper caption generator – produces styled ASS subtitle files.
Uses faster-whisper for word-level timestamps.
Pattern adapted from numbpill3d/ffmpeg-ai captions.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

PAUSE_THRESHOLD = 0.2  # seconds between words to force a new caption chunk
MAX_CHUNK_WORDS = 3


@dataclass
class CaptionChunk:
    words: list[str] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0

    def text(self) -> str:
        return " ".join(self.words)


def transcribe_to_ass(audio_path: Path, dest: Path, landscape: bool = True) -> Path:
    """
    Transcribe audio with faster-whisper, chunk into caption groups,
    and write an ASS file with karaoke-style word highlighting.
    """
    from faster_whisper import WhisperModel

    log.info("[whisper] loading model…")
    model = WhisperModel("base", device="cpu", compute_type="int8")

    log.info("[whisper] transcribing %s", audio_path.name)
    segments, _info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        beam_size=5,
        language="en",
    )

    chunks = _build_chunks(segments)
    ass_content = _render_ass(chunks, landscape=landscape)
    dest.write_text(ass_content, encoding="utf-8")
    log.info("[whisper] subtitles -> %s (%d chunks)", dest.name, len(chunks))
    return dest


# ── internal helpers ──────────────────────────────────────────────────────────

def _build_chunks(segments) -> list[CaptionChunk]:
    chunks: list[CaptionChunk] = []
    current = CaptionChunk()
    prev_end = 0.0

    for segment in segments:
        for word_info in (segment.words or []):
            word = word_info.word.strip()
            if not word:
                continue

            gap = word_info.start - prev_end
            full = len(current.words) >= MAX_CHUNK_WORDS
            pause = gap > PAUSE_THRESHOLD and current.words

            if full or pause:
                if current.words:
                    chunks.append(current)
                current = CaptionChunk(
                    words=[word],
                    start=word_info.start,
                    end=word_info.end,
                )
            else:
                if not current.words:
                    current.start = word_info.start
                current.words.append(word)
                current.end = word_info.end

            prev_end = word_info.end

    if current.words:
        chunks.append(current)

    return chunks


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _render_ass(chunks: list[CaptionChunk], landscape: bool = True) -> str:
    if landscape:
        res_x, res_y = 1920, 1080
        font_size = 52
        margin_v = 80
    else:
        res_x, res_y = 1080, 1920
        font_size = 60
        margin_v = 200

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,{font_size},&H0000E1FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []
    for chunk in chunks:
        text = chunk.text()
        lines.append(
            f"Dialogue: 0,{_ts(chunk.start)},{_ts(chunk.end)},Default,,0,0,0,,{text}"
        )

    return header + "\n".join(lines) + "\n"
