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


def transcribe_to_ass(audio_path: Path, dest: Path, landscape: bool = True, language: str = "en") -> Path:
    """
    Transcribe audio with faster-whisper, chunk into caption groups,
    and write an ASS file with karaoke-style word highlighting.

    language: ASR language so subtitles match the narration (e.g. "hi" for Hindi
    gives Devanagari captions; "en" for English/Hinglish).
    """
    from faster_whisper import WhisperModel

    log.info("[whisper] loading model…")
    model = WhisperModel("base", device="cpu", compute_type="int8")

    log.info("[whisper] transcribing %s (lang=%s)", audio_path.name, language)
    segments, _info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        beam_size=5,
        language=language,
    )

    chunks = _build_chunks(segments)
    # Devanagari (Hindi) needs a script-capable font or it renders as tofu boxes.
    font = "Noto Sans Devanagari" if language == "hi" else "Arial"
    ass_content = _render_ass(chunks, landscape=landscape, font=font)
    dest.write_text(ass_content, encoding="utf-8")
    log.info("[whisper] subtitles -> %s (%d chunks)", dest.name, len(chunks))
    return dest


def subtitles_from_scenes(items: list[tuple[str, int]], dest: Path,
                          landscape: bool = True, font: str = "Arial") -> Path:
    """Build subtitles from the KNOWN narration text + per-scene durations.

    Used for languages where ASR is unreliable (Hindi/Hinglish): we already have
    the exact script, so transcribing is both lossy and wrong-script. Each scene's
    narration is split into short chunks spread evenly across that scene's audio
    window, giving accurate captions in the correct script for any language.

    items: list of (narration_text, duration_ms) in scene order.
    """
    chunks: list[CaptionChunk] = []
    t = 0.0
    for text, dur_ms in items:
        dur = max(0.1, dur_ms / 1000.0)
        words = (text or "").split()
        groups = [words[i:i + MAX_CHUNK_WORDS] for i in range(0, len(words), MAX_CHUNK_WORDS)] or [[""]]
        per = dur / len(groups)
        for gi, g in enumerate(groups):
            start = t + gi * per
            chunks.append(CaptionChunk(words=g, start=start, end=start + per))
        t += dur
    dest.write_text(_render_ass(chunks, landscape=landscape, font=font), encoding="utf-8")
    log.info("[subs] built from script text -> %s (%d chunks, font=%s)", dest.name, len(chunks), font)
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


def _render_ass(chunks: list[CaptionChunk], landscape: bool = True, font: str = "Arial") -> str:
    """
    Clean educational subtitle style:
    - Bold white text, large and legible
    - Semi-transparent dark pill background for contrast on any scene
    - Bottom-center position with generous margin
    - Outline + shadow so text reads on both bright and dark frames
    """
    if landscape:
        res_x, res_y = 1920, 1080
        font_size = 58
        margin_v = 90
    else:
        res_x, res_y = 1080, 1920
        font_size = 64
        margin_v = 220

    # Colours in ASS &HAABBGGRR format:
    #   Primary  = opaque white    (&H00FFFFFF)
    #   Outline  = near-black      (&H00111111)
    #   Back     = 65% opaque dark (&HA6000000) — pill background
    #   Shadow   = pure black      (&H00000000)
    # Latin scripts get a touch of letter-spacing for that clean look; complex
    # scripts (Devanagari) need 0 so conjunct ligatures aren't broken apart.
    spacing = 0 if "Devanagari" in font else 1.2

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H00111111,&HA6000000,-1,0,0,0,100,100,{spacing},0,4,3,1,2,30,30,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []
    for chunk in chunks:
        # Scale text to 110% on display for the pop-in effect (no animation tag needed,
        # but we use \blur0 to ensure crisp rendering on re-encode)
        text = r"{\blur0}" + chunk.text()
        lines.append(
            f"Dialogue: 0,{_ts(chunk.start)},{_ts(chunk.end)},Default,,0,0,0,,{text}"
        )

    return header + "\n".join(lines) + "\n"
