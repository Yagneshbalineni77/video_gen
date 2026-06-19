"""
Portal script → VideoScript ingestion (the curriculum-bridge path; NO Phase 2).

The Dextora education portal authors complete scripts via its curriculum API:
  - audio_script      : continuous, TTS-ready narration (the exact spoken words)
  - visual_directions : timestamped on-screen cues ("0:00-0:05 → ...")
  - title, metadata (hook/hashtags), duration_seconds, language

This module turns ONE portal script item into our VideoScript so Phase 3/4 can
render it WITHOUT Phase 2 — the portal IS the script author. The narration is kept
verbatim (split across scenes on sentence boundaries); each visual-direction segment
becomes a scene's visual prompt, which Phase 3 then enriches into a cinematic Veo
prompt. If a script has no parsable visual directions, we fall back to narration-only
scenes (Phase 3 enriches the narration itself into visuals).
"""
from __future__ import annotations

import json
import re

from schemas.models import SceneScript, VideoMetadata, VideoScript

# "M:SS-M:SS → desc"  (also tolerates  -, –, ->, :  separators)
_TS_LINE = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})\s*(?:→|->|:|–|-)?\s*(.*)$"
)

# Style brief per subject so the visuals fit the topic; falls back to educational.
SUBJECT_STYLE_BRIEF: dict[str, str] = {
    "chemistry": "clean modern 3D educational visuals, molecular models, crisp labelled diagrams, vibrant and accurate",
    "physics": "clean modern 3D educational visuals, physics diagrams, motion and force vectors, vibrant and accurate",
    "biology": "clean modern 3D educational visuals, anatomical and cellular detail, vibrant and accurate",
    "mathematics": "clean modern educational visuals, geometric figures and graphs, crisp labelled diagrams",
    "history": "warm cinematic 35mm documentary look, authentic period detail, evocative and grounded",
    "english": "soft hand-painted illustrative storytelling visuals, warm and evocative",
    "economics": "clean modern infographic visuals, charts and flow diagrams, professional",
}
DEFAULT_STYLE_BRIEF = "clean modern educational visuals, crisp labelled diagrams, vibrant and clear"


def _extract_plain_narration(raw: str) -> str:
    """If audio_script is JSON-structured (portal sends sections/dialogue), extract text.
    Otherwise return the string as-is."""
    raw = (raw or "").strip()
    if not raw.startswith("{") and not raw.startswith("["):
        return raw
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw

    parts: list[str] = []

    def _walk(node):
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            # Prefer known dialogue fields; skip label/timestamp keys
            for key in ("dialogue", "narration", "text", "content", "script"):
                if key in node:
                    _walk(node[key])
                    return
            # Fallback: walk all string values except structural keys
            for k, v in node.items():
                if k not in ("label", "timestamp", "start", "end", "type", "id"):
                    _walk(v)

    _walk(data)
    return " ".join(parts)


def _sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _parse_visual_segments(visual_directions: str) -> list[tuple[int, int, str]]:
    """Parse the timestamped visual directions into [(start_s, end_s, description)]."""
    segments: list[tuple[int, int, str]] = []
    for raw in (visual_directions or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _TS_LINE.match(line)
        if m:
            sm, ss, em, es, desc = m.groups()
            start, end = int(sm) * 60 + int(ss), int(em) * 60 + int(es)
            segments.append((start, end, desc.strip()))
        elif segments:
            # a wrapped continuation line of the previous segment's description
            s, e, d = segments[-1]
            segments[-1] = (s, e, (d + " " + line).strip())
    return [(s, e, d) for (s, e, d) in segments if d]


def _group(items: list, n: int) -> list[list]:
    """Split a list into n contiguous, near-even groups (n clamped to len)."""
    n = max(1, min(n, len(items)))
    out, per, rem, i = [], len(items) // n, len(items) % n, 0
    for g in range(n):
        size = per + (1 if g < rem else 0)
        out.append(items[i:i + size])
        i += size
    return out


def _split_narration(sentences: list[str], weights: list[float]) -> list[str]:
    """Distribute whole sentences across len(weights) scenes, proportional to each
    scene's visual-segment duration (so longer beats get more narration)."""
    n = len(weights)
    if n <= 1 or len(sentences) <= 1:
        return [" ".join(sentences)]
    total_w = sum(weights) or float(n)
    total_words = sum(len(s.split()) for s in sentences) or 1
    targets = [total_words * (w / total_w) for w in weights]
    groups: list[list[str]] = [[] for _ in range(n)]
    counts = [0.0] * n
    gi = 0
    for sent in sentences:
        # advance to the next scene once the current one has met its word target
        while gi < n - 1 and counts[gi] >= targets[gi]:
            gi += 1
        groups[gi].append(sent)
        counts[gi] += len(sent.split())
    return [" ".join(g).strip() for g in groups]


_WORDS_PER_SCENE = 40   # target ~16s of speech per scene at 150 wpm
_MAX_SCENES = 20        # Veo concurrency cap


def _ideal_scene_count(total_words: int, n_segments: int) -> int:
    """Return a scene count that keeps each scene to ~_WORDS_PER_SCENE.

    If visual_directions are present (n_segments>0) we use them as the
    primary split but expand toward the word-count ideal when they'd create
    scenes that are too long. Capped at _MAX_SCENES."""
    word_based = max(3, (total_words + _WORDS_PER_SCENE - 1) // _WORDS_PER_SCENE)
    if n_segments > 0:
        # honour the content author's visual beats but allow expansion
        n = max(n_segments, min(word_based, _MAX_SCENES))
    else:
        n = min(word_based, _MAX_SCENES)
    return n


def build_script_from_portal(item: dict, *, subject_name: str = "") -> VideoScript:
    """Convert a portal script item → VideoScript (verbatim narration, no Phase 2)."""
    title = (item.get("title") or "Untitled").strip()
    raw_script = item.get("audio_script") or item.get("script_content") or ""
    narration_src = _extract_plain_narration(raw_script)
    sentences = _sentences(narration_src)
    if not sentences:
        sentences = [title]

    total_words = sum(len(s.split()) for s in sentences)
    segments = _parse_visual_segments(item.get("visual_directions") or "")

    if segments:
        n = _ideal_scene_count(total_words, len(segments))
        # never create more scenes than sentences (avoids empty/silent scenes)
        n = min(n, len(sentences))
        grouped = _group(segments, n)
        descs = [" ".join(d for _, _, d in g).strip() for g in grouped]
        weights = [max(1, sum((e - s) for s, e, _ in g)) for g in grouped]
        narration = _split_narration(sentences, weights)
    else:
        # no visual directions → scale by word count, target ~16s per scene
        n = _ideal_scene_count(total_words, 0)
        n = min(n, len(sentences))
        grouped_sents = _group(sentences, n)
        descs = [""] * n
        narration = [" ".join(g).strip() for g in grouped_sents]

    scenes: list[SceneScript] = []
    for i in range(n):
        narr = narration[i] if i < len(narration) and narration[i] else (descs[i] or title)
        visual = descs[i] or narr  # Phase 3 enriches this into a cinematic Veo prompt
        scenes.append(SceneScript(
            scene_id=i + 1,
            narration=narr,
            image_prompt=visual,
            video_prompt=visual,
            engine_type="veo_cinematic",
            veo_mode="keyframe",
        ))

    meta_in = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    metadata = VideoMetadata(
        title=title,
        description=str(meta_in.get("hook", "") or "")[:500],
        tags=[str(t) for t in (meta_in.get("hashtags") or [])][:15],
        thumbnail_concept=str(meta_in.get("thumbnail_idea", "") or "")[:300],
    )
    brief = SUBJECT_STYLE_BRIEF.get((subject_name or "").strip().lower(), DEFAULT_STYLE_BRIEF)
    return VideoScript(metadata=metadata, scenes=scenes, visual_style_brief=brief)
