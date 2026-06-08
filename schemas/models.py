from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Timezone-aware UTC now (datetime.utcnow() is deprecated in Python 3.12+)."""
    return datetime.now(timezone.utc)


# ─── Phase 1 ──────────────────────────────────────────────────────────────────

class TrendCandidate(BaseModel):
    video_id: str
    title: str
    channel_title: str
    views: int
    published_at: datetime
    days_since_upload: float
    view_velocity: float  # views / days_since_upload
    niche_keyword: str
    velocity_score: float = 0.0  # normalized 0-1


class TrendReport(BaseModel):
    scraped_at: datetime = Field(default_factory=_utcnow)
    candidates: list[TrendCandidate]
    top_concepts: list[TrendCandidate]


# ─── Phase 2 ──────────────────────────────────────────────────────────────────

class SceneScript(BaseModel):
    scene_id: int
    narration: str
    image_prompt: str
    video_prompt: str
    engine_type: Literal["veo_cinematic", "code_animator"] = "veo_cinematic"
    # For veo_cinematic scenes: how the clip is made.
    #   "keyframe" (default) → Imagen keyframe → Veo animates it (control + character
    #                          consistency; best for composed/character/static shots)
    #   "direct"             → Veo text-to-video, NO keyframe (free, living motion;
    #                          best for action/motion/event "hero" scenes)
    veo_mode: Literal["keyframe", "direct"] = "keyframe"
    # Exact formula / equation / key term to pin on screen as a guaranteed-accurate
    # graphic (e.g. "F = ma", "CO₂ + 6H₂O → C₆H₁₂O₆"). Empty when the scene has no
    # critical text. Burned as a crisp overlay in Phase 4 — never AI-rendered, so it
    # is always correct (fixes Imagen/Veo garbling formulas like F=ma → f=a).
    overlay_text: Optional[str] = None
    duration_ms: Optional[int] = None  # filled in Phase 3 after TTS


class VideoMetadata(BaseModel):
    title: str
    description: str
    tags: list[str]
    thumbnail_concept: str = ""


class CharacterRef(BaseModel):
    """A recurring character with a locked-down physical description
    so every scene generates the same person."""
    name: str
    physical_description: str  # age, build, skin tone, hair, clothing, distinguishing marks
    role: str = ""  # e.g. "protagonist", "antagonist", "narrator subject"
    scenes_present: list[int] = []  # scene_ids where this character appears
    reference_image_path: Optional[Path] = None  # filled in Phase 3


class SceneComposition(BaseModel):
    """Per-scene breakdown of which characters appear and what they're doing."""
    scene_id: int
    characters_present: list[str] = []  # character names from the cast
    composition_notes: str = ""  # e.g. "close-up on protagonist's face, tears welling"


class Storyboard(BaseModel):
    """Visual consistency anchor — generated once per video, referenced by every scene."""
    characters: list[CharacterRef] = []
    scene_compositions: list[SceneComposition] = []


class VideoScript(BaseModel):
    metadata: VideoMetadata
    scenes: list[SceneScript]
    visual_style_brief: str = ""  # cinematic DNA applied to all scenes
    storyboard: Optional[Storyboard] = None  # character cast + scene compositions
    trend_source: Optional[TrendCandidate] = None


# ─── Phase 3 ──────────────────────────────────────────────────────────────────

class AudioAsset(BaseModel):
    scene_id: int
    path: Path
    duration_ms: int


class VisualAsset(BaseModel):
    scene_id: int
    raw_video_path: Path
    keyframe_image_path: Optional[Path] = None


class AssetBundle(BaseModel):
    script: VideoScript
    audio_assets: list[AudioAsset]
    visual_assets: list[VisualAsset]
    bgm_path: Optional[Path] = None


# ─── Phase 4 ──────────────────────────────────────────────────────────────────

class RenderResult(BaseModel):
    final_video_path: Path
    duration_seconds: float
    subtitle_path: Optional[Path] = None


# ─── Phase 5 ──────────────────────────────────────────────────────────────────

class PublishResult(BaseModel):
    youtube_video_id: str
    youtube_url: str
    title: str
    published_at: datetime = Field(default_factory=_utcnow)


# ─── Pipeline state ───────────────────────────────────────────────────────────

class PipelineRun(BaseModel):
    run_id: str
    started_at: datetime = Field(default_factory=_utcnow)
    trend: Optional[TrendCandidate] = None
    script: Optional[VideoScript] = None
    assets: Optional[AssetBundle] = None
    render: Optional[RenderResult] = None
    publish: Optional[PublishResult] = None
    error: Optional[str] = None
    completed: bool = False
