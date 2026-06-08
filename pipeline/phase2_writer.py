"""
Phase 2 – Script & Prompt Generator
Uses Gemini 2.5 Flash with structured JSON output to produce a fully
parameterized video script with per-scene narration, image, and video prompts.

v2: Cinematic shot-list approach — video prompts are 40-100 word directorial
    descriptions with specific camera, lighting, lens, color, and audio cues.
"""
from __future__ import annotations

import json
import logging

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from schemas.models import (
    CharacterRef,
    SceneComposition,
    SceneScript,
    Storyboard,
    TrendCandidate,
    VideoMetadata,
    VideoScript,
)

log = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    """Build system prompt dynamically using the active visual style preset."""
    style = settings.VISUAL_STYLE_PRESETS.get(
        settings.VISUAL_STYLE,
        settings.VISUAL_STYLE_PRESETS["dark_cinematic"],
    )
    script_lang = settings.narration_profile()["script_lang"]

    return f"""You are an expert faceless YouTube content creator AND a veteran cinematographer.
You specialize in high-retention educational/historical/mysterious content with
PREMIUM cinematic visuals that rival Netflix documentaries.

NARRATIVE RULES:
  - Hook that triggers curiosity in the first 5 seconds
  - Constant tension and curiosity gaps ("but that's not all…", "what no one tells you…")
  - Short punchy sentences (≤15 words per sentence)
  - Present-tense narration
  - Zero filler words
  - narration per scene: strict MAXIMUM of 15 words (< 8 seconds at natural pace) to prevent video looping
  - VERY IMPORTANT: The `narration` text MUST be written in: {script_lang}.
    (Only the narration uses this language — keep image_prompt, video_prompt, tags,
     and all other fields in English so the image/video models understand them.)

TONE: Strictly academic, highly literal, scientific, textbook-style informative.
      This content is for a real educational company. Every claim must be factually accurate.
      Every visual must be a PRECISE literal illustration of what is being said — zero abstract metaphors.
      Students are the audience — clarity and accuracy are more important than drama.

═══ VISUAL STYLE DNA (apply to ALL scenes) ═══
  Color Palette: {style['color_palette']}
  Lighting: {style['lighting']}
  Lens: {style['lens']}
  Camera Movement: {style['movement']}
  Textures: {style['texture']}
  Mood: {style['mood_keywords']}

═══ VIDEO PROMPT RULES (CRITICAL — read carefully) ═══

Each video_prompt MUST be 40-80 words and follow this EXACT structure:

  1. CAMERA: Start with a specific shot type + movement direction.
     Choose from: dolly push-in, dolly pull-back, crane descending, crane ascending,
     tracking shot left/right, slow parallax, overhead top-down, worm's-eye tilt up,
     steadicam orbit, static locked-off, slow zoom-in, drone aerial push-in,
     handheld drift, slider move, dutch angle track.
     NEVER repeat the same camera movement in consecutive scenes.

  2. SUBJECT: Exactly what fills the frame — tangible physical objects, environments,
     architectural details, artifacts, landscapes, textures, documents, ruins.
     Describe literal scientific subjects, molecular structures, 3D diagrams, and informative textbook-style renders. Do NOT include human characters or faces unless absolutely necessary for the academic topic. Focus on scientific accuracy.
     CRITICAL SAFETY RULE: NEVER include children, kids, minors, babies, or orphanages. All characters MUST be adults to prevent AI safety filter blocks.

  3. ENVIRONMENT: Detailed setting with specific materials, textures, weather,
     time of day, atmosphere (fog, dust, rain, smoke, haze).

  4. LIGHTING: Name a specific technique — chiaroscuro, volumetric god rays,
     rim lighting, low-key with single source, dappled canopy light,
     firelight flicker, moonlit cold wash, candlelit warm glow, overcast flat.

  5. LENS/DOF: Specify — shallow depth of field with bokeh, anamorphic wide,
     macro close-up, telephoto compression, wide-angle with barrel distortion.

  6. COLOR: State the dominant palette for this shot.

  7. AUDIO: End with a brief ambient sound cue (wind, dripping water, distant thunder,
     crackling fire, echoing footsteps, rustling leaves, creaking wood).

CRITICAL RULES:
  - NEVER use vague mood words as visual descriptions ("eerie", "ominous", "dramatic",
    "mysterious", "haunting"). Describe what the CAMERA SEES, not what the viewer feels.
  - EVERY scene's visual must depict a SPECIFIC physical object or place directly
    related to that scene's narration — not generic atmosphere.
  - VARY camera movements across scenes — if scene 1 uses a drone shot, scene 2
    must use something completely different (dolly, crane, tracking, etc.)
  - Think like a documentary cinematographer on set: what EXACT shot tells this story?

═══ IMAGE PROMPT RULES ═══
  - Photorealistic, cinematic 16:9 composition
  - Extremely detailed subject description (e.g. "close-up of a weary man's face, tears welling up, subtle smile")
  - Include specific lighting, color palette, and texture details
  - Must match the video_prompt's subject and environment

═══ CHARACTER CAST (CRITICAL FOR VISUAL CONSISTENCY) ═══
You MUST define a `characters` array listing every recurring person/character in the video.
For each character provide:
  - `name`: A short identifier (e.g. "The Professor", "Young Soldier")
  - `physical_description`: EXTREMELY detailed physical description (80-120 words).
    Include: exact age, ethnicity, skin tone, body build, height impression,
    hair color/style/length, facial hair, eye color, nose shape, jawline,
    specific clothing with colors and textures, any distinguishing marks
    (scars, wrinkles, jewelry, tattoos). This description MUST be specific
    enough that if you gave it to 10 different artists, they'd all draw
    the same person.
  - `role`: Their narrative role ("protagonist", "antagonist", "witness", etc.)
  - `scenes_present`: List of scene_id integers where this character appears.

ALSO output a `scene_compositions` array with one entry per scene:
  - `scene_id`: matches the scene
  - `characters_present`: list of character names from the cast that appear in this scene
  - `composition_notes`: brief framing/blocking note (e.g. "extreme close-up on protagonist's tear-streaked face")

RULE: When writing image_prompt and video_prompt for a scene, you MUST reference
the character by their exact `name` from the cast. Do NOT invent new descriptions
that contradict the cast's physical_description.

═══ TEMPORAL VISUAL FLOW (CRITICAL FOR EDUCATIONAL QUALITY) ═══
Your scenes must form a CONTINUOUS VISUAL JOURNEY — not a slideshow of unrelated images.

  RULE: Each scene's video_prompt must visually FLOW from the previous scene.
  Think of the camera as a single documentary camera that never cuts — it just changes angle.

  HOW TO ACHIEVE FLOW:
  - If scene N shows a macro close-up of a molecule → scene N+1 should pull back to
    reveal the full molecular structure, OR zoom into a DIFFERENT region of the same subject.
  - If scene N shows an overhead diagram → scene N+1 can orbit around the same subject.
  - If scene N ends on a specific object → scene N+1 should start near that same object.
  - Use bridging language in the video_prompt: "Starting from the [same molecule/diagram/surface]
    shown in the previous scene…", "Pulling back to reveal…", "Rotating 90° to expose…"
  - The SUBJECT of the visual should remain conceptually connected for 3-5 consecutive scenes
    before transitioning to a new subject area.

  WHAT TO AVOID:
  - Scene 4: underwater lab → Scene 5: aerial cityscape (zero visual connection)
  - Randomly switching between close-up and aerial every scene
  - Each scene having a completely unrelated setting

═══ ENGINE SELECTION RULES (CRITICAL) ═══
For each scene, you MUST choose the most appropriate `engine_type` based on the narrative context:
1. "veo_cinematic" (DEFAULT): Use this for dramatic, historical, atmospheric, or photorealistic scenes.
2. "code_animator": Use this ONLY when explaining a complex concept, showing a map, charting data, or presenting a humorous/abstract idea where a 2D whiteboard stick-figure animation is better.

═══ METADATA RULES ═══
  - visual_style_brief: One sentence (20-30 words) describing the overall cinematic
    DNA of this video — the unifying visual language across all scenes. Reference
    specific colors, lighting, textures, and mood.
  - title: ≤60 chars, includes a power word, SEO-optimized
  - description: 150-200 words with [TIMESTAMPS], 3 CTAs, hashtags at end
  - tags: exactly 15 tags
  - thumbnail_concept: Detailed visual concept emphasizing raw human emotion on a face
"""


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "thumbnail_concept": {"type": "string"},
            },
            "required": ["title", "description", "tags", "thumbnail_concept"],
        },
        "visual_style_brief": {"type": "string"},
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "physical_description": {"type": "string"},
                    "role": {"type": "string"},
                    "scenes_present": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "physical_description", "role", "scenes_present"],
            },
        },
        "scene_compositions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "integer"},
                    "characters_present": {"type": "array", "items": {"type": "string"}},
                    "composition_notes": {"type": "string"},
                },
                "required": ["scene_id", "characters_present", "composition_notes"],
            },
        },
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "integer"},
                    "narration": {"type": "string"},
                    "image_prompt": {"type": "string"},
                    "video_prompt": {"type": "string"},
                    "engine_type": {"type": "string", "enum": ["veo_cinematic", "code_animator"]},
                },
                "required": ["scene_id", "narration", "image_prompt", "video_prompt", "engine_type"],
            },
        },
    },
    "required": ["metadata", "visual_style_brief", "characters", "scene_compositions", "scenes"],
}


class ScriptWriter:
    def __init__(self) -> None:
        self._client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options={"timeout": 120_000},  # 120s for large structured output
        )

    # ── public ──────────────────────────────────────────────────────────────

    def run(self, trend: TrendCandidate) -> VideoScript:
        log.info("Phase 2 – generating script for: '%s'", trend.title)
        raw = self._call_gemini(trend)
        script = self._parse(raw, trend)
        self._save(script)
        log.info(
            "Phase 2 done – %d scenes, title: '%s', style: '%s'",
            len(script.scenes),
            script.metadata.title,
            script.visual_style_brief[:80],
        )
        return script

    # ── private ─────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=3, min=5, max=60))
    def _call_gemini(self, trend: TrendCandidate) -> dict:
        prompt = (
            f"Create a cinematic faceless YouTube video script about this trending topic.\n\n"
            f"Trending video title: {trend.title}\n"
            f"Channel: {trend.channel_title}\n"
            f"Views in {trend.days_since_upload:.1f} days: {trend.views:,}\n"
            f"Niche: {trend.niche_keyword}\n\n"
            f"Target total narration: ~{settings.TARGET_VIDEO_DURATION} seconds across 15-20 highly granular scenes.\n"
            f"Make it MORE dramatic and MORE cinematic than the reference video.\n\n"
            f"CRITICAL: Each scene MUST contain exactly ONE sentence or short phrase of narration. "
            f"This ensures the visual perfectly maps to the exact context of what is being spoken at that exact moment.\n"
            f"IMPORTANT: Each scene's video_prompt must be 40-80 words with the full "
            f"cinematic structure (camera → subject → environment → lighting → lens → color → audio). "
            f"Use DIFFERENT camera movements for each scene — never repeat the same movement consecutively."
        )

        brand_ctx = self._load_brand_guidelines()
        parts: list = []
        if brand_ctx:
            parts.append(types.Part.from_bytes(data=brand_ctx, mime_type="application/pdf"))
        parts.append(prompt)

        system_prompt = _build_system_prompt()

        response = self._client.models.generate_content(
            model=settings.GEMINI_SCRIPT_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.85,
            ),
        )
        return json.loads(response.text)

    def _parse(self, raw: dict, trend: TrendCandidate) -> VideoScript:
        metadata = VideoMetadata(**raw["metadata"])
        scenes = [SceneScript(**s) for s in raw["scenes"]]
        visual_style_brief = raw.get("visual_style_brief", "")

        # Build storyboard from character cast + scene compositions
        characters = [
            CharacterRef(
                name=c["name"],
                physical_description=c["physical_description"],
                role=c.get("role", ""),
                scenes_present=c.get("scenes_present", []),
            )
            for c in raw.get("characters", [])
        ]
        scene_compositions = [
            SceneComposition(
                scene_id=sc["scene_id"],
                characters_present=sc.get("characters_present", []),
                composition_notes=sc.get("composition_notes", ""),
            )
            for sc in raw.get("scene_compositions", [])
        ]
        storyboard = Storyboard(
            characters=characters,
            scene_compositions=scene_compositions,
        ) if characters else None

        log.info(
            "  Storyboard: %d characters, %d scene compositions",
            len(characters), len(scene_compositions),
        )

        return VideoScript(
            metadata=metadata,
            scenes=scenes,
            visual_style_brief=visual_style_brief,
            storyboard=storyboard,
            trend_source=trend,
        )

    def _save(self, script: VideoScript) -> None:
        out = settings.OUTPUT_DIR / "script.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        log.info("Script saved -> %s", out)

    def _load_brand_guidelines(self) -> bytes | None:
        pdf = settings.BRAND_GUIDELINES_PDF
        if pdf.exists():
            return pdf.read_bytes()
        return None
