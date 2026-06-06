from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

# ── Google AI ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
GEMINI_SCRIPT_MODEL: str = os.getenv("GEMINI_SCRIPT_MODEL", "gemini-2.5-flash")
GEMINI_TTS_MODEL: str = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
VEO_MODEL: str = os.getenv("VEO_MODEL", "veo-3.0-generate-001")
GEMINI_TTS_VOICE: str = os.getenv("GEMINI_TTS_VOICE", "Charon")

# ── Veo / Vertex AI ───────────────────────────────────────────────────────────
VEO_SERVICE_ACCOUNT_PATH: Path = ROOT / os.getenv(
    "VEO_SERVICE_ACCOUNT_PATH", "credentials/veo-service-account.json"
)
VEO_PROJECT_ID: str = os.getenv("VEO_PROJECT_ID", "veo-test-487816")
VEO_LOCATION: str = os.getenv("VEO_LOCATION", "us-central1")

# ── YouTube ───────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_CLIENT_SECRETS_PATH: Path = ROOT / os.getenv(
    "YOUTUBE_CLIENT_SECRETS_PATH", "credentials/client_secrets.json"
)
YOUTUBE_TOKEN_PATH: Path = ROOT / os.getenv(
    "YOUTUBE_TOKEN_PATH", "credentials/youtube_token.json"
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
NICHE_KEYWORDS: list[str] = [
    k.strip()
    for k in os.getenv("NICHE_KEYWORDS", "dark history,untold history").split(",")
    if k.strip()
]
TOP_TRENDS_COUNT: int = int(os.getenv("TOP_TRENDS_COUNT", "1"))
TARGET_VIDEO_DURATION: int = int(os.getenv("TARGET_VIDEO_DURATION", "90"))

OUTPUT_DIR: Path = ROOT / os.getenv("OUTPUT_DIR", "output")
BGM_DIR: Path = ROOT / os.getenv("BGM_DIR", "assets/bgm")
BRAND_GUIDELINES_PDF: Path = ROOT / os.getenv(
    "BRAND_GUIDELINES_PDF", "config/brand_guidelines.pdf"
)

# ── YouTube upload ────────────────────────────────────────────────────────────
YT_CATEGORY_ID: str = os.getenv("YT_CATEGORY_ID", "27")
YT_DEFAULT_LANGUAGE: str = os.getenv("YT_DEFAULT_LANGUAGE", "en")
YT_PRIVACY_STATUS: str = os.getenv("YT_PRIVACY_STATUS", "public")

# ── Misc ──────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"

# ── Visual Style ──────────────────────────────────────────────────────────
VISUAL_STYLE: str = os.getenv("VISUAL_STYLE", "cinematic_portrait")

VISUAL_STYLE_PRESETS: dict = {
    "cinematic_portrait": {
        "color_palette": "natural skin tones, rich emotional contrast, soft warm highlights",
        "lighting": "soft diffused lighting, Rembrandt lighting, subtle rim light to separate subject from background",
        "lens": "85mm portrait lens, very shallow depth of field, beautiful bokeh, anamorphic widescreen",
        "movement": "extremely slow and subtle handheld breathing, lingering push-ins on faces",
        "texture": "hyper-realistic skin details, raw organic textures, film grain, capturing tears or sweat",
        "mood_keywords": "intimate, emotional, raw, authentic, profound",
    },
    "dark_cinematic": {
        "color_palette": "teal-orange with deep shadows, desaturated midtones, crushed blacks",
        "lighting": "low-key chiaroscuro, volumetric fog, god rays through gaps, rim lighting on edges",
        "lens": "anamorphic widescreen, shallow depth of field, subtle lens flare, cinematic bokeh",
        "movement": "extremely slow deliberate camera movements, smooth dolly and crane work",
        "texture": "photorealistic textures, weathered surfaces, dust particles in light beams",
        "mood_keywords": "haunting, solemn, oppressive, ancient, decayed",
    },
    "ghibli_anime": {
        "color_palette": "Vibrant greens, lush earth tones, soft pastel skies, deep watercolors",
        "lighting": "Warm glowing sunlight, dappled forest shadows, soft and atmospheric",
        "lens": "Hand-painted 2D anime illustration, Studio Ghibli style, traditional animation layout",
        "texture": "Watercolor backgrounds, highly detailed foliage, cel-shaded foreground elements",
        "movement": "Slow pans, wind blowing through grass, gentle character breathing, subtle parallax",
        "mood_keywords": "magical, nostalgic, peaceful, wondrous, lush",
    },
    "warm_documentary": {
        "color_palette": "warm amber and golden tones, rich earth colors, soft highlights",
        "lighting": "golden hour natural light, soft diffused window light, warm practicals",
        "lens": "35mm film grain, medium depth of field, natural vignette",
        "movement": "handheld subtle drift, slow push-ins, gentle parallax",
        "texture": "organic textures, warm wood, aged paper, natural materials",
        "mood_keywords": "intimate, nostalgic, reverent, contemplative",
    },
    "cold_thriller": {
        "color_palette": "cold blue-steel, high contrast, crushed blacks, desaturated",
        "lighting": "harsh overhead fluorescent, neon accents, strong rim lighting, clinical",
        "lens": "wide-angle slight distortion, deep focus, surveillance camera aesthetic",
        "movement": "slow tracking shots, occasional dutch angle, static surveillance holds",
        "texture": "concrete, metal, glass, rain-wet surfaces, reflections",
        "mood_keywords": "paranoid, clinical, sterile, oppressive, voyeuristic",
    },
    "academic_science": {
        "color_palette": "clean stark white backgrounds, vivid primary colors for molecular structures, deep dark backgrounds for macro photography",
        "lighting": "studio photography lighting, completely even illumination, perfectly soft shadows, bright clinical rims",
        "lens": "hyper-macro lens, deep depth of field so everything is perfectly in focus, perfectly undistorted orthographic framing",
        "movement": "perfectly smooth 3D orbits around a subject, extremely slow dolly-in, perfect 3D rendering tracks",
        "texture": "glossy plastics, shimmering glass, detailed cellular membranes, hyper-realistic textbook diagrams",
        "mood_keywords": "academic, literal, educational, highly informative, perfectly accurate, 3D textbook",
    },
}
