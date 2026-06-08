from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

# ── Google AI ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
GEMINI_SCRIPT_MODEL: str = os.getenv("GEMINI_SCRIPT_MODEL", "gemini-2.5-flash")
# Imagen model for keyframes. Default to Imagen 4 ULTRA — it renders text/formulas
# (CO2, F=ma) far more reliably than the standard model. Set GEMINI_IMAGE_MODEL=
# imagen-4.0-generate-001 for higher throughput / lower cost if text accuracy matters less.
GEMINI_IMAGE_MODEL: str = os.getenv("GEMINI_IMAGE_MODEL", "imagen-4.0-ultra-generate-001")
GEMINI_TTS_MODEL: str = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
VEO_MODEL: str = os.getenv("VEO_MODEL", "veo-3.0-generate-001")
GEMINI_TTS_VOICE: str = os.getenv("GEMINI_TTS_VOICE", "Charon")

# ── Veo / Vertex AI ───────────────────────────────────────────────────────────
VEO_SERVICE_ACCOUNT_PATH: Path = ROOT / os.getenv(
    "VEO_SERVICE_ACCOUNT_PATH", "credentials/veo-service-account.json"
)
VEO_PROJECT_ID: str = os.getenv("VEO_PROJECT_ID", "veo-test-487816")
VEO_LOCATION: str = os.getenv("VEO_LOCATION", "us-central1")
# First+last-frame interpolation for perfect scene handoffs. OFF by default:
# it is Veo's slow/expensive path (10x+ render time). Phase 4 xfade handles transitions.
VEO_LAST_FRAME: bool = os.getenv("VEO_LAST_FRAME", "false").lower() == "true"
# How many Veo/Imagen jobs to run concurrently in Phase 3. Veo polling is I/O-bound,
# so concurrency turns a serial 15-min render into ~one-clip wall-time. Keep modest to
# avoid 429 rate-limit errors on AI Studio keys; raise on Vertex with provisioned quota.
VEO_CONCURRENCY: int = int(os.getenv("VEO_CONCURRENCY", "4"))

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

# Brand watermark burned into the top-right of every video. Set WATERMARK_TEXT="" to disable.
WATERMARK_TEXT: str = os.getenv("WATERMARK_TEXT", "Dextora")

# ── Visual Style ──────────────────────────────────────────────────────────
VISUAL_STYLE: str = os.getenv("VISUAL_STYLE", "cinematic_portrait")

VISUAL_STYLE_PRESETS: dict = {
    "cinematic_portrait": {
        "medium": "photorealistic live-action cinematography, real human actors, shot on a digital cinema camera",
        "photoreal": True,
        "color_palette": "natural skin tones, rich emotional contrast, soft warm highlights",
        "lighting": "soft diffused lighting, Rembrandt lighting, subtle rim light to separate subject from background",
        "lens": "85mm portrait lens, very shallow depth of field, beautiful bokeh, anamorphic widescreen",
        "movement": "extremely slow and subtle handheld breathing, lingering push-ins on faces",
        "texture": "hyper-realistic skin details, raw organic textures, film grain, capturing tears or sweat",
        "mood_keywords": "intimate, emotional, raw, authentic, profound",
    },
    "dark_cinematic": {
        "medium": "photorealistic cinematic film, live-action, shot on anamorphic lenses",
        "photoreal": True,
        "color_palette": "teal-orange with deep shadows, desaturated midtones, crushed blacks",
        "lighting": "low-key chiaroscuro, volumetric fog, god rays through gaps, rim lighting on edges",
        "lens": "anamorphic widescreen, shallow depth of field, subtle lens flare, cinematic bokeh",
        "movement": "extremely slow deliberate camera movements, smooth dolly and crane work",
        "texture": "photorealistic textures, weathered surfaces, dust particles in light beams",
        "mood_keywords": "haunting, solemn, oppressive, ancient, decayed",
    },
    "ghibli_anime": {
        "medium": "hand-painted 2D Studio Ghibli anime, cel-shaded, traditional hand-drawn animation, watercolor backgrounds — STRICTLY NOT photorealistic, NOT 3D render, NOT live-action",
        "photoreal": False,
        "negative_extra": "photorealistic, photograph, live-action footage, realistic human skin, real camera, 3D render, CGI, octane render",
        "color_palette": "Vibrant greens, lush earth tones, soft pastel skies, deep watercolors",
        "lighting": "Warm glowing sunlight, dappled forest shadows, soft and atmospheric",
        "lens": "Hand-painted 2D anime illustration, Studio Ghibli style, traditional animation layout",
        "texture": "Watercolor backgrounds, highly detailed foliage, cel-shaded foreground elements",
        "movement": "Slow pans, wind blowing through grass, gentle character breathing, subtle parallax",
        "mood_keywords": "magical, nostalgic, peaceful, wondrous, lush",
    },
    "warm_documentary": {
        "medium": "photorealistic documentary film, live-action, natural 35mm film stock",
        "photoreal": True,
        "color_palette": "warm amber and golden tones, rich earth colors, soft highlights",
        "lighting": "golden hour natural light, soft diffused window light, warm practicals",
        "lens": "35mm film grain, medium depth of field, natural vignette",
        "movement": "handheld subtle drift, slow push-ins, gentle parallax",
        "texture": "organic textures, warm wood, aged paper, natural materials",
        "mood_keywords": "intimate, nostalgic, reverent, contemplative",
    },
    "cold_thriller": {
        "medium": "photorealistic cinematic film, live-action, clinical surveillance aesthetic",
        "photoreal": True,
        "color_palette": "cold blue-steel, high contrast, crushed blacks, desaturated",
        "lighting": "harsh overhead fluorescent, neon accents, strong rim lighting, clinical",
        "lens": "wide-angle slight distortion, deep focus, surveillance camera aesthetic",
        "movement": "slow tracking shots, occasional dutch angle, static surveillance holds",
        "texture": "concrete, metal, glass, rain-wet surfaces, reflections",
        "mood_keywords": "paranoid, clinical, sterile, oppressive, voyeuristic",
    },
    "academic_science": {
        "medium": "clean 3D rendered scientific animation, polished textbook illustration style, Pixar-grade 3D render — NOT live-action, NOT hand-drawn",
        "photoreal": False,
        "negative_extra": "photorealistic photograph, live-action footage, real human actors, hand-drawn sketch, 2D anime",
        "color_palette": "clean stark white backgrounds, vivid primary colors for molecular structures, deep dark backgrounds for macro photography",
        "lighting": "studio photography lighting, completely even illumination, perfectly soft shadows, bright clinical rims",
        "lens": "hyper-macro lens, deep depth of field so everything is perfectly in focus, perfectly undistorted orthographic framing",
        "movement": "perfectly smooth 3D orbits around a subject, extremely slow dolly-in, perfect 3D rendering tracks",
        "texture": "glossy plastics, shimmering glass, detailed cellular membranes, hyper-realistic textbook diagrams",
        "mood_keywords": "academic, literal, educational, highly informative, perfectly accurate, 3D textbook",
    },
}

# ── TTS voice + delivery per style ───────────────────────────────────────────
# Gemini TTS accepts natural-language delivery direction (a "style prompt"), which
# is the single biggest quality lever — it turns flat read-aloud into a directed
# performance. Each style maps to a fitting prebuilt voice + a delivery instruction.
# Override the whole thing with GEMINI_TTS_VOICE env if you want one fixed voice.
# ── Narration profiles (accent + language) ───────────────────────────────────
# A profile drives BOTH the script language Phase 2 writes AND how Phase 3 TTS
# speaks it. Accents (Indian/British/American) keep an English script; Hindi and
# Hinglish change the script language itself (so subtitles match the audio).
#   script_lang   → injected into the Phase 2 system prompt (what language to write)
#   tts_directive → woven into the Gemini TTS delivery prompt (how to speak)
#   whisper_lang  → ASR language for subtitle transcription
NARRATION_PROFILES: dict = {
    "indian_english": {
        "label": "Indian English",
        "script_lang": "English",
        "tts_directive": "speaking in a natural, clear Indian English accent",
        "whisper_lang": "en", "use_asr": False, "sub_font": "Arial",
    },
    "british_english": {
        "label": "British English",
        "script_lang": "English",
        "tts_directive": "speaking in a refined British English (Received Pronunciation) accent",
        "whisper_lang": "en", "use_asr": False, "sub_font": "Arial",
    },
    "american_english": {
        "label": "American English",
        "script_lang": "English",
        "tts_directive": "speaking in a neutral American English accent",
        "whisper_lang": "en", "use_asr": False, "sub_font": "Arial",
    },
    "hindi": {
        "label": "Hindi",
        "script_lang": "Hindi, written in Devanagari script (natural spoken Hindi, not overly formal)",
        "tts_directive": "speaking naturally and fluently in Hindi",
        # ASR (Whisper base) is unreliable for Hindi (wrong script) — build subtitles
        # from the known Devanagari script text instead, rendered with a Devanagari font.
        "whisper_lang": "hi", "use_asr": False, "sub_font": "Noto Sans Devanagari",
    },
    "hinglish": {
        "label": "Hinglish",
        "script_lang": ("Hinglish — a natural conversational mix of Hindi and English as spoken by "
                        "young urban Indians, written in Roman (English) script "
                        "(e.g. 'Yeh concept bahut interesting hai, let's understand it step by step')"),
        "tts_directive": "speaking in a natural Indian accent, blending Hindi and English casually",
        # Roman script but Hindi words → ASR mis-hears; use the known text.
        "whisper_lang": "en", "use_asr": False, "sub_font": "Arial",
    },
}
NARRATION_LANGUAGE: str = os.getenv("NARRATION_LANGUAGE", "indian_english")


def narration_profile() -> dict:
    """Active narration profile (falls back to Indian English)."""
    return NARRATION_PROFILES.get(NARRATION_LANGUAGE, NARRATION_PROFILES["indian_english"])


TTS_DEFAULT = {
    "voice": "Charon",
    "tone": "Read this aloud as a clear, confident narrator with a natural, measured pace",
}
TTS_STYLE_PRESETS: dict = {
    "academic_science":   {"voice": "Charon", "tone": "Read this aloud as a clear, authoritative science educator — measured pace, confident and articulate, with crisp diction"},
    "cinematic_portrait": {"voice": "Aoede",  "tone": "Read this aloud as a warm, intimate narrator — natural, reflective, unhurried, with genuine feeling"},
    "warm_documentary":   {"voice": "Aoede",  "tone": "Read this aloud as a warm, nostalgic documentary narrator — gentle, reverent and contemplative"},
    "ghibli_anime":       {"voice": "Leda",   "tone": "Read this aloud as a warm, gentle storyteller narrating a magical fable — soft, wonder-filled and soothing"},
    "dark_cinematic":     {"voice": "Charon", "tone": "Read this aloud as a deep, dramatic documentary narrator — measured, suspenseful and weighty, with gravitas"},
    "cold_thriller":      {"voice": "Fenrir", "tone": "Read this aloud in a tense, hushed, urgent tone — clipped, serious and controlled"},
}
