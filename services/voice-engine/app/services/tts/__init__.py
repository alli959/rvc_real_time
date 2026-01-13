"""
TTS Service Module

Text-to-speech generation using multiple backends:
- Bark (Suno): Best for emotions and sound effects
- Edge TTS: Fallback with audio processing for emotion simulation
"""

from app.services.tts.bark_tts import (
    generate_with_bark,
    BARK_AVAILABLE,
    BARK_SPEAKERS,
    BARK_SAMPLE_RATE,
)
from app.services.tts.edge_tts import (
    generate_with_edge_tts,
    get_available_voices,
    EDGE_VOICES,
)
from app.services.tts.text_parser import (
    parse_text_for_tts,
    convert_tags_for_bark,
    EMOTION_PROSODY,
    SOUND_REPLACEMENTS,
)
from app.services.tts.audio_effects import (
    apply_emotion_effects,
    EMOTION_AUDIO_EFFECTS,
)
from app.services.tts.service import TTSService

__all__ = [
    # Main service
    "TTSService",
    # Bark
    "generate_with_bark",
    "BARK_AVAILABLE",
    "BARK_SPEAKERS",
    "BARK_SAMPLE_RATE",
    # Edge TTS
    "generate_with_edge_tts",
    "get_available_voices",
    "EDGE_VOICES",
    # Text parsing
    "parse_text_for_tts",
    "convert_tags_for_bark",
    "EMOTION_PROSODY",
    "SOUND_REPLACEMENTS",
    # Audio effects
    "apply_emotion_effects",
    "EMOTION_AUDIO_EFFECTS",
]
