"""
Voice Engine Services

Business logic layer for the voice engine API.
"""

# Voice Conversion
from app.services.voice_conversion import (
    ModelManager,
    RVCInferParams,
)

# Audio Analysis
from app.services.audio_analysis import (
    VoiceDetectionResult,
    detect_voice_count,
    separate_vocals,
    list_available_models as list_uvr5_models,
    AVAILABLE_MODELS as UVR5_MODELS,
)

# TTS
from app.services.tts import (
    TTSService,
    BARK_AVAILABLE,
    BARK_SPEAKERS,
    EDGE_VOICES,
)

# YouTube
from app.services.youtube import (
    YouTubeService,
    SearchResult as YouTubeSearchResult,
    search_youtube,
    download_youtube_audio,
)

# Model Cache
from app.services.model_cache import (
    ModelCache,
    CacheConfig,
    get_model_cache,
    configure_model_cache,
)

__all__ = [
    # Voice Conversion
    "ModelManager",
    "RVCInferParams",
    # Audio Analysis
    "VoiceDetectionResult",
    "detect_voice_count",
    "separate_vocals",
    "list_uvr5_models",
    "UVR5_MODELS",
    # TTS
    "TTSService",
    "BARK_AVAILABLE",
    "BARK_SPEAKERS",
    "EDGE_VOICES",
    # YouTube
    "YouTubeService",
    "YouTubeSearchResult",
    "search_youtube",
    "download_youtube_audio",
    # Model Cache
    "ModelCache",
    "CacheConfig",
    "get_model_cache",
    "configure_model_cache",
]
