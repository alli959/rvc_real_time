"""Pydantic models for API requests and responses."""

from app.models.tts import (
    TTSRequest,
    TTSResponse,
    MultiVoiceSegment,
    TTSCapabilitiesResponse,
)
from app.models.conversion import (
    ConvertRequest,
    ConvertResponse,
    ApplyEffectsRequest,
)
from app.models.audio import (
    AudioProcessRequest,
    AudioProcessResponse,
    VoiceModelConfig,
)
from app.models.youtube import (
    YouTubeSearchRequest,
    YouTubeSearchResponse,
    YouTubeDownloadRequest,
    YouTubeDownloadResponse,
)
from app.models.common import (
    HealthResponse,
    ErrorResponse,
)

__all__ = [
    # TTS
    "TTSRequest",
    "TTSResponse",
    "MultiVoiceSegment",
    "TTSCapabilitiesResponse",
    # Conversion
    "ConvertRequest",
    "ConvertResponse",
    "ApplyEffectsRequest",
    # Audio
    "AudioProcessRequest",
    "AudioProcessResponse",
    "VoiceModelConfig",
    # YouTube
    "YouTubeSearchRequest",
    "YouTubeSearchResponse",
    "YouTubeDownloadRequest",
    "YouTubeDownloadResponse",
    # Common
    "HealthResponse",
    "ErrorResponse",
]
