"""
Audio Analysis Services

Voice detection and vocal separation utilities.
"""

from app.services.audio_analysis.voice_detector import (
    VoiceDetectionResult,
    detect_voice_count,
    estimate_spectral_complexity,
    count_simultaneous_pitches,
    analyze_harmonic_intervals,
)
from app.services.audio_analysis.vocal_separator import (
    separate_vocals,
    list_available_models,
    get_uvr5_model_path,
    AVAILABLE_MODELS,
)

__all__ = [
    # Voice detection
    "VoiceDetectionResult",
    "detect_voice_count",
    "estimate_spectral_complexity",
    "count_simultaneous_pitches",
    "analyze_harmonic_intervals",
    # Vocal separation
    "separate_vocals",
    "list_available_models",
    "get_uvr5_model_path",
    "AVAILABLE_MODELS",
]
