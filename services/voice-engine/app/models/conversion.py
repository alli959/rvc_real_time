"""Voice conversion request/response models."""

from typing import Optional
from pydantic import BaseModel, Field


# Quality presets for voice conversion
QUALITY_PRESETS = {
    'natural': {
        'index_rate': 0.4,
        'rms_mix_rate': 0.15,
        'protect': 0.45,
        'description': 'Most natural sounding, preserves original speech characteristics'
    },
    'balanced': {
        'index_rate': 0.55,
        'rms_mix_rate': 0.25,
        'protect': 0.35,
        'description': 'Balance between naturalness and voice accuracy'
    },
    'accurate': {
        'index_rate': 0.75,
        'rms_mix_rate': 0.35,
        'protect': 0.25,
        'description': 'Most similar to target voice, may sound less natural'
    },
    'studio': {
        'index_rate': 0.5,
        'rms_mix_rate': 0.2,
        'protect': 0.4,
        'description': 'Optimized for studio-quality recordings'
    },
}


class ConvertRequest(BaseModel):
    """Voice conversion request."""
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=16000, description="Input sample rate")
    model_path: str = Field(..., description="Path to .pth model file")
    index_path: Optional[str] = Field(default=None, description="Path to .index file")
    f0_up_key: int = Field(default=0, description="Pitch shift (-12 to 12)")
    f0_method: str = Field(default="rmvpe", description="F0 extraction method")
    index_rate: float = Field(default=0.5, description="Index blend rate (0.0-1.0)")
    filter_radius: int = Field(default=3, description="Filter radius (0-7)")
    rms_mix_rate: float = Field(default=0.2, description="RMS mix rate (0.0-1.0)")
    protect: float = Field(default=0.4, description="Protect consonants (0.0-0.5)")
    quality_preset: Optional[str] = Field(
        default=None, 
        description="Quality preset: 'natural', 'balanced', 'accurate', 'studio'"
    )
    apply_effects: Optional[str] = Field(
        default=None, 
        description="Emotion/effect name to apply after conversion"
    )


class ConvertResponse(BaseModel):
    """Voice conversion response."""
    audio: str = Field(..., description="Base64 encoded converted audio")
    sample_rate: int = Field(default=16000, description="Output sample rate")
    format: str = Field(default="wav", description="Audio format")


class ApplyEffectsRequest(BaseModel):
    """Request to apply audio effects to existing audio."""
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=16000, description="Input sample rate")
    effect: str = Field(..., description="Effect name to apply (e.g. 'robot', 'whisper')")


class ApplyEffectsResponse(BaseModel):
    """Response from applying audio effects."""
    audio: str = Field(..., description="Base64 encoded processed audio")
    sample_rate: int = Field(..., description="Output sample rate")
    format: str = Field(default="wav", description="Audio format")
    effect_applied: str = Field(..., description="Name of the effect that was applied")
