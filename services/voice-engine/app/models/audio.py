"""Audio processing request/response models."""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class VoiceModelConfig(BaseModel):
    """Configuration for a voice model in multi-voice processing."""
    model_path: str = Field(..., description="Path to .pth model file")
    index_path: Optional[str] = Field(default=None, description="Path to .index file")
    f0_up_key: int = Field(default=0, description="Pitch shift (-12 to 12)")
    extraction_mode: Literal["main", "all"] = Field(
        default="main",
        description="'main' = HP5 (main vocal only), 'all' = HP3 (all vocals including harmonies)"
    )


class AudioProcessRequest(BaseModel):
    """Audio processing request for vocal separation and swapping."""
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=44100, description="Input sample rate")
    mode: Literal["split", "convert", "swap"] = Field(
        default="split",
        description="Processing mode: split (separate), convert (voice change), swap (full pipeline)"
    )
    
    # Vocal separation options
    uvr_model: str = Field(
        default="HP5_only_main_vocal",
        description="UVR5 model for separation: HP5_only_main_vocal or HP3_all_vocals"
    )
    
    # Voice conversion options  
    model_path: Optional[str] = Field(default=None, description="Path to .pth model file")
    index_path: Optional[str] = Field(default=None, description="Path to .index file")
    f0_up_key: int = Field(default=0, description="Pitch shift (-12 to 12)")
    
    # Multi-voice swap options
    voice_count: int = Field(default=1, ge=1, le=4, description="Number of voices to extract and swap")
    voice_configs: Optional[List[VoiceModelConfig]] = Field(
        default=None,
        description="Voice configurations for multi-voice swap"
    )
    
    # Quality settings
    quality_preset: Optional[str] = Field(default="natural", description="Quality preset")
    
    # Pitch shift options
    pitch_shift_all: int = Field(default=0, description="Pitch shift all audio")
    instrumental_pitch: Optional[int] = Field(default=None, description="Pitch shift instrumental only")


class AudioProcessResponse(BaseModel):
    """Audio processing response."""
    mode: str = Field(..., description="Processing mode that was used")
    
    # Split mode outputs
    vocals: Optional[str] = Field(default=None, description="Base64 encoded vocals audio")
    instrumental: Optional[str] = Field(default=None, description="Base64 encoded instrumental audio")
    
    # Convert/Swap mode output
    converted: Optional[str] = Field(default=None, description="Base64 encoded converted audio")
    
    # Metadata
    sample_rate: int = Field(default=44100, description="Output sample rate")
    format: str = Field(default="wav", description="Audio format")


class VoiceDetectionRequest(BaseModel):
    """Voice detection request."""
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=44100, description="Input sample rate")
    max_voices: int = Field(default=4, ge=1, le=8, description="Maximum voices to detect")


class VoiceDetectionResponse(BaseModel):
    """Voice detection response."""
    voice_count: int = Field(..., description="Number of detected voices")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    details: Optional[dict] = Field(default=None, description="Additional detection details")
