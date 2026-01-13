"""TTS request/response models."""

from typing import Optional, List
from pydantic import BaseModel, Field


class MultiVoiceSegment(BaseModel):
    """A segment for multi-voice generation."""
    text: str = Field(..., description="Text for this segment")
    voice: Optional[str] = Field(default=None, description="Override TTS voice for this segment")
    voice_model_id: Optional[int] = Field(default=None, description="Voice model ID for RVC conversion")
    model_path: Optional[str] = Field(default=None, description="Path to .pth model file")
    index_path: Optional[str] = Field(default=None, description="Path to .index file")
    f0_up_key: int = Field(default=0, description="Pitch shift for this segment")
    index_rate: float = Field(default=0.75, description="Index rate for this segment")
    rate: Optional[str] = Field(default=None, description="Override speech rate for this segment")


class TTSRequest(BaseModel):
    """TTS generation request."""
    text: str = Field(..., description="Text to convert to speech", max_length=10000)
    voice: str = Field(default="en-US-GuyNeural", description="Edge TTS voice ID")
    style: str = Field(default="default", description="Speaking style/emotion")
    rate: str = Field(default="+0%", description="Speech rate adjustment")
    pitch: str = Field(default="+0Hz", description="Pitch adjustment")
    use_bark: bool = Field(default=True, description="Use Bark TTS for native emotion support")
    bark_speaker: str = Field(default="default", description="Bark speaker preset")
    include_segments: Optional[List[dict]] = Field(
        default=None, 
        description="List of segments with different voice models"
    )


class TTSResponse(BaseModel):
    """TTS generation response."""
    audio: str = Field(..., description="Base64 encoded WAV audio")
    sample_rate: int = Field(default=24000, description="Audio sample rate")
    format: str = Field(default="wav", description="Audio format")


class TTSCapabilitiesResponse(BaseModel):
    """TTS capabilities response."""
    bark_available: bool = Field(..., description="Whether Bark TTS is available")
    edge_tts_available: bool = Field(default=True, description="Whether Edge TTS is available")
    supported_emotions: List[str] = Field(default=[], description="List of supported emotions")
    bark_speakers: List[str] = Field(default=[], description="Available Bark speaker presets")
