"""
TTS Router

Text-to-speech generation endpoints.
"""

import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.tts import (
    TTSRequest,
    TTSResponse,
    TTSCapabilitiesResponse,
    MultiVoiceTTSRequest,
    MultiVoiceTTSResponse,
)
from app.services.tts import (
    TTSService,
    BARK_AVAILABLE,
    BARK_SPEAKERS,
    EDGE_VOICES,
)
from app.services.tts.text_parser import (
    EMOTION_PROSODY,
    SOUND_REPLACEMENTS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])

# Initialize TTS service
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service


@router.post("", response_model=TTSResponse)
async def generate_tts(request: TTSRequest):
    """
    Generate speech from text.
    
    Supports:
    - Emotion tags: [happy]text[/happy], [sad], [angry], etc.
    - Sound effects: [laugh], [sigh], [gasp], etc.
    - Speed control: <speed rate="+20%">text</speed>
    - Multiple backends: Bark (GPU) or Edge TTS (CPU)
    
    Returns audio as base64-encoded WAV.
    """
    try:
        service = get_tts_service()
        
        audio, sample_rate = service.generate(
            text=request.text,
            voice=request.voice,
            emotion=request.emotion,
            use_bark=request.use_bark,
            apply_effects=request.apply_effects if hasattr(request, 'apply_effects') else True,
        )
        
        # Convert to WAV bytes
        import soundfile as sf
        import base64
        
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sample_rate, format='WAV')
        wav_buffer.seek(0)
        
        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        return TTSResponse(
            audio=audio_base64,
            sample_rate=sample_rate,
            format="wav",
            duration=len(audio) / sample_rate,
        )
        
    except Exception as e:
        logger.exception(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities", response_model=TTSCapabilitiesResponse)
async def get_capabilities():
    """
    Get TTS capabilities.
    
    Returns available voices, emotions, sound effects, and backend info.
    """
    return TTSCapabilitiesResponse(
        bark_available=BARK_AVAILABLE,
        emotions=list(EMOTION_PROSODY.keys()),
        sound_effects=list(SOUND_REPLACEMENTS.keys()),
        voices=list(EDGE_VOICES.keys()),
        bark_speakers=list(BARK_SPEAKERS.keys()) if BARK_AVAILABLE else [],
    )


@router.post("/multi-voice", response_model=MultiVoiceTTSResponse)
async def generate_multi_voice_tts(request: MultiVoiceTTSRequest):
    """
    Generate multi-voice TTS with voice conversion.
    
    Each segment can have a different voice model applied via RVC.
    Uses <include voice_model_id="X">text</include> syntax.
    """
    try:
        from app.services import ModelManager
        
        service = get_tts_service()
        model_manager = ModelManager()
        
        segments_audio = []
        target_sr = 44100
        
        for segment in request.segments:
            # Generate base TTS
            audio, sr = service.generate(
                text=segment.text,
                voice=segment.voice,
                emotion=segment.emotion,
            )
            
            # Apply voice conversion if specified
            if segment.voice_model_id:
                try:
                    # Resample to 44100 for RVC
                    if sr != target_sr:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                        sr = target_sr
                    
                    audio = model_manager.convert(
                        audio=audio,
                        model_id=segment.voice_model_id,
                        f0_up_key=segment.f0_up_key or 0,
                        index_rate=segment.index_rate or 0.5,
                    )
                except Exception as e:
                    logger.warning(f"Voice conversion failed for segment: {e}")
            
            # Resample if needed
            if sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            segments_audio.append(audio)
        
        # Concatenate
        import numpy as np
        combined = np.concatenate(segments_audio)
        
        # Convert to WAV
        import soundfile as sf
        import base64
        
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, combined, target_sr, format='WAV')
        wav_buffer.seek(0)
        
        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        return MultiVoiceTTSResponse(
            audio=audio_base64,
            sample_rate=target_sr,
            format="wav",
            duration=len(combined) / target_sr,
            segments_count=len(segments_audio),
        )
        
    except Exception as e:
        logger.exception(f"Multi-voice TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/emotions")
async def list_emotions():
    """Get list of available emotions with their prosody settings."""
    return {
        "emotions": EMOTION_PROSODY,
        "sound_effects": list(SOUND_REPLACEMENTS.keys()),
    }


@router.get("/voices")
async def list_voices():
    """Get list of available TTS voices."""
    return {
        "edge_voices": list(EDGE_VOICES.keys()),
        "bark_speakers": list(BARK_SPEAKERS.keys()) if BARK_AVAILABLE else [],
        "bark_available": BARK_AVAILABLE,
    }
