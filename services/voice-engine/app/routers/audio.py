"""
Audio Processing Router

Voice detection, vocal separation, and audio effects endpoints.
"""

import io
import base64
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.models.audio import (
    AudioProcessRequest,
    AudioProcessResponse,
    VoiceDetectRequest,
    VoiceDetectResponse,
)
from app.services.audio_analysis import (
    detect_voice_count,
    separate_vocals,
    VoiceDetectionResult,
    AVAILABLE_MODELS as UVR5_MODELS,
)
from app.services.tts.audio_effects import (
    apply_emotion_effects,
    EMOTION_AUDIO_EFFECTS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/process", response_model=AudioProcessResponse)
async def process_audio(request: AudioProcessRequest):
    """
    Process audio with various effects.
    
    Supports:
    - Vocal separation (UVR5)
    - Voice detection
    - Audio effects (reverb, filter, etc.)
    """
    try:
        import numpy as np
        import soundfile as sf
        
        # Decode input audio
        audio_bytes = base64.b64decode(request.audio)
        audio_buffer = io.BytesIO(audio_bytes)
        audio, sample_rate = sf.read(audio_buffer)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        audio = audio.astype(np.float32)
        result_audio = audio
        metadata = {}
        
        # Vocal separation
        if request.separate_vocals:
            model = request.uvr5_model or "HP5_only_main_vocal"
            vocals, instrumental = separate_vocals(
                audio=audio,
                sample_rate=sample_rate,
                model_name=model,
            )
            
            if request.output_type == "vocals":
                result_audio = vocals
            elif request.output_type == "instrumental":
                result_audio = instrumental
            else:
                result_audio = vocals  # Default to vocals
            
            metadata["separation"] = {
                "model": model,
                "output_type": request.output_type,
            }
        
        # Voice detection
        if request.detect_voices:
            detection = detect_voice_count(
                audio=result_audio,
                sr=sample_rate,
                use_vocals_only=False,  # Already separated if requested
            )
            metadata["voice_detection"] = {
                "count": detection.voice_count,
                "confidence": detection.confidence,
                "method": detection.method,
            }
        
        # Apply effects
        if request.effects:
            for effect_name in request.effects:
                if effect_name in EMOTION_AUDIO_EFFECTS:
                    result_audio = apply_emotion_effects(
                        result_audio, 
                        sample_rate, 
                        effect_name
                    )
            metadata["effects_applied"] = request.effects
        
        # Encode output
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, result_audio, sample_rate, format='WAV')
        wav_buffer.seek(0)
        
        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        return AudioProcessResponse(
            audio=audio_base64,
            sample_rate=sample_rate,
            format="wav",
            duration=len(result_audio) / sample_rate,
            metadata=metadata,
        )
        
    except Exception as e:
        logger.exception(f"Audio processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice-count/detect", response_model=VoiceDetectResponse)
async def detect_voices(request: VoiceDetectRequest):
    """
    Detect number of simultaneous voices in audio.
    
    Uses spectral and harmonic analysis to estimate voice count.
    Can optionally separate vocals first for better accuracy on music.
    """
    try:
        import numpy as np
        import soundfile as sf
        
        # Decode audio
        audio_bytes = base64.b64decode(request.audio)
        audio_buffer = io.BytesIO(audio_bytes)
        audio, sample_rate = sf.read(audio_buffer)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Detect voices
        result = detect_voice_count(
            audio=audio.astype(np.float32),
            sr=sample_rate,
            use_vocals_only=request.separate_vocals_first if hasattr(request, 'separate_vocals_first') else False,
        )
        
        return VoiceDetectResponse(
            voice_count=result.voice_count,
            confidence=result.confidence,
            method=result.method,
            details=result.details,
        )
        
    except Exception as e:
        logger.exception(f"Voice detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/separate-vocals")
async def separate_vocals_endpoint(
    file: UploadFile = File(...),
    model: str = Form("HP5_only_main_vocal"),
    output_type: str = Form("vocals"),
):
    """
    Separate vocals from instrumental in uploaded audio.
    
    Models:
    - HP5_only_main_vocal: Main vocal only (excludes harmonies)
    - HP3_all_vocals: All vocals including harmonies
    
    Output types:
    - vocals: Return only vocal track
    - instrumental: Return only instrumental track
    - both: Return both as separate files (not implemented)
    """
    try:
        import numpy as np
        import soundfile as sf
        import librosa
        
        # Read uploaded file
        content = await file.read()
        audio_buffer = io.BytesIO(content)
        
        try:
            audio, sample_rate = sf.read(audio_buffer)
        except Exception:
            audio_buffer.seek(0)
            audio, sample_rate = librosa.load(audio_buffer, sr=None)
        
        # Ensure mono for processing
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Separate
        vocals, instrumental = separate_vocals(
            audio=audio.astype(np.float32),
            sample_rate=sample_rate,
            model_name=model,
        )
        
        # Select output
        if output_type == "instrumental":
            result_audio = instrumental
        else:
            result_audio = vocals
        
        # Encode output
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, result_audio, 44100, format='WAV')  # UVR5 outputs at 44100
        wav_buffer.seek(0)
        
        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "sample_rate": 44100,
            "format": "wav",
            "model": model,
            "output_type": output_type,
        }
        
    except Exception as e:
        logger.exception(f"Vocal separation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uvr5-models")
async def list_uvr5_models():
    """List available UVR5 models for vocal separation."""
    return {
        "models": UVR5_MODELS,
        "default": "HP5_only_main_vocal",
    }


@router.get("/effects")
async def list_effects():
    """List available audio effects."""
    return {
        "effects": list(EMOTION_AUDIO_EFFECTS.keys()),
        "descriptions": {
            "whisper": "Soft, breathy effect",
            "shouting": "Loud, aggressive effect",
            "scared": "Trembling, fearful effect",
            "sad": "Muted, melancholic effect",
            "happy": "Bright, cheerful effect",
            "robot": "Robotic, processed effect",
            "phone": "Telephone line effect",
            "radio": "AM radio effect",
            "underwater": "Muffled underwater effect",
        }
    }


@router.post("/apply-effects")
async def apply_effects(
    file: UploadFile = File(...),
    effect: str = Form(...),
):
    """Apply audio effect to uploaded file."""
    try:
        import numpy as np
        import soundfile as sf
        import librosa
        
        if effect not in EMOTION_AUDIO_EFFECTS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown effect: {effect}. Available: {list(EMOTION_AUDIO_EFFECTS.keys())}"
            )
        
        # Read uploaded file
        content = await file.read()
        audio_buffer = io.BytesIO(content)
        
        try:
            audio, sample_rate = sf.read(audio_buffer)
        except Exception:
            audio_buffer.seek(0)
            audio, sample_rate = librosa.load(audio_buffer, sr=None)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Apply effect
        processed = apply_emotion_effects(
            audio.astype(np.float32),
            sample_rate,
            effect,
        )
        
        # Encode output
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, processed, sample_rate, format='WAV')
        wav_buffer.seek(0)
        
        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "sample_rate": sample_rate,
            "format": "wav",
            "effect": effect,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Effect application failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
