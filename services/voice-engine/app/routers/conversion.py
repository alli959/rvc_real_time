"""
Voice Conversion Router

RVC voice conversion endpoints.
"""

import io
import base64
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.models.conversion import (
    ConvertRequest,
    ConvertResponse,
    QUALITY_PRESETS,
)
from app.services.voice_conversion import ModelManager, RVCInferParams

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/convert", tags=["conversion"])

# Model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


@router.post("", response_model=ConvertResponse)
async def convert_voice(request: ConvertRequest):
    """
    Convert voice using RVC model.
    
    Takes base64-encoded audio and returns converted audio.
    Supports quality presets and fine-grained parameter control.
    """
    try:
        import numpy as np
        import soundfile as sf
        
        model_manager = get_model_manager()
        
        # Decode input audio
        audio_bytes = base64.b64decode(request.audio)
        audio_buffer = io.BytesIO(audio_bytes)
        audio, sample_rate = sf.read(audio_buffer)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Get preset parameters if specified
        params = {}
        if request.quality_preset and request.quality_preset in QUALITY_PRESETS:
            params = QUALITY_PRESETS[request.quality_preset].copy()
        
        # Override with explicit parameters
        if request.f0_up_key is not None:
            params['f0_up_key'] = request.f0_up_key
        if request.index_rate is not None:
            params['index_rate'] = request.index_rate
        if request.filter_radius is not None:
            params['filter_radius'] = request.filter_radius
        if request.protect is not None:
            params['protect'] = request.protect
        
        # Convert
        converted = model_manager.convert(
            audio=audio.astype(np.float32),
            model_id=request.model_id,
            f0_up_key=params.get('f0_up_key', 0),
            index_rate=params.get('index_rate', 0.5),
            filter_radius=params.get('filter_radius', 3),
            protect=params.get('protect', 0.33),
        )
        
        # Encode output - use actual output sample rate from model
        out_sr = int(getattr(getattr(model_manager, "default_params", None), "resample_sr", 0) or 0)
        if out_sr <= 0:
            out_sr = int(getattr(getattr(model_manager, "vc", None), "tgt_sr", sample_rate) or sample_rate)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, converted, out_sr, format='WAV')
        wav_buffer.seek(0)
        
        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        return ConvertResponse(
            audio=audio_base64,
            sample_rate=out_sr,
            format="wav",
            duration=len(converted) / out_sr,
            model_id=request.model_id,
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Voice conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file", response_model=ConvertResponse)
async def convert_voice_file(
    file: UploadFile = File(...),
    model_id: int = Form(...),
    f0_up_key: int = Form(0),
    index_rate: float = Form(0.5),
    quality_preset: Optional[str] = Form(None),
):
    """
    Convert voice from uploaded audio file.
    
    Accepts WAV, MP3, FLAC, and other common audio formats.
    """
    try:
        import numpy as np
        import soundfile as sf
        import librosa
        
        model_manager = get_model_manager()
        
        # Read uploaded file
        content = await file.read()
        audio_buffer = io.BytesIO(content)
        
        # Try to read with soundfile first
        try:
            audio, sample_rate = sf.read(audio_buffer)
        except Exception:
            # Fallback to librosa for MP3 etc.
            audio_buffer.seek(0)
            audio, sample_rate = librosa.load(audio_buffer, sr=None)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Get preset parameters
        params = {'f0_up_key': f0_up_key, 'index_rate': index_rate}
        if quality_preset and quality_preset in QUALITY_PRESETS:
            params = QUALITY_PRESETS[quality_preset].copy()
            params['f0_up_key'] = f0_up_key  # Keep user-specified pitch
        
        # Convert
        converted = model_manager.convert(
            audio=audio.astype(np.float32),
            model_id=model_id,
            f0_up_key=params.get('f0_up_key', 0),
            index_rate=params.get('index_rate', 0.5),
        )
        
        # Encode output - use actual output sample rate from model
        out_sr = int(getattr(getattr(model_manager, "default_params", None), "resample_sr", 0) or 0)
        if out_sr <= 0:
            out_sr = int(getattr(getattr(model_manager, "vc", None), "tgt_sr", sample_rate) or sample_rate)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, converted, out_sr, format='WAV')
        wav_buffer.seek(0)
        
        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        return ConvertResponse(
            audio=audio_base64,
            sample_rate=out_sr,
            format="wav",
            duration=len(converted) / out_sr,
            model_id=model_id,
        )
        
    except Exception as e:
        logger.exception(f"Voice conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets")
async def get_presets():
    """Get available quality presets."""
    return {
        "presets": QUALITY_PRESETS,
        "description": {
            "fast": "Fastest processing, lower quality",
            "balanced": "Good balance of speed and quality",
            "high": "High quality, slower processing",
            "max": "Maximum quality, slowest processing",
        }
    }


@router.get("/models")
async def list_models():
    """
    List available voice models.
    
    Returns all loaded RVC models with their metadata.
    """
    try:
        model_manager = get_model_manager()
        models = model_manager.list_models()
        
        return {
            "models": models,
            "count": len(models),
        }
        
    except Exception as e:
        logger.exception(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}")
async def get_model(model_id: int):
    """Get details for a specific model."""
    try:
        model_manager = get_model_manager()
        models = model_manager.list_models()
        
        for model in models:
            if model.get('id') == model_id:
                return model
        
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
