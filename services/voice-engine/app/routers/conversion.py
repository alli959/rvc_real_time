"""
Voice Conversion Router

RVC voice conversion endpoints.
"""

import io
import base64
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header

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


@router.post("/models/validate/{model_name}")
async def validate_model(
    model_name: str, 
    run_smoke_test: bool = True,
    x_internal_key: str = Header(None, alias="X-Internal-Key")
):
    """
    Validate a model's quality and detect training failures.
    
    This endpoint runs comprehensive quality checks on a trained model:
    - Training log analysis (detects stuck/NaN loss patterns)
    - F0/pitch extraction quality (detects bad preprocessing)
    - Checkpoint smoke test (detects collapsed/broken generators)
    
    Args:
        model_name: Name of the model directory (e.g., "lexi-11")
        run_smoke_test: Whether to run inference smoke test (slower but comprehensive)
    
    Returns:
        Validation result with pass/fail status, issues, warnings, and metrics.
        
    Example failure response for a collapsed model:
    {
        "passed": false,
        "issues": [
            "[Training] Mel loss STUCK at 75.0 for ALL iterations! Generator has COLLAPSED.",
            "[SmokeTest] Low crest factor: 1.06 (min: 3.0). Output may be collapsed/compressed."
        ],
        "warnings": [...],
        "metrics": {...}
    }
    """
    # Security: Validate model_name to prevent path traversal
    import os
    if '..' in model_name or model_name.startswith('/') or os.path.isabs(model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    # Security: Optional internal key check (configure INTERNAL_API_KEY env var)
    expected_key = os.environ.get('INTERNAL_API_KEY')
    if expected_key and x_internal_key != expected_key:
        logger.warning(f"Unauthorized validation attempt for model: {model_name}")
        raise HTTPException(status_code=403, detail="Forbidden - internal endpoint")
    
    try:
        from pathlib import Path
        from app.services.voice_conversion.training_quality_validator import validate_trained_model
        
        model_dir = Path("assets/models") / model_name
        
        # Security: Ensure resolved path is under assets/models
        model_dir_resolved = model_dir.resolve()
        base_dir_resolved = Path("assets/models").resolve()
        if not str(model_dir_resolved).startswith(str(base_dir_resolved)):
            raise HTTPException(status_code=400, detail="Invalid model path")
        
        if not model_dir.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Model directory not found: {model_name}"
            )
        
        result = validate_trained_model(str(model_dir), run_smoke_test=run_smoke_test)
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to validate model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/smoke-test/{model_name}")
async def smoke_test_model(
    model_name: str,
    x_internal_key: str = Header(None, alias="X-Internal-Key")
):
    """
    Quick smoke test to check if a model produces valid output.
    
    This is a lightweight check that:
    1. Runs inference with synthetic test audio
    2. Checks output quality metrics (DC offset, crest factor, spectral flatness)
    3. Returns pass/fail with detailed metrics
    
    Use this for quick validation before deploying a model.
    
    Args:
        model_name: Name of the model (e.g., "lexi-11/lexi-11.pth")
    
    Returns:
        {
            "passed": true/false,
            "issues": [...],
            "metrics": {
                "dc_offset": ...,
                "crest_factor": ...,
                "spectral_flatness": ...,
                "zero_crossing_rate": ...
            }
        }
    """
    import os
    
    # Security: Validate model_name to prevent path traversal
    if '..' in model_name or model_name.startswith('/') or os.path.isabs(model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    # Security: Optional internal key check
    expected_key = os.environ.get('INTERNAL_API_KEY')
    if expected_key and x_internal_key != expected_key:
        logger.warning(f"Unauthorized smoke test attempt for model: {model_name}")
        raise HTTPException(status_code=403, detail="Forbidden - internal endpoint")
    
    try:
        from pathlib import Path
        from app.services.voice_conversion.training_quality_validator import smoke_test_checkpoint
        
        # Find model file
        model_dir = Path("assets/models")
        
        # Try exact path first
        model_path = model_dir / model_name
        if not model_path.exists():
            # Try as directory
            model_path = model_dir / model_name / f"{model_name}.pth"
        if not model_path.exists():
            # Search for .pth file
            matches = list((model_dir / model_name).glob("*.pth")) if (model_dir / model_name).exists() else []
            inference_models = [f for f in matches if not f.name.startswith(('G_', 'D_'))]
            if inference_models:
                model_path = inference_models[0]
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {model_name}"
                )
        
        result = smoke_test_checkpoint(str(model_path))
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to smoke test model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
