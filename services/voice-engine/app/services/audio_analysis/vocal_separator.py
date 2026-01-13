"""
UVR5 Vocal Separator Service

Provides vocal/instrumental separation using UVR5 models.
Supports HP3 (all vocals) and HP5 (main vocal only) models.
"""

import os
import logging
import tempfile
from typing import Tuple, List, Optional

import numpy as np
import soundfile as sf
import torch
import librosa

logger = logging.getLogger(__name__)

# Lazy-loaded model cache
_uvr_model = None
_model_name = None


def get_uvr5_model_path(model_name: str = "HP5_only_main_vocal") -> str:
    """
    Get the path to a UVR5 model.
    
    Checks:
      1. Project root assets/uvr5_weights/
      2. Voice-engine assets/uvr5_weights/
    """
    # Navigate from app/services/audio_analysis/ to voice-engine root
    current_dir = os.path.dirname(__file__)  # audio_analysis/
    app_dir = os.path.dirname(current_dir)  # services/
    app_root = os.path.dirname(app_dir)  # app/
    voice_engine_dir = os.path.dirname(app_root)  # voice-engine/
    project_root = os.path.dirname(os.path.dirname(voice_engine_dir))  # project root
    
    # Possible paths
    paths_to_check = [
        os.path.join(project_root, "assets", "uvr5_weights", f"{model_name}.pth"),
        os.path.join(voice_engine_dir, "assets", "uvr5_weights", f"{model_name}.pth"),
        os.path.join(project_root, "assets", "uvr5_weights", model_name),
        os.path.join(voice_engine_dir, "assets", "uvr5_weights", model_name),
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        f"UVR5 model not found: {model_name}. "
        "Run scripts/download_uvr5_assets.sh to download models."
    )


def separate_vocals(
    audio: np.ndarray,
    sample_rate: int,
    model_name: str = "HP5_only_main_vocal",
    agg: int = 10,
    device: Optional[str] = None,
    is_half: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate vocals from instrumental using UVR5 model.
    
    Args:
        audio: Input audio as numpy array (mono or stereo)
        sample_rate: Sample rate of input audio
        model_name: Name of UVR5 model:
            - HP5_only_main_vocal: Main vocal only (excludes harmonies)
            - HP3_all_vocals: All vocals including harmonies
        agg: Aggressiveness of separation (0-20, default: 10)
        device: Device to run on ('cuda' or 'cpu'), auto-detected if None
        is_half: Use half precision (faster on GPU, slightly less accurate)
    
    Returns:
        Tuple of (vocals, instrumental) as numpy arrays at 44100Hz
    
    Raises:
        RuntimeError: If UVR5 modules not available or separation fails
    """
    global _uvr_model, _model_name
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from rvc.modules.uvr5.vr import AudioPre
    except ImportError as e:
        logger.error(f"Failed to import UVR5 modules: {e}")
        raise RuntimeError("UVR5 modules not available. Check rvc installation.")
    
    model_path = get_uvr5_model_path(model_name)
    logger.info(f"Using UVR5 model: {model_path}")
    
    # Load or reload model if needed
    if _uvr_model is None or _model_name != model_name:
        logger.info(f"Loading UVR5 model: {model_name}")
        _uvr_model = AudioPre(
            agg=agg,
            model_path=model_path,
            device=device,
            is_half=is_half
        )
        _model_name = model_name
    
    # Prepare audio: must be stereo at 44100Hz
    audio = _prepare_audio_for_uvr5(audio, sample_rate)
    
    # Process through UVR5 (uses temp files)
    return _run_uvr5_separation(audio, model_name)


def _prepare_audio_for_uvr5(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert audio to stereo at 44100Hz for UVR5 processing."""
    target_sr = 44100
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio])  # Mono to stereo
    elif audio.shape[0] != 2 and audio.shape[1] == 2:
        audio = audio.T  # Transpose if channels are last dimension
    
    # Resample to 44100 if needed
    if sample_rate != target_sr:
        logger.info(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
        audio_left = librosa.resample(
            audio[0].astype(np.float32), 
            orig_sr=sample_rate, 
            target_sr=target_sr
        )
        audio_right = librosa.resample(
            audio[1].astype(np.float32), 
            orig_sr=sample_rate, 
            target_sr=target_sr
        )
        audio = np.stack([audio_left, audio_right])
    
    return audio


def _run_uvr5_separation(
    audio: np.ndarray, 
    model_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Run UVR5 separation using temporary files."""
    global _uvr_model
    
    target_sr = 44100
    
    # Save to temp file (UVR5 works with files)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
        tmp_input_path = tmp_input.name
        sf.write(tmp_input_path, audio.T, target_sr)
    
    # Create temp output directories
    with tempfile.TemporaryDirectory() as tmp_dir:
        vocals_dir = os.path.join(tmp_dir, "vocals")
        instrumental_dir = os.path.join(tmp_dir, "instrumental")
        os.makedirs(vocals_dir, exist_ok=True)
        os.makedirs(instrumental_dir, exist_ok=True)
        
        try:
            # Run separation
            is_hp3 = "HP3" in model_name
            _uvr_model._path_audio_(
                tmp_input_path,
                ins_root=instrumental_dir,
                vocal_root=vocals_dir,
                format="wav",
                is_hp3=is_hp3
            )
            
            # Load outputs
            vocal_files = [f for f in os.listdir(vocals_dir) if f.endswith(".wav")]
            instrumental_files = [f for f in os.listdir(instrumental_dir) if f.endswith(".wav")]
            
            if not vocal_files or not instrumental_files:
                raise RuntimeError("UVR5 separation produced no output")
            
            vocals, _ = sf.read(
                os.path.join(vocals_dir, vocal_files[0]), 
                dtype='float32'
            )
            instrumental, _ = sf.read(
                os.path.join(instrumental_dir, instrumental_files[0]), 
                dtype='float32'
            )
            
            # Convert to mono if stereo
            if vocals.ndim > 1:
                vocals = np.mean(vocals, axis=1)
            if instrumental.ndim > 1:
                instrumental = np.mean(instrumental, axis=1)
            
            logger.info(f"Separation complete: vocals={len(vocals)}, instrumental={len(instrumental)}")
            
            return vocals.astype(np.float32), instrumental.astype(np.float32)
            
        finally:
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)


def list_available_models() -> List[str]:
    """List available UVR5 models in the assets directory."""
    current_dir = os.path.dirname(__file__)
    app_dir = os.path.dirname(current_dir)
    app_root = os.path.dirname(app_dir)
    voice_engine_dir = os.path.dirname(app_root)
    project_root = os.path.dirname(os.path.dirname(voice_engine_dir))
    
    models = set()
    
    # Check both possible locations
    uvr5_dirs = [
        os.path.join(project_root, "assets", "uvr5_weights"),
        os.path.join(voice_engine_dir, "assets", "uvr5_weights"),
    ]
    
    for uvr5_dir in uvr5_dirs:
        if os.path.exists(uvr5_dir):
            for f in os.listdir(uvr5_dir):
                if f.endswith(".pth"):
                    models.add(f.replace(".pth", ""))
    
    return sorted(list(models))


# Model descriptions for documentation
AVAILABLE_MODELS = {
    "HP3_all_vocals": "Extracts all vocals including harmonies and backing vocals",
    "HP5_only_main_vocal": "Extracts only the main/lead vocal, excludes harmonies",
}
