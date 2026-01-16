"""
UVR5 Vocal Separator Module

Provides vocal/instrumental separation using UVR5 models.
Based on the implementation from Retrieval-based-Voice-Conversion-WebUI.

Memory optimization: Processes long audio in chunks to avoid OOM.
"""

import os
import gc
import logging
import tempfile
import numpy as np
import soundfile as sf
import torch
import librosa
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Lazy load the heavy imports
_uvr_model = None
_model_name = None

# Max audio length in samples before chunking (5 minutes at 44100Hz)
MAX_CHUNK_SAMPLES = 44100 * 60 * 5  # 5 minutes


def get_uvr5_model_path(model_name: str = "HP5_only_main_vocal") -> str:
    """Get the path to a UVR5 model"""
    # Check in project root assets/uvr5_weights - go up three levels from app/ to project root
    # __file__ = /services/voice-engine/app/vocal_separator.py
    # up 1 = /services/voice-engine/app
    # up 2 = /services/voice-engine
    # up 3 = /services
    # up 4 = / (project root)
    voice_engine_path = os.path.dirname(os.path.dirname(__file__))  # /services/voice-engine
    project_root = os.path.dirname(os.path.dirname(voice_engine_path))  # project root
    
    # First check in project root assets/
    model_path = os.path.join(project_root, "assets", "uvr5_weights", f"{model_name}.pth")
    
    if os.path.exists(model_path):
        return model_path
    
    # Also check in voice-engine/assets/ as fallback
    fallback_path = os.path.join(voice_engine_path, "assets", "uvr5_weights", f"{model_name}.pth")
    if os.path.exists(fallback_path):
        return fallback_path
    
    # Also check without .pth extension
    if not model_name.endswith(".pth"):
        model_path = os.path.join(project_root, "assets", "uvr5_weights", model_name)
        if os.path.exists(model_path):
            return model_path
    
    raise FileNotFoundError(f"UVR5 model not found: {model_name}. Run scripts/download_uvr5_assets.sh")


def separate_vocals(
    audio: np.ndarray,
    sample_rate: int,
    model_name: str = "HP5_only_main_vocal",
    agg: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    is_half: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate vocals from instrumental using UVR5 model.
    
    Args:
        audio: Input audio as numpy array (mono or stereo)
        sample_rate: Sample rate of input audio
        model_name: Name of UVR5 model to use (default: HP5_only_main_vocal)
        agg: Aggressiveness of separation (0-20, default: 10)
        device: Device to run on ('cuda' or 'cpu')
        is_half: Use half precision (faster, slightly less accurate)
    
    Returns:
        Tuple of (vocals, instrumental) as numpy arrays at 44100Hz
    """
    global _uvr_model, _model_name
    
    try:
        from rvc.modules.uvr5.vr import AudioPre
        from rvc.lib.uvr5_pack.lib_v5 import spec_utils
    except ImportError as e:
        logger.error(f"Failed to import UVR5 modules: {e}")
        raise RuntimeError("UVR5 modules not available. Check rvc installation.")
    
    # Get model path
    model_path = get_uvr5_model_path(model_name)
    logger.info(f"Using UVR5 model: {model_path}")
    
    # Clear GPU cache before loading model to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Check if we need to reload model
    if _uvr_model is None or _model_name != model_name:
        logger.info(f"Loading UVR5 model: {model_name}")
        _uvr_model = AudioPre(
            agg=agg,
            model_path=model_path,
            device=device,
            is_half=is_half
        )
        _model_name = model_name
    
    # Audio must be stereo at 44100Hz for UVR5
    if audio.ndim == 1:
        audio = np.stack([audio, audio])  # Mono to stereo
    elif audio.shape[0] != 2 and audio.shape[1] == 2:
        audio = audio.T  # Transpose if channels are last dimension
    
    # Resample to 44100 if needed
    target_sr = 44100
    if sample_rate != target_sr:
        logger.info(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
        audio_left = librosa.resample(audio[0].astype(np.float32), orig_sr=sample_rate, target_sr=target_sr)
        audio_right = librosa.resample(audio[1].astype(np.float32), orig_sr=sample_rate, target_sr=target_sr)
        audio = np.stack([audio_left, audio_right])
    
    # Check if audio is too long - process in chunks to avoid OOM
    audio_length = audio.shape[1]
    if audio_length > MAX_CHUNK_SAMPLES:
        logger.info(f"Audio is {audio_length / target_sr / 60:.1f} minutes - processing in chunks to save memory")
        return _separate_vocals_chunked(audio, target_sr, model_name, agg, device, is_half)
    
    # Process normally for shorter audio
    return _separate_vocals_single(audio, target_sr, model_name, agg, device, is_half)


def _separate_vocals_chunked(
    audio: np.ndarray,
    sample_rate: int,
    model_name: str,
    agg: int,
    device: str,
    is_half: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Process long audio in chunks to avoid OOM"""
    global _uvr_model
    
    audio_length = audio.shape[1]
    chunk_size = MAX_CHUNK_SAMPLES
    overlap = sample_rate * 5  # 5 second overlap for smooth transitions
    
    all_vocals = []
    all_instrumental = []
    
    num_chunks = (audio_length + chunk_size - overlap - 1) // (chunk_size - overlap)
    logger.info(f"Processing {num_chunks} chunks...")
    
    for i, start in enumerate(range(0, audio_length, chunk_size - overlap)):
        end = min(start + chunk_size, audio_length)
        chunk = audio[:, start:end]
        
        logger.info(f"Processing chunk {i+1}/{num_chunks} ({start/sample_rate:.1f}s - {end/sample_rate:.1f}s)")
        
        # Clear memory before each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Process this chunk
        vocals, instrumental = _separate_vocals_single(chunk, sample_rate, model_name, agg, device, is_half)
        
        # Handle overlap with crossfade
        if len(all_vocals) > 0 and i > 0:
            # Crossfade with previous chunk
            fade_samples = min(overlap, len(vocals), len(all_vocals[-1]))
            if fade_samples > 0:
                # Create fade curves
                fade_out = np.linspace(1.0, 0.0, fade_samples).astype(np.float32)
                fade_in = np.linspace(0.0, 1.0, fade_samples).astype(np.float32)
                
                # Apply crossfade to the overlap region
                overlap_vocals = all_vocals[-1][-fade_samples:] * fade_out + vocals[:fade_samples] * fade_in
                overlap_inst = all_instrumental[-1][-fade_samples:] * fade_out + instrumental[:fade_samples] * fade_in
                
                # Replace end of previous chunk and start of current
                all_vocals[-1] = all_vocals[-1][:-fade_samples]
                all_instrumental[-1] = all_instrumental[-1][:-fade_samples]
                
                all_vocals.append(overlap_vocals)
                all_instrumental.append(overlap_inst)
                
                vocals = vocals[fade_samples:]
                instrumental = instrumental[fade_samples:]
        
        all_vocals.append(vocals)
        all_instrumental.append(instrumental)
        
        # Check if this is the last chunk
        if end >= audio_length:
            break
    
    # Concatenate all chunks
    final_vocals = np.concatenate(all_vocals)
    final_instrumental = np.concatenate(all_instrumental)
    
    logger.info(f"Chunked separation complete: vocals={len(final_vocals)}, instrumental={len(final_instrumental)}")
    
    return final_vocals, final_instrumental


def _separate_vocals_single(
    audio: np.ndarray,
    sample_rate: int,
    model_name: str,
    agg: int,
    device: str,
    is_half: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single audio chunk"""
    global _uvr_model
    
    # Save to temp file (UVR5 works with files)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
        tmp_input_path = tmp_input.name
        sf.write(tmp_input_path, audio.T, sample_rate)
    
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
            # Find the output files (they have _agg suffix)
            vocal_files = [f for f in os.listdir(vocals_dir) if f.endswith(".wav")]
            instrumental_files = [f for f in os.listdir(instrumental_dir) if f.endswith(".wav")]
            
            if not vocal_files or not instrumental_files:
                raise RuntimeError("UVR5 separation produced no output")
            
            # IMPORTANT: UVR5 vr.py has inverted output logic for HP3 models!
            # When is_hp3=True:
            #   - ins_root gets VOCALS (with head="vocal_")
            #   - vocal_root gets INSTRUMENTALS (with head="instrument_")
            # When is_hp3=False (HP5, etc):
            #   - ins_root gets INSTRUMENTALS (with head="instrument_")
            #   - vocal_root gets VOCALS (with head="vocal_")
            # So we need to swap the reads for HP3 models.
            
            if is_hp3:
                # HP3: vocals are in instrumental_dir, instrumentals are in vocals_dir
                vocals, _ = sf.read(os.path.join(instrumental_dir, instrumental_files[0]), dtype='float32')
                instrumental, _ = sf.read(os.path.join(vocals_dir, vocal_files[0]), dtype='float32')
                logger.info(f"HP3 mode: swapped reads (vocals from ins_dir, instrumental from vocal_dir)")
            else:
                # HP5 and others: normal read
                vocals, _ = sf.read(os.path.join(vocals_dir, vocal_files[0]), dtype='float32')
                instrumental, _ = sf.read(os.path.join(instrumental_dir, instrumental_files[0]), dtype='float32')
            
            # Convert to mono if stereo
            if vocals.ndim > 1:
                vocals = np.mean(vocals, axis=1)
            if instrumental.ndim > 1:
                instrumental = np.mean(instrumental, axis=1)
            
            logger.info(f"Separation complete: vocals={len(vocals)}, instrumental={len(instrumental)}")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return vocals.astype(np.float32), instrumental.astype(np.float32)
            
        finally:
            # Clean up input file
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)


def list_available_models() -> list:
    """List available UVR5 models"""
    voice_engine_path = os.path.dirname(os.path.dirname(__file__))  # /services/voice-engine
    project_root = os.path.dirname(os.path.dirname(voice_engine_path))  # project root
    
    models = []
    
    # Check project root assets first
    uvr5_dir = os.path.join(project_root, "assets", "uvr5_weights")
    if os.path.exists(uvr5_dir):
        for f in os.listdir(uvr5_dir):
            if f.endswith(".pth"):
                models.append(f.replace(".pth", ""))
    
    # Also check voice-engine/assets as fallback
    fallback_dir = os.path.join(voice_engine_path, "assets", "uvr5_weights")
    if os.path.exists(fallback_dir):
        for f in os.listdir(fallback_dir):
            if f.endswith(".pth"):
                name = f.replace(".pth", "")
                if name not in models:
                    models.append(name)
    
    return models
