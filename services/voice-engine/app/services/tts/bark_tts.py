"""
Bark TTS Backend

High-quality TTS with native emotion and sound effect support.
Requires GPU (8GB+ VRAM) for reasonable performance.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Bark Availability
# =============================================================================

BARK_AVAILABLE = False
BARK_SAMPLE_RATE = 24000

try:
    from bark import generate_audio as _bark_generate, SAMPLE_RATE
    from bark.generation import preload_models
    BARK_AVAILABLE = True
    BARK_SAMPLE_RATE = SAMPLE_RATE
    logger.info("Bark TTS is available")
except ImportError:
    logger.info("Bark TTS not installed - install with: pip install git+https://github.com/suno-ai/bark.git")

# =============================================================================
# Speaker Presets
# =============================================================================

BARK_SPEAKERS = {
    'default': 'v2/en_speaker_6',
    'male1': 'v2/en_speaker_6',
    'male2': 'v2/en_speaker_3',
    'female1': 'v2/en_speaker_9',
    'female2': 'v2/en_speaker_0',
    'dramatic': 'v2/en_speaker_5',
    'calm': 'v2/en_speaker_1',
}

# =============================================================================
# Cache Setup
# =============================================================================

_bark_loaded = False


def setup_bark_cache() -> bool:
    """
    Configure Bark to use local models if available.
    
    Sets XDG_CACHE_HOME and TORCH_HOME environment variables.
    """
    local_bark_dir = Path(__file__).parent.parent.parent.parent / "assets" / "bark"
    required_models = ["text_2.pt", "coarse_2.pt", "fine_2.pt"]
    encodec_model = "encodec_24khz-d7cc33bc.th"
    
    if not local_bark_dir.exists():
        logger.info("No local Bark models - will download on first use (~13GB)")
        return False
    
    existing_models = [m for m in required_models if (local_bark_dir / m).exists()]
    has_encodec = (local_bark_dir / encodec_model).exists()
    
    if len(existing_models) != len(required_models):
        missing = set(required_models) - set(existing_models)
        logger.info(f"Local Bark models incomplete, missing: {missing}")
        return False
    
    # Set up cache directory structure
    cache_dir = local_bark_dir.parent.parent
    bark_cache = cache_dir / "suno" / "bark_v0"
    bark_cache.parent.mkdir(parents=True, exist_ok=True)
    
    if not bark_cache.exists():
        try:
            bark_cache.symlink_to(local_bark_dir)
            logger.info(f"Linked Bark cache to local models: {local_bark_dir}")
        except OSError as e:
            logger.warning(f"Could not create Bark symlink: {e}")
    
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    
    # Set up encodec cache
    if has_encodec:
        torch_cache = cache_dir / "torch" / "hub" / "checkpoints"
        torch_cache.mkdir(parents=True, exist_ok=True)
        encodec_target = torch_cache / encodec_model
        
        if not encodec_target.exists():
            try:
                encodec_target.symlink_to(local_bark_dir / encodec_model)
                logger.info(f"Linked encodec model: {encodec_target}")
            except OSError as e:
                logger.warning(f"Could not create encodec symlink: {e}")
        
        os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    
    logger.info(f"Using local Bark models from {local_bark_dir}")
    return True


def ensure_bark_loaded():
    """Preload Bark models if not already loaded."""
    global _bark_loaded
    
    if not BARK_AVAILABLE:
        raise RuntimeError("Bark TTS is not installed")
    
    if _bark_loaded:
        return
    
    setup_bark_cache()
    
    logger.info("Loading Bark models...")
    preload_models()
    _bark_loaded = True
    logger.info("Bark models loaded")


def generate_with_bark(
    text: str,
    speaker: str = "default",
    temperature: float = 0.7,
    semantic_temperature: float = 0.7,
) -> Tuple[np.ndarray, int]:
    """
    Generate speech using Bark.
    
    Args:
        text: Text to speak (can include Bark's native tags like [laughter])
        speaker: Speaker preset name from BARK_SPEAKERS
        temperature: Generation temperature (lower = more consistent)
        semantic_temperature: Semantic model temperature
    
    Returns:
        Tuple of (audio_samples, sample_rate)
    
    Note:
        Bark natively supports:
        - [laughter], [laughs], [sighs], [gasps], [clears throat]
        - ♪ for singing/music
        - CAPITALIZATION for emphasis
        - ... or — for hesitations
    """
    if not BARK_AVAILABLE:
        raise RuntimeError("Bark TTS is not installed")
    
    ensure_bark_loaded()
    
    # Get speaker preset
    speaker_preset = BARK_SPEAKERS.get(speaker, BARK_SPEAKERS['default'])
    
    logger.info(f"Generating with Bark: '{text[:50]}...' (speaker={speaker_preset})")
    
    # Generate audio
    audio = _bark_generate(
        text,
        history_prompt=speaker_preset,
        text_temp=temperature,
        waveform_temp=semantic_temperature,
    )
    
    return audio.astype(np.float32), BARK_SAMPLE_RATE
