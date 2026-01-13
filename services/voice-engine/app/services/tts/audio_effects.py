"""
Audio Effects for TTS Post-processing

Applies emotion-based audio effects to TTS output.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Effect Presets
# =============================================================================

EMOTION_AUDIO_EFFECTS: Dict[str, Dict] = {
    # Whisper effects
    'whisper': {'volume': 0.4, 'highpass': 300, 'noise': 0.025},
    'mysterious': {'volume': 0.65, 'reverb': 0.5, 'lowpass': 4800},
    
    # Shouting/Angry - louder with saturation
    'shouting': {'volume': 1.7, 'saturation': 0.4, 'compression': True},
    'angry': {'volume': 1.5, 'saturation': 0.3, 'highpass': 180},
    'furious': {'volume': 1.6, 'saturation': 0.35, 'compression': True},
    
    # Fear - tremolo for shaking voice
    'scared': {'tremolo': {'rate': 9, 'depth': 0.3}, 'pitch_wobble': 0.12},
    'terrified': {'tremolo': {'rate': 14, 'depth': 0.45}, 'pitch_wobble': 0.18, 'volume': 1.15},
    'nervous': {'tremolo': {'rate': 6, 'depth': 0.18}},
    'anxious': {'tremolo': {'rate': 7, 'depth': 0.22}},
    
    # Sad - muffled and quieter
    'sad': {'volume': 0.7, 'lowpass': 5200, 'reverb': 0.18},
    'melancholy': {'volume': 0.62, 'lowpass': 4800, 'reverb': 0.25},
    'depressed': {'volume': 0.52, 'lowpass': 4300, 'reverb': 0.3},
    'crying': {'volume': 0.75, 'tremolo': {'rate': 4.5, 'depth': 0.25}, 'lowpass': 5800},
    'sob': {'volume': 0.7, 'tremolo': {'rate': 5.5, 'depth': 0.3}, 'lowpass': 5300},
    
    # Happy - brighter
    'happy': {'volume': 1.18, 'highpass': 120, 'brightness': 0.25},
    'excited': {'volume': 1.3, 'highpass': 140, 'brightness': 0.35},
    'cheerful': {'volume': 1.12, 'brightness': 0.2},
    'joyful': {'volume': 1.25, 'brightness': 0.3},
    
    # Dramatic effects
    'dramatic': {'reverb': 0.5, 'volume': 1.15},
    
    # Special effects
    'robot': {'bitcrush': 6, 'lowpass': 3300, 'formant_shift': 0.8},
    'spooky': {'reverb': 0.6, 'lowpass': 4200, 'volume': 0.78, 'echo': True},
    'ethereal': {'reverb': 0.7, 'highpass': 400, 'shimmer': 0.35},
    'phone': {'lowpass': 3100, 'highpass': 400, 'volume': 0.85, 'saturation': 0.12},
    'radio': {'lowpass': 3600, 'highpass': 280, 'saturation': 0.18, 'noise': 0.018},
    'megaphone': {'lowpass': 4200, 'highpass': 550, 'saturation': 0.35, 'volume': 1.45, 'compression': True},
    'echo': {'reverb': 0.55, 'echo': True},
    'underwater': {'lowpass': 1100, 'volume': 0.72, 'reverb': 0.35, 'pitch_wobble': 0.1},
    
    # Sound effect enhancements
    'laugh': {'brightness': 0.25, 'volume': 1.2},
    'laughing': {'brightness': 0.3, 'volume': 1.25},
    'giggle': {'brightness': 0.35, 'volume': 1.15, 'highpass': 180},
    'scream': {'volume': 1.6, 'saturation': 0.25, 'highpass': 220},
    'shriek': {'volume': 1.5, 'saturation': 0.18, 'highpass': 320, 'brightness': 0.35},
    'sigh': {'volume': 0.8, 'lowpass': 5500, 'reverb': 0.15},
    'gasp': {'volume': 1.2, 'brightness': 0.2},
    'yawn': {'volume': 0.85, 'lowpass': 4500},
    'groan': {'volume': 0.9, 'lowpass': 4000, 'saturation': 0.1},
}


def apply_emotion_effects(
    audio: np.ndarray,
    sample_rate: int,
    emotion: Optional[str] = None,
    effects: Optional[Dict] = None
) -> np.ndarray:
    """
    Apply emotion-based audio effects.
    
    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate
        emotion: Emotion name (looks up in EMOTION_AUDIO_EFFECTS)
        effects: Override effects dict
        
    Returns:
        Processed audio
    """
    if effects is None and emotion:
        effects = EMOTION_AUDIO_EFFECTS.get(emotion.lower(), {})
    
    if not effects:
        return audio
    
    result = audio.copy().astype(np.float32)
    
    # Volume adjustment
    if 'volume' in effects:
        result = result * effects['volume']
    
    # Lowpass filter
    if 'lowpass' in effects:
        result = _apply_lowpass(result, sample_rate, effects['lowpass'])
    
    # Highpass filter
    if 'highpass' in effects:
        result = _apply_highpass(result, sample_rate, effects['highpass'])
    
    # Saturation/distortion
    if 'saturation' in effects:
        result = _apply_saturation(result, effects['saturation'])
    
    # Tremolo
    if 'tremolo' in effects:
        trem = effects['tremolo']
        result = _apply_tremolo(result, sample_rate, trem['rate'], trem['depth'])
    
    # Add noise
    if 'noise' in effects:
        result = _add_noise(result, effects['noise'])
    
    # Simple reverb
    if 'reverb' in effects:
        result = _apply_reverb(result, sample_rate, effects['reverb'])
    
    # Brightness (high shelf boost)
    if 'brightness' in effects:
        result = _apply_brightness(result, sample_rate, effects['brightness'])
    
    # Compression
    if effects.get('compression'):
        result = _apply_compression(result)
    
    # Bitcrush
    if 'bitcrush' in effects:
        result = _apply_bitcrush(result, effects['bitcrush'])
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(result))
    if max_val > 0.99:
        result = result * (0.99 / max_val)
    
    return result


def _apply_lowpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """Simple first-order lowpass filter."""
    try:
        from scipy.signal import butter, filtfilt
        nyq = sr / 2
        normalized_cutoff = min(cutoff / nyq, 0.99)
        b, a = butter(2, normalized_cutoff, btype='low')
        return filtfilt(b, a, audio).astype(np.float32)
    except Exception:
        return audio


def _apply_highpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """Simple first-order highpass filter."""
    try:
        from scipy.signal import butter, filtfilt
        nyq = sr / 2
        normalized_cutoff = min(cutoff / nyq, 0.99)
        b, a = butter(2, normalized_cutoff, btype='high')
        return filtfilt(b, a, audio).astype(np.float32)
    except Exception:
        return audio


def _apply_saturation(audio: np.ndarray, amount: float) -> np.ndarray:
    """Apply soft clipping/saturation."""
    return np.tanh(audio * (1 + amount * 3)).astype(np.float32)


def _apply_tremolo(
    audio: np.ndarray, 
    sr: int, 
    rate: float, 
    depth: float
) -> np.ndarray:
    """Apply amplitude modulation tremolo."""
    t = np.arange(len(audio)) / sr
    mod = 1 - depth * 0.5 * (1 + np.sin(2 * np.pi * rate * t))
    return (audio * mod).astype(np.float32)


def _add_noise(audio: np.ndarray, amount: float) -> np.ndarray:
    """Add white noise."""
    noise = np.random.randn(len(audio)) * amount
    return (audio + noise).astype(np.float32)


def _apply_reverb(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Simple delay-based reverb approximation."""
    delay_samples = int(sr * 0.05)  # 50ms delay
    decay = amount * 0.6
    
    result = audio.copy()
    delayed = np.zeros_like(audio)
    delayed[delay_samples:] = audio[:-delay_samples] * decay
    
    result = result + delayed
    
    # Second tap
    delay2 = int(sr * 0.08)
    if delay2 < len(audio):
        delayed2 = np.zeros_like(audio)
        delayed2[delay2:] = audio[:-delay2] * decay * 0.5
        result = result + delayed2
    
    return result.astype(np.float32)


def _apply_brightness(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Boost high frequencies for brightness."""
    try:
        from scipy.signal import butter, filtfilt
        nyq = sr / 2
        cutoff = min(3000 / nyq, 0.99)
        b, a = butter(1, cutoff, btype='high')
        highs = filtfilt(b, a, audio)
        return (audio + highs * amount).astype(np.float32)
    except Exception:
        return audio


def _apply_compression(audio: np.ndarray, threshold: float = 0.5, ratio: float = 4.0) -> np.ndarray:
    """Simple dynamic range compression."""
    result = audio.copy()
    above_threshold = np.abs(result) > threshold
    result[above_threshold] = np.sign(result[above_threshold]) * (
        threshold + (np.abs(result[above_threshold]) - threshold) / ratio
    )
    return result.astype(np.float32)


def _apply_bitcrush(audio: np.ndarray, bits: int) -> np.ndarray:
    """Reduce bit depth for lo-fi effect."""
    levels = 2 ** bits
    return (np.round(audio * levels) / levels).astype(np.float32)
