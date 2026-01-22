"""
Audio Utilities Module

Audio loading and processing utilities matching the WebUI reference implementation.
"""

import os
import re
import platform
import traceback
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import librosa
import soundfile as sf


def clean_path(path_str: str) -> str:
    """
    Clean file path string (from WebUI reference).
    
    Handles:
    - Unicode control characters
    - Whitespace and quotes
    - Platform-specific path separators
    """
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    # Remove Unicode control characters
    path_str = re.sub(r'[\u202a\u202b\u202c\u202d\u202e]', '', path_str)
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


def load_audio_ffmpeg(file: str, sr: int) -> np.ndarray:
    """
    Load audio using ffmpeg subprocess (from WebUI reference).
    
    This is more robust than librosa for various formats and corrupted files.
    
    Args:
        file: Path to audio file
        sr: Target sample rate
        
    Returns:
        Audio as float32 numpy array
    """
    file = clean_path(file)
    if not os.path.exists(file):
        raise RuntimeError(f"Audio file does not exist: {file}")
    
    try:
        # Use ffmpeg to decode audio
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",  # Mono
            "-ar", str(sr),
            "-"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {err.decode()}")
        
        return np.frombuffer(out, np.float32).flatten()
        
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio with ffmpeg: {e}")


def load_audio(file: str, sr: int, use_ffmpeg: bool = True) -> np.ndarray:
    """
    Load audio file with fallback methods.
    
    Args:
        file: Path to audio file
        sr: Target sample rate
        use_ffmpeg: Whether to use ffmpeg (more robust) or librosa
        
    Returns:
        Audio as float32 numpy array (mono)
    """
    if use_ffmpeg:
        try:
            return load_audio_ffmpeg(file, sr)
        except Exception:
            pass  # Fall back to librosa
    
    # Fallback to librosa
    try:
        audio, _ = librosa.load(file, sr=sr, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")


def save_audio(file: str, audio: np.ndarray, sr: int) -> None:
    """
    Save audio to file.
    
    Args:
        file: Output path
        audio: Audio data (float32)
        sr: Sample rate
    """
    # Ensure parent directory exists
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    sf.write(file, audio.astype(np.float32), sr)


def get_audio_info(file: str) -> Tuple[int, float, int]:
    """
    Get audio file information.
    
    Args:
        file: Path to audio file
        
    Returns:
        Tuple of (sample_rate, duration_seconds, channels)
    """
    info = sf.info(file)
    return info.samplerate, info.duration, info.channels


def validate_audio(file: str, min_duration: float = 0.5) -> Tuple[bool, Optional[str]]:
    """
    Validate audio file for preprocessing.
    
    Args:
        file: Path to audio file
        min_duration: Minimum acceptable duration in seconds
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not os.path.exists(file):
            return False, "File does not exist"
        
        info = sf.info(file)
        
        if info.duration < min_duration:
            return False, f"Duration too short: {info.duration:.2f}s < {min_duration}s"
        
        if info.duration > 3600:  # 1 hour max
            return False, f"Duration too long: {info.duration:.2f}s"
        
        # Try to actually load a sample
        audio, _ = sf.read(file, frames=1000)
        if np.isnan(audio).any():
            return False, "Audio contains NaN values"
        if np.isinf(audio).any():
            return False, "Audio contains Inf values"
        
        return True, None
        
    except Exception as e:
        return False, f"Error reading audio: {str(e)}"
