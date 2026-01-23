"""
F0 Extraction Module

Implements F0 (fundamental frequency) extraction using RMVPE,
matching the WebUI reference implementation.

Creates:
- {exp_dir}/2a_f0/    - Coarse F0 contours (quantized to 256 bins)
- {exp_dir}/2b_f0nsf/ - Fine F0 contours (for NSF vocoder)
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from functools import wraps

import numpy as np
import torch

from .config import settings
from .audio import load_audio_ffmpeg

logger = logging.getLogger(__name__)


# Patch torch.load to use weights_only=False for RMVPE model compatibility
# PyTorch 2.6+ changed the default to weights_only=True
_original_torch_load = torch.load

@wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load


@dataclass
class F0Config:
    """Configuration for F0 extraction."""
    exp_name: str
    device: str = settings.device
    is_half: bool = True  # Use half precision on GPU
    
    # F0 parameters (matching WebUI)
    f0_bin: int = 256
    f0_max: float = 1100.0
    f0_min: float = 50.0


class F0Extractor:
    """
    F0 extractor using RMVPE model.
    
    Matches WebUI extract_f0_rmvpe.py exactly.
    """
    
    def __init__(self, config: F0Config):
        """Initialize F0 extractor."""
        self.config = config
        self.device = config.device
        self.is_half = config.is_half and "cuda" in self.device
        
        self.exp_dir = Path(settings.models_dir) / config.exp_name
        self.wav16k_dir = self.exp_dir / "1_16k_wavs"
        self.f0_dir = self.exp_dir / "2a_f0"
        self.f0nsf_dir = self.exp_dir / "2b_f0nsf"
        
        # F0 mel conversion parameters
        self.f0_bin = config.f0_bin
        self.f0_max = config.f0_max
        self.f0_min = config.f0_min
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        
        # RMVPE model (loaded lazily)
        self._rmvpe_model = None
        
        # Create output directories
        self.f0_dir.mkdir(exist_ok=True)
        self.f0nsf_dir.mkdir(exist_ok=True)
        
        # Log file
        self.log_file = self.exp_dir / "extract_f0_feature.log"
    
    def _log(self, message: str):
        """Log to both logger and file."""
        logger.info(message)
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")
    
    def _load_rmvpe(self):
        """Load RMVPE model if not already loaded."""
        if self._rmvpe_model is not None:
            return
        
        # Import RMVPE (requires rvc module to be available)
        try:
            # Try to import from the shared rvc module
            sys.path.insert(0, "/app")
            from rvc.lib.rmvpe import RMVPE
            
            rmvpe_path = Path(settings.rmvpe_path) / "rmvpe.pt"
            if not rmvpe_path.exists():
                raise FileNotFoundError(f"RMVPE model not found at {rmvpe_path}")
            
            self._log(f"Loading RMVPE model from {rmvpe_path}")
            self._rmvpe_model = RMVPE(
                str(rmvpe_path),
                is_half=self.is_half,
                device=self.device
            )
            
        except ImportError as e:
            logger.error(f"Failed to import RMVPE: {e}")
            raise
    
    def compute_f0(self, wav_path: str) -> np.ndarray:
        """
        Compute F0 using RMVPE.
        
        Args:
            wav_path: Path to 16kHz WAV file
            
        Returns:
            F0 array
        """
        self._load_rmvpe()
        
        # Load audio at 16kHz
        audio = load_audio_ffmpeg(wav_path, 16000)
        
        # Extract F0
        f0 = self._rmvpe_model.infer_from_audio(audio, thred=0.03)
        
        return f0
    
    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        """
        Convert F0 to coarse representation (quantized to 256 bins).
        
        Matches WebUI exactly.
        
        Args:
            f0: F0 array
            
        Returns:
            Coarse F0 array (integers 1-255)
        """
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # Clamp values
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        
        return f0_coarse
    
    def extract_file(self, wav_path: str) -> Tuple[bool, Optional[str]]:
        """
        Extract F0 for a single file.
        
        Args:
            wav_path: Path to 16kHz WAV file
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            wav_path = Path(wav_path)
            # Use .stem to get filename without extension for consistent naming
            f0_path = self.f0_dir / (wav_path.stem + ".npy")
            f0nsf_path = self.f0nsf_dir / (wav_path.stem + ".npy")
            
            # Skip if already processed
            if f0_path.exists() and f0nsf_path.exists():
                return True, None
            
            # Compute F0
            f0 = self.compute_f0(str(wav_path))
            
            # Save fine F0 (for NSF vocoder)
            np.save(str(f0nsf_path), f0, allow_pickle=False)
            
            # Save coarse F0
            f0_coarse = self.coarse_f0(f0)
            np.save(str(f0_path), f0_coarse, allow_pickle=False)
            
            return True, None
            
        except Exception as e:
            return False, f"f0fail-{wav_path}-{traceback.format_exc()}"
    
    def extract_all(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, List[str]]:
        """
        Extract F0 for all WAV files.
        
        Args:
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Tuple of (files_processed, errors)
        """
        wav_files = sorted(self.wav16k_dir.glob("*.wav"))
        
        if not wav_files:
            self._log("no-f0-todo")
            return 0, ["No WAV files found"]
        
        self._log(f"todo-f0-{len(wav_files)}")
        
        processed = 0
        errors = []
        
        for idx, wav_file in enumerate(wav_files):
            success, error = self.extract_file(str(wav_file))
            
            if success:
                processed += 1
            else:
                errors.append(error)
                self._log(error)
            
            if progress_callback:
                progress_callback(idx + 1, len(wav_files), wav_file.name)
            
            # Log progress periodically
            if idx % max(1, len(wav_files) // 5) == 0:
                self._log(f"f0ing,now-{idx},all-{len(wav_files)},{wav_file.name}")
        
        return processed, errors
    
    def validate_output(self) -> Tuple[bool, dict]:
        """
        Validate F0 extraction output.
        
        Returns:
            Tuple of (is_valid, stats_dict)
        """
        wav_files = set(f.stem for f in self.wav16k_dir.glob("*.wav"))
        f0_files = set(f.stem.replace(".wav", "") for f in self.f0_dir.glob("*.npy"))
        f0nsf_files = set(f.stem.replace(".wav", "") for f in self.f0nsf_dir.glob("*.npy"))
        
        missing_f0 = wav_files - f0_files
        missing_f0nsf = wav_files - f0nsf_files
        
        stats = {
            "wav_count": len(wav_files),
            "f0_count": len(f0_files),
            "f0nsf_count": len(f0nsf_files),
            "missing_f0": list(missing_f0),
            "missing_f0nsf": list(missing_f0nsf),
        }
        
        is_valid = len(missing_f0) == 0 and len(missing_f0nsf) == 0
        
        return is_valid, stats


def run_f0_extraction(
    config: F0Config,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[bool, dict]:
    """
    Run F0 extraction.
    
    Args:
        config: F0 extraction configuration
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (success, result_dict)
    """
    try:
        extractor = F0Extractor(config)
        processed, errors = extractor.extract_all(progress_callback)
        
        is_valid, stats = extractor.validate_output()
        
        return is_valid and len(errors) == 0, {
            "files_processed": processed,
            "errors": errors,
            "validation": stats,
        }
        
    except Exception as e:
        logger.exception("F0 extraction failed")
        return False, {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
