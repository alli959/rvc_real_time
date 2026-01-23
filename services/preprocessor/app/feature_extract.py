"""
Feature Extraction Module

Implements HuBERT feature extraction matching the WebUI reference.

Creates:
- {exp_dir}/3_feature256/ (v1) or 3_feature768/ (v2)
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
import torch.nn.functional as F
import soundfile as sf

from .config import settings

logger = logging.getLogger(__name__)


# Patch torch.load to use weights_only=False for fairseq compatibility
# PyTorch 2.6+ changed the default to weights_only=True which breaks fairseq
_original_torch_load = torch.load

@wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    exp_name: str
    version: str = "v2"  # v1 = 256 dim, v2 = 768 dim
    device: str = settings.device
    is_half: bool = True  # Use half precision on GPU


class FeatureExtractor:
    """
    HuBERT feature extractor.
    
    Matches WebUI extract_feature_print.py exactly.
    """
    
    def __init__(self, config: FeatureConfig):
        """Initialize feature extractor."""
        self.config = config
        self.version = config.version
        self.device = config.device
        self.is_half = config.is_half and "cuda" in self.device
        
        # Feature dimension based on version
        self.feature_dim = 256 if config.version == "v1" else 768
        
        self.exp_dir = Path(settings.models_dir) / config.exp_name
        self.wav16k_dir = self.exp_dir / "1_16k_wavs"
        self.feature_dir = self.exp_dir / f"3_feature{self.feature_dim}"
        
        # HuBERT model (loaded lazily)
        self._model = None
        self._saved_cfg = None
        
        # Create output directory
        self.feature_dir.mkdir(exist_ok=True)
        
        # Log file
        self.log_file = self.exp_dir / "extract_f0_feature.log"
    
    def _log(self, message: str):
        """Log to both logger and file."""
        logger.info(message)
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")
    
    def _load_model(self):
        """Load HuBERT model if not already loaded."""
        if self._model is not None:
            return
        
        import fairseq
        
        hubert_path = settings.hubert_path
        
        if not Path(hubert_path).exists():
            raise FileNotFoundError(f"HuBERT model not found at {hubert_path}")
        
        self._log(f"load model(s) from {hubert_path}")
        
        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path],
            suffix="",
        )
        
        self._model = models[0]
        self._saved_cfg = saved_cfg
        
        self._model = self._model.to(self.device)
        self._log(f"move model to {self.device}")
        
        if self.is_half and self.device not in ["mps", "cpu"]:
            self._model = self._model.half()
        
        self._model.eval()
    
    def _read_wave(self, wav_path: str, normalize: bool = False) -> torch.Tensor:
        """
        Read 16kHz WAV file.
        
        Args:
            wav_path: Path to WAV file
            normalize: Whether to apply layer norm
            
        Returns:
            Audio tensor [1, length]
        """
        wav, sr = sf.read(wav_path)
        assert sr == 16000, f"Expected 16kHz, got {sr}Hz"
        
        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:  # stereo -> mono
            feats = feats.mean(-1)
        
        assert feats.dim() == 1, f"Expected 1D, got {feats.dim()}D"
        
        if normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        
        feats = feats.view(1, -1)
        return feats
    
    def extract_file(self, wav_path: str) -> Tuple[bool, Optional[str]]:
        """
        Extract features for a single file.
        
        Args:
            wav_path: Path to 16kHz WAV file
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            self._load_model()
            
            wav_path = Path(wav_path)
            out_path = self.feature_dir / wav_path.name.replace(".wav", ".npy")
            
            # Skip if already processed
            if out_path.exists():
                return True, None
            
            # Read audio
            normalize = getattr(self._saved_cfg.task, 'normalize', False)
            feats = self._read_wave(str(wav_path), normalize=normalize)
            
            # Create padding mask
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
            
            # Prepare inputs
            inputs = {
                "source": (
                    feats.half().to(self.device)
                    if self.is_half and self.device not in ["mps", "cpu"]
                    else feats.to(self.device)
                ),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 9 if self.version == "v1" else 12,
            }
            
            # Extract features
            with torch.no_grad():
                logits = self._model.extract_features(**inputs)
                feats = (
                    self._model.final_proj(logits[0])
                    if self.version == "v1"
                    else logits[0]
                )
            
            feats = feats.squeeze(0).float().cpu().numpy()
            
            # Check for NaN
            if np.isnan(feats).sum() != 0:
                self._log(f"{wav_path.name}-contains nan")
                return False, f"{wav_path.name} contains NaN values"
            
            # Save features
            np.save(str(out_path), feats, allow_pickle=False)
            
            return True, None
            
        except Exception as e:
            return False, f"feature-fail-{wav_path}-{traceback.format_exc()}"
    
    def extract_all(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, List[str]]:
        """
        Extract features for all WAV files.
        
        Args:
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Tuple of (files_processed, errors)
        """
        wav_files = sorted(self.wav16k_dir.glob("*.wav"))
        
        if not wav_files:
            self._log("no-feature-todo")
            return 0, ["No WAV files found"]
        
        self._log(f"all-feature-{len(wav_files)}")
        
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
            if idx % max(1, len(wav_files) // 10) == 0:
                self._log(f"now-{len(wav_files)},all-{idx},{wav_file.name}")
        
        self._log("all-feature-done")
        
        return processed, errors
    
    def validate_output(self) -> Tuple[bool, dict]:
        """
        Validate feature extraction output.
        
        Returns:
            Tuple of (is_valid, stats_dict)
        """
        wav_files = set(f.stem for f in self.wav16k_dir.glob("*.wav"))
        feature_files = set(f.stem for f in self.feature_dir.glob("*.npy"))
        
        missing = wav_files - feature_files
        
        stats = {
            "wav_count": len(wav_files),
            "feature_count": len(feature_files),
            "feature_dim": self.feature_dim,
            "missing": list(missing),
        }
        
        is_valid = len(missing) == 0
        
        return is_valid, stats


def run_feature_extraction(
    config: FeatureConfig,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[bool, dict]:
    """
    Run feature extraction.
    
    Args:
        config: Feature extraction configuration
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (success, result_dict)
    """
    try:
        extractor = FeatureExtractor(config)
        processed, errors = extractor.extract_all(progress_callback)
        
        is_valid, stats = extractor.validate_output()
        
        return is_valid and len(errors) == 0, {
            "files_processed": processed,
            "errors": errors,
            "validation": stats,
        }
        
    except Exception as e:
        logger.exception("Feature extraction failed")
        return False, {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
