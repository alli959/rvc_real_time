"""
Model Manager Module - Handles RVC model loading and inference

This project vendors the open-source RVC inference pipeline (based on the
RVC-Project WebUI / python package) under ./rvc so that models trained
with the WebUI can be used here.
"""

from __future__ import annotations

import logging
import os
from math import gcd
from pathlib import Path
from typing import Optional, List, Callable

import numpy as np
from scipy import signal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global usage tracking callback (set by assets_api to avoid circular imports)
_usage_callback: Optional[Callable[[str], None]] = None

def set_usage_callback(callback: Callable[[str], None]) -> None:
    """Set the global callback for tracking model usage."""
    global _usage_callback
    _usage_callback = callback
    logger.info("Voice model usage callback registered")

def _notify_usage(model_name: str) -> None:
    """Notify that a model was used for inference."""
    if _usage_callback and model_name:
        try:
            _usage_callback(model_name)
        except Exception as e:
            logger.debug(f"Usage callback error: {e}")


@dataclass
class RVCInferParams:
    """Runtime inference parameters for RVC."""

    sid: int = 0
    f0_up_key: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    filter_radius: int = 3
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    resample_sr: int = 0  # output resample rate (0 = keep model's native rate)


class ModelManager:
    """Manages RVC model loading and chunk inference."""

    def __init__(
        self,
        model_dir: str = "assets/models",
        index_dir: str = "assets/index",
        hubert_path: str = "assets/hubert/hubert_base.pt",
        rmvpe_dir: str = "assets/rmvpe",
        input_sample_rate: int = 16000,
        device: str = "auto",
        default_params: Optional[RVCInferParams] = None,
    ):
        self.model_dir = Path(model_dir)
        self.index_dir = Path(index_dir)
        self.hubert_path = Path(hubert_path)
        self.rmvpe_dir = Path(rmvpe_dir)
        self.input_sample_rate = int(input_sample_rate)

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.hubert_path.parent.mkdir(parents=True, exist_ok=True)
        self.rmvpe_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables for RVC
        os.environ["weight_root"] = str(self.model_dir.resolve())
        os.environ["index_root"] = str(self.index_dir.resolve())
        os.environ["hubert_path"] = str(self.hubert_path.resolve())
        os.environ["rmvpe_root"] = str(self.rmvpe_dir.resolve())

        # Import RVC modules
        from rvc.modules.vc.modules import VC
        from rvc.modules.vc.utils import load_hubert
        import rvc.lib.rmvpe as rmvpe_lib

        self._VC = VC
        self.vc = VC()

        # Load HuBERT at startup
        logger.info("Loading HuBERT model at startup...")
        self.vc.hubert_model = load_hubert(self.vc.config, str(self.hubert_path))
        self.vc.hubert_model = self.vc.hubert_model.to(self.vc.config.device)
        logger.info(f"HuBERT model loaded and moved to device: {self.vc.config.device}")

        # Load RMVPE at startup
        logger.info("Loading RMVPE model at startup...")
        rmvpe_pt = self.rmvpe_dir / "rmvpe.pt"
        if rmvpe_pt.exists():
            self.vc.rmvpe_model = rmvpe_lib.RMVPE(
                str(rmvpe_pt), 
                is_half=getattr(self.vc.config, "is_half", False), 
                device=self.vc.config.device
            )
            logger.info("RMVPE model loaded and cached.")
        else:
            self.vc.rmvpe_model = None
            logger.warning(f"RMVPE model not found at {rmvpe_pt}")

        # Model state
        self.model_name: Optional[str] = None
        self.index_path: Optional[str] = None
        self._chunk_counter = 0
        self.default_params = default_params or RVCInferParams()

        # Device configuration
        import torch
        if device == "cuda" and torch.cuda.is_available():
            self.vc.config.device = "cuda"
        elif device == "cpu":
            self.vc.config.device = "cpu"
        else:
            self.vc.config.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _resample_poly(self, y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Fast, high-quality resampling using polyphase filtering."""
        y = np.asarray(y, dtype=np.float32).flatten()
        if orig_sr == target_sr or y.size == 0:
            return y
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return signal.resample_poly(y.astype(np.float64), up, down).astype(np.float32)

    def load_model(self, model_path: str, index_path: Optional[str] = None) -> bool:
        """Load an RVC model (.pth file).

        Args:
            model_path: Filename inside model_dir or a full/relative path.
            index_path: Optional .index file path for retrieval.

        Returns:
            True if successfully loaded.
        """
        try:
            p = Path(model_path)
            model_file = None
            logger.info(f"Attempting to load model: {model_path}")
            
            # Try different path resolution strategies
            if p.exists():
                logger.info(f"Model file exists at given path: {p}")
                model_file = p
            else:
                if p.is_absolute():
                    candidate = self.model_dir / p.name
                    if candidate.exists():
                        model_file = candidate
                
                if model_file is None:
                    candidate = self.model_dir / model_path
                    if candidate.exists():
                        model_file = candidate
            
            # Fallback to recursive search
            if not model_file or not model_file.exists():
                matches = list(self.model_dir.rglob(p.name))
                if matches:
                    expected_parent = p.parent.name if p.parent.name else None
                    if expected_parent and len(matches) > 1:
                        for m in matches:
                            if expected_parent.lower() in str(m.parent).lower():
                                model_file = m
                                break
                    if not model_file:
                        model_file = matches[0]
                else:
                    logger.error(f"Model file not found: {model_path}")
                    return False

            # Load model
            info = self.vc.get_vc(str(model_file))
            self.model_name = model_file.name

            # Handle index file
            if index_path:
                idx = Path(index_path)
                idx_file = idx if idx.exists() else (self.index_dir / index_path)
                self.index_path = str(idx_file) if idx_file.exists() else None
            else:
                stem = model_file.stem
                candidates = sorted(self.index_dir.glob(f"{stem}*.index"))
                self.index_path = str(candidates[0]) if candidates else None

            if self.index_path:
                logger.info(f"Using index file: {self.index_path}")

            logger.info(f"Successfully loaded model: {self.model_name} ({info})")
            return True

        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            return False

    def infer(self, audio: np.ndarray, params: Optional[RVCInferParams] = None) -> np.ndarray:
        """Convert audio using the loaded RVC model.
        
        Args:
            audio: Mono audio (float32, -1..1)
            params: Inference parameters
            
        Returns:
            Converted audio as numpy array
        """
        if self.model_name is None or self.vc.net_g is None:
            logger.warning("No model loaded, returning input as-is")
            return audio

        p = params or self.default_params
        x = np.asarray(audio, dtype=np.float32).flatten()

        # Normalize
        audio_max = np.abs(x).max() / 0.95 if x.size else 0.0
        if audio_max > 1.0:
            x = x / audio_max

        # Resample to 16kHz for HuBERT
        if self.input_sample_rate != 16000:
            x = self._resample_poly(x, orig_sr=self.input_sample_rate, target_sr=16000)

        # Validate assets
        hubert = Path(os.environ["hubert_path"])
        if not hubert.exists():
            logger.error(f"Missing HuBERT checkpoint: {hubert}")
            return audio

        if p.f0_method == "rmvpe":
            rmvpe_pt = Path(os.environ["rmvpe_root"]) / "rmvpe.pt"
            if not rmvpe_pt.exists():
                logger.error(f"Missing RMVPE checkpoint: {rmvpe_pt}")
                return audio

        # Run inference
        times = {"npy": 0.0, "f0": 0.0, "infer": 0.0}
        self._chunk_counter += 1
        cache_key = f"realtime_{self._chunk_counter}"

        try:
            audio_int16 = self.vc.pipeline.pipeline(
                self.vc.hubert_model,
                self.vc.net_g,
                p.sid,
                x,
                cache_key,
                times,
                p.f0_up_key,
                p.f0_method,
                self.index_path or "",
                float(p.index_rate),
                self.vc.if_f0,
                int(p.filter_radius),
                self.vc.tgt_sr,
                int(p.resample_sr),
                float(p.rms_mix_rate),
                self.vc.version,
                float(p.protect),
                f0_file=None,
            )
            
            # Track usage for the loaded model
            _notify_usage(self.model_name)
            
            return audio_int16.astype(np.float32) / 32768.0

        except Exception as e:
            logger.exception(f"Error during inference: {e}")
            return audio

    def list_available_models(self) -> List[str]:
        """List available models in the model directory."""
        model_files = []
        for ext in ["*.pth", "*.pt", "*.ckpt"]:
            model_files.extend(self.model_dir.glob(ext))
        return [f.name for f in model_files]

    def unload_model(self):
        """Unload current model and free memory."""
        try:
            self.vc.net_g = None
        except Exception:
            pass
        self.model_name = None
        self.index_path = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded (HuBERT and RMVPE remain cached)")
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model_name is not None and self.vc.net_g is not None
