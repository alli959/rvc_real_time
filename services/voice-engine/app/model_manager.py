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

from scipy import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


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
    resample_sr: int = 16000  # output resample rate (0 disables)


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

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.hubert_path.parent.mkdir(parents=True, exist_ok=True)
        self.rmvpe_dir.mkdir(parents=True, exist_ok=True)

        # RVC code uses a few environment variables to locate resources.
        # We *override* them so running with other RVC tools in the same shell won't accidentally point
        # to a different model/index directory.
        os.environ["weight_root"] = str(self.model_dir.resolve())
        os.environ["index_root"] = str(self.index_dir.resolve())
        os.environ["hubert_path"] = str(self.hubert_path.resolve())
        os.environ["rmvpe_root"] = str(self.rmvpe_dir.resolve())


        # Import after env vars are set.
        from rvc.modules.vc.modules import VC  # type: ignore
        from rvc.modules.vc.utils import load_hubert  # type: ignore
        import rvc.lib.rmvpe as rmvpe_lib

        self._VC = VC
        self.vc = VC()

        # Always load HuBERT and RMVPE at startup
        logger.info("Loading HuBERT model at startup...")
        self.vc.hubert_model = load_hubert(self.vc.config, str(self.hubert_path))
        self.vc.hubert_model = self.vc.hubert_model.to(self.vc.config.device)
        logger.info(f"HuBERT model loaded and moved to device: {self.vc.config.device}")

        logger.info("Loading RMVPE model at startup...")
        rmvpe_pt = self.rmvpe_dir / "rmvpe.pt"
        if rmvpe_pt.exists():
            self.vc.rmvpe_model = rmvpe_lib.RMVPE(str(rmvpe_pt), is_half=getattr(self.vc.config, "is_half", False), device=self.vc.config.device)
            logger.info("RMVPE model loaded and cached.")
        else:
            self.vc.rmvpe_model = None
            logger.warning(f"RMVPE model not found at {rmvpe_pt}, pitch extraction may fail.")

        # Expose loaded model state
        self.model_name: Optional[str] = None
        self.index_path: Optional[str] = None

        # Counter to avoid reusing the same cache key inside the pipeline.
        self._chunk_counter = 0

        self.default_params = default_params or RVCInferParams()

        # Always use CUDA if available, unless explicitly set to CPU
        import torch
        if device == "cuda" and torch.cuda.is_available():
            self.vc.config.device = "cuda"
        elif device == "cpu":
            self.vc.config.device = "cpu"
        else:
            # Fallback: auto-detect
            self.vc.config.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _resample_poly(self, y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Fast, high-quality resampling using polyphase filtering."""
        y = np.asarray(y, dtype=np.float32).flatten()
        if orig_sr == target_sr or y.size == 0:
            return y
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        # resample_poly expects float64 for best accuracy
        return signal.resample_poly(y.astype(np.float64), up, down).astype(np.float32)

    def load_model(self, model_path: str, index_path: Optional[str] = None) -> bool:
        """Load an RVC model (a .pth file exported by the WebUI).

        Args:
            model_path: Filename inside model_dir or a full/relative path.
            index_path: Optional .index file path for retrieval (recommended).

        Returns:
            True if successfully loaded.
        """
        try:
            p = Path(model_path)
            model_file = None
            logger.info(f"Attempting to load model: {model_path}")
            # If path exists as given, use it
            if p.exists():
                logger.info(f"Model file exists at given path: {p}")
                model_file = p
            else:
                # If absolute path but doesn't exist, try just the filename in model_dir
                if p.is_absolute():
                    candidate = self.model_dir / p.name
                    logger.info(f"Checking model_dir/name: {candidate}")
                    if candidate.exists():
                        logger.info(f"Model file found at model_dir/name: {candidate}")
                        model_file = candidate
                # Otherwise, try as relative to model_dir
                if model_file is None:
                    candidate = self.model_dir / model_path
                    logger.info(f"Checking model_dir/model_path: {candidate}")
                    if candidate.exists():
                        logger.info(f"Model file found at model_dir/model_path: {candidate}")
                        model_file = candidate
            if not model_file or not model_file.exists():
                logger.info(f"Falling back to rglob search for: {p.name} in {self.model_dir}")
                matches = list(self.model_dir.rglob(p.name))
                logger.info(f"rglob matches: {matches}")
                if matches:
                    # If we have the parent directory hint from the original path, prefer that match
                    # e.g., if model_path was "/storage/models/jokull-0.4/G_570.pth", prefer match in jokull-0.4
                    expected_parent = p.parent.name if p.parent.name else None
                    if expected_parent and len(matches) > 1:
                        for m in matches:
                            if expected_parent.lower() in str(m.parent).lower():
                                model_file = m
                                logger.info(f"Model file found by rglob (preferred parent '{expected_parent}'): {model_file}")
                                break
                    if not model_file:
                        model_file = matches[0]
                        logger.info(f"Model file found by rglob: {model_file}")
                else:
                    logger.error(f"Model file not found: {model_path} (tried: {model_file})")
                    return False

            # Load model into self.vc.net_g etc. (force GPU if device is cuda)
            import torch
            info = self.vc.get_vc(str(model_file))
            self.model_name = model_file.name

            # Auto-pick index if not provided: try matching by stem.
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
            else:
                logger.info("No index file configured (conversion will still work)")

            logger.info(f"Successfully loaded model: {self.model_name} ({info})")
            return True

        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            return False

    def infer(self, audio: np.ndarray, params: Optional[RVCInferParams] = None) -> np.ndarray:
        """Convert a mono audio chunk (float32, -1..1) using the loaded RVC model."""

        if self.model_name is None or self.vc.net_g is None:
            # Not loaded
            logger.warning("No model loaded, returning input as-is")
            return audio

        p = params or self.default_params

        # Defensive: ensure 1D float32
        x = np.asarray(audio, dtype=np.float32).flatten()

        # Keep level similar to WebUI
        audio_max = np.abs(x).max() / 0.95 if x.size else 0.0
        if audio_max > 1.0:
            x = x / audio_max

        # RVC expects the *input* waveform at 16 kHz for HuBERT features.
        # If your device/client runs at a different sample rate, we resample here.
        if self.input_sample_rate != 16000:
            x = self._resample_poly(x, orig_sr=self.input_sample_rate, target_sr=16000)

        # Validate required auxiliary assets
        hubert = Path(os.environ["hubert_path"])
        if not hubert.exists():
            logger.error(
                f"Missing HuBERT checkpoint: {hubert}. Put hubert_base.pt in assets/hubert/ (or set HUBERT_PATH)."
            )
            return audio

        if p.f0_method == "rmvpe":
            rmvpe_pt = Path(os.environ["rmvpe_root"]) / "rmvpe.pt"
            if not rmvpe_pt.exists():
                logger.error(
                    f"Missing RMVPE checkpoint: {rmvpe_pt}. Put rmvpe.pt in assets/rmvpe/ (or set RMVPE_DIR)."
                )
                return audio

        # Ensure hubert is loaded (lazy).
        if self.vc.hubert_model is None:
            from rvc.modules.vc.utils import load_hubert  # type: ignore

            logger.info("Loading HuBERT model for feature extraction...")
            self.vc.hubert_model = load_hubert(self.vc.config, os.environ["hubert_path"])

        # Run conversion through the pipeline.
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

            # Convert int16 -> float32
            y = audio_int16.astype(np.float32) / 32768.0
            return y

        except Exception as e:
            logger.exception(f"Error during inference: {e}")
            return audio

    def list_available_models(self) -> list[str]:
        """List .pth/.pt/.ckpt models in the model directory."""
        model_files = []
        for ext in ["*.pth", "*.pt", "*.ckpt"]:
            model_files.extend(self.model_dir.glob(ext))
        return [f.name for f in model_files]

    def unload_model(self):
        """Unload current model and free memory (except HuBERT and RMVPE)."""
        try:
            self.vc.net_g = None
            # Do NOT unload HuBERT or RMVPE
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
