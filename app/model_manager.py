"""
Model Manager Module - Handles RVC model loading and inference
<<<<<<< HEAD
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging
=======

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
>>>>>>> 2c01fa3 (feat(rvc): run WebUI-trained RVC models in realtime pipeline)

logger = logging.getLogger(__name__)


<<<<<<< HEAD
class ModelManager:
    """Manages RVC model loading and inference"""
    
    def __init__(self, model_dir: str = "assets/models"):
        """
        Initialize model manager
        
        Args:
            model_dir: Directory containing RVC models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_model: Optional[torch.nn.Module] = None
        self.model_name: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load an RVC model from file
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_file = self.model_dir / model_path
            
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return False
            
            # Load model checkpoint
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # For demonstration, we'll use a placeholder model structure
            # In production, this would load the actual RVC model architecture
            self.current_model = self._create_model_from_checkpoint(checkpoint)
            self.model_name = model_path
            
            logger.info(f"Successfully loaded model: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _create_model_from_checkpoint(self, checkpoint: Dict[str, Any]) -> torch.nn.Module:
        """
        Create model from checkpoint
        
        Args:
            checkpoint: Model checkpoint dictionary
            
        Returns:
            Model instance
        """
        # Placeholder for actual RVC model architecture
        # In production, this would instantiate the proper model class
        class PlaceholderRVCModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(1, 1, kernel_size=3, padding=1)
            
            def forward(self, x):
                # Simple passthrough for demonstration
                return x
        
        model = PlaceholderRVCModel()
        
        # Load state dict if available
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                logger.warning("Could not load model state dict, using random initialization")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def infer(self, audio_features: np.ndarray) -> np.ndarray:
        """
        Perform inference on audio features
        
        Args:
            audio_features: Input audio features
            
        Returns:
            Converted audio features
        """
        if self.current_model is None:
            logger.warning("No model loaded, returning input as-is")
            return audio_features
        
        try:
            # Convert to tensor
            input_tensor = torch.from_numpy(audio_features).float()
            
            # Add batch dimension if needed
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            elif input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
            
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output_tensor = self.current_model(input_tensor)
            
            # Convert back to numpy
            output = output_tensor.cpu().numpy().squeeze()
            
            return output
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return audio_features
    
    def list_available_models(self) -> list:
        """
        List all available models in the model directory
        
        Returns:
            List of model filenames
        """
        model_files = []
        for ext in ['*.pth', '*.pt', '*.ckpt']:
            model_files.extend(self.model_dir.glob(ext))
        
        return [f.name for f in model_files]
    
    def unload_model(self):
        """Unload current model and free memory"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.model_name = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")
=======
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

        self._VC = VC
        self.vc = VC()

        # Expose loaded model state
        self.model_name: Optional[str] = None
        self.index_path: Optional[str] = None

        # Counter to avoid reusing the same cache key inside the pipeline.
        self._chunk_counter = 0

        self.default_params = default_params or RVCInferParams()

        if device != "auto":
            # Override device selection in the underlying config
            self.vc.config.device = device

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
            model_file = p if p.exists() else (self.model_dir / model_path)
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return False

            # Load model into self.vc.net_g etc.
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
        """Unload current model and free memory."""
        try:
            self.vc.net_g = None
            self.vc.hubert_model = None
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

        logger.info("Model unloaded")
>>>>>>> 2c01fa3 (feat(rvc): run WebUI-trained RVC models in realtime pipeline)
