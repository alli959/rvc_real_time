"""
Unified Storage Paths Module

This module provides centralized path configuration for all Python services.
All paths are derived from STORAGE_ROOT environment variable with sensible defaults.

Usage:
    from storage_paths import StoragePaths
    
    paths = StoragePaths()
    model_dir = paths.models
    hubert_path = paths.hubert
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class StoragePaths:
    """
    Centralized storage path configuration.
    
    All paths are derived from STORAGE_ROOT with the following structure:
    
    /storage/
    ├── logs/<service>/       # Service-specific logs
    ├── data/
    │   ├── uploads/          # Uploaded audio files
    │   ├── preprocess/       # Preprocessing artifacts
    │   ├── training/         # Training checkpoints
    │   └── outputs/          # Generated outputs
    ├── assets/
    │   ├── hubert/           # HuBERT model
    │   ├── rmvpe/            # RMVPE model
    │   ├── pretrained_v2/    # RVC pretrained models
    │   ├── uvr5_weights/     # Vocal separation models
    │   ├── bark/             # Bark TTS models
    │   └── whisper/          # Whisper models
    └── models/               # User voice models
    """
    
    # Base paths - can be overridden via environment
    storage_root: str = field(default_factory=lambda: os.getenv("STORAGE_ROOT", "/storage"))
    
    # Service name for logs (set by each service)
    service_name: str = "default"
    
    def __post_init__(self):
        """Ensure all directories exist."""
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Create required directories if they don't exist."""
        dirs = [
            self.logs,
            self.uploads,
            self.preprocess,
            self.training,
            self.outputs,
            self.models,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Computed paths
    # =========================================================================
    
    @property
    def logs(self) -> str:
        """Service-specific logs directory."""
        return os.path.join(self.storage_root, "logs", self.service_name)
    
    @property
    def data(self) -> str:
        """Data root directory."""
        return os.path.join(self.storage_root, "data")
    
    @property
    def uploads(self) -> str:
        """Uploaded audio files directory."""
        return os.path.join(self.data, "uploads")
    
    @property
    def preprocess(self) -> str:
        """Preprocessing artifacts directory."""
        return os.path.join(self.data, "preprocess")
    
    @property
    def training(self) -> str:
        """Training artifacts directory."""
        return os.path.join(self.data, "training")
    
    @property
    def outputs(self) -> str:
        """Generated outputs directory."""
        return os.path.join(self.data, "outputs")
    
    @property
    def assets(self) -> str:
        """Shared assets directory."""
        return os.path.join(self.storage_root, "assets")
    
    @property
    def models(self) -> str:
        """User voice models directory."""
        return os.path.join(self.storage_root, "models")
    
    # =========================================================================
    # Asset paths
    # =========================================================================
    
    @property
    def hubert(self) -> str:
        """HuBERT model path."""
        # Allow override via environment
        return os.getenv("HUBERT_PATH", os.path.join(self.assets, "hubert", "hubert_base.pt"))
    
    @property
    def hubert_dir(self) -> str:
        """HuBERT directory."""
        return os.path.join(self.assets, "hubert")
    
    @property
    def rmvpe(self) -> str:
        """RMVPE model directory."""
        return os.getenv("RMVPE_PATH", os.path.join(self.assets, "rmvpe"))
    
    @property
    def pretrained_v2(self) -> str:
        """RVC pretrained models directory."""
        return os.path.join(self.assets, "pretrained_v2")
    
    @property
    def uvr5_weights(self) -> str:
        """UVR5 vocal separation weights directory."""
        return os.path.join(self.assets, "uvr5_weights")
    
    @property
    def bark(self) -> str:
        """Bark TTS models directory."""
        return os.path.join(self.assets, "bark")
    
    @property
    def whisper(self) -> str:
        """Whisper models directory."""
        return os.path.join(self.assets, "whisper")
    
    @property
    def index(self) -> str:
        """FAISS index files directory."""
        return os.path.join(self.assets, "index")
    
    # =========================================================================
    # Experiment paths
    # =========================================================================
    
    def get_experiment_dir(self, exp_name: str) -> str:
        """Get experiment directory for preprocessing/training."""
        return os.path.join(self.preprocess, exp_name)
    
    def get_upload_dir(self, exp_name: str) -> str:
        """Get upload directory for an experiment."""
        return os.path.join(self.uploads, exp_name)
    
    def get_training_dir(self, exp_name: str) -> str:
        """Get training directory for an experiment."""
        return os.path.join(self.training, exp_name)
    
    def get_model_path(self, model_name: str) -> str:
        """Get model .pth file path."""
        return os.path.join(self.models, f"{model_name}.pth")
    
    def get_index_path(self, model_name: str) -> str:
        """Get model .index file path."""
        return os.path.join(self.models, f"{model_name}.index")
    
    def get_model_metadata_dir(self, model_name: str) -> str:
        """Get model metadata directory (for config, images, etc.)."""
        return os.path.join(self.models, model_name)
    
    def get_model_image_path(self, model_name: str, ext: str = "png") -> str:
        """Get model image path."""
        return os.path.join(self.get_model_metadata_dir(model_name), f"image.{ext}")
    
    def get_output_dir(self, model_name: str) -> str:
        """Get output directory for a model."""
        return os.path.join(self.outputs, model_name)
    
    # =========================================================================
    # Pretrained model paths
    # =========================================================================
    
    def get_pretrained_g(self, sample_rate: int = 48000) -> str:
        """Get pretrained generator path."""
        return os.path.join(self.pretrained_v2, f"f0G{sample_rate // 1000}k.pth")
    
    def get_pretrained_d(self, sample_rate: int = 48000) -> str:
        """Get pretrained discriminator path."""
        return os.path.join(self.pretrained_v2, f"f0D{sample_rate // 1000}k.pth")
    
    # =========================================================================
    # Log file paths
    # =========================================================================
    
    def get_log_file(self, name: str = "app.log") -> str:
        """Get log file path."""
        return os.path.join(self.logs, name)
    
    # =========================================================================
    # Validation
    # =========================================================================
    
    def validate_assets(self) -> dict:
        """Check if required assets exist."""
        return {
            "hubert": os.path.exists(self.hubert),
            "rmvpe": os.path.exists(self.rmvpe),
            "pretrained_g_48k": os.path.exists(self.get_pretrained_g(48000)),
            "pretrained_d_48k": os.path.exists(self.get_pretrained_d(48000)),
        }
    
    def to_dict(self) -> dict:
        """Export all paths as a dictionary."""
        return {
            "storage_root": self.storage_root,
            "logs": self.logs,
            "data": self.data,
            "uploads": self.uploads,
            "preprocess": self.preprocess,
            "training": self.training,
            "outputs": self.outputs,
            "assets": self.assets,
            "models": self.models,
            "hubert": self.hubert,
            "rmvpe": self.rmvpe,
            "pretrained_v2": self.pretrained_v2,
        }


# Singleton instances for each service
_instances = {}


def get_storage_paths(service_name: str = "default") -> StoragePaths:
    """
    Get or create a StoragePaths instance for a service.
    
    Args:
        service_name: Name of the service (used for log subdirectory)
    
    Returns:
        StoragePaths instance
    """
    if service_name not in _instances:
        _instances[service_name] = StoragePaths(service_name=service_name)
    return _instances[service_name]


# Legacy compatibility - create default paths
def get_paths() -> StoragePaths:
    """Legacy function - use get_storage_paths() instead."""
    return get_storage_paths()


# For direct import
paths = get_storage_paths()
