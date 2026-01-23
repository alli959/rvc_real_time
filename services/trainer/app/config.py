"""
Trainer Service Configuration
Training hyperparameters and paths matching RVC WebUI defaults
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration matching RVC v2 defaults."""
    
    # Model architecture
    version: str = "v2"
    sample_rate: int = 48000
    
    # Training hyperparameters
    batch_size: int = 8
    total_epochs: int = 100
    save_every_epoch: int = 10
    
    # Learning rates
    learning_rate: float = 1e-4
    lr_decay: float = 0.999875  # Per step decay
    
    # Model dimensions
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.0
    
    # Speaker embedding
    spk_embed_dim: int = 109
    gin_channels: int = 256
    
    # F0 conditioning
    use_f0: bool = True
    
    # Augmentation
    aug_shift_max: int = 12  # Semitones for pitch augmentation
    
    # Gradient settings
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000


@dataclass
class PathConfig:
    """Path configuration for training."""
    
    # Base directories
    data_root: str = field(default_factory=lambda: os.getenv("DATA_ROOT", "/data"))
    models_root: str = field(default_factory=lambda: os.getenv("MODELS_ROOT", "/models"))
    
    # RVC assets (shared with voice-engine)
    rvc_root: str = field(default_factory=lambda: os.getenv("RVC_ROOT", "/app/rvc"))
    assets_root: str = field(default_factory=lambda: os.getenv("ASSETS_ROOT", "/app/assets"))
    
    # Pretrained models
    pretrained_g: str = ""
    pretrained_d: str = ""
    
    def __post_init__(self):
        """Set default pretrained model paths based on sample rate."""
        if not self.pretrained_g:
            self.pretrained_g = os.path.join(
                self.assets_root, 
                "pretrained_v2",
                "f0G48k.pth"
            )
        if not self.pretrained_d:
            self.pretrained_d = os.path.join(
                self.assets_root,
                "pretrained_v2", 
                "f0D48k.pth"
            )
    
    def get_experiment_dir(self, exp_name: str) -> str:
        """Get experiment directory path."""
        return os.path.join(self.data_root, exp_name)
    
    def get_logs_dir(self, exp_name: str) -> str:
        """Get training logs directory."""
        return os.path.join(self.data_root, exp_name, "logs")
    
    def get_weights_dir(self, exp_name: str) -> str:
        """Get model weights directory."""
        return os.path.join(self.data_root, exp_name, "weights")
    
    def get_output_model_path(self, exp_name: str) -> str:
        """Get final output model path."""
        return os.path.join(self.models_root, exp_name, f"{exp_name}.pth")
    
    def get_index_path(self, exp_name: str) -> str:
        """Get FAISS index path."""
        return os.path.join(self.models_root, exp_name, f"{exp_name}.index")


@dataclass
class Settings:
    """Combined settings for trainer service."""
    
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Service settings
    host: str = field(default_factory=lambda: os.getenv("TRAINER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("TRAINER_PORT", "8002")))
    workers: int = field(default_factory=lambda: int(os.getenv("TRAINER_WORKERS", "1")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Preprocessor service URL (for validation)
    preprocessor_url: str = field(
        default_factory=lambda: os.getenv("PREPROCESSOR_URL", "http://preprocess:8003")
    )


# Global settings instance
settings = Settings()
