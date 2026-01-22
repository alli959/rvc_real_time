"""
Preprocessor Service - Configuration

Environment-based configuration for the preprocessor service.
"""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""
    
    # Server settings
    http_port: int = int(os.getenv("HTTP_PORT", "8003"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Directory paths
    models_dir: str = os.getenv("MODELS_DIR", "/app/assets/models")
    uploads_dir: str = os.getenv("UPLOADS_DIR", "/app/uploads")
    
    # Asset paths
    hubert_path: str = os.getenv("HUBERT_PATH", "/app/assets/hubert/hubert_base.pt")
    rmvpe_path: str = os.getenv("RMVPE_PATH", "/app/assets/rmvpe")
    
    # Processing settings
    device: str = os.getenv("DEVICE", "cuda:0")
    n_threads: int = int(os.getenv("N_THREADS", "4"))
    
    # Preprocessing defaults (from WebUI reference)
    default_sample_rate: int = 48000
    default_version: str = "v2"
    
    # Slicer parameters (matching WebUI exactly)
    slicer_threshold_db: float = -42.0  # Silence threshold in dB
    slicer_min_length_ms: int = 1500    # Minimum clip length (1.5s)
    slicer_min_interval_ms: int = 400   # Minimum silence for split (400ms)
    slicer_hop_size_ms: int = 15        # Frame hop (15ms)
    slicer_max_sil_kept_ms: int = 500   # Max silence to keep (500ms)
    
    # Chunk parameters (matching WebUI)
    chunk_length_sec: float = 3.7       # Chunk length in seconds
    chunk_overlap_sec: float = 0.3      # Overlap between chunks
    
    # Normalization parameters
    norm_max: float = 0.9               # Target max amplitude
    norm_alpha: float = 0.75            # Normalization alpha
    norm_clip_threshold: float = 2.5    # Filter out clips above this
    
    # High-pass filter (matching WebUI: butter N=5, Wn=48Hz)
    highpass_order: int = 5
    highpass_cutoff: int = 48           # Hz
    
    # F0 extraction
    f0_bin: int = 256
    f0_max: float = 1100.0
    f0_min: float = 50.0
    
    # Feature extraction
    feature_dim_v1: int = 256
    feature_dim_v2: int = 768
    
    def __post_init__(self):
        """Ensure directories exist."""
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.uploads_dir).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
