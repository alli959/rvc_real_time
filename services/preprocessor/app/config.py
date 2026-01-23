"""
Preprocessor Service - Configuration

Environment-based configuration for the preprocessor service.

IMPORTANT: Path Configuration
=============================
- DATA_ROOT: Writable directory for experiment outputs (preprocessing artifacts)
  Default: /data - shared volume with trainer service
  Contains: <exp_name>/0_gt_wavs, 1_16k_wavs, 2a_f0, 2b_f0nsf, 3_feature768
  
- UPLOADS_DIR: Where uploaded audio files are stored
  Default: /data/uploads

- ASSETS_ROOT: Read-only shared assets directory  
  Default: /app/assets
  Contains: hubert/, rmvpe/, pretrained_v2/ (from voice-engine)

The preprocessor writes to DATA_ROOT, which is mounted as a shared volume
between preprocessor and trainer services. The trainer validates and reads
from the same DATA_ROOT.
"""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""
    
    # Server settings
    http_port: int = int(os.getenv("HTTP_PORT", os.getenv("PREPROCESS_PORT", "8003")))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Directory paths - UNIFIED with trainer service
    # DATA_ROOT is where preprocessing outputs go (shared with trainer)
    data_root: str = os.getenv("DATA_ROOT", "/data")
    
    # UPLOADS_DIR is where uploaded training audio is stored
    uploads_dir: str = os.getenv("UPLOADS_DIR", os.getenv("DATA_ROOT", "/data") + "/uploads")
    
    # ASSETS_ROOT is for read-only shared assets (hubert, rmvpe, etc)
    assets_root: str = os.getenv("ASSETS_ROOT", "/app/assets")
    
    # Legacy compatibility: models_dir now points to data_root for experiment outputs
    # Final trained models go to models_root (handled by trainer)
    @property
    def models_dir(self) -> str:
        """Experiment output directory (for preprocessing artifacts)."""
        return self.data_root
    
    # Asset paths (read-only)
    @property
    def hubert_path(self) -> str:
        return os.getenv("HUBERT_PATH", f"{self.assets_root}/hubert/hubert_base.pt")
    
    @property
    def rmvpe_path(self) -> str:
        return os.getenv("RMVPE_PATH", f"{self.assets_root}/rmvpe")
    
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
        """Ensure writable directories exist."""
        # Only create directories in DATA_ROOT (writable volume)
        Path(self.data_root).mkdir(parents=True, exist_ok=True)
        Path(self.uploads_dir).mkdir(parents=True, exist_ok=True)
        
        # Don't try to create directories in assets_root (read-only)


# Global settings instance
settings = Settings()
