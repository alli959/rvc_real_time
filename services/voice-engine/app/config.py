"""Configuration Module - Handles application configuration.

RVC inference uses a few auxiliary assets in addition to the .pth weight file:
- HuBERT checkpoint (hubert_base.pt) for content features
- RMVPE checkpoint (rmvpe.pt) for pitch extraction (f0)
- Optional .index file for retrieval-based enhancement

All of these are configurable via environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 16000
    chunk_size: int = 1024
    overlap: int = 0
    channels: int = 1

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", 16000)),
            chunk_size=int(os.getenv("AUDIO_CHUNK_SIZE", 1024)),
            overlap=int(os.getenv("AUDIO_OVERLAP", 0)),
            channels=int(os.getenv("AUDIO_CHANNELS", 1)),
        )


@dataclass
class ModelConfig:
    """Model & inference configuration."""

    model_dir: str = "assets/models"
    index_dir: str = "assets/index"
    hubert_path: str = "assets/hubert/hubert_base.pt"
    rmvpe_dir: str = "assets/rmvpe"

    default_model: Optional[str] = None
    default_index: Optional[str] = None

    # Device selection for torch
    device: str = "cuda"  # force GPU usage by default

    # Inference defaults
    f0_method: str = "rmvpe"
    f0_up_key: int = 0
    index_rate: float = 0.75
    filter_radius: int = 3
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    resample_sr: int = 16000

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        # Force device to cuda unless overridden by env
        device_env = os.getenv("DEVICE", "cuda")
        return cls(
            model_dir=os.getenv("MODEL_DIR", "assets/models"),
            index_dir=os.getenv("INDEX_DIR", "assets/index"),
            hubert_path=os.getenv("HUBERT_PATH", "assets/hubert/hubert_base.pt"),
            rmvpe_dir=os.getenv("RMVPE_DIR", "assets/rmvpe"),
            default_model=os.getenv("DEFAULT_MODEL"),
            default_index=os.getenv("DEFAULT_INDEX"),
            device=device_env,
            f0_method=os.getenv("F0_METHOD", "rmvpe"),
            f0_up_key=int(os.getenv("F0_UP_KEY", 0)),
            index_rate=float(os.getenv("INDEX_RATE", 0.75)),
            filter_radius=int(os.getenv("FILTER_RADIUS", 3)),
            rms_mix_rate=float(os.getenv("RMS_MIX_RATE", 0.25)),
            protect=float(os.getenv("PROTECT", 0.33)),
            resample_sr=int(os.getenv("RESAMPLE_SR", int(os.getenv("AUDIO_SAMPLE_RATE", 16000)))),
        )


@dataclass
class ServerConfig:
    """Server configuration."""

    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765
    socket_host: str = "0.0.0.0"
    socket_port: int = 9876

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            websocket_host=os.getenv("WEBSOCKET_HOST", "0.0.0.0"),
            websocket_port=int(os.getenv("WEBSOCKET_PORT", 8765)),
            socket_host=os.getenv("SOCKET_HOST", "0.0.0.0"),
            socket_port=int(os.getenv("SOCKET_PORT", 9876)),
        )


@dataclass
class AppConfig:
    """Main application configuration."""

    audio: AudioConfig
    model: ModelConfig
    server: ServerConfig
    mode: str = "streaming"  # streaming, local, api
    log_level: str = "INFO"

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            audio=AudioConfig.from_env(),
            model=ModelConfig.from_env(),
            server=ServerConfig.from_env(),
            mode=os.getenv("APP_MODE", "streaming"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    @classmethod
    def default(cls):
        """Get default configuration."""
        return cls(audio=AudioConfig(), model=ModelConfig(), server=ServerConfig())
