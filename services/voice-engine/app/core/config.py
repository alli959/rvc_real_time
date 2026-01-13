"""Configuration Module - Handles application configuration.

RVC inference uses a few auxiliary assets in addition to the .pth weight file:
- HuBERT checkpoint (hubert_base.pt) for content features
- RMVPE checkpoint (rmvpe.pt) for pitch extraction (f0)
- Optional .index file for retrieval-based enhancement

All of these are configurable via environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 16000
    chunk_size: int = 1024
    overlap: int = 0
    channels: int = 1

    @classmethod
    def from_env(cls) -> "AudioConfig":
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
    def from_env(cls) -> "ModelConfig":
        """Load configuration from environment variables."""
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

    http_host: str = "0.0.0.0"
    http_port: int = 8001
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765
    socket_host: str = "0.0.0.0"
    socket_port: int = 9876

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        return cls(
            http_host=os.getenv("HTTP_HOST", "0.0.0.0"),
            http_port=int(os.getenv("HTTP_PORT", 8001)),
            websocket_host=os.getenv("WEBSOCKET_HOST", "0.0.0.0"),
            websocket_port=int(os.getenv("WEBSOCKET_PORT", 8765)),
            socket_host=os.getenv("SOCKET_HOST", "0.0.0.0"),
            socket_port=int(os.getenv("SOCKET_PORT", 9876)),
        )


@dataclass
class TTSConfig:
    """TTS service configuration."""
    
    bark_model_dir: str = "assets/bark"
    use_bark_by_default: bool = True
    edge_tts_voice: str = "en-US-GuyNeural"
    
    @classmethod
    def from_env(cls) -> "TTSConfig":
        """Load configuration from environment variables."""
        return cls(
            bark_model_dir=os.getenv("BARK_MODEL_DIR", "assets/bark"),
            use_bark_by_default=os.getenv("USE_BARK", "true").lower() == "true",
            edge_tts_voice=os.getenv("EDGE_TTS_VOICE", "en-US-GuyNeural"),
        )


@dataclass
class AppConfig:
    """Main application configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    mode: str = "streaming"  # streaming, local, api
    log_level: str = "INFO"
    debug: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            audio=AudioConfig.from_env(),
            model=ModelConfig.from_env(),
            server=ServerConfig.from_env(),
            tts=TTSConfig.from_env(),
            mode=os.getenv("APP_MODE", "streaming"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )

    @classmethod
    def default(cls) -> "AppConfig":
        """Get default configuration."""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "chunk_size": self.audio.chunk_size,
                "channels": self.audio.channels,
            },
            "model": {
                "device": self.model.device,
                "f0_method": self.model.f0_method,
            },
            "server": {
                "http_port": self.server.http_port,
                "websocket_port": self.server.websocket_port,
            },
            "mode": self.mode,
            "log_level": self.log_level,
        }


# Global config instance (lazy loaded)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
