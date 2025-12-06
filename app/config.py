"""
Configuration Module - Handles application configuration
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    overlap: int = 256
    channels: int = 1
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            sample_rate=int(os.getenv('AUDIO_SAMPLE_RATE', 16000)),
            chunk_size=int(os.getenv('AUDIO_CHUNK_SIZE', 1024)),
            overlap=int(os.getenv('AUDIO_OVERLAP', 256)),
            channels=int(os.getenv('AUDIO_CHANNELS', 1))
        )


@dataclass
class ModelConfig:
    """Model configuration"""
    model_dir: str = "assets/models"
    default_model: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            model_dir=os.getenv('MODEL_DIR', 'assets/models'),
            default_model=os.getenv('DEFAULT_MODEL'),
            device=os.getenv('DEVICE', 'auto')
        )


@dataclass
class ServerConfig:
    """Server configuration"""
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765
    socket_host: str = "0.0.0.0"
    socket_port: int = 9876
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            websocket_host=os.getenv('WEBSOCKET_HOST', '0.0.0.0'),
            websocket_port=int(os.getenv('WEBSOCKET_PORT', 8765)),
            socket_host=os.getenv('SOCKET_HOST', '0.0.0.0'),
            socket_port=int(os.getenv('SOCKET_PORT', 9876))
        )


@dataclass
class AppConfig:
    """Main application configuration"""
    audio: AudioConfig
    model: ModelConfig
    server: ServerConfig
    mode: str = "streaming"  # streaming, local, api
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            audio=AudioConfig.from_env(),
            model=ModelConfig.from_env(),
            server=ServerConfig.from_env(),
            mode=os.getenv('APP_MODE', 'streaming'),
            log_level=os.getenv('LOG_LEVEL', 'INFO')
        )
    
    @classmethod
    def default(cls):
        """Get default configuration"""
        return cls(
            audio=AudioConfig(),
            model=ModelConfig(),
            server=ServerConfig()
        )
