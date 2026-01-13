"""
API Routers

FastAPI routers organized by domain.
"""

from app.routers.health import router as health_router
from app.routers.tts import router as tts_router
from app.routers.conversion import router as conversion_router
from app.routers.youtube import router as youtube_router
from app.routers.audio import router as audio_router

__all__ = [
    "health_router",
    "tts_router",
    "conversion_router",
    "youtube_router",
    "audio_router",
]
