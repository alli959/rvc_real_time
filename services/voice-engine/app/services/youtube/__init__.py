"""
YouTube Service Module

Search and download audio from YouTube using yt-dlp.
"""

from app.services.youtube.service import (
    YouTubeService,
    SearchResult,
    search_youtube,
    download_youtube_audio,
    get_video_info,
)
from app.services.youtube.cache import (
    AudioCache,
    get_cache_path,
    is_cached,
    clean_cache,
)

__all__ = [
    # Main service
    "YouTubeService",
    "SearchResult",
    # Functions
    "search_youtube",
    "download_youtube_audio",
    "get_video_info",
    # Cache
    "AudioCache",
    "get_cache_path",
    "is_cached",
    "clean_cache",
]
