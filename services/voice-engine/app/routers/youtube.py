"""
YouTube Router

YouTube search and download endpoints.
"""

import io
import base64
import logging

from fastapi import APIRouter, HTTPException

from app.models.youtube import (
    YouTubeSearchRequest,
    YouTubeSearchResponse,
    YouTubeDownloadRequest,
    YouTubeDownloadResponse,
)
from app.services.youtube import (
    YouTubeService,
    search_youtube,
    download_youtube_audio,
    get_video_info,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/youtube", tags=["youtube"])


@router.post("/search", response_model=YouTubeSearchResponse)
async def search(request: YouTubeSearchRequest):
    """
    Search YouTube for songs.
    
    Returns list of matching videos with title, artist, duration, etc.
    """
    try:
        results = await search_youtube(
            query=request.query,
            max_results=request.max_results,
        )
        
        return YouTubeSearchResponse(
            results=[r.to_dict() for r in results],
            count=len(results),
            query=request.query,
        )
        
    except Exception as e:
        logger.exception(f"YouTube search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download", response_model=YouTubeDownloadResponse)
async def download(request: YouTubeDownloadRequest):
    """
    Download audio from YouTube video.
    
    Returns base64-encoded WAV audio.
    Uses caching to speed up repeated downloads.
    """
    try:
        audio_bytes, sample_rate = await download_youtube_audio(
            video_id=request.video_id,
            use_cache=request.use_cache if hasattr(request, 'use_cache') else True,
        )
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return YouTubeDownloadResponse(
            audio=audio_base64,
            sample_rate=sample_rate,
            format="wav",
            video_id=request.video_id,
        )
        
    except Exception as e:
        logger.exception(f"YouTube download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{video_id}")
async def video_info(video_id: str):
    """
    Get information about a YouTube video.
    
    Returns title, artist, duration, thumbnail, etc.
    """
    try:
        info = await get_video_info(video_id)
        return info
        
    except Exception as e:
        logger.exception(f"Failed to get video info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def cache_stats():
    """Get YouTube audio cache statistics."""
    from app.services.youtube.cache import get_cache_stats
    return get_cache_stats()


@router.post("/cache/clear")
async def clear_cache():
    """Clear the YouTube audio cache."""
    from app.services.youtube.cache import clear_cache
    clear_cache()
    return {"status": "cleared"}
