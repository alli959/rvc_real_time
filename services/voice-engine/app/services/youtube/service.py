"""
YouTube Service

Search and download audio from YouTube using yt-dlp.
"""

import asyncio
import io
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """YouTube search result."""
    id: str
    title: str
    artist: str
    duration: int  # seconds
    thumbnail: str
    url: str
    view_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


async def search_youtube(
    query: str,
    max_results: int = 10
) -> List[SearchResult]:
    """
    Search YouTube for songs matching the query.
    
    Args:
        query: Search query (artist, song name, etc.)
        max_results: Maximum number of results to return
        
    Returns:
        List of SearchResult objects
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp not installed. Install with: pip install yt-dlp")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'default_search': 'ytsearch',
        'max_downloads': max_results,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        },
    }
    
    results = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            )
            
            if not search_results or 'entries' not in search_results:
                return []
            
            for entry in search_results.get('entries', []):
                if not entry:
                    continue
                
                # Parse artist from title
                title = entry.get('title', 'Unknown')
                artist = entry.get('uploader', entry.get('channel', 'Unknown Artist'))
                
                if ' - ' in title:
                    parts = title.split(' - ', 1)
                    artist = parts[0].strip()
                    title = parts[1].strip()
                
                results.append(SearchResult(
                    id=entry.get('id', ''),
                    title=title,
                    artist=artist,
                    duration=entry.get('duration', 0) or 0,
                    thumbnail=entry.get('thumbnail', '') or '',
                    url=entry.get('url', '') or entry.get('webpage_url', ''),
                    view_count=entry.get('view_count', 0) or 0,
                ))
        
        logger.info(f"Found {len(results)} results for query: {query}")
        return results
        
    except Exception as e:
        logger.exception(f"YouTube search failed: {e}")
        raise


async def download_youtube_audio(
    video_id: str,
    use_cache: bool = True
) -> Tuple[bytes, int]:
    """
    Download audio from YouTube video.
    
    Args:
        video_id: YouTube video ID
        use_cache: Whether to use cached audio if available
        
    Returns:
        Tuple of (audio_bytes, sample_rate)
    """
    try:
        import yt_dlp
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise ImportError(f"Required package not installed: {e}")
    
    from app.services.youtube.cache import (
        get_cache_path, 
        get_cached_file, 
        is_cached, 
        clean_cache
    )
    
    # Check cache first
    if use_cache and is_cached(video_id):
        logger.info(f"Using cached audio for {video_id}")
        cached_file = get_cached_file(video_id)
        audio, sr = librosa.load(str(cached_file), sr=None, mono=True)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sr, format='WAV')
        wav_buffer.seek(0)
        return wav_buffer.read(), sr
    
    # Download audio
    url = f"https://www.youtube.com/watch?v={video_id}"
    cache_path = get_cache_path(video_id, "wav")
    
    logger.info(f"Downloading audio for {video_id}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
        'outtmpl': str(cache_path.with_suffix('.%(ext)s')),
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'retries': 5,
        'fragment_retries': 5,
        'file_access_retries': 5,
        'ignoreerrors': False,
        'geo_bypass': True,
        'cookiefile': os.environ.get('YOUTUBE_COOKIES_FILE'),
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ydl.download([url])
            )
        
        clean_cache()
        
        # Find downloaded file
        downloaded_file = None
        for ext in ['.wav', '.mp3', '.m4a', '.webm', '.opus', '.ogg']:
            potential_file = cache_path.with_suffix(ext)
            if potential_file.exists():
                downloaded_file = potential_file
                break
        
        if downloaded_file is None:
            raise FileNotFoundError(f"Downloaded file not found for: {video_id}")
        
        audio, sr = librosa.load(str(downloaded_file), sr=None, mono=True)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sr, format='WAV')
        wav_buffer.seek(0)
        
        logger.info(f"Downloaded audio for {video_id}: {len(audio)} samples at {sr}Hz")
        return wav_buffer.read(), sr
        
    except Exception as e:
        error_str = str(e)
        logger.warning(f"Download failed: {e}")
        
        # Clean up partial downloads
        for ext in ['.wav', '.mp3', '.m4a', '.webm', '.mp4', '.opus', '.ogg', '.part']:
            temp_file = cache_path.with_suffix(ext)
            if temp_file.exists():
                temp_file.unlink()
        
        if "403" in error_str:
            raise Exception(
                "This video is blocked from downloading. YouTube may be restricting access. "
                "Try a different song or use a popular/official music video."
            )
        elif "not available" in error_str.lower():
            raise Exception(
                "No downloadable format available for this video. "
                "Try a different version of the song."
            )
        else:
            raise Exception(f"Download failed: {e}")


async def get_video_info(video_id: str) -> Dict:
    """
    Get detailed information about a YouTube video.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Dict with video information
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp not installed")
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ydl.extract_info(url, download=False)
            )
            
            title = info.get('title', 'Unknown')
            artist = info.get('uploader', info.get('channel', 'Unknown Artist'))
            
            if ' - ' in title:
                parts = title.split(' - ', 1)
                artist = parts[0].strip()
                title = parts[1].strip()
            
            return {
                'id': video_id,
                'title': title,
                'artist': artist,
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', ''),
                'description': info.get('description', ''),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', ''),
            }
            
    except Exception as e:
        logger.exception(f"Failed to get video info: {e}")
        raise


class YouTubeService:
    """
    YouTube audio service for search and download.
    
    Usage:
        yt = YouTubeService()
        results = await yt.search("Daft Punk Get Lucky")
        audio, sr = await yt.download(results[0].id)
    """
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for videos."""
        return await search_youtube(query, max_results)
    
    async def download(self, video_id: str) -> Tuple[bytes, int]:
        """Download audio from video."""
        return await download_youtube_audio(video_id, self.use_cache)
    
    async def get_info(self, video_id: str) -> Dict:
        """Get video information."""
        return await get_video_info(video_id)
    
    def search_sync(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Synchronous search wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.search(query, max_results)
        )
    
    def download_sync(self, video_id: str) -> Tuple[bytes, int]:
        """Synchronous download wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.download(video_id)
        )
