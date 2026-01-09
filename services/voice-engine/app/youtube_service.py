"""
YouTube Audio Search and Download Service

Provides functionality to search for songs and download audio from YouTube
using yt-dlp for integration with vocal splitter/swap features.
"""

import asyncio
import logging
import os
import tempfile
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Cache directory for downloaded audio
CACHE_DIR = Path(os.environ.get("YOUTUBE_CACHE_DIR", "/tmp/youtube_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Maximum cached files
MAX_CACHE_FILES = 50


@dataclass
class SearchResult:
    """YouTube search result"""
    id: str
    title: str
    artist: str
    duration: int  # seconds
    thumbnail: str
    url: str
    view_count: int = 0


def get_cache_path(video_id: str) -> Path:
    """Get cache file path for a video ID"""
    return CACHE_DIR / f"{video_id}.mp3"


def is_cached(video_id: str) -> bool:
    """Check if audio is already cached"""
    return get_cache_path(video_id).exists()


def clean_cache():
    """Clean old cache files if limit exceeded"""
    try:
        files = list(CACHE_DIR.glob("*.mp3"))
        if len(files) > MAX_CACHE_FILES:
            # Sort by modification time, remove oldest
            files.sort(key=lambda f: f.stat().st_mtime)
            for f in files[:len(files) - MAX_CACHE_FILES]:
                f.unlink()
                logger.info(f"Removed old cache file: {f}")
    except Exception as e:
        logger.warning(f"Cache cleanup failed: {e}")


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
    
    results = []
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'default_search': 'ytsearch',
        'max_downloads': max_results,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        },
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search YouTube
            search_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            )
            
            if not search_results or 'entries' not in search_results:
                return []
            
            for entry in search_results.get('entries', []):
                if not entry:
                    continue
                    
                # Parse artist from title if possible
                title = entry.get('title', 'Unknown')
                artist = entry.get('uploader', entry.get('channel', 'Unknown Artist'))
                
                # Try to extract artist from title (common format: "Artist - Song")
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
        import io
        import numpy as np
    except ImportError as e:
        raise ImportError(f"Required package not installed: {e}")
    
    cache_path = get_cache_path(video_id)
    
    # Check cache first
    if use_cache and cache_path.exists():
        logger.info(f"Using cached audio for {video_id}")
        audio, sr = librosa.load(str(cache_path), sr=None, mono=True)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sr, format='WAV')
        wav_buffer.seek(0)
        return wav_buffer.read(), sr
    
    # Download audio
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    logger.info(f"Downloading audio for {video_id}")
    
    ydl_opts = {
        # Let yt-dlp auto-select the best format
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Convert to WAV for consistency
            'preferredquality': '0',  # Best quality (lossless for WAV)
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
        # Use cookies if available
        'cookiefile': os.environ.get('YOUTUBE_COOKIES_FILE'),
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ydl.download([url])
            )
        
        # Clean old cache files
        clean_cache()
        
        # Find the downloaded file (could be various extensions)
        downloaded_file = None
        for ext in ['.wav', '.mp3', '.m4a', '.webm', '.opus', '.ogg']:
            potential_file = cache_path.with_suffix(ext)
            if potential_file.exists():
                downloaded_file = potential_file
                break
        
        # Also check the original cache_path
        if downloaded_file is None and cache_path.exists():
            downloaded_file = cache_path
            
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
        
        # Clean up partial download
        if cache_path.exists():
            cache_path.unlink()
        for ext in ['.wav', '.mp3', '.m4a', '.webm', '.mp4', '.opus', '.ogg', '.part']:
            temp_file = cache_path.with_suffix(ext)
            if temp_file.exists():
                temp_file.unlink()
        
        # Provide helpful error message
        if "403" in error_str:
            logger.exception(f"YouTube download blocked (403 Forbidden): {e}")
            raise Exception(
                "This video is blocked from downloading. YouTube may be restricting access "
                "from this server. Try a different song or use a popular/official music video."
            )
        elif "not available" in error_str.lower():
            logger.exception(f"YouTube format not available: {e}")
            raise Exception(
                "No downloadable format available for this video. "
                "Try a different version of the song."
            )
        else:
            logger.exception(f"YouTube download failed: {e}")
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
        raise ImportError("yt-dlp not installed. Install with: pip install yt-dlp")
    
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
            
            # Try to extract artist from title
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


# Search music databases (optional enhancement)
async def search_music_metadata(query: str) -> List[Dict]:
    """
    Search for music metadata from various sources.
    This provides a more music-focused search experience.
    
    Currently uses YouTube search but could be extended to use:
    - MusicBrainz
    - Spotify API
    - Last.fm API
    """
    results = await search_youtube(query)
    return [asdict(r) for r in results]
