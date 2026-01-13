"""
YouTube Audio Cache

Manages cached downloaded audio files.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path(os.environ.get("YOUTUBE_CACHE_DIR", "/tmp/youtube_cache"))
MAX_CACHE_FILES = int(os.environ.get("YOUTUBE_MAX_CACHE", "50"))


def ensure_cache_dir():
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(video_id: str, ext: str = "mp3") -> Path:
    """Get cache file path for a video ID."""
    ensure_cache_dir()
    return CACHE_DIR / f"{video_id}.{ext}"


def is_cached(video_id: str) -> bool:
    """Check if audio is already cached."""
    # Check for any audio format
    for ext in ["mp3", "wav", "m4a", "webm", "opus", "ogg"]:
        if get_cache_path(video_id, ext).exists():
            return True
    return False


def get_cached_file(video_id: str) -> Optional[Path]:
    """Get cached file path if exists."""
    for ext in ["wav", "mp3", "m4a", "webm", "opus", "ogg"]:
        path = get_cache_path(video_id, ext)
        if path.exists():
            return path
    return None


def clean_cache():
    """Clean old cache files if limit exceeded."""
    try:
        # Get all audio files
        files = []
        for ext in ["*.mp3", "*.wav", "*.m4a", "*.webm", "*.opus", "*.ogg"]:
            files.extend(CACHE_DIR.glob(ext))
        
        if len(files) > MAX_CACHE_FILES:
            # Sort by modification time, remove oldest
            files.sort(key=lambda f: f.stat().st_mtime)
            for f in files[:len(files) - MAX_CACHE_FILES]:
                f.unlink()
                logger.info(f"Removed old cache file: {f}")
    except Exception as e:
        logger.warning(f"Cache cleanup failed: {e}")


def clear_cache():
    """Clear all cached files."""
    try:
        for ext in ["*.mp3", "*.wav", "*.m4a", "*.webm", "*.opus", "*.ogg"]:
            for f in CACHE_DIR.glob(ext):
                f.unlink()
        logger.info("Cache cleared")
    except Exception as e:
        logger.warning(f"Cache clear failed: {e}")


def get_cache_size() -> int:
    """Get total cache size in bytes."""
    total = 0
    try:
        for ext in ["*.mp3", "*.wav", "*.m4a", "*.webm", "*.opus", "*.ogg"]:
            for f in CACHE_DIR.glob(ext):
                total += f.stat().st_size
    except Exception:
        pass
    return total


def get_cache_stats() -> dict:
    """Get cache statistics."""
    files = []
    for ext in ["*.mp3", "*.wav", "*.m4a", "*.webm", "*.opus", "*.ogg"]:
        files.extend(CACHE_DIR.glob(ext))
    
    return {
        "count": len(files),
        "max_files": MAX_CACHE_FILES,
        "total_size_mb": round(get_cache_size() / (1024 * 1024), 2),
        "cache_dir": str(CACHE_DIR),
    }


class AudioCache:
    """
    Audio cache manager with LRU-style cleanup.
    
    Usage:
        cache = AudioCache()
        if cache.has(video_id):
            path = cache.get(video_id)
        else:
            # download...
            cache.put(video_id, audio_path)
    """
    
    def __init__(
        self, 
        cache_dir: Optional[Path] = None,
        max_files: Optional[int] = None
    ):
        self.cache_dir = cache_dir or CACHE_DIR
        self.max_files = max_files or MAX_CACHE_FILES
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def has(self, video_id: str) -> bool:
        """Check if video is cached."""
        return is_cached(video_id)
    
    def get(self, video_id: str) -> Optional[Path]:
        """Get cached file path."""
        return get_cached_file(video_id)
    
    def put(self, video_id: str, source_path: Path) -> Path:
        """
        Add file to cache.
        
        Args:
            video_id: YouTube video ID
            source_path: Path to audio file to cache
        
        Returns:
            Path to cached file
        """
        import shutil
        
        # Determine extension
        ext = source_path.suffix.lstrip(".")
        cache_path = self.cache_dir / f"{video_id}.{ext}"
        
        # Copy to cache
        shutil.copy2(source_path, cache_path)
        
        # Cleanup if needed
        self._cleanup()
        
        return cache_path
    
    def remove(self, video_id: str):
        """Remove video from cache."""
        for ext in ["mp3", "wav", "m4a", "webm", "opus", "ogg"]:
            path = self.cache_dir / f"{video_id}.{ext}"
            if path.exists():
                path.unlink()
    
    def _cleanup(self):
        """Run cleanup if over limit."""
        files = list(self.cache_dir.glob("*"))
        audio_files = [f for f in files if f.suffix in [".mp3", ".wav", ".m4a", ".webm", ".opus", ".ogg"]]
        
        if len(audio_files) > self.max_files:
            # Sort by modification time
            audio_files.sort(key=lambda f: f.stat().st_mtime)
            # Remove oldest
            for f in audio_files[:len(audio_files) - self.max_files]:
                f.unlink()
                logger.info(f"Cache cleanup: removed {f.name}")
    
    def clear(self):
        """Clear all cached files."""
        clear_cache()
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return get_cache_stats()
