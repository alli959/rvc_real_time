"""
Model Cache Service - Memory-efficient model management with LRU eviction

Provides lazy loading and automatic unloading of ML models to optimize memory usage.
Supports RVC models, UVR5 models, and Bark TTS with configurable cache limits.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """Represents a cached model with metadata."""
    name: str
    model: Any
    load_time: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    size_estimate_mb: float = 0.0  # Estimated memory footprint
    
    def touch(self):
        """Update last access time and increment counter."""
        self.last_access = time.time()
        self.access_count += 1


@dataclass  
class CacheConfig:
    """Configuration for model cache."""
    # Maximum number of RVC models to keep loaded
    max_rvc_models: int = 2
    
    # Maximum number of UVR5 models to keep loaded
    max_uvr5_models: int = 1
    
    # Unload models after this many seconds of inactivity
    idle_timeout_seconds: float = 300.0  # 5 minutes
    
    # Whether to preload Bark TTS at startup (uses ~1.5GB GPU memory)
    preload_bark: bool = False
    
    # Whether to keep HuBERT/RMVPE always loaded (required for RVC)
    keep_base_models_loaded: bool = True
    
    # Run cleanup every N seconds
    cleanup_interval_seconds: float = 60.0
    
    # Memory threshold (MB) - trigger cleanup if estimated usage exceeds this
    memory_threshold_mb: float = 2500.0


class ModelCache:
    """
    LRU cache for ML models with automatic eviction and cleanup.
    
    Features:
    - Lazy loading: Models loaded on first use
    - LRU eviction: Least recently used models evicted when cache full
    - Idle timeout: Models unloaded after period of inactivity
    - Thread-safe: Safe for concurrent access
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Separate caches for different model types (OrderedDict for LRU)
        self._rvc_cache: OrderedDict[str, CachedModel] = OrderedDict()
        self._uvr5_cache: OrderedDict[str, CachedModel] = OrderedDict()
        self._bark_loaded: bool = False
        self._bark_model: Optional[Any] = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        # Statistics
        self._stats = {
            'rvc_loads': 0,
            'rvc_evictions': 0,
            'uvr5_loads': 0,
            'uvr5_evictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        logger.info(f"ModelCache initialized: max_rvc={self.config.max_rvc_models}, "
                   f"max_uvr5={self.config.max_uvr5_models}, "
                   f"idle_timeout={self.config.idle_timeout_seconds}s")
    
    def start_cleanup_thread(self):
        """Start background thread for periodic cleanup."""
        if self._cleanup_thread is not None:
            return
            
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="ModelCacheCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Model cache cleanup thread started")
    
    def stop_cleanup_thread(self):
        """Stop the cleanup thread."""
        if self._cleanup_thread is None:
            return
            
        self._stop_cleanup.set()
        self._cleanup_thread.join(timeout=5.0)
        self._cleanup_thread = None
        logger.info("Model cache cleanup thread stopped")
    
    def _cleanup_loop(self):
        """Background loop for periodic cleanup."""
        while not self._stop_cleanup.wait(self.config.cleanup_interval_seconds):
            try:
                self._cleanup_idle_models()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_idle_models(self):
        """Remove models that haven't been used recently."""
        now = time.time()
        timeout = self.config.idle_timeout_seconds
        
        with self._lock:
            # Clean RVC models
            to_remove = []
            for name, cached in self._rvc_cache.items():
                if now - cached.last_access > timeout:
                    to_remove.append(name)
            
            for name in to_remove:
                self._evict_rvc_model(name, reason="idle timeout")
            
            # Clean UVR5 models
            to_remove = []
            for name, cached in self._uvr5_cache.items():
                if now - cached.last_access > timeout:
                    to_remove.append(name)
            
            for name in to_remove:
                self._evict_uvr5_model(name, reason="idle timeout")
        
        # Force garbage collection after cleanup
        if to_remove:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def _evict_rvc_model(self, name: str, reason: str = "LRU"):
        """Evict an RVC model from cache."""
        if name in self._rvc_cache:
            cached = self._rvc_cache.pop(name)
            logger.info(f"Evicted RVC model '{name}' ({reason}), "
                       f"was accessed {cached.access_count} times")
            self._stats['rvc_evictions'] += 1
            del cached.model
    
    def _evict_uvr5_model(self, name: str, reason: str = "LRU"):
        """Evict a UVR5 model from cache."""
        if name in self._uvr5_cache:
            cached = self._uvr5_cache.pop(name)
            logger.info(f"Evicted UVR5 model '{name}' ({reason}), "
                       f"was accessed {cached.access_count} times")
            self._stats['uvr5_evictions'] += 1
            del cached.model
    
    def _ensure_rvc_capacity(self):
        """Ensure RVC cache has room for a new model."""
        while len(self._rvc_cache) >= self.config.max_rvc_models:
            # Evict oldest (first) item in OrderedDict
            oldest_name = next(iter(self._rvc_cache))
            self._evict_rvc_model(oldest_name, reason="LRU capacity")
    
    def _ensure_uvr5_capacity(self):
        """Ensure UVR5 cache has room for a new model."""
        while len(self._uvr5_cache) >= self.config.max_uvr5_models:
            oldest_name = next(iter(self._uvr5_cache))
            self._evict_uvr5_model(oldest_name, reason="LRU capacity")
    
    def get_rvc_model(
        self,
        name: str,
        loader: Callable[[], Any],
        size_estimate_mb: float = 100.0
    ) -> Any:
        """
        Get an RVC model from cache, loading if necessary.
        
        Args:
            name: Unique identifier for the model
            loader: Function to load the model if not cached
            size_estimate_mb: Estimated memory usage in MB
            
        Returns:
            The loaded model
        """
        with self._lock:
            if name in self._rvc_cache:
                # Cache hit - move to end (most recently used)
                self._rvc_cache.move_to_end(name)
                self._rvc_cache[name].touch()
                self._stats['cache_hits'] += 1
                return self._rvc_cache[name].model
            
            # Cache miss - load model
            self._stats['cache_misses'] += 1
            self._ensure_rvc_capacity()
            
            logger.info(f"Loading RVC model '{name}' into cache...")
            load_start = time.time()
            model = loader()
            load_time = time.time() - load_start
            
            cached = CachedModel(
                name=name,
                model=model,
                size_estimate_mb=size_estimate_mb
            )
            self._rvc_cache[name] = cached
            self._stats['rvc_loads'] += 1
            
            logger.info(f"RVC model '{name}' loaded in {load_time:.2f}s "
                       f"(cache: {len(self._rvc_cache)}/{self.config.max_rvc_models})")
            
            return model
    
    def get_uvr5_model(
        self,
        name: str,
        loader: Callable[[], Any],
        size_estimate_mb: float = 200.0
    ) -> Any:
        """
        Get a UVR5 model from cache, loading if necessary.
        
        Args:
            name: Model name (e.g., "HP5_only_main_vocal")
            loader: Function to load the model
            size_estimate_mb: Estimated memory usage in MB
            
        Returns:
            The loaded model
        """
        with self._lock:
            if name in self._uvr5_cache:
                self._uvr5_cache.move_to_end(name)
                self._uvr5_cache[name].touch()
                self._stats['cache_hits'] += 1
                return self._uvr5_cache[name].model
            
            self._stats['cache_misses'] += 1
            self._ensure_uvr5_capacity()
            
            logger.info(f"Loading UVR5 model '{name}' into cache...")
            load_start = time.time()
            model = loader()
            load_time = time.time() - load_start
            
            cached = CachedModel(
                name=name,
                model=model,
                size_estimate_mb=size_estimate_mb
            )
            self._uvr5_cache[name] = cached
            self._stats['uvr5_loads'] += 1
            
            logger.info(f"UVR5 model '{name}' loaded in {load_time:.2f}s")
            
            return model
    
    def unload_bark(self):
        """Unload Bark TTS models to free memory."""
        if not self._bark_loaded:
            return
            
        logger.info("Unloading Bark TTS models...")
        try:
            # Clear Bark's internal model cache
            from bark.generation import clean_models
            clean_models()
        except (ImportError, AttributeError):
            pass
        
        self._bark_loaded = False
        self._bark_model = None
        
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Bark TTS models unloaded")
    
    def is_bark_loaded(self) -> bool:
        """Check if Bark models are currently loaded."""
        return self._bark_loaded
    
    def mark_bark_loaded(self):
        """Mark Bark as loaded (called after preload_models())."""
        self._bark_loaded = True
    
    def clear_all(self):
        """Clear all cached models and free memory."""
        with self._lock:
            # Clear RVC models
            for name in list(self._rvc_cache.keys()):
                self._evict_rvc_model(name, reason="clear_all")
            
            # Clear UVR5 models
            for name in list(self._uvr5_cache.keys()):
                self._evict_uvr5_model(name, reason="clear_all")
            
            # Unload Bark
            self.unload_bark()
        
        # Force garbage collection
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("All models cleared from cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                **self._stats,
                'rvc_models_loaded': len(self._rvc_cache),
                'uvr5_models_loaded': len(self._uvr5_cache),
                'bark_loaded': self._bark_loaded,
                'rvc_models': list(self._rvc_cache.keys()),
                'uvr5_models': list(self._uvr5_cache.keys()),
            }
    
    def get_memory_estimate_mb(self) -> float:
        """Estimate current memory usage of cached models."""
        with self._lock:
            total = 0.0
            for cached in self._rvc_cache.values():
                total += cached.size_estimate_mb
            for cached in self._uvr5_cache.values():
                total += cached.size_estimate_mb
            if self._bark_loaded:
                total += 1500.0  # Bark uses ~1.5GB
            return total


# Global cache instance
_model_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get or create the global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
        _model_cache.start_cleanup_thread()
    return _model_cache


def configure_model_cache(config: CacheConfig) -> ModelCache:
    """Configure and return the global model cache."""
    global _model_cache
    if _model_cache is not None:
        _model_cache.stop_cleanup_thread()
    _model_cache = ModelCache(config)
    _model_cache.start_cleanup_thread()
    return _model_cache
