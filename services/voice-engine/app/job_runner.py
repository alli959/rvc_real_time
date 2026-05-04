"""Async job runner using ThreadPoolExecutor with busy tracking and heartbeat."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class HeartbeatTimer:
    """Periodic timer that re-sends progress to prevent stale reaping."""

    def __init__(self, interval: float, callback):
        self._timer = None
        self.interval = interval
        self.callback = callback
        self._running = False

    def start(self):
        self._running = True
        self._schedule()

    def _schedule(self):
        if self._running:
            self._timer = threading.Timer(self.interval, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self):
        if self._running:
            try:
                self.callback()
            except Exception as e:
                logger.warning(f"Heartbeat callback failed: {e}")
            self._schedule()

    def stop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None


class JobRunner:
    """Single-worker executor with busy tracking."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._busy = False
        self._lock = threading.Lock()

    @property
    def is_busy(self) -> bool:
        return self._busy

    def submit(self, fn, *args, **kwargs) -> bool:
        """Submit a job. Returns False if busy (caller should return 503)."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def wrapper():
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Job failed with unhandled exception: {e}")
            finally:
                with self._lock:
                    self._busy = False

        self._executor.submit(wrapper)
        return True


# Global singleton
runner = JobRunner()
