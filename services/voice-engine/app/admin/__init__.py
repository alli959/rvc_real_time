"""
Admin API Module

Provides admin endpoints for:
- Log streaming and management
- System metrics
- Asset management
"""

from .logs_api import router as logs_router
from .metrics_api import router as metrics_router
from .assets_api import router as assets_router
from .log_discovery import LogDiscovery

__all__ = [
    'logs_router',
    'metrics_router', 
    'assets_router',
    'LogDiscovery',
]
