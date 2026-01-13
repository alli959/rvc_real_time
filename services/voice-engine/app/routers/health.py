"""
Health Check Router

Provides service health and status endpoints.
"""

import logging
from fastapi import APIRouter

from app.models.common import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and version information.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        message="Voice Engine is running"
    )


@router.get("/")
async def root():
    """Root endpoint - returns basic service info."""
    return {
        "service": "Voice Engine API",
        "version": "1.0.0",
        "docs": "/docs",
    }
