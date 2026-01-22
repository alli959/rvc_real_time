"""
Preprocessor Service - Main Entry Point

Audio preprocessing service for RVC voice model training.
Handles slicing, resampling, F0 extraction, and feature extraction.
"""

import os
import sys
import logging
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import router as api_router

# Setup logging
def setup_logging():
    """Configure logging with both console and file handlers."""
    log_dir = Path("/app/logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "preprocessor.log"),
        ]
    )
    
    # Suppress noisy loggers
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fairseq").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MorphVox Preprocessor Service",
    description="Audio preprocessing service for RVC voice model training",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "preprocessor",
        "version": "1.0.0"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Starting MorphVox Preprocessor Service")
    logger.info(f"HTTP Port: {settings.http_port}")
    logger.info(f"Models Dir: {settings.models_dir}")
    logger.info(f"Device: {settings.device}")
    
    # Verify required assets exist
    if not Path(settings.hubert_path).exists():
        logger.warning(f"HuBERT model not found at {settings.hubert_path}")
    if not Path(settings.rmvpe_path).exists():
        logger.warning(f"RMVPE model not found at {settings.rmvpe_path}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down MorphVox Preprocessor Service")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.http_port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        workers=1,  # Single worker for GPU operations
    )
