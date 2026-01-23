"""
Trainer Service - RVC Voice Model Training
FastAPI application entry point
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Trainer service starting up...")
    
    # Initialize GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {device_name}")
            # Set memory growth to avoid OOM
            torch.cuda.empty_cache()
        else:
            logger.warning("CUDA not available, training will be slow on CPU")
    except Exception as e:
        logger.error(f"Error initializing CUDA: {e}")
    
    # Import and include routers after startup
    from app.api import router
    app.include_router(router, prefix="/api/v1/trainer")
    
    yield
    
    # Shutdown
    logger.info("Trainer service shutting down...")
    # Clean up any running training jobs
    from app.api import cleanup_jobs
    await cleanup_jobs()


app = FastAPI(
    title="MorphVox Trainer Service",
    description="RVC Voice Model Training Service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch
    return {
        "status": "healthy",
        "service": "trainer",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "MorphVox Trainer",
        "version": "1.0.0",
        "docs": "/docs"
    }
