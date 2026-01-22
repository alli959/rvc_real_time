"""
Trainer Service - FastAPI Routes
API endpoints for training job management
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .jobs import TrainingJob, TrainingStatus, job_manager, cleanup_jobs
from .training import executor

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class StartTrainingRequest(BaseModel):
    """Request to start training"""
    exp_name: str = Field(..., description="Experiment/model name")
    epochs: int = Field(default=100, ge=1, le=10000, description="Number of epochs")
    batch_size: int = Field(default=8, ge=1, le=64, description="Training batch size")
    save_every_epoch: int = Field(default=10, ge=1, description="Save checkpoint every N epochs")
    sample_rate: int = Field(default=48000, description="Audio sample rate")
    version: str = Field(default="v2", pattern="^v[12]$", description="RVC version")
    use_pitch_guidance: bool = Field(default=True, description="Use F0 pitch guidance")
    gpus: str = Field(default="0", description="GPU device IDs")
    pretrain_g: Optional[str] = Field(default=None, description="Path to pretrained generator")
    pretrain_d: Optional[str] = Field(default=None, description="Path to pretrained discriminator")


class TrainingResponse(BaseModel):
    """Training operation response"""
    success: bool
    job_id: Optional[str] = None
    message: str


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    exp_name: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    message: str
    error: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None
    logs: List[str] = []
    result: Optional[Dict[str, Any]] = None


class ExtractModelRequest(BaseModel):
    """Request to extract model from checkpoint"""
    exp_name: str = Field(..., description="Experiment/model name")
    epoch: Optional[int] = Field(default=None, description="Epoch to extract (None for latest)")
    sample_rate: int = Field(default=48000, description="Audio sample rate")
    version: str = Field(default="v2", description="RVC version")


class BuildIndexRequest(BaseModel):
    """Request to build FAISS index"""
    exp_name: str = Field(..., description="Experiment/model name")
    version: str = Field(default="v2", description="RVC version")


# ============================================================================
# Training Endpoints
# ============================================================================

@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: StartTrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new training job.
    
    Prerequisites:
    - Preprocessing must be complete for the experiment
    - All required directories must exist (0_gt_wavs, 1_16k_wavs, 2a_f0, 2b_f0nsf, 3_feature768)
    """
    # Check for existing active job
    active_job = job_manager.get_active_job(request.exp_name)
    if active_job:
        raise HTTPException(
            status_code=409,
            detail=f"Training already in progress for {request.exp_name} (job_id: {active_job.job_id})"
        )
    
    # Create job
    config = request.model_dump()
    job = job_manager.create_job(request.exp_name, config)
    
    # Run training in background
    background_tasks.add_task(run_training_job, job)
    
    return TrainingResponse(
        success=True,
        job_id=job.job_id,
        message=f"Training job started for {request.exp_name}"
    )


async def run_training_job(job: TrainingJob):
    """Background task to run the full training pipeline"""
    try:
        # Step 1: Validate preprocessing
        job.status = TrainingStatus.VALIDATING
        job.log("Validating preprocessing outputs...")
        
        valid, message = await executor.validate_preprocessing(job.exp_name)
        if not valid:
            job.status = TrainingStatus.FAILED
            job.error = message
            job.log(f"Validation failed: {message}")
            return
        
        job.log(message)
        
        # Step 2: Generate filelist
        job.status = TrainingStatus.GENERATING_FILELIST
        job.log("Generating filelist...")
        
        config = job.config
        filelist_path = await executor.generate_filelist(
            exp_name=job.exp_name,
            sample_rate=config.get("sample_rate", 48000),
            version=config.get("version", "v2"),
            use_pitch_guidance=config.get("use_pitch_guidance", True)
        )
        
        # Create training config
        await executor.create_training_config(
            exp_name=job.exp_name,
            sample_rate=config.get("sample_rate", 48000),
            batch_size=config.get("batch_size", 8),
            version=config.get("version", "v2")
        )
        
        job.log(f"Generated filelist: {filelist_path}")
        
        # Step 3: Run training
        success = await executor.run_training(job)
        
        if job._cancel_requested:
            job.status = TrainingStatus.CANCELLED
            job.log("Training cancelled by user")
            return
        
        if not success:
            job.status = TrainingStatus.FAILED
            job.error = "Training subprocess failed"
            job.log("Training failed")
            return
        
        # Step 4: Extract model
        job.status = TrainingStatus.EXTRACTING_MODEL
        job.log("Extracting inference model...")
        
        model_path = await executor.extract_model(
            exp_name=job.exp_name,
            sample_rate=config.get("sample_rate", 48000),
            version=config.get("version", "v2")
        )
        
        # Step 5: Build index
        job.status = TrainingStatus.BUILDING_INDEX
        job.log("Building FAISS index...")
        
        index_path = await executor.build_index(
            exp_name=job.exp_name,
            version=config.get("version", "v2")
        )
        
        # Complete
        job.status = TrainingStatus.COMPLETED
        job.progress = 1.0
        job.completed_at = datetime.utcnow().isoformat() + "Z"
        job.result = {
            "model_path": model_path,
            "index_path": index_path
        }
        job.log(f"Training completed! Model: {model_path}")
        
    except Exception as e:
        logger.exception(f"Training job failed: {e}")
        job.status = TrainingStatus.FAILED
        job.error = str(e)
        job.log(f"Error: {e}")


from datetime import datetime


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get training job status"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    data = job.to_dict()
    return JobStatusResponse(**data)


@router.post("/stop/{job_id}", response_model=TrainingResponse)
async def stop_training(job_id: str):
    """Stop a training job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    if job.status not in [TrainingStatus.TRAINING, TrainingStatus.PENDING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot stop job in status: {job.status.value}"
        )
    
    success = job_manager.cancel_job(job_id)
    
    return TrainingResponse(
        success=success,
        job_id=job_id,
        message="Stop requested" if success else "Failed to stop job"
    )


@router.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    return {"jobs": job_manager.list_jobs()}


# ============================================================================
# Model Extraction Endpoints
# ============================================================================

@router.post("/extract-model", response_model=TrainingResponse)
async def extract_model(request: ExtractModelRequest):
    """
    Extract inference model from training checkpoint.
    
    This creates a lightweight .pth file that can be used for inference,
    without the optimizer state and other training artifacts.
    """
    model_path = await executor.extract_model(
        exp_name=request.exp_name,
        sample_rate=request.sample_rate,
        version=request.version,
        epoch=request.epoch
    )
    
    if not model_path:
        raise HTTPException(status_code=500, detail="Failed to extract model")
    
    return TrainingResponse(
        success=True,
        message=f"Model extracted to: {model_path}"
    )


@router.post("/build-index", response_model=TrainingResponse)
async def build_index(request: BuildIndexRequest):
    """
    Build FAISS index from extracted features.
    
    This creates an index for fast voice matching during inference.
    """
    index_path = await executor.build_index(
        exp_name=request.exp_name,
        version=request.version
    )
    
    if not index_path:
        raise HTTPException(status_code=500, detail="Failed to build index")
    
    return TrainingResponse(
        success=True,
        message=f"Index built at: {index_path}"
    )


# ============================================================================
# Validation Endpoints
# ============================================================================

@router.get("/validate/{exp_name}")
async def validate_preprocessing(exp_name: str):
    """
    Validate that preprocessing is complete for an experiment.
    
    Checks for all required directories and file counts.
    """
    valid, message = await executor.validate_preprocessing(exp_name)
    
    return {
        "valid": valid,
        "message": message,
        "exp_name": exp_name
    }
