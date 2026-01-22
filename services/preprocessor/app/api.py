"""
Preprocessor API Module

HTTP API endpoints for audio preprocessing.
"""

import asyncio
import logging
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form
from pydantic import BaseModel, Field

from .config import settings
from .preprocess import PreprocessConfig, run_preprocessing
from .f0_extract import F0Config, run_f0_extraction
from .feature_extract import FeatureConfig, run_feature_extraction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/preprocess", tags=["preprocess"])


# =============================================================================
# Data Models
# =============================================================================

class PreprocessStatus(str, Enum):
    """Preprocessing job status."""
    PENDING = "pending"
    SLICING = "slicing"
    F0_EXTRACTION = "f0_extraction"
    FEATURE_EXTRACTION = "feature_extraction"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PreprocessJob:
    """Preprocessing job tracking."""
    job_id: str
    exp_name: str
    status: PreprocessStatus
    stage: str
    progress: float  # 0.0 to 1.0
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "exp_name": self.exp_name,
            "status": self.status.value,
            "stage": self.stage,
            "progress": round(self.progress * 100, 1),
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stats": self.stats,
        }


class StartPreprocessRequest(BaseModel):
    """Request to start preprocessing."""
    exp_name: str = Field(..., description="Experiment/model name")
    input_dir: Optional[str] = Field(None, description="Input directory (defaults to uploads/{exp_name})")
    sample_rate: int = Field(48000, description="Target sample rate (32000, 40000, or 48000)")
    version: str = Field("v2", description="RVC version (v1 or v2)")
    n_threads: int = Field(4, description="Number of processing threads")


class PreprocessResponse(BaseModel):
    """Response from start preprocessing."""
    job_id: str
    status: str
    message: str


class ValidationResponse(BaseModel):
    """Response from validation endpoint."""
    valid: bool
    exp_name: str
    stats: Dict[str, Any]


# =============================================================================
# Job Management
# =============================================================================

# Global job tracking
_jobs: Dict[str, PreprocessJob] = {}


def get_job(job_id: str) -> Optional[PreprocessJob]:
    """Get job by ID."""
    return _jobs.get(job_id)


def update_job(job: PreprocessJob):
    """Update job in registry."""
    _jobs[job.job_id] = job


# =============================================================================
# Background Tasks
# =============================================================================

async def run_preprocessing_task(job: PreprocessJob, config: PreprocessConfig):
    """
    Run full preprocessing pipeline as background task.
    
    Stages:
    1. Audio slicing and resampling (creates 0_gt_wavs and 1_16k_wavs)
    2. F0 extraction (creates 2a_f0 and 2b_f0nsf)
    3. Feature extraction (creates 3_feature768)
    """
    try:
        # Stage 1: Slicing and resampling
        job.status = PreprocessStatus.SLICING
        job.stage = "Audio slicing and resampling"
        job.progress = 0.0
        update_job(job)
        
        def slice_progress(current, total, filename):
            job.progress = current / total * 0.33  # First 33%
            job.message = f"Processing {filename}"
            update_job(job)
        
        success, result = run_preprocessing(config, slice_progress)
        
        if not success:
            job.status = PreprocessStatus.FAILED
            job.error = result.get("error", "Preprocessing failed")
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            update_job(job)
            return
        
        job.stats["preprocessing"] = result
        
        # Stage 2: F0 extraction
        job.status = PreprocessStatus.F0_EXTRACTION
        job.stage = "F0 extraction"
        job.progress = 0.33
        update_job(job)
        
        f0_config = F0Config(
            exp_name=config.exp_name,
            device=settings.device,
            is_half=True
        )
        
        def f0_progress(current, total, filename):
            job.progress = 0.33 + (current / total * 0.33)  # 33% to 66%
            job.message = f"Extracting F0: {filename}"
            update_job(job)
        
        success, result = run_f0_extraction(f0_config, f0_progress)
        
        if not success:
            job.status = PreprocessStatus.FAILED
            job.error = result.get("error", "F0 extraction failed")
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            update_job(job)
            return
        
        job.stats["f0_extraction"] = result
        
        # Stage 3: Feature extraction
        job.status = PreprocessStatus.FEATURE_EXTRACTION
        job.stage = "Feature extraction"
        job.progress = 0.66
        update_job(job)
        
        feature_config = FeatureConfig(
            exp_name=config.exp_name,
            version=config.version,
            device=settings.device,
            is_half=True
        )
        
        def feature_progress(current, total, filename):
            job.progress = 0.66 + (current / total * 0.34)  # 66% to 100%
            job.message = f"Extracting features: {filename}"
            update_job(job)
        
        success, result = run_feature_extraction(feature_config, feature_progress)
        
        if not success:
            job.status = PreprocessStatus.FAILED
            job.error = result.get("error", "Feature extraction failed")
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            update_job(job)
            return
        
        job.stats["feature_extraction"] = result
        
        # Completed successfully
        job.status = PreprocessStatus.COMPLETED
        job.stage = "Completed"
        job.progress = 1.0
        job.message = "Preprocessing completed successfully"
        job.completed_at = datetime.utcnow().isoformat() + "Z"
        update_job(job)
        
        logger.info(f"Preprocessing completed for {config.exp_name}")
        
    except Exception as e:
        logger.exception(f"Preprocessing failed for {config.exp_name}")
        job.status = PreprocessStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.utcnow().isoformat() + "Z"
        update_job(job)


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/start", response_model=PreprocessResponse)
async def start_preprocessing(
    request: StartPreprocessRequest,
    background_tasks: BackgroundTasks
):
    """
    Start preprocessing for a model.
    
    This creates the training data directories:
    - 0_gt_wavs: Ground truth audio at target sample rate
    - 1_16k_wavs: Resampled audio at 16kHz for HuBERT
    - 2a_f0: Coarse F0 contours
    - 2b_f0nsf: Fine F0 contours
    - 3_feature768: HuBERT features (v2)
    """
    # Validate sample rate
    if request.sample_rate not in [32000, 40000, 48000]:
        raise HTTPException(400, f"Invalid sample rate: {request.sample_rate}")
    
    # Validate version
    if request.version not in ["v1", "v2"]:
        raise HTTPException(400, f"Invalid version: {request.version}")
    
    # Determine input directory
    input_dir = request.input_dir or str(Path(settings.uploads_dir) / request.exp_name)
    
    if not Path(input_dir).exists():
        raise HTTPException(404, f"Input directory not found: {input_dir}")
    
    # Check for existing active job
    for job in _jobs.values():
        if job.exp_name == request.exp_name and job.status in [
            PreprocessStatus.PENDING,
            PreprocessStatus.SLICING,
            PreprocessStatus.F0_EXTRACTION,
            PreprocessStatus.FEATURE_EXTRACTION
        ]:
            raise HTTPException(409, f"Preprocessing already in progress for {request.exp_name}")
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = PreprocessJob(
        job_id=job_id,
        exp_name=request.exp_name,
        status=PreprocessStatus.PENDING,
        stage="Initializing",
        progress=0.0,
        started_at=datetime.utcnow().isoformat() + "Z",
    )
    update_job(job)
    
    # Create config
    config = PreprocessConfig(
        exp_name=request.exp_name,
        input_dir=input_dir,
        sample_rate=request.sample_rate,
        version=request.version,
        n_threads=request.n_threads,
    )
    
    # Start background task
    background_tasks.add_task(run_preprocessing_task, job, config)
    
    return PreprocessResponse(
        job_id=job_id,
        status="started",
        message=f"Preprocessing started for {request.exp_name}"
    )


@router.get("/status/{job_id}")
async def get_preprocessing_status(job_id: str):
    """Get status of a preprocessing job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job.to_dict()


@router.get("/validate/{exp_name}", response_model=ValidationResponse)
async def validate_preprocessing(exp_name: str):
    """
    Validate preprocessing output for a model.
    
    Checks that all directories exist and have matching file counts.
    """
    exp_dir = Path(settings.models_dir) / exp_name
    
    if not exp_dir.exists():
        raise HTTPException(404, f"Experiment directory not found: {exp_name}")
    
    # Check directories
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    wav16k_dir = exp_dir / "1_16k_wavs"
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b_f0nsf"
    feature_dir_v1 = exp_dir / "3_feature256"
    feature_dir_v2 = exp_dir / "3_feature768"
    
    stats = {
        "gt_wavs": len(list(gt_wavs_dir.glob("*.wav"))) if gt_wavs_dir.exists() else 0,
        "wav16k": len(list(wav16k_dir.glob("*.wav"))) if wav16k_dir.exists() else 0,
        "f0": len(list(f0_dir.glob("*.npy"))) if f0_dir.exists() else 0,
        "f0nsf": len(list(f0nsf_dir.glob("*.npy"))) if f0nsf_dir.exists() else 0,
        "features_v1": len(list(feature_dir_v1.glob("*.npy"))) if feature_dir_v1.exists() else 0,
        "features_v2": len(list(feature_dir_v2.glob("*.npy"))) if feature_dir_v2.exists() else 0,
    }
    
    # Determine version
    feature_count = max(stats["features_v1"], stats["features_v2"])
    stats["feature_count"] = feature_count
    stats["version"] = "v2" if stats["features_v2"] > stats["features_v1"] else "v1"
    
    # Validate counts match
    counts = [stats["gt_wavs"], stats["wav16k"], stats["f0"], stats["f0nsf"], feature_count]
    valid = len(set(counts)) == 1 and counts[0] > 0
    
    stats["mismatches"] = []
    if not valid:
        if stats["gt_wavs"] != stats["wav16k"]:
            stats["mismatches"].append(f"gt_wavs ({stats['gt_wavs']}) != wav16k ({stats['wav16k']})")
        if stats["wav16k"] != stats["f0"]:
            stats["mismatches"].append(f"wav16k ({stats['wav16k']}) != f0 ({stats['f0']})")
        if stats["f0"] != stats["f0nsf"]:
            stats["mismatches"].append(f"f0 ({stats['f0']}) != f0nsf ({stats['f0nsf']})")
        if stats["f0"] != feature_count:
            stats["mismatches"].append(f"f0 ({stats['f0']}) != features ({feature_count})")
    
    return ValidationResponse(
        valid=valid,
        exp_name=exp_name,
        stats=stats
    )


@router.get("/jobs")
async def list_jobs():
    """List all preprocessing jobs."""
    return [job.to_dict() for job in _jobs.values()]


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a preprocessing job (if running)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    
    # Note: This only marks it as cancelled, actual cancellation
    # would require more complex task management
    if job.status in [PreprocessStatus.PENDING, PreprocessStatus.SLICING,
                      PreprocessStatus.F0_EXTRACTION, PreprocessStatus.FEATURE_EXTRACTION]:
        job.status = PreprocessStatus.FAILED
        job.error = "Cancelled by user"
        job.completed_at = datetime.utcnow().isoformat() + "Z"
        update_job(job)
        return {"status": "cancelled", "job_id": job_id}
    
    return {"status": "already_finished", "job_id": job_id}


# =============================================================================
# Upload Endpoint (for training audio files)
# =============================================================================


class UploadResponse(BaseModel):
    """Response from upload endpoint."""
    success: bool
    exp_name: str
    files_uploaded: int
    upload_dir: str
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_training_audio(
    exp_name: str = Form(..., description="Experiment/model name"),
    files: List[UploadFile] = File(..., description="Audio files for training")
):
    """
    Upload training audio files.
    
    Files are stored at: {uploads_dir}/{exp_name}/
    This is the input directory for preprocessing.
    
    Supported formats: WAV, MP3, FLAC, OGG
    """
    if not exp_name:
        raise HTTPException(400, "exp_name is required")
    
    if not files:
        raise HTTPException(400, "At least one file is required")
    
    # Create upload directory
    upload_dir = Path(settings.uploads_dir) / exp_name
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_count = 0
    errors = []
    
    # Allowed extensions
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    for file in files:
        try:
            # Validate extension
            ext = Path(file.filename).suffix.lower()
            if ext not in allowed_extensions:
                errors.append(f"Skipped {file.filename}: unsupported format {ext}")
                continue
            
            # Save file
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            uploaded_count += 1
            logger.info(f"Uploaded: {file.filename} -> {file_path}")
            
        except Exception as e:
            errors.append(f"Failed to upload {file.filename}: {str(e)}")
            logger.error(f"Upload error for {file.filename}: {e}")
        finally:
            await file.close()
    
    if uploaded_count == 0:
        raise HTTPException(400, f"No files uploaded. Errors: {'; '.join(errors)}")
    
    message = f"Uploaded {uploaded_count} files"
    if errors:
        message += f". Warnings: {'; '.join(errors)}"
    
    return UploadResponse(
        success=True,
        exp_name=exp_name,
        files_uploaded=uploaded_count,
        upload_dir=str(upload_dir),
        message=message
    )
