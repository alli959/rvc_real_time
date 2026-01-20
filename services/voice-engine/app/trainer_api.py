"""
Trainer API Module

HTTP API endpoints for voice model training:
- Upload audio files
- Start training jobs
- Monitor training progress
- Recording wizard sessions
- Model scanning and analysis
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks
from pydantic import BaseModel, Field

from app.trainer import (
    RVCTrainingPipeline,
    TrainingConfig,
    TrainingStatus,
    SampleRate,
    F0Method,
    RVCVersion,
    create_training_pipeline,
)
from app.trainer.auto_config import (
    get_auto_config,
    TrainingMode,
    AudioSourceType,
    QualityLevel,
    AutoTrainingConfig,
    AudioAnalysis,
)
from app.analyzer import (
    ModelScanner,
    scan_model,
    analyze_model_gaps,
    LANGUAGE_PHONEMES,
)
from app.wizard import RecordingWizard, SessionStatus
from app.prompts import get_prompt_loader, get_available_languages
from app.model_storage import (
    ModelStorage, 
    ModelMetadata, 
    RecordingInfo,
    get_model_storage,
    init_model_storage
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/trainer", tags=["trainer"])

# Global instances (initialized on startup)
_training_pipeline: Optional[RVCTrainingPipeline] = None
_model_scanner: Optional[ModelScanner] = None
_recording_wizard: Optional[RecordingWizard] = None
_model_storage: Optional[ModelStorage] = None

# Config
LOGS_DIR = Path(__file__).parent.parent / "logs"
ASSETS_DIR = Path(__file__).parent.parent / "assets"
MODELS_DIR = ASSETS_DIR / "models"
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"


def detect_existing_model(model_dir: Path) -> tuple[Optional[str], int]:
    """
    Detect if a model directory has existing training artifacts.
    
    Checks for:
    1. Final extracted model ({name}.pth)
    2. Training checkpoints (G_*.pth, D_*.pth)
    3. Metadata with epoch info
    
    Returns:
        Tuple of (model_path or None, estimated_epochs)
    """
    import re
    
    if not model_dir.exists():
        return None, 0
    
    # Check for final model
    final_models = list(model_dir.glob("*.pth"))
    # Exclude checkpoint files (G_*, D_*)
    final_models = [m for m in final_models if not re.match(r'^[GD]_\d+\.pth$', m.name)]
    
    if final_models:
        # Has a final model - read epochs from metadata
        metadata_path = model_dir / "model_metadata.json"
        epochs = 0
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                epochs = metadata.get("training_config", {}).get("epochs", 0)
            except:
                pass
        return str(final_models[0]), epochs
    
    # Check for checkpoints (training in progress or resumable)
    g_checkpoints = sorted(model_dir.glob("G_*.pth"), key=lambda x: int(re.search(r'G_(\d+)\.pth', x.name).group(1)) if re.search(r'G_(\d+)\.pth', x.name) else 0)
    if g_checkpoints:
        # Has checkpoints - estimate epochs from step number
        # Step = epoch * batches_per_epoch, but we don't know batches
        # Use rough estimate: assume ~50 batches/epoch average
        max_step = 0
        for ckpt in g_checkpoints:
            match = re.search(r'G_(\d+)\.pth', ckpt.name)
            if match:
                step = int(match.group(1))
                max_step = max(max_step, step)
        
        # Rough epoch estimate (conservative)
        estimated_epochs = max_step // 30  # Assume ~30 steps per epoch average
        
        # Return the LATEST checkpoint (last in sorted list)
        return str(g_checkpoints[-1]), estimated_epochs
    
    return None, 0


def init_trainer_api(
    logs_dir: Optional[str] = None,
    assets_dir: Optional[str] = None,
    device: str = "cuda:0"
):
    """Initialize the trainer API components"""
    global _training_pipeline, _model_scanner, _recording_wizard, _model_storage
    
    logs = Path(logs_dir) if logs_dir else LOGS_DIR
    assets = Path(assets_dir) if assets_dir else ASSETS_DIR
    models = assets / "models"
    
    logs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use models directory as base_dir so checkpoints persist with models
    _training_pipeline = create_training_pipeline(
        base_dir=str(models),
        assets_dir=str(assets),
        device=device
    )
    
    _model_scanner = ModelScanner(logs_dir=str(models), use_gpu=True)
    
    # Initialize model storage first - stores recordings in assets/models/{model}/recordings/
    _model_storage = ModelStorage(str(models))
    init_model_storage(str(models))
    
    # Recording wizard uses model storage for recording path
    _recording_wizard = RecordingWizard(
        base_dir=str(logs / "wizard_sessions"),
        models_dir=str(models)
    )
    
    logger.info(f"Trainer API initialized (models_dir={models})")


def get_pipeline() -> RVCTrainingPipeline:
    """Get the training pipeline"""
    if _training_pipeline is None:
        init_trainer_api()
    return _training_pipeline


def get_scanner() -> ModelScanner:
    """Get the model scanner"""
    if _model_scanner is None:
        init_trainer_api()
    return _model_scanner


def get_wizard() -> RecordingWizard:
    """Get the recording wizard"""
    if _recording_wizard is None:
        init_trainer_api()
    return _recording_wizard


def get_storage() -> ModelStorage:
    """Get the model storage manager"""
    if _model_storage is None:
        init_trainer_api()
    return _model_storage


# ============================================================================
# Request/Response Models
# ============================================================================

# Training Models
class TrainingConfigRequest(BaseModel):
    """Training configuration request"""
    exp_name: str = Field(..., description="Experiment/model name")
    sample_rate: int = Field(default=40000, description="Sample rate (32000, 40000, 48000)")
    f0_method: str = Field(default="rmvpe", description="F0 method (rmvpe, pm, harvest)")
    epochs: int = Field(default=200, description="Training epochs")
    batch_size: int = Field(default=8, description="Batch size")
    save_every_epoch: int = Field(default=50, description="Save checkpoint interval")
    version: str = Field(default="v2", description="RVC version (v1, v2)")
    use_pitch_guidance: bool = Field(default=True, description="Use pitch guidance")


class TrainingConfigInput(BaseModel):
    """Training config input for model training (no exp_name required)
    
    Note: epochs, batch_size, save_every_epoch default to None to allow
    auto-configuration based on audio analysis. Set explicit values to override.
    """
    sample_rate: Optional[int] = Field(default=None, description="Sample rate (32000, 40000, 48000). Default: 48000")
    f0_method: Optional[str] = Field(default=None, description="F0 method (rmvpe, pm, harvest). Default: rmvpe")
    epochs: Optional[int] = Field(default=None, description="Training epochs. Auto-calculated if not specified")
    batch_size: Optional[int] = Field(default=None, description="Batch size. Auto-calculated if not specified")
    save_every_epoch: Optional[int] = Field(default=None, description="Save checkpoint interval. Auto-calculated if not specified")
    version: Optional[str] = Field(default=None, description="RVC version (v1, v2). Default: v2")
    use_pitch_guidance: Optional[bool] = Field(default=None, description="Use pitch guidance. Default: True")


class StartTrainingRequest(BaseModel):
    """Start training request"""
    exp_name: str = Field(..., description="Experiment name")
    config: Optional[TrainingConfigRequest] = None
    audio_paths: Optional[List[str]] = Field(default=None, description="Paths to audio files")


class TrainingProgressResponse(BaseModel):
    """Training progress response"""
    job_id: str
    status: str
    step: str
    progress: float
    current_epoch: int
    total_epochs: int
    message: str
    error: Optional[str] = None


# Scanner Models
class ScanModelRequest(BaseModel):
    """Scan model request"""
    model_path: str = Field(..., description="Path to model .pth file")
    languages: List[str] = Field(default=["en", "is"], description="Languages to analyze")


class GapAnalysisRequest(BaseModel):
    """Gap analysis request"""
    model_path: str = Field(..., description="Path to model .pth file")
    language: str = Field(default="en", description="Language to analyze")


# Wizard Models
class CreateSessionRequest(BaseModel):
    """Create wizard session request"""
    language: str = Field(..., description="Language code (en, is)")
    exp_name: str = Field(..., description="Experiment/model name")
    prompt_count: int = Field(default=50, description="Number of prompts")
    target_phonemes: Optional[List[str]] = Field(default=None, description="Specific phonemes to target")


class SubmitRecordingRequest(BaseModel):
    """Submit recording request"""
    audio: str = Field(..., description="Base64 encoded audio")
    sample_rate: int = Field(default=16000, description="Sample rate")
    auto_advance: bool = Field(default=False, description="Auto-advance to next prompt")
    format: Optional[str] = Field(default=None, description="Audio format hint (webm, wav, etc)")


# Prompt Models
class PromptsResponse(BaseModel):
    """Prompts response"""
    language: str
    categories: Dict[str, Any]
    total_prompts: int


# ============================================================================
# Training Endpoints
# ============================================================================

@router.post("/upload")
async def upload_training_audio(
    exp_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload audio files for training.
    
    Accepts individual audio files or ZIP archives.
    Files are stored directly in the model's recordings directory.
    """
    # Store uploads directly in model directory under recordings/
    model_dir = MODELS_DIR / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir = model_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = recordings_dir
    
    uploaded_files = []
    
    for file in files:
        try:
            content = await file.read()
            filename = file.filename or "upload"
            
            if filename.endswith(".zip"):
                # Extract ZIP
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for member in zf.namelist():
                        if member.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                            zf.extract(member, exp_dir)
                            uploaded_files.append(str(exp_dir / member))
            else:
                # Save individual file
                file_path = exp_dir / filename
                with open(file_path, "wb") as f:
                    f.write(content)
                uploaded_files.append(str(file_path))
                
        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {e}")
    
    return {
        "success": True,
        "exp_name": exp_name,
        "uploaded_files": len(uploaded_files),
        "upload_dir": str(exp_dir)
    }


@router.post("/start")
async def start_training(
    request: StartTrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a training job.
    
    If audio_paths not provided, uses previously uploaded files.
    """
    pipeline = get_pipeline()
    
    # Check for active training
    active_job = pipeline.get_active_training(request.exp_name)
    if active_job:
        raise HTTPException(
            status_code=409,  # Conflict
            detail={
                "message": f"Training already in progress for model '{request.exp_name}'",
                "job_id": active_job.job_id,
                "status": active_job.status.value,
                "progress": active_job.progress,
                "current_epoch": active_job.current_epoch
            }
        )
    
    # Build config
    if request.config:
        config = TrainingConfig(
            exp_name=request.exp_name,
            sample_rate=SampleRate(request.config.sample_rate),
            f0_method=F0Method(request.config.f0_method),
            epochs=request.config.epochs,
            batch_size=request.config.batch_size,
            save_every_epoch=request.config.save_every_epoch,
            version=RVCVersion(request.config.version),
            use_pitch_guidance=request.config.use_pitch_guidance
        )
    else:
        config = TrainingConfig(exp_name=request.exp_name)
    
    # Get audio paths
    audio_paths = request.audio_paths
    if not audio_paths:
        # Check recordings directory under model folder (primary location)
        recordings_dir = MODELS_DIR / request.exp_name / "recordings"
        if recordings_dir.exists():
            audio_paths = []
            for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.webm"]:
                audio_paths.extend([str(p) for p in recordings_dir.glob(f"**/{ext}")])
        
        # Fallback to legacy uploads directory if no recordings found
        if not audio_paths:
            upload_dir = UPLOAD_DIR / request.exp_name
            if upload_dir.exists():
                audio_paths = []
                for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
                    audio_paths.extend([str(p) for p in upload_dir.glob(f"**/{ext}")])
    
    if not audio_paths:
        raise HTTPException(status_code=400, detail="No audio files found")
    
    # Create job
    job_id = pipeline.create_job(config)
    
    # Start training in background
    async def run_training():
        try:
            await pipeline.train(config, audio_paths, job_id)
        except Exception as e:
            logger.exception(f"Training error: {e}")
    
    background_tasks.add_task(asyncio.create_task, run_training())
    
    return {
        "job_id": job_id,
        "status": "started",
        "exp_name": request.exp_name,
        "audio_files": len(audio_paths)
    }


@router.get("/status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status"""
    pipeline = get_pipeline()
    progress = pipeline.get_progress(job_id)
    
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return progress.to_dict()


@router.post("/cancel/{job_id}")
async def cancel_training(job_id: str):
    """Cancel a training job"""
    pipeline = get_pipeline()
    success = pipeline.cancel_job(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"job_id": job_id, "status": "cancellation_requested"}


class CheckpointRequest(BaseModel):
    """Request body for checkpoint endpoint"""
    stop_after: bool = False


@router.post("/checkpoint/{job_id}")
async def request_checkpoint(
    job_id: str,
    body: Optional[CheckpointRequest] = None,
    stop_after: bool = False
):
    """
    Request the training to save a checkpoint.
    
    Args:
        job_id: Training job ID
        stop_after: If True, stop training after saving checkpoint ("Save checkpoint & cancel")
                   If False, continue training after saving checkpoint
        body: Optional JSON body with stop_after field
    
    Returns:
        Status of the checkpoint request
    """
    # Accept stop_after from either body or query param
    should_stop = stop_after
    if body is not None and body.stop_after:
        should_stop = body.stop_after
    pipeline = get_pipeline()
    
    # Check job exists and is training
    progress = pipeline.get_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if progress.status != TrainingStatus.TRAINING:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not training (status: {progress.status.value})"
        )
    
    # Request checkpoint
    if should_stop:
        success = pipeline.request_checkpoint_and_stop(job_id)
    else:
        success = pipeline.request_checkpoint(job_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to request checkpoint")
    
    return {
        "job_id": job_id,
        "checkpoint_requested": True,
        "stop_after": should_stop,
        "status": "checkpoint_requested",
        "message": "Checkpoint will be saved at the end of the current epoch"
    }


@router.get("/checkpoint/{job_id}/status")
async def get_checkpoint_status(job_id: str):
    """
    Get the status of a pending checkpoint request.
    
    Returns:
        Response from the training subprocess if available
    """
    pipeline = get_pipeline()
    
    # Check job exists
    progress = pipeline.get_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check for checkpoint response
    response = pipeline.get_checkpoint_response(job_id)
    
    if response:
        return {
            "job_id": job_id,
            "checkpoint_completed": True,
            **response
        }
    
    return {
        "job_id": job_id,
        "checkpoint_completed": False,
        "message": "Checkpoint request pending"
    }


@router.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    pipeline = get_pipeline()
    
    jobs = []
    for job_id, progress in pipeline._jobs.items():
        jobs.append(progress.to_dict())
    
    return {"jobs": jobs}


@router.get("/active")
async def get_active_training():
    """
    Check if any training is currently active.
    
    Returns active job info if training is running, null otherwise.
    Useful for UI to show "Training in progress" state.
    """
    pipeline = get_pipeline()
    active_job = pipeline.get_active_training()
    
    if active_job:
        return {
            "active": True,
            "job": active_job.to_dict()
        }
    
    return {
        "active": False,
        "job": None
    }


@router.get("/model/{exp_name}/active")
async def get_model_active_training(exp_name: str):
    """
    Check if training is active for a specific model.
    
    Returns active job info if training is running for this model.
    """
    pipeline = get_pipeline()
    active_job = pipeline.get_active_training(exp_name)
    
    if active_job:
        return {
            "active": True,
            "job": active_job.to_dict()
        }
    
    return {
        "active": False,
        "job": None
    }


@router.get("/logs")
async def get_training_logs(
    lines: int = 200,
    job_id: Optional[str] = None,
):
    """
    Get training logs from all or a specific job.
    
    Args:
        lines: Maximum number of lines to return
        job_id: Optional job ID to filter logs
    
    Returns combined logs from all training jobs if no job_id specified.
    """
    pipeline = get_pipeline()
    all_logs = []
    
    if job_id:
        # Get logs for specific job
        progress = pipeline.get_progress(job_id)
        if progress and progress.logs:
            all_logs = progress.logs[-lines:]
    else:
        # Combine logs from all jobs (most recent first)
        for jid, progress in pipeline._jobs.items():
            if progress.logs:
                # Add job prefix to logs
                for log in progress.logs:
                    all_logs.append(f"[{progress.exp_name or jid[:8]}] {log}")
        
        # Sort by timestamp if possible, else just get last N lines
        all_logs = all_logs[-lines:]
    
    return {
        "lines": all_logs,
        "line_count": len(all_logs),
        "job_id": job_id,
    }


@router.get("/logs/{job_id}")
async def get_job_logs(job_id: str, lines: int = 500):
    """Get logs for a specific training job"""
    pipeline = get_pipeline()
    progress = pipeline.get_progress(job_id)
    
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "exp_name": progress.exp_name,
        "status": progress.status.value,
        "lines": progress.logs[-lines:] if progress.logs else [],
        "line_count": len(progress.logs) if progress.logs else 0,
    }


# ============================================================================
# Model Storage Endpoints (New Architecture)
# ============================================================================

class BulkUploadRequest(BaseModel):
    """Bulk upload request for base64 encoded files"""
    files: List[Dict[str, Any]] = Field(..., description="List of {name, data, format} dicts")
    language: str = Field(default="en", description="Language for phoneme analysis")


@router.post("/model/{exp_name}/upload-audio")
async def upload_audio_to_model(
    exp_name: str,
    files: List[UploadFile] = File(...)
):
    """
    Upload audio files directly to a model's recordings directory.
    
    This is the unified upload endpoint - all audio files go to:
    assets/models/{exp_name}/recordings/
    
    Accepts:
    - Individual audio files (wav, mp3, flac, ogg, webm)
    - ZIP archives containing audio files
    
    Returns recording info and updated stats.
    """
    storage = get_storage()
    recordings_dir = storage.get_recordings_dir(exp_name)
    
    uploaded = []
    errors = []
    
    for file in files:
        try:
            content = await file.read()
            filename = file.filename or f"upload_{len(uploaded)}"
            
            if filename.endswith(".zip"):
                # Extract ZIP
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for member in zf.namelist():
                        if member.endswith(('.wav', '.mp3', '.flac', '.ogg', '.webm')):
                            extracted_path = recordings_dir / Path(member).name
                            with zf.open(member) as src, open(extracted_path, 'wb') as dst:
                                # Convert if needed
                                audio_bytes = src.read()
                                try:
                                    audio, sr = sf.read(io.BytesIO(audio_bytes))
                                except Exception:
                                    # Try ffmpeg conversion
                                    ext = Path(member).suffix.lstrip('.')
                                    wav_bytes = convert_audio_to_wav(audio_bytes, ext)
                                    audio, sr = sf.read(io.BytesIO(wav_bytes))
                                
                                # Save as WAV
                                wav_name = Path(member).stem + '.wav'
                                wav_path = recordings_dir / wav_name
                                sf.write(str(wav_path), audio, sr)
                                uploaded.append(wav_name)
            else:
                # Process single file
                ext = Path(filename).suffix.lstrip('.') or 'wav'
                
                try:
                    audio, sr = sf.read(io.BytesIO(content))
                except Exception:
                    # Try ffmpeg conversion
                    wav_bytes = convert_audio_to_wav(content, ext)
                    audio, sr = sf.read(io.BytesIO(wav_bytes))
                
                # Save as WAV
                wav_name = Path(filename).stem + '.wav'
                wav_path = recordings_dir / wav_name
                
                # Avoid name collisions
                counter = 1
                while wav_path.exists():
                    wav_name = f"{Path(filename).stem}_{counter}.wav"
                    wav_path = recordings_dir / wav_name
                    counter += 1
                
                sf.write(str(wav_path), audio, sr)
                uploaded.append(wav_name)
                
        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {e}")
            errors.append({"file": file.filename, "error": str(e)})
    
    # Update metadata by scanning directory
    storage.scan_existing_model(exp_name)
    stats = storage.get_recording_stats(exp_name)
    
    return {
        "success": len(errors) == 0,
        "exp_name": exp_name,
        "uploaded_files": uploaded,
        "upload_count": len(uploaded),
        "errors": errors,
        "recordings_dir": str(recordings_dir),
        "stats": stats
    }


@router.post("/model/{exp_name}/upload-audio-base64")
async def upload_audio_base64(exp_name: str, request: BulkUploadRequest):
    """
    Upload audio files as base64 encoded data.
    
    Useful for browser-based recording uploads.
    Each file should have: {name, data (base64), format (optional)}
    """
    storage = get_storage()
    recordings_dir = storage.get_recordings_dir(exp_name)
    
    uploaded = []
    errors = []
    
    for i, file_data in enumerate(request.files):
        try:
            name = file_data.get('name', f'recording_{i:04d}')
            data_b64 = file_data.get('data', '')
            format_hint = file_data.get('format', 'webm')
            
            audio_bytes = base64.b64decode(data_b64)
            
            try:
                audio, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception:
                wav_bytes = convert_audio_to_wav(audio_bytes, format_hint)
                audio, sr = sf.read(io.BytesIO(wav_bytes))
            
            # Save as WAV
            wav_name = Path(name).stem + '.wav'
            wav_path = recordings_dir / wav_name
            
            counter = 1
            while wav_path.exists():
                wav_name = f"{Path(name).stem}_{counter}.wav"
                wav_path = recordings_dir / wav_name
                counter += 1
            
            sf.write(str(wav_path), audio, sr)
            uploaded.append(wav_name)
            
        except Exception as e:
            logger.error(f"Error uploading file {i}: {e}")
            errors.append({"index": i, "name": file_data.get('name'), "error": str(e)})
    
    # Update metadata
    storage.scan_existing_model(exp_name)
    stats = storage.get_recording_stats(exp_name)
    
    return {
        "success": len(errors) == 0,
        "exp_name": exp_name,
        "uploaded_files": uploaded,
        "upload_count": len(uploaded),
        "errors": errors,
        "stats": stats
    }


@router.get("/model/{exp_name}/info")
async def get_model_info(exp_name: str):
    """
    Get comprehensive info about a model including:
    - Recording stats
    - Training status
    - Phoneme coverage (if trained)
    - Category status
    """
    storage = get_storage()
    
    # Scan/update model metadata
    metadata = storage.scan_existing_model(exp_name)
    
    model_dir = storage.get_model_dir(exp_name)
    
    # Check for trained model files (exclude checkpoint files G_*, D_*)
    import re
    pth_files = list(model_dir.glob("*.pth"))
    final_model_files = [m for m in pth_files if not re.match(r'^[GD]_\d+\.pth$', m.name)]
    index_files = list(model_dir.glob("*.index"))
    
    # Check for latest generator checkpoint (sorted by step number)
    g_files = sorted(model_dir.glob("G_*.pth"), key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0)
    latest_checkpoint = str(g_files[-1]) if g_files else None
    
    # Extract actual epochs trained from train.log (checkpoint filenames use step numbers, not epochs)
    epochs_trained = metadata.training_epochs
    target_epochs_from_log = None
    train_log = model_dir / "train.log"
    if train_log.exists():
        try:
            # Parse train.log to find last completed epoch: "====> Epoch: X"
            # And also extract total_epoch from the config dump
            import re
            with open(train_log, 'r') as f:
                content = f.read()
            # Find all completed epochs
            epoch_matches = re.findall(r'====> Epoch: (\d+)', content)
            if epoch_matches:
                epochs_trained = max(int(e) for e in epoch_matches)
            # Find the most recent total_epoch from config dumps in the log
            # Format: 'total_epoch': 78 or "total_epoch": 78
            total_epoch_matches = re.findall(r"['\"]total_epoch['\"]:\s*(\d+)", content)
            if total_epoch_matches:
                # Use the last one (most recent training config)
                target_epochs_from_log = int(total_epoch_matches[-1])
        except Exception as e:
            logger.warning(f"Could not parse train.log for epochs: {e}")
    
    # Check for preprocessed data
    gt_wavs = model_dir / "0_gt_wavs"
    preprocessed_count = len(list(gt_wavs.glob("*.wav"))) if gt_wavs.exists() else 0
    
    # Get target epochs: prefer train.log value, then metadata, then config default
    target_epochs = 100  # default
    if target_epochs_from_log:
        target_epochs = target_epochs_from_log
    elif metadata.training_config:
        if "total_epoch" in metadata.training_config:
            target_epochs = metadata.training_config["total_epoch"]
        elif "train" in metadata.training_config and "epochs" in metadata.training_config["train"]:
            # Only use train.epochs if it's a reasonable value (not the default 20000)
            config_epochs = metadata.training_config["train"]["epochs"]
            if config_epochs < 1000:  # Reasonable training target
                target_epochs = config_epochs
    
    return {
        "name": exp_name,
        "model_dir": str(model_dir),
        "recordings": {
            "count": metadata.total_recordings,
            "duration_seconds": metadata.total_duration_seconds,
            "duration_minutes": round(metadata.total_duration_seconds / 60, 1)
        },
        "preprocessed": {
            "count": preprocessed_count,
            "has_data": preprocessed_count > 0
        },
        "training": {
            "has_model": metadata.has_model or len(final_model_files) > 0,
            "has_index": metadata.has_index or len(index_files) > 0,
            "epochs_trained": epochs_trained,
            "target_epochs": target_epochs,
            "last_trained": metadata.last_trained_at,
            "latest_checkpoint": latest_checkpoint,
            "checkpoint_count": len(g_files)
        },
        "coverage": {
            "phoneme_percent": metadata.phoneme_coverage_percent,
            "phonemes_covered": len(metadata.phonemes_covered),
            "phonemes_missing": len(metadata.phonemes_missing)
        },
        "categories": {
            cat_id: {
                "name": cat.name,
                "recordings": cat.recordings_count,
                "satisfied": cat.is_satisfied
            }
            for cat_id, cat in metadata.categories.items()
        },
        "metadata": metadata.to_dict()
    }


@router.get("/model/{exp_name}/recordings-list")
async def list_model_recordings(exp_name: str):
    """List all recordings for a model"""
    storage = get_storage()
    recording_paths = storage.get_all_recording_paths(exp_name)
    
    recordings = []
    for path in recording_paths:
        try:
            info = sf.info(path)
            recordings.append({
                "path": path,
                "filename": Path(path).name,
                "duration_seconds": round(info.duration, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels
            })
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            recordings.append({
                "path": path,
                "filename": Path(path).name,
                "error": str(e)
            })
    
    return {
        "exp_name": exp_name,
        "recording_count": len(recordings),
        "recordings": recordings
    }


@router.delete("/model/{exp_name}/recordings")
async def delete_model_recordings(exp_name: str):
    """Delete all recordings for a model (keeps trained model files)"""
    storage = get_storage()
    deleted = storage.delete_recordings(exp_name)
    
    return {
        "exp_name": exp_name,
        "deleted_count": deleted,
        "success": True
    }


@router.get("/models")
async def list_all_models():
    """List all models in the models directory with summary info"""
    storage = get_storage()
    model_names = storage.list_models()
    
    models = []
    for name in model_names:
        try:
            metadata = storage.load_metadata(name)
            model_dir = storage.get_model_dir(name)
            
            models.append({
                "name": name,
                "recordings": metadata.total_recordings,
                "duration_minutes": round(metadata.total_duration_seconds / 60, 1),
                "has_model": metadata.has_model or any(model_dir.glob("*.pth")),
                "has_index": metadata.has_index or any(model_dir.glob("*.index")),
                "phoneme_coverage": metadata.phoneme_coverage_percent,
                "last_trained": metadata.last_trained_at
            })
        except Exception as e:
            logger.warning(f"Could not load metadata for {name}: {e}")
            models.append({
                "name": name,
                "error": str(e)
            })
    
    return {
        "model_count": len(models),
        "models": models
    }


# ============================================================================
# Scanner Endpoints
# ============================================================================

@router.post("/scan")
async def scan_model_endpoint(request: ScanModelRequest):
    """
    Scan a model for language readiness.
    
    Returns scores and recommendations for each language.
    """
    scanner = get_scanner()
    
    try:
        result = scanner.scan_model(request.model_path, request.languages)
        return result.to_dict()
    except Exception as e:
        logger.exception(f"Scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-gaps")
async def analyze_gaps_endpoint(request: GapAnalysisRequest):
    """
    Analyze phoneme coverage gaps for a language.
    
    Returns missing phonemes and suggested prompts.
    """
    scanner = get_scanner()
    
    try:
        # First scan the model
        scan_result = scanner.scan_model(request.model_path, [request.language])
        
        # Then analyze gaps
        gaps = scanner.analyze_gaps(scan_result, request.language)
        return gaps.to_dict()
        
    except Exception as e:
        logger.exception(f"Gap analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phonemes/{language}")
async def get_language_phonemes(language: str):
    """Get phoneme set for a language"""
    if language not in LANGUAGE_PHONEMES:
        raise HTTPException(status_code=404, detail=f"Language not supported: {language}")
    
    return {
        "language": language,
        "phonemes": sorted(list(LANGUAGE_PHONEMES[language])),
        "count": len(LANGUAGE_PHONEMES[language])
    }


# ============================================================================
# Wizard Endpoints
# ============================================================================

@router.post("/wizard/sessions")
async def create_wizard_session(request: CreateSessionRequest):
    """Create a new recording wizard session"""
    wizard = get_wizard()
    
    target_phonemes = set(request.target_phonemes) if request.target_phonemes else None
    
    session = wizard.create_session(
        language=request.language,
        exp_name=request.exp_name,
        prompt_count=request.prompt_count,
        target_phonemes=target_phonemes
    )
    
    return session.to_dict()


@router.get("/wizard/sessions/{session_id}")
async def get_wizard_session(session_id: str):
    """Get wizard session details"""
    wizard = get_wizard()
    session = wizard.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()


@router.post("/wizard/sessions/{session_id}/start")
async def start_wizard_session(session_id: str):
    """Start a wizard session"""
    wizard = get_wizard()
    session = wizard.start_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()


@router.get("/wizard/sessions/{session_id}/current")
async def get_current_prompt(session_id: str):
    """Get the current prompt to record"""
    wizard = get_wizard()
    result = wizard.get_current_prompt(session_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Session not found or completed")
    
    return result


def convert_audio_to_wav(audio_bytes: bytes, format_hint: Optional[str] = None) -> bytes:
    """Convert audio to WAV format using ffmpeg.
    
    Handles webm, opus, mp3, ogg, and other formats that soundfile can't read.
    """
    import subprocess
    
    # Create temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format_hint or "webm"}') as in_file:
        in_file.write(audio_bytes)
        in_path = in_file.name
    
    out_path = in_path.rsplit('.', 1)[0] + '.wav'
    
    try:
        # Convert using ffmpeg
        result = subprocess.run([
            'ffmpeg', '-y', '-i', in_path,
            '-ar', '48000',  # Standard sample rate
            '-ac', '1',       # Mono
            '-f', 'wav',
            out_path
        ], capture_output=True, timeout=30)
        
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr.decode()}")
            raise ValueError(f"ffmpeg conversion failed: {result.stderr.decode()[:200]}")
        
        # Read converted file
        with open(out_path, 'rb') as f:
            return f.read()
    finally:
        # Cleanup
        if os.path.exists(in_path):
            os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


@router.post("/wizard/sessions/{session_id}/submit")
async def submit_recording(session_id: str, request: SubmitRecordingRequest):
    """Submit a recording for the current prompt"""
    wizard = get_wizard()
    
    # Decode audio
    try:
        audio_bytes = base64.b64decode(request.audio)
        
        # Try to read directly first (works for wav, flac, ogg vorbis)
        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            logger.debug(f"Direct audio read successful, sr={sr}, shape={audio.shape}")
        except Exception as e:
            # If direct read fails, try converting with ffmpeg (for webm, opus, etc)
            logger.debug(f"Direct read failed ({e}), trying ffmpeg conversion")
            format_hint = request.format or 'webm'
            wav_bytes = convert_audio_to_wav(audio_bytes, format_hint)
            audio, sr = sf.read(io.BytesIO(wav_bytes))
            logger.debug(f"FFmpeg conversion successful, sr={sr}, shape={audio.shape}")
        
        audio = audio.astype(np.float32)
        
        # Normalize if needed
        if len(audio) > 0 and np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
            
    except Exception as e:
        logger.exception(f"Failed to process audio: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {e}")
    
    result = wizard.submit_recording(
        session_id=session_id,
        audio=audio,
        sample_rate=request.sample_rate,
        auto_advance=request.auto_advance
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/wizard/sessions/{session_id}/next")
async def next_prompt(session_id: str):
    """Move to the next prompt"""
    wizard = get_wizard()
    result = wizard.next_prompt(session_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Session not found or at end")
    
    return result


@router.post("/wizard/sessions/{session_id}/previous")
async def previous_prompt(session_id: str):
    """Move to the previous prompt"""
    wizard = get_wizard()
    result = wizard.previous_prompt(session_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Session not found or at beginning")
    
    return result


@router.post("/wizard/sessions/{session_id}/skip")
async def skip_prompt(session_id: str):
    """Skip the current prompt"""
    wizard = get_wizard()
    result = wizard.skip_prompt(session_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return result


@router.post("/wizard/sessions/{session_id}/pause")
async def pause_session(session_id: str):
    """Pause the session"""
    wizard = get_wizard()
    session = wizard.pause_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()


@router.post("/wizard/sessions/{session_id}/complete")
async def complete_session(session_id: str):
    """Complete the session"""
    wizard = get_wizard()
    session = wizard.complete_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return wizard.get_session_summary(session_id)


@router.delete("/wizard/sessions/{session_id}")
async def cancel_session(session_id: str):
    """Cancel and delete a session"""
    wizard = get_wizard()
    session = wizard.cancel_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session_id": session_id, "status": "cancelled"}


# ============================================================================
# Model Training Data Endpoints
# ============================================================================

@router.get("/model/{exp_name}/recordings")
async def get_model_recordings(exp_name: str):
    """
    Get all recordings for a model across all wizard sessions.
    Used to gather training data from multiple recording sessions.
    """
    wizard = get_wizard()
    return wizard.get_all_recordings_for_model(exp_name)


@router.get("/model/{exp_name}/category-status/{language}")
async def get_category_status(exp_name: str, language: str):
    """
    Get recording status for each category.
    Shows how many recordings exist per category for the model.
    """
    wizard = get_wizard()
    return wizard.get_category_status(exp_name, language)


@router.get("/model/{exp_name}/training-config-preview")
async def preview_training_config(exp_name: str):
    """
    Preview the auto-generated training configuration without starting training.
    
    This lets users see what parameters will be used and understand why,
    before committing to a potentially long training run.
    """
    wizard = get_wizard()
    
    # Get all recordings for this model
    recordings_data = wizard.get_all_recordings_for_model(exp_name)
    audio_paths = recordings_data.get("audio_paths", [])
    total_duration = recordings_data.get("total_duration_seconds", 0)
    categories = recordings_data.get("categories", {})
    
    if not audio_paths:
        return {
            "ready_to_train": False,
            "error": "No recordings found. Please record some audio first or upload audio files.",
            "audio_files": 0,
            "total_duration": 0
        }
    
    # Check for existing model (including checkpoints)
    model_dir = MODELS_DIR / exp_name
    existing_model, existing_epochs = detect_existing_model(model_dir)
    
    # Get auto-configuration
    auto_config, audio_analysis = get_auto_config(
        audio_paths=audio_paths,
        source_categories={cat: data.get("audio_paths", []) for cat, data in categories.items()},
        existing_model_path=existing_model,
        gpu_memory_gb=12.0
    )
    
    # Check minimum requirements
    min_recordings_for_short = 10
    min_duration_for_long = 120
    ready_to_train = (
        len(audio_paths) >= min_recordings_for_short or 
        total_duration >= min_duration_for_long
    )
    
    return {
        "ready_to_train": ready_to_train,
        "exp_name": exp_name,
        "audio_files": len(audio_paths),
        "total_duration_seconds": total_duration,
        "total_duration_formatted": f"{int(total_duration // 60)}m {int(total_duration % 60)}s",
        "recommended_config": {
            "epochs": auto_config.epochs,
            "batch_size": auto_config.batch_size,
            "save_every_epoch": auto_config.save_every_epoch,
            "training_mode": auto_config.training_mode.value,
            "estimated_time_minutes": auto_config.estimated_minutes,
            "estimated_time_formatted": f"{int(auto_config.estimated_minutes // 60)}h {int(auto_config.estimated_minutes % 60)}m" if auto_config.estimated_minutes >= 60 else f"{int(auto_config.estimated_minutes)}m"
        },
        "audio_analysis": {
            "quality_level": audio_analysis.quality_level.value,
            "quality_description": {
                "excellent": "Studio quality - optimal for training",
                "good": "Clean audio - good for training",
                "fair": "Some noise present - acceptable for training",
                "poor": "Significant noise - may affect results"
            }.get(audio_analysis.quality_level.value, "Unknown"),
            "snr_db": round(audio_analysis.avg_snr_db, 1),
            "estimated_chunks": audio_analysis.estimated_chunks,
            "source_type": audio_analysis.source_type.value,
            "has_existing_model": audio_analysis.has_existing_model
        },
        "summary": auto_config.config_summary,
        "recommendations": auto_config.recommendations,
        "warnings": auto_config.warnings,
        "min_requirements": {
            "recordings_for_wizard": min_recordings_for_short,
            "duration_for_uploads_seconds": min_duration_for_long,
            "met": ready_to_train
        }
    }


@router.post("/model/{exp_name}/train")
async def train_model(
    exp_name: str,
    background_tasks: BackgroundTasks,
    config: Optional[TrainingConfigInput] = None
):
    """
    Start training a model using all collected recordings from wizard sessions.
    
    Training parameters are automatically optimized based on:
    - Audio duration and number of samples
    - Audio quality (SNR estimation)
    - Whether it's a new model or fine-tuning existing
    - Source type (wizard recordings vs uploads)
    
    You can override auto-config by providing explicit config values.
    """
    wizard = get_wizard()
    pipeline = get_pipeline()
    
    # Check for active training (for this model or any model)
    active_job = pipeline.get_active_training(exp_name)
    if active_job:
        raise HTTPException(
            status_code=409,  # Conflict
            detail={
                "message": f"Training already in progress for model '{exp_name}'",
                "job_id": active_job.job_id,
                "status": active_job.status.value,
                "progress": active_job.progress,
                "current_epoch": active_job.current_epoch,
                "step": active_job.step
            }
        )
    
    # Also check if any other training is running (single GPU constraint)
    any_active = pipeline.get_active_training()
    if any_active:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"Another training is already in progress (model: '{any_active.exp_name}'). Please wait for it to complete or cancel it.",
                "job_id": any_active.job_id,
                "exp_name": any_active.exp_name,
                "status": any_active.status.value,
                "progress": any_active.progress
            }
        )
    
    # Get all recordings for this model (includes wizard recordings AND uploaded files)
    recordings_data = wizard.get_all_recordings_for_model(exp_name)
    audio_paths = recordings_data.get("audio_paths", [])
    total_duration = recordings_data.get("total_duration_seconds", 0)
    categories = recordings_data.get("categories", {})
    
    logger.info(f"Training request for {exp_name}: {len(audio_paths)} files, {total_duration}s total")
    
    if not audio_paths:
        raise HTTPException(
            status_code=400, 
            detail="No recordings found. Please record some audio first or upload audio files."
        )
    
    # Allow training if:
    # 1. We have at least 10 short recordings (from wizard), OR
    # 2. We have at least 2 minutes of audio (for uploaded longer files)
    # This supports both use cases: many short prompts OR few long audio files
    min_recordings_for_short = 10
    min_duration_for_long = 120  # 2 minutes
    
    if len(audio_paths) < min_recordings_for_short and total_duration < min_duration_for_long:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {min_recordings_for_short} recordings OR {min_duration_for_long // 60} minutes of audio. "
                   f"Currently have {len(audio_paths)} files with {total_duration:.0f}s total."
        )
    
    # Check for existing model (including checkpoints for resume/fine-tune)
    model_dir = MODELS_DIR / exp_name
    existing_model, existing_epochs = detect_existing_model(model_dir)
    
    # Determine training mode from config if provided
    training_mode = None
    if config and hasattr(config, 'training_mode') and config.training_mode:
        training_mode = TrainingMode(config.training_mode)
    
    # ==========================================================================
    # AUTO-CONFIGURATION
    # ==========================================================================
    
    # Get automatic configuration based on audio analysis
    auto_config, audio_analysis = get_auto_config(
        audio_paths=audio_paths,
        source_categories={cat: data.get("audio_paths", []) for cat, data in categories.items()},
        existing_model_path=existing_model,
        training_mode=training_mode,
        gpu_memory_gb=12.0  # RTX 4070 Ti has 12GB
    )
    
    logger.info(f"Auto-config for {exp_name}: {auto_config.config_summary}")
    logger.info(f"Audio analysis: quality={audio_analysis.quality_level.value}, "
                f"SNR={audio_analysis.avg_snr_db:.1f}dB, "
                f"chunks={audio_analysis.estimated_chunks}")
    
    # Build training config - use auto values unless explicitly overridden
    training_config = TrainingConfig(
        exp_name=exp_name,
        sample_rate=SampleRate(config.sample_rate) if config and config.sample_rate else SampleRate.SR_48K,
        f0_method=F0Method(config.f0_method) if config and config.f0_method else F0Method.RMVPE,
        # Use auto-calculated epochs unless user specified
        epochs=config.epochs if config and config.epochs else auto_config.epochs,
        batch_size=config.batch_size if config and config.batch_size else auto_config.batch_size,
        save_every_epoch=config.save_every_epoch if config and config.save_every_epoch else auto_config.save_every_epoch,
        version=RVCVersion(config.version) if config and config.version else RVCVersion.V2,
        use_pitch_guidance=config.use_pitch_guidance if config and config.use_pitch_guidance is not None else True
    )
    
    logger.info(f"Final training config for {exp_name}: epochs={training_config.epochs}, batch_size={training_config.batch_size}, save_every={training_config.save_every_epoch}")
    
    # Create job
    job_id = pipeline.create_job(training_config)
    
    # Start training in background
    # NOTE: Pass the coroutine function itself, NOT the result of calling it
    # BackgroundTasks will call and await the coroutine
    async def run_training():
        try:
            logger.info(f"Starting training job {job_id} for model {exp_name}")
            result = await pipeline.train(training_config, audio_paths, job_id)
            
            # Copy final model files to the models directory
            if result.success:
                logs_exp_dir = LOGS_DIR / exp_name
                models_exp_dir = MODELS_DIR / exp_name
                models_exp_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy model files (.pth, .index, metadata)
                # Handle symlinks by comparing resolved paths to avoid SameFileError
                import shutil
                for pattern in ["*.pth", "*.index", "*.json"]:
                    for src_file in logs_exp_dir.glob(pattern):
                        dest_file = models_exp_dir / src_file.name
                        try:
                            # Check if source and dest are the same file (can happen with symlinks)
                            if src_file.resolve() == dest_file.resolve():
                                logger.debug(f"Skipping {src_file.name} - source and dest are same file")
                                continue
                            shutil.copy2(src_file, dest_file)
                            logger.info(f"Copied {src_file.name} to {models_exp_dir}")
                        except shutil.SameFileError:
                            logger.debug(f"Skipping {src_file.name} - already exists at destination")
                        except Exception as e:
                            logger.warning(f"Failed to copy {src_file.name}: {e}")
                
                logger.info(f"Training job {job_id} completed successfully, model saved to {models_exp_dir}")
            else:
                logger.error(f"Training job {job_id} failed: {result.error}")
        except Exception as e:
            logger.exception(f"Training error for job {job_id}: {e}")
            # Update job status to failed
            try:
                pipeline.update_job_status(job_id, "failed", error=str(e))
            except Exception:
                pass
    
    # Correctly add the async function - BackgroundTasks handles awaiting
    background_tasks.add_task(run_training)
    
    return {
        "job_id": job_id,
        "status": "started",
        "exp_name": exp_name,
        "audio_files": len(audio_paths),
        "total_duration": recordings_data.get("total_duration_seconds", 0),
        "config": {
            "epochs": training_config.epochs,
            "batch_size": training_config.batch_size,
            "sample_rate": training_config.sample_rate.value,
            "save_every_epoch": training_config.save_every_epoch
        },
        "auto_config": {
            "summary": auto_config.config_summary,
            "training_mode": auto_config.training_mode.value,
            "estimated_minutes": auto_config.estimated_minutes,
            "recommendations": auto_config.recommendations,
            "warnings": auto_config.warnings
        },
        "audio_analysis": {
            "quality_level": audio_analysis.quality_level.value,
            "snr_db": round(audio_analysis.avg_snr_db, 1),
            "estimated_chunks": audio_analysis.estimated_chunks,
            "source_type": audio_analysis.source_type.value
        }
    }


# ============================================================================
# Prompt Endpoints
# ============================================================================

@router.get("/prompts/languages")
async def list_languages():
    """List available languages"""
    return {
        "languages": get_available_languages()
    }


@router.get("/prompts/{language}")
async def get_prompts(language: str):
    """Get prompts for a language"""
    loader = get_prompt_loader()
    prompts = loader.get_language(language)
    
    if not prompts:
        raise HTTPException(status_code=404, detail=f"Language not found: {language}")
    
    return {
        "language": prompts.language,
        "language_name": prompts.language_name,
        "total_prompts": prompts.total_prompts,
        "categories": {
            name: {
                "description": cat.description,
                "prompt_count": len(cat.prompts),
                "phonemes_covered": cat.phonemes_covered
            }
            for name, cat in prompts.categories.items()
        }
    }


@router.get("/prompts/{language}/{category}")
async def get_category_prompts(language: str, category: str):
    """Get prompts for a specific category"""
    loader = get_prompt_loader()
    prompts = loader.get_language(language)
    
    if not prompts:
        raise HTTPException(status_code=404, detail=f"Language not found: {language}")
    
    if category not in prompts.categories:
        raise HTTPException(status_code=404, detail=f"Category not found: {category}")
    
    cat = prompts.categories[category]
    return {
        "language": language,
        "category": category,
        "description": cat.description,
        "phonemes_covered": cat.phonemes_covered,
        "prompts": cat.prompts
    }


@router.post("/prompts/{language}/for-phonemes")
async def get_prompts_for_phonemes(language: str, phonemes: List[str]):
    """Get prompts that cover specific phonemes"""
    loader = get_prompt_loader()
    prompts = loader.get_prompts_for_missing_phonemes(
        language=language,
        missing_phonemes=set(phonemes),
        max_prompts=30
    )
    
    return {
        "language": language,
        "target_phonemes": phonemes,
        "prompts": prompts
    }


# ============================================================================
# Model Inference Testing
# ============================================================================

class InferenceTestRequest(BaseModel):
    """Request for inference-based model testing"""
    model_path: str = Field(..., description="Path to model .pth file")
    index_path: Optional[str] = Field(default=None, description="Path to .index file")
    test_sentences: Optional[List[str]] = Field(
        default=None, 
        description="Custom test sentences (uses defaults if not provided)"
    )
    languages: List[str] = Field(default=["en"], description="Languages to test")
    voice: str = Field(default="en-US-GuyNeural", description="TTS voice for test audio")


class InferenceTestResult(BaseModel):
    """Result from inference test"""
    model_path: str
    model_name: str
    overall_score: float
    language_scores: Dict[str, Any]
    test_details: List[Dict[str, Any]]
    recommendations: List[str]


# Default test sentences covering various phonemes
DEFAULT_TEST_SENTENCES = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
        "Peter Piper picked a peck of pickled peppers.",
        "Hello, how are you doing today?",
        "I really appreciate your help with this project.",
        "The weather is absolutely beautiful this morning.",
        "Can you please pass me the salt and pepper?",
        "What time does the train arrive at the station?",
        "I'm looking forward to meeting you tomorrow."
    ],
    "is": [
        "Góðan daginn, hvað segirðu gott?",
        "Veðrið er mjög fallegt í dag.",
        "Þetta er frábært, takk fyrir hjálpina.",
        "Hvar er næsta strætóstöð?",
        "Ég heiti Jón og ég er frá Íslandi."
    ]
}


@router.post("/test", response_model=InferenceTestResult)
async def test_model_inference(request: InferenceTestRequest):
    """
    Test a model's performance using inference.
    
    This endpoint:
    1. Generates test audio using TTS for each test sentence
    2. Runs each through the voice model
    3. Analyzes the output quality (pitch stability, audio clarity, artifacts)
    4. Returns scores and recommendations
    
    Useful for models without training data to assess quality.
    """
    import librosa
    from scipy import signal
    
    try:
        from app.http_api import model_manager
    except ImportError:
        raise HTTPException(
            status_code=500, 
            detail="Model manager not available for inference testing"
        )
    
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    model_path = Path(request.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
    
    # Get test sentences
    test_sentences = {}
    for lang in request.languages:
        if request.test_sentences:
            test_sentences[lang] = request.test_sentences
        else:
            test_sentences[lang] = DEFAULT_TEST_SENTENCES.get(lang, DEFAULT_TEST_SENTENCES["en"])
    
    # Load the model
    try:
        success = model_manager.load_model(str(model_path), request.index_path)
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to load model: {model_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")
    
    test_results = []
    language_scores = {}
    
    for lang, sentences in test_sentences.items():
        lang_results = []
        
        for sentence in sentences[:5]:  # Limit to 5 sentences per language
            try:
                # Generate TTS audio
                tts_result = await generate_tts_for_test(sentence, request.voice)
                if not tts_result:
                    continue
                
                tts_audio, tts_sr = tts_result
                
                # Resample to 16kHz for RVC
                if tts_sr != 16000:
                    tts_audio = librosa.resample(tts_audio, orig_sr=tts_sr, target_sr=16000)
                
                # Run inference
                from app.model_manager import RVCInferParams
                params = RVCInferParams(
                    f0_up_key=0,
                    f0_method="rmvpe",
                    index_rate=0.5,
                    filter_radius=3,
                    rms_mix_rate=0.2,
                    protect=0.4
                )
                
                output_audio = model_manager.infer(tts_audio, params=params)
                
                if output_audio is None or len(output_audio) == 0:
                    lang_results.append({
                        "sentence": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                        "success": False,
                        "error": "Inference returned empty audio"
                    })
                    continue
                
                # Analyze output quality
                quality_score = _analyze_audio_quality(output_audio, 16000)
                
                lang_results.append({
                    "sentence": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                    "success": True,
                    "quality_score": quality_score["overall"],
                    "pitch_stability": quality_score["pitch_stability"],
                    "audio_clarity": quality_score["clarity"],
                    "artifact_score": quality_score["artifacts"],
                    "duration_seconds": len(output_audio) / 16000
                })
                
            except Exception as e:
                logger.error(f"Error testing sentence '{sentence[:30]}...': {e}")
                lang_results.append({
                    "sentence": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate language score
        successful_tests = [r for r in lang_results if r.get("success", False)]
        if successful_tests:
            avg_quality = np.mean([r["quality_score"] for r in successful_tests])
            avg_pitch = np.mean([r["pitch_stability"] for r in successful_tests])
            avg_clarity = np.mean([r["audio_clarity"] for r in successful_tests])
            avg_artifacts = np.mean([r["artifact_score"] for r in successful_tests])
            
            language_scores[lang] = {
                "overall_score": float(round(avg_quality, 1)),
                "pitch_stability": float(round(avg_pitch, 1)),
                "audio_clarity": float(round(avg_clarity, 1)),
                "artifact_score": float(round(avg_artifacts, 1)),
                "tests_run": len(lang_results),
                "tests_passed": len(successful_tests)
            }
        else:
            language_scores[lang] = {
                "overall_score": 0,
                "pitch_stability": 0,
                "audio_clarity": 0,
                "artifact_score": 0,
                "tests_run": len(lang_results),
                "tests_passed": 0
            }
        
        test_results.extend(lang_results)
    
    # Calculate overall score
    if language_scores:
        overall_score = np.mean([s["overall_score"] for s in language_scores.values()])
    else:
        overall_score = 0
    
    # Generate recommendations
    recommendations = _generate_inference_recommendations(language_scores, test_results)
    
    return InferenceTestResult(
        model_path=str(model_path),
        model_name=model_path.stem,
        overall_score=float(round(overall_score, 1)),
        language_scores=language_scores,
        test_details=test_results,
        recommendations=recommendations
    )


async def generate_tts_for_test(text: str, voice: str) -> Optional[Tuple[np.ndarray, int]]:
    """Generate TTS audio for testing"""
    try:
        import edge_tts
        import io
        import soundfile as sf
        
        communicate = edge_tts.Communicate(text, voice)
        audio_data = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        audio_data.seek(0)
        
        # Read the MP3 and convert to numpy array
        import librosa
        audio, sr = librosa.load(audio_data, sr=None, mono=True)
        
        return audio.astype(np.float32), sr
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return None


def _analyze_audio_quality(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Analyze audio quality metrics.
    
    Returns scores 0-100 for:
    - pitch_stability: How stable the pitch is (higher = more stable)
    - clarity: Audio clarity/SNR estimation
    - artifacts: Artifact detection (higher = fewer artifacts)
    - overall: Combined score
    """
    import librosa
    from scipy import signal
    
    try:
        # Pitch stability analysis
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 10:
            pitch_std = np.std(pitch_values)
            pitch_mean = np.mean(pitch_values)
            # Coefficient of variation - lower is more stable
            pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 1.0
            pitch_stability = max(0, min(100, 100 - pitch_cv * 100))
        else:
            pitch_stability = 50.0
        
        # Audio clarity (based on spectral centroid consistency)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_std = np.std(spectral_centroids)
        centroid_mean = np.mean(spectral_centroids)
        clarity = max(0, min(100, 100 - (centroid_std / centroid_mean * 50) if centroid_mean > 0 else 50))
        
        # Artifact detection (look for sudden amplitude changes)
        rms = librosa.feature.rms(y=audio)[0]
        rms_diff = np.abs(np.diff(rms))
        artifact_count = np.sum(rms_diff > np.mean(rms_diff) * 3)  # Spikes > 3x average
        max_artifacts = len(rms_diff) * 0.1  # 10% of frames
        artifact_score = max(0, min(100, 100 - (artifact_count / max_artifacts * 100) if max_artifacts > 0 else 0))
        
        # Overall score (weighted average)
        overall = pitch_stability * 0.4 + clarity * 0.3 + artifact_score * 0.3
        
        return {
            "pitch_stability": float(round(pitch_stability, 1)),
            "clarity": float(round(clarity, 1)),
            "artifacts": float(round(artifact_score, 1)),
            "overall": float(round(overall, 1))
        }
        
    except Exception as e:
        logger.error(f"Audio quality analysis failed: {e}")
        return {
            "pitch_stability": 50.0,
            "clarity": 50.0,
            "artifacts": 50.0,
            "overall": 50.0
        }


def _generate_inference_recommendations(
    language_scores: Dict[str, Any],
    test_results: List[Dict[str, Any]]
) -> List[str]:
    """Generate recommendations based on inference test results"""
    recommendations = []
    
    for lang, scores in language_scores.items():
        overall = scores.get("overall_score", 0)
        pitch = scores.get("pitch_stability", 0)
        clarity = scores.get("audio_clarity", 0)
        artifacts = scores.get("artifact_score", 0)
        
        if overall < 40:
            recommendations.append(
                f"Model shows poor performance in {lang.upper()}. "
                "Consider training with more diverse audio samples."
            )
        elif overall < 60:
            recommendations.append(
                f"Model has moderate {lang.upper()} performance. "
                "Additional training data could improve quality."
            )
        
        if pitch < 50:
            recommendations.append(
                f"Pitch instability detected in {lang.upper()} output. "
                "Try using pitch correction or training with more consistent source audio."
            )
        
        if clarity < 50:
            recommendations.append(
                f"Audio clarity is low for {lang.upper()}. "
                "Ensure training audio is clean without background noise."
            )
        
        if artifacts < 50:
            recommendations.append(
                f"Audio artifacts detected in {lang.upper()} output. "
                "May indicate model overfitting or insufficient training data."
            )
    
    # Check failure rate
    failed = [r for r in test_results if not r.get("success", False)]
    if len(failed) > len(test_results) * 0.5:
        recommendations.append(
            "High failure rate during testing. Check model file integrity and compatibility."
        )
    
    if not recommendations:
        recommendations.append(
            "Model performs well across all tested metrics. Ready for production use."
        )
    
    return recommendations


# ============================================================================
# Model Extraction & Index Building
# ============================================================================

class ModelConfigResponse(BaseModel):
    """Response with model's training configuration (sample rate, version)"""
    model_dir: str
    sample_rate: Optional[str] = None
    sample_rate_hz: Optional[int] = None
    version: Optional[str] = None
    config_found: bool = False
    has_checkpoints: bool = False
    has_final_model: bool = False
    has_index: bool = False
    message: str = ""


@router.get("/model/{model_dir:path}/config", response_model=ModelConfigResponse)
async def get_model_config(model_dir: str):
    """
    Get the training configuration for a model directory.
    
    Auto-detects sample_rate and version from config.json if present.
    This is useful for pre-populating the Extract & Index form.
    """
    import json
    import re
    
    # Resolve model directory
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = MODELS_DIR / model_dir
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")
    
    result = {
        "model_dir": model_dir,
        "sample_rate": None,
        "sample_rate_hz": None,
        "version": None,
        "config_found": False,
        "has_checkpoints": False,
        "has_final_model": False,
        "has_index": False,
        "message": "",
    }
    
    # Check for checkpoints
    g_checkpoints = list(model_path.glob("G_*.pth"))
    result["has_checkpoints"] = len(g_checkpoints) > 0
    
    # Check for final model (*.pth that isn't G_*.pth or D_*.pth)
    all_pth = list(model_path.glob("*.pth"))
    final_models = [f for f in all_pth if not re.match(r'^[GD]_\d+\.pth$', f.name)]
    result["has_final_model"] = len(final_models) > 0
    
    # Check for index
    indexes = list(model_path.glob("*.index"))
    result["has_index"] = len(indexes) > 0
    
    # Try to read config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        result["config_found"] = True
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract sample rate
            data_sr = config.get("data", {}).get("sampling_rate")
            if data_sr:
                result["sample_rate_hz"] = data_sr
                sr_map = {32000: "32k", 40000: "40k", 48000: "48k"}
                result["sample_rate"] = sr_map.get(data_sr)
            
            # Detect version from model architecture
            model_config = config.get("model", {})
            upsample_rates = model_config.get("upsample_rates", [])
            if upsample_rates == [12, 10, 2, 2]:  # v2 48k pattern
                result["version"] = "v2"
            elif upsample_rates == [10, 10, 2, 2]:  # v2 40k pattern
                result["version"] = "v2"
            elif len(upsample_rates) == 5:  # v1 patterns have 5 stages
                result["version"] = "v1"
            else:
                result["version"] = "v2"  # Default to v2
            
            result["message"] = f"Config found: {result['sample_rate']} ({data_sr}Hz), {result['version']}"
        except Exception as e:
            result["message"] = f"Config found but failed to parse: {e}"
    else:
        result["message"] = "No config.json found - using defaults (48k, v2)"
        result["sample_rate"] = "48k"
        result["version"] = "v2"
    
    return ModelConfigResponse(**result)


class ExtractModelRequest(BaseModel):
    """Request to extract a model from checkpoint and/or build FAISS index"""
    model_dir: str = Field(..., description="Path to model directory (absolute or relative to models dir)")
    extract_model: bool = Field(default=True, description="Extract .pth from G_*.pth checkpoint")
    build_index: bool = Field(default=True, description="Build FAISS index from features")
    sample_rate: Optional[str] = Field(default=None, description="Sample rate: 32k, 40k, or 48k. If not provided, auto-detected from config.json")
    version: Optional[str] = Field(default=None, description="RVC version: v1 or v2. If not provided, auto-detected from config.json")
    model_name: Optional[str] = Field(default=None, description="Custom model name (defaults to directory name)")


class ExtractModelResponse(BaseModel):
    """Response from model extraction"""
    success: bool
    model_path: Optional[str] = None
    index_path: Optional[str] = None
    message: str
    details: Dict[str, Any] = {}



@router.post("/model/extract", response_model=ExtractModelResponse)
async def extract_model_and_build_index(request: ExtractModelRequest):
    """
    Extract a final model from training checkpoint and/or build FAISS index.
    
    This endpoint handles two operations:
    1. Extract model: Convert G_*.pth checkpoint to final {name}.pth model
    2. Build index: Create FAISS IVF index from training features
    
    Use this when:
    - Training completed but final model wasn't extracted
    - You have a G_*.pth checkpoint and need a usable model
    - Index file is missing and needs to be rebuilt
    """
    import faiss
    import torch
    import re
    from collections import OrderedDict
    
    # Resolve model directory
    model_dir = Path(request.model_dir)
    if not model_dir.is_absolute():
        model_dir = MODELS_DIR / request.model_dir
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")
    
    model_name = request.model_name or model_dir.name
    results = {"extracted": False, "indexed": False}
    final_model_path = None
    final_index_path = None
    
    # Auto-detect sample_rate and version from config.json if not provided
    config_path = model_dir / "config.json"
    detected_sr = None
    detected_version = None
    
    if config_path.exists():
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract sample rate from config
            data_sr = config.get("data", {}).get("sampling_rate")
            if data_sr:
                sr_map = {32000: "32k", 40000: "40k", 48000: "48k"}
                detected_sr = sr_map.get(data_sr)
                if detected_sr:
                    logger.info(f"Auto-detected sample rate from config.json: {detected_sr} ({data_sr}Hz)")
            
            # Try to detect version from model architecture
            # v2 uses 768 feature dimensions, v1 uses 256
            model_config = config.get("model", {})
            # v2 typically has different upsample_rates
            upsample_rates = model_config.get("upsample_rates", [])
            if upsample_rates == [12, 10, 2, 2]:  # v2 48k pattern
                detected_version = "v2"
            elif upsample_rates == [10, 10, 2, 2]:  # v2 40k pattern
                detected_version = "v2"
            elif len(upsample_rates) == 5:  # v1 patterns have 5 stages
                detected_version = "v1"
            else:
                detected_version = "v2"  # Default to v2
            
            logger.info(f"Auto-detected version from config.json: {detected_version}")
        except Exception as e:
            logger.warning(f"Failed to parse config.json for auto-detection: {e}")
    
    # Use provided values or fall back to detected/defaults
    sr = request.sample_rate or detected_sr or "48k"
    version = request.version or detected_version or "v2"
    
    results["sample_rate"] = sr
    results["version"] = version
    results["auto_detected"] = {
        "sample_rate": detected_sr,
        "version": detected_version,
        "config_found": config_path.exists()
    }
    
    try:
        # Step 1: Extract model from checkpoint
        if request.extract_model:
            # Find the largest G_*.pth checkpoint
            g_checkpoints = list(model_dir.glob("G_*.pth"))
            
            if not g_checkpoints:
                # Check if final model already exists
                existing_models = [f for f in model_dir.glob("*.pth") 
                                   if not re.match(r'^[GD]_\d+\.pth$', f.name)]
                if existing_models:
                    final_model_path = str(existing_models[0])
                    results["extracted"] = True
                    results["model_existed"] = True
                    logger.info(f"Final model already exists: {final_model_path}")
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail="No G_*.pth checkpoints found and no existing model"
                    )
            else:
                # Find largest step checkpoint
                def extract_step(path: Path) -> int:
                    match = re.search(r'G_(\d+)\.pth', path.name)
                    return int(match.group(1)) if match else 0
                
                best_checkpoint = max(g_checkpoints, key=extract_step)
                logger.info(f"Extracting model from checkpoint: {best_checkpoint}")
                
                # Load checkpoint
                ckpt = torch.load(str(best_checkpoint), map_location="cpu", weights_only=False)
                
                # Extract weights (remove discriminator-only weights)
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                
                opt = OrderedDict()
                opt["weight"] = {}
                for key in ckpt.keys():
                    if "enc_q" in key:  # Skip encoder query weights
                        continue
                    opt["weight"][key] = ckpt[key].half()
                
                # Set config based on sample rate and version (using auto-detected values)
                if sr == "40k":
                    opt["config"] = [
                        1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                        [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                        [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000
                    ]
                elif sr == "48k":
                    if version == "v1":
                        opt["config"] = [
                            1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                            [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                            [10, 6, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 48000
                        ]
                    else:  # v2
                        opt["config"] = [
                            1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                            [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                            [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000
                        ]
                else:  # 32k
                    opt["config"] = [
                        513, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                        [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                        [10, 4, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 32000
                    ]
                
                # Set sample rate as integer
                sr_map = {"32k": 32000, "40k": 40000, "48k": 48000}
                opt["sr"] = sr_map.get(sr, 48000)
                opt["f0"] = 1  # Pitch guidance enabled
                opt["version"] = version
                opt["info"] = f"Extracted from {best_checkpoint.name}"
                
                # Save final model
                final_model_path = str(model_dir / f"{model_name}.pth")
                torch.save(opt, final_model_path)
                
                results["extracted"] = True
                results["checkpoint_used"] = best_checkpoint.name
                results["step"] = extract_step(best_checkpoint)
                logger.info(f"Extracted model saved to: {final_model_path}")
        
        # Step 2: Build FAISS index
        if request.build_index:
            # version already resolved above (auto-detected or provided)
            feature_dim = 768 if version == "v2" else 256
            feature_dir = model_dir / f"3_feature{feature_dim}"
            
            # Check if index already exists
            existing_indexes = list(model_dir.glob("added_IVF*_Flat_nprobe_*.index"))
            
            if not feature_dir.exists() or not any(feature_dir.glob("*.npy")):
                if existing_indexes:
                    final_index_path = str(existing_indexes[0])
                    results["indexed"] = True
                    results["index_existed"] = True
                    logger.info(f"Index already exists: {final_index_path}")
                else:
                    logger.warning(f"No features found in {feature_dir} and no existing index")
                    results["index_error"] = f"No features found in {feature_dir}"
            else:
                # Collect all features
                all_features = []
                for npy_file in feature_dir.glob("*.npy"):
                    features = np.load(str(npy_file))
                    all_features.append(features)
                
                if all_features:
                    # Concatenate features
                    big_npy = np.concatenate(all_features, axis=0).astype(np.float32)
                    logger.info(f"Building index from {big_npy.shape[0]} feature vectors")
                    
                    # Save total features
                    np.save(str(model_dir / "total_fea.npy"), big_npy)
                    
                    # Calculate IVF clusters
                    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
                    n_ivf = max(n_ivf, 1)
                    
                    # Build index
                    index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
                    index.train(big_npy)
                    index.add(big_npy)
                    
                    # Save index
                    final_index_path = str(
                        model_dir / f"added_IVF{n_ivf}_Flat_nprobe_1_{model_name}_{version}.index"
                    )
                    faiss.write_index(index, final_index_path)
                    
                    results["indexed"] = True
                    results["n_vectors"] = big_npy.shape[0]
                    results["n_ivf_clusters"] = n_ivf
                    logger.info(f"Index saved to: {final_index_path}")
                else:
                    results["index_error"] = "No feature vectors found"
        
        # Build response
        success = results.get("extracted", False) or results.get("indexed", False)
        
        messages = []
        if results.get("extracted"):
            if results.get("model_existed"):
                messages.append(f"Model already exists")
            else:
                messages.append(f"Model extracted from step {results.get('step', 'unknown')}")
        if results.get("indexed"):
            if results.get("index_existed"):
                messages.append(f"Index already exists")
            else:
                messages.append(f"Index built with {results.get('n_vectors', 0)} vectors, {results.get('n_ivf_clusters', 0)} IVF clusters")
        
        return ExtractModelResponse(
            success=success,
            model_path=final_model_path,
            index_path=final_index_path,
            message=" | ".join(messages) if messages else "No operations performed",
            details=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Model extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def trainer_health():
    """Health check for trainer API"""
    return {
        "status": "healthy",
        "components": {
            "pipeline": _training_pipeline is not None,
            "scanner": _model_scanner is not None,
            "wizard": _recording_wizard is not None
        },
        "available_languages": get_available_languages()
    }
