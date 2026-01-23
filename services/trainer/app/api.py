"""
Trainer Service - FastAPI Routes
API endpoints for training job management
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .jobs import TrainingJob, TrainingStatus, job_manager, cleanup_jobs
from .training import executor

logger = logging.getLogger(__name__)


async def trigger_preprocessing(exp_name: str, sample_rate: int = 48000, version: str = "v2") -> tuple[bool, str, Optional[str]]:
    """
    Trigger preprocessing via the preprocessor service.
    
    Returns:
        (success, message, job_id)
    """
    preprocessor_url = settings.preprocessor_url
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{preprocessor_url}/api/v1/preprocess/start",
                json={
                    "exp_name": exp_name,
                    "sample_rate": sample_rate,
                    "version": version,
                    "n_threads": 4
                }
            )
            if response.status_code == 200:
                data = response.json()
                return True, data.get("message", "Preprocessing started"), data.get("job_id")
            elif response.status_code == 409:
                # Already in progress
                return True, "Preprocessing already in progress", None
            else:
                return False, f"Preprocessor returned {response.status_code}: {response.text}", None
    except httpx.ConnectError:
        return False, f"Could not connect to preprocessor service at {preprocessor_url}", None
    except Exception as e:
        return False, f"Error calling preprocessor: {e}", None


async def wait_for_preprocessing(job_id: str, timeout: int = 600) -> tuple[bool, str]:
    """
    Wait for preprocessing job to complete.
    
    Args:
        job_id: Preprocessing job ID
        timeout: Max seconds to wait (default 10 minutes)
        
    Returns:
        (success, message)
    """
    preprocessor_url = settings.preprocessor_url
    start_time = asyncio.get_event_loop().time()
    
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            return False, f"Preprocessing timed out after {timeout} seconds"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{preprocessor_url}/api/v1/preprocess/status/{job_id}")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    progress = data.get("progress", 0)
                    
                    if status == "completed":
                        return True, "Preprocessing completed successfully"
                    elif status == "failed":
                        return False, f"Preprocessing failed: {data.get('error', 'Unknown error')}"
                    else:
                        logger.info(f"Preprocessing progress: {progress}% ({status})")
                else:
                    return False, f"Failed to get preprocessing status: {response.status_code}"
        except Exception as e:
            logger.warning(f"Error checking preprocessing status: {e}")
        
        await asyncio.sleep(3)  # Poll every 3 seconds

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
        config = job.config
        
        # Step 1: Validate preprocessing
        job.status = TrainingStatus.VALIDATING
        job.log("Validating preprocessing outputs...")
        
        valid, message = await executor.validate_preprocessing(job.exp_name)
        
        if not valid:
            # Try auto-triggering preprocessing
            job.log(f"Preprocessing not found: {message}")
            job.log("Auto-triggering preprocessing...")
            
            success, preprocess_msg, preprocess_job_id = await trigger_preprocessing(
                job.exp_name,
                sample_rate=config.get("sample_rate", 48000),
                version=config.get("version", "v2")
            )
            
            if not success:
                job.status = TrainingStatus.FAILED
                job.error = f"Failed to start preprocessing: {preprocess_msg}"
                job.log(job.error)
                return
            
            job.log(f"Preprocessing started: {preprocess_msg}")
            
            if preprocess_job_id:
                # Wait for preprocessing to complete
                job.log("Waiting for preprocessing to complete...")
                success, wait_msg = await wait_for_preprocessing(preprocess_job_id)
                
                if not success:
                    job.status = TrainingStatus.FAILED
                    job.error = wait_msg
                    job.log(wait_msg)
                    return
                
                job.log(wait_msg)
                
                # Re-validate
                valid, message = await executor.validate_preprocessing(job.exp_name)
                if not valid:
                    job.status = TrainingStatus.FAILED
                    job.error = f"Still invalid after preprocessing: {message}"
                    job.log(job.error)
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


# ============================================================================
# Prompts & Phonemes Endpoints (for training wizard)
# ============================================================================

# English phonemes (IPA-based)
PHONEMES = {
    "en": {
        "consonants": [
            "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ",
            "h", "tʃ", "dʒ", "m", "n", "ŋ", "l", "r", "w", "j"
        ],
        "vowels": [
            "iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ʌ", "ɜː", "ə",
            "eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "ɪə", "eə", "ʊə"
        ],
        "description": "English (US/UK)"
    },
    "is": {
        "consonants": [
            "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "h",
            "m", "n", "ŋ", "l", "r", "j", "x", "c"
        ],
        "vowels": [
            "a", "á", "e", "é", "i", "í", "o", "ó", "u", "ú", "y", "ý",
            "æ", "ö", "au", "ei", "ey"
        ],
        "description": "Icelandic"
    }
}

# Training prompts for phoneme coverage
PROMPTS = {
    "en": [
        {"id": 1, "text": "The quick brown fox jumps over the lazy dog.", "phonemes": ["θ", "k", "w", "ɪ", "k", "b", "r", "aʊ", "n", "f", "ɒ", "k", "s", "dʒ", "ʌ", "m", "p", "s", "əʊ", "v", "ə", "ð", "ə", "l", "eɪ", "z", "i", "d", "ɒ", "g"]},
        {"id": 2, "text": "She sells seashells by the seashore.", "phonemes": ["ʃ", "iː", "s", "e", "l", "z", "s", "iː", "ʃ", "e", "l", "z", "b", "aɪ", "ð", "ə", "s", "iː", "ʃ", "ɔː", "r"]},
        {"id": 3, "text": "How much wood would a woodchuck chuck?", "phonemes": ["h", "aʊ", "m", "ʌ", "tʃ", "w", "ʊ", "d", "w", "ʊ", "d", "ə", "w", "ʊ", "d", "tʃ", "ʌ", "k", "tʃ", "ʌ", "k"]},
        {"id": 4, "text": "Peter Piper picked a peck of pickled peppers.", "phonemes": ["p", "iː", "t", "ə", "p", "aɪ", "p", "ə", "p", "ɪ", "k", "t", "ə", "p", "e", "k", "ɒ", "v", "p", "ɪ", "k", "l", "d", "p", "e", "p", "ə", "z"]},
        {"id": 5, "text": "The rain in Spain stays mainly in the plain.", "phonemes": ["ð", "ə", "r", "eɪ", "n", "ɪ", "n", "s", "p", "eɪ", "n", "s", "t", "eɪ", "z", "m", "eɪ", "n", "l", "i", "ɪ", "n", "ð", "ə", "p", "l", "eɪ", "n"]},
        {"id": 6, "text": "A journey of a thousand miles begins with a single step.", "phonemes": ["ə", "dʒ", "ɜː", "n", "i", "ɒ", "v", "ə", "θ", "aʊ", "z", "ə", "n", "d", "m", "aɪ", "l", "z", "b", "ɪ", "g", "ɪ", "n", "z", "w", "ɪ", "ð", "ə", "s", "ɪ", "ŋ", "g", "l", "s", "t", "e", "p"]},
        {"id": 7, "text": "To be or not to be, that is the question.", "phonemes": ["t", "uː", "b", "iː", "ɔː", "n", "ɒ", "t", "t", "uː", "b", "iː", "ð", "æ", "t", "ɪ", "z", "ð", "ə", "k", "w", "e", "s", "tʃ", "ə", "n"]},
        {"id": 8, "text": "Jack and Jill went up the hill to fetch a pail of water.", "phonemes": ["dʒ", "æ", "k", "ə", "n", "d", "dʒ", "ɪ", "l", "w", "e", "n", "t", "ʌ", "p", "ð", "ə", "h", "ɪ", "l", "t", "uː", "f", "e", "tʃ", "ə", "p", "eɪ", "l", "ɒ", "v", "w", "ɔː", "t", "ə"]},
        {"id": 9, "text": "Unique New York, you know you need unique New York.", "phonemes": ["j", "uː", "n", "iː", "k", "n", "j", "uː", "j", "ɔː", "k", "j", "uː", "n", "əʊ", "j", "uː", "n", "iː", "d", "j", "uː", "n", "iː", "k", "n", "j", "uː", "j", "ɔː", "k"]},
        {"id": 10, "text": "Red leather, yellow leather, red leather, yellow leather.", "phonemes": ["r", "e", "d", "l", "e", "ð", "ə", "j", "e", "l", "əʊ", "l", "e", "ð", "ə", "r", "e", "d", "l", "e", "ð", "ə", "j", "e", "l", "əʊ", "l", "e", "ð", "ə"]},
    ],
    "is": [
        {"id": 1, "text": "Það var eitt sinn karl og kerling.", "phonemes": ["θ", "a", "ð", "v", "a", "r", "ei", "t", "s", "i", "n", "k", "a", "r", "l", "o", "g", "k", "e", "r", "l", "i", "ŋ", "g"]},
        {"id": 2, "text": "Sólskin og skýjum létt.", "phonemes": ["s", "ó", "l", "s", "c", "i", "n", "o", "g", "s", "c", "y", "j", "u", "m", "l", "j", "é", "t"]},
        {"id": 3, "text": "Þetta er fallegt land.", "phonemes": ["θ", "e", "t", "a", "e", "r", "f", "a", "l", "e", "g", "t", "l", "a", "n", "d"]},
    ]
}


@router.get("/phonemes/{language}")
async def get_phonemes(language: str):
    """Get phoneme set for a language."""
    if language not in PHONEMES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    return PHONEMES[language]


@router.get("/prompts/{language}")
async def get_prompts(language: str):
    """Get training prompts for a language."""
    if language not in PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    return {
        "language": language,
        "prompts": PROMPTS[language],
        "total": len(PROMPTS[language])
    }


@router.get("/prompts/languages")
async def get_available_languages():
    """Get available languages for prompts."""
    return {
        "languages": list(PROMPTS.keys())
    }


@router.post("/prompts/{language}/for-phonemes")
async def get_prompts_for_phonemes(language: str, phonemes: list[str]):
    """Get prompts that contain specific phonemes."""
    if language not in PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    # Filter prompts that contain any of the requested phonemes
    matching_prompts = []
    phonemes_set = set(phonemes)
    
    for prompt in PROMPTS[language]:
        prompt_phonemes = set(prompt.get("phonemes", []))
        if prompt_phonemes & phonemes_set:  # Intersection
            matching_prompts.append(prompt)
    
    return {
        "language": language,
        "requested_phonemes": phonemes,
        "prompts": matching_prompts,
        "total": len(matching_prompts)
    }


# ============================================================================
# Model Info Endpoints (for frontend training wizard)
# ============================================================================

import os
import glob
import json
from datetime import datetime

def get_audio_duration(filepath: str) -> float:
    """Get audio file duration in seconds. Returns 0 if unable to read."""
    try:
        import wave
        with wave.open(filepath, 'r') as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        # For non-wav files or errors, estimate 3 seconds
        return 3.0


def scan_model_recordings(exp_name: str) -> dict:
    """Scan for recordings for an experiment.
    
    Looks in multiple locations:
    1. /data/uploads/{exp_name}/ - where preprocessor stores uploaded files
    2. /data/{exp_name}/audio/ - alternative location
    3. /data/{exp_name}/uploads/ - legacy location
    """
    data_root = settings.paths.data_root
    data_dir = settings.paths.get_experiment_dir(exp_name)
    
    result = {
        "exp_name": exp_name,
        "total_recordings": 0,
        "total_duration_seconds": 0.0,
        "audio_paths": [],
        "categories": {},
        "sessions": []
    }
    
    # Primary location: /data/uploads/{exp_name}/ (where preprocessor stores uploads)
    uploads_dir = os.path.join(data_root, "uploads", exp_name)
    logger.info(f"Scanning for recordings in: {uploads_dir}")
    if os.path.exists(uploads_dir):
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            for filepath in glob.glob(os.path.join(uploads_dir, '**', ext), recursive=True):
                if filepath not in result["audio_paths"]:
                    result["audio_paths"].append(filepath)
                    result["total_duration_seconds"] += get_audio_duration(filepath)
    
    # Secondary location: /data/{exp_name}/audio/
    audio_dir = os.path.join(data_dir, "audio")
    if os.path.exists(audio_dir):
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            for filepath in glob.glob(os.path.join(audio_dir, '**', ext), recursive=True):
                if filepath not in result["audio_paths"]:
                    result["audio_paths"].append(filepath)
                    result["total_duration_seconds"] += get_audio_duration(filepath)
    
    # Legacy location: /data/{exp_name}/uploads/
    legacy_uploads_dir = os.path.join(data_dir, "uploads")
    if os.path.exists(legacy_uploads_dir):
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            for filepath in glob.glob(os.path.join(legacy_uploads_dir, '**', ext), recursive=True):
                if filepath not in result["audio_paths"]:
                    result["audio_paths"].append(filepath)
                    result["total_duration_seconds"] += get_audio_duration(filepath)
    
    result["total_recordings"] = len(result["audio_paths"])
    logger.info(f"Found {result['total_recordings']} recordings, {result['total_duration_seconds']:.2f}s total")
    
    return result


def scan_model_info(exp_name: str) -> dict:
    """Scan an experiment for training info."""
    data_dir = settings.paths.get_experiment_dir(exp_name)
    weights_dir = settings.paths.get_weights_dir(exp_name)
    model_dir = os.path.join(settings.paths.models_root, exp_name)
    
    # Get recordings info
    recordings = scan_model_recordings(exp_name)
    
    # Check preprocessed data
    preprocessed_dir = os.path.join(data_dir, "3_feature768")
    preprocessed_count = 0
    if os.path.exists(preprocessed_dir):
        preprocessed_count = len(glob.glob(os.path.join(preprocessed_dir, "*.npy")))
    
    # Check for training outputs
    has_model = os.path.exists(settings.paths.get_output_model_path(exp_name))
    has_index = os.path.exists(settings.paths.get_index_path(exp_name))
    
    # Check for checkpoints and epochs
    epochs_trained = 0
    latest_checkpoint = None
    if os.path.exists(weights_dir):
        checkpoints = glob.glob(os.path.join(weights_dir, "*.pth"))
        if checkpoints:
            # Parse epoch numbers from checkpoint names (e.g., G_100.pth, D_100.pth)
            for cp in checkpoints:
                basename = os.path.basename(cp)
                if basename.startswith("G_"):
                    try:
                        epoch = int(basename.replace("G_", "").replace(".pth", ""))
                        if epoch > epochs_trained:
                            epochs_trained = epoch
                            latest_checkpoint = cp
                    except ValueError:
                        pass
    
    # Check for config
    target_epochs = 100
    config_path = os.path.join(data_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                target_epochs = config.get("total_epochs", 100)
        except Exception:
            pass
    
    return {
        "name": exp_name,
        "model_dir": model_dir,
        "recordings": {
            "count": recordings["total_recordings"],
            "duration_seconds": recordings["total_duration_seconds"],
            "duration_minutes": round(recordings["total_duration_seconds"] / 60, 2)
        },
        "preprocessed": {
            "count": preprocessed_count,
            "has_data": preprocessed_count > 0
        },
        "training": {
            "has_model": has_model,
            "has_index": has_index,
            "epochs_trained": epochs_trained,
            "target_epochs": target_epochs,
            "last_trained": None,  # Could parse from file mtime
            "latest_checkpoint": latest_checkpoint
        },
        "coverage": {
            "phoneme_percent": None,
            "phonemes_covered": 0,
            "phonemes_missing": 0
        },
        "categories": {},
        "model": {
            "id": 0,
            "name": exp_name,
            "slug": exp_name,
            "status": "ready" if has_model else ("training" if epochs_trained > 0 else "new"),
            "en_readiness_score": None,
            "language_scanned_at": None
        }
    }


@router.get("/model/{exp_name}/recordings")
async def get_model_recordings(exp_name: str):
    """Get all recordings for a model."""
    try:
        result = scan_model_recordings(exp_name)
        return result
    except Exception as e:
        logger.error(f"Failed to get model recordings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/{exp_name}/category-status/{language}")
async def get_category_status(exp_name: str, language: str):
    """Get category recording status for a model."""
    if language not in PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    try:
        recordings = scan_model_recordings(exp_name)
        
        # Build category status (simplified - real implementation would analyze recordings)
        categories = {}
        for prompt in PROMPTS[language]:
            cat_id = str(prompt["id"])
            categories[cat_id] = {
                "name": prompt["text"][:50] + "..." if len(prompt["text"]) > 50 else prompt["text"],
                "total_prompts": 1,
                "recordings": 0,  # Would need to match recordings to prompts
                "has_recordings": False,
                "phonemes_covered": prompt.get("phonemes", [])
            }
        
        return {
            "exp_name": exp_name,
            "language": language,
            "categories": categories,
            "model": {
                "id": 0,
                "name": exp_name,
                "en_phoneme_coverage": None,
                "en_missing_phonemes": [],
                "is_phoneme_coverage": None,
                "is_missing_phonemes": [],
                "language_scanned_at": None
            }
        }
    except Exception as e:
        logger.error(f"Failed to get category status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/{exp_name}/info")
async def get_model_info(exp_name: str):
    """Get comprehensive model training info."""
    try:
        return scan_model_info(exp_name)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/{exp_name}/active")
async def get_active_training(exp_name: str):
    """Check if there's active training for this model."""
    active_job = job_manager.get_active_job(exp_name)
    if active_job:
        return {
            "active": True,
            "job": {
                "job_id": active_job.job_id,
                "status": active_job.status.value,
                "current_epoch": active_job.current_epoch,
                "total_epochs": active_job.total_epochs,
                "progress": active_job.progress
            }
        }
    return {"active": False}


class ModelTrainRequest(BaseModel):
    """Request to train a model using its collected recordings."""
    config: Optional[Dict[str, Any]] = Field(default=None, description="Training configuration")


@router.post("/model/{exp_name}/train")
async def train_model(exp_name: str, request: ModelTrainRequest, background_tasks: BackgroundTasks):
    """
    Start training for a model using all its collected recordings.
    
    This is a convenience endpoint that:
    1. Scans for existing recordings in the model directory
    2. Starts the training job with the specified configuration
    """
    # Check for existing active job
    active_job = job_manager.get_active_job(exp_name)
    if active_job:
        raise HTTPException(
            status_code=409,
            detail=f"Training already in progress for {exp_name} (job_id: {active_job.job_id})"
        )
    
    # Scan recordings
    recordings = scan_model_recordings(exp_name)
    
    if recordings["total_recordings"] == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No recordings found for {exp_name}. Upload audio files first."
        )
    
    # Build training config
    config = request.config or {}
    training_config = {
        "exp_name": exp_name,
        "epochs": config.get("epochs", 100),
        "batch_size": config.get("batch_size", 8),
        "save_every_epoch": config.get("save_every_epoch", 10),
        "sample_rate": config.get("sample_rate", 48000),
        "version": config.get("version", "v2"),
        "use_pitch_guidance": config.get("use_pitch_guidance", True),
        "gpus": config.get("gpus", "0"),
    }
    
    # Create job
    job = job_manager.create_job(exp_name, training_config)
    
    # Run training in background
    background_tasks.add_task(run_training_job, job)
    
    return {
        "success": True,
        "job_id": job.job_id,
        "exp_name": exp_name,
        "audio_files": recordings["total_recordings"],
        "total_duration": recordings["total_duration_seconds"],
        "config": {
            "epochs": training_config["epochs"],
            "batch_size": training_config["batch_size"],
            "sample_rate": training_config["sample_rate"]
        },
        "message": f"Training started for {exp_name} with {recordings['total_recordings']} audio files"
    }
