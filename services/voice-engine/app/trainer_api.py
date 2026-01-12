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
from app.analyzer import (
    ModelScanner,
    scan_model,
    analyze_model_gaps,
    LANGUAGE_PHONEMES,
)
from app.wizard import RecordingWizard, SessionStatus
from app.prompts import get_prompt_loader, get_available_languages

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/trainer", tags=["trainer"])

# Global instances (initialized on startup)
_training_pipeline: Optional[RVCTrainingPipeline] = None
_model_scanner: Optional[ModelScanner] = None
_recording_wizard: Optional[RecordingWizard] = None

# Config
LOGS_DIR = Path(__file__).parent.parent / "logs"
ASSETS_DIR = Path(__file__).parent.parent / "assets"
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"


def init_trainer_api(
    logs_dir: Optional[str] = None,
    assets_dir: Optional[str] = None,
    device: str = "cuda:0"
):
    """Initialize the trainer API components"""
    global _training_pipeline, _model_scanner, _recording_wizard
    
    logs = Path(logs_dir) if logs_dir else LOGS_DIR
    assets = Path(assets_dir) if assets_dir else ASSETS_DIR
    
    logs.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    _training_pipeline = create_training_pipeline(
        base_dir=str(logs),
        assets_dir=str(assets),
        device=device
    )
    
    _model_scanner = ModelScanner(logs_dir=str(logs), use_gpu=True)
    _recording_wizard = RecordingWizard(base_dir=str(logs / "wizard_sessions"))
    
    logger.info("Trainer API initialized")


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
    """
    exp_dir = UPLOAD_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
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
        # Check upload directory
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


@router.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    pipeline = get_pipeline()
    
    jobs = []
    for job_id, progress in pipeline._jobs.items():
        jobs.append(progress.to_dict())
    
    return {"jobs": jobs}


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


@router.post("/model/{exp_name}/train")
async def train_model(
    exp_name: str,
    background_tasks: BackgroundTasks,
    config: Optional[TrainingConfigInput] = None
):
    """
    Start training a model using all collected recordings from wizard sessions.
    """
    wizard = get_wizard()
    pipeline = get_pipeline()
    
    # Get all recordings for this model
    recordings_data = wizard.get_all_recordings_for_model(exp_name)
    audio_paths = recordings_data.get("audio_paths", [])
    
    if not audio_paths:
        raise HTTPException(
            status_code=400, 
            detail="No recordings found. Please record some audio first."
        )
    
    if len(audio_paths) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 10 recordings to train. Currently have {len(audio_paths)}."
        )
    
    # Build training config
    if config:
        training_config = TrainingConfig(
            exp_name=exp_name,
            sample_rate=SampleRate(config.sample_rate) if config.sample_rate else SampleRate.SR_48K,
            f0_method=F0Method(config.f0_method) if config.f0_method else F0Method.RMVPE,
            epochs=config.epochs or 100,
            batch_size=config.batch_size or 8,
            save_every_epoch=config.save_every_epoch or 25,
            version=RVCVersion(config.version) if config.version else RVCVersion.V2,
            use_pitch_guidance=config.use_pitch_guidance if config.use_pitch_guidance is not None else True
        )
    else:
        training_config = TrainingConfig(exp_name=exp_name)
    
    # Create job
    job_id = pipeline.create_job(training_config)
    
    # Start training in background
    async def run_training():
        try:
            await pipeline.train(training_config, audio_paths, job_id)
        except Exception as e:
            logger.exception(f"Training error: {e}")
    
    background_tasks.add_task(asyncio.create_task, run_training())
    
    return {
        "job_id": job_id,
        "status": "started",
        "exp_name": exp_name,
        "audio_files": len(audio_paths),
        "total_duration": recordings_data.get("total_duration_seconds", 0),
        "config": {
            "epochs": training_config.epochs,
            "batch_size": training_config.batch_size,
            "sample_rate": training_config.sample_rate.value
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
