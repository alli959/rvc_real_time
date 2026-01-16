"""
Job Progress Contract Types

Defines the standard job progress contract for all long-running tasks.
All jobs must report progress using this step-based model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class JobStatus(str, Enum):
    """Job execution status"""
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    FINALIZING = "finalizing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    PAUSED = "paused"


class JobType(str, Enum):
    """Supported job types"""
    # Training variants
    TRAINING_RVC = "training_rvc"
    TRAINING_FINE_TUNE = "training_fine_tune"
    
    # Generation tasks
    GENERATE_SONG = "generate_song"
    TTS_GENERATE = "tts_generate"
    VOICE_CONVERT = "voice_convert"
    VOCAL_SPLIT = "vocal_split"
    VOICE_SWAP = "voice_swap"
    
    # Preprocessing
    PREPROCESS_AUDIO = "preprocess_audio"
    DOWNLOAD_MODEL = "download_model"
    
    # Generic
    GENERIC = "generic"


class CheckpointStatus(str, Enum):
    """Checkpoint save status for training jobs"""
    IDLE = "idle"
    REQUESTED = "requested"
    SAVING = "saving"
    SAVED = "saved"
    FAILED = "failed"


@dataclass
class JobStep:
    """Represents a single step in job execution"""
    key: str                      # Stable identifier e.g., "preprocessing"
    label: str                    # User-friendly e.g., "Preprocessing Audio"
    step_index: int               # 0-based index
    step_count: int               # Total number of steps
    step_progress_percent: float  # 0-100 within this step
    
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "step_index": self.step_index,
            "step_count": self.step_count,
            "step_progress_percent": round(self.step_progress_percent, 1),
        }


@dataclass
class TrainingJobDetails:
    """Details specific to training jobs"""
    exp_name: str
    phase: str  # 'dataset_prep', 'feature_extraction', 'training', 'evaluation', 'packaging'
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0         # Step within current epoch
    total_steps: int = 0          # Total steps in current epoch
    gpu_utilization: Optional[float] = None  # 0-100
    vram_usage_mb: Optional[float] = None
    checkpoint_status: CheckpointStatus = CheckpointStatus.IDLE
    last_checkpoint_at: Optional[str] = None
    last_checkpoint_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "exp_name": self.exp_name,
            "phase": self.phase,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "gpu_utilization": self.gpu_utilization,
            "vram_usage_mb": self.vram_usage_mb,
            "checkpoint_status": self.checkpoint_status.value,
            "last_checkpoint_at": self.last_checkpoint_at,
            "last_checkpoint_path": self.last_checkpoint_path,
        }


@dataclass
class GenerationJobDetails:
    """Details specific to generation jobs (song/TTS/convert)"""
    phase: str  # 'input_prep', 'model_load', 'inference', 'postprocess', 'export'
    input_file: Optional[str] = None
    model_name: Optional[str] = None
    output_artifacts: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "input_file": self.input_file,
            "model_name": self.model_name,
            "output_artifacts": self.output_artifacts,
        }


@dataclass
class DownloadJobDetails:
    """Details specific to download jobs"""
    file_name: str
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None
    download_speed_mbps: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "download_speed_mbps": self.download_speed_mbps,
        }


@dataclass
class JobProgress:
    """
    Standard Job Progress Contract
    
    All long-running tasks must report progress using this model.
    Progress is computed as weighted completion of steps.
    """
    job_id: str
    job_type: JobType
    status: JobStatus
    progress_percent: float  # 0-100
    
    current_step: JobStep
    
    # Type-specific details (one of these will be populated)
    details: Union[TrainingJobDetails, GenerationJobDetails, DownloadJobDetails, Dict[str, Any]]
    
    # Optional log stream pointer
    log_stream: Optional[str] = None
    
    # Timestamps
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Error info
    error: Optional[str] = None
    
    # Recent log messages
    recent_logs: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> dict:
        details_dict = self.details.to_dict() if hasattr(self.details, 'to_dict') else self.details
        
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "progress_percent": round(self.progress_percent, 1),
            "current_step": self.current_step.to_dict(),
            "details": details_dict,
            "log_stream": self.log_stream,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "recent_logs": self.recent_logs[-20:],  # Last 20 entries
        }


# =============================================================================
# Step Definitions for Each Job Type
# =============================================================================

TRAINING_STEPS = [
    {"key": "preprocessing", "label": "Preprocessing Audio", "weight": 0.10},
    {"key": "f0_extraction", "label": "Extracting Pitch (F0)", "weight": 0.15},
    {"key": "feature_extraction", "label": "Extracting Features (HuBERT)", "weight": 0.15},
    {"key": "training", "label": "Training Model", "weight": 0.50},
    {"key": "index_building", "label": "Building FAISS Index", "weight": 0.05},
    {"key": "packaging", "label": "Packaging Model", "weight": 0.05},
]

GENERATION_STEPS = [
    {"key": "input_prep", "label": "Preparing Input", "weight": 0.10},
    {"key": "model_load", "label": "Loading Model", "weight": 0.15},
    {"key": "inference", "label": "Running Inference", "weight": 0.50},
    {"key": "postprocess", "label": "Post-processing", "weight": 0.15},
    {"key": "export", "label": "Exporting Output", "weight": 0.10},
]

VOCAL_SPLIT_STEPS = [
    {"key": "loading", "label": "Loading Audio", "weight": 0.10},
    {"key": "separation", "label": "Separating Vocals", "weight": 0.70},
    {"key": "export", "label": "Exporting Tracks", "weight": 0.20},
]

VOICE_CONVERT_STEPS = [
    {"key": "loading", "label": "Loading Audio", "weight": 0.10},
    {"key": "f0_extraction", "label": "Extracting Pitch", "weight": 0.20},
    {"key": "inference", "label": "Converting Voice", "weight": 0.50},
    {"key": "postprocess", "label": "Post-processing", "weight": 0.10},
    {"key": "export", "label": "Exporting Audio", "weight": 0.10},
]

TTS_STEPS = [
    {"key": "text_prep", "label": "Preparing Text", "weight": 0.05},
    {"key": "synthesis", "label": "Synthesizing Speech", "weight": 0.60},
    {"key": "voice_convert", "label": "Applying Voice Model", "weight": 0.25},
    {"key": "export", "label": "Exporting Audio", "weight": 0.10},
]

DOWNLOAD_STEPS = [
    {"key": "connecting", "label": "Connecting", "weight": 0.05},
    {"key": "downloading", "label": "Downloading", "weight": 0.85},
    {"key": "verifying", "label": "Verifying", "weight": 0.10},
]


def get_steps_for_job_type(job_type: JobType) -> List[Dict[str, Any]]:
    """Get step definitions for a job type"""
    mapping = {
        JobType.TRAINING_RVC: TRAINING_STEPS,
        JobType.TRAINING_FINE_TUNE: TRAINING_STEPS,
        JobType.GENERATE_SONG: GENERATION_STEPS,
        JobType.TTS_GENERATE: TTS_STEPS,
        JobType.VOICE_CONVERT: VOICE_CONVERT_STEPS,
        JobType.VOCAL_SPLIT: VOCAL_SPLIT_STEPS,
        JobType.VOICE_SWAP: GENERATION_STEPS,
        JobType.PREPROCESS_AUDIO: TRAINING_STEPS[:3],  # First 3 steps
        JobType.DOWNLOAD_MODEL: DOWNLOAD_STEPS,
    }
    return mapping.get(job_type, GENERATION_STEPS)


def compute_progress(
    steps: List[Dict[str, Any]],
    current_step_index: int,
    step_progress: float  # 0-100 for current step
) -> float:
    """
    Compute overall progress based on step weights.
    
    This ensures progress never jumps and accurately reflects work done.
    """
    if current_step_index >= len(steps):
        return 100.0
    
    # Sum weights of completed steps
    completed_weight = sum(s["weight"] for s in steps[:current_step_index])
    
    # Add weighted progress of current step
    current_weight = steps[current_step_index]["weight"]
    current_contribution = current_weight * (step_progress / 100.0)
    
    total_progress = (completed_weight + current_contribution) * 100.0
    return min(100.0, max(0.0, total_progress))


class JobProgressManager:
    """
    Manages job progress tracking with the standard contract.
    
    Ensures:
    - No fake progress jumps
    - Accurate step-based tracking
    - Proper status transitions
    """
    
    def __init__(self, job_id: str, job_type: JobType):
        self.job_id = job_id
        self.job_type = job_type
        self.steps = get_steps_for_job_type(job_type)
        self.current_step_index = 0
        self.step_progress = 0.0
        self.status = JobStatus.QUEUED
        self.details: Any = {}
        self.recent_logs: List[str] = []
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow().isoformat() + "Z"
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
    
    def start(self):
        """Mark job as starting"""
        self.status = JobStatus.STARTING
        self.started_at = datetime.utcnow().isoformat() + "Z"
    
    def run(self):
        """Mark job as running"""
        self.status = JobStatus.RUNNING
    
    def advance_step(self, step_key: str):
        """
        Advance to a new step.
        
        Validates that step_key matches expected step to prevent jumping.
        """
        if self.current_step_index >= len(self.steps):
            return
        
        expected_key = self.steps[self.current_step_index]["key"]
        if step_key != expected_key:
            # Find the step index for this key
            for i, s in enumerate(self.steps):
                if s["key"] == step_key:
                    # Only allow forward progress
                    if i >= self.current_step_index:
                        # Mark previous step as complete
                        if i > self.current_step_index:
                            self.step_progress = 100.0
                        self.current_step_index = i
                        self.step_progress = 0.0
                    break
        else:
            # Advance to next step
            if self.current_step_index < len(self.steps) - 1:
                self.current_step_index += 1
                self.step_progress = 0.0
    
    def update_step_progress(self, progress: float):
        """Update progress within current step (0-100)"""
        # Never allow progress to go backwards
        self.step_progress = max(self.step_progress, min(100.0, progress))
    
    def set_step(self, step_key: str, progress: float = 0.0):
        """Set current step and progress"""
        for i, s in enumerate(self.steps):
            if s["key"] == step_key:
                if i >= self.current_step_index:
                    self.current_step_index = i
                    self.step_progress = max(0.0, min(100.0, progress))
                break
    
    def add_log(self, message: str):
        """Add a log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.recent_logs.append(f"[{timestamp}] {message}")
        # Keep only last 100 entries in memory
        if len(self.recent_logs) > 100:
            self.recent_logs = self.recent_logs[-100:]
    
    def complete(self):
        """Mark job as completed successfully"""
        self.status = JobStatus.SUCCEEDED
        self.current_step_index = len(self.steps)
        self.step_progress = 100.0
        self.completed_at = datetime.utcnow().isoformat() + "Z"
    
    def fail(self, error: str):
        """Mark job as failed"""
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow().isoformat() + "Z"
    
    def cancel(self):
        """Mark job as canceled"""
        self.status = JobStatus.CANCELED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
    
    def pause(self):
        """Mark job as paused"""
        self.status = JobStatus.PAUSED
    
    def resume(self):
        """Resume a paused job"""
        self.status = JobStatus.RUNNING
    
    def get_progress(self) -> JobProgress:
        """Get current progress as JobProgress object"""
        current_step_def = self.steps[min(self.current_step_index, len(self.steps) - 1)]
        
        current_step = JobStep(
            key=current_step_def["key"],
            label=current_step_def["label"],
            step_index=self.current_step_index,
            step_count=len(self.steps),
            step_progress_percent=self.step_progress,
        )
        
        progress_percent = compute_progress(
            self.steps,
            self.current_step_index,
            self.step_progress
        )
        
        return JobProgress(
            job_id=self.job_id,
            job_type=self.job_type,
            status=self.status,
            progress_percent=progress_percent,
            current_step=current_step,
            details=self.details,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            error=self.error,
            recent_logs=self.recent_logs,
        )
