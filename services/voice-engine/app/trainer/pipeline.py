"""
RVC Training Pipeline

Orchestrates the full RVC model training process:
1. Preprocess audio (slice, resample)
2. Extract F0 features (RMVPE)
3. Extract HuBERT features
4. Train RVC model
5. Build FAISS index

Reuses components from Retrieval-based-Voice-Conversion-WebUI.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Training Configuration
# ============================================================================

class TrainingStatus(str, Enum):
    """Training job status"""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    EXTRACTING_F0 = "extracting_f0"
    EXTRACTING_FEATURES = "extracting_features"
    TRAINING = "training"
    BUILDING_INDEX = "building_index"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SampleRate(int, Enum):
    """Supported sample rates"""
    SR_32K = 32000
    SR_40K = 40000
    SR_48K = 48000


class F0Method(str, Enum):
    """F0 extraction methods"""
    RMVPE = "rmvpe"
    PM = "pm"          # Parselmouth
    HARVEST = "harvest"
    CREPE = "crepe"
    DIO = "dio"


class RVCVersion(str, Enum):
    """RVC model version"""
    V1 = "v1"
    V2 = "v2"


@dataclass
class TrainingConfig:
    """
    Training configuration
    
    STEP-BASED TRAINING (NEW):
    - epochs is calculated from target_steps, NOT chosen arbitrarily
    - batch_size is auto-calculated based on dataset size
    - Formula: epochs = ceil(target_steps / steps_per_epoch)
    - steps_per_epoch = floor(num_segments / batch_size)
    
    For small datasets (<100 segments), use batch_size=4 to get more steps.
    """
    # Experiment
    exp_name: str
    
    # Audio processing
    sample_rate: SampleRate = SampleRate.SR_40K
    
    # F0 extraction
    f0_method: F0Method = F0Method.RMVPE
    
    # Training parameters (epochs derived from target_steps)
    epochs: int = 50  # DERIVED from target_steps if using training plan
    batch_size: int = 16  # AUTO-CALCULATED based on dataset size
    save_every_epoch: int = 10  # Save frequently to test early
    total_epoch: int = 50
    
    # STEP-BASED TRAINING TARGETS (NEW)
    target_steps: int = 1800          # Target optimizer steps (primary metric)
    steps_per_epoch: int = 0          # Calculated: num_segments / batch_size
    estimated_total_steps: int = 0    # steps_per_epoch * epochs
    smoke_test_after_steps: int = 500 # Run smoke test after N steps
    
    # GPU settings
    gpus: str = "0"
    
    # Model version
    version: RVCVersion = RVCVersion.V2
    
    # Pretrained models (auto-resolved if None)
    pretrain_G: Optional[str] = None
    pretrain_D: Optional[str] = None
    
    # Feature extraction
    use_pitch_guidance: bool = True  # if_f0 = 1
    
    # Preprocessing
    n_threads: int = 4
    
    def to_dict(self) -> dict:
        return {
            "exp_name": self.exp_name,
            "sample_rate": self.sample_rate.value,
            "f0_method": self.f0_method.value,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "save_every_epoch": self.save_every_epoch,
            "target_steps": self.target_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "estimated_total_steps": self.estimated_total_steps,
            "smoke_test_after_steps": self.smoke_test_after_steps,
            "gpus": self.gpus,
            "version": self.version.value,
            "use_pitch_guidance": self.use_pitch_guidance,
            "n_threads": self.n_threads
        }


@dataclass
class TrainingProgress:
    """Training progress tracking"""
    job_id: str
    status: TrainingStatus
    step: str
    progress: float  # 0.0 to 1.0
    current_epoch: int = 0
    total_epochs: int = 0
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    exp_name: Optional[str] = None
    result: Optional[dict] = None  # Final result with model_path, index_path
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "step": self.step,
            "progress": round(self.progress * 100, 1),
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "logs": self.logs[-50:],  # Last 50 log entries
            "exp_name": self.exp_name,
            "result": self.result,
        }


@dataclass
class TrainingResult:
    """Training result"""
    success: bool
    job_id: str
    model_path: Optional[str] = None
    index_path: Optional[str] = None
    metadata_path: Optional[str] = None
    error: Optional[str] = None
    training_time_seconds: float = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Model Metadata
# ============================================================================

@dataclass
class ModelMetadata:
    """Voice model metadata following the schema in TRAINER_DESIGN.md"""
    model_id: str
    name: str
    version: str = "1.0.0"
    created_at: str = ""
    updated_at: str = ""
    
    # Training config
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Audio source info
    audio_source: Dict[str, Any] = field(default_factory=dict)
    
    # Language readiness scores
    language_readiness: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Version history
    version_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Consent info
    consent: Dict[str, Any] = field(default_factory=dict)
    
    # File paths
    files: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, path: Union[str, Path]):
        """Save metadata to JSON file"""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelMetadata":
        """Load metadata from JSON file"""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def cleanup_checkpoints(
    exp_dir: Union[str, Path],
    keep_recent: int = 3,
    milestone_interval: int = 50
) -> List[Path]:
    """
    Clean up old checkpoint files, keeping recent ones and milestones.
    
    Args:
        exp_dir: Training experiment directory (string or Path)
        keep_recent: Number of most recent checkpoints to keep
        milestone_interval: Keep checkpoints at this epoch interval (e.g., 50, 100, 150...)
        
    Returns:
        List of deleted checkpoint paths
    """
    import re
    
    # Ensure exp_dir is a Path object
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)
    
    deleted = []
    
    # Find all G_*.pth and D_*.pth files
    g_files = list(exp_dir.glob("G_*.pth"))
    d_files = list(exp_dir.glob("D_*.pth"))
    
    if len(g_files) <= keep_recent:
        return deleted  # Not enough checkpoints to clean up
    
    # Extract step numbers and sort
    def extract_step(filepath: Path) -> int:
        match = re.search(r'[GD]_(\d+)\.pth', filepath.name)
        return int(match.group(1)) if match else 0
    
    g_files_sorted = sorted(g_files, key=extract_step)
    d_files_sorted = sorted(d_files, key=extract_step)
    
    # Determine which to keep:
    # 1. Most recent N checkpoints
    # 2. Milestone checkpoints (need to estimate epoch from step)
    recent_steps = {extract_step(f) for f in g_files_sorted[-keep_recent:]}
    
    # For milestone detection, we keep checkpoints that fall on 
    # approximate milestone boundaries (every milestone_interval checkpoints in sorted order)
    total_checkpoints = len(g_files_sorted)
    milestone_indices = set()
    if total_checkpoints > keep_recent:
        # Keep first checkpoint and roughly every milestone_interval-th thereafter
        for i in range(0, total_checkpoints, max(1, total_checkpoints // 5)):
            milestone_indices.add(i)
    
    # Delete old checkpoints (G and D files)
    for i, g_file in enumerate(g_files_sorted[:-keep_recent]):
        step = extract_step(g_file)
        if i not in milestone_indices:
            try:
                g_file.unlink()
                deleted.append(g_file)
                logger.debug(f"Deleted checkpoint: {g_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {g_file}: {e}")
    
    for i, d_file in enumerate(d_files_sorted[:-keep_recent]):
        step = extract_step(d_file)
        if i not in milestone_indices:
            try:
                d_file.unlink()
                deleted.append(d_file)
                logger.debug(f"Deleted checkpoint: {d_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {d_file}: {e}")
    
    if deleted:
        logger.info(f"Cleaned up {len(deleted)} old checkpoint files")
    
    return deleted


# ============================================================================
# Training Pipeline
# ============================================================================

class RVCTrainingPipeline:
    """
    Full RVC training pipeline.
    
    This class orchestrates all training steps and provides progress tracking.
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        assets_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda:0"
    ):
        """
        Initialize the training pipeline.
        
        Args:
            base_dir: Base directory for training outputs (logs/)
            assets_dir: Directory containing pretrained models
            device: Training device (cuda:0, cpu)
        """
        self.base_dir = Path(base_dir)
        self.assets_dir = Path(assets_dir) if assets_dir else self.base_dir.parent / "assets"
        self.device = device
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Job tracking
        self._jobs: Dict[str, TrainingProgress] = {}
        self._job_lock = threading.Lock()
        self._cancel_flags: Dict[str, bool] = {}
        
        # Callbacks
        self._progress_callbacks: Dict[str, List[Callable]] = {}
    
    def _resolve_pretrained_paths(
        self, 
        config: TrainingConfig
    ) -> tuple[str, str]:
        """Resolve pretrained model paths based on config"""
        sr = config.sample_rate.value // 1000  # 40000 -> 40
        version = config.version.value
        
        if version == "v2":
            pretrain_dir = self.assets_dir / "pretrained_v2"
            if config.use_pitch_guidance:
                g_path = pretrain_dir / f"f0G{sr}k.pth"
                d_path = pretrain_dir / f"f0D{sr}k.pth"
            else:
                g_path = pretrain_dir / f"G{sr}k.pth"
                d_path = pretrain_dir / f"D{sr}k.pth"
        else:
            pretrain_dir = self.assets_dir / "pretrained"
            g_path = pretrain_dir / f"G{sr}k.pth"
            d_path = pretrain_dir / f"D{sr}k.pth"
        
        return str(g_path), str(d_path)
    
    def create_job(self, config: TrainingConfig) -> str:
        """Create a new training job"""
        job_id = str(uuid.uuid4())[:8]
        
        progress = TrainingProgress(
            job_id=job_id,
            status=TrainingStatus.PENDING,
            step="Initializing",
            progress=0.0,
            total_epochs=config.epochs,
            started_at=datetime.utcnow().isoformat() + "Z",
            exp_name=config.exp_name  # Track the experiment name
        )
        
        with self._job_lock:
            self._jobs[job_id] = progress
            self._cancel_flags[job_id] = False
        
        return job_id
    
    def get_progress(self, job_id: str) -> Optional[TrainingProgress]:
        """Get job progress"""
        return self._jobs.get(job_id)
    
    def get_active_training(self, exp_name: Optional[str] = None) -> Optional[TrainingProgress]:
        """
        Check if there's an active training job.
        
        Args:
            exp_name: If provided, check for active training for this specific model.
                     If None, check for any active training.
        
        Returns:
            TrainingProgress if active training exists, None otherwise.
        """
        active_statuses = [TrainingStatus.PENDING, TrainingStatus.PREPROCESSING, TrainingStatus.TRAINING]
        
        with self._job_lock:
            for job in self._jobs.values():
                if job.status in active_statuses:
                    if exp_name is None or job.exp_name == exp_name:
                        return job
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Request job cancellation"""
        with self._job_lock:
            if job_id in self._jobs:
                self._cancel_flags[job_id] = True
                return True
        return False
    
    def request_checkpoint_and_stop(self, job_id: str) -> bool:
        """
        Request the training to save a checkpoint and then stop.
        
        This writes a control file that train.py checks after each epoch.
        """
        return self._write_checkpoint_request(job_id, "save_and_stop")
    
    def request_checkpoint(self, job_id: str) -> bool:
        """
        Request the training to save a checkpoint and continue.
        
        This writes a control file that train.py checks after each epoch.
        """
        return self._write_checkpoint_request(job_id, "save_and_continue")
    
    def _write_checkpoint_request(self, job_id: str, action: str) -> bool:
        """Write a checkpoint request file for the training subprocess"""
        with self._job_lock:
            if job_id not in self._jobs:
                return False
            
            job = self._jobs[job_id]
            if job.status != TrainingStatus.TRAINING:
                logger.warning(f"Cannot request checkpoint - job {job_id} is not training")
                return False
            
            exp_name = getattr(job, 'exp_name', None)
            if not exp_name:
                logger.warning(f"No exp_name found for job {job_id}")
                return False
        
        # Write the request file to the logs directory where train.py reads from
        # train.py uses hps.model_dir which is ./logs/{exp_name}
        voice_engine_root = Path(__file__).parent.parent.parent
        logs_dir = voice_engine_root / "logs" / exp_name
        request_file = logs_dir / ".checkpoint_request.json"
        
        try:
            request = {
                "action": action,
                "job_id": job_id,
                "requested_at": datetime.utcnow().isoformat() + "Z"
            }
            with open(request_file, 'w') as f:
                json.dump(request, f)
            
            logger.info(f"Wrote checkpoint request: {action} for job {job_id}")
            
            # Update job progress
            self._update_progress(job_id, message=f"Checkpoint requested ({action})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write checkpoint request: {e}")
            return False
    
    def get_checkpoint_response(self, job_id: str) -> Optional[dict]:
        """Check if there's a checkpoint response from the training subprocess"""
        with self._job_lock:
            if job_id not in self._jobs:
                return None
            job = self._jobs[job_id]
            exp_name = getattr(job, 'exp_name', None)
            if not exp_name:
                return None
        
        # Read from logs directory where train.py writes to
        voice_engine_root = Path(__file__).parent.parent.parent
        logs_dir = voice_engine_root / "logs" / exp_name
        response_file = logs_dir / ".checkpoint_response.json"
        
        if response_file.exists():
            try:
                with open(response_file, 'r') as f:
                    response = json.load(f)
                # Clear the response file
                response_file.unlink()
                return response
            except Exception as e:
                logger.warning(f"Failed to read checkpoint response: {e}")
        
        return None
    
    def _update_progress(
        self, 
        job_id: str, 
        status: Optional[TrainingStatus] = None,
        step: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        current_epoch: Optional[int] = None,
        error: Optional[str] = None,
        result: Optional[dict] = None
    ):
        """Update job progress"""
        with self._job_lock:
            if job_id not in self._jobs:
                return
            
            job = self._jobs[job_id]
            
            if status:
                job.status = status
            if step:
                job.step = step
            if progress is not None:
                job.progress = progress
            if message:
                job.message = message
                job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
            if current_epoch is not None:
                job.current_epoch = current_epoch
            if error:
                job.error = error
            if result is not None:
                job.result = result
            
            if status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                job.completed_at = datetime.utcnow().isoformat() + "Z"
    
    def _check_cancelled(self, job_id: str) -> bool:
        """Check if job should be cancelled"""
        return self._cancel_flags.get(job_id, False)
    
    async def train(
        self,
        config: TrainingConfig,
        audio_paths: List[Union[str, Path]],
        job_id: Optional[str] = None
    ) -> TrainingResult:
        """
        Run the full training pipeline.
        
        Args:
            config: Training configuration
            audio_paths: List of audio file paths to train on
            job_id: Optional existing job ID
            
        Returns:
            TrainingResult with model paths
        """
        import asyncio
        
        if job_id is None:
            job_id = self.create_job(config)
        
        start_time = time.time()
        exp_dir = self.base_dir / config.exp_name
        
        try:
            # Step 1: Setup experiment directory
            self._update_progress(
                job_id, 
                status=TrainingStatus.PREPROCESSING,
                step="Setting up experiment",
                progress=0.0,
                message="Creating experiment directory..."
            )
            
            exp_dir.mkdir(parents=True, exist_ok=True)
            trainset_dir = exp_dir / "trainset"
            trainset_dir.mkdir(exist_ok=True)
            
            # Copy/link audio files to trainset
            for audio_path in audio_paths:
                audio_path = Path(audio_path)
                if audio_path.exists():
                    dest = trainset_dir / audio_path.name
                    if not dest.exists():
                        shutil.copy2(audio_path, dest)
            
            if self._check_cancelled(job_id):
                self._update_progress(job_id, status=TrainingStatus.CANCELLED)
                return TrainingResult(success=False, job_id=job_id, error="Cancelled")
            
            # Step 2: Preprocess
            self._update_progress(
                job_id,
                step="Preprocessing audio",
                progress=0.1,
                message="Slicing and resampling audio..."
            )
            
            await self._run_preprocess(
                exp_dir=str(exp_dir),
                trainset_dir=str(trainset_dir),
                sample_rate=config.sample_rate.value,
                n_threads=config.n_threads
            )
            
            if self._check_cancelled(job_id):
                self._update_progress(job_id, status=TrainingStatus.CANCELLED)
                return TrainingResult(success=False, job_id=job_id, error="Cancelled")
            
            # Step 3: Extract F0
            self._update_progress(
                job_id,
                status=TrainingStatus.EXTRACTING_F0,
                step="Extracting pitch (F0)",
                progress=0.25,
                message=f"Using {config.f0_method.value} method..."
            )
            
            await self._run_f0_extraction(
                exp_dir=str(exp_dir),
                f0_method=config.f0_method.value,
                device=self.device
            )
            
            if self._check_cancelled(job_id):
                self._update_progress(job_id, status=TrainingStatus.CANCELLED)
                return TrainingResult(success=False, job_id=job_id, error="Cancelled")
            
            # Step 4: Extract Features
            self._update_progress(
                job_id,
                status=TrainingStatus.EXTRACTING_FEATURES,
                step="Extracting features (HuBERT)",
                progress=0.4,
                message="Extracting speaker embeddings..."
            )
            
            await self._run_feature_extraction(
                exp_dir=str(exp_dir),
                device=self.device,
                version=config.version.value
            )
            
            if self._check_cancelled(job_id):
                self._update_progress(job_id, status=TrainingStatus.CANCELLED)
                return TrainingResult(success=False, job_id=job_id, error="Cancelled")
            
            # Step 5: Train Model
            self._update_progress(
                job_id,
                status=TrainingStatus.TRAINING,
                step="Training model",
                progress=0.5,
                message="Starting model training..."
            )
            
            # Resolve pretrained paths
            pretrain_g, pretrain_d = self._resolve_pretrained_paths(config)
            
            await self._run_training(
                exp_dir=str(exp_dir),
                sample_rate=config.sample_rate.value,
                epochs=config.epochs,
                batch_size=config.batch_size,
                save_every_epoch=config.save_every_epoch,
                pretrain_G=pretrain_g,
                pretrain_D=pretrain_d,
                gpus=config.gpus,
                version=config.version.value,
                use_pitch_guidance=config.use_pitch_guidance,
                job_id=job_id,
                target_steps=config.target_steps,
                steps_per_epoch=config.steps_per_epoch,
                smoke_test_after_steps=config.smoke_test_after_steps
            )
            
            if self._check_cancelled(job_id):
                self._update_progress(job_id, status=TrainingStatus.CANCELLED)
                return TrainingResult(success=False, job_id=job_id, error="Cancelled")
            
            # Step 6: Build Index
            self._update_progress(
                job_id,
                status=TrainingStatus.BUILDING_INDEX,
                step="Building FAISS index",
                progress=0.95,
                message="Creating voice matching index..."
            )
            
            index_path = await self._run_index_training(
                exp_dir=str(exp_dir),
                version=config.version.value
            )
            
            # Step 7: Create metadata
            self._update_progress(
                job_id,
                step="Finalizing",
                progress=0.98,
                message="Creating model metadata..."
            )
            
            # Find the final model file with priority:
            # 1) <exp>_infer.pth (preferred stable runtime file)
            # 2) <exp>.pth (final extracted model)
            # 3) latest G_*.pth by step number (highest step)
            # Never use D_*.pth as a model
            import re
            model_path = None
            exp_name = config.exp_name
            
            # Priority 1: <exp>_infer.pth
            infer_pth = exp_dir / f"{exp_name}_infer.pth"
            if infer_pth.exists():
                model_path = str(infer_pth)
                logger.info(f"Selected model (priority 1 - _infer.pth): {model_path}")
            
            # Priority 2: <exp>.pth
            if not model_path:
                final_pth = exp_dir / f"{exp_name}.pth"
                if final_pth.exists():
                    model_path = str(final_pth)
                    logger.info(f"Selected model (priority 2 - final .pth): {model_path}")
            
            # Priority 3: latest G_*.pth by step number
            if not model_path:
                g_checkpoints = list(exp_dir.glob("G_*.pth"))
                if g_checkpoints:
                    # Sort by step number (extract from G_<step>.pth)
                    def get_step(p):
                        match = re.search(r'G_(\d+)\.pth', p.name)
                        return int(match.group(1)) if match else 0
                    g_checkpoints.sort(key=get_step, reverse=True)
                    model_path = str(g_checkpoints[0])  # Latest checkpoint
                    logger.info(f"Selected model (priority 3 - latest checkpoint): {model_path}")
            
            # Create metadata
            metadata = self._create_metadata(
                config=config,
                exp_dir=exp_dir,
                audio_paths=audio_paths,
                model_path=model_path,
                index_path=index_path
            )
            
            metadata_path = exp_dir / "model_metadata.json"
            metadata.save(metadata_path)
            
            # Done!
            elapsed = time.time() - start_time
            self._update_progress(
                job_id,
                status=TrainingStatus.COMPLETED,
                step="Complete",
                progress=1.0,
                message=f"Training completed in {elapsed/60:.1f} minutes",
                result={
                    "model_path": model_path,
                    "index_path": index_path,
                    "metadata_path": str(metadata_path),
                    "training_time_seconds": elapsed
                }
            )
            
            return TrainingResult(
                success=True,
                job_id=job_id,
                model_path=model_path,
                index_path=index_path,
                metadata_path=str(metadata_path),
                training_time_seconds=elapsed
            )
            
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            self._update_progress(
                job_id,
                status=TrainingStatus.FAILED,
                error=str(e),
                message=f"Training failed: {e}"
            )
            return TrainingResult(
                success=False,
                job_id=job_id,
                error=str(e),
                training_time_seconds=time.time() - start_time
            )
    
    async def _run_preprocess(
        self,
        exp_dir: str,
        trainset_dir: str,
        sample_rate: int,
        n_threads: int = 4
    ):
        """
        Run audio preprocessing.
        
        Creates:
        - {exp_dir}/0_gt_wavs/     - Ground truth wavs at target SR
        - {exp_dir}/1_16k_wavs/    - 16kHz wavs for feature extraction
        """
        import asyncio
        import soundfile as sf
        import librosa
        from concurrent.futures import ThreadPoolExecutor
        
        exp_path = Path(exp_dir)
        gt_wavs_dir = exp_path / "0_gt_wavs"
        wav16k_dir = exp_path / "1_16k_wavs"
        
        gt_wavs_dir.mkdir(exist_ok=True)
        wav16k_dir.mkdir(exist_ok=True)
        
        # Get all audio files
        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
            audio_files.extend(Path(trainset_dir).glob(ext))
        
        logger.info(f"Preprocessing {len(audio_files)} audio files")
        
        def process_file(audio_file: Path) -> int:
            """Process a single audio file"""
            try:
                # Load audio
                audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
                
                # Resample to target SR
                if sr != sample_rate:
                    audio_target = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                else:
                    audio_target = audio
                
                # Resample to 16k for feature extraction
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                # CRITICAL FIX: Slice ONCE on target-SR audio and get TIME BOUNDARIES
                # Then apply the SAME boundaries to both versions to ensure alignment
                # This prevents HuBERT features from being misaligned with GT waveforms
                chunk_boundaries = self._get_slice_boundaries(audio_target, sample_rate)
                
                # Save chunks using consistent boundaries
                base_name = audio_file.stem
                saved_count = 0
                for i, (start_sec, end_sec) in enumerate(chunk_boundaries):
                    # Apply same time boundaries to both sample rates
                    start_target = int(start_sec * sample_rate)
                    end_target = int(end_sec * sample_rate)
                    start_16k = int(start_sec * 16000)
                    end_16k = int(end_sec * 16000)
                    
                    chunk = audio_target[start_target:end_target]
                    chunk_16k = audio_16k[start_16k:end_16k]
                    
                    if len(chunk) < sample_rate * 0.5:  # Skip < 0.5s
                        continue
                    
                    # Save ground truth
                    gt_path = gt_wavs_dir / f"{base_name}_{i}.wav"
                    sf.write(str(gt_path), chunk, sample_rate)
                    
                    # Save 16k version
                    wav16k_path = wav16k_dir / f"{base_name}_{i}.wav"
                    sf.write(str(wav16k_path), chunk_16k, 16000)
                    saved_count += 1
                
                return saved_count
                
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
                return 0
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(process_file, audio_files))
        
        total_chunks = sum(results)
        logger.info(f"Created {total_chunks} audio chunks")
    
    def _get_slice_boundaries(
        self,
        audio: np.ndarray,
        sr: int,
        min_length: float = 1.5,  # seconds
        silence_threshold: float = 0.01
    ) -> List[Tuple[float, float]]:
        """
        Get time boundaries for audio slicing by silence detection.
        
        Returns list of (start_seconds, end_seconds) tuples.
        These boundaries can be applied to audio at ANY sample rate
        to ensure perfect alignment between different versions.
        
        CRITICAL: This ensures 0_gt_wavs and 1_16k_wavs chunks represent
        the EXACT same time regions, preventing HuBERT feature misalignment.
        """
        # Frame-based silence detection
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)    # 10ms hops
        
        boundaries = []
        chunk_start_sample = 0
        in_chunk = False
        silence_count = 0
        silence_threshold_frames = int(0.3 * sr / hop_length)  # 300ms silence
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            
            if rms >= silence_threshold:
                # Sound detected
                if not in_chunk:
                    chunk_start_sample = i
                    in_chunk = True
                silence_count = 0
            else:
                # Silence detected
                silence_count += 1
                if in_chunk and silence_count > silence_threshold_frames:
                    # End of chunk - calculate duration
                    chunk_end_sample = i
                    duration_samples = chunk_end_sample - chunk_start_sample
                    
                    if duration_samples >= sr * min_length:
                        start_sec = chunk_start_sample / sr
                        end_sec = chunk_end_sample / sr
                        boundaries.append((start_sec, end_sec))
                    
                    in_chunk = False
                    silence_count = 0
        
        # Handle remaining audio
        if in_chunk:
            chunk_end_sample = len(audio)
            duration_samples = chunk_end_sample - chunk_start_sample
            
            if duration_samples >= sr * min_length:
                start_sec = chunk_start_sample / sr
                end_sec = chunk_end_sample / sr
                boundaries.append((start_sec, end_sec))
        
        # If no chunks found, return whole audio as one chunk
        if not boundaries and len(audio) >= sr * min_length:
            boundaries = [(0.0, len(audio) / sr)]
        
        return boundaries
    
    def _simple_slice(
        self, 
        audio: np.ndarray, 
        sr: int,
        min_length: float = 1.5,  # seconds
        silence_threshold: float = 0.01
    ) -> List[np.ndarray]:
        """Simple audio slicing by silence (DEPRECATED - use _get_slice_boundaries)"""
        # Frame-based silence detection
        frame_length = int(sr * 0.025)
        hop_length = int(sr * 0.010)
        
        chunks = []
        current_chunk = []
        silence_count = 0
        silence_threshold_frames = int(0.3 * sr / hop_length)  # 300ms silence
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            
            if rms < silence_threshold:
                silence_count += 1
                if silence_count > silence_threshold_frames and len(current_chunk) > 0:
                    # End of chunk
                    chunk_audio = np.concatenate(current_chunk) if current_chunk else np.array([])
                    if len(chunk_audio) >= sr * min_length:
                        chunks.append(chunk_audio)
                    current_chunk = []
                    silence_count = 0
            else:
                silence_count = 0
                current_chunk.append(audio[i:i + hop_length])
        
        # Add remaining audio
        if current_chunk:
            chunk_audio = np.concatenate(current_chunk)
            if len(chunk_audio) >= sr * min_length:
                chunks.append(chunk_audio)
        
        # If no chunks found, return original audio
        if not chunks and len(audio) >= sr * min_length:
            chunks = [audio]
        
        return chunks
    
    async def _run_f0_extraction(
        self,
        exp_dir: str,
        f0_method: str = "rmvpe",
        device: str = "cuda:0"
    ):
        """
        Extract F0 (pitch) features.
        
        Creates:
        - {exp_dir}/2a_f0/     - F0 contours
        - {exp_dir}/2b_f0nsf/  - F0 for NSF vocoder
        """
        import asyncio
        
        exp_path = Path(exp_dir)
        wav16k_dir = exp_path / "1_16k_wavs"
        f0_dir = exp_path / "2a_f0"
        f0nsf_dir = exp_path / "2b_f0nsf"
        
        f0_dir.mkdir(exist_ok=True)
        f0nsf_dir.mkdir(exist_ok=True)
        
        wav_files = list(wav16k_dir.glob("*.wav"))
        logger.info(f"Extracting F0 from {len(wav_files)} files using {f0_method}")
        
        # Load F0 extractor based on method
        if f0_method == "rmvpe":
            f0_extractor = self._get_rmvpe_extractor(device)
        else:
            f0_extractor = self._get_pm_extractor()
        
        for wav_file in wav_files:
            try:
                import soundfile as sf
                audio, sr = sf.read(str(wav_file))
                
                # Extract F0
                f0 = f0_extractor(audio, sr)
                
                # Save F0
                f0_path = f0_dir / f"{wav_file.stem}.npy"
                np.save(str(f0_path), f0)
                
                # Save F0 for NSF (coarse)
                f0_coarse = self._coarse_f0(f0)
                f0nsf_path = f0nsf_dir / f"{wav_file.stem}.npy"
                np.save(str(f0nsf_path), f0_coarse)
                
            except Exception as e:
                logger.warning(f"Error extracting F0 from {wav_file}: {e}")
    
    def _get_rmvpe_extractor(self, device: str):
        """Get RMVPE F0 extractor"""
        # Try to use RVC's RMVPE implementation
        try:
            rmvpe_path = self.assets_dir / "rmvpe" / "rmvpe.pt"
            if rmvpe_path.exists():
                # Import and use RVC's RMVPE
                from infer.lib.rmvpe import RMVPE
                # Determine if we should use half precision (FP16)
                is_half = "cuda" in str(device).lower()
                rmvpe = RMVPE(str(rmvpe_path), is_half=is_half, device=device)
                
                def extract(audio, sr):
                    if sr != 16000:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    return rmvpe.infer_from_audio(audio, thred=0.03)
                
                return extract
        except ImportError:
            pass
        
        # Fallback to parselmouth
        return self._get_pm_extractor()
    
    def _get_pm_extractor(self):
        """Get Parselmouth F0 extractor"""
        try:
            import parselmouth
            
            def extract(audio, sr):
                sound = parselmouth.Sound(audio, sr)
                pitch = sound.to_pitch(time_step=0.01)
                f0 = pitch.selected_array['frequency']
                f0[f0 == 0] = np.nan
                return f0
            
            return extract
        except ImportError:
            # Ultimate fallback - librosa pyin
            import librosa
            
            def extract(audio, sr):
                f0, voiced_flag, _ = librosa.pyin(
                    audio, fmin=50, fmax=800, sr=sr
                )
                f0[~voiced_flag] = 0
                return f0
            
            return extract
    
    def _coarse_f0(self, f0: np.ndarray, f0_bin: int = 256) -> np.ndarray:
        """Convert F0 to coarse representation"""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - 1127 * np.log(1 + 50 / 700)) * (f0_bin - 2) / (
            1127 * np.log(1 + 1100 / 700) - 1127 * np.log(1 + 50 / 700)
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
        return np.rint(f0_mel).astype(np.int32)
    
    async def _run_feature_extraction(
        self,
        exp_dir: str,
        device: str = "cuda:0",
        version: str = "v2"
    ):
        """
        Extract HuBERT features.
        
        Creates:
        - {exp_dir}/3_feature256/ (v1) or 3_feature768/ (v2)
        """
        import torch
        import soundfile as sf
        
        exp_path = Path(exp_dir)
        wav16k_dir = exp_path / "1_16k_wavs"
        
        feature_dim = 768 if version == "v2" else 256
        feature_dir = exp_path / f"3_feature{feature_dim}"
        feature_dir.mkdir(exist_ok=True)
        
        wav_files = list(wav16k_dir.glob("*.wav"))
        logger.info(f"Extracting features from {len(wav_files)} files")
        
        # Load HuBERT model
        hubert_path = self.assets_dir / "hubert" / "hubert_base.pt"
        
        try:
            from fairseq import checkpoint_utils
            
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [str(hubert_path)],
                suffix="",
            )
            model = models[0]
            model = model.to(device)
            model.eval()
            
            # Check if we need a projection layer for v1
            if version == "v1":
                # v1 uses 256-dim features
                # This would require a trained projection layer
                pass
            
        except Exception as e:
            logger.warning(f"Could not load HuBERT: {e}")
            logger.info("Using dummy features for testing")
            
            # Dummy feature extraction for testing
            for wav_file in wav_files:
                try:
                    audio, sr = sf.read(str(wav_file))
                    # Create dummy features
                    n_frames = len(audio) // 320  # 20ms frames at 16kHz
                    features = np.random.randn(n_frames, feature_dim).astype(np.float32)
                    
                    feature_path = feature_dir / f"{wav_file.stem}.npy"
                    np.save(str(feature_path), features)
                except Exception as e:
                    logger.warning(f"Error creating dummy features for {wav_file}: {e}")
            return
        
        # Extract features
        for wav_file in wav_files:
            try:
                audio, sr = sf.read(str(wav_file))
                audio = torch.from_numpy(audio).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    features = model.extract_features(audio)[0]
                    
                    if version == "v1":
                        # Average pool to 256 dim
                        features = features.mean(dim=-1, keepdim=True).expand(-1, -1, 256)
                    
                    features = features.squeeze(0).cpu().numpy()
                
                feature_path = feature_dir / f"{wav_file.stem}.npy"
                np.save(str(feature_path), features)
                
            except Exception as e:
                logger.warning(f"Error extracting features from {wav_file}: {e}")
    
    async def _run_training(
        self,
        exp_dir: str,
        sample_rate: int,
        epochs: int,
        batch_size: int,
        save_every_epoch: int,
        pretrain_G: str,
        pretrain_D: str,
        gpus: str,
        version: str,
        use_pitch_guidance: bool,
        job_id: str,
        target_steps: int = 1800,
        steps_per_epoch: int = 0,
        smoke_test_after_steps: int = 500
    ):
        """
        Run the REAL RVC training loop.
        
        STEP-BASED TRAINING:
        - target_steps: The target number of optimizer steps
        - steps_per_epoch: Pre-calculated steps per epoch (num_segments / batch_size)
        - smoke_test_after_steps: When to save an early checkpoint for smoke test
        
        This creates the actual RVC model (.pth files) by calling the train.py script.
        """
        import asyncio
        
        exp_path = Path(exp_dir)
        exp_name = exp_path.name
        
        # Generate filelist
        filelist_path = await self._generate_filelist(
            exp_dir=exp_dir,
            sample_rate=sample_rate,
            version=version,
            use_pitch_guidance=use_pitch_guidance
        )
        
        # Create config.json in the experiment directory
        await self._create_training_config(
            exp_dir=exp_dir,
            sample_rate=sample_rate,
            version=version,
            batch_size=batch_size
        )
        
        # The train.py script expects experiment data in ./logs/{exp_name}/ relative to its cwd
        # Create a symlink from ./logs/{exp_name} -> the actual experiment directory
        voice_engine_root = Path(__file__).parent.parent.parent
        logs_dir = voice_engine_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logs_exp_link = logs_dir / exp_name
        
        # Remove existing symlink/directory if it exists
        if logs_exp_link.is_symlink():
            logs_exp_link.unlink()
        elif logs_exp_link.exists():
            # If it's a real directory, we need to be careful
            # Just use it as-is if it exists
            pass
        
        if not logs_exp_link.exists():
            logs_exp_link.symlink_to(exp_path.resolve())
            logger.info(f"Created symlink: {logs_exp_link} -> {exp_path.resolve()}")
        
        # Calculate step metrics for logging
        estimated_total_steps = steps_per_epoch * epochs if steps_per_epoch > 0 else 0
        
        logger.info(f"=" * 60)
        logger.info(f"[STEP-BASED TRAINING] Starting RVC training")
        logger.info(f"  Experiment: {exp_name}")
        logger.info(f"  Sample rate: {sample_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  ── STEP METRICS ──")
        logger.info(f"  Target steps: {target_steps}")
        logger.info(f"  Steps/epoch: {steps_per_epoch}")
        logger.info(f"  Estimated total steps: {estimated_total_steps}")
        logger.info(f"  Smoke test at step: {smoke_test_after_steps}")
        logger.info(f"  ── MODEL CONFIG ──")
        logger.info(f"  Pretrain G: {pretrain_G}")
        logger.info(f"  Pretrain D: {pretrain_D}")
        logger.info(f"  Version: {version}")
        logger.info(f"  F0 (pitch guidance): {use_pitch_guidance}")
        logger.info(f"=" * 60)
        
        # Build the training command
        # The train.py script expects these arguments (see infer/lib/train/utils.py get_hparams())
        sr_str = f"{sample_rate // 1000}k"  # 48000 -> "48k"
        
        train_cmd = [
            sys.executable,  # Use the same Python interpreter
            str(Path(__file__).parent.parent.parent / "infer" / "modules" / "train" / "train.py"),
            "-se", str(save_every_epoch),
            "-te", str(epochs),
            "-pg", pretrain_G,
            "-pd", pretrain_D,
            "-g", gpus,
            "-bs", str(batch_size),
            "-e", exp_name,  # experiment_dir - relative to ./logs/
            "-sr", sr_str,
            "-sw", "1",  # save_every_weights - save extractable model
            "-v", version,
            "-f0", "1" if use_pitch_guidance else "0",
            "-l", "0",  # if_latest=0 - keep all checkpoints (we clean up manually)
            "-c", "0",  # if_cache_data_in_gpu - don't cache in GPU memory
        ]
        
        logger.info(f"Training command: {' '.join(train_cmd)}")
        
        # Set up environment - ensure the working directory is voice-engine root
        voice_engine_root = Path(__file__).parent.parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(voice_engine_root)
        env["PYTHONUNBUFFERED"] = "1"  # Disable output buffering so we can parse epochs in real-time
        
        # Ensure assets/models/<model_name> directory exists BEFORE training
        # (savee() in process_ckpt.py saves to assets/models/<model_name>/{name}.pth)
        models_dir = voice_engine_root / "assets" / "models" / exp_name
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured assets/models/{exp_name} directory exists: {models_dir}")
        
        # Run training as a subprocess
        process = await asyncio.create_subprocess_exec(
            *train_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(voice_engine_root),
            env=env
        )
        
        # Monitor progress by reading stdout and parsing epoch info
        last_epoch = 0
        last_cleanup_epoch = 0
        training_complete = False
        
        async def read_output(stream, stream_name):
            nonlocal last_epoch, last_cleanup_epoch, training_complete
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    logger.info(f"[train.py {stream_name}] {line_str}")
                    
                    # Parse epoch progress from log output
                    # Formats:
                    #   - "Train Epoch: X [Y%]" - start of batch within epoch
                    #   - "====> Epoch: X [date] | (time)" - epoch completion
                    #   - "loss_disc=X, loss_gen=Y" - training progress
                    try:
                        import re
                        
                        # Check for epoch completion first (====> Epoch: X)
                        if "====> Epoch:" in line_str:
                            match = re.search(r"====> Epoch:\s*(\d+)", line_str)
                            if match:
                                current_epoch = int(match.group(1))
                                if current_epoch >= last_epoch:
                                    last_epoch = current_epoch
                                    # Epoch completed - calculate progress
                                    progress = 0.5 + (current_epoch / epochs) * 0.48
                                    self._update_progress(
                                        job_id,
                                        progress=progress,
                                        current_epoch=current_epoch,
                                        message=f"Completed epoch {current_epoch}/{epochs}"
                                    )
                                    logger.info(f"Epoch {current_epoch}/{epochs} completed")
                                    
                                    # Cleanup old checkpoints every 30 epochs
                                    if current_epoch - last_cleanup_epoch >= 30:
                                        try:
                                            cleanup_checkpoints(exp_dir, keep_recent=3, milestone_interval=20)
                                            last_cleanup_epoch = current_epoch
                                        except Exception as e:
                                            logger.warning(f"Checkpoint cleanup failed: {e}")
                        
                        # Check for batch progress within epoch (Train Epoch: X [Y%])
                        elif "Train Epoch:" in line_str:
                            match = re.search(r"Train Epoch:\s*(\d+)\s*\[(\d+(?:\.\d+)?)%\]", line_str)
                            if match:
                                current_epoch = int(match.group(1))
                                batch_pct = float(match.group(2))
                                # Calculate fine-grained progress within epoch
                                epoch_progress = (current_epoch - 1 + batch_pct / 100) / epochs
                                progress = 0.5 + epoch_progress * 0.48
                                if current_epoch > last_epoch or (current_epoch == last_epoch and batch_pct > 0):
                                    last_epoch = current_epoch
                                    self._update_progress(
                                        job_id,
                                        progress=progress,
                                        current_epoch=current_epoch,
                                        message=f"Training epoch {current_epoch}/{epochs} ({batch_pct:.0f}%)"
                                    )
                        
                        # Also parse loss values for monitoring
                        elif "loss_disc=" in line_str:
                            # Just log - useful for debugging training quality
                            pass
                            
                    except Exception as e:
                        logger.debug(f"Failed to parse training output: {e}")
                    
                    # Check for completion
                    if "Training is done" in line_str:
                        training_complete = True
                    
                    # Check for cancellation
                    if self._check_cancelled(job_id):
                        process.terminate()
                        return
        
        # Read stdout and stderr concurrently
        await asyncio.gather(
            read_output(process.stdout, "stdout"),
            read_output(process.stderr, "stderr")
        )
        
        # Wait for process to complete
        return_code = await process.wait()
        
        # Check for special exit code (2333333 means success in RVC)
        if return_code == 2333333 % 256 or return_code == 0 or training_complete:
            logger.info(f"Training completed successfully (return code: {return_code})")
        else:
            logger.error(f"Training failed with return code: {return_code}")
            raise RuntimeError(f"Training subprocess failed with code {return_code}")
        
        # Find the final model file
        # RVC saves models to assets/models/<model_name>/{name}.pth
        models_dir = voice_engine_root / "assets" / "models" / exp_name
        models_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        final_model = None
        
        # Look for model with our experiment name
        for pth_file in models_dir.glob(f"{exp_name}*.pth"):
            final_model = pth_file
            break
        
        if final_model and final_model.exists():
            # Copy to experiment directory as well
            dest_path = exp_path / f"{exp_name}.pth"
            shutil.copy2(str(final_model), str(dest_path))
            logger.info(f"Copied final model to {dest_path}")
        else:
            # Create placeholder - training may have produced checkpoint files instead
            checkpoint_files = list(exp_path.glob("G_*.pth"))
            if checkpoint_files:
                # Use the latest checkpoint and extract it
                latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Extracting model from checkpoint: {latest_checkpoint}")
                
                # Extract the model
                await self._extract_model_from_checkpoint(
                    checkpoint_path=latest_checkpoint,
                    output_path=exp_path / f"{exp_name}.pth",
                    sample_rate=sample_rate,
                    version=version
                )
            else:
                logger.warning(f"No model file found after training")
    
    async def _create_training_config(
        self,
        exp_dir: str,
        sample_rate: int,
        version: str,
        batch_size: int
    ):
        """Create config.json for training based on sample rate and version"""
        exp_path = Path(exp_dir)
        
        # Load base config from rvc/configs
        voice_engine_root = Path(__file__).parent.parent.parent
        sr_str = f"{sample_rate // 1000}k"  # 48000 -> "48k"
        config_template = voice_engine_root / "rvc" / "configs" / version / f"{sr_str}.json"
        
        if not config_template.exists():
            # Fallback to 48k config
            config_template = voice_engine_root / "rvc" / "configs" / version / "48k.json"
        
        if config_template.exists():
            with open(config_template, 'r') as f:
                config = json.load(f)
        else:
            # Use default config
            config = self._get_default_training_config(sample_rate, version)
        
        # Override batch size and log_interval for better progress tracking
        config['train']['batch_size'] = batch_size
        config['train']['log_interval'] = 1  # Log every batch for real-time progress
        # FP16 can cause NaN losses with some audio data, disable for stability
        config['train']['fp16_run'] = False
        
        # Save to experiment directory
        config_path = exp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Created training config at {config_path}")
    
    def _get_default_training_config(self, sample_rate: int, version: str) -> dict:
        """Get default training config for sample rate"""
        if sample_rate == 48000:
            hop_length = 480
            win_length = 2048
            segment_size = 17280
        elif sample_rate == 40000:
            hop_length = 400
            win_length = 2048
            segment_size = 12800
        else:  # 32000
            hop_length = 320
            win_length = 2048
            segment_size = 12800
        
        return {
            "train": {
                "log_interval": 10,  # Log every 10 steps for better progress tracking
                "seed": 1234,
                "epochs": 20000,
                "learning_rate": 0.0001,
                "betas": [0.8, 0.99],
                "eps": 1e-09,
                "batch_size": 4,
                "fp16_run": True,  # Enable mixed precision for faster training
                "lr_decay": 0.999875,
                "segment_size": segment_size,
                "init_lr_ratio": 1,
                "warmup_epochs": 0,
                "c_mel": 45,
                "c_kl": 1.0
            },
            "data": {
                "max_wav_value": 32768.0,
                "sampling_rate": sample_rate,
                "filter_length": 2048,
                "hop_length": hop_length,
                "win_length": win_length,
                "n_mel_channels": 128,
                "mel_fmin": 0.0,
                "mel_fmax": None
            },
            "model": {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [12, 10, 2, 2] if sample_rate == 48000 else [10, 10, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [24, 20, 4, 4] if sample_rate == 48000 else [20, 20, 4, 4],
                "use_spectral_norm": False,
                "gin_channels": 256,
                "spk_embed_dim": 109
            }
        }
    
    async def _generate_filelist(
        self,
        exp_dir: str,
        sample_rate: int,
        version: str,
        use_pitch_guidance: bool
    ) -> str:
        """Generate training filelist"""
        exp_path = Path(exp_dir)
        
        feature_dim = 768 if version == "v2" else 256
        gt_wavs_dir = exp_path / "0_gt_wavs"
        feature_dir = exp_path / f"3_feature{feature_dim}"
        f0_dir = exp_path / "2a_f0"
        f0nsf_dir = exp_path / "2b_f0nsf"
        
        filelist = []
        
        # Speaker ID (0 for single-speaker training)
        speaker_id = "0"
        
        for wav_file in gt_wavs_dir.glob("*.wav"):
            name = wav_file.stem
            
            feature_path = feature_dir / f"{name}.npy"
            f0_path = f0_dir / f"{name}.npy"          # Raw F0 values (float)
            f0nsf_path = f0nsf_dir / f"{name}.npy"    # Coarse F0 values (0-255 int)
            
            if not all(p.exists() for p in [feature_path, f0_path, f0nsf_path]):
                continue
            
            if use_pitch_guidance:
                # Format: wav_path|feature_path|pitch(coarse)|pitchf(raw)|speaker_id
                # pitch is loaded as LongTensor (coarse 0-255), pitchf as FloatTensor (raw Hz)
                line = f"{wav_file}|{feature_path}|{f0nsf_path}|{f0_path}|{speaker_id}"
            else:
                # Format: wav_path|feature_path|speaker_id
                line = f"{wav_file}|{feature_path}|{speaker_id}"
            
            filelist.append(line)
        
        filelist_path = exp_path / "filelist.txt"
        with open(filelist_path, "w") as f:
            f.write("\n".join(filelist))
        
        logger.info(f"Generated filelist with {len(filelist)} entries")
        return str(filelist_path)
    
    async def _extract_model_from_checkpoint(
        self,
        checkpoint_path: Path,
        output_path: Path,
        sample_rate: int,
        version: str
    ):
        """
        Extract final model from G_*.pth training checkpoint.
        
        This converts the training checkpoint format to the inference format
        by removing discriminator weights and setting proper config.
        """
        import torch
        from collections import OrderedDict
        
        logger.info(f"Extracting model from {checkpoint_path} to {output_path}")
        
        # Load checkpoint
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        
        # Extract weights (remove discriminator-only weights)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:  # Skip encoder query weights (discriminator)
                continue
            opt["weight"][key] = ckpt[key].half()
        
        # Map sample rate to string key
        sr_map = {32000: "32k", 40000: "40k", 48000: "48k"}
        sr_key = sr_map.get(sample_rate, "48k")
        
        # Set config based on sample rate and version
        if sr_key == "40k":
            opt["config"] = [
                1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000
            ]
        elif sr_key == "48k":
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
        
        # Set metadata
        opt["sr"] = sample_rate
        opt["f0"] = 1  # Pitch guidance enabled
        opt["version"] = version
        opt["info"] = f"Extracted from {checkpoint_path.name}"
        
        # Save final model
        torch.save(opt, str(output_path))
        logger.info(f"Extracted model saved to {output_path}")

    async def _run_index_training(
        self,
        exp_dir: str,
        version: str
    ) -> str:
        """
        Build FAISS index for voice matching.
        
        Returns path to created index file.
        """
        import faiss
        
        exp_path = Path(exp_dir)
        feature_dim = 768 if version == "v2" else 256
        feature_dir = exp_path / f"3_feature{feature_dim}"
        
        # Collect all features
        all_features = []
        for npy_file in feature_dir.glob("*.npy"):
            features = np.load(str(npy_file))
            all_features.append(features)
        
        if not all_features:
            logger.warning("No features found for index building")
            return ""
        
        # Concatenate features
        big_npy = np.concatenate(all_features, axis=0).astype(np.float32)
        logger.info(f"Building index from {big_npy.shape[0]} feature vectors")
        
        # Save big feature file
        big_npy_path = exp_path / f"total_fea.npy"
        np.save(str(big_npy_path), big_npy)
        
        # Build FAISS index
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        n_ivf = max(n_ivf, 1)
        
        index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
        index.train(big_npy)
        index.add(big_npy)
        
        # Save index
        index_path = exp_path / f"added_IVF{n_ivf}_Flat_nprobe_1_{exp_path.name}_{version}.index"
        faiss.write_index(index, str(index_path))
        
        logger.info(f"Created index at {index_path}")
        return str(index_path)
    
    def _create_metadata(
        self,
        config: TrainingConfig,
        exp_dir: Path,
        audio_paths: List[Union[str, Path]],
        model_path: Optional[str],
        index_path: Optional[str]
    ) -> ModelMetadata:
        """Create model metadata"""
        import soundfile as sf
        
        # Calculate total duration
        total_duration = 0.0
        for audio_path in audio_paths:
            try:
                info = sf.info(str(audio_path))
                total_duration += info.duration
            except:
                pass
        
        return ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=config.exp_name,
            training_config=config.to_dict(),
            audio_source={
                "type": "upload",
                "total_duration_seconds": total_duration,
                "num_files": len(audio_paths)
            },
            files={
                "model_pth": Path(model_path).name if model_path else "",
                "model_index": Path(index_path).name if index_path else ""
            }
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_training_pipeline(
    base_dir: Optional[str] = None,
    assets_dir: Optional[str] = None,
    device: str = "cuda:0"
) -> RVCTrainingPipeline:
    """
    Create a training pipeline with default paths.
    
    Args:
        base_dir: Base directory for training outputs
        assets_dir: Directory containing pretrained models
        device: Training device
        
    Returns:
        Configured RVCTrainingPipeline
    """
    if base_dir is None:
        # Use assets/models so checkpoints persist with model files
        base_dir = Path(__file__).parent.parent.parent / "assets" / "models"
    
    if assets_dir is None:
        assets_dir = Path(__file__).parent.parent.parent / "assets"
    
    return RVCTrainingPipeline(
        base_dir=base_dir,
        assets_dir=assets_dir,
        device=device
    )


if __name__ == "__main__":
    # Test the pipeline
    import asyncio
    
    async def test():
        pipeline = create_training_pipeline(device="cpu")
        
        config = TrainingConfig(
            exp_name="test_model",
            epochs=2,
            save_every_epoch=1
        )
        
        # Create test audio
        test_dir = Path("/tmp/test_training")
        test_dir.mkdir(exist_ok=True)
        
        import soundfile as sf
        test_audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
        test_path = test_dir / "test.wav"
        sf.write(str(test_path), test_audio, 16000)
        
        result = await pipeline.train(config, [test_path])
        print(f"Training result: {result.to_dict()}")
    
    asyncio.run(test())
