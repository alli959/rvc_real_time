"""
Training Plan Recommender - Intelligent RVC Training Configuration

This module provides a comprehensive "detector + config rules + watchdogs" system
that ensures RVC training settings are chosen correctly based on:

1. MODE DETECTION: NEW model vs RESUME/FINE-TUNE from checkpoint
2. DATA ANALYSIS: Input data type/quality (speech vs singing, clean vs FX)
3. CONFIG RULES: Safe, deterministic parameter selection
4. WATCHDOGS: Fail-fast gates that abort bad training early

KEY PRINCIPLES:
- PREVENT "silent collapse" models (buzzy/static output, DC offset, low crest factor)
- FAIL FAST - abort early rather than letting bad jobs run for hours
- DETERMINISTIC and TESTABLE - no random or unpredictable behavior
- HUMAN-READABLE - produce clear "training plan report" explaining decisions

Usage:
    from app.trainer.training_plan import recommend_training_plan
    
    plan = recommend_training_plan(
        model_name="my_voice",
        audio_paths=["/path/to/audio1.wav", "/path/to/audio2.wav"],
        model_dir="/path/to/models/my_voice",
        gpu_memory_gb=12.0,
    )
    
    if plan.errors:
        print("Cannot proceed:", plan.errors)
    else:
        print(plan.report)
        # Use plan.suggested_config for training

Author: RVC Training System
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class TrainingPlanMode(str, Enum):
    """Training mode based on existing model state"""
    NEW_MODEL = "new_model"       # Training from scratch
    RESUME = "resume"             # Resuming interrupted training (same data)
    FINE_TUNE = "fine_tune"       # Adding new data to existing model


class DataType(str, Enum):
    """Detected data type classification"""
    SPEECH = "speech"             # Conversational speech, low pitch variance
    SINGING = "singing"           # Singing voice, high pitch range
    MIXED = "mixed"               # Mixed speech and singing
    UNKNOWN = "unknown"           # Could not classify


class DataQuality(str, Enum):
    """Detected data quality tier"""
    EXCELLENT = "excellent"       # Studio quality, high SNR, clean
    GOOD = "good"                 # Clean recordings, minimal issues
    FAIR = "fair"                 # Some noise/artifacts, usable
    POOR = "poor"                 # Significant issues, risky to train
    UNUSABLE = "unusable"         # Cannot train on this data


class WatchdogGate(str, Enum):
    """Available watchdog gates"""
    PREPROCESS = "preprocess"           # After slicing/resampling
    F0_EXTRACTION = "f0_extraction"     # After pitch extraction
    EARLY_TRAINING = "early_training"   # Loss patterns during training
    SMOKE_TEST = "smoke_test"           # First checkpoint inference test


# =============================================================================
# CONFIGURABLE THRESHOLDS
# =============================================================================

@dataclass
class PlanThresholds:
    """
    All configurable thresholds for training plan decisions.
    
    These are the initial justified values - adjust based on empirical testing.
    """
    
    # === Dataset Size Thresholds ===
    # Duration thresholds for sample rate selection
    min_duration_for_40k: float = 600.0      # 10 minutes - safe baseline
    min_duration_for_48k: float = 1200.0     # 20 minutes - high quality only
    min_usable_duration: float = 60.0        # 1 minute absolute minimum
    recommended_duration: float = 300.0       # 5 minutes for good results
    
    # Segment count thresholds (after preprocessing, ~6s chunks)
    min_segments_for_40k: int = 80           # ~8 minutes of data
    min_segments_for_48k: int = 160          # ~16 minutes of data
    min_segments_absolute: int = 20          # Minimum to even attempt training
    min_batches_per_epoch: int = 3           # Avoid unstable training signals
    
    # === Audio Quality Thresholds ===
    min_rms_level: float = 0.01              # Below = too quiet
    max_silence_percent: float = 30.0        # Above = too much silence
    max_clipping_percent: float = 1.0        # Above = distorted
    max_dc_offset: float = 0.05              # Above = DC bias issues
    
    # RMS percentile thresholds for consistency
    rms_p10_floor: float = 0.005             # 10th percentile shouldn't be this low
    rms_p90_ceiling: float = 0.9             # 90th percentile shouldn't be this high
    
    # === Data Type Classification (Speech vs Singing) ===
    # Pitch range in octaves (log2(f0_max/f0_min))
    speech_max_octave_range: float = 2.5     # Speech typically spans ~1.5-2.5 octaves
    singing_min_octave_range: float = 2.5    # Singing often spans 2.5+ octaves
    
    # F0 standard deviation thresholds (in Hz)
    speech_max_f0_std: float = 60.0          # Speech has moderate pitch variation
    singing_min_f0_std: float = 80.0         # Singing has wider pitch swings
    
    # Voiced frame percentage for classification
    min_voiced_for_speech: float = 30.0      # Speech should have decent voicing
    min_voiced_warning: float = 20.0         # Below this, warn about pitch extraction
    min_voiced_fail: float = 10.0            # Below this, pitch extraction failed
    
    # === FX/Contamination Detection ===
    # Spectral flatness (0=tonal, 1=noise-like)
    clean_max_spectral_flatness: float = 0.25    # Clean speech is tonal
    fx_min_spectral_flatness: float = 0.35       # FX/reverb increases flatness
    noisy_spectral_flatness: float = 0.50        # Definitely noisy/contaminated
    
    # === STEP-BASED TRAINING TARGETS ===
    # Key insight: optimizer steps matter, not epochs!
    # Small dataset + large batch = few steps/epoch = collapsed models
    # Formula: total_steps = (num_segments / batch_size) * epochs
    
    # Target step counts for different training scenarios
    target_steps_speech: int = 1800          # Speech models: 1500-2500 steps
    target_steps_singing: int = 3000         # Singing: 2500-5000 steps (wider pitch range)
    target_steps_fine_tune: int = 800        # Fine-tuning: 500-1000 steps
    
    min_total_steps: int = 1200              # Absolute minimum for convergence
    max_total_steps: int = 6000              # Diminishing returns beyond this
    
    min_steps_per_epoch: int = 10            # Below this, reduce batch size
    
    # === Epoch Limits (secondary to step targets) ===
    max_epochs_new_model: int = 300          # Raised - epochs calculated from steps
    max_epochs_tiny_data: int = 500          # Very small datasets need more epochs
    max_epochs_singing: int = 400            # Cap for singing (artifact risk)
    max_epochs_fx_noisy: int = 200           # Cap for FX/noisy data
    max_epochs_fine_tune: int = 100          # Cap for fine-tuning
    
    fine_tune_epoch_mult: float = 0.4        # Fine-tune uses 40% of new epochs
    fine_tune_lr_mult: float = 0.5           # Fine-tune uses 50% learning rate
    
    # === Batch Size Rules (based on dataset size) ===
    # Small datasets need smaller batch sizes to get more steps per epoch
    batch_size_rules: Dict[str, Any] = field(default_factory=lambda: {
        # num_segments -> recommended batch_size
        "tiny": {"max_segments": 100, "batch_size": 4},      # <100 segments
        "small": {"max_segments": 300, "batch_size": 6},     # 100-300 segments
        "medium": {"max_segments": 800, "batch_size": 8},    # 300-800 segments
        "large": {"max_segments": 2000, "batch_size": 12},   # 800-2000 segments
        "huge": {"max_segments": float('inf'), "batch_size": 16},  # 2000+ segments
    })
    
    max_batch_size_48k: int = 10             # Lower batch for 48k (more VRAM)
    min_batch_size: int = 4                  # Absolute minimum
    
    # === Watchdog Thresholds ===
    # Early training stuck detection
    stuck_loss_check_steps: int = 30         # Check after this many steps
    stuck_loss_tolerance: float = 0.01       # Loss variance below this = stuck
    max_stuck_iterations: int = 5            # Max iterations with stuck loss
    
    # F0 extraction quality
    f0_max_outlier_percent: float = 30.0     # Warning threshold
    f0_max_outlier_percent_fail: float = 50.0  # Hard fail threshold
    f0_valid_range: Tuple[float, float] = (40.0, 1100.0)  # Human pitch range
    
    # Smoke test
    smoke_test_min_crest: float = 2.0        # Below = collapsed output
    smoke_test_max_dc: float = 0.02          # Above = bias issue
    smoke_test_max_flatness: float = 0.35    # Above = noise-like
    
    # Checkpoint schedule
    early_checkpoint_epoch: int = 1          # Always save at epoch 1
    risky_save_interval: int = 5             # Save every N epochs if risky
    normal_save_interval: int = 10           # Normal checkpoint interval
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return asdict(self)


# Global default thresholds instance
DEFAULT_THRESHOLDS = PlanThresholds()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LockedParams:
    """
    Parameters locked when resuming/fine-tuning.
    
    These CANNOT be changed without starting a new model because the pretrained
    weights and existing training are bound to these values.
    """
    sample_rate: int                  # 32000, 40000, or 48000
    version: str                      # "v1" or "v2"
    use_pitch_guidance: bool          # if_f0 = True/False
    source_checkpoint: Optional[str] = None  # Path to checkpoint being resumed
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class DatasetAnalysis:
    """
    Results of pilot data analysis.
    
    Computed on a sample of preprocessed segments to classify data type
    and quality without processing everything.
    """
    # Size metrics
    total_duration_seconds: float
    num_source_files: int
    num_segments: int                 # After preprocessing
    estimated_batches_per_epoch: int
    
    # Audio quality metrics
    rms_median: float
    rms_p10: float                    # 10th percentile
    rms_p90: float                    # 90th percentile
    silence_ratio: float              # % segments below RMS threshold
    clipping_ratio: float             # % samples > 0.99
    dc_offset_median: float
    dc_offset_max: float
    
    # Pitch/type classification
    voiced_percent: float             # % frames with valid F0
    f0_min: float                     # Excluding zeros
    f0_max: float
    f0_mean: float
    f0_std: float
    pitch_range_octaves: float        # log2(f0_max/f0_min)
    data_type: DataType
    
    # FX/contamination
    spectral_flatness_median: float
    has_fx_contamination: bool
    
    # Overall quality
    quality_tier: DataQuality
    quality_score: float              # 0-100 composite score
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['data_type'] = self.data_type.value
        d['quality_tier'] = self.quality_tier.value
        return d


@dataclass
class SuggestedConfig:
    """
    Suggested training configuration with justifications.
    
    STEP-BASED TRAINING (NEW):
    - target_steps is the PRIMARY training target
    - epochs is DERIVED from: epochs = ceil(target_steps / steps_per_epoch)
    - steps_per_epoch = floor(num_segments / batch_size)
    
    This ensures small datasets get enough optimizer updates by:
    1. Reducing batch_size for small datasets
    2. Calculating epochs to reach target_steps
    """
    # Core params
    sample_rate: int
    f0_method: str
    epochs: int                       # DERIVED from target_steps
    batch_size: int                   # Auto-calculated based on dataset size
    save_every_epoch: int
    
    # STEP-BASED TRAINING (PRIMARY)
    target_steps: int                 # Target total optimizer steps
    steps_per_epoch: int              # Calculated: num_segments / batch_size
    estimated_total_steps: int        # steps_per_epoch * epochs
    
    # Learning rates
    learning_rate_g: float
    learning_rate_d: float
    
    # Model settings
    version: str                      # "v1" or "v2"
    use_pitch_guidance: bool
    fp16_run: bool
    
    # Pretrained paths (resolved)
    pretrain_G: str
    pretrain_D: str
    
    # Smoke test configuration
    smoke_test_after_steps: int = 500  # Run smoke test after this many steps
    smoke_test_abort_on_fail: bool = True
    
    # Justifications for each choice
    justifications: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatchdogConfig:
    """
    Configuration for enabled watchdog gates.
    """
    gate: WatchdogGate
    enabled: bool
    thresholds: Dict[str, Any]
    abort_on_fail: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['gate'] = self.gate.value
        return d


@dataclass
class TrainingPlan:
    """
    Complete training plan with mode, config, watchdogs, and report.
    
    This is the main output of recommend_training_plan().
    """
    # Mode detection result
    mode: TrainingPlanMode
    
    # Locked params if resume/fine-tune
    locked_params: Optional[LockedParams] = None
    
    # Data analysis
    dataset_analysis: Optional[DatasetAnalysis] = None
    
    # Suggested configuration
    suggested_config: Optional[SuggestedConfig] = None
    
    # Watchdog configuration
    required_watchdogs: List[WatchdogConfig] = field(default_factory=list)
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Human-readable report
    report: str = ""
    
    # Whether training can proceed
    can_proceed: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'locked_params': self.locked_params.to_dict() if self.locked_params else None,
            'dataset_analysis': self.dataset_analysis.to_dict() if self.dataset_analysis else None,
            'suggested_config': self.suggested_config.to_dict() if self.suggested_config else None,
            'required_watchdogs': [w.to_dict() for w in self.required_watchdogs],
            'warnings': self.warnings,
            'errors': self.errors,
            'report': self.report,
            'can_proceed': self.can_proceed,
        }


# =============================================================================
# MODE DETECTION
# =============================================================================

def detect_training_mode(
    model_dir: Path,
    force_mode: Optional[TrainingPlanMode] = None,
) -> Tuple[TrainingPlanMode, Optional[LockedParams], List[str]]:
    """
    Detect training mode based on existing model/checkpoint state.
    
    Priority:
    1. Check for resumable checkpoints (G_*.pth) -> RESUME mode
    2. Check for final model ({name}.pth) -> FINE_TUNE mode  
    3. No existing artifacts -> NEW_MODEL mode
    
    Args:
        model_dir: Path to model directory
        force_mode: If set, override detected mode (with validation)
        
    Returns:
        (mode, locked_params if applicable, warnings/errors)
    """
    warnings = []
    
    if not model_dir.exists():
        if force_mode and force_mode != TrainingPlanMode.NEW_MODEL:
            return TrainingPlanMode.NEW_MODEL, None, [
                f"Cannot {force_mode.value}: model directory doesn't exist"
            ]
        return TrainingPlanMode.NEW_MODEL, None, []
    
    # Check for checkpoints
    g_checkpoints = sorted(
        model_dir.glob("G_*.pth"),
        key=lambda x: _extract_step_number(x.name)
    )
    d_checkpoints = sorted(
        model_dir.glob("D_*.pth"),
        key=lambda x: _extract_step_number(x.name)
    )
    
    # Check for final model (exclude G_/D_ checkpoints)
    all_pth = list(model_dir.glob("*.pth"))
    final_models = [p for p in all_pth if not re.match(r'^[GD]_\d+\.pth$', p.name)]
    
    # Load existing config to get locked params
    locked_params = None
    config_path = model_dir / "config.json"
    metadata_path = model_dir / "metadata.json"
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Extract locked params from config
            # RVC config stores sampling_rate in data section
            sr = config.get('data', {}).get('sampling_rate', 40000)
            version = _detect_version_from_config(config)
            if_f0 = config.get('model', {}).get('if_f0', 1) == 1
            
            locked_params = LockedParams(
                sample_rate=sr,
                version=version,
                use_pitch_guidance=if_f0,
                source_checkpoint=str(g_checkpoints[-1]) if g_checkpoints else None
            )
        except Exception as e:
            warnings.append(f"Could not read config.json: {e}")
    
    # Determine mode
    has_checkpoints = bool(g_checkpoints and d_checkpoints)
    has_final_model = bool(final_models)
    
    if has_checkpoints:
        # Can resume training
        detected_mode = TrainingPlanMode.RESUME
        
        # Verify checkpoint pair exists
        if len(g_checkpoints) != len(d_checkpoints):
            warnings.append("Mismatched G/D checkpoint counts - will use latest matching pair")
        
    elif has_final_model:
        # Has completed model, can fine-tune
        detected_mode = TrainingPlanMode.FINE_TUNE
        
    else:
        detected_mode = TrainingPlanMode.NEW_MODEL
    
    # Handle force_mode
    if force_mode:
        if force_mode == TrainingPlanMode.NEW_MODEL:
            if has_checkpoints or has_final_model:
                warnings.append("Forcing NEW_MODEL mode - existing artifacts will be overwritten")
            return TrainingPlanMode.NEW_MODEL, None, warnings
            
        elif force_mode == TrainingPlanMode.RESUME:
            if not has_checkpoints:
                return detected_mode, locked_params, [
                    "Cannot RESUME: no checkpoints found (G_*.pth, D_*.pth)"
                ]
            return TrainingPlanMode.RESUME, locked_params, warnings
            
        elif force_mode == TrainingPlanMode.FINE_TUNE:
            if not has_final_model and not has_checkpoints:
                return detected_mode, locked_params, [
                    "Cannot FINE_TUNE: no existing model or checkpoint found"
                ]
            return TrainingPlanMode.FINE_TUNE, locked_params, warnings
    
    return detected_mode, locked_params, warnings


def _extract_step_number(filename: str) -> int:
    """Extract step number from checkpoint filename like G_12345.pth"""
    match = re.search(r'[GD]_(\d+)\.pth', filename)
    return int(match.group(1)) if match else 0


def _detect_version_from_config(config: dict) -> str:
    """Detect RVC version from config structure"""
    # v2 configs have different model structure
    if config.get('model', {}).get('spk_embed_dim'):
        return "v2"
    return "v1"


# =============================================================================
# DATA ANALYSIS
# =============================================================================

def analyze_dataset(
    audio_paths: Optional[List[str]] = None,
    preprocessed_dir: Optional[Path] = None,
    f0_dir: Optional[Path] = None,
    sample_size: int = 10,
    thresholds: PlanThresholds = DEFAULT_THRESHOLDS,
) -> DatasetAnalysis:
    """
    Analyze dataset to determine type and quality.
    
    This runs "pilot analysis" on a sample of segments to classify:
    - Data type (speech vs singing)
    - Quality tier (excellent to unusable)
    - FX/contamination level
    
    Args:
        audio_paths: Original audio files (before preprocessing)
        preprocessed_dir: Directory with preprocessed WAVs (0_gt_wavs/)
        f0_dir: Directory with F0 files (2a_f0/) for pitch analysis
        sample_size: Number of segments to sample for pilot analysis
        thresholds: Threshold configuration
        
    Returns:
        DatasetAnalysis with all metrics and classifications
    """
    import soundfile as sf
    
    # Gather source file info
    num_source_files = len(audio_paths) if audio_paths else 0
    total_duration = 0.0
    
    if audio_paths:
        for path in audio_paths:
            try:
                info = sf.info(path)
                total_duration += info.duration
            except:
                pass
    
    # Default values if we can't analyze
    default_analysis = DatasetAnalysis(
        total_duration_seconds=total_duration,
        num_source_files=num_source_files,
        num_segments=0,
        estimated_batches_per_epoch=0,
        rms_median=0.0,
        rms_p10=0.0,
        rms_p90=0.0,
        silence_ratio=1.0,
        clipping_ratio=0.0,
        dc_offset_median=0.0,
        dc_offset_max=0.0,
        voiced_percent=0.0,
        f0_min=0.0,
        f0_max=0.0,
        f0_mean=0.0,
        f0_std=0.0,
        pitch_range_octaves=0.0,
        data_type=DataType.UNKNOWN,
        spectral_flatness_median=0.5,
        has_fx_contamination=False,
        quality_tier=DataQuality.UNUSABLE,
        quality_score=0.0,
    )
    
    # Analyze preprocessed segments if available
    if preprocessed_dir and preprocessed_dir.exists():
        wav_files = list(preprocessed_dir.glob("*.wav"))
        num_segments = len(wav_files)
        
        if num_segments == 0:
            return default_analysis
        
        # Sample files for pilot analysis
        sample_files = _sample_files(wav_files, sample_size)
        
        # Collect metrics from samples
        rms_values = []
        dc_offsets = []
        clipping_counts = []
        total_samples = 0
        silence_count = 0
        spectral_flatness_values = []
        
        for wav_file in sample_files:
            try:
                audio, sr = sf.read(str(wav_file))
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
                
                # RMS
                rms = np.sqrt(np.mean(audio ** 2))
                rms_values.append(rms)
                
                # DC offset
                dc = np.mean(audio)
                dc_offsets.append(abs(dc))
                
                # Clipping
                clipped = np.sum(np.abs(audio) > 0.99)
                clipping_counts.append(clipped)
                total_samples += len(audio)
                
                # Silence check
                if rms < thresholds.min_rms_level:
                    silence_count += 1
                
                # Spectral flatness
                sf_val = _compute_spectral_flatness(audio)
                spectral_flatness_values.append(sf_val)
                
            except Exception as e:
                logger.debug(f"Could not analyze {wav_file}: {e}")
        
        if not rms_values:
            return default_analysis
        
        # Aggregate quality metrics
        rms_median = float(np.median(rms_values))
        rms_p10 = float(np.percentile(rms_values, 10))
        rms_p90 = float(np.percentile(rms_values, 90))
        silence_ratio = silence_count / len(sample_files)
        clipping_ratio = sum(clipping_counts) / total_samples if total_samples > 0 else 0
        dc_offset_median = float(np.median(dc_offsets))
        dc_offset_max = float(np.max(dc_offsets))
        spectral_flatness_median = float(np.median(spectral_flatness_values)) if spectral_flatness_values else 0.5
        
    else:
        # No preprocessed data - use defaults
        num_segments = int(total_duration / 6.0)  # Estimate ~6s segments
        rms_median = 0.1
        rms_p10 = 0.05
        rms_p90 = 0.3
        silence_ratio = 0.0
        clipping_ratio = 0.0
        dc_offset_median = 0.0
        dc_offset_max = 0.0
        spectral_flatness_median = 0.2
    
    # Analyze F0 for pitch classification
    voiced_percent = 0.0
    f0_min = 0.0
    f0_max = 0.0
    f0_mean = 0.0
    f0_std = 0.0
    pitch_range_octaves = 0.0
    
    if f0_dir and f0_dir.exists():
        f0_files = list(f0_dir.glob("*.npy"))
        
        if f0_files:
            sample_f0_files = _sample_files(f0_files, sample_size)
            all_voiced_f0 = []
            total_frames = 0
            voiced_frames = 0
            
            for f0_file in sample_f0_files:
                try:
                    f0 = np.load(str(f0_file))
                    total_frames += len(f0)
                    voiced_mask = f0 > 0
                    voiced_frames += np.sum(voiced_mask)
                    
                    if voiced_mask.any():
                        all_voiced_f0.extend(f0[voiced_mask].tolist())
                except:
                    pass
            
            if total_frames > 0:
                voiced_percent = 100.0 * voiced_frames / total_frames
            
            if all_voiced_f0:
                f0_arr = np.array(all_voiced_f0)
                f0_min = float(np.min(f0_arr))
                f0_max = float(np.max(f0_arr))
                f0_mean = float(np.mean(f0_arr))
                f0_std = float(np.std(f0_arr))
                
                if f0_min > 0:
                    pitch_range_octaves = np.log2(f0_max / f0_min)
    
    # Classify data type
    data_type = _classify_data_type(
        voiced_percent=voiced_percent,
        pitch_range_octaves=pitch_range_octaves,
        f0_std=f0_std,
        thresholds=thresholds,
    )
    
    # Check for FX contamination
    has_fx_contamination = spectral_flatness_median > thresholds.fx_min_spectral_flatness
    
    # Calculate quality tier and score
    quality_tier, quality_score = _calculate_quality_tier(
        rms_median=rms_median,
        rms_p10=rms_p10,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        dc_offset_max=dc_offset_max,
        spectral_flatness=spectral_flatness_median,
        voiced_percent=voiced_percent,
        total_duration=total_duration,
        thresholds=thresholds,
    )
    
    # Estimate batches per epoch
    batch_size = 16  # Default
    estimated_batches = max(1, num_segments // batch_size)
    
    return DatasetAnalysis(
        total_duration_seconds=total_duration,
        num_source_files=num_source_files,
        num_segments=num_segments,
        estimated_batches_per_epoch=estimated_batches,
        rms_median=rms_median,
        rms_p10=rms_p10,
        rms_p90=rms_p90,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        dc_offset_median=dc_offset_median,
        dc_offset_max=dc_offset_max,
        voiced_percent=voiced_percent,
        f0_min=f0_min,
        f0_max=f0_max,
        f0_mean=f0_mean,
        f0_std=f0_std,
        pitch_range_octaves=pitch_range_octaves,
        data_type=data_type,
        spectral_flatness_median=spectral_flatness_median,
        has_fx_contamination=has_fx_contamination,
        quality_tier=quality_tier,
        quality_score=quality_score,
    )


def _sample_files(files: List[Path], n: int) -> List[Path]:
    """Sample n files evenly from list (deterministic)"""
    if len(files) <= n:
        return files
    
    # Deterministic sampling - take evenly spaced files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def _compute_spectral_flatness(audio: np.ndarray) -> float:
    """Compute spectral flatness (Wiener entropy)"""
    fft = np.abs(np.fft.rfft(audio))
    fft = fft[fft > 1e-10]  # Remove zeros
    
    if len(fft) == 0:
        return 1.0  # Noise-like
    
    geometric_mean = np.exp(np.mean(np.log(fft)))
    arithmetic_mean = np.mean(fft)
    
    if arithmetic_mean < 1e-10:
        return 1.0
    
    return geometric_mean / arithmetic_mean


def _classify_data_type(
    voiced_percent: float,
    pitch_range_octaves: float,
    f0_std: float,
    thresholds: PlanThresholds,
) -> DataType:
    """Classify data as speech, singing, or mixed"""
    
    if voiced_percent < thresholds.min_voiced_fail:
        return DataType.UNKNOWN  # Pitch extraction failed
    
    # Check pitch range
    is_high_range = pitch_range_octaves > thresholds.speech_max_octave_range
    is_high_std = f0_std > thresholds.singing_min_f0_std
    
    if is_high_range and is_high_std:
        return DataType.SINGING
    elif is_high_range or is_high_std:
        return DataType.MIXED
    else:
        return DataType.SPEECH


def _calculate_quality_tier(
    rms_median: float,
    rms_p10: float,
    silence_ratio: float,
    clipping_ratio: float,
    dc_offset_max: float,
    spectral_flatness: float,
    voiced_percent: float,
    total_duration: float,
    thresholds: PlanThresholds,
) -> Tuple[DataQuality, float]:
    """Calculate quality tier and composite score (0-100)"""
    
    score = 100.0
    issues = []
    
    # Deduct for low RMS
    if rms_median < thresholds.min_rms_level:
        score -= 30
        issues.append("very_quiet")
    elif rms_median < thresholds.min_rms_level * 2:
        score -= 15
        issues.append("quiet")
    
    # Deduct for silence
    if silence_ratio > 0.5:
        score -= 40
        issues.append("mostly_silent")
    elif silence_ratio > thresholds.max_silence_percent / 100:
        score -= 20
        issues.append("high_silence")
    
    # Deduct for clipping
    if clipping_ratio > thresholds.max_clipping_percent / 100:
        score -= 25
        issues.append("clipped")
    elif clipping_ratio > thresholds.max_clipping_percent / 200:
        score -= 10
    
    # Deduct for DC offset
    if dc_offset_max > thresholds.max_dc_offset:
        score -= 15
        issues.append("dc_offset")
    
    # Deduct for FX/noise
    if spectral_flatness > thresholds.noisy_spectral_flatness:
        score -= 25
        issues.append("noisy")
    elif spectral_flatness > thresholds.fx_min_spectral_flatness:
        score -= 15
        issues.append("fx_contamination")
    
    # Deduct for low voiced frames
    if voiced_percent < thresholds.min_voiced_fail:
        score -= 30
        issues.append("pitch_fail")
    elif voiced_percent < thresholds.min_voiced_warning:
        score -= 15
        issues.append("low_voicing")
    
    # Deduct for short duration
    if total_duration < thresholds.min_usable_duration:
        score -= 20
        issues.append("too_short")
    elif total_duration < thresholds.recommended_duration:
        score -= 10
    
    score = max(0, score)
    
    # Map to tier
    if score >= 80:
        tier = DataQuality.EXCELLENT
    elif score >= 60:
        tier = DataQuality.GOOD
    elif score >= 40:
        tier = DataQuality.FAIR
    elif score >= 20:
        tier = DataQuality.POOR
    else:
        tier = DataQuality.UNUSABLE
    
    return tier, score


# =============================================================================
# CONFIG RULES ENGINE
# =============================================================================

def calculate_suggested_config(
    mode: TrainingPlanMode,
    locked_params: Optional[LockedParams],
    analysis: DatasetAnalysis,
    gpu_memory_gb: float,
    assets_dir: Path,
    thresholds: PlanThresholds = DEFAULT_THRESHOLDS,
    user_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[SuggestedConfig, List[str], List[str]]:
    """
    Calculate suggested training configuration based on mode and analysis.
    
    Implements the CONFIG RULES from the spec:
    - Sample rate policy (32k for risky, 40k baseline, 48k for excellent)
    - Epochs policy (caps based on data type/quality)
    - Batch size policy (GPU-aware with safety limits)
    - FP16 policy (off by default for safety)
    - Pretrained selection (strict validation)
    
    Returns:
        (config, warnings, errors)
    """
    warnings = []
    errors = []
    justifications = {}
    
    # Start with defaults
    sample_rate = 40000
    version = "v2"
    use_pitch_guidance = True
    f0_method = "rmvpe"
    epochs = 50
    batch_size = 16
    save_every = 10
    lr_g = 1e-4
    lr_d = 1e-4
    fp16_run = False
    target_steps = 1800
    steps_per_epoch = 0
    estimated_total_steps = 0
    
    # Handle locked params (RESUME/FINE_TUNE)
    if locked_params:
        sample_rate = locked_params.sample_rate
        version = locked_params.version
        use_pitch_guidance = locked_params.use_pitch_guidance
        
        justifications['sample_rate'] = f"LOCKED from existing model/checkpoint ({sample_rate}Hz)"
        justifications['version'] = f"LOCKED from existing model ({version})"
        justifications['use_pitch_guidance'] = f"LOCKED from existing model ({use_pitch_guidance})"
        
        # Validate user overrides don't conflict
        if user_overrides:
            if 'sample_rate' in user_overrides and user_overrides['sample_rate'] != sample_rate:
                errors.append(
                    f"Cannot change sample_rate when {mode.value}: "
                    f"locked to {sample_rate}Hz from existing checkpoint"
                )
            if 'version' in user_overrides and user_overrides['version'] != version:
                errors.append(
                    f"Cannot change version when {mode.value}: "
                    f"locked to {version} from existing checkpoint"
                )
            if 'use_pitch_guidance' in user_overrides and user_overrides['use_pitch_guidance'] != use_pitch_guidance:
                errors.append(
                    f"Cannot change use_pitch_guidance when {mode.value}: "
                    f"locked to {use_pitch_guidance} from existing checkpoint"
                )
    
    else:
        # NEW_MODEL - apply sample rate policy
        sample_rate, sr_justification = _select_sample_rate(analysis, thresholds)
        justifications['sample_rate'] = sr_justification
    
    # STEP 1: Apply batch size policy FIRST (needed for epoch calculation)
    batch_size, bs_justification = _select_batch_size(
        sample_rate=sample_rate,
        gpu_memory_gb=gpu_memory_gb,
        analysis=analysis,
        thresholds=thresholds,
    )
    justifications['batch_size'] = bs_justification
    
    # STEP 2: Apply STEP-BASED epochs policy (uses batch_size)
    epochs, target_steps, steps_per_epoch, estimated_total_steps, epochs_justification = _select_epochs(
        mode=mode,
        analysis=analysis,
        thresholds=thresholds,
        batch_size=batch_size,
    )
    justifications['epochs'] = epochs_justification
    justifications['training_steps'] = (
        f"target={target_steps} steps, actual={estimated_total_steps} steps "
        f"({steps_per_epoch} steps/epoch × {epochs} epochs)"
    )
    
    # Apply learning rate policy
    if mode == TrainingPlanMode.FINE_TUNE:
        lr_g = 1e-4 * thresholds.fine_tune_lr_mult
        lr_d = 1e-4 * thresholds.fine_tune_lr_mult
        justifications['learning_rate'] = f"Reduced for fine-tuning ({lr_g:.0e})"
    else:
        justifications['learning_rate'] = f"Default learning rate ({lr_g:.0e})"
    
    # Apply FP16 policy (default OFF for safety)
    is_risky = (
        analysis.quality_tier in [DataQuality.POOR, DataQuality.FAIR] or
        analysis.data_type == DataType.SINGING or
        analysis.has_fx_contamination or
        analysis.total_duration_seconds < thresholds.min_duration_for_40k
    )
    
    if is_risky:
        fp16_run = False
        justifications['fp16'] = "Disabled (safe mode: risky data detected)"
    else:
        fp16_run = False  # Still default off for now
        justifications['fp16'] = "Disabled (default for stability)"
    
    # Apply save schedule policy
    if is_risky:
        save_every = thresholds.risky_save_interval
        justifications['save_every'] = f"Frequent saves ({save_every} epochs) for risky data"
    else:
        save_every = thresholds.normal_save_interval
        justifications['save_every'] = f"Normal checkpoint interval ({save_every} epochs)"
    
    # Resolve pretrained paths
    pretrain_g, pretrain_d, pretrain_warnings = _resolve_pretrained(
        sample_rate=sample_rate,
        version=version,
        use_pitch_guidance=use_pitch_guidance,
        assets_dir=assets_dir,
    )
    warnings.extend(pretrain_warnings)
    justifications['pretrained'] = f"Matched to SR={sample_rate}, version={version}, f0={use_pitch_guidance}"
    
    if not pretrain_g or not pretrain_d:
        errors.append(f"Pretrained models not found for SR={sample_rate}, version={version}")
    
    # Apply user overrides (only for non-locked params)
    if user_overrides and not errors:
        for key, value in user_overrides.items():
            if key == 'epochs' and isinstance(value, int):
                epochs = value
                # Recalculate estimated_total_steps with override
                estimated_total_steps = steps_per_epoch * epochs
                justifications['epochs'] = f"User override: {epochs} epochs ({estimated_total_steps} total steps)"
            elif key == 'batch_size' and isinstance(value, int):
                batch_size = value
                # Recalculate step metrics with new batch size
                steps_per_epoch = max(1, analysis.num_segments // batch_size)
                estimated_total_steps = steps_per_epoch * epochs
                justifications['batch_size'] = f"User override: batch_size={batch_size} ({steps_per_epoch} steps/epoch)"
            elif key == 'target_steps' and isinstance(value, int):
                target_steps = value
                # Recalculate epochs from target_steps
                epochs = max(1, (target_steps + steps_per_epoch - 1) // steps_per_epoch)
                epochs = min(epochs, thresholds.max_epochs_new_model)
                estimated_total_steps = steps_per_epoch * epochs
                justifications['epochs'] = f"User override target_steps={target_steps}: {epochs} epochs needed"
            elif key == 'f0_method' and isinstance(value, str):
                f0_method = value
                justifications['f0_method'] = f"User override: {f0_method}"
    
    # Determine smoke test timing
    smoke_test_after_steps = min(500, estimated_total_steps // 3)
    
    config = SuggestedConfig(
        sample_rate=sample_rate,
        f0_method=f0_method,
        epochs=epochs,
        batch_size=batch_size,
        save_every_epoch=save_every,
        target_steps=target_steps,
        steps_per_epoch=steps_per_epoch,
        estimated_total_steps=estimated_total_steps,
        learning_rate_g=lr_g,
        learning_rate_d=lr_d,
        version=version,
        use_pitch_guidance=use_pitch_guidance,
        fp16_run=fp16_run,
        pretrain_G=pretrain_g or "",
        pretrain_D=pretrain_d or "",
        smoke_test_after_steps=smoke_test_after_steps,
        smoke_test_abort_on_fail=True,
        justifications=justifications,
    )
    
    return config, warnings, errors


def _select_sample_rate(
    analysis: DatasetAnalysis,
    thresholds: PlanThresholds,
) -> Tuple[int, str]:
    """
    Select sample rate based on data analysis.
    
    Policy:
    - Force 32k if: tiny data, singing, FX/noisy, poor quality
    - Use 40k if: adequate data, clean speech
    - Allow 48k only if: large data, excellent quality, clean speech
    """
    duration = analysis.total_duration_seconds
    segments = analysis.num_segments
    quality = analysis.quality_tier
    data_type = analysis.data_type
    has_fx = analysis.has_fx_contamination
    
    reasons = []
    
    # Check for forced 32k conditions
    force_32k = False
    
    if duration < thresholds.min_duration_for_40k / 2:  # <5 min
        force_32k = True
        reasons.append(f"duration only {duration/60:.1f}min (<5min)")
    
    if segments < thresholds.min_segments_for_40k / 2:  # <40 segments
        force_32k = True
        reasons.append(f"only {segments} segments (<40)")
    
    if data_type == DataType.SINGING:
        force_32k = True
        reasons.append("singing detected (high pitch complexity)")
    
    if has_fx:
        force_32k = True
        reasons.append("FX/reverb contamination detected")
    
    if quality in [DataQuality.POOR, DataQuality.UNUSABLE]:
        force_32k = True
        reasons.append(f"quality tier is {quality.value}")
    
    if force_32k:
        return 32000, f"32kHz (safe mode): {'; '.join(reasons)}"
    
    # Check for 48k eligibility
    allow_48k = (
        duration >= thresholds.min_duration_for_48k and
        segments >= thresholds.min_segments_for_48k and
        quality == DataQuality.EXCELLENT and
        data_type == DataType.SPEECH and
        not has_fx
    )
    
    if allow_48k:
        return 48000, f"48kHz (high quality): {duration/60:.0f}min clean speech, excellent quality"
    
    # Default to 40k
    return 40000, f"40kHz (standard): {duration/60:.1f}min, {quality.value} quality"


def _select_epochs(
    mode: TrainingPlanMode,
    analysis: DatasetAnalysis,
    thresholds: PlanThresholds,
    batch_size: int,
) -> Tuple[int, int, int, int, str]:
    """
    Select epochs based on TARGET STEPS, not arbitrary epoch counts.
    
    STEP-BASED TRAINING (NEW):
    - Primary target: total optimizer steps
    - epochs = ceil(target_steps / steps_per_epoch)
    - steps_per_epoch = floor(num_segments / batch_size)
    
    This ensures small datasets get enough training by calculating
    how many epochs are needed to reach the target step count.
    
    Returns:
        (epochs, target_steps, steps_per_epoch, estimated_total_steps, justification)
    """
    segments = analysis.num_segments
    data_type = analysis.data_type
    quality = analysis.quality_tier
    has_fx = analysis.has_fx_contamination
    
    # Calculate steps_per_epoch
    steps_per_epoch = max(1, segments // batch_size)
    
    # Select target steps based on data type
    if mode == TrainingPlanMode.FINE_TUNE:
        target_steps = thresholds.target_steps_fine_tune
        base_reason = f"fine-tune: {target_steps} steps"
    elif data_type == DataType.SINGING:
        target_steps = thresholds.target_steps_singing
        base_reason = f"singing: {target_steps} steps"
    else:
        target_steps = thresholds.target_steps_speech
        base_reason = f"speech: {target_steps} steps"
    
    # Quality adjustment
    quality_mult = {
        DataQuality.EXCELLENT: 1.2,
        DataQuality.GOOD: 1.0,
        DataQuality.FAIR: 0.9,
        DataQuality.POOR: 0.8,
        DataQuality.UNUSABLE: 0.6,
    }.get(quality, 1.0)
    
    target_steps = int(target_steps * quality_mult)
    
    # Apply FX penalty
    if has_fx:
        target_steps = int(target_steps * 0.8)
        base_reason += ", FX detected (-20%)"
    
    # Ensure within bounds
    target_steps = max(thresholds.min_total_steps, min(target_steps, thresholds.max_total_steps))
    
    # Calculate epochs needed to reach target steps
    epochs = max(1, (target_steps + steps_per_epoch - 1) // steps_per_epoch)  # ceiling division
    
    # Apply epoch caps
    caps_applied = []
    
    if mode == TrainingPlanMode.FINE_TUNE:
        cap = thresholds.max_epochs_fine_tune
        if epochs > cap:
            epochs = cap
            caps_applied.append(f"fine-tune cap {cap}")
    
    if data_type == DataType.SINGING:
        cap = thresholds.max_epochs_singing
        if epochs > cap:
            epochs = cap
            caps_applied.append(f"singing cap {cap}")
    
    if has_fx:
        cap = thresholds.max_epochs_fx_noisy
        if epochs > cap:
            epochs = cap
            caps_applied.append(f"FX cap {cap}")
    
    # Absolute max
    epochs = min(epochs, thresholds.max_epochs_new_model)
    
    # Recalculate actual total steps after epoch cap
    estimated_total_steps = steps_per_epoch * epochs
    
    # Build justification
    justification = (
        f"{epochs} epochs: target {target_steps} steps, "
        f"{steps_per_epoch} steps/epoch × {epochs} epochs = {estimated_total_steps} total steps "
        f"({base_reason}, {quality.value} quality ×{quality_mult})"
    )
    if caps_applied:
        justification += f" [capped: {', '.join(caps_applied)}]"
    
    if estimated_total_steps < thresholds.min_total_steps:
        justification += f" ⚠️ WARNING: only {estimated_total_steps} steps (min {thresholds.min_total_steps})"
    
    return epochs, target_steps, steps_per_epoch, estimated_total_steps, justification


def _select_batch_size(
    sample_rate: int,
    gpu_memory_gb: float,
    analysis: DatasetAnalysis,
    thresholds: PlanThresholds,
) -> Tuple[int, str]:
    """
    Select batch size based on dataset size and GPU memory.
    
    STEP-BASED LOGIC (NEW):
    - Small datasets MUST use smaller batch sizes to get enough steps per epoch
    - Formula: steps_per_epoch = num_segments / batch_size
    - Goal: steps_per_epoch >= min_steps_per_epoch (default 10)
    
    Batch size rules by segment count:
    - <100 segments: batch=4 (tiny datasets need every gradient update)
    - 100-300 segments: batch=6 (small datasets)
    - 300-800 segments: batch=8 (medium datasets)
    - 800-2000 segments: batch=12 (large datasets)
    - 2000+ segments: batch=16 (huge datasets)
    """
    segments = analysis.num_segments
    reasons = []
    
    # STEP 1: Get batch size from dataset size rules (PRIMARY)
    batch_rules = thresholds.batch_size_rules
    if segments < batch_rules["tiny"]["max_segments"]:
        dataset_batch = batch_rules["tiny"]["batch_size"]
        dataset_tier = "tiny"
    elif segments < batch_rules["small"]["max_segments"]:
        dataset_batch = batch_rules["small"]["batch_size"]
        dataset_tier = "small"
    elif segments < batch_rules["medium"]["max_segments"]:
        dataset_batch = batch_rules["medium"]["batch_size"]
        dataset_tier = "medium"
    elif segments < batch_rules["large"]["max_segments"]:
        dataset_batch = batch_rules["large"]["batch_size"]
        dataset_tier = "large"
    else:
        dataset_batch = batch_rules["huge"]["batch_size"]
        dataset_tier = "huge"
    
    reasons.append(f"batch={dataset_batch} for {dataset_tier} dataset ({segments} segments)")
    
    # STEP 2: Apply GPU memory cap (SECONDARY)
    if gpu_memory_gb >= 16:
        gpu_batch = 16
    elif gpu_memory_gb >= 12:
        gpu_batch = 14
    elif gpu_memory_gb >= 8:
        gpu_batch = 12
    elif gpu_memory_gb >= 6:
        gpu_batch = 8
    else:
        gpu_batch = 4
    
    batch_size = min(dataset_batch, gpu_batch)
    if batch_size < dataset_batch:
        reasons.append(f"limited to {batch_size} by {gpu_memory_gb:.0f}GB VRAM")
    
    # STEP 3: Apply 48k constraint
    if sample_rate == 48000:
        if batch_size > thresholds.max_batch_size_48k:
            batch_size = thresholds.max_batch_size_48k
            reasons.append(f"reduced to {batch_size} for 48kHz")
    
    # STEP 4: Ensure minimum steps per epoch
    if segments > 0:
        steps_per_epoch = segments // batch_size
        if steps_per_epoch < thresholds.min_steps_per_epoch:
            # Further reduce batch size to get more steps
            new_batch = max(thresholds.min_batch_size, segments // thresholds.min_steps_per_epoch)
            if new_batch < batch_size:
                old_steps = steps_per_epoch
                batch_size = new_batch
                steps_per_epoch = segments // batch_size
                reasons.append(f"reduced to {batch_size} for {steps_per_epoch} steps/epoch (was {old_steps})")
    
    # Ensure minimum
    batch_size = max(thresholds.min_batch_size, batch_size)
    
    return batch_size, f"batch_size={batch_size}: {'; '.join(reasons)}"


def _resolve_pretrained(
    sample_rate: int,
    version: str,
    use_pitch_guidance: bool,
    assets_dir: Path,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Resolve pretrained model paths with validation"""
    warnings = []
    
    sr_k = sample_rate // 1000  # 40000 -> 40
    
    if version == "v2":
        pretrain_dir = assets_dir / "pretrained_v2"
        if use_pitch_guidance:
            g_name = f"f0G{sr_k}k.pth"
            d_name = f"f0D{sr_k}k.pth"
        else:
            g_name = f"G{sr_k}k.pth"
            d_name = f"D{sr_k}k.pth"
    else:
        pretrain_dir = assets_dir / "pretrained"
        g_name = f"G{sr_k}k.pth"
        d_name = f"D{sr_k}k.pth"
    
    g_path = pretrain_dir / g_name
    d_path = pretrain_dir / d_name
    
    if not g_path.exists():
        warnings.append(f"Pretrained G not found: {g_path}")
        g_path = None
    
    if not d_path.exists():
        warnings.append(f"Pretrained D not found: {d_path}")
        d_path = None
    
    return (
        str(g_path) if g_path else None,
        str(d_path) if d_path else None,
        warnings
    )


# =============================================================================
# WATCHDOG CONFIGURATION
# =============================================================================

def configure_watchdogs(
    mode: TrainingPlanMode,
    analysis: DatasetAnalysis,
    thresholds: PlanThresholds = DEFAULT_THRESHOLDS,
) -> List[WatchdogConfig]:
    """
    Configure required watchdog gates based on mode and risk assessment.
    
    All watchdogs are ALWAYS enabled, but thresholds may vary based on risk level.
    """
    watchdogs = []
    
    # Risk assessment
    is_risky = (
        analysis.quality_tier in [DataQuality.POOR, DataQuality.FAIR] or
        analysis.data_type == DataType.SINGING or
        analysis.has_fx_contamination or
        analysis.total_duration_seconds < 300  # <5 min
    )
    
    # 1. Preprocess gate (ALWAYS enabled)
    watchdogs.append(WatchdogConfig(
        gate=WatchdogGate.PREPROCESS,
        enabled=True,
        thresholds={
            'min_rms': thresholds.min_rms_level,
            'max_silence_pct': thresholds.max_silence_percent,
            'max_clipping_pct': thresholds.max_clipping_percent,
            'min_segments': thresholds.min_segments_absolute,
        },
        abort_on_fail=True,
    ))
    
    # 2. F0 extraction gate (ALWAYS enabled)
    watchdogs.append(WatchdogConfig(
        gate=WatchdogGate.F0_EXTRACTION,
        enabled=True,
        thresholds={
            'min_voiced_pct': thresholds.min_voiced_warning,
            'min_voiced_pct_hard': thresholds.min_voiced_fail,
            'max_outlier_pct': thresholds.f0_max_outlier_percent,
            'max_outlier_pct_hard': thresholds.f0_max_outlier_percent_fail,
            'f0_range': thresholds.f0_valid_range,
        },
        abort_on_fail=True,
    ))
    
    # 3. Early training stuck detector (ALWAYS enabled)
    watchdogs.append(WatchdogConfig(
        gate=WatchdogGate.EARLY_TRAINING,
        enabled=True,
        thresholds={
            'check_after_steps': thresholds.stuck_loss_check_steps,
            'loss_variance_threshold': thresholds.stuck_loss_tolerance,
            'max_stuck_iterations': thresholds.max_stuck_iterations,
            'abort_on_stuck_alone': False,  # Only abort if smoke test also fails
            'abort_on_nan': True,
        },
        abort_on_fail=False,  # Deferred to smoke test
    ))
    
    # 4. Smoke test (ALWAYS enabled, but trigger timing varies)
    smoke_test_epoch = 1 if is_risky else thresholds.early_checkpoint_epoch
    
    watchdogs.append(WatchdogConfig(
        gate=WatchdogGate.SMOKE_TEST,
        enabled=True,
        thresholds={
            'trigger_epoch': smoke_test_epoch,
            'min_crest_factor': thresholds.smoke_test_min_crest,
            'max_dc_offset': thresholds.smoke_test_max_dc,
            'max_spectral_flatness': thresholds.smoke_test_max_flatness,
        },
        abort_on_fail=True,
    ))
    
    return watchdogs


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    mode: TrainingPlanMode,
    locked_params: Optional[LockedParams],
    analysis: DatasetAnalysis,
    config: SuggestedConfig,
    watchdogs: List[WatchdogConfig],
    warnings: List[str],
    errors: List[str],
) -> str:
    """Generate human-readable training plan report"""
    
    lines = []
    lines.append("=" * 70)
    lines.append("RVC TRAINING PLAN REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Mode
    lines.append(f"MODE: {mode.value.upper()}")
    if locked_params:
        lines.append(f"  Locked parameters from existing model:")
        lines.append(f"    - Sample rate: {locked_params.sample_rate}Hz")
        lines.append(f"    - Version: {locked_params.version}")
        lines.append(f"    - Pitch guidance: {locked_params.use_pitch_guidance}")
        if locked_params.source_checkpoint:
            lines.append(f"    - Source checkpoint: {locked_params.source_checkpoint}")
    lines.append("")
    
    # Dataset Analysis
    lines.append("-" * 70)
    lines.append("DATASET ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Source files: {analysis.num_source_files}")
    lines.append(f"  Total duration: {analysis.total_duration_seconds/60:.1f} minutes")
    lines.append(f"  Preprocessed segments: {analysis.num_segments}")
    lines.append(f"  Batches per epoch: ~{analysis.estimated_batches_per_epoch}")
    lines.append("")
    lines.append(f"  Data type: {analysis.data_type.value.upper()}")
    lines.append(f"    - Voiced frames: {analysis.voiced_percent:.1f}%")
    lines.append(f"    - Pitch range: {analysis.pitch_range_octaves:.2f} octaves")
    lines.append(f"    - F0: {analysis.f0_min:.0f}-{analysis.f0_max:.0f}Hz (mean={analysis.f0_mean:.0f}, std={analysis.f0_std:.0f})")
    lines.append("")
    lines.append(f"  Quality tier: {analysis.quality_tier.value.upper()} (score: {analysis.quality_score:.0f}/100)")
    lines.append(f"    - RMS level: {analysis.rms_median:.4f} (p10={analysis.rms_p10:.4f}, p90={analysis.rms_p90:.4f})")
    lines.append(f"    - Silence ratio: {analysis.silence_ratio*100:.1f}%")
    lines.append(f"    - Clipping ratio: {analysis.clipping_ratio*100:.2f}%")
    lines.append(f"    - DC offset: {analysis.dc_offset_max:.4f}")
    lines.append(f"    - Spectral flatness: {analysis.spectral_flatness_median:.3f}")
    lines.append(f"    - FX contamination: {'YES' if analysis.has_fx_contamination else 'No'}")
    lines.append("")
    
    # Suggested Config
    lines.append("-" * 70)
    lines.append("SUGGESTED CONFIGURATION")
    lines.append("-" * 70)
    lines.append(f"  Sample rate: {config.sample_rate}Hz")
    lines.append(f"    → {config.justifications.get('sample_rate', 'N/A')}")
    lines.append(f"  Epochs: {config.epochs}")
    lines.append(f"    → {config.justifications.get('epochs', 'N/A')}")
    lines.append(f"  Batch size: {config.batch_size}")
    lines.append(f"    → {config.justifications.get('batch_size', 'N/A')}")
    lines.append(f"  Learning rate: {config.learning_rate_g:.0e}")
    lines.append(f"    → {config.justifications.get('learning_rate', 'N/A')}")
    lines.append(f"  FP16: {config.fp16_run}")
    lines.append(f"    → {config.justifications.get('fp16', 'N/A')}")
    lines.append(f"  Save every: {config.save_every_epoch} epochs")
    lines.append(f"    → {config.justifications.get('save_every', 'N/A')}")
    lines.append(f"  F0 method: {config.f0_method}")
    lines.append(f"  Version: {config.version}")
    lines.append(f"  Pitch guidance: {config.use_pitch_guidance}")
    lines.append("")
    lines.append(f"  Pretrained G: {config.pretrain_G}")
    lines.append(f"  Pretrained D: {config.pretrain_D}")
    lines.append("")
    
    # Watchdogs
    lines.append("-" * 70)
    lines.append("WATCHDOG GATES (fail-fast protection)")
    lines.append("-" * 70)
    for wd in watchdogs:
        status = "ENABLED" if wd.enabled else "disabled"
        abort = "ABORT" if wd.abort_on_fail else "warn"
        lines.append(f"  [{status}] {wd.gate.value}: {abort} on fail")
        for key, val in wd.thresholds.items():
            lines.append(f"         {key}: {val}")
    lines.append("")
    
    # Warnings
    if warnings:
        lines.append("-" * 70)
        lines.append("⚠️  WARNINGS")
        lines.append("-" * 70)
        for w in warnings:
            lines.append(f"  • {w}")
        lines.append("")
    
    # Errors
    if errors:
        lines.append("-" * 70)
        lines.append("❌ ERRORS (cannot proceed)")
        lines.append("-" * 70)
        for e in errors:
            lines.append(f"  • {e}")
        lines.append("")
    
    # Ready rule
    lines.append("-" * 70)
    lines.append("READY RULE")
    lines.append("-" * 70)
    lines.append("  Model will NOT be marked ready unless:")
    lines.append("    ✓ Preprocess gate PASS")
    lines.append("    ✓ F0 extraction gate PASS")
    lines.append("    ✓ At least one smoke test PASS")
    lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def recommend_training_plan(
    model_name: str,
    audio_paths: List[str],
    model_dir: Optional[str] = None,
    preprocessed_dir: Optional[str] = None,
    f0_dir: Optional[str] = None,
    assets_dir: Optional[str] = None,
    gpu_memory_gb: float = 8.0,
    force_mode: Optional[TrainingPlanMode] = None,
    user_overrides: Optional[Dict[str, Any]] = None,
    thresholds: Optional[PlanThresholds] = None,
) -> TrainingPlan:
    """
    Generate a comprehensive training plan for RVC model training.
    
    This is the main entry point for the training plan system. It:
    1. Detects training mode (new/resume/fine-tune)
    2. Analyzes the dataset (type, quality)
    3. Calculates optimal configuration
    4. Configures watchdog gates
    5. Generates a human-readable report
    
    Args:
        model_name: Name for the model
        audio_paths: List of audio file paths to train on
        model_dir: Directory for model artifacts (checkpoints, logs)
        preprocessed_dir: Directory with preprocessed WAVs (optional)
        f0_dir: Directory with F0 files (optional)
        assets_dir: Directory with pretrained models
        gpu_memory_gb: Available GPU memory
        force_mode: Override detected mode (with validation)
        user_overrides: User-specified config overrides
        thresholds: Custom threshold configuration
        
    Returns:
        TrainingPlan with all recommendations and report
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    # Resolve paths
    model_path = Path(model_dir) if model_dir else Path(f"logs/{model_name}")
    assets_path = Path(assets_dir) if assets_dir else Path("assets")
    
    preproc_path = Path(preprocessed_dir) if preprocessed_dir else model_path / "0_gt_wavs"
    f0_path = Path(f0_dir) if f0_dir else model_path / "2a_f0"
    
    all_warnings = []
    all_errors = []
    
    # 1. MODE DETECTION
    mode, locked_params, mode_issues = detect_training_mode(
        model_dir=model_path,
        force_mode=force_mode,
    )
    
    # Mode issues are errors if they prevented the requested mode
    if force_mode and mode != force_mode:
        all_errors.extend(mode_issues)
    else:
        all_warnings.extend(mode_issues)
    
    # 2. DATASET ANALYSIS
    analysis = analyze_dataset(
        audio_paths=audio_paths,
        preprocessed_dir=preproc_path if preproc_path.exists() else None,
        f0_dir=f0_path if f0_path.exists() else None,
        thresholds=thresholds,
    )
    
    # Check for unusable data
    if analysis.quality_tier == DataQuality.UNUSABLE:
        all_errors.append(
            f"Data quality is UNUSABLE (score: {analysis.quality_score:.0f}/100). "
            "Cannot proceed with training. Issues: "
            f"silence={analysis.silence_ratio*100:.0f}%, "
            f"voiced={analysis.voiced_percent:.0f}%, "
            f"duration={analysis.total_duration_seconds/60:.1f}min"
        )
    
    # 3. CONFIG RULES
    config, config_warnings, config_errors = calculate_suggested_config(
        mode=mode,
        locked_params=locked_params,
        analysis=analysis,
        gpu_memory_gb=gpu_memory_gb,
        assets_dir=assets_path,
        thresholds=thresholds,
        user_overrides=user_overrides,
    )
    all_warnings.extend(config_warnings)
    all_errors.extend(config_errors)
    
    # 4. WATCHDOG CONFIGURATION
    watchdogs = configure_watchdogs(
        mode=mode,
        analysis=analysis,
        thresholds=thresholds,
    )
    
    # 5. GENERATE REPORT
    report = generate_report(
        mode=mode,
        locked_params=locked_params,
        analysis=analysis,
        config=config,
        watchdogs=watchdogs,
        warnings=all_warnings,
        errors=all_errors,
    )
    
    # Determine if we can proceed
    can_proceed = len(all_errors) == 0
    
    return TrainingPlan(
        mode=mode,
        locked_params=locked_params,
        dataset_analysis=analysis,
        suggested_config=config,
        required_watchdogs=watchdogs,
        warnings=all_warnings,
        errors=all_errors,
        report=report,
        can_proceed=can_proceed,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python training_plan.py <model_name> <audio_file1> [audio_file2] ...")
        print("\nGenerates a training plan for the given audio files.")
        sys.exit(1)
    
    model_name = sys.argv[1]
    audio_paths = sys.argv[2:]
    
    logging.basicConfig(level=logging.INFO)
    
    plan = recommend_training_plan(
        model_name=model_name,
        audio_paths=audio_paths,
        gpu_memory_gb=12.0,
    )
    
    print(plan.report)
    
    if not plan.can_proceed:
        print("\n❌ CANNOT PROCEED WITH TRAINING")
        sys.exit(1)
    else:
        print("\n✅ TRAINING PLAN READY")
        sys.exit(0)


# =============================================================================
# HELPER FUNCTIONS FOR STEP-BASED CONFIGURATION
# =============================================================================

def calculate_step_based_config(
    num_segments: int,
    data_type: str = "speech",
    gpu_memory_gb: float = 12.0,
    thresholds: Optional[PlanThresholds] = None,
) -> Dict[str, Any]:
    """
    Quick helper to calculate step-based training config from segment count.
    
    This is the key insight from debugging collapsed models:
    - Optimizer STEPS matter, not epochs
    - Small datasets need smaller batch sizes to get more steps/epoch
    - Formula: total_steps = (num_segments / batch_size) * epochs
    
    Args:
        num_segments: Number of preprocessed audio segments (~6s each)
        data_type: "speech" or "singing" (affects target steps)
        gpu_memory_gb: Available GPU VRAM (caps batch size)
        thresholds: Custom thresholds (optional)
        
    Returns:
        Dict with recommended batch_size, epochs, target_steps, steps_per_epoch, etc.
    
    Example:
        >>> config = calculate_step_based_config(84)  # Lexi's segment count
        >>> print(config)
        {
            'batch_size': 4,          # Reduced from 16 for small dataset
            'steps_per_epoch': 21,    # 84 / 4 = 21
            'target_steps': 1800,     # Speech target
            'epochs': 86,             # 1800 / 21 = 86 (rounded up)
            'estimated_total_steps': 1806,
            'justification': '...'
        }
        
        # Compare to original collapsed config:
        # batch_size=16, epochs=55 → only 84/16 * 55 = 290 steps (COLLAPSED!)
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    # Select batch size based on dataset size
    batch_rules = thresholds.batch_size_rules
    if num_segments < batch_rules["tiny"]["max_segments"]:
        batch_size = batch_rules["tiny"]["batch_size"]
        tier = "tiny"
    elif num_segments < batch_rules["small"]["max_segments"]:
        batch_size = batch_rules["small"]["batch_size"]
        tier = "small"
    elif num_segments < batch_rules["medium"]["max_segments"]:
        batch_size = batch_rules["medium"]["batch_size"]
        tier = "medium"
    elif num_segments < batch_rules["large"]["max_segments"]:
        batch_size = batch_rules["large"]["batch_size"]
        tier = "large"
    else:
        batch_size = batch_rules["huge"]["batch_size"]
        tier = "huge"
    
    # Apply GPU memory cap
    if gpu_memory_gb >= 16:
        max_batch = 16
    elif gpu_memory_gb >= 12:
        max_batch = 14
    elif gpu_memory_gb >= 8:
        max_batch = 12
    elif gpu_memory_gb >= 6:
        max_batch = 8
    else:
        max_batch = 4
    
    batch_size = min(batch_size, max_batch)
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, num_segments // batch_size)
    
    # Select target steps based on data type
    if data_type == "singing":
        target_steps = thresholds.target_steps_singing
    else:
        target_steps = thresholds.target_steps_speech
    
    # Calculate epochs needed
    epochs = max(1, (target_steps + steps_per_epoch - 1) // steps_per_epoch)
    epochs = min(epochs, thresholds.max_epochs_new_model)
    
    # Actual total steps
    estimated_total_steps = steps_per_epoch * epochs
    
    # Build justification
    justification = (
        f"{tier} dataset ({num_segments} segments) → batch_size={batch_size}, "
        f"{steps_per_epoch} steps/epoch × {epochs} epochs = {estimated_total_steps} total steps "
        f"(target: {target_steps} for {data_type})"
    )
    
    # Warning if below minimum
    warning = None
    if estimated_total_steps < thresholds.min_total_steps:
        warning = (
            f"⚠️ Only {estimated_total_steps} total steps - below minimum {thresholds.min_total_steps}! "
            f"Consider adding more training data."
        )
    
    return {
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "target_steps": target_steps,
        "epochs": epochs,
        "estimated_total_steps": estimated_total_steps,
        "dataset_tier": tier,
        "justification": justification,
        "warning": warning,
    }
