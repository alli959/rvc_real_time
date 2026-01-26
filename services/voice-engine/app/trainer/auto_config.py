"""
Auto-Configuration for RVC Training

Automatically determines optimal training parameters based on:
- Audio duration and number of samples
- Audio quality (SNR estimation)
- Whether it's a new model or fine-tuning existing
- Source type (wizard recordings vs uploads)
- Base model quality metrics
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrainingMode(str, Enum):
    """Training mode based on use case"""
    NEW_MODEL = "new_model"           # Training from scratch
    FINE_TUNE = "fine_tune"           # Adding data to existing model
    QUICK_TEST = "quick_test"         # Fast training for testing


class AudioSourceType(str, Enum):
    """Type of audio source"""
    WIZARD = "wizard"                 # Controlled wizard recordings
    UPLOAD = "upload"                 # User uploaded files
    MIXED = "mixed"                   # Combination of both


class QualityLevel(str, Enum):
    """Estimated audio quality level"""
    EXCELLENT = "excellent"           # Studio quality, high SNR
    GOOD = "good"                     # Clean recordings
    FAIR = "fair"                     # Some noise/issues
    POOR = "poor"                     # Significant quality issues


@dataclass
class AudioAnalysis:
    """Analysis results for training audio"""
    total_duration_seconds: float
    num_files: int
    estimated_chunks: int             # ~6 second chunks after preprocessing
    avg_snr_db: float                 # Signal-to-noise ratio estimate
    quality_level: QualityLevel
    source_type: AudioSourceType
    has_existing_model: bool
    existing_model_epochs: int = 0    # How many epochs the existing model was trained
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_duration_seconds": self.total_duration_seconds,
            "num_files": self.num_files,
            "estimated_chunks": self.estimated_chunks,
            "avg_snr_db": round(self.avg_snr_db, 1),
            "quality_level": self.quality_level.value,
            "source_type": self.source_type.value,
            "has_existing_model": self.has_existing_model,
            "existing_model_epochs": self.existing_model_epochs
        }


@dataclass
class AutoTrainingConfig:
    """Auto-generated training configuration with explanations"""
    # Core parameters
    epochs: int
    batch_size: int
    save_every_epoch: int
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 1e-4
    
    # Mode
    training_mode: TrainingMode = TrainingMode.NEW_MODEL
    
    # Explanations for user
    config_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Estimated training time
    estimated_minutes: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "save_every_epoch": self.save_every_epoch,
            "learning_rate_g": self.learning_rate_g,
            "learning_rate_d": self.learning_rate_d,
            "training_mode": self.training_mode.value,
            "config_summary": self.config_summary,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "estimated_minutes": round(self.estimated_minutes, 1)
        }


def estimate_snr(audio_data: np.ndarray, sample_rate: int) -> float:
    """
    Estimate Signal-to-Noise Ratio (SNR) of audio.
    
    Uses a simple method based on energy distribution.
    Higher values = cleaner audio.
    """
    try:
        # Normalize
        audio = audio_data.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-10:
            return 0.0
        
        # Estimate noise floor from quietest segments
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = frame_length // 2
        
        num_frames = (len(audio) - frame_length) // hop_length
        if num_frames < 10:
            return 20.0  # Default for very short audio
        
        frame_energies = []
        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            if energy > 1e-10:
                frame_energies.append(energy)
        
        if not frame_energies:
            return 20.0
        
        frame_energies = np.array(frame_energies)
        
        # Noise floor estimate: 10th percentile of frame energies
        noise_floor = np.percentile(frame_energies, 10)
        # Signal estimate: 90th percentile
        signal_level = np.percentile(frame_energies, 90)
        
        if noise_floor < 1e-10:
            noise_floor = 1e-10
        
        snr_db = 20 * np.log10(signal_level / noise_floor)
        
        return min(max(snr_db, 0), 60)  # Clamp to reasonable range
        
    except Exception as e:
        logger.warning(f"SNR estimation failed: {e}")
        return 20.0  # Default moderate SNR


def analyze_audio_files(
    audio_paths: List[str],
    source_categories: Dict[str, List[str]] = None,
    existing_model_path: Optional[str] = None
) -> AudioAnalysis:
    """
    Analyze audio files to determine training parameters.
    
    Args:
        audio_paths: List of audio file paths
        source_categories: Dict mapping categories to file paths (wizard vs uploaded)
        existing_model_path: Path to existing model if fine-tuning
    """
    import soundfile as sf
    
    total_duration = 0.0
    snr_values = []
    
    # Determine source type
    wizard_count = 0
    upload_count = 0
    
    if source_categories:
        for cat, paths in source_categories.items():
            if cat == "uploaded":
                upload_count += len(paths)
            else:
                wizard_count += len(paths)
    else:
        upload_count = len(audio_paths)
    
    # Analyze each file
    for audio_path in audio_paths:
        try:
            info = sf.info(audio_path)
            total_duration += info.duration
            
            # Sample a portion of longer files for SNR estimation
            if info.duration > 30:
                # Read middle 30 seconds
                start_frame = int((info.duration / 2 - 15) * info.samplerate)
                audio_data, sr = sf.read(
                    audio_path, 
                    start=start_frame, 
                    frames=int(30 * info.samplerate)
                )
            else:
                audio_data, sr = sf.read(audio_path)
            
            # Handle stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            snr = estimate_snr(audio_data, sr)
            snr_values.append(snr)
            
        except Exception as e:
            logger.warning(f"Could not analyze {audio_path}: {e}")
    
    # Calculate averages
    avg_snr = np.mean(snr_values) if snr_values else 20.0
    
    # Estimate chunks (RVC uses ~6 second chunks)
    chunk_duration = 6.0
    estimated_chunks = int(total_duration / chunk_duration)
    
    # Determine quality level based on SNR
    if avg_snr >= 35:
        quality_level = QualityLevel.EXCELLENT
    elif avg_snr >= 25:
        quality_level = QualityLevel.GOOD
    elif avg_snr >= 15:
        quality_level = QualityLevel.FAIR
    else:
        quality_level = QualityLevel.POOR
    
    # Determine source type
    if wizard_count > 0 and upload_count > 0:
        source_type = AudioSourceType.MIXED
    elif wizard_count > 0:
        source_type = AudioSourceType.WIZARD
    else:
        source_type = AudioSourceType.UPLOAD
    
    # Check for existing model
    has_existing_model = False
    existing_epochs = 0
    if existing_model_path:
        model_path = Path(existing_model_path)
        if model_path.exists():
            has_existing_model = True
            # Try to read epochs from metadata
            metadata_path = model_path.parent / "model_metadata.json"
            if metadata_path.exists():
                try:
                    import json
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    existing_epochs = metadata.get("training_config", {}).get("epochs", 0)
                except:
                    pass
    
    return AudioAnalysis(
        total_duration_seconds=total_duration,
        num_files=len(audio_paths),
        estimated_chunks=estimated_chunks,
        avg_snr_db=avg_snr,
        quality_level=quality_level,
        source_type=source_type,
        has_existing_model=has_existing_model,
        existing_model_epochs=existing_epochs
    )


def calculate_optimal_config(
    analysis: AudioAnalysis,
    training_mode: Optional[TrainingMode] = None,
    gpu_memory_gb: float = 8.0
) -> AutoTrainingConfig:
    """
    Calculate optimal training configuration based on audio analysis.
    
    The algorithm considers:
    1. Data volume (duration, chunks) - more data = more epochs safe
    2. Audio quality - higher quality = can train longer without overfitting
    3. Training mode - fine-tuning needs fewer epochs
    4. Source type - wizard recordings are more consistent
    """
    recommendations = []
    warnings = []
    
    duration = analysis.total_duration_seconds
    chunks = analysis.estimated_chunks
    quality = analysis.quality_level
    
    # Determine training mode if not specified
    if training_mode is None:
        if analysis.has_existing_model:
            training_mode = TrainingMode.FINE_TUNE
        else:
            training_mode = TrainingMode.NEW_MODEL
    
    # ==========================================================================
    # EPOCH CALCULATION
    # ==========================================================================
    
    # Base epochs based on audio duration
    # RVC models converge quickly - 30-80 epochs is usually sufficient
    # Excessive epochs cause overfitting and waste time
    if duration < 60:  # < 1 minute
        base_epochs = 20
        warnings.append("Very limited audio data. Results may be inconsistent.")
    elif duration < 120:  # 1-2 minutes
        base_epochs = 30
        recommendations.append("Consider adding more audio for better results.")
    elif duration < 300:  # 2-5 minutes
        base_epochs = 40
    elif duration < 600:  # 5-10 minutes
        base_epochs = 50
    elif duration < 900:  # 10-15 minutes
        base_epochs = 60
    elif duration < 1800:  # 15-30 minutes
        base_epochs = 70
    else:  # > 30 minutes
        base_epochs = 80
        recommendations.append("Excellent amount of training data!")
    
    # Quality multiplier
    quality_multipliers = {
        QualityLevel.EXCELLENT: 1.3,   # Can train longer with clean data
        QualityLevel.GOOD: 1.1,
        QualityLevel.FAIR: 0.9,
        QualityLevel.POOR: 0.7         # Risk of learning noise patterns
    }
    quality_mult = quality_multipliers.get(quality, 1.0)
    
    if quality == QualityLevel.POOR:
        warnings.append("Audio quality is low. Consider cleaning or re-recording.")
    elif quality == QualityLevel.EXCELLENT:
        recommendations.append("High quality audio detected - training should produce excellent results.")
    
    # Source type adjustment
    source_multipliers = {
        AudioSourceType.WIZARD: 1.1,   # Consistent, controlled recordings
        AudioSourceType.UPLOAD: 1.0,   # Unknown consistency
        AudioSourceType.MIXED: 1.05    # Diverse data
    }
    source_mult = source_multipliers.get(analysis.source_type, 1.0)
    
    # Training mode adjustment
    if training_mode == TrainingMode.FINE_TUNE:
        # Fine-tuning needs fewer epochs to avoid catastrophic forgetting
        mode_mult = 0.4
        # Also consider how much the existing model was trained
        if analysis.existing_model_epochs > 0:
            # If model was already heavily trained, use even fewer epochs
            if analysis.existing_model_epochs >= 500:
                mode_mult = 0.25
            elif analysis.existing_model_epochs >= 300:
                mode_mult = 0.3
        recommendations.append("Fine-tuning existing model - using conservative epochs to preserve quality.")
    elif training_mode == TrainingMode.QUICK_TEST:
        mode_mult = 0.3
        recommendations.append("Quick test mode - reduced epochs for faster iteration.")
    else:
        mode_mult = 1.0
    
    # Calculate final epochs
    epochs = int(base_epochs * quality_mult * source_mult * mode_mult)
    
    # Ensure reasonable bounds - cap at 100 to prevent excessive training
    # RVC models converge fast; more epochs = overfitting risk
    epochs = max(20, min(epochs, 100))
    
    # ==========================================================================
    # STEP-BASED BATCH SIZE CALCULATION
    # ==========================================================================
    # The key insight: training quality depends on TOTAL OPTIMIZER STEPS, not epochs.
    # Minimum ~1500 steps needed to avoid model collapse (output = noise).
    # 
    # With small datasets, large batch sizes = too few steps per epoch = collapse.
    # Example: 84 segments / batch=16 = 5 steps/epoch × 55 epochs = 275 steps (BAD!)
    #          84 segments / batch=4  = 21 steps/epoch × 143 epochs = 3003 steps (GOOD!)
    
    # Target steps based on training type
    TARGET_STEPS_SPEECH = 1800    # Minimum steps for speech voice model
    TARGET_STEPS_SINGING = 3000   # More steps for singing (broader range)
    MIN_TOTAL_STEPS = 1500        # Absolute minimum to avoid collapse
    
    # For very small datasets, we need to reduce batch size further to get more steps
    # Start with batch_size based on chunk count, then reduce if needed
    if chunks < 20:
        batch_size = 2      # Tiny dataset: use batch=2 for maximum steps
    elif chunks < 100:
        batch_size = 4      # Small dataset: maximize steps
    elif chunks < 300:
        batch_size = 6      # Medium-small dataset
    elif chunks < 800:
        batch_size = 8      # Medium dataset
    elif chunks < 2000:
        batch_size = 12     # Large dataset
    else:
        batch_size = 16     # Huge dataset: can use larger batches
    
    # Calculate steps per epoch with chosen batch size
    steps_per_epoch = max(1, chunks // batch_size)
    
    # Now recalculate epochs to achieve target steps
    target_steps = TARGET_STEPS_SPEECH  # Use speech target (can be parameterized later)
    required_epochs = max(20, math.ceil(target_steps / steps_per_epoch))
    
    # Apply the quality/mode multipliers to required epochs
    final_epochs = int(required_epochs * quality_mult * source_mult * mode_mult)
    
    # Ensure we have enough total steps
    estimated_total_steps = final_epochs * steps_per_epoch
    if estimated_total_steps < MIN_TOTAL_STEPS:
        # Need more epochs to meet minimum steps
        final_epochs = max(final_epochs, math.ceil(MIN_TOTAL_STEPS / steps_per_epoch))
    
    # For tiny datasets, allow high epoch counts to achieve minimum steps
    # With 7 chunks @ batch=2, we get ~3 steps/epoch, need 600 epochs for 1800 steps
    # Max 2000 epochs should be plenty for any reasonable dataset
    max_epochs = 2000 if chunks < 50 else 500 if chunks < 200 else 300
    epochs = max(20, min(final_epochs, max_epochs))
    
    # Recalculate for logging
    estimated_total_steps = epochs * steps_per_epoch
    
    # Warn if we still can't reach minimum steps (extremely small dataset)
    if estimated_total_steps < MIN_TOTAL_STEPS:
        logger.warning(f"⚠️ Dataset too small: {chunks} chunks can only achieve {estimated_total_steps} steps "
                      f"(minimum recommended: {MIN_TOTAL_STEPS}). Consider adding more training data.")
    
    # Log step-based calculation
    logger.info(f"Step-based config: {chunks} segments / batch={batch_size} = {steps_per_epoch} steps/epoch × {epochs} epochs = {estimated_total_steps} total steps")
    
    # ==========================================================================
    # SAVE FREQUENCY
    # ==========================================================================
    
    # Save checkpoints at step-based intervals
    # Target ~6-8 checkpoints during training
    target_checkpoints = 6
    save_every = max(10, epochs // target_checkpoints)
    
    # ==========================================================================
    # LEARNING RATE
    # ==========================================================================
    
    # Default learning rates work well for most cases
    # Reduce for fine-tuning to preserve existing knowledge
    if training_mode == TrainingMode.FINE_TUNE:
        lr_g = 5e-5
        lr_d = 5e-5
    else:
        lr_g = 1e-4
        lr_d = 1e-4
    
    # ==========================================================================
    # TIME ESTIMATION
    # ==========================================================================
    
    # Time per epoch depends on dataset size and batch size
    # Based on empirical testing with RTX 4070 Ti (12GB):
    # - Each epoch processes all chunks in batches
    # - Approximate time: 1.5-3.5 minutes per epoch depending on chunk count
    # Formula: base_time + (chunks / batch_size) * time_per_batch
    batches_per_epoch = max(1, chunks / batch_size)
    # ~1.5-2 seconds per batch on modern GPU
    seconds_per_batch = 2.0
    seconds_per_epoch = batches_per_epoch * seconds_per_batch
    # Add overhead for epoch transitions, checkpointing, etc.
    seconds_per_epoch += 10  # ~10 seconds overhead per epoch
    
    estimated_minutes = (epochs * seconds_per_epoch) / 60
    
    # Add preprocessing time estimate
    preprocessing_minutes = (duration / 60) * 0.5  # ~30 seconds per minute of audio
    estimated_minutes += preprocessing_minutes
    
    # ==========================================================================
    # BUILD SUMMARY
    # ==========================================================================
    
    summary_parts = [
        f"Training {epochs} epochs ({estimated_total_steps} steps)",
        f"with batch size {batch_size}",
        f"for {duration/60:.1f} minutes of {quality.value} quality audio"
    ]
    
    if training_mode == TrainingMode.FINE_TUNE:
        summary_parts.insert(0, "Fine-tuning mode:")
    
    config_summary = " ".join(summary_parts)
    
    # Add step-based training info
    recommendations.insert(0, f"Step-based training: {estimated_total_steps} optimizer steps (target: {target_steps})")
    
    # Add duration-specific recommendations
    if duration >= 600:  # 10+ minutes
        recommendations.append(f"With {duration/60:.0f} minutes of audio, the model should capture voice characteristics well.")
    
    if chunks >= 100:
        recommendations.append(f"Dataset will create ~{chunks} training samples - good diversity.")
    
    return AutoTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        save_every_epoch=save_every,
        learning_rate_g=lr_g,
        learning_rate_d=lr_d,
        training_mode=training_mode,
        config_summary=config_summary,
        recommendations=recommendations,
        warnings=warnings,
        estimated_minutes=estimated_minutes
    )


def get_auto_config(
    audio_paths: List[str],
    source_categories: Dict[str, List[str]] = None,
    existing_model_path: Optional[str] = None,
    training_mode: Optional[TrainingMode] = None,
    gpu_memory_gb: float = 8.0
) -> Tuple[AutoTrainingConfig, AudioAnalysis]:
    """
    High-level function to get automatic training configuration.
    
    Returns both the config and the analysis for transparency.
    """
    logger.info(f"Analyzing {len(audio_paths)} audio files for auto-configuration...")
    
    # Analyze the audio
    analysis = analyze_audio_files(
        audio_paths=audio_paths,
        source_categories=source_categories,
        existing_model_path=existing_model_path
    )
    
    logger.info(f"Audio analysis: {analysis.total_duration_seconds:.1f}s total, "
                f"{analysis.estimated_chunks} chunks, "
                f"quality={analysis.quality_level.value}, "
                f"SNR={analysis.avg_snr_db:.1f}dB")
    
    # Calculate optimal config
    config = calculate_optimal_config(
        analysis=analysis,
        training_mode=training_mode,
        gpu_memory_gb=gpu_memory_gb
    )
    
    logger.info(f"Auto-config: {config.epochs} epochs, "
                f"batch_size={config.batch_size}, "
                f"mode={config.training_mode.value}, "
                f"estimated_time={config.estimated_minutes:.1f}min")
    
    return config, analysis
