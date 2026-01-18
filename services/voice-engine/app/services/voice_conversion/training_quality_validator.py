"""
Training Quality Validator - Catches collapsed/broken models EARLY

This module provides comprehensive quality gates for RVC model training:

1. PREPROCESSING GATE: Validates input audio quality before training
2. F0 EXTRACTION GATE: Validates pitch extraction quality per RVC wiki
3. CHECKPOINT SMOKE TEST: Tests first checkpoint for collapsed output
4. TRAINING LOSS MONITOR: Detects stuck/divergent loss patterns

Usage:
    from app.services.voice_conversion.training_quality_validator import (
        validate_preprocessing_quality,
        validate_f0_extraction,
        smoke_test_checkpoint,
        validate_training_logs,
    )

ROOT CAUSE DOCUMENTATION:
========================
The lexi-11 model failure was caused by TRAINING COLLAPSE where:
- loss_mel=75.000 was STUCK for ALL 78 epochs (never decreased)
- The generator never learned to reconstruct mel spectrograms
- Output has:
  - High DC offset (~-0.023)
  - Low crest factor (~1.4 vs 4.8 for working model)
  - 4x more zero crossings (5911/s vs 1507/s)
  - Dominant frequency at 2400Hz vs 561Hz

This module prevents such failures by:
1. Detecting stuck loss patterns during training
2. Smoke testing checkpoints with real inference
3. Validating preprocessing and F0 extraction quality
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of quality validation."""
    passed: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'issues': self.issues,
            'warnings': self.warnings,
            'metrics': self.metrics
        }
    
    def fail(self, msg: str):
        self.passed = False
        self.issues.append(msg)
        
    def warn(self, msg: str):
        self.warnings.append(msg)


# =============================================================================
# PREPROCESSING QUALITY GATE
# =============================================================================

def validate_preprocessing_quality(
    audio_dir: str,
    sample_rate: int = 48000,
    min_rms: float = 0.01,
    max_silence_pct: float = 30.0,
    max_clipping_pct: float = 1.0,
) -> QualityResult:
    """
    Validate preprocessing quality of training audio.
    
    Checks:
    - RMS level (too quiet = bad features)
    - Silence percentage (too much silence = sparse training)
    - Clipping percentage (clipped audio = distorted features)
    - Sample rate consistency
    
    Args:
        audio_dir: Directory containing preprocessed WAV files
        sample_rate: Expected sample rate
        min_rms: Minimum acceptable RMS level
        max_silence_pct: Maximum acceptable silence percentage
        max_clipping_pct: Maximum acceptable clipping percentage
        
    Returns:
        QualityResult with validation status and metrics
    """
    from scipy.io import wavfile
    
    result = QualityResult()
    path = Path(audio_dir)
    
    if not path.exists():
        result.fail(f"Audio directory not found: {audio_dir}")
        return result
    
    wav_files = list(path.glob('*.wav'))
    if not wav_files:
        result.fail("No WAV files found in directory")
        return result
    
    result.metrics['num_files'] = len(wav_files)
    
    total_duration = 0.0
    total_silence = 0.0
    total_clipped = 0
    total_samples = 0
    rms_values = []
    sr_mismatches = 0
    
    for wav_file in wav_files:
        try:
            sr, audio = wavfile.read(str(wav_file))
            
            # Check sample rate
            if sr != sample_rate:
                sr_mismatches += 1
                
            # Convert to float
            if audio.dtype == np.int16:
                audio_f = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio_f = audio.astype(np.float32) / 2147483648.0
            else:
                audio_f = audio.astype(np.float32)
            
            # Make mono if stereo
            if len(audio_f.shape) > 1:
                audio_f = audio_f.mean(axis=1)
            
            # Calculate metrics
            duration = len(audio_f) / sr
            total_duration += duration
            total_samples += len(audio_f)
            
            # RMS
            rms = np.sqrt(np.mean(audio_f ** 2))
            rms_values.append(rms)
            
            # Silence (RMS below threshold)
            window_size = sr // 10  # 100ms windows
            for i in range(0, len(audio_f), window_size):
                window = audio_f[i:i+window_size]
                if len(window) > 0:
                    window_rms = np.sqrt(np.mean(window ** 2))
                    if window_rms < 0.001:  # -60 dB
                        total_silence += len(window) / sr
            
            # Clipping
            clipped = np.sum(np.abs(audio_f) > 0.99)
            total_clipped += clipped
            
        except Exception as e:
            result.warn(f"Could not process {wav_file.name}: {e}")
    
    # Calculate aggregate metrics
    result.metrics['total_duration_sec'] = total_duration
    result.metrics['mean_rms'] = float(np.mean(rms_values)) if rms_values else 0
    result.metrics['min_rms'] = float(np.min(rms_values)) if rms_values else 0
    result.metrics['silence_pct'] = 100.0 * total_silence / total_duration if total_duration > 0 else 0
    result.metrics['clipping_pct'] = 100.0 * total_clipped / total_samples if total_samples > 0 else 0
    result.metrics['sr_mismatches'] = sr_mismatches
    
    # Validate thresholds
    if result.metrics['mean_rms'] < min_rms:
        result.fail(f"Audio too quiet: mean RMS {result.metrics['mean_rms']:.4f} < {min_rms}")
    
    if result.metrics['silence_pct'] > max_silence_pct:
        result.fail(f"Too much silence: {result.metrics['silence_pct']:.1f}% > {max_silence_pct}%")
    
    if result.metrics['clipping_pct'] > max_clipping_pct:
        result.fail(f"Too much clipping: {result.metrics['clipping_pct']:.2f}% > {max_clipping_pct}%")
    
    if sr_mismatches > 0:
        result.warn(f"{sr_mismatches} files have mismatched sample rate (expected {sample_rate})")
    
    if total_duration < 60:
        result.warn(f"Very short training data: {total_duration:.1f}s. Recommend 5+ minutes.")
    elif total_duration < 300:
        result.warn(f"Short training data: {total_duration:.1f}s. Quality may be limited.")
    
    return result


# =============================================================================
# F0 EXTRACTION QUALITY GATE (Per RVC Wiki)
# =============================================================================

def validate_f0_extraction(
    f0_dir: str,
    f0nsf_dir: Optional[str] = None,
    feature_dir: Optional[str] = None,
    min_voiced_pct: float = 20.0,
    min_voiced_pct_hard: float = 10.0,  # Hard fail threshold
    max_outlier_pct: float = 30.0,
    max_outlier_pct_hard: float = 50.0,  # Hard fail threshold
    f0_range: Tuple[float, float] = (40.0, 1100.0),  # Human speech range
) -> QualityResult:
    """
    Validate F0/pitch extraction quality per RVC wiki requirements.
    
    Checks:
    - Voiced frame percentage (too low = silence/noise input)
    - F0 outlier percentage (abnormal pitch values)
    - F0 range validity (within human speech range)
    - Feature/F0 alignment
    
    Args:
        f0_dir: Directory containing F0 .npy files (2a_f0/)
        f0nsf_dir: Optional directory for F0 NSF files (2b_f0nsf/)
        feature_dir: Optional directory for HuBERT features (3_feature768/)
        min_voiced_pct: Minimum percentage of voiced frames
        max_outlier_pct: Maximum percentage of outlier F0 values
        f0_range: Valid F0 frequency range in Hz
        
    Returns:
        QualityResult with validation status and metrics
    """
    result = QualityResult()
    path = Path(f0_dir)
    
    if not path.exists():
        result.fail(f"F0 directory not found: {f0_dir}")
        return result
    
    f0_files = list(path.glob('*.npy'))
    if not f0_files:
        result.fail("No F0 .npy files found")
        return result
    
    result.metrics['num_files'] = len(f0_files)
    
    total_frames = 0
    voiced_frames = 0
    outlier_frames = 0
    all_voiced_f0 = []
    alignment_issues = 0
    
    for f0_file in f0_files:
        try:
            f0 = np.load(str(f0_file))
            total_frames += len(f0)
            
            # Count voiced frames (f0 > 0)
            voiced_mask = f0 > 0
            voiced_frames += np.sum(voiced_mask)
            
            # Count outliers (outside valid range)
            if voiced_mask.any():
                voiced_f0 = f0[voiced_mask]
                all_voiced_f0.extend(voiced_f0.tolist())
                outliers = np.sum((voiced_f0 < f0_range[0]) | (voiced_f0 > f0_range[1]))
                outlier_frames += outliers
            
            # Check F0/F0nsf alignment
            if f0nsf_dir:
                f0nsf_file = Path(f0nsf_dir) / f0_file.name
                if f0nsf_file.exists():
                    f0nsf = np.load(str(f0nsf_file))
                    # F0nsf is typically half the frames
                    expected_ratio = len(f0) / len(f0nsf)
                    if not (1.8 < expected_ratio < 2.2):
                        alignment_issues += 1
            
            # Check F0/feature alignment
            if feature_dir:
                feat_file = Path(feature_dir) / f0_file.name
                if feat_file.exists():
                    feat = np.load(str(feat_file))
                    # Features should have same or similar frame count
                    if abs(len(f0) - feat.shape[0]) > 5:  # Allow small difference
                        alignment_issues += 1
                        
        except Exception as e:
            result.warn(f"Could not process {f0_file.name}: {e}")
    
    # Calculate metrics
    voiced_pct = 100.0 * voiced_frames / total_frames if total_frames > 0 else 0
    outlier_pct = 100.0 * outlier_frames / voiced_frames if voiced_frames > 0 else 0
    
    result.metrics['total_frames'] = total_frames
    result.metrics['voiced_frames'] = voiced_frames
    result.metrics['voiced_pct'] = voiced_pct
    result.metrics['outlier_pct'] = outlier_pct
    result.metrics['alignment_issues'] = alignment_issues
    
    if all_voiced_f0:
        all_voiced_f0 = np.array(all_voiced_f0)
        result.metrics['f0_min'] = float(np.min(all_voiced_f0))
        result.metrics['f0_max'] = float(np.max(all_voiced_f0))
        result.metrics['f0_mean'] = float(np.mean(all_voiced_f0))
        result.metrics['f0_std'] = float(np.std(all_voiced_f0))
    
    # Validate thresholds with adaptive levels (hard fail vs warning)
    if voiced_pct < min_voiced_pct_hard:
        result.fail(f"Too few voiced frames: {voiced_pct:.1f}% < {min_voiced_pct_hard}%. "
                   "Input is mostly silence/noise - training will fail.")
    elif voiced_pct < min_voiced_pct:
        result.warn(f"Low voiced frames: {voiced_pct:.1f}% < {min_voiced_pct}%. "
                   "Consider checking audio quality or slicing settings.")
    
    if outlier_pct > max_outlier_pct_hard:
        result.fail(f"Too many F0 outliers: {outlier_pct:.1f}% > {max_outlier_pct_hard}%. "
                   "Pitch extraction has failed - check for noisy/corrupted audio.")
    elif outlier_pct > max_outlier_pct:
        result.warn(f"High F0 outliers: {outlier_pct:.1f}% > {max_outlier_pct}%. "
                   "Pitch extraction may be struggling with some samples.")
    
    if alignment_issues > 0:
        result.warn(f"{alignment_issues} files have F0/feature alignment issues")
    
    if voiced_frames == 0:
        result.fail("NO VOICED FRAMES FOUND - pitch extraction completely failed!")
    
    return result


# =============================================================================
# CHECKPOINT SMOKE TEST
# =============================================================================

def smoke_test_checkpoint(
    checkpoint_path: str,
    test_audio: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    min_crest_factor: float = 2.0,  # Lowered for windowed analysis on active frames
    max_dc_offset: float = 0.02,
    max_spectral_flatness: float = 0.35,  # Slightly relaxed for windowed analysis
) -> QualityResult:
    """
    Smoke test a checkpoint by running inference and checking output quality.
    
    This catches collapsed/broken generators EARLY in training.
    
    Checks:
    - DC offset (high DC = generator bias)
    - Crest factor (low crest = compressed/collapsed output)
    - Spectral flatness (high flatness = noise-like output)
    - Zero crossing rate (abnormal rate = buzz/artifacts)
    
    Args:
        checkpoint_path: Path to checkpoint file
        test_audio: Test audio array (16kHz float32), uses synthetic if None
        sample_rate: Sample rate of test audio
        min_crest_factor: Minimum acceptable peak-to-RMS ratio
        max_dc_offset: Maximum acceptable DC offset magnitude
        max_spectral_flatness: Maximum acceptable spectral flatness
        
    Returns:
        QualityResult with validation status and metrics
    """
    result = QualityResult()
    
    # Generate test audio if not provided
    if test_audio is None:
        # Synthetic voice-like test signal
        t = np.linspace(0, 3, 3 * sample_rate, dtype=np.float32)
        f0 = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Vibrato-like pitch
        test_audio = 0.3 * np.sin(2 * np.pi * f0.cumsum() / sample_rate)
        test_audio = test_audio.astype(np.float32)
    
    try:
        # Import and run inference
        import sys
        sys.path.insert(0, '/app')
        from app.services.voice_conversion.model_manager import ModelManager, RVCInferParams
        
        mm = ModelManager(model_dir="assets/models", input_sample_rate=sample_rate)
        
        # Load model
        if not mm.load_model(checkpoint_path, None):
            result.fail(f"Failed to load checkpoint: {checkpoint_path}")
            return result
        
        # Run inference
        params = RVCInferParams(
            f0_up_key=0,
            f0_method="rmvpe",
            index_rate=0.0,  # No index for smoke test
            resample_sr=0
        )
        
        output = mm.infer(test_audio, params)
        
        if output is None or len(output) == 0:
            result.fail("Inference returned empty output")
            return result
        
        # Calculate quality metrics on ACTIVE FRAMES only (to avoid silence skewing metrics)
        output = output.astype(np.float32)
        output_sr = 48000  # RVC outputs at 48kHz typically
        
        # Compute per-window metrics and use median (robust to silence/outliers)
        window_size = int(output_sr * 0.025)  # 25ms windows
        hop_size = int(output_sr * 0.010)  # 10ms hop
        
        window_rms = []
        window_crest = []
        window_sf = []
        window_zc = []
        
        # RMS threshold for "active" frames (about -40dB)
        rms_threshold = 0.01
        
        for i in range(0, len(output) - window_size, hop_size):
            window = output[i:i + window_size]
            w_rms = np.sqrt(np.mean(window ** 2))
            
            # Skip silent frames
            if w_rms < rms_threshold:
                continue
            
            window_rms.append(w_rms)
            
            # Crest factor per window
            w_peak = np.abs(window).max()
            window_crest.append(w_peak / w_rms if w_rms > 0.001 else 0)
            
            # Spectral flatness per window
            fft = np.abs(np.fft.rfft(window))
            fft = fft[fft > 0]
            if len(fft) > 0:
                sf = np.exp(np.mean(np.log(fft))) / np.mean(fft)
                window_sf.append(sf)
            
            # Zero crossings per window
            zc = np.sum(np.diff(np.sign(window)) != 0)
            window_zc.append(zc * output_sr / window_size)  # Convert to per-second
        
        # Use median of active frames (robust to outliers)
        if len(window_rms) < 10:
            # Not enough active frames - use global metrics as fallback
            result.warn(f"Very few active frames ({len(window_rms)}) - audio may be mostly silent")
            dc_offset = float(np.mean(output))
            rms = float(np.sqrt(np.mean(output ** 2)))
            peak = float(np.abs(output).max())
            crest_factor = peak / rms if rms > 0.001 else 0
            fft = np.abs(np.fft.rfft(output))
            fft = fft[fft > 0]
            spectral_flatness = float(np.exp(np.mean(np.log(fft))) / np.mean(fft)) if len(fft) > 0 else 1.0
            zc_rate = float(np.sum(np.diff(np.sign(output)) != 0) / (len(output) / output_sr))
        else:
            # DC offset is still computed globally (bias affects entire signal)
            dc_offset = float(np.mean(output))
            
            # Use median of active frames for other metrics
            rms = float(np.median(window_rms))
            peak = float(np.abs(output).max())  # Peak is still global
            crest_factor = float(np.median(window_crest))
            spectral_flatness = float(np.median(window_sf)) if window_sf else 1.0
            zc_rate = float(np.median(window_zc)) if window_zc else 0
        
        result.metrics['dc_offset'] = dc_offset
        result.metrics['rms'] = rms
        result.metrics['peak'] = peak
        result.metrics['crest_factor'] = crest_factor
        result.metrics['spectral_flatness'] = spectral_flatness
        result.metrics['zero_crossing_rate'] = zc_rate
        result.metrics['active_frames'] = len(window_rms)
        
        # Validate thresholds
        if abs(dc_offset) > max_dc_offset:
            result.fail(f"High DC offset: {dc_offset:.4f} (max: ±{max_dc_offset}). "
                       "Generator may have learned a bias.")
        
        if crest_factor < min_crest_factor:
            result.fail(f"Low crest factor: {crest_factor:.2f} (min: {min_crest_factor}). "
                       "Output may be collapsed/compressed.")
        
        if spectral_flatness > max_spectral_flatness:
            result.fail(f"High spectral flatness: {spectral_flatness:.3f} (max: {max_spectral_flatness}). "
                       "Output may be noise-like.")
        
        # Warning for abnormal zero crossing rate
        if zc_rate > 4000:
            result.warn(f"High zero crossing rate: {zc_rate:.0f}/s. May indicate buzz/artifacts.")
        elif zc_rate < 100:
            result.warn(f"Very low zero crossing rate: {zc_rate:.0f}/s. May indicate collapsed output.")
        
        mm.unload_model()
        
    except Exception as e:
        result.fail(f"Smoke test failed with error: {e}")
        import traceback
        result.metrics['error_traceback'] = traceback.format_exc()
    
    return result


# =============================================================================
# TRAINING LOSS MONITOR
# =============================================================================

def validate_training_logs(
    log_path: str,
    min_loss_decrease_pct: float = 10.0,
    max_stuck_epochs: int = 5,
    check_mel_loss: bool = True,
    expected_loss_pattern: str = 'decreasing',
) -> QualityResult:
    """
    Validate training logs for loss patterns indicating training failure.
    
    Detects:
    - Stuck loss (loss_mel=75.000 for many epochs = COLLAPSE)
    - NaN losses (training diverged)
    - Increasing loss (training going wrong direction)
    - Loss not decreasing enough
    
    Args:
        log_path: Path to train.log file
        min_loss_decrease_pct: Minimum expected decrease from first to last epoch
        max_stuck_epochs: Maximum consecutive epochs with same loss before warning
        check_mel_loss: Whether to specifically check mel loss (critical for RVC)
        expected_loss_pattern: 'decreasing', 'stable', or 'any'
        
    Returns:
        QualityResult with validation status and metrics
    """
    result = QualityResult()
    path = Path(log_path)
    
    if not path.exists():
        result.fail(f"Log file not found: {log_path}")
        return result
    
    # Parse loss values from log
    loss_pattern = re.compile(
        r'loss_disc=([\d.]+|nan), loss_gen=([\d.]+|nan), '
        r'loss_fm=([\d.]+|nan),loss_mel=([\d.]+|nan), loss_kl=([\d.]+|nan)'
    )
    
    mel_losses = []
    gen_losses = []
    nan_count = 0
    
    with open(path, 'r') as f:
        for line in f:
            match = loss_pattern.search(line)
            if match:
                disc, gen, fm, mel, kl = match.groups()
                
                # Check for NaN
                if 'nan' in [disc, gen, fm, mel, kl]:
                    nan_count += 1
                    continue
                
                try:
                    mel_losses.append(float(mel))
                    gen_losses.append(float(gen))
                except ValueError:
                    pass
    
    result.metrics['total_iterations'] = len(mel_losses)
    result.metrics['nan_iterations'] = nan_count
    
    if len(mel_losses) == 0:
        if nan_count > 0:
            result.fail(f"Training produced ONLY NaN losses ({nan_count} iterations). "
                       "Training completely diverged.")
        else:
            result.fail("No loss values found in log")
        return result
    
    result.metrics['first_mel_loss'] = mel_losses[0]
    result.metrics['last_mel_loss'] = mel_losses[-1]
    result.metrics['min_mel_loss'] = min(mel_losses)
    result.metrics['max_mel_loss'] = max(mel_losses)
    
    # Check for NaN
    if nan_count > len(mel_losses) * 0.1:  # >10% NaN
        result.fail(f"High NaN rate: {nan_count}/{len(mel_losses)+nan_count} iterations. "
                   "Training may be unstable (try lower learning rate or disable fp16).")
    elif nan_count > 0:
        result.warn(f"{nan_count} NaN iterations detected")
    
    # Check for stuck mel loss (CRITICAL - this is what caused lexi-11 failure)
    # NOTE: "mel loss stuck" alone is a WARNING, not a hard fail.
    # It becomes a FAIL only when combined with smoke test failures.
    # This avoids false positives from logging quirks while still flagging the issue.
    if check_mel_loss:
        unique_mel = set(mel_losses)
        most_common_mel = max(set(mel_losses), key=mel_losses.count)
        stuck_count = mel_losses.count(most_common_mel)
        stuck_pct = 100.0 * stuck_count / len(mel_losses)
        result.metrics['mel_loss_stuck_pct'] = stuck_pct
        result.metrics['mel_loss_stuck_value'] = most_common_mel
        
        if len(unique_mel) == 1:
            # 100% stuck - this is a SEVERE warning, but we defer to smoke test for final verdict
            result.warn(f"Mel loss STUCK at {mel_losses[0]} for ALL {len(mel_losses)} iterations! "
                       "Generator may have collapsed - smoke test required for confirmation.")
            result.metrics['mel_loss_collapse_suspected'] = True
        elif stuck_pct > 90:
            result.warn(f"Mel loss stuck at {most_common_mel} for {stuck_pct:.0f}% of training. "
                       "Generator may be collapsed - recommend smoke test.")
            result.metrics['mel_loss_collapse_suspected'] = True
        elif stuck_pct > 50:
            result.warn(f"Mel loss unchanged for {stuck_pct:.0f}% of training")
            result.metrics['mel_loss_collapse_suspected'] = False
        else:
            result.metrics['mel_loss_collapse_suspected'] = False
    
    # Check for decreasing trend
    if expected_loss_pattern == 'decreasing':
        first_window = np.mean(mel_losses[:min(100, len(mel_losses))])
        last_window = np.mean(mel_losses[-min(100, len(mel_losses)):])
        
        decrease_pct = 100.0 * (first_window - last_window) / first_window if first_window > 0 else 0
        result.metrics['loss_decrease_pct'] = decrease_pct
        
        if decrease_pct < min_loss_decrease_pct:
            if decrease_pct < 0:
                result.fail(f"Loss INCREASED by {-decrease_pct:.1f}% during training!")
            else:
                result.warn(f"Loss only decreased {decrease_pct:.1f}% (expected {min_loss_decrease_pct}%+)")
    
    # Check for consecutive stuck epochs
    current_val = None
    current_count = 0
    max_stuck_count = 0
    
    for loss in mel_losses:
        if loss == current_val:
            current_count += 1
            max_stuck_count = max(max_stuck_count, current_count)
        else:
            current_val = loss
            current_count = 1
    
    result.metrics['max_consecutive_stuck'] = max_stuck_count
    
    if max_stuck_count > max_stuck_epochs * 10:  # iterations, not epochs
        result.warn(f"Loss stuck for {max_stuck_count} consecutive iterations")
    
    return result


# =============================================================================
# COMPREHENSIVE MODEL QUALITY CHECK
# =============================================================================

def validate_trained_model(
    model_dir: str,
    run_smoke_test: bool = True,
) -> QualityResult:
    """
    Comprehensive validation of a trained model.
    
    Runs all applicable quality gates:
    1. Preprocessing quality (if 0_gt_wavs exists)
    2. F0 extraction quality (if 2a_f0 exists)
    3. Training log analysis (if train.log exists)
    4. Checkpoint smoke test (if inference model exists)
    
    Args:
        model_dir: Path to model training directory
        run_smoke_test: Whether to run inference smoke test
        
    Returns:
        QualityResult with comprehensive validation results
    """
    result = QualityResult()
    path = Path(model_dir)
    
    if not path.exists():
        result.fail(f"Model directory not found: {model_dir}")
        return result
    
    # 1. Preprocessing quality
    gt_wavs = path / '0_gt_wavs'
    if gt_wavs.exists():
        logger.info("Validating preprocessing quality...")
        preproc_result = validate_preprocessing_quality(str(gt_wavs))
        result.metrics['preprocessing'] = preproc_result.to_dict()
        if not preproc_result.passed:
            result.issues.extend([f"[Preprocessing] {i}" for i in preproc_result.issues])
        result.warnings.extend([f"[Preprocessing] {w}" for w in preproc_result.warnings])
    
    # 2. F0 extraction quality
    f0_dir = path / '2a_f0'
    if f0_dir.exists():
        logger.info("Validating F0 extraction quality...")
        f0nsf_dir = path / '2b_f0nsf'
        feat_dir = path / '3_feature768'
        f0_result = validate_f0_extraction(
            str(f0_dir),
            str(f0nsf_dir) if f0nsf_dir.exists() else None,
            str(feat_dir) if feat_dir.exists() else None
        )
        result.metrics['f0_extraction'] = f0_result.to_dict()
        if not f0_result.passed:
            result.issues.extend([f"[F0] {i}" for i in f0_result.issues])
        result.warnings.extend([f"[F0] {w}" for w in f0_result.warnings])
    
    # 3. Training log analysis
    log_result = None
    train_log = path / 'train.log'
    if train_log.exists():
        logger.info("Validating training logs...")
        log_result = validate_training_logs(str(train_log))
        result.metrics['training_logs'] = log_result.to_dict()
        if not log_result.passed:
            result.issues.extend([f"[Training] {i}" for i in log_result.issues])
        result.warnings.extend([f"[Training] {w}" for w in log_result.warnings])
    
    # 4. Checkpoint smoke test
    smoke_result = None
    if run_smoke_test:
        # Find inference model
        model_files = list(path.glob('*.pth'))
        inference_models = [f for f in model_files 
                          if not f.name.startswith('G_') and not f.name.startswith('D_')]
        
        if inference_models:
            logger.info(f"Running smoke test on {inference_models[0].name}...")
            smoke_result = smoke_test_checkpoint(str(inference_models[0]))
            result.metrics['smoke_test'] = smoke_result.to_dict()
            if not smoke_result.passed:
                result.issues.extend([f"[SmokeTest] {i}" for i in smoke_result.issues])
            result.warnings.extend([f"[SmokeTest] {w}" for w in smoke_result.warnings])
    
    # 5. Cross-validate: mel loss stuck + smoke test failure = definite collapse
    # This makes the validator robust even if logging metrics aren't perfect
    if log_result and smoke_result:
        mel_collapse_suspected = log_result.metrics.get('mel_loss_collapse_suspected', False)
        smoke_failed = not smoke_result.passed
        
        if mel_collapse_suspected and smoke_failed:
            # Upgrade warning to failure with combined evidence
            result.fail("[CONFIRMED COLLAPSE] Training logs show stuck mel loss AND smoke test failed. "
                       f"Model is broken (mel stuck at {log_result.metrics.get('mel_loss_stuck_value')}, "
                       f"crest={smoke_result.metrics.get('crest_factor', 'N/A')}).")
        elif mel_collapse_suspected and not smoke_failed:
            # Stuck loss but passing smoke test - just keep as warning
            logger.info("Mel loss appeared stuck but smoke test passed - model may be okay")
    
    # Final verdict
    result.passed = len(result.issues) == 0
    
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python training_quality_validator.py <model_dir> [--no-smoke-test]")
        print("\nValidates a trained RVC model directory for quality issues.")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    run_smoke = '--no-smoke-test' not in sys.argv
    
    logging.basicConfig(level=logging.INFO)
    
    result = validate_trained_model(model_dir, run_smoke_test=run_smoke)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    
    if not result.passed:
        print("\n❌ MODEL QUALITY CHECK FAILED")
        for issue in result.issues:
            print(f"  • {issue}")
    else:
        print("\n✅ MODEL QUALITY CHECK PASSED")
    
    if result.warnings:
        print("\n⚠️  WARNINGS:")
        for warning in result.warnings:
            print(f"  • {warning}")
    
    sys.exit(0 if result.passed else 1)
