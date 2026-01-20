"""
Training Watchdogs - Fail-Fast Gates for RVC Training

This module implements runtime watchdog gates that monitor training quality
and can ABORT training early when collapse or failure is detected.

WATCHDOG GATES:
1. PREPROCESS_GATE: Validates input audio quality before training starts
2. F0_GATE: Validates pitch extraction quality with fallback method retry
3. EARLY_TRAINING_GATE: Detects stuck/divergent loss patterns during training
4. SMOKE_TEST_GATE: Tests checkpoint inference output for collapse signatures

CRITICAL BEHAVIOR:
- Watchdogs can STOP training and mark job FAILED
- Debug artifacts are saved for post-mortem analysis
- Human-readable failure reports are generated

Usage:
    from app.trainer.training_watchdogs import WatchdogManager
    
    manager = WatchdogManager(
        job_id="abc123",
        model_dir="/path/to/model",
        thresholds=thresholds,
    )
    
    # Run preprocess gate
    result = manager.run_preprocess_gate(preprocessed_dir)
    if not result.passed:
        # Abort training
        
    # Monitor during training
    manager.update_loss_metrics(step=100, loss_mel=75.0, loss_kl=9.0)
    if manager.should_abort():
        # Stop training
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class WatchdogStatus(str, Enum):
    """Status of a watchdog gate"""
    NOT_RUN = "not_run"
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class WatchdogResult:
    """Result of a watchdog gate check"""
    gate_name: str
    status: WatchdogStatus
    passed: bool
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    remediation: List[str] = field(default_factory=list)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        return d


@dataclass
class LossMetrics:
    """Training loss metrics for a single step"""
    step: int
    epoch: int
    loss_disc: float
    loss_gen: float
    loss_fm: float
    loss_mel: float
    loss_kl: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingLossHistory:
    """History of training losses for stuck detection"""
    metrics: List[LossMetrics] = field(default_factory=list)
    mel_values: List[float] = field(default_factory=list)
    kl_values: List[float] = field(default_factory=list)
    nan_count: int = 0
    
    def add(self, metrics: LossMetrics):
        self.metrics.append(metrics)
        self.mel_values.append(metrics.loss_mel)
        self.kl_values.append(metrics.loss_kl)
        
        if np.isnan(metrics.loss_mel) or np.isnan(metrics.loss_kl):
            self.nan_count += 1


# =============================================================================
# WATCHDOG MANAGER
# =============================================================================

class WatchdogManager:
    """
    Manages all watchdog gates for a training job.
    
    Coordinates gate execution, tracks state, and provides abort signaling.
    """
    
    def __init__(
        self,
        job_id: str,
        model_dir: str,
        thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
        save_debug_artifacts: bool = True,
    ):
        """
        Initialize watchdog manager.
        
        Args:
            job_id: Unique training job ID
            model_dir: Model directory for saving artifacts
            thresholds: Per-gate threshold configuration
            save_debug_artifacts: Whether to save debug data on failure
        """
        self.job_id = job_id
        self.model_dir = Path(model_dir)
        self.thresholds = thresholds or {}
        self.save_debug_artifacts = save_debug_artifacts
        
        # Gate results
        self.results: Dict[str, WatchdogResult] = {}
        
        # Training loss history
        self.loss_history = TrainingLossHistory()
        
        # Abort flag
        self._abort_requested = False
        self._abort_reason = ""
        
        # Ready rule tracking
        self._preprocess_passed = False
        self._f0_passed = False
        self._smoke_test_passed = False
        
        # Debug directory
        self.debug_dir = self.model_dir / "watchdog_debug"
        
        logger.info(f"WatchdogManager initialized for job {job_id}")
    
    # =========================================================================
    # PREPROCESS GATE
    # =========================================================================
    
    def run_preprocess_gate(
        self,
        preprocessed_dir: str,
        sample_rate: int = 48000,
    ) -> WatchdogResult:
        """
        Run preprocessing quality gate.
        
        Validates:
        - Enough segments exist
        - Audio levels are acceptable
        - Silence ratio is not too high
        - Clipping is minimal
        
        Args:
            preprocessed_dir: Directory with preprocessed WAVs
            sample_rate: Expected sample rate
            
        Returns:
            WatchdogResult with pass/fail status
        """
        gate_name = "preprocess"
        thresholds = self.thresholds.get(gate_name, {})
        
        min_rms = thresholds.get('min_rms', 0.01)
        max_silence_pct = thresholds.get('max_silence_pct', 30.0)
        max_clipping_pct = thresholds.get('max_clipping_pct', 1.0)
        min_segments = thresholds.get('min_segments', 20)
        
        issues = []
        warnings = []
        remediation = []
        metrics = {}
        
        try:
            from scipy.io import wavfile
            
            path = Path(preprocessed_dir)
            if not path.exists():
                return self._fail_result(
                    gate_name,
                    f"Preprocessed directory not found: {preprocessed_dir}",
                    remediation=["Check preprocessing step completed successfully"]
                )
            
            wav_files = list(path.glob("*.wav"))
            num_segments = len(wav_files)
            metrics['num_segments'] = num_segments
            
            if num_segments == 0:
                return self._fail_result(
                    gate_name,
                    "No WAV files found in preprocessed directory",
                    remediation=["Ensure audio files were properly sliced"]
                )
            
            if num_segments < min_segments:
                issues.append(f"Too few segments: {num_segments} < {min_segments}")
                remediation.append(f"Add more audio data (need at least {min_segments} segments)")
            
            # Sample analysis
            sample_size = min(20, num_segments)
            step = max(1, num_segments // sample_size)
            sample_files = [wav_files[i] for i in range(0, num_segments, step)][:sample_size]
            
            rms_values = []
            silence_count = 0
            total_clipped = 0
            total_samples = 0
            sr_mismatches = 0
            
            for wav_file in sample_files:
                try:
                    sr, audio = wavfile.read(str(wav_file))
                    
                    if sr != sample_rate:
                        sr_mismatches += 1
                    
                    # Convert to float
                    if audio.dtype == np.int16:
                        audio_f = audio.astype(np.float32) / 32768.0
                    elif audio.dtype == np.int32:
                        audio_f = audio.astype(np.float32) / 2147483648.0
                    else:
                        audio_f = audio.astype(np.float32)
                    
                    if len(audio_f.shape) > 1:
                        audio_f = audio_f.mean(axis=1)
                    
                    # RMS
                    rms = np.sqrt(np.mean(audio_f ** 2))
                    rms_values.append(rms)
                    
                    if rms < min_rms:
                        silence_count += 1
                    
                    # Clipping
                    total_clipped += np.sum(np.abs(audio_f) > 0.99)
                    total_samples += len(audio_f)
                    
                except Exception as e:
                    warnings.append(f"Could not analyze {wav_file.name}: {e}")
            
            if not rms_values:
                return self._fail_result(
                    gate_name,
                    "Could not analyze any audio files",
                    remediation=["Check audio file format and integrity"]
                )
            
            # Calculate metrics
            mean_rms = float(np.mean(rms_values))
            silence_pct = 100.0 * silence_count / len(sample_files)
            clipping_pct = 100.0 * total_clipped / total_samples if total_samples > 0 else 0
            
            metrics['mean_rms'] = mean_rms
            metrics['silence_pct'] = silence_pct
            metrics['clipping_pct'] = clipping_pct
            metrics['sr_mismatches'] = sr_mismatches
            
            # Validate
            if mean_rms < min_rms:
                issues.append(f"Audio too quiet: RMS {mean_rms:.4f} < {min_rms}")
                remediation.append("Normalize audio to higher levels")
            
            if silence_pct > max_silence_pct:
                issues.append(f"Too much silence: {silence_pct:.1f}% > {max_silence_pct}%")
                remediation.append("Remove silent segments or add more speech")
            
            if clipping_pct > max_clipping_pct:
                issues.append(f"Too much clipping: {clipping_pct:.2f}% > {max_clipping_pct}%")
                remediation.append("Reduce audio gain to prevent clipping")
            
            if sr_mismatches > 0:
                warnings.append(f"{sr_mismatches} files have wrong sample rate")
            
            # Determine status
            if issues:
                result = self._fail_result(
                    gate_name,
                    f"Preprocessing quality check failed: {len(issues)} issues",
                    metrics=metrics,
                    issues=issues,
                    warnings=warnings,
                    remediation=remediation,
                )
            else:
                result = WatchdogResult(
                    gate_name=gate_name,
                    status=WatchdogStatus.PASSED,
                    passed=True,
                    message=f"Preprocessing quality OK: {num_segments} segments, RMS={mean_rms:.3f}",
                    metrics=metrics,
                    warnings=warnings,
                )
                self._preprocess_passed = True
            
        except Exception as e:
            result = self._fail_result(
                gate_name,
                f"Preprocessing gate error: {e}",
                remediation=["Check logs for details"]
            )
        
        self.results[gate_name] = result
        self._save_result(result)
        return result
    
    # =========================================================================
    # F0 EXTRACTION GATE
    # =========================================================================
    
    def run_f0_gate(
        self,
        f0_dir: str,
        f0nsf_dir: Optional[str] = None,
        feature_dir: Optional[str] = None,
        retry_methods: Optional[List[str]] = None,
    ) -> WatchdogResult:
        """
        Run F0 extraction quality gate.
        
        Validates:
        - Voiced frame percentage is sufficient
        - F0 outlier percentage is acceptable
        - F0/feature alignment is correct
        
        If validation fails, can retry with alternative F0 methods.
        
        Args:
            f0_dir: Directory with F0 .npy files
            f0nsf_dir: Optional F0 NSF directory
            feature_dir: Optional HuBERT feature directory
            retry_methods: Alternative F0 methods to try on failure
            
        Returns:
            WatchdogResult with pass/fail status
        """
        gate_name = "f0_extraction"
        thresholds = self.thresholds.get(gate_name, {})
        
        min_voiced_pct = thresholds.get('min_voiced_pct', 20.0)
        min_voiced_pct_hard = thresholds.get('min_voiced_pct_hard', 10.0)
        max_outlier_pct = thresholds.get('max_outlier_pct', 30.0)
        max_outlier_pct_hard = thresholds.get('max_outlier_pct_hard', 50.0)
        f0_range = thresholds.get('f0_range', (40.0, 1100.0))
        
        issues = []
        warnings = []
        remediation = []
        metrics = {}
        
        try:
            path = Path(f0_dir)
            if not path.exists():
                return self._fail_result(
                    gate_name,
                    f"F0 directory not found: {f0_dir}",
                    remediation=["Check F0 extraction step completed"]
                )
            
            f0_files = list(path.glob("*.npy"))
            if not f0_files:
                return self._fail_result(
                    gate_name,
                    "No F0 .npy files found",
                    remediation=["Re-run F0 extraction"]
                )
            
            metrics['num_files'] = len(f0_files)
            
            # Analyze F0 files
            total_frames = 0
            voiced_frames = 0
            outlier_frames = 0
            all_voiced_f0 = []
            alignment_issues = 0
            
            for f0_file in f0_files:
                try:
                    f0 = np.load(str(f0_file))
                    total_frames += len(f0)
                    
                    voiced_mask = f0 > 0
                    voiced_frames += np.sum(voiced_mask)
                    
                    if voiced_mask.any():
                        voiced_f0 = f0[voiced_mask]
                        all_voiced_f0.extend(voiced_f0.tolist())
                        
                        # Count outliers
                        outliers = np.sum(
                            (voiced_f0 < f0_range[0]) | (voiced_f0 > f0_range[1])
                        )
                        outlier_frames += outliers
                    
                    # Check alignment if feature_dir provided
                    if feature_dir:
                        feat_file = Path(feature_dir) / f0_file.name
                        if feat_file.exists():
                            feat = np.load(str(feat_file))
                            if abs(len(f0) - feat.shape[0]) > 5:
                                alignment_issues += 1
                                
                except Exception as e:
                    warnings.append(f"Could not process {f0_file.name}: {e}")
            
            # Calculate metrics
            voiced_pct = 100.0 * voiced_frames / total_frames if total_frames > 0 else 0
            outlier_pct = 100.0 * outlier_frames / voiced_frames if voiced_frames > 0 else 0
            
            metrics['total_frames'] = total_frames
            metrics['voiced_frames'] = voiced_frames
            metrics['voiced_pct'] = voiced_pct
            metrics['outlier_pct'] = outlier_pct
            metrics['alignment_issues'] = alignment_issues
            
            if all_voiced_f0:
                f0_arr = np.array(all_voiced_f0)
                metrics['f0_min'] = float(np.min(f0_arr))
                metrics['f0_max'] = float(np.max(f0_arr))
                metrics['f0_mean'] = float(np.mean(f0_arr))
                metrics['f0_std'] = float(np.std(f0_arr))
            
            # Validate - use two-tier thresholds
            
            # Hard fails (cannot proceed)
            if voiced_frames == 0:
                issues.append("NO VOICED FRAMES - pitch extraction completely failed!")
                remediation.append("Try alternative F0 method (harvest, crepe)")
                remediation.append("Check audio contains speech, not just noise/music")
            
            elif voiced_pct < min_voiced_pct_hard:
                issues.append(f"Critical: Too few voiced frames: {voiced_pct:.1f}% < {min_voiced_pct_hard}%")
                remediation.append("Try alternative F0 method")
                remediation.append("Check audio quality - may be too noisy")
            
            if outlier_pct > max_outlier_pct_hard:
                issues.append(f"Critical: Too many F0 outliers: {outlier_pct:.1f}% > {max_outlier_pct_hard}%")
                remediation.append("Try alternative F0 method")
                remediation.append("Audio may have pitch artifacts or non-speech content")
            
            # Soft warnings
            if voiced_pct < min_voiced_pct and voiced_pct >= min_voiced_pct_hard:
                warnings.append(f"Low voiced frames: {voiced_pct:.1f}% < {min_voiced_pct}%")
            
            if outlier_pct > max_outlier_pct and outlier_pct <= max_outlier_pct_hard:
                warnings.append(f"High F0 outliers: {outlier_pct:.1f}% > {max_outlier_pct}%")
            
            if alignment_issues > 0:
                warnings.append(f"{alignment_issues} files have F0/feature alignment issues")
            
            # Determine result
            if issues:
                result = self._fail_result(
                    gate_name,
                    f"F0 extraction quality check failed: {len(issues)} critical issues",
                    metrics=metrics,
                    issues=issues,
                    warnings=warnings,
                    remediation=remediation,
                )
            else:
                status = WatchdogStatus.WARNING if warnings else WatchdogStatus.PASSED
                result = WatchdogResult(
                    gate_name=gate_name,
                    status=status,
                    passed=True,
                    message=f"F0 extraction OK: {voiced_pct:.1f}% voiced, {outlier_pct:.1f}% outliers",
                    metrics=metrics,
                    warnings=warnings,
                )
                self._f0_passed = True
                
        except Exception as e:
            result = self._fail_result(
                gate_name,
                f"F0 gate error: {e}",
                remediation=["Check logs for details"]
            )
        
        self.results[gate_name] = result
        self._save_result(result)
        return result
    
    # =========================================================================
    # EARLY TRAINING STUCK DETECTOR
    # =========================================================================
    
    def update_loss_metrics(
        self,
        step: int,
        epoch: int,
        loss_disc: float,
        loss_gen: float,
        loss_fm: float,
        loss_mel: float,
        loss_kl: float,
    ) -> Optional[WatchdogResult]:
        """
        Update loss metrics and check for stuck patterns.
        
        Should be called after each training iteration.
        
        Returns:
            WatchdogResult if a check was triggered, None otherwise
        """
        metrics = LossMetrics(
            step=step,
            epoch=epoch,
            loss_disc=loss_disc,
            loss_gen=loss_gen,
            loss_fm=loss_fm,
            loss_mel=loss_mel,
            loss_kl=loss_kl,
        )
        self.loss_history.add(metrics)
        
        # Get thresholds
        thresholds = self.thresholds.get('early_training', {})
        check_after_steps = thresholds.get('check_after_steps', 30)
        
        # Only check after minimum steps
        if step < check_after_steps:
            return None
        
        # Check for issues
        return self._check_training_health()
    
    def _check_training_health(self) -> Optional[WatchdogResult]:
        """Check training loss patterns for issues"""
        gate_name = "early_training"
        thresholds = self.thresholds.get(gate_name, {})
        
        loss_variance_threshold = thresholds.get('loss_variance_threshold', 0.01)
        max_stuck_iterations = thresholds.get('max_stuck_iterations', 5)
        abort_on_nan = thresholds.get('abort_on_nan', True)
        abort_on_stuck_alone = thresholds.get('abort_on_stuck_alone', False)
        
        issues = []
        warnings = []
        metrics = {}
        
        history = self.loss_history
        
        # Check for NaN
        if abort_on_nan and history.nan_count > 0:
            nan_pct = 100.0 * history.nan_count / len(history.metrics)
            if nan_pct > 10:
                issues.append(f"Training produced {history.nan_count} NaN losses ({nan_pct:.1f}%)")
                
                result = self._fail_result(
                    gate_name,
                    f"Training diverged: {history.nan_count} NaN losses",
                    metrics={'nan_count': history.nan_count, 'nan_pct': nan_pct},
                    issues=issues,
                    remediation=[
                        "Try lower learning rate",
                        "Disable FP16 training",
                        "Check for corrupted audio data"
                    ]
                )
                self._request_abort("NaN losses detected - training diverged")
                return result
        
        # Check for stuck mel loss
        if len(history.mel_values) >= max_stuck_iterations:
            recent_mel = history.mel_values[-max_stuck_iterations:]
            mel_variance = np.var(recent_mel)
            mel_unique = len(set(recent_mel))
            
            metrics['mel_variance'] = float(mel_variance)
            metrics['mel_unique_values'] = mel_unique
            metrics['recent_mel_loss'] = recent_mel[-1]
            
            if mel_unique == 1 or mel_variance < loss_variance_threshold:
                warnings.append(
                    f"Mel loss stuck at {recent_mel[-1]:.3f} for {max_stuck_iterations} iterations "
                    f"(variance={mel_variance:.6f})"
                )
                metrics['mel_stuck'] = True
        
        # Check for stuck KL loss
        if len(history.kl_values) >= max_stuck_iterations:
            recent_kl = history.kl_values[-max_stuck_iterations:]
            kl_variance = np.var(recent_kl)
            kl_unique = len(set(recent_kl))
            
            metrics['kl_variance'] = float(kl_variance)
            metrics['kl_unique_values'] = kl_unique
            
            if kl_unique == 1 or kl_variance < loss_variance_threshold:
                warnings.append(
                    f"KL loss stuck at {recent_kl[-1]:.3f} for {max_stuck_iterations} iterations"
                )
                metrics['kl_stuck'] = True
        
        # Both stuck = likely collapse
        both_stuck = metrics.get('mel_stuck', False) and metrics.get('kl_stuck', False)
        
        if both_stuck:
            msg = "Both mel and KL losses are stuck - potential training collapse"
            issues.append(msg)
            
            if abort_on_stuck_alone:
                result = self._fail_result(
                    gate_name,
                    msg,
                    metrics=metrics,
                    issues=issues,
                    warnings=warnings,
                    remediation=[
                        "Check pretrained model compatibility",
                        "Try different sample rate (32k/40k)",
                        "Verify preprocessing output quality"
                    ]
                )
                self._request_abort("Stuck losses - training collapse detected")
                return result
            else:
                # Don't abort yet - wait for smoke test confirmation
                warnings.append("Will run smoke test to confirm collapse")
        
        # Return result if we have warnings
        if warnings:
            return WatchdogResult(
                gate_name=gate_name,
                status=WatchdogStatus.WARNING,
                passed=True,  # Not failing yet
                message="Training health check: warnings detected",
                metrics=metrics,
                warnings=warnings,
                issues=issues,
            )
        
        return None
    
    # =========================================================================
    # SMOKE TEST GATE
    # =========================================================================
    
    def run_smoke_test(
        self,
        checkpoint_path: str,
        test_audio: Optional[np.ndarray] = None,
    ) -> WatchdogResult:
        """
        Run inference smoke test on a checkpoint.
        
        Tests the checkpoint by running inference and checking output for
        collapse signatures (DC offset, low crest factor, noise-like spectrum).
        
        Args:
            checkpoint_path: Path to checkpoint file
            test_audio: Optional test audio (generates synthetic if not provided)
            
        Returns:
            WatchdogResult with pass/fail status
        """
        gate_name = "smoke_test"
        thresholds = self.thresholds.get(gate_name, {})
        
        min_crest_factor = thresholds.get('min_crest_factor', 2.0)
        max_dc_offset = thresholds.get('max_dc_offset', 0.02)
        max_spectral_flatness = thresholds.get('max_spectral_flatness', 0.35)
        
        issues = []
        warnings = []
        remediation = []
        metrics = {}
        
        try:
            # Import inference components
            import sys
            sys.path.insert(0, '/app')
            
            # Generate test audio if not provided
            sample_rate = 16000
            if test_audio is None:
                t = np.linspace(0, 3, 3 * sample_rate, dtype=np.float32)
                f0 = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)
                test_audio = 0.3 * np.sin(2 * np.pi * f0.cumsum() / sample_rate)
                test_audio = test_audio.astype(np.float32)
            
            # Try to load and run inference
            try:
                from app.services.voice_conversion.model_manager import ModelManager, RVCInferParams
                
                mm = ModelManager(model_dir="assets/models", input_sample_rate=sample_rate)
                
                if not mm.load_model(checkpoint_path, None):
                    return self._fail_result(
                        gate_name,
                        f"Failed to load checkpoint: {checkpoint_path}",
                        remediation=["Checkpoint may be corrupted"]
                    )
                
                params = RVCInferParams(
                    f0_up_key=0,
                    f0_method="rmvpe",
                    index_rate=0.0,
                    resample_sr=0
                )
                
                output = mm.infer(test_audio, params)
                mm.unload_model()
                
                if output is None or len(output) == 0:
                    return self._fail_result(
                        gate_name,
                        "Inference returned empty output",
                        remediation=["Checkpoint may be broken"]
                    )
                
                output = output.astype(np.float32)
                output_sr = 48000
                
            except Exception as e:
                return self._fail_result(
                    gate_name,
                    f"Inference failed: {e}",
                    remediation=["Check checkpoint integrity and dependencies"]
                )
            
            # Analyze output on active frames
            window_size = int(output_sr * 0.025)
            hop_size = int(output_sr * 0.010)
            rms_threshold = 0.01
            
            window_crest = []
            window_sf = []
            
            for i in range(0, len(output) - window_size, hop_size):
                window = output[i:i + window_size]
                w_rms = np.sqrt(np.mean(window ** 2))
                
                if w_rms < rms_threshold:
                    continue
                
                # Crest factor
                w_peak = np.abs(window).max()
                window_crest.append(w_peak / w_rms if w_rms > 0.001 else 0)
                
                # Spectral flatness
                fft = np.abs(np.fft.rfft(window))
                fft = fft[fft > 0]
                if len(fft) > 0:
                    sf = np.exp(np.mean(np.log(fft))) / np.mean(fft)
                    window_sf.append(sf)
            
            # Calculate metrics
            dc_offset = float(np.mean(output))
            
            if len(window_crest) >= 10:
                crest_factor = float(np.median(window_crest))
                spectral_flatness = float(np.median(window_sf)) if window_sf else 1.0
            else:
                # Fallback to global metrics
                rms = float(np.sqrt(np.mean(output ** 2)))
                peak = float(np.abs(output).max())
                crest_factor = peak / rms if rms > 0.001 else 0
                fft = np.abs(np.fft.rfft(output))
                fft = fft[fft > 0]
                spectral_flatness = float(np.exp(np.mean(np.log(fft))) / np.mean(fft)) if len(fft) > 0 else 1.0
                warnings.append(f"Only {len(window_crest)} active frames - using global metrics")
            
            metrics['dc_offset'] = dc_offset
            metrics['crest_factor'] = crest_factor
            metrics['spectral_flatness'] = spectral_flatness
            metrics['active_frames'] = len(window_crest)
            
            # Validate
            if abs(dc_offset) > max_dc_offset:
                issues.append(f"High DC offset: {dc_offset:.4f} (max: ±{max_dc_offset})")
                remediation.append("Generator learned a DC bias - training collapsed")
            
            if crest_factor < min_crest_factor:
                issues.append(f"Low crest factor: {crest_factor:.2f} (min: {min_crest_factor})")
                remediation.append("Output is compressed/collapsed - not speech-like")
            
            if spectral_flatness > max_spectral_flatness:
                issues.append(f"High spectral flatness: {spectral_flatness:.3f} (max: {max_spectral_flatness})")
                remediation.append("Output is noise-like - generator collapsed")
            
            # Save debug audio on failure
            if issues and self.save_debug_artifacts:
                self._save_smoke_test_audio(output, output_sr, checkpoint_path)
            
            # Determine result
            if issues:
                result = self._fail_result(
                    gate_name,
                    f"Smoke test FAILED: Output shows collapse signature",
                    metrics=metrics,
                    issues=issues,
                    warnings=warnings,
                    remediation=remediation,
                )
                self._request_abort("Smoke test failed - model output collapsed")
            else:
                result = WatchdogResult(
                    gate_name=gate_name,
                    status=WatchdogStatus.PASSED,
                    passed=True,
                    message=f"Smoke test PASSED: crest={crest_factor:.2f}, DC={dc_offset:.4f}",
                    metrics=metrics,
                    warnings=warnings,
                )
                self._smoke_test_passed = True
                
        except Exception as e:
            import traceback
            result = self._fail_result(
                gate_name,
                f"Smoke test error: {e}",
                metrics={'error_traceback': traceback.format_exc()},
                remediation=["Check logs for details"]
            )
        
        self.results[gate_name] = result
        self._save_result(result)
        return result
    
    def _save_smoke_test_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        checkpoint_path: str,
    ):
        """Save smoke test audio for debugging"""
        try:
            from scipy.io import wavfile
            
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_name = Path(checkpoint_path).stem
            output_path = self.debug_dir / f"smoke_test_{checkpoint_name}.wav"
            
            # Convert to int16
            audio_int = (audio * 32767).astype(np.int16)
            wavfile.write(str(output_path), sample_rate, audio_int)
            
            logger.info(f"Saved smoke test audio: {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not save smoke test audio: {e}")
    
    # =========================================================================
    # ABORT MANAGEMENT
    # =========================================================================
    
    def _request_abort(self, reason: str):
        """Request training abort"""
        self._abort_requested = True
        self._abort_reason = reason
        logger.warning(f"WATCHDOG ABORT REQUESTED: {reason}")
    
    def should_abort(self) -> bool:
        """Check if training should be aborted"""
        return self._abort_requested
    
    def get_abort_reason(self) -> str:
        """Get reason for abort request"""
        return self._abort_reason
    
    def is_ready_for_model(self) -> bool:
        """
        Check if model meets READY requirements.
        
        A model is NOT ready unless:
        - Preprocess gate PASSED
        - F0 gate PASSED  
        - At least one smoke test PASSED
        """
        return (
            self._preprocess_passed and
            self._f0_passed and
            self._smoke_test_passed
        )
    
    def get_ready_status(self) -> Dict[str, bool]:
        """Get status of each ready requirement"""
        return {
            'preprocess_passed': self._preprocess_passed,
            'f0_passed': self._f0_passed,
            'smoke_test_passed': self._smoke_test_passed,
            'is_ready': self.is_ready_for_model(),
        }
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _fail_result(
        self,
        gate_name: str,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        issues: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        remediation: Optional[List[str]] = None,
    ) -> WatchdogResult:
        """Create a failed watchdog result"""
        return WatchdogResult(
            gate_name=gate_name,
            status=WatchdogStatus.FAILED,
            passed=False,
            message=message,
            metrics=metrics or {},
            issues=issues or [message],
            warnings=warnings or [],
            remediation=remediation or [],
        )
    
    def _save_result(self, result: WatchdogResult):
        """Save watchdog result to debug directory"""
        if not self.save_debug_artifacts:
            return
        
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
            result_path = self.debug_dir / f"{result.gate_name}_result.json"
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Could not save watchdog result: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all watchdog results"""
        return {
            'job_id': self.job_id,
            'abort_requested': self._abort_requested,
            'abort_reason': self._abort_reason,
            'ready_status': self.get_ready_status(),
            'results': {
                name: result.to_dict()
                for name, result in self.results.items()
            },
            'loss_history': {
                'total_steps': len(self.loss_history.metrics),
                'nan_count': self.loss_history.nan_count,
                'last_mel': self.loss_history.mel_values[-1] if self.loss_history.mel_values else None,
                'last_kl': self.loss_history.kl_values[-1] if self.loss_history.kl_values else None,
            },
        }
    
    def generate_failure_report(self) -> str:
        """Generate human-readable failure report"""
        lines = []
        lines.append("=" * 60)
        lines.append("WATCHDOG FAILURE REPORT")
        lines.append("=" * 60)
        lines.append(f"Job ID: {self.job_id}")
        lines.append(f"Abort Requested: {self._abort_requested}")
        if self._abort_reason:
            lines.append(f"Abort Reason: {self._abort_reason}")
        lines.append("")
        
        for name, result in self.results.items():
            lines.append("-" * 60)
            lines.append(f"GATE: {name.upper()}")
            lines.append(f"Status: {result.status.value}")
            lines.append(f"Passed: {result.passed}")
            lines.append(f"Message: {result.message}")
            
            if result.issues:
                lines.append("Issues:")
                for issue in result.issues:
                    lines.append(f"  ❌ {issue}")
            
            if result.warnings:
                lines.append("Warnings:")
                for warning in result.warnings:
                    lines.append(f"  ⚠️  {warning}")
            
            if result.remediation:
                lines.append("Remediation:")
                for rem in result.remediation:
                    lines.append(f"  → {rem}")
            
            lines.append("")
        
        lines.append("-" * 60)
        lines.append("READY STATUS")
        lines.append("-" * 60)
        ready = self.get_ready_status()
        lines.append(f"  Preprocess gate: {'✓' if ready['preprocess_passed'] else '✗'}")
        lines.append(f"  F0 gate: {'✓' if ready['f0_passed'] else '✗'}")
        lines.append(f"  Smoke test: {'✓' if ready['smoke_test_passed'] else '✗'}")
        lines.append(f"  MODEL READY: {'✓ YES' if ready['is_ready'] else '✗ NO'}")
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_watchdog_manager(
    job_id: str,
    model_dir: str,
    watchdog_configs: List[Dict[str, Any]],
) -> WatchdogManager:
    """
    Create a WatchdogManager from watchdog configuration list.
    
    Converts the WatchdogConfig format from training_plan to the
    thresholds dict format expected by WatchdogManager.
    """
    thresholds = {}
    
    for config in watchdog_configs:
        gate = config.get('gate', '')
        gate_thresholds = config.get('thresholds', {})
        thresholds[gate] = gate_thresholds
    
    return WatchdogManager(
        job_id=job_id,
        model_dir=model_dir,
        thresholds=thresholds,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python training_watchdogs.py <model_dir> <gate>")
        print("\nGates: preprocess, f0, smoke_test")
        print("\nExample:")
        print("  python training_watchdogs.py /path/to/model preprocess")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    gate = sys.argv[2]
    
    logging.basicConfig(level=logging.INFO)
    
    manager = WatchdogManager(
        job_id="test",
        model_dir=model_dir,
    )
    
    if gate == "preprocess":
        preproc_dir = Path(model_dir) / "0_gt_wavs"
        result = manager.run_preprocess_gate(str(preproc_dir))
    elif gate == "f0":
        f0_dir = Path(model_dir) / "2a_f0"
        result = manager.run_f0_gate(str(f0_dir))
    elif gate == "smoke_test":
        # Find a checkpoint
        checkpoints = list(Path(model_dir).glob("*.pth"))
        checkpoints = [c for c in checkpoints if not c.name.startswith('G_') and not c.name.startswith('D_')]
        if checkpoints:
            result = manager.run_smoke_test(str(checkpoints[0]))
        else:
            print("No checkpoint found for smoke test")
            sys.exit(1)
    else:
        print(f"Unknown gate: {gate}")
        sys.exit(1)
    
    print(json.dumps(result.to_dict(), indent=2, default=str))
    
    if not result.passed:
        print("\n" + manager.generate_failure_report())
        sys.exit(1)
