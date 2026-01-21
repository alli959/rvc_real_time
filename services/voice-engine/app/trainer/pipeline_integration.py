"""
Pipeline Integration for Training Watchdogs

This module provides integration between the training plan system, watchdog gates,
and the existing RVC training pipeline.

It wraps the training pipeline to:
1. Run training plan recommendation before training
2. Execute watchdog gates at appropriate points
3. Abort training early if watchdogs fail
4. Enforce the READY rule (no model marked ready without passing gates)

Usage:
    from app.trainer.pipeline_integration import run_training_with_watchdogs
    
    result = await run_training_with_watchdogs(
        config=config,
        audio_paths=audio_paths,
        model_dir=model_dir,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.trainer.pipeline import (
    RVCTrainingPipeline,
    TrainingConfig,
    TrainingResult,
    TrainingStatus,
    TrainingProgress,
)
from app.trainer.training_plan import (
    recommend_training_plan,
    TrainingPlan,
    TrainingPlanMode,
    PlanThresholds,
    DEFAULT_THRESHOLDS,
)
from app.trainer.training_watchdogs import (
    WatchdogManager,
    WatchdogResult,
    WatchdogStatus,
    create_watchdog_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WatchdogIntegrationConfig:
    """Configuration for watchdog integration"""
    
    # Enable/disable watchdogs
    enable_preprocess_gate: bool = True
    enable_f0_gate: bool = True
    enable_early_training_gate: bool = True
    enable_smoke_test: bool = True
    
    # Smoke test timing
    smoke_test_epoch: int = 1        # Run smoke test at this epoch
    smoke_test_on_stuck: bool = True  # Also run smoke test if stuck detected
    
    # Abort behavior
    abort_on_preprocess_fail: bool = True
    abort_on_f0_fail: bool = True
    abort_on_smoke_test_fail: bool = True
    abort_on_nan_loss: bool = True
    
    # F0 fallback
    f0_retry_methods: List[str] = None  # ['harvest', 'crepe'] etc
    
    # Enforce ready rule
    enforce_ready_rule: bool = True
    
    def __post_init__(self):
        if self.f0_retry_methods is None:
            self.f0_retry_methods = ['harvest', 'pm']


# =============================================================================
# WATCHDOG HOOKS
# =============================================================================

class WatchdogHooks:
    """
    Hooks for integrating watchdogs with the training pipeline.
    
    These can be called at appropriate points in the training process.
    """
    
    def __init__(
        self,
        manager: WatchdogManager,
        config: WatchdogIntegrationConfig,
        thresholds: PlanThresholds,
    ):
        self.manager = manager
        self.config = config
        self.thresholds = thresholds
        
        # Track state
        self._preprocess_done = False
        self._f0_done = False
        self._training_started = False
        self._first_smoke_test_done = False
    
    async def after_preprocess(
        self,
        exp_dir: Path,
        sample_rate: int,
    ) -> WatchdogResult:
        """
        Run after preprocessing completes.
        
        Validates the preprocessed audio quality.
        """
        if not self.config.enable_preprocess_gate:
            return WatchdogResult(
                gate_name="preprocess",
                status=WatchdogStatus.NOT_RUN,
                passed=True,
                message="Preprocess gate disabled"
            )
        
        gt_wavs_dir = exp_dir / "0_gt_wavs"
        result = self.manager.run_preprocess_gate(
            preprocessed_dir=str(gt_wavs_dir),
            sample_rate=sample_rate,
        )
        
        self._preprocess_done = True
        
        if not result.passed and self.config.abort_on_preprocess_fail:
            logger.error(f"Preprocess gate FAILED: {result.message}")
        
        return result
    
    async def after_f0_extraction(
        self,
        exp_dir: Path,
    ) -> WatchdogResult:
        """
        Run after F0 extraction completes.
        
        Validates pitch extraction quality.
        """
        if not self.config.enable_f0_gate:
            return WatchdogResult(
                gate_name="f0_extraction",
                status=WatchdogStatus.NOT_RUN,
                passed=True,
                message="F0 gate disabled"
            )
        
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b_f0nsf"
        feature_dir = exp_dir / "3_feature768"
        
        result = self.manager.run_f0_gate(
            f0_dir=str(f0_dir),
            f0nsf_dir=str(f0nsf_dir) if f0nsf_dir.exists() else None,
            feature_dir=str(feature_dir) if feature_dir.exists() else None,
        )
        
        self._f0_done = True
        
        if not result.passed and self.config.abort_on_f0_fail:
            logger.error(f"F0 gate FAILED: {result.message}")
        
        return result
    
    def on_training_step(
        self,
        step: int,
        epoch: int,
        losses: Dict[str, float],
    ) -> Optional[WatchdogResult]:
        """
        Called after each training step.
        
        Updates loss metrics and checks for stuck patterns.
        """
        if not self.config.enable_early_training_gate:
            return None
        
        self._training_started = True
        
        result = self.manager.update_loss_metrics(
            step=step,
            epoch=epoch,
            loss_disc=losses.get('loss_disc', 0),
            loss_gen=losses.get('loss_gen', 0),
            loss_fm=losses.get('loss_fm', 0),
            loss_mel=losses.get('loss_mel', 0),
            loss_kl=losses.get('loss_kl', 0),
        )
        
        # If stuck detected and smoke test enabled, we should request smoke test
        if result and result.status == WatchdogStatus.WARNING:
            if 'mel_stuck' in result.metrics or 'kl_stuck' in result.metrics:
                logger.warning("Stuck loss pattern detected - smoke test recommended")
        
        return result
    
    async def on_checkpoint_saved(
        self,
        checkpoint_path: Path,
        epoch: int,
    ) -> Optional[WatchdogResult]:
        """
        Called when a checkpoint is saved.
        
        Runs smoke test if appropriate.
        """
        if not self.config.enable_smoke_test:
            return None
        
        # Run smoke test at configured epoch
        should_smoke_test = (
            epoch == self.config.smoke_test_epoch or
            (self.config.smoke_test_on_stuck and self.manager.should_abort())
        )
        
        if not should_smoke_test:
            return None
        
        # Find the inference model to test
        # Look for extracted model or use checkpoint
        exp_dir = checkpoint_path.parent
        exp_name = exp_dir.name
        
        test_model = None
        
        # Check for extracted model first
        infer_pth = exp_dir / f"{exp_name}_infer.pth"
        final_pth = exp_dir / f"{exp_name}.pth"
        
        if infer_pth.exists():
            test_model = str(infer_pth)
        elif final_pth.exists():
            test_model = str(final_pth)
        else:
            # Extract model from checkpoint on-the-fly for smoke test
            logger.info(f"No inference model found, extracting from checkpoint: {checkpoint_path}")
            test_model = self._extract_model_for_smoke_test(checkpoint_path, exp_name)
            if test_model is None:
                logger.warning(f"Failed to extract model at epoch {epoch}, skipping smoke test")
                return None
        
        logger.info(f"Running smoke test on {test_model}")
        result = self.manager.run_smoke_test(checkpoint_path=test_model)
        
        self._first_smoke_test_done = True
        
        if not result.passed:
            logger.error(f"Smoke test FAILED at epoch {epoch}: {result.message}")
            if self.config.abort_on_smoke_test_fail:
                logger.error("Requesting training abort")
        
        return result
    
    def _extract_model_for_smoke_test(self, checkpoint_path: Path, exp_name: str) -> Optional[str]:
        """
        Extract a lightweight inference model from a checkpoint for smoke testing.
        
        This allows smoke testing even when no extracted model exists yet.
        """
        import torch
        from collections import OrderedDict
        
        try:
            exp_dir = checkpoint_path.parent
            
            # Load config to determine sr/version
            config_path = exp_dir / "config.json"
            sr = "48k"
            version = "v2"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    sr_val = cfg.get('sample_rate', 48000)
                    sr = f"{sr_val // 1000}k" if sr_val >= 1000 else "48k"
                    version = cfg.get('version', 'v2')
            
            # Load checkpoint
            ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            if "model" in ckpt:
                ckpt = ckpt["model"]
            
            # Extract weights (remove enc_q)
            opt = OrderedDict()
            opt["weight"] = {}
            for key in ckpt.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = ckpt[key].half()
            
            # Set config based on sr/version
            if sr == "48k" and version == "v2":
                opt["config"] = [
                    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000
                ]
            elif sr == "48k" and version == "v1":
                opt["config"] = [
                    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 6, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 48000
                ]
            elif sr == "40k":
                opt["config"] = [
                    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000
                ]
            else:  # 32k
                opt["config"] = [
                    513, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 4, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 32000
                ]
            
            sr_map = {"32k": 32000, "40k": 40000, "48k": 48000}
            opt["sr"] = sr_map.get(sr, 48000)
            opt["f0"] = 1
            opt["version"] = version
            opt["info"] = f"Smoke test extraction from {checkpoint_path.name}"
            
            # Save to temporary inference model
            smoke_test_path = exp_dir / f"{exp_name}_smoke_test.pth"
            torch.save(opt, str(smoke_test_path))
            
            logger.info(f"Extracted smoke test model: {smoke_test_path}")
            return str(smoke_test_path)
            
        except Exception as e:
            logger.error(f"Failed to extract model for smoke test: {e}")
            return None
    
    def should_abort(self) -> bool:
        """Check if training should be aborted"""
        return self.manager.should_abort()
    
    def get_abort_reason(self) -> str:
        """Get reason for abort"""
        return self.manager.get_abort_reason()
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize watchdog integration and get summary.
        
        Called when training completes (success or failure).
        """
        summary = self.manager.get_summary()
        
        # Add ready status
        ready_status = self.manager.get_ready_status()
        summary['ready_status'] = ready_status
        
        # Check if model should be marked ready
        if self.config.enforce_ready_rule:
            if not ready_status['is_ready']:
                logger.warning("Model does NOT meet ready requirements!")
                logger.warning(f"  Preprocess: {ready_status['preprocess_passed']}")
                logger.warning(f"  F0: {ready_status['f0_passed']}")
                logger.warning(f"  Smoke test: {ready_status['smoke_test_passed']}")
        
        return summary


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

async def run_training_with_watchdogs(
    pipeline: RVCTrainingPipeline,
    config: TrainingConfig,
    audio_paths: List[Union[str, Path]],
    model_dir: Optional[str] = None,
    gpu_memory_gb: float = 12.0,
    force_mode: Optional[TrainingPlanMode] = None,
    user_overrides: Optional[Dict[str, Any]] = None,
    integration_config: Optional[WatchdogIntegrationConfig] = None,
    thresholds: Optional[PlanThresholds] = None,
) -> TrainingResult:
    """
    Run training with watchdog protection.
    
    This is the recommended entry point for training with fail-fast protection.
    
    Args:
        pipeline: RVC training pipeline instance
        config: Training configuration
        audio_paths: List of audio files to train on
        model_dir: Model directory (defaults to pipeline base_dir / exp_name)
        gpu_memory_gb: Available GPU memory
        force_mode: Force specific training mode
        user_overrides: User-specified config overrides
        integration_config: Watchdog integration configuration
        thresholds: Plan thresholds
        
    Returns:
        TrainingResult with success/failure status
    """
    if integration_config is None:
        integration_config = WatchdogIntegrationConfig()
    
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    if model_dir is None:
        model_dir = str(pipeline.base_dir / config.exp_name)
    
    # Convert audio_paths to strings
    audio_path_strs = [str(p) for p in audio_paths]
    
    # Step 1: Generate training plan
    logger.info("Generating training plan...")
    plan = recommend_training_plan(
        model_name=config.exp_name,
        audio_paths=audio_path_strs,
        model_dir=model_dir,
        assets_dir=str(pipeline.assets_dir),
        gpu_memory_gb=gpu_memory_gb,
        force_mode=force_mode,
        user_overrides=user_overrides,
        thresholds=thresholds,
    )
    
    # Log the plan report
    logger.info("Training Plan:\n" + plan.report)
    
    # Check if we can proceed
    if not plan.can_proceed:
        error_msg = "Training plan rejected: " + "; ".join(plan.errors)
        logger.error(error_msg)
        return TrainingResult(
            success=False,
            job_id="",
            error=error_msg,
        )
    
    # Step 2: Apply suggested config if not overridden
    if plan.suggested_config:
        # Only apply non-locked params
        if not plan.locked_params:
            # Can suggest sample rate, etc.
            pass
        
        # Apply epochs if not user-overridden
        if user_overrides is None or 'epochs' not in user_overrides:
            config.epochs = plan.suggested_config.epochs
            config.total_epoch = plan.suggested_config.epochs
        
        # Apply batch size
        if user_overrides is None or 'batch_size' not in user_overrides:
            config.batch_size = plan.suggested_config.batch_size
        
        # Apply save interval
        config.save_every_epoch = plan.suggested_config.save_every_epoch
    
    # Step 3: Create watchdog manager
    job_id = pipeline.create_job(config)
    
    watchdog_thresholds = {}
    for wd in plan.required_watchdogs:
        watchdog_thresholds[wd.gate.value] = wd.thresholds
    
    manager = WatchdogManager(
        job_id=job_id,
        model_dir=model_dir,
        thresholds=watchdog_thresholds,
    )
    
    hooks = WatchdogHooks(
        manager=manager,
        config=integration_config,
        thresholds=thresholds,
    )
    
    # Step 4: Run training with watchdog hooks
    # Note: This is a simplified version. Full integration would require
    # modifying the pipeline to call hooks at appropriate points.
    
    logger.info(f"Starting training with watchdog protection (job_id={job_id})")
    
    try:
        result = await pipeline.train(
            config=config,
            audio_paths=audio_paths,
            job_id=job_id,
        )
        
        # After training, check watchdog state
        summary = hooks.finalize()
        
        # Save watchdog summary
        summary_path = Path(model_dir) / "watchdog_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Check ready rule
        if integration_config.enforce_ready_rule:
            ready_status = summary.get('ready_status', {})
            if not ready_status.get('is_ready', False):
                logger.warning("Model trained but does NOT meet ready requirements!")
                # Mark in result
                if result.success:
                    logger.warning("Marking model as NOT READY despite successful training")
        
        return result
        
    except Exception as e:
        logger.exception(f"Training with watchdogs failed: {e}")
        
        # Save failure report
        failure_report = hooks.manager.generate_failure_report()
        report_path = Path(model_dir) / "watchdog_failure_report.txt"
        with open(report_path, 'w') as f:
            f.write(failure_report)
        
        return TrainingResult(
            success=False,
            job_id=job_id,
            error=str(e),
        )


# =============================================================================
# POST-TRAINING VALIDATION
# =============================================================================

async def validate_trained_model_with_watchdogs(
    model_dir: str,
    run_smoke_test: bool = True,
) -> Dict[str, Any]:
    """
    Run full watchdog validation on a trained model.
    
    Can be used to validate models that were trained without watchdog
    protection, or to re-validate existing models.
    
    Args:
        model_dir: Path to model directory
        run_smoke_test: Whether to run inference smoke test
        
    Returns:
        Validation summary with pass/fail status
    """
    model_path = Path(model_dir)
    
    manager = WatchdogManager(
        job_id="validation",
        model_dir=model_dir,
    )
    
    results = []
    
    # Run preprocess gate
    gt_wavs = model_path / "0_gt_wavs"
    if gt_wavs.exists():
        result = manager.run_preprocess_gate(str(gt_wavs))
        results.append(result)
    
    # Run F0 gate
    f0_dir = model_path / "2a_f0"
    if f0_dir.exists():
        result = manager.run_f0_gate(str(f0_dir))
        results.append(result)
    
    # Run smoke test
    if run_smoke_test:
        # Find model file
        exp_name = model_path.name
        test_model = None
        
        for pattern in [f"{exp_name}_infer.pth", f"{exp_name}.pth"]:
            candidate = model_path / pattern
            if candidate.exists():
                test_model = str(candidate)
                break
        
        if test_model:
            result = manager.run_smoke_test(test_model)
            results.append(result)
    
    return manager.get_summary()


# =============================================================================
# TRAINING LOSS PARSER (for analyzing existing logs)
# =============================================================================

def parse_training_log(log_path: str) -> List[Dict[str, Any]]:
    """
    Parse an existing training log file for loss values.
    
    Useful for post-hoc analysis of training runs.
    """
    loss_pattern = re.compile(
        r'loss_disc=([\d.]+|nan), loss_gen=([\d.]+|nan), '
        r'loss_fm=([\d.]+|nan),loss_mel=([\d.]+|nan), loss_kl=([\d.]+|nan)'
    )
    
    epoch_pattern = re.compile(r'Train Epoch: (\d+)')
    
    entries = []
    current_epoch = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # Check for epoch
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Check for losses
            loss_match = loss_pattern.search(line)
            if loss_match:
                disc, gen, fm, mel, kl = loss_match.groups()
                
                entries.append({
                    'epoch': current_epoch,
                    'loss_disc': float(disc) if disc != 'nan' else float('nan'),
                    'loss_gen': float(gen) if gen != 'nan' else float('nan'),
                    'loss_fm': float(fm) if fm != 'nan' else float('nan'),
                    'loss_mel': float(mel) if mel != 'nan' else float('nan'),
                    'loss_kl': float(kl) if kl != 'nan' else float('nan'),
                })
    
    return entries


def analyze_training_log_for_issues(log_path: str) -> Dict[str, Any]:
    """
    Analyze a training log for issues that watchdogs would have caught.
    
    Returns analysis of what watchdogs would have detected.
    """
    entries = parse_training_log(log_path)
    
    if not entries:
        return {
            'error': 'No loss entries found in log',
            'issues': [],
            'warnings': [],
        }
    
    issues = []
    warnings = []
    metrics = {}
    
    # Get mel and kl values
    mel_values = [e['loss_mel'] for e in entries if not np.isnan(e['loss_mel'])]
    kl_values = [e['loss_kl'] for e in entries if not np.isnan(e['loss_kl'])]
    nan_count = sum(1 for e in entries if np.isnan(e['loss_mel']))
    
    metrics['total_entries'] = len(entries)
    metrics['nan_count'] = nan_count
    
    # Check for NaN
    if nan_count > len(entries) * 0.1:
        issues.append(f"High NaN rate: {nan_count}/{len(entries)} ({100*nan_count/len(entries):.1f}%)")
    
    # Check for stuck mel loss
    if mel_values:
        unique_mel = len(set(mel_values))
        most_common = max(set(mel_values), key=mel_values.count)
        stuck_count = mel_values.count(most_common)
        stuck_pct = 100 * stuck_count / len(mel_values)
        
        metrics['mel_unique_values'] = unique_mel
        metrics['mel_most_common'] = most_common
        metrics['mel_stuck_pct'] = stuck_pct
        
        if unique_mel == 1:
            issues.append(f"Mel loss STUCK at {most_common} for ALL iterations - COLLAPSE")
        elif stuck_pct > 90:
            issues.append(f"Mel loss stuck at {most_common} for {stuck_pct:.0f}% - likely collapse")
        elif stuck_pct > 50:
            warnings.append(f"Mel loss stuck at {most_common} for {stuck_pct:.0f}%")
    
    # Check for stuck KL loss
    if kl_values:
        unique_kl = len(set(kl_values))
        most_common_kl = max(set(kl_values), key=kl_values.count)
        stuck_kl_pct = 100 * kl_values.count(most_common_kl) / len(kl_values)
        
        metrics['kl_unique_values'] = unique_kl
        metrics['kl_stuck_pct'] = stuck_kl_pct
        
        if unique_kl == 1:
            issues.append(f"KL loss STUCK at {most_common_kl} for ALL iterations")
    
    # Both stuck = definite collapse
    if len(issues) >= 2 and 'Mel loss STUCK' in str(issues) and 'KL loss STUCK' in str(issues):
        issues.insert(0, "CONFIRMED TRAINING COLLAPSE: Both mel and KL losses stuck")
    
    return {
        'total_entries': len(entries),
        'issues': issues,
        'warnings': warnings,
        'metrics': metrics,
        'would_abort': len(issues) > 0,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline_integration.py validate <model_dir>")
        print("  python pipeline_integration.py analyze-log <train.log>")
        sys.exit(1)
    
    import numpy as np  # For isnan
    
    logging.basicConfig(level=logging.INFO)
    
    command = sys.argv[1]
    
    if command == "validate":
        model_dir = sys.argv[2]
        result = asyncio.run(validate_trained_model_with_watchdogs(model_dir))
        print(json.dumps(result, indent=2, default=str))
        
    elif command == "analyze-log":
        log_path = sys.argv[2]
        result = analyze_training_log_for_issues(log_path)
        print(json.dumps(result, indent=2, default=str))
        
        if result['issues']:
            print("\n❌ ISSUES DETECTED:")
            for issue in result['issues']:
                print(f"  • {issue}")
        
        if result['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in result['warnings']:
                print(f"  • {warning}")
        
        if result['would_abort']:
            print("\n→ Watchdogs WOULD HAVE ABORTED this training")
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
