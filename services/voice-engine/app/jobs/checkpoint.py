"""
Checkpoint Management System

Provides safe checkpoint saving for training jobs with:
- Atomic writes (temp file + rename)
- Proper naming convention
- Request/save flow for "Save checkpoint & cancel"
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable
import re

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint"""
    path: str
    model_name: str
    version: str
    epoch: int
    step: int
    created_at: str
    size_bytes: int
    is_resumable: bool = True
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def filename(self) -> str:
        return Path(self.path).name


def generate_checkpoint_name(
    model_name: str,
    version: str,
    epoch: int,
    step: int,
    date: Optional[datetime] = None
) -> str:
    """
    Generate checkpoint filename using the naming convention:
    {model_name}-v{version}-e{epoch}-s{step}-{date}.pth
    
    Examples:
    - anton-v0.5-e100-s4300-20240115.pth
    - bjarni-v1.0-e200-s8600-20240116.pth
    """
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y%m%d")
    
    # Sanitize model name (remove spaces, special chars)
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '-', model_name.lower())
    safe_name = re.sub(r'-+', '-', safe_name).strip('-')
    
    return f"{safe_name}-v{version}-e{epoch}-s{step}-{date_str}.pth"


def parse_checkpoint_name(filename: str) -> Optional[Dict]:
    """
    Parse checkpoint filename to extract metadata.
    
    Returns dict with: model_name, version, epoch, step, date
    """
    # Match pattern: {name}-v{version}-e{epoch}-s{step}-{date}.pth
    pattern = r'^(.+)-v([\d.]+)-e(\d+)-s(\d+)-(\d{8})\.pth$'
    match = re.match(pattern, filename)
    
    if match:
        return {
            "model_name": match.group(1),
            "version": match.group(2),
            "epoch": int(match.group(3)),
            "step": int(match.group(4)),
            "date": match.group(5),
        }
    
    # Try legacy format: G_{step}.pth or D_{step}.pth
    legacy_pattern = r'^([GD])_(\d+)\.pth$'
    legacy_match = re.match(legacy_pattern, filename)
    
    if legacy_match:
        return {
            "type": legacy_match.group(1),
            "step": int(legacy_match.group(2)),
            "legacy": True,
        }
    
    return None


class CheckpointSaveRequest:
    """Request to save a checkpoint"""
    def __init__(self, job_id: str, callback: Optional[Callable] = None):
        self.job_id = job_id
        self.callback = callback
        self.created_at = datetime.utcnow()
        self.completed = threading.Event()
        self.success = False
        self.error: Optional[str] = None
        self.checkpoint_info: Optional[CheckpointInfo] = None
    
    def wait(self, timeout: float = 120.0) -> bool:
        """Wait for checkpoint save to complete"""
        return self.completed.wait(timeout)
    
    def complete_success(self, info: CheckpointInfo):
        """Mark request as completed successfully"""
        self.success = True
        self.checkpoint_info = info
        self.completed.set()
        if self.callback:
            self.callback(self, True)
    
    def complete_failure(self, error: str):
        """Mark request as failed"""
        self.success = False
        self.error = error
        self.completed.set()
        if self.callback:
            self.callback(self, False)


class CheckpointManager:
    """
    Manages checkpoint saving for training jobs.
    
    Features:
    - Atomic checkpoint writes
    - Request queue for "Save checkpoint & cancel"
    - Automatic cleanup of old checkpoints
    - Proper naming convention
    """
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self._pending_requests: Dict[str, CheckpointSaveRequest] = {}
        self._lock = threading.Lock()
    
    def request_checkpoint(
        self, 
        job_id: str, 
        callback: Optional[Callable] = None
    ) -> CheckpointSaveRequest:
        """
        Request a checkpoint save for a training job.
        
        The training loop should poll has_pending_request() and call
        save_checkpoint() when appropriate.
        """
        with self._lock:
            request = CheckpointSaveRequest(job_id, callback)
            self._pending_requests[job_id] = request
            logger.info(f"Checkpoint save requested for job {job_id}")
            return request
    
    def has_pending_request(self, job_id: str) -> bool:
        """Check if there's a pending checkpoint request"""
        with self._lock:
            return job_id in self._pending_requests
    
    def get_pending_request(self, job_id: str) -> Optional[CheckpointSaveRequest]:
        """Get pending checkpoint request"""
        with self._lock:
            return self._pending_requests.get(job_id)
    
    def clear_request(self, job_id: str):
        """Clear a pending request"""
        with self._lock:
            self._pending_requests.pop(job_id, None)
    
    def save_checkpoint(
        self,
        job_id: str,
        model_name: str,
        version: str,
        epoch: int,
        step: int,
        generator_state: dict,
        discriminator_state: Optional[dict] = None,
        optimizer_g_state: Optional[dict] = None,
        optimizer_d_state: Optional[dict] = None,
        training_config: Optional[dict] = None,
    ) -> CheckpointInfo:
        """
        Save a checkpoint atomically.
        
        This saves both generator and discriminator states in a single
        checkpoint file that can be used to resume training.
        
        Args:
            job_id: Training job ID
            model_name: Name of the model being trained
            version: Version string (e.g., "0.5")
            epoch: Current epoch number
            step: Current global step number
            generator_state: Generator model state dict
            discriminator_state: Optional discriminator state dict
            optimizer_g_state: Optional generator optimizer state
            optimizer_d_state: Optional discriminator optimizer state
            training_config: Optional training configuration
        
        Returns:
            CheckpointInfo with saved checkpoint details
        """
        import torch
        
        # Generate filename
        filename = generate_checkpoint_name(model_name, version, epoch, step)
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = model_dir / filename
        temp_path = model_dir / f".{filename}.tmp"
        
        logger.info(f"Saving checkpoint: {checkpoint_path}")
        
        try:
            # Build checkpoint dict
            checkpoint = {
                "generator": generator_state,
                "epoch": epoch,
                "step": step,
                "model_name": model_name,
                "version": version,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            
            if discriminator_state is not None:
                checkpoint["discriminator"] = discriminator_state
            
            if optimizer_g_state is not None:
                checkpoint["optimizer_g"] = optimizer_g_state
            
            if optimizer_d_state is not None:
                checkpoint["optimizer_d"] = optimizer_d_state
            
            if training_config is not None:
                checkpoint["training_config"] = training_config
            
            # Save to temp file first
            torch.save(checkpoint, str(temp_path))
            
            # Verify the temp file
            if not temp_path.exists():
                raise RuntimeError("Temp checkpoint file not created")
            
            # Verify it can be loaded
            test_load = torch.load(str(temp_path), map_location="cpu")
            if "generator" not in test_load:
                raise RuntimeError("Checkpoint verification failed")
            del test_load
            
            # Atomic rename
            temp_path.rename(checkpoint_path)
            
            # Get file size
            size_bytes = checkpoint_path.stat().st_size
            
            # Create checkpoint info
            info = CheckpointInfo(
                path=str(checkpoint_path),
                model_name=model_name,
                version=version,
                epoch=epoch,
                step=step,
                created_at=checkpoint["created_at"],
                size_bytes=size_bytes,
                is_resumable=True,
            )
            
            logger.info(f"Checkpoint saved successfully: {filename} ({size_bytes / 1024 / 1024:.1f} MB)")
            
            # Complete pending request if any
            with self._lock:
                request = self._pending_requests.pop(job_id, None)
                if request:
                    request.complete_success(info)
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
            # Clean up temp file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            
            # Complete pending request with failure
            with self._lock:
                request = self._pending_requests.pop(job_id, None)
                if request:
                    request.complete_failure(str(e))
            
            raise
    
    def save_legacy_checkpoint(
        self,
        model_dir: Path,
        generator_state: dict,
        discriminator_state: dict,
        step: int,
    ):
        """
        Save checkpoint in legacy format (G_{step}.pth, D_{step}.pth)
        
        This maintains compatibility with existing RVC training code.
        """
        import torch
        
        g_path = model_dir / f"G_{step}.pth"
        d_path = model_dir / f"D_{step}.pth"
        
        g_temp = model_dir / f".G_{step}.pth.tmp"
        d_temp = model_dir / f".D_{step}.pth.tmp"
        
        try:
            # Save generator
            torch.save(generator_state, str(g_temp))
            g_temp.rename(g_path)
            
            # Save discriminator
            torch.save(discriminator_state, str(d_temp))
            d_temp.rename(d_path)
            
            logger.debug(f"Saved legacy checkpoints: G_{step}.pth, D_{step}.pth")
            
        except Exception as e:
            # Clean up
            for p in [g_temp, d_temp]:
                if p.exists():
                    try:
                        p.unlink()
                    except:
                        pass
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """Load a checkpoint file"""
        import torch
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint
    
    def list_checkpoints(self, model_name: str) -> list[CheckpointInfo]:
        """List all checkpoints for a model"""
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return []
        
        checkpoints = []
        
        # Find new-format checkpoints
        for pth_file in model_dir.glob("*.pth"):
            parsed = parse_checkpoint_name(pth_file.name)
            if parsed and not parsed.get("legacy"):
                checkpoints.append(CheckpointInfo(
                    path=str(pth_file),
                    model_name=parsed["model_name"],
                    version=parsed["version"],
                    epoch=parsed["epoch"],
                    step=parsed["step"],
                    created_at=parsed["date"],
                    size_bytes=pth_file.stat().st_size,
                ))
        
        # Sort by step (most recent first)
        checkpoints.sort(key=lambda c: c.step, reverse=True)
        
        return checkpoints
    
    def get_latest_checkpoint(self, model_name: str) -> Optional[CheckpointInfo]:
        """Get the most recent checkpoint for a model"""
        checkpoints = self.list_checkpoints(model_name)
        return checkpoints[0] if checkpoints else None
    
    def cleanup_old_checkpoints(
        self,
        model_name: str,
        keep_count: int = 3,
        keep_milestones: bool = True,
        milestone_interval: int = 50,
    ) -> list[str]:
        """
        Clean up old checkpoints, keeping recent ones and milestones.
        
        Args:
            model_name: Model name
            keep_count: Number of recent checkpoints to keep
            keep_milestones: Whether to keep milestone checkpoints
            milestone_interval: Epoch interval for milestones (e.g., 50, 100, 150)
        
        Returns:
            List of deleted file paths
        """
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return []
        
        deleted = []
        
        # Handle legacy checkpoints (G_*.pth, D_*.pth)
        g_files = sorted(model_dir.glob("G_*.pth"), key=lambda p: self._extract_step(p))
        d_files = sorted(model_dir.glob("D_*.pth"), key=lambda p: self._extract_step(p))
        
        # Keep recent and milestones
        for files in [g_files, d_files]:
            if len(files) <= keep_count:
                continue
            
            for f in files[:-keep_count]:
                step = self._extract_step(f)
                
                # Check if milestone (roughly every milestone_interval epochs)
                # Assuming ~30 steps per epoch
                approx_epoch = step // 30
                is_milestone = keep_milestones and (approx_epoch % milestone_interval == 0)
                
                if not is_milestone:
                    try:
                        f.unlink()
                        deleted.append(str(f))
                        logger.debug(f"Deleted old checkpoint: {f.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {f}: {e}")
        
        # Handle new-format checkpoints
        checkpoints = self.list_checkpoints(model_name)
        if len(checkpoints) > keep_count:
            for ckpt in checkpoints[keep_count:]:
                is_milestone = keep_milestones and (ckpt.epoch % milestone_interval == 0)
                
                if not is_milestone:
                    try:
                        Path(ckpt.path).unlink()
                        deleted.append(ckpt.path)
                        logger.debug(f"Deleted old checkpoint: {ckpt.filename}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {ckpt.path}: {e}")
        
        if deleted:
            logger.info(f"Cleaned up {len(deleted)} old checkpoints for {model_name}")
        
        return deleted
    
    def _extract_step(self, path: Path) -> int:
        """Extract step number from legacy checkpoint filename"""
        match = re.search(r'[GD]_(\d+)\.pth', path.name)
        return int(match.group(1)) if match else 0


# Global checkpoint manager instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(models_dir: Optional[str] = None) -> CheckpointManager:
    """Get or create the global checkpoint manager"""
    global _checkpoint_manager
    
    if _checkpoint_manager is None:
        if models_dir is None:
            # Default path
            models_dir = str(Path(__file__).parent.parent.parent / "assets" / "models")
        _checkpoint_manager = CheckpointManager(models_dir)
    
    return _checkpoint_manager


def init_checkpoint_manager(models_dir: str):
    """Initialize the global checkpoint manager"""
    global _checkpoint_manager
    _checkpoint_manager = CheckpointManager(models_dir)
