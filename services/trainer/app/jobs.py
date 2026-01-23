"""
Trainer Service - Training Job Management
Handles training job queue, progress tracking, and subprocess management
"""

import asyncio
import logging
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    """Training job status"""
    PENDING = "pending"
    VALIDATING = "validating"  # Validating preprocessing outputs
    GENERATING_FILELIST = "generating_filelist"
    TRAINING = "training"
    EXTRACTING_MODEL = "extracting_model"
    BUILDING_INDEX = "building_index"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob:
    """Training job state"""
    
    def __init__(
        self,
        job_id: str,
        exp_name: str,
        config: Dict[str, Any]
    ):
        self.job_id = job_id
        self.exp_name = exp_name
        self.config = config
        self.status = TrainingStatus.PENDING
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = config.get("epochs", 100)
        self.current_step = 0
        self.total_steps = 0
        self.message = ""
        self.error: Optional[str] = None
        self.started_at = datetime.utcnow().isoformat() + "Z"
        self.completed_at: Optional[str] = None
        self.logs: List[str] = []
        self.result: Optional[Dict[str, Any]] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        self._cancel_requested = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "exp_name": self.exp_name,
            "status": self.status.value,
            "progress": round(self.progress * 100, 1),
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "logs": self.logs[-50:],  # Last 50 log entries
            "result": self.result,
        }
    
    def log(self, message: str):
        """Add log entry"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.logs.append(f"[{timestamp}] {message}")
        self.message = message


class TrainingJobManager:
    """Manages training jobs"""
    
    def __init__(self):
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
    
    def create_job(self, exp_name: str, config: Dict[str, Any]) -> TrainingJob:
        """Create a new training job"""
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(job_id, exp_name, config)
        
        with self._lock:
            self._jobs[job_id] = job
        
        return job
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    def get_active_job(self, exp_name: Optional[str] = None) -> Optional[TrainingJob]:
        """Get active job, optionally filtered by experiment name"""
        active_statuses = [
            TrainingStatus.PENDING,
            TrainingStatus.VALIDATING,
            TrainingStatus.GENERATING_FILELIST,
            TrainingStatus.TRAINING,
            TrainingStatus.EXTRACTING_MODEL,
            TrainingStatus.BUILDING_INDEX
        ]
        
        with self._lock:
            for job in self._jobs.values():
                if job.status in active_statuses:
                    if exp_name is None or job.exp_name == exp_name:
                        return job
        return None
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs"""
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]
    
    def cancel_job(self, job_id: str) -> bool:
        """Request job cancellation"""
        job = self.get_job(job_id)
        if job:
            job._cancel_requested = True
            if job.process:
                job.process.terminate()
            return True
        return False
    
    async def cleanup(self):
        """Clean up all jobs on shutdown"""
        for job in list(self._jobs.values()):
            if job.process:
                try:
                    job.process.terminate()
                    await asyncio.wait_for(job.process.wait(), timeout=5.0)
                except:
                    job.process.kill()


# Global job manager
job_manager = TrainingJobManager()


async def cleanup_jobs():
    """Clean up jobs on service shutdown"""
    await job_manager.cleanup()
