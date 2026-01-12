"""
Recording Wizard Module

Manages guided recording sessions for voice model training.
Features:
- Progressive prompt delivery
- Real-time recording validation
- Coverage tracking
- Session persistence
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import tempfile

import numpy as np

from app.analyzer import (
    AudioQualityAnalyzer,
    AudioQualityMetrics,
    PhonemeAnalyzer,
    PhonemeCoverageReport,
)
from app.prompts import get_prompt_loader, PromptLoader

logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Recording session status"""
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class RecordingValidation:
    """Validation result for a single recording"""
    is_valid: bool
    duration_seconds: float
    audio_quality: AudioQualityMetrics
    
    # Validation checks
    duration_ok: bool
    snr_ok: bool
    clipping_ok: bool
    silence_ok: bool
    
    # Issues (if any)
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "is_valid": bool(self.is_valid),
            "duration_seconds": round(float(self.duration_seconds), 2),
            "audio_quality": self.audio_quality.to_dict(),
            "checks": {
                "duration_ok": bool(self.duration_ok),
                "snr_ok": bool(self.snr_ok),
                "clipping_ok": bool(self.clipping_ok),
                "silence_ok": bool(self.silence_ok)
            },
            "issues": self.issues
        }


@dataclass
class RecordedPrompt:
    """A recorded prompt with metadata"""
    prompt_id: str
    prompt_text: str
    category: str
    language: str
    
    # Recording info
    audio_path: Optional[str] = None
    duration_seconds: float = 0.0
    recorded_at: Optional[str] = None
    
    # Validation
    validation: Optional[RecordingValidation] = None
    is_accepted: bool = False
    
    # Retry tracking
    attempt_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "category": self.category,
            "language": self.language,
            "audio_path": self.audio_path,
            "duration_seconds": float(round(self.duration_seconds, 2)),
            "recorded_at": self.recorded_at,
            "validation": self.validation.to_dict() if self.validation else None,
            "is_accepted": bool(self.is_accepted),
            "attempt_count": int(self.attempt_count)
        }


@dataclass
class WizardSession:
    """A recording wizard session"""
    session_id: str
    language: str
    exp_name: str
    status: SessionStatus = SessionStatus.CREATED
    
    # Timing
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Prompts
    total_prompts: int = 0
    prompts: List[RecordedPrompt] = field(default_factory=list)
    
    # Progress
    current_index: int = 0
    completed_count: int = 0
    skipped_count: int = 0
    
    # Target phonemes (if improving existing model)
    target_phonemes: Set[str] = field(default_factory=set)
    
    # Coverage tracking
    initial_coverage: Optional[PhonemeCoverageReport] = None
    current_coverage: Optional[PhonemeCoverageReport] = None
    
    # Output
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
    
    @property
    def progress_percentage(self) -> float:
        if self.total_prompts == 0:
            return 0.0
        return (self.completed_count / self.total_prompts) * 100
    
    @property
    def total_duration(self) -> float:
        return sum(p.duration_seconds for p in self.prompts if p.is_accepted)
    
    @property
    def current_prompt(self) -> Optional[RecordedPrompt]:
        if 0 <= self.current_index < len(self.prompts):
            return self.prompts[self.current_index]
        return None
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "language": self.language,
            "exp_name": self.exp_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_prompts": int(self.total_prompts),
            "current_index": int(self.current_index),
            "completed_count": int(self.completed_count),
            "skipped_count": int(self.skipped_count),
            "progress_percentage": float(round(self.progress_percentage, 1)),
            "total_duration_seconds": float(round(self.total_duration, 1)),
            "target_phonemes": list(self.target_phonemes),
            "output_dir": self.output_dir,
            "current_prompt": self.current_prompt.to_dict() if self.current_prompt else None
        }
    
    def save(self, path: Path):
        """Save session to JSON file"""
        data = self.to_dict()
        data["prompts"] = [p.to_dict() for p in self.prompts]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "WizardSession":
        """Load session from JSON file"""
        with open(path) as f:
            data = json.load(f)
        
        # Parse prompts
        prompts = []
        for p_data in data.pop("prompts", []):
            prompts.append(RecordedPrompt(**{
                k: v for k, v in p_data.items() 
                if k in RecordedPrompt.__dataclass_fields__
            }))
        
        # Remove computed fields
        data.pop("progress_percentage", None)
        data.pop("total_duration_seconds", None)
        data.pop("current_prompt", None)
        
        # Parse status
        data["status"] = SessionStatus(data["status"])
        
        session = cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })
        session.prompts = prompts
        session.target_phonemes = set(data.get("target_phonemes", []))
        
        return session


class RecordingWizard:
    """
    Manages guided recording sessions for voice model training.
    
    Usage:
        wizard = RecordingWizard(base_dir="/path/to/sessions")
        
        # Create new session
        session = wizard.create_session(
            language="en",
            exp_name="my_voice",
            prompt_count=50
        )
        
        # Get current prompt
        prompt = wizard.get_current_prompt(session.session_id)
        
        # Submit recording
        result = wizard.submit_recording(
            session_id=session.session_id,
            audio_data=audio_bytes,
            sample_rate=16000
        )
        
        # Move to next prompt
        wizard.next_prompt(session.session_id)
        
        # Complete session
        wizard.complete_session(session.session_id)
    """
    
    # Validation thresholds
    MIN_DURATION = 0.5  # seconds
    MAX_DURATION = 30.0  # seconds
    MIN_SNR = 15.0  # dB
    MAX_CLIPPING = 2.0  # percentage
    MAX_SILENCE = 70.0  # percentage
    
    def __init__(
        self,
        base_dir: Optional[str] = None,
        prompt_loader: Optional[PromptLoader] = None
    ):
        """
        Initialize the recording wizard.
        
        Args:
            base_dir: Directory to store session data
            prompt_loader: PromptLoader instance (uses default if None)
        """
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.gettempdir()) / "wizard_sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.prompt_loader = prompt_loader or get_prompt_loader()
        self.phoneme_analyzer = PhonemeAnalyzer()
        
        # Active sessions
        self._sessions: Dict[str, WizardSession] = {}
    
    def create_session(
        self,
        language: str,
        exp_name: str,
        prompt_count: int = 50,
        target_phonemes: Optional[Set[str]] = None,
        existing_coverage: Optional[PhonemeCoverageReport] = None
    ) -> WizardSession:
        """
        Create a new recording session.
        
        Args:
            language: Language code (en, is)
            exp_name: Experiment/model name
            prompt_count: Number of prompts for the session
            target_phonemes: Specific phonemes to target (for gap filling)
            existing_coverage: Existing coverage (for incremental training)
            
        Returns:
            New WizardSession
        """
        session_id = str(uuid.uuid4())[:8]
        
        # Create output directory
        output_dir = self.base_dir / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get prompts
        prompts_data = self.prompt_loader.get_wizard_session_prompts(
            language=language,
            session_length=prompt_count,
            target_phonemes=target_phonemes
        )
        
        # Create RecordedPrompt objects
        prompts = []
        for p_data in prompts_data:
            prompts.append(RecordedPrompt(
                prompt_id=f"{session_id}-{p_data['index']:03d}",
                prompt_text=p_data["prompt"],
                category=p_data["category"],
                language=language
            ))
        
        session = WizardSession(
            session_id=session_id,
            language=language,
            exp_name=exp_name,
            total_prompts=len(prompts),
            prompts=prompts,
            target_phonemes=target_phonemes or set(),
            initial_coverage=existing_coverage,
            output_dir=str(output_dir)
        )
        
        # Save session
        self._sessions[session_id] = session
        session.save(output_dir / "session.json")
        
        logger.info(f"Created session {session_id} with {len(prompts)} prompts")
        return session
    
    def get_session(self, session_id: str) -> Optional[WizardSession]:
        """Get a session by ID"""
        if session_id in self._sessions:
            return self._sessions[session_id]
        
        # Try loading from disk
        session_path = self.base_dir / session_id / "session.json"
        if session_path.exists():
            session = WizardSession.load(session_path)
            self._sessions[session_id] = session
            return session
        
        return None
    
    def start_session(self, session_id: str) -> Optional[WizardSession]:
        """Start a recording session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        if session.status not in [SessionStatus.CREATED, SessionStatus.PAUSED]:
            logger.warning(f"Cannot start session in status: {session.status}")
            return session
        
        session.status = SessionStatus.IN_PROGRESS
        session.started_at = datetime.utcnow().isoformat() + "Z"
        self._save_session(session)
        
        logger.info(f"Started session {session_id}")
        return session
    
    def get_current_prompt(self, session_id: str) -> Optional[Dict]:
        """
        Get the current prompt to record.
        
        Returns:
            Dict with prompt info and progress
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        prompt = session.current_prompt
        if not prompt:
            return None
        
        return {
            "session_id": session_id,
            "prompt": prompt.to_dict(),
            "progress": {
                "current": session.current_index + 1,
                "total": session.total_prompts,
                "completed": session.completed_count,
                "skipped": session.skipped_count,
                "percentage": session.progress_percentage,
                "total_duration": session.total_duration
            }
        }
    
    def validate_recording(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> RecordingValidation:
        """
        Validate a recording for quality.
        
        Args:
            audio: Audio waveform (float32, -1 to 1)
            sample_rate: Sample rate
            
        Returns:
            RecordingValidation with quality assessment
        """
        # Get audio quality metrics
        quality = AudioQualityAnalyzer.analyze(audio, sample_rate)
        duration = len(audio) / sample_rate
        
        issues = []
        
        # Check duration
        duration_ok = self.MIN_DURATION <= duration <= self.MAX_DURATION
        if duration < self.MIN_DURATION:
            issues.append(f"Recording too short ({duration:.1f}s, min {self.MIN_DURATION}s)")
        elif duration > self.MAX_DURATION:
            issues.append(f"Recording too long ({duration:.1f}s, max {self.MAX_DURATION}s)")
        
        # Check SNR
        snr_ok = quality.snr_db >= self.MIN_SNR
        if not snr_ok:
            issues.append(f"Audio quality too low (SNR {quality.snr_db:.1f}dB, need {self.MIN_SNR}dB)")
        
        # Check clipping
        clipping_ok = quality.clipping_percentage <= self.MAX_CLIPPING
        if not clipping_ok:
            issues.append(f"Audio clipping detected ({quality.clipping_percentage:.1f}%)")
        
        # Check silence
        silence_ok = quality.silence_percentage <= self.MAX_SILENCE
        if not silence_ok:
            issues.append(f"Too much silence ({quality.silence_percentage:.1f}%)")
        
        is_valid = all([duration_ok, snr_ok, clipping_ok, silence_ok])
        
        return RecordingValidation(
            is_valid=is_valid,
            duration_seconds=duration,
            audio_quality=quality,
            duration_ok=duration_ok,
            snr_ok=snr_ok,
            clipping_ok=clipping_ok,
            silence_ok=silence_ok,
            issues=issues
        )
    
    def submit_recording(
        self,
        session_id: str,
        audio: np.ndarray,
        sample_rate: int,
        auto_advance: bool = False
    ) -> Dict[str, Any]:
        """
        Submit a recording for the current prompt.
        
        Args:
            session_id: Session ID
            audio: Audio waveform (float32, -1 to 1)
            sample_rate: Sample rate
            auto_advance: Automatically move to next prompt if valid
            
        Returns:
            Dict with validation result and status
        """
        import soundfile as sf
        
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        if session.status != SessionStatus.IN_PROGRESS:
            return {"error": f"Session is not in progress (status: {session.status.value})"}
        
        prompt = session.current_prompt
        if not prompt:
            return {"error": "No current prompt"}
        
        # Validate recording
        validation = self.validate_recording(audio, sample_rate)
        
        # Save audio
        audio_filename = f"{prompt.prompt_id}.wav"
        audio_path = Path(session.output_dir) / "recordings" / audio_filename
        audio_path.parent.mkdir(exist_ok=True)
        
        sf.write(str(audio_path), audio, sample_rate)
        
        # Update prompt
        prompt.audio_path = str(audio_path)
        prompt.duration_seconds = validation.duration_seconds
        prompt.recorded_at = datetime.utcnow().isoformat() + "Z"
        prompt.validation = validation
        prompt.attempt_count += 1
        
        result = {
            "success": validation.is_valid,
            "validation": validation.to_dict(),
            "prompt_id": prompt.prompt_id
        }
        
        if validation.is_valid:
            prompt.is_accepted = True
            session.completed_count += 1
            
            result["message"] = "Recording accepted!"
            
            # Auto-advance if requested
            if auto_advance and session.current_index < len(session.prompts) - 1:
                session.current_index += 1
                result["advanced"] = True
                result["next_prompt"] = session.current_prompt.to_dict() if session.current_prompt else None
        else:
            result["message"] = "Recording has quality issues. Please try again."
            result["issues"] = validation.issues
        
        self._save_session(session)
        return result
    
    def next_prompt(self, session_id: str) -> Optional[Dict]:
        """Move to the next prompt"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        if session.current_index < len(session.prompts) - 1:
            session.current_index += 1
            self._save_session(session)
        
        return self.get_current_prompt(session_id)
    
    def previous_prompt(self, session_id: str) -> Optional[Dict]:
        """Move to the previous prompt"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        if session.current_index > 0:
            session.current_index -= 1
            self._save_session(session)
        
        return self.get_current_prompt(session_id)
    
    def skip_prompt(self, session_id: str) -> Optional[Dict]:
        """Skip the current prompt"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        session.skipped_count += 1
        
        if session.current_index < len(session.prompts) - 1:
            session.current_index += 1
        
        self._save_session(session)
        return self.get_current_prompt(session_id)
    
    def pause_session(self, session_id: str) -> Optional[WizardSession]:
        """Pause a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        if session.status == SessionStatus.IN_PROGRESS:
            session.status = SessionStatus.PAUSED
            self._save_session(session)
        
        return session
    
    def complete_session(self, session_id: str) -> Optional[WizardSession]:
        """Complete a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        session.status = SessionStatus.COMPLETED
        session.completed_at = datetime.utcnow().isoformat() + "Z"
        
        # Analyze final coverage
        if session.completed_count > 0:
            session.current_coverage = self._analyze_session_coverage(session)
        
        self._save_session(session)
        logger.info(f"Completed session {session_id} with {session.completed_count} recordings")
        
        return session
    
    def cancel_session(self, session_id: str) -> Optional[WizardSession]:
        """Cancel a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        session.status = SessionStatus.CANCELLED
        session.completed_at = datetime.utcnow().isoformat() + "Z"
        self._save_session(session)
        
        return session
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get a summary of the session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        # Collect all accepted audio paths
        audio_paths = [
            p.audio_path for p in session.prompts 
            if p.is_accepted and p.audio_path
        ]
        
        return {
            "session": session.to_dict(),
            "audio_paths": audio_paths,
            "ready_for_training": session.status == SessionStatus.COMPLETED and session.completed_count >= 10,
            "coverage_improvement": self._calculate_coverage_improvement(session)
        }
    
    def _save_session(self, session: WizardSession):
        """Save session to disk"""
        if session.output_dir:
            session.save(Path(session.output_dir) / "session.json")
    
    def _analyze_session_coverage(
        self,
        session: WizardSession
    ) -> Optional[PhonemeCoverageReport]:
        """Analyze phoneme coverage of session recordings"""
        audio_dir = Path(session.output_dir) / "recordings"
        if not audio_dir.exists():
            return None
        
        return self.phoneme_analyzer.analyze_audio_directory(
            str(audio_dir), session.language
        )
    
    def _calculate_coverage_improvement(
        self,
        session: WizardSession
    ) -> Optional[Dict]:
        """Calculate coverage improvement from initial to current"""
        if not session.initial_coverage or not session.current_coverage:
            return None
        
        return {
            "initial_coverage": session.initial_coverage.coverage_percentage,
            "current_coverage": session.current_coverage.coverage_percentage,
            "improvement": session.current_coverage.coverage_percentage - session.initial_coverage.coverage_percentage,
            "phonemes_added": len(session.current_coverage.found_phonemes - session.initial_coverage.found_phonemes),
            "phonemes_still_missing": len(session.current_coverage.missing_phonemes)
        }


# Convenience functions

def create_wizard_session(
    language: str,
    exp_name: str,
    prompt_count: int = 50,
    base_dir: Optional[str] = None
) -> WizardSession:
    """Create a new wizard session"""
    wizard = RecordingWizard(base_dir=base_dir)
    return wizard.create_session(language, exp_name, prompt_count)


def get_wizard_session(session_id: str, base_dir: Optional[str] = None) -> Optional[WizardSession]:
    """Get an existing wizard session"""
    wizard = RecordingWizard(base_dir=base_dir)
    return wizard.get_session(session_id)


if __name__ == "__main__":
    # Test the wizard
    wizard = RecordingWizard()
    
    # Create session
    session = wizard.create_session(
        language="en",
        exp_name="test_voice",
        prompt_count=5
    )
    
    print(f"Created session: {session.session_id}")
    print(f"Prompts: {session.total_prompts}")
    
    # Start session
    wizard.start_session(session.session_id)
    
    # Get current prompt
    current = wizard.get_current_prompt(session.session_id)
    print(f"Current prompt: {current['prompt']['prompt_text']}")
    
    # Simulate recording submission
    import numpy as np
    fake_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    result = wizard.submit_recording(session.session_id, fake_audio, 16000)
    print(f"Submission result: {result}")
