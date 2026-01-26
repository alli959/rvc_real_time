"""
Model Storage Management

Manages the model directory structure for voice models including:
- Raw recordings storage
- Preprocessed data (gt_wavs, features, f0)
- Trained model files (G_*.pth, D_*.pth, index)
- Model metadata and phoneme coverage

Directory structure:
    assets/models/{model_name}/
    ├── recordings/           # Raw uploaded audio files
    │   └── *.wav
    ├── 0_gt_wavs/           # Ground truth wavs (from preprocessing)
    ├── 1_16k_wavs/          # 16kHz resampled wavs
    ├── 2a_f0/               # F0 pitch features
    ├── 2b-f0nsf/            # F0 NSF features  
    ├── 3_feature768/        # HuBERT features
    ├── eval/                # Evaluation outputs
    ├── config.json          # Training config
    ├── filelist.txt         # Training manifest
    ├── model.pth            # Final model weights (or G_*.pth checkpoints)
    ├── model.index          # Final FAISS index
    ├── metadata.json        # Model metadata, phoneme coverage, categories
    └── *.log                # Training logs
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecordingInfo:
    """Information about a single recording"""
    filename: str
    duration_seconds: float
    sample_rate: int
    uploaded_at: str
    category: Optional[str] = None
    phonemes_detected: List[str] = field(default_factory=list)
    transcript: Optional[str] = None


@dataclass 
class CategoryStatus:
    """Status of a training category"""
    category_id: str
    name: str
    recordings_count: int = 0
    target_count: int = 10  # Minimum recordings for this category
    phonemes_covered: List[str] = field(default_factory=list)
    is_satisfied: bool = False  # True after training if model handles this category well
    

@dataclass
class ModelMetadata:
    """Metadata for a voice model"""
    name: str
    created_at: str
    updated_at: str
    language: str = "en"
    
    # Recording stats
    total_recordings: int = 0
    total_duration_seconds: float = 0.0
    recordings: List[RecordingInfo] = field(default_factory=list)
    
    # Category coverage
    categories: Dict[str, CategoryStatus] = field(default_factory=dict)
    
    # Phoneme coverage (updated after training)
    phoneme_coverage_percent: float = 0.0
    phonemes_covered: List[str] = field(default_factory=list)
    phonemes_missing: List[str] = field(default_factory=list)
    
    # Training info
    training_epochs: int = 0
    last_trained_at: Optional[str] = None
    training_config: Optional[Dict] = None
    
    # Model files
    has_model: bool = False
    has_index: bool = False
    model_version: str = "v2"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        # Handle nested dataclasses
        recordings = [RecordingInfo(**r) if isinstance(r, dict) else r for r in data.get('recordings', [])]
        categories = {}
        for cat_id, cat_data in data.get('categories', {}).items():
            if isinstance(cat_data, dict):
                categories[cat_id] = CategoryStatus(**cat_data)
            else:
                categories[cat_id] = cat_data
        
        data['recordings'] = recordings
        data['categories'] = categories
        return cls(**data)


class ModelStorage:
    """
    Manages storage for voice models and their associated data.
    
    Stores everything under assets/models/{model_name}/ to keep
    all model-related files together.
    
    Also scans /data/uploads/{model_name}/ for recordings from preprocessor.
    """
    
    def __init__(self, models_dir: str, uploads_dir: str = "/data/uploads"):
        """
        Initialize model storage.
        
        Args:
            models_dir: Path to the models directory (e.g., assets/models)
            uploads_dir: Path to uploads directory (e.g., /data/uploads)
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = Path(uploads_dir) if uploads_dir else None
        logger.info(f"ModelStorage initialized at {self.models_dir}")
    
    def get_model_dir(self, model_name: str) -> Path:
        """Get the directory for a model, creating if needed"""
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def get_recordings_dir(self, model_name: str) -> Path:
        """Get the recordings directory for a model"""
        recordings_dir = self.get_model_dir(model_name) / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        return recordings_dir
    
    def get_metadata_path(self, model_name: str) -> Path:
        """Get the metadata file path for a model"""
        return self.get_model_dir(model_name) / "metadata.json"
    
    def load_metadata(self, model_name: str) -> ModelMetadata:
        """Load or create metadata for a model"""
        metadata_path = self.get_metadata_path(model_name)
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                return ModelMetadata.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {model_name}: {e}")
        
        # Create new metadata
        now = datetime.utcnow().isoformat() + "Z"
        return ModelMetadata(
            name=model_name,
            created_at=now,
            updated_at=now
        )
    
    def save_metadata(self, model_name: str, metadata: ModelMetadata):
        """Save metadata for a model"""
        metadata.updated_at = datetime.utcnow().isoformat() + "Z"
        metadata_path = self.get_metadata_path(model_name)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        logger.debug(f"Saved metadata for {model_name}")
    
    def add_recording(
        self, 
        model_name: str, 
        audio_data: np.ndarray, 
        sample_rate: int,
        filename: Optional[str] = None,
        category: Optional[str] = None,
        transcript: Optional[str] = None
    ) -> RecordingInfo:
        """
        Add a recording to the model's recordings directory.
        
        Args:
            model_name: Name of the model
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate
            filename: Optional filename (auto-generated if not provided)
            category: Optional category for the recording
            transcript: Optional transcript/prompt text
            
        Returns:
            RecordingInfo for the saved recording
        """
        recordings_dir = self.get_recordings_dir(model_name)
        
        # Generate filename if not provided
        if not filename:
            existing = list(recordings_dir.glob("*.wav"))
            idx = len(existing) + 1
            filename = f"recording_{idx:04d}.wav"
        
        # Ensure .wav extension
        if not filename.endswith('.wav'):
            filename = filename.rsplit('.', 1)[0] + '.wav'
        
        # Save audio
        audio_path = recordings_dir / filename
        sf.write(str(audio_path), audio_data, sample_rate)
        
        # Calculate duration
        duration = len(audio_data) / sample_rate
        
        # Create recording info
        recording = RecordingInfo(
            filename=filename,
            duration_seconds=round(duration, 2),
            sample_rate=sample_rate,
            uploaded_at=datetime.utcnow().isoformat() + "Z",
            category=category,
            transcript=transcript
        )
        
        # Update metadata
        metadata = self.load_metadata(model_name)
        metadata.recordings.append(recording)
        metadata.total_recordings = len(metadata.recordings)
        metadata.total_duration_seconds = sum(r.duration_seconds for r in metadata.recordings)
        
        # Update category counts
        if category and category in metadata.categories:
            metadata.categories[category].recordings_count += 1
        
        self.save_metadata(model_name, metadata)
        
        logger.info(f"Added recording {filename} to {model_name} ({duration:.1f}s)")
        return recording
    
    def add_recordings_bulk(
        self,
        model_name: str,
        audio_files: List[Dict[str, Any]],
        language: str = "en"
    ) -> List[RecordingInfo]:
        """
        Add multiple recordings at once.
        
        Args:
            model_name: Name of the model
            audio_files: List of dicts with 'data', 'sample_rate', optional 'filename'
            language: Language for phoneme analysis
            
        Returns:
            List of RecordingInfo for saved recordings
        """
        recordings = []
        recordings_dir = self.get_recordings_dir(model_name)
        
        # Get starting index
        existing = list(recordings_dir.glob("*.wav"))
        start_idx = len(existing) + 1
        
        metadata = self.load_metadata(model_name)
        metadata.language = language
        
        for i, audio_file in enumerate(audio_files):
            audio_data = audio_file['data']
            sample_rate = audio_file['sample_rate']
            filename = audio_file.get('filename', f"recording_{start_idx + i:04d}.wav")
            
            if not filename.endswith('.wav'):
                filename = filename.rsplit('.', 1)[0] + '.wav'
            
            # Save audio
            audio_path = recordings_dir / filename
            sf.write(str(audio_path), audio_data, sample_rate)
            
            duration = len(audio_data) / sample_rate
            
            recording = RecordingInfo(
                filename=filename,
                duration_seconds=round(duration, 2),
                sample_rate=sample_rate,
                uploaded_at=datetime.utcnow().isoformat() + "Z"
            )
            
            metadata.recordings.append(recording)
            recordings.append(recording)
        
        # Update totals
        metadata.total_recordings = len(metadata.recordings)
        metadata.total_duration_seconds = sum(r.duration_seconds for r in metadata.recordings)
        
        self.save_metadata(model_name, metadata)
        
        logger.info(f"Added {len(recordings)} recordings to {model_name}")
        return recordings
    
    def get_all_recording_paths(self, model_name: str) -> List[str]:
        """Get paths to all recordings for a model"""
        recordings_dir = self.get_recordings_dir(model_name)
        return [str(p) for p in recordings_dir.glob("*.wav")]
    
    def get_recording_stats(self, model_name: str) -> Dict[str, Any]:
        """Get statistics about recordings for a model"""
        metadata = self.load_metadata(model_name)
        
        return {
            "total_recordings": metadata.total_recordings,
            "total_duration_seconds": metadata.total_duration_seconds,
            "total_duration_minutes": round(metadata.total_duration_seconds / 60, 1),
            "language": metadata.language,
            "categories": {
                cat_id: {
                    "name": cat.name,
                    "recordings": cat.recordings_count,
                    "target": cat.target_count,
                    "satisfied": cat.is_satisfied
                }
                for cat_id, cat in metadata.categories.items()
            }
        }
    
    def initialize_categories(self, model_name: str, categories: Dict[str, Dict]):
        """
        Initialize category tracking for a model.
        
        Args:
            model_name: Name of the model
            categories: Dict mapping category_id to category info
        """
        metadata = self.load_metadata(model_name)
        
        for cat_id, cat_info in categories.items():
            if cat_id not in metadata.categories:
                metadata.categories[cat_id] = CategoryStatus(
                    category_id=cat_id,
                    name=cat_info.get('name', cat_id),
                    phonemes_covered=cat_info.get('phonemes', []),
                    target_count=cat_info.get('target_count', 10)
                )
        
        self.save_metadata(model_name, metadata)
    
    def update_after_training(
        self,
        model_name: str,
        phoneme_coverage: float,
        phonemes_covered: List[str],
        phonemes_missing: List[str],
        training_epochs: int,
        training_config: Optional[Dict] = None
    ):
        """
        Update metadata after training completes.
        
        Args:
            model_name: Name of the model
            phoneme_coverage: Coverage percentage (0-100)
            phonemes_covered: List of covered phonemes
            phonemes_missing: List of missing phonemes
            training_epochs: Number of epochs trained
            training_config: Training configuration used
        """
        metadata = self.load_metadata(model_name)
        
        metadata.phoneme_coverage_percent = phoneme_coverage
        metadata.phonemes_covered = phonemes_covered
        metadata.phonemes_missing = phonemes_missing
        metadata.training_epochs = training_epochs
        metadata.last_trained_at = datetime.utcnow().isoformat() + "Z"
        metadata.training_config = training_config
        
        # Check for model files
        model_dir = self.get_model_dir(model_name)
        metadata.has_model = any(model_dir.glob("*.pth"))
        metadata.has_index = any(model_dir.glob("*.index"))
        
        # Update category satisfaction based on phoneme coverage
        covered_set = set(phonemes_covered)
        for cat_id, cat_status in metadata.categories.items():
            cat_phonemes = set(cat_status.phonemes_covered)
            if cat_phonemes:
                overlap = len(cat_phonemes & covered_set) / len(cat_phonemes)
                cat_status.is_satisfied = overlap >= 0.8  # 80% coverage = satisfied
        
        self.save_metadata(model_name, metadata)
        logger.info(f"Updated {model_name} after training: {phoneme_coverage:.1f}% coverage")
    
    def scan_existing_model(self, model_name: str) -> ModelMetadata:
        """
        Scan an existing model directory and create/update metadata.
        
        Useful for models that were trained outside this system.
        Also scans uploads directory for recordings from preprocessor.
        """
        model_dir = self.get_model_dir(model_name)
        metadata = self.load_metadata(model_name)
        
        # Check for recordings in model directory
        recordings_dir = model_dir / "recordings"
        if recordings_dir.exists():
            existing_files = {r.filename for r in metadata.recordings}
            for wav_path in recordings_dir.glob("*.wav"):
                if wav_path.name not in existing_files:
                    try:
                        data, sr = sf.read(str(wav_path))
                        duration = len(data) / sr
                        metadata.recordings.append(RecordingInfo(
                            filename=wav_path.name,
                            duration_seconds=round(duration, 2),
                            sample_rate=sr,
                            uploaded_at=datetime.fromtimestamp(wav_path.stat().st_mtime).isoformat() + "Z"
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to read {wav_path}: {e}")
        
        # Also check trainset directory (where audio is copied for training)
        trainset_dir = model_dir / "trainset"
        if trainset_dir.exists():
            existing_files = {r.filename for r in metadata.recordings}
            for pattern in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.webm"]:
                for audio_path in trainset_dir.glob(pattern):
                    if audio_path.name not in existing_files:
                        try:
                            data, sr = sf.read(str(audio_path))
                            duration = len(data) / sr
                            metadata.recordings.append(RecordingInfo(
                                filename=audio_path.name,
                                duration_seconds=round(duration, 2),
                                sample_rate=sr,
                                uploaded_at=datetime.fromtimestamp(audio_path.stat().st_mtime).isoformat() + "Z"
                            ))
                            logger.debug(f"Found trainset file: {audio_path.name} ({duration:.1f}s)")
                        except Exception as e:
                            logger.warning(f"Failed to read {audio_path}: {e}")
            if metadata.recordings:
                logger.info(f"Found {len(metadata.recordings)} files in trainset for {model_name}")
        
        # NOTE: We no longer scan a separate uploads directory.
        # All uploaded files now go directly to {model_dir}/trainset/
        # This simplifies the data flow - one canonical location for all training audio.
        
        # Also check 0_gt_wavs for existing training data
        gt_wavs_dir = model_dir / "0_gt_wavs"
        if gt_wavs_dir.exists():
            wav_count = len(list(gt_wavs_dir.glob("*.wav")))
            if wav_count > 0:
                logger.info(f"Found {wav_count} preprocessed wavs in 0_gt_wavs")
        
        # Check for model files
        metadata.has_model = any(model_dir.glob("*.pth"))
        metadata.has_index = any(model_dir.glob("*.index"))
        
        # Check config.json
        config_path = model_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    metadata.training_config = json.load(f)
            except Exception:
                pass
        
        # Update totals
        metadata.total_recordings = len(metadata.recordings)
        metadata.total_duration_seconds = sum(r.duration_seconds for r in metadata.recordings)
        
        self.save_metadata(model_name, metadata)
        return metadata
    
    def list_models(self) -> List[str]:
        """List all model directories"""
        models = []
        for item in self.models_dir.iterdir():
            if item.is_dir():
                models.append(item.name)
        return sorted(models)
    
    def delete_recordings(self, model_name: str) -> int:
        """Delete all recordings for a model (keeps trained model)"""
        recordings_dir = self.get_recordings_dir(model_name)
        count = 0
        for wav_path in recordings_dir.glob("*.wav"):
            wav_path.unlink()
            count += 1
        
        # Clear recordings from metadata
        metadata = self.load_metadata(model_name)
        metadata.recordings = []
        metadata.total_recordings = 0
        metadata.total_duration_seconds = 0
        
        for cat_status in metadata.categories.values():
            cat_status.recordings_count = 0
        
        self.save_metadata(model_name, metadata)
        
        logger.info(f"Deleted {count} recordings from {model_name}")
        return count


# Global instance
_model_storage: Optional[ModelStorage] = None

# Path to uploads directory (local voice-engine uploads)
UPLOADS_DIR = "/app/uploads"


def get_model_storage() -> ModelStorage:
    """Get the global model storage instance"""
    global _model_storage
    if _model_storage is None:
        from pathlib import Path
        models_dir = Path(__file__).parent.parent / "assets" / "models"
        # Pass uploads_dir so we can scan preprocessor uploads
        uploads_dir = UPLOADS_DIR if Path(UPLOADS_DIR).exists() else None
        _model_storage = ModelStorage(str(models_dir), uploads_dir)
    return _model_storage


def init_model_storage(models_dir: str, uploads_dir: str = None):
    """Initialize the global model storage"""
    global _model_storage
    _model_storage = ModelStorage(models_dir, uploads_dir or UPLOADS_DIR)
