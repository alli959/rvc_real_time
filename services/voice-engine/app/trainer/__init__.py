"""
RVC Trainer Package

Provides voice model training capabilities:
- Full training pipeline
- Audio preprocessing
- F0 extraction
- Feature extraction (HuBERT)
- Model training
- FAISS index building
"""

from .pipeline import (
    RVCTrainingPipeline,
    TrainingConfig,
    TrainingProgress,
    TrainingResult,
    TrainingStatus,
    ModelMetadata,
    SampleRate,
    F0Method,
    RVCVersion,
    create_training_pipeline,
)

__all__ = [
    # Main classes
    "RVCTrainingPipeline",
    "TrainingConfig",
    "TrainingProgress",
    "TrainingResult",
    "TrainingStatus",
    "ModelMetadata",
    # Enums
    "SampleRate",
    "F0Method",
    "RVCVersion",
    # Functions
    "create_training_pipeline",
]
