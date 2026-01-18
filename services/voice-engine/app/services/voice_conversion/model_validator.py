"""
Model Validator Module - Validates RVC model artifacts before deployment

This module provides comprehensive validation of RVC model files to catch
common issues BEFORE they cause problems in production:

1. Sample rate mismatches (most common cause of "chipmunk" audio)
2. Corrupted checkpoints (NaN/Inf weights)
3. Missing/mismatched index files
4. Version compatibility issues (v1 vs v2)

Usage:
    from app.services.voice_conversion.model_validator import validate_model
    
    result = validate_model('/path/to/model.pth', '/path/to/model.index')
    if not result['valid']:
        print(f"Issues: {result['issues']}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""
    valid: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'valid': self.valid,
            'issues': self.issues,
            'warnings': self.warnings,
            'metadata': self.metadata
        }
    
    def add_issue(self, msg: str):
        self.valid = False
        self.issues.append(msg)
        
    def add_warning(self, msg: str):
        self.warnings.append(msg)


def validate_checkpoint(model_path: str) -> ValidationResult:
    """
    Validate an RVC model checkpoint file.
    
    Checks:
    - File exists and is loadable
    - Contains expected keys (weight, config, etc.)
    - Sample rate is valid (32000, 40000, 48000)
    - Version is specified (v1 or v2)
    - No NaN or Inf values in weights
    - f0 guidance status
    
    Returns:
        ValidationResult with validation status and metadata
    """
    result = ValidationResult()
    path = Path(model_path)
    
    if not path.exists():
        result.add_issue(f"Model file not found: {model_path}")
        return result
    
    if not path.suffix.lower() in ['.pth', '.pt', '.ckpt']:
        result.add_warning(f"Unexpected file extension: {path.suffix}")
    
    try:
        import torch
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
    except Exception as e:
        result.add_issue(f"Failed to load checkpoint: {e}")
        return result
    
    # Check if it's a training checkpoint vs inference model
    if 'model' in checkpoint and 'optimizer' in checkpoint:
        result.add_warning("This appears to be a training checkpoint (has optimizer state). "
                         "For inference, use a model exported without optimizer state.")
        result.metadata['format'] = 'training'
    else:
        result.metadata['format'] = 'inference'
    
    # Extract model weight dict
    if 'weight' in checkpoint:
        weight_dict = checkpoint['weight']
    elif 'model' in checkpoint:
        weight_dict = checkpoint['model']
    else:
        # Try the checkpoint itself as weight dict
        weight_dict = checkpoint
        result.add_warning("Checkpoint structure is non-standard")
    
    # Count parameters
    total_params = 0
    nan_params = 0
    inf_params = 0
    
    for key, tensor in weight_dict.items():
        if hasattr(tensor, 'numel'):
            total_params += tensor.numel()
            if torch.isnan(tensor).any():
                nan_params += 1
                result.add_issue(f"NaN values found in weight: {key}")
            if torch.isinf(tensor).any():
                inf_params += 1
                result.add_issue(f"Inf values found in weight: {key}")
    
    result.metadata['total_params'] = total_params
    
    # Check config
    config = checkpoint.get('config', [])
    if isinstance(config, list) and len(config) >= 3:
        # Format: [tgt_sr, json_config, if_f0, version]
        result.metadata['tgt_sr'] = config[0]
        result.metadata['if_f0'] = config[2] if len(config) > 2 else None
        result.metadata['version'] = config[3] if len(config) > 3 else 'v1'
        
        # Validate sample rate
        valid_sr = [16000, 32000, 40000, 48000]
        if config[0] not in valid_sr:
            result.add_warning(f"Unusual target sample rate: {config[0]}Hz. "
                             f"Expected one of {valid_sr}")
        
        # Check f0 guidance
        if config[2] == 0:
            result.add_warning("f0 guidance disabled - model won't preserve pitch well")
            
    elif 'params' in checkpoint:
        # Alternative format with 'params' dict
        params = checkpoint['params']
        result.metadata['tgt_sr'] = params.get('sr', 'unknown')
        result.metadata['version'] = params.get('version', 'unknown')
        result.metadata['if_f0'] = params.get('f0', 'unknown')
    else:
        result.add_warning("Config section missing or malformed - cannot verify sample rate")
    
    # Check for info section
    if 'info' in checkpoint:
        result.metadata['info'] = checkpoint['info']
        
    # Check embedding dimension for version
    for key in weight_dict:
        if 'emb_g' in key and hasattr(weight_dict[key], 'shape'):
            emb_dim = weight_dict[key].shape[0]
            result.metadata['emb_dim'] = emb_dim
            if emb_dim == 768:
                inferred_version = 'v2'
            elif emb_dim == 256:
                inferred_version = 'v1'
            else:
                inferred_version = 'unknown'
            
            declared_version = result.metadata.get('version', 'unknown')
            if declared_version != 'unknown' and inferred_version != 'unknown':
                if declared_version != inferred_version:
                    result.add_issue(f"Version mismatch: declared={declared_version}, "
                                   f"inferred from emb_dim={inferred_version}")
            break
    
    return result


def validate_index(index_path: str, model_version: str = 'v2') -> ValidationResult:
    """
    Validate a FAISS index file.
    
    Checks:
    - File exists and is loadable
    - Dimension matches model version (256 for v1, 768 for v2)
    - Has reasonable number of vectors
    
    Returns:
        ValidationResult with validation status and metadata
    """
    result = ValidationResult()
    path = Path(index_path)
    
    if not path.exists():
        result.add_issue(f"Index file not found: {index_path}")
        return result
        
    if not path.suffix.lower() == '.index':
        result.add_warning(f"Unexpected file extension: {path.suffix}")
    
    try:
        import faiss
        index = faiss.read_index(str(path))
    except Exception as e:
        result.add_issue(f"Failed to load index: {e}")
        return result
    
    # Get index properties
    dimension = index.d
    n_vectors = index.ntotal
    
    result.metadata['dimension'] = dimension
    result.metadata['n_vectors'] = n_vectors
    result.metadata['index_type'] = type(index).__name__
    
    # Check dimension matches version
    expected_dim = 768 if model_version == 'v2' else 256
    if dimension != expected_dim:
        result.add_issue(f"Index dimension {dimension} doesn't match model version {model_version} "
                        f"(expected {expected_dim})")
    
    # Check vector count
    if n_vectors < 100:
        result.add_warning(f"Very few vectors in index ({n_vectors}). "
                         "Quality may be poor. Recommend at least 5000 vectors.")
    elif n_vectors < 1000:
        result.add_warning(f"Low vector count ({n_vectors}). Consider more training data.")
    
    return result


def validate_model_directory(model_dir: str) -> ValidationResult:
    """
    Validate a complete model directory.
    
    Expects structure:
        model_dir/
            model_name.pth           # Required: inference model
            *.index                  # Optional: FAISS index
            config.json              # Optional: training config
            metadata.json            # Optional: model metadata
    
    Returns:
        ValidationResult with validation status and metadata
    """
    result = ValidationResult()
    path = Path(model_dir)
    
    if not path.exists():
        result.add_issue(f"Model directory not found: {model_dir}")
        return result
    
    if not path.is_dir():
        result.add_issue(f"Not a directory: {model_dir}")
        return result
    
    # Find model file
    model_files = list(path.glob('*.pth'))
    # Filter out training checkpoints (G_*.pth, D_*.pth)
    inference_models = [f for f in model_files 
                       if not f.name.startswith('G_') and not f.name.startswith('D_')]
    
    if not inference_models:
        result.add_issue("No inference model (.pth) found in directory")
        return result
    
    if len(inference_models) > 1:
        result.add_warning(f"Multiple inference models found: {[f.name for f in inference_models]}")
    
    model_file = inference_models[0]
    result.metadata['model_file'] = str(model_file)
    
    # Validate model
    model_result = validate_checkpoint(str(model_file))
    result.issues.extend(model_result.issues)
    result.warnings.extend(model_result.warnings)
    result.metadata['model'] = model_result.metadata
    if not model_result.valid:
        result.valid = False
    
    # Find and validate index
    index_files = list(path.glob('*.index'))
    if index_files:
        index_file = index_files[0]
        result.metadata['index_file'] = str(index_file)
        
        model_version = model_result.metadata.get('version', 'v2')
        index_result = validate_index(str(index_file), model_version)
        result.issues.extend(index_result.issues)
        result.warnings.extend(index_result.warnings)
        result.metadata['index'] = index_result.metadata
        if not index_result.valid:
            result.valid = False
    else:
        result.add_warning("No index file found - inference will work but quality may be lower")
    
    # Check for config.json
    config_file = path / 'config.json'
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            result.metadata['config'] = config
            
            # Validate SR consistency
            config_sr = config.get('sample_rate') or config.get('sr')
            model_sr = model_result.metadata.get('tgt_sr')
            if config_sr and model_sr and config_sr != model_sr:
                result.add_warning(f"Sample rate mismatch: config.json says {config_sr}Hz "
                                 f"but model is {model_sr}Hz")
        except Exception as e:
            result.add_warning(f"Could not parse config.json: {e}")
    
    # Check for metadata.json  
    metadata_file = path / 'metadata.json'
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            result.metadata['training_metadata'] = metadata
        except Exception as e:
            result.add_warning(f"Could not parse metadata.json: {e}")
    
    return result


def validate_model(
    model_path: str,
    index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level validation function for a model + optional index.
    
    Args:
        model_path: Path to model file or directory
        index_path: Optional path to index file
        
    Returns:
        Dict with validation results
    """
    path = Path(model_path)
    
    if path.is_dir():
        result = validate_model_directory(str(path))
    else:
        result = validate_checkpoint(str(path))
        
        if index_path:
            model_version = result.metadata.get('version', 'v2')
            index_result = validate_index(index_path, model_version)
            result.issues.extend(index_result.issues)
            result.warnings.extend(index_result.warnings)
            result.metadata['index'] = index_result.metadata
            if not index_result.valid:
                result.valid = False
    
    return result.to_dict()


def quick_check_model_sr(model_path: str) -> Tuple[bool, int]:
    """
    Quick check to get model's target sample rate.
    
    Returns:
        (success, sample_rate) - sample_rate is 0 if unknown
    """
    try:
        import torch
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        config = checkpoint.get('config', [])
        if isinstance(config, list) and len(config) >= 1:
            return True, int(config[0])
        return False, 0
    except Exception:
        return False, 0


# CLI interface
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_validator.py <model_path_or_dir> [index_path]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    index_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = validate_model(model_path, index_path)
    print(json.dumps(result, indent=2, default=str))
    
    sys.exit(0 if result['valid'] else 1)
