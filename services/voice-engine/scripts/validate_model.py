#!/usr/bin/env python3
"""
Validate RVC model checkpoint format.
Returns exit code 0 for valid inference models, 1 for invalid/raw checkpoints.

Usage:
    python validate_model.py /path/to/model.pth
    
Output (JSON):
    {"valid": true, "format": "inference", "version": "v2"}
    {"valid": false, "format": "raw_checkpoint", "reason": "Training checkpoint, not exported"}
"""

import sys
import json
import os

def validate_model(model_path: str) -> dict:
    """Validate an RVC model checkpoint."""
    result = {
        "valid": False,
        "format": "unknown",
        "reason": "",
        "path": model_path
    }
    
    if not os.path.exists(model_path):
        result["reason"] = "File not found"
        return result
    
    # Skip D_*.pth files (discriminator checkpoints)
    basename = os.path.basename(model_path)
    if basename.startswith('D_') and basename.endswith('.pth'):
        result["format"] = "discriminator"
        result["reason"] = "Discriminator checkpoint (D_*.pth), not a voice model"
        return result
    
    try:
        import torch
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        keys = list(ckpt.keys())
        
        # Valid inference model has 'weight' key
        if 'weight' in ckpt:
            result["valid"] = True
            result["format"] = "inference"
            
            # Detect version
            if 'config' in ckpt and len(ckpt['config']) >= 17:
                result["version"] = "v2"
            elif 'version' in ckpt:
                result["version"] = ckpt.get('version', 'v1')
            else:
                result["version"] = "v1"
            
            return result
        
        # Some RVC checkpoints use 'model' key but are still usable for inference
        # if they have the proper config metadata
        if 'model' in ckpt and 'config' in ckpt and 'version' in ckpt:
            # This is a special export format that can be converted
            # Check if it has the right config structure
            config = ckpt.get('config', [])
            if isinstance(config, (list, tuple)) and len(config) >= 17:
                result["valid"] = True
                result["format"] = "inference_alt"
                result["version"] = ckpt.get('version', 'v1')
                result["note"] = "Alternative inference format (model key instead of weight)"
                return result
        
        # Raw training checkpoint has 'model' + 'iteration' but NO config/version
        if 'model' in ckpt and 'iteration' in ckpt:
            # Check if it's truly raw (no config) vs usable
            if 'config' not in ckpt or 'version' not in ckpt:
                result["format"] = "raw_checkpoint"
                result["reason"] = "Training checkpoint (not exported). Use RVC WebUI 'Export Model' to create inference model."
                return result
            else:
                # Has some metadata, might be usable
                result["format"] = "partial_export"
                result["reason"] = "Partially exported model - may need re-export"
                # For now, allow these but with warning
                result["valid"] = True
                return result
        
        # Unknown format
        result["format"] = "unknown"
        result["reason"] = f"Unknown checkpoint format. Keys: {keys[:5]}"
        return result
        
    except Exception as e:
        result["format"] = "error"
        result["reason"] = f"Failed to load: {str(e)}"
        return result

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"valid": False, "reason": "No model path provided"}))
        sys.exit(1)
    
    model_path = sys.argv[1]
    result = validate_model(model_path)
    print(json.dumps(result))
    sys.exit(0 if result["valid"] else 1)

if __name__ == "__main__":
    main()
