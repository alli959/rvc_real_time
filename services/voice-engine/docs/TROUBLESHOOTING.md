# RVC Model Troubleshooting Guide

This document covers common issues with RVC (Retrieval-based Voice Conversion) models and how to diagnose/fix them.

## Issue: "Chipmunk" or High-Pitched Audio Output

### Symptoms
- Output audio sounds high-pitched, squeaky, or "chipmunk-like"
- Audio plays too fast
- Audio quality is degraded

### Root Causes

#### 1. Sample Rate Mismatch (Most Common)
**Problem**: The output audio is at one sample rate (e.g., 48kHz) but is being played back or processed as another rate (e.g., 16kHz).

**How to Diagnose**:
```bash
# Check model's target sample rate
cd services/voice-engine
python -c "
import torch
ckpt = torch.load('assets/models/YOUR_MODEL/YOUR_MODEL.pth', map_location='cpu')
print('Target SR:', ckpt['config'][0])
print('Version:', ckpt['config'][3] if len(ckpt['config']) > 3 else 'v1')
"
```

**Fix**: Ensure `resample_sr` parameter is set to `0` (keep model's native rate):
```python
# In RVCInferParams
resample_sr: int = 0  # DON'T force resampling!
```

#### 2. Wrong `resample_sr` Default
**Problem**: Code has `resample_sr=16000` instead of `resample_sr=0`, forcing 48kHz output to be resampled to 16kHz but played back at 48kHz.

**Location to Check**: `app/services/voice_conversion/model_manager.py`
```python
@dataclass
class RVCInferParams:
    resample_sr: int = 0  # ✓ Correct
    # resample_sr: int = 16000  # ✗ WRONG - causes chipmunk!
```

#### 3. PHP Frontend Fallback
**Problem**: PHP code falls back to wrong sample rate when API response is missing `sample_rate`.

**Location**: `apps/api/app/Http/Controllers/Api/TTSController.php`
```php
// OLD (wrong):
$sampleRate = $convertResponse->json('sample_rate', 16000);

// NEW (correct):
$sampleRate = $convertResponse->json('sample_rate', 48000);  // Safe fallback
```

### Prevention
1. Always use the model validator before deployment:
   ```bash
   curl -X POST http://localhost:8000/models/validate \
     -H "Content-Type: application/json" \
     -d '{"model_path": "assets/models/my-model/my-model.pth"}'
   ```

2. Check model metadata after training:
   ```bash
   python scripts/validate_model.py assets/models/my-model/my-model.pth
   ```

---

## Issue: Low Quality / Robotic Audio

### Symptoms
- Audio sounds artificial or "robotic"
- Loss of natural voice characteristics
- Distortion or artifacts

### Root Causes

#### 1. Low Index Rate
**Problem**: Not using enough of the index for voice retrieval.

**Fix**: Increase `index_rate` (0.0-1.0):
```python
params = RVCInferParams(
    index_rate=0.75,  # Default, try higher for more original character
)
```

#### 2. Missing Index File
**Problem**: Model loaded without .index file.

**Check**:
```bash
ls assets/models/YOUR_MODEL/*.index
```

#### 3. Too Much Protection
**Problem**: `protect` value too high removes too much pitch variance.

**Fix**: Lower `protect` value (0.0-0.5):
```python
params = RVCInferParams(
    protect=0.33,  # Default, try lower for more expressiveness
)
```

---

## Issue: Wrong Pitch / Gender Mismatch

### Symptoms
- Male voice sounds female or vice versa
- Pitch doesn't match expected output

### Root Causes

#### 1. Wrong `f0_up_key`
**Problem**: Pitch shift is set incorrectly.

**Guidelines**:
- Male → Female: `f0_up_key=12` (one octave up)
- Female → Male: `f0_up_key=-12` (one octave down)
- Same gender: `f0_up_key=0`

#### 2. Model Trained Without f0 Guidance
**Problem**: Model was trained with `if_f0=0`.

**Check**:
```python
import torch
ckpt = torch.load('model.pth', map_location='cpu')
print('f0 guidance:', ckpt['config'][2])  # 1=enabled, 0=disabled
```

**Note**: Models without f0 guidance cannot preserve pitch well.

---

## Issue: Inference Fails / Returns Original Audio

### Symptoms
- Output sounds exactly like input
- Error messages about missing files

### Root Causes

#### 1. Missing HuBERT Model
**Check**: `assets/hubert/hubert_base.pt` exists

**Fix**:
```bash
./scripts/download_pretrained_models.sh
```

#### 2. Missing RMVPE Model
**Check**: `assets/rmvpe/rmvpe.pt` exists

#### 3. Model Not Loaded
**Check logs for**:
```
"No model loaded, returning input as-is"
```

---

## Issue: CUDA Out of Memory

### Symptoms
- `CUDA out of memory` errors
- Inference crashes on GPU

### Solutions

1. **Reduce batch size** (for training)
2. **Use CPU inference**:
   ```python
   ModelManager(device='cpu')
   ```
3. **Clear GPU cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

## Model Validation Checklist

Before deploying a new model, validate:

- [ ] Target sample rate matches expected (32k/40k/48k)
- [ ] Version correct (v1 or v2)
- [ ] f0 guidance enabled (if_f0=1)
- [ ] No NaN/Inf values in weights
- [ ] Index file present and dimension matches version
- [ ] Index has sufficient vectors (>5000 recommended)

Use the validation endpoint:
```bash
curl -X POST http://localhost:8000/models/validate \
  -H "Content-Type: application/json" \
  -d '{"model_path": "assets/models/lexi-11"}'
```

---

## Training Parameter Recommendations

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample Rate | 40000 | Best compatibility |
| Version | v2 | Better quality |
| f0 Method | rmvpe | Most accurate pitch |
| Batch Size | 16 | Adjust for VRAM |
| Epochs | 100-300 | Monitor for overfitting |
| f0 Guidance | Enabled | Required for pitch preservation |

### Source Audio Guidelines
- **Duration**: 10-30 minutes of clean speech
- **Quality**: Studio quality, no background noise
- **Content**: Varied sentences, emotions, pitches
- **Format**: WAV, 44.1kHz or higher
