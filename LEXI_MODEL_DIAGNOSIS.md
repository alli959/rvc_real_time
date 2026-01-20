# Lexi RVC Model Diagnosis Report

## Problem Summary
The trained RVC model produces **noise-like output** instead of proper voice conversion, despite training completing successfully with decreasing losses.

## Key Diagnostic Findings

### 1. Model Extraction Bug (FIXED)
**Critical Issue Discovered**: The `lexi.pth` inference model was extracted from an **early checkpoint** (G_396.pth = 22 epochs) instead of the final checkpoint (G_2376.pth = 132 epochs).

- `lexi.pth` timestamp: 04:17 (first checkpoint)
- `G_2376.pth` timestamp: 04:40 (final checkpoint)
- Training completed at: 04:41

**Action Taken**: Re-extracted model from G_2376.pth (132 epochs, 2376 steps)

### 2. Output Analysis Shows Noise
Even after extracting from the final checkpoint:

| Metric | Input (Original) | Output (Converted) | Healthy Range |
|--------|------------------|-------------------|---------------|
| Spectral Flatness | 0.0019 | **0.4118** | <0.1 for speech |
| Zero Crossing Rate | 1059 Hz | **9747 Hz** | Similar to input |
| Max Amplitude | 0.2937 | 0.1584 | ~same |
| RMS | 0.0764 | 0.1258 | ~same |

**Interpretation**: Spectral flatness of 0.41 (vs 0.002) indicates the output is almost white noise-like rather than tonal speech content.

### 3. Training Configuration
```
Model: lexi
Version: v2
Sample Rate: 48kHz
F0 Extraction: RMVPE
Total Steps: 2376 (from checkpoint filename) / 2465 (from logs)
Epochs: 132-137
Batch Size: 6
Dataset: 84 audio segments (~10 minutes total)
```

### 4. Loss Progression (Training Log)
```
Start (Step ~1):
  loss_mel=333.991, loss_kl=72.584

End (Step 2465):
  loss_mel=207.552, loss_kl=0.751
```
**Critical Observation**: Mel loss dropped quickly in epoch 1 (333→207) but then oscillated between 170-210 for the entire training! It never converged below 170.

This indicates **training stagnation** - the model learned the rough structure quickly but couldn't refine the details.

### 4b. Pretrained Model Used
```
pretrainG: /app/assets/pretrained_v2/f0G48k.pth
pretrainD: /app/assets/pretrained_v2/f0D48k.pth
```
Training started from pretrained weights (not from scratch), which is correct.

### 5. Preprocessing Verification
| Check | Result |
|-------|--------|
| GT wavs (48kHz) count | 84 files |
| 16k wavs count | 84 files |
| Duration alignment | ✅ Match within 0.02s |
| Feature768 files | 84 files, no NaN/Inf |
| F0 files | 84 files, reasonable values |
| F0nsf files | 84 files, reasonable values |

### 6. Model Weights Analysis
```python
# From G_2376.pth checkpoint
Total weight layers: 560 (training) → 457 (inference, excluding enc_q)
Has NaN: False
Has Inf: False
Weight statistics: Normal ranges (mean ~0, std ~0.05-0.1)
```

### 7. Comparison with Working Model (Biden)
| Attribute | Biden (Works) | Lexi (Broken) |
|-----------|---------------|---------------|
| Info | 500epoch | 132epoch |
| Sample Rate | 40k | 48k |
| Weight Layers | 457 | 457 ✅ |
| Config Length | 18 params | 18 params ✅ |
| Model Size | ~55MB | ~55MB ✅ |

## Hypotheses for Broken Output

### Hypothesis 1: Insufficient Training
- Biden trained for 500 epochs
- Lexi only trained for 132 epochs
- With small dataset (84 segments), may need more epochs

### Hypothesis 2: Feature Extraction Misalignment
Despite file counts matching, there may be subtle misalignment between:
- 48kHz ground truth audio
- 16kHz audio for HuBERT features
- 768-dimensional HuBERT features

### Hypothesis 3: Pretrained Model Mismatch
Training from scratch with small dataset instead of fine-tuning from a pretrained model may cause collapse.

### Hypothesis 4: Learning Rate / Optimizer Issues
- Final learning rate: 0.0001 (normal)
- But may have been too high initially for small dataset

### Hypothesis 5: Batch Size Too Small
- batch_size=6 with 84 samples = 14 steps/epoch
- May cause noisy gradients

## Files for Analysis

### Model Files
- `/app/logs/lexi/lexi.pth` - Extracted inference model (132 epochs)
- `/app/logs/lexi/G_2376.pth` - Final training checkpoint
- `/app/logs/lexi/config.json` - Training configuration

### Audio Files
- `/app/logs/lexi/0_gt_wavs/` - 84 ground truth segments (48kHz)
- `/app/logs/lexi/1_16k_wavs/` - 84 resampled segments (16kHz)
- `/tmp/lexi_test.wav` - Test conversion output

### Feature Files
- `/app/logs/lexi/3_feature768/` - HuBERT features
- `/app/logs/lexi/2a_f0/` - F0 pitch contours
- `/app/logs/lexi/2b_f0nsf/` - F0 NSF features

### Logs
- `/app/logs/lexi/train.log` - Training progress log

## Questions for Review

1. Is 132 epochs enough for a 48kHz v2 model with 84 segments?
2. Should we use a pretrained base model instead of training from scratch?
3. Is there something special about 48kHz config vs 40kHz that requires different training?
4. Could the feature extraction (HuBERT) be using wrong sample rate internally?
5. What mel loss value indicates a properly trained model? (Current: 207)

## Code Changes Made

### 1. Step-Based Training (auto_config.py)
Ensured minimum ~1800 optimizer steps for small datasets by adjusting batch_size and epochs.

### 2. Preprocessing Alignment (pipeline.py)  
Added `_get_slice_boundaries()` to slice audio once and apply same time boundaries to both 48kHz and 16kHz versions.

### 3. datetime Bug Fix (train.py)
Fixed `datetime.now()` → `datetime.datetime.now()` for resume functionality.

## Next Steps to Try

1. **Train longer**: Increase to 500+ epochs like Biden model
2. **Use pretrained model**: Fine-tune from an existing 48kHz model
3. **Reduce learning rate**: Try lr=0.0001 from start
4. **Increase dataset**: Add more training audio
5. **Check inference code**: Verify the pipeline uses correct parameters

---
Generated: 2026-01-20
Model: lexi (132 epochs, 48kHz, RVC v2)
Status: **BROKEN** - Produces noise instead of voice conversion
