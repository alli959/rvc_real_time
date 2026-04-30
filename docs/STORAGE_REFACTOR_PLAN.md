# Storage Refactoring Plan: Separating Model Files from Training Artifacts

## Executive Summary

Currently, checkpoint files (G_*.pth, D_*.pth), extracted intermediate models (*_e*_s*.pth), and training artifacts are stored in `/storage/models/<model_name>/` alongside final inference models. This creates:
- Large model directories (10GB+ per model with checkpoints)
- Confusion between inference models and training artifacts
- Cleanup complexity
- Backup/sync overhead for production

**Goal**: `/storage/models/<model_name>/` should ONLY contain:
- `<model_name>.pth` - Final inference model
- `<model_name>.index` - FAISS index for voice matching
- `config.json` - Model configuration (optional)
- `metadata.json` - Model metadata (optional)

All training-related files go to:
- `/storage/data/preprocess/<model_name>/` - Preprocessed audio, features
- `/storage/data/training/<model_name>/` - Checkpoints, TensorBoard logs

---

## Implementation Status

| Component | File | Status | Changes Made |
|-----------|------|--------|--------------|
| Voice-Engine | `pipeline.py` | ✅ Complete | Checkpoint handling uses training_dir, symlink targets training_dir |
| Voice-Engine | `trainer_api.py` | ✅ Complete | Checkpoint scanning searches training_dir first |
| Voice-Engine | `train.py` | ✅ Complete | `savee()` calls pass `output_dir=hps.model_dir` (training dir via symlink) |
| Laravel API | `VoiceModelScanner.php` | ✅ Complete | Removed G_*.pth fallback - models dir is inference-only |
| Migration Script | `scripts/migrate-checkpoints-to-training-dir.sh` | ✅ Created | Moves existing checkpoints from models/ to training/ |
| Voice-Engine | `jobs/checkpoint.py` | ⬜ Optional | Not actively used - future enhancement |
| Laravel API | `TrainingRunService.php` | ⬜ No changes needed | checkpoint_directory derived from paths automatically |

---

## Current State Analysis

### Files Currently in `/storage/models/<model_name>/`

| File Pattern | Type | Should Be In |
|--------------|------|--------------|
| `<model>.pth` | Final inference model | ✅ Keep here |
| `<model>.index` | FAISS index | ✅ Keep here |
| `config.json` | Model config | ✅ Keep here |
| `metadata.json` | Metadata | ✅ Keep here |
| `G_*.pth` | Generator checkpoints (800MB each) | ❌ Move to `/storage/data/training/` |
| `D_*.pth` | Discriminator checkpoints (800MB each) | ❌ Move to `/storage/data/training/` |
| `*_e*_s*.pth` | Intermediate extracted models | ❌ Move to `/storage/data/training/` |
| `added_*.index` | Training index | ❌ Move to `/storage/data/training/` |
| `total_fea.npy` | Feature aggregation | ❌ Move to `/storage/data/preprocess/` |
| `.smoke_test_ready.json` | Training marker | ❌ Move to `/storage/data/training/` |
| `trainset/` | Training audio | ❌ Move to `/storage/data/preprocess/` |

---

## Components Requiring Changes

### 1. Voice-Engine Service (Python)

#### A. `services/voice-engine/infer/modules/train/train.py`

**Current behavior**: Writes checkpoints to `hps.model_dir` which is symlinked to preprocess dir
**Problem**: Sometimes symlink points to `/storage/models/` or checkpoints get copied there

| Change | Lines | Description |
|--------|-------|-------------|
| Checkpoint output path | ~892-920 | Change `hps.model_dir` handling to explicitly use training dir |
| Model extraction path | savee() calls | Pass explicit path to training dir, not models dir |

**Changes needed**:
```python
# train.py should write checkpoints to dedicated training directory
# NOT to hps.model_dir if it points to models directory
```

#### B. `services/voice-engine/infer/modules/train/extract/process_ckpt.py`

**Current behavior**: `savee()` writes to `assets/models/` or provided path

| Function | Lines | Current | Target |
|----------|-------|---------|--------|
| `savee()` | 72-121 | Writes to `assets/models/{exp_name}/` | Write intermediate to training dir |
| `save_infer_model()` | 122-156 | Writes to provided path | Keep - used for final export |
| `merge()` | 192-243 | Writes to model dir | Keep - final merged model |

**Changes needed**:
- Add new function `save_training_checkpoint()` that writes to training dir
- Modify `savee()` default path to training directory
- Keep final model export to `/storage/models/`

#### C. `services/voice-engine/app/trainer/pipeline.py`

**Current behavior**: Mixed - writes to preprocess_dir but also copies to models_exp_dir

| Method | Lines | Current | Target |
|--------|-------|---------|--------|
| `train()` | ~720-730 | Sets up `models_exp_dir` for checkpoints | Use `training_exp_dir` |
| Checkpoint copy logic | ~946-960 | Copies from models to preprocess | Reverse: training as source |
| `_resolve_pretrained_paths_with_checkpoint()` | ~415-520 | Searches exp_dir only | Search training_dir first |
| Post-training | ~1080-1100 | Copies model to models dir | Keep - for final model only |

**Changes needed**:
```python
# In train():
# 1. Checkpoints go to training_exp_dir, NOT models_exp_dir
# 2. Symlink ./logs/{exp_name} -> training_exp_dir (not preprocess)
# 3. Only copy final .pth and .index to models_exp_dir at completion
```

#### D. `services/voice-engine/app/trainer_api.py`

**Current behavior**: `detect_existing_model()` and endpoints scan models directory

| Function/Endpoint | Lines | Current | Target |
|-------------------|-------|---------|--------|
| `detect_existing_model()` | ~100-130 | Checks model_dir for G_*.pth | Check training_dir |
| `GET /model/{expName}/info` | ~880-940 | Scans preprocess_dir for checkpoints | Scan training_dir |
| `GET /model/{expName}/checkpoints` | ~970-1020 | Scans preprocess_dir | Scan training_dir |
| `POST /extract_model` | ~2440-2510 | Searches model_dir then preprocess | Search training_dir |

**Changes needed**:
```python
# Define TRAINING_DIR = DATA_DIR / "training"
# All checkpoint scanning uses TRAINING_DIR / exp_name
# Only final model verification uses MODELS_DIR / exp_name
```

#### E. `services/voice-engine/app/trainer/jobs.py` (CheckpointManager)

| Method | Lines | Current | Target |
|--------|-------|---------|--------|
| `_scan_checkpoints()` | ~370-430 | Scans model_dir | Scan training_dir |
| `_cleanup_old_checkpoints()` | ~275-320 | Cleans from model_dir | Clean from training_dir |

#### F. `services/voice-engine/app/trainer/training_watchdogs.py`

| Method | Lines | Current | Target |
|--------|-------|---------|--------|
| `smoke_test()` | ~1080-1150 | Scans model_dir for checkpoints | Scan training_dir |

---

### 2. Trainer Service (Python)

#### `services/trainer/app/trainer/pipeline.py`

| Method | Lines | Current | Target |
|--------|-------|---------|--------|
| `extract_model()` | ~400-450 | Scans exp_dir for G_*.pth | Scan training subdirectory |

---

### 3. Laravel API (PHP)

#### A. `apps/api/app/Services/VoiceModelScanner.php`

**Current behavior**: `findLocalModelFile()` falls back to G_*.pth in model directory

| Method | Lines | Current | Target |
|--------|-------|---------|--------|
| `findLocalModelFile()` | 224-269 | Falls back to G_*.pth in model dir | Remove G_*.pth fallback for inference |
| `findLocalIndexFile()` | 276-300 | Searches model dir | Keep - final index is here |

**Changes needed**:
```php
// Remove Priority 3 (G_*.pth) from inference model lookup
// G_*.pth should NEVER be used for inference - only extracted models
// This was a dangerous fallback that could use incomplete checkpoints
```

#### B. `apps/api/app/Services/TrainingRunService.php`

**Current behavior**: Stores checkpoint paths pointing to models directory

| Method | Lines | Current | Target |
|--------|-------|---------|--------|
| `recordCheckpoint()` | 433-494 | Stores paths to model dir | Store paths to training dir |
| `startTrainingOnEngine()` | 380-425 | Sends model dir paths for resume | Send training dir paths |

**Changes needed**:
```php
// Checkpoint paths should reference /storage/data/training/{model}/
// Update checkpoint_directory, generator_path, discriminator_path
```

#### C. `apps/api/app/Services/TrainerService.php`

| Method | Lines | Current | Target |
|--------|-------|---------|--------|
| `handleTrainingCompleted()` | 377-413 | Assumes model at `/storage/models/` | Keep - final model location is correct |
| `getModelPath()` translation | 718-744 | Translates to `/app/assets/models/` | Update path translations |

#### D. `apps/api/config/storage_paths.php`

**Add new path definitions**:
```php
'training' => env('STORAGE_ROOT', '/storage') . '/data/training',
'preprocess' => env('STORAGE_ROOT', '/storage') . '/data/preprocess',
```

---

### 4. Frontend (TypeScript/React)

#### `apps/web/src/lib/api.ts`

| Function | Lines | Current | Target |
|----------|-------|---------|--------|
| `getModelCheckpoints()` | ~1380 | Returns checkpoints from API | No change needed - API handles paths |
| `getModelTrainingInfo()` | ~1350 | Returns training info | No change needed |

**No frontend changes required** - frontend only displays data from API, doesn't construct paths.

---

## Migration Steps

### Phase 1: Update Code (Non-Breaking)

1. **Add new path constants** in all services for TRAINING_DIR
2. **Update checkpoint write paths** to use training directory
3. **Update checkpoint read paths** to check training directory first
4. **Keep fallback reads** from models directory for existing data

### Phase 2: Data Migration Script

```bash
#!/bin/bash
# migrate-training-artifacts.sh

for model_dir in /storage/models/*/; do
    model_name=$(basename "$model_dir")
    training_dir="/storage/data/training/$model_name"
    
    mkdir -p "$training_dir"
    
    # Move checkpoint files
    mv "$model_dir"/G_*.pth "$training_dir/" 2>/dev/null
    mv "$model_dir"/D_*.pth "$training_dir/" 2>/dev/null
    
    # Move intermediate extracted models
    mv "$model_dir"/*_e*_s*.pth "$training_dir/" 2>/dev/null
    
    # Move training artifacts
    mv "$model_dir"/added_*.index "$training_dir/" 2>/dev/null
    mv "$model_dir"/.smoke_test_ready.json "$training_dir/" 2>/dev/null
    mv "$model_dir"/events.out.* "$training_dir/" 2>/dev/null
    
    # Move trainset if present
    if [ -d "$model_dir/trainset" ]; then
        preprocess_dir="/storage/data/preprocess/$model_name"
        mkdir -p "$preprocess_dir"
        mv "$model_dir/trainset" "$preprocess_dir/" 2>/dev/null
    fi
done
```

### Phase 3: Remove Fallbacks

After migration verified:
1. Remove fallback code that reads from models directory
2. Remove G_*.pth fallback in VoiceModelScanner.php
3. Update documentation

---

## Files to Modify (Prioritized)

### High Priority (Core Training Flow)

| File | Changes |
|------|---------|
| `services/voice-engine/app/trainer/pipeline.py` | Change checkpoint output directory |
| `services/voice-engine/app/trainer_api.py` | Update checkpoint scanning paths |
| `services/voice-engine/infer/modules/train/train.py` | Update model_dir handling |
| `services/voice-engine/infer/modules/train/extract/process_ckpt.py` | Separate intermediate vs final output |

### Medium Priority (API Layer)

| File | Changes |
|------|---------|
| `apps/api/app/Services/TrainingRunService.php` | Update checkpoint path storage |
| `apps/api/app/Services/VoiceModelScanner.php` | Remove G_*.pth fallback |
| `apps/api/config/storage_paths.php` | Add training/preprocess path configs |

### Low Priority (Cleanup & Validation)

| File | Changes |
|------|---------|
| `services/voice-engine/app/trainer/jobs.py` | Update CheckpointManager paths |
| `services/voice-engine/app/trainer/training_watchdogs.py` | Update smoke test paths |
| `services/trainer/app/trainer/pipeline.py` | Update extract_model paths |

---

## Testing Plan

1. **Unit Tests**:
   - Verify checkpoint writes go to training directory
   - Verify checkpoint reads find files in training directory
   - Verify final model export still goes to models directory

2. **Integration Tests**:
   - Start fresh training → checkpoints in training dir
   - Continue training → finds checkpoints in training dir
   - Complete training → final model in models dir only

3. **Migration Test**:
   - Run migration script on test data
   - Verify existing models still work
   - Verify training continuation works after migration

---

## Rollback Plan

If issues arise:
1. Keep migration script's reverse operation ready
2. Maintain fallback reads for 1 release cycle
3. Models directory paths still exist, just empty of checkpoints

---

## Estimated Impact

| Metric | Current | After Migration |
|--------|---------|-----------------|
| `/storage/models/` size per model | ~10GB | ~200MB |
| Checkpoint management complexity | High (mixed locations) | Low (single location) |
| Inference model clarity | Confusing (G_*.pth fallback) | Clear (only final .pth) |
| Backup size for production | ~10GB per model | ~200MB per model |
