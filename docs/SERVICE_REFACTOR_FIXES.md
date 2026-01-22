# Training Service Refactor - Fixes Applied

This document summarizes all the fixes applied to repair the training/preprocessing pipeline after the service refactor that split training and preprocessing out of `voice-engine` into separate services.

## Issues Identified and Fixed

### 1. ❌ Trainer `/upload` endpoint did not exist

**Problem:** `TrainerService.php::uploadTrainingAudio()` was calling `POST {$baseUrl}/upload` on the trainer service, but trainer only had `/start`, `/status`, `/stop`, `/validate` endpoints.

**Solution:** 
- Changed the API to upload training audio via the **preprocessor** service instead
- Added `/api/v1/preprocess/upload` endpoint to preprocessor
- The preprocessor manages the shared `/data/uploads` directory
- Modified `apps/api/app/Services/TrainerService.php` to use preprocessor URL for uploads

### 2. ❌ Path mismatch between preprocessor and trainer

**Problem:**
- Preprocessor was writing to `MODELS_DIR` = `/app/assets/models/<exp_name>`
- Trainer was validating from `DATA_ROOT` = `/data/<exp_name>`
- These are completely different paths!

**Solution:**
- Unified paths: Both services now use `DATA_ROOT=/data` for experiment outputs
- Changed preprocessor config (`services/preprocessor/app/config.py`):
  - `DATA_ROOT` for experiment outputs (preprocessing artifacts)
  - `UPLOADS_DIR` for uploaded training audio
  - `ASSETS_ROOT` for read-only shared assets
- Both services use the same `training_data` Docker volume mounted at `/data`

### 3. ❌ Read-only assets mount crash

**Problem:** Preprocessor tried to create directories inside `/app/assets/models` but assets were mounted as `:ro` (read-only) in Docker Compose.

**Solution:**
- Changed preprocessor to never write to `/app/assets` (read-only)
- All writable outputs go to `DATA_ROOT=/data` (writable volume)
- Updated `Settings.__post_init__()` to only create directories in writable paths
- Split asset mounts to be granular:
  - `/app/assets/hubert:ro` - HuBERT model (read-only)
  - `/app/assets/rmvpe:ro` - RMVPE model (read-only)
  - `/app/rvc:ro` - RVC code (read-only)
  - `/data` - writable volume for outputs

### 4. ❌ Preprocessor missing RVC module import

**Problem:** `services/preprocessor/app/f0_extract.py` imports `from rvc.lib.rmvpe import RMVPE` but the `rvc` module wasn't available in preprocessor container.

**Solution:**
- Added RVC code mount to preprocessor in both dev and prod compose:
  ```yaml
  volumes:
    - ../../services/voice-engine/rvc:/app/rvc:ro
  ```
- For local development, added `PYTHONPATH` to include voice-engine directory

### 5. ❌ Inconsistent environment variables

**Problem:** Preprocessor config used `MODELS_DIR` and `UPLOADS_DIR`, but compose set `DATA_ROOT` and `ASSETS_ROOT`, leading to path mismatches.

**Solution:**
- Standardized environment variables across all services:
  - `DATA_ROOT` - Writable directory for preprocessing/training data
  - `UPLOADS_DIR` - Where uploaded audio goes (inside DATA_ROOT)
  - `ASSETS_ROOT` - Read-only shared assets directory
  - `MODELS_ROOT` - Where final trained models are saved (trainer only)
  - `HUBERT_PATH` - Explicit path to HuBERT model
  - `RMVPE_PATH` - Explicit path to RMVPE directory

### 6. ❌ Missing Docker socket in dev compose

**Problem:** Production compose mounted Docker socket for admin panel container logs, but dev compose didn't, causing feature parity issues.

**Solution:**
- Added Docker socket mount to API container in dev compose:
  ```yaml
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
    - /var/lib/docker/containers:/var/lib/docker/containers:ro
  ```

### 7. ❌ Model path handling in API

**Problem:** `TrainerService.php::handleTrainingCompleted()` used absolute paths like `/app/assets/models/{exp_name}/{exp_name}.pth` which didn't match the API's scanning path.

**Solution:**
- Changed to use relative paths that work with the API's `VOICE_MODELS_PATH` mount:
  ```php
  $modelPath = "{$expName}/{$expName}.pth";
  ```

---

## Files Modified

### Services

| File | Changes |
|------|---------|
| `services/preprocessor/app/config.py` | Complete rewrite - unified path configuration with DATA_ROOT, UPLOADS_DIR, ASSETS_ROOT |
| `services/preprocessor/app/api.py` | Added `/upload` endpoint for training audio files |
| `services/preprocessor/main.py` | Updated startup logging to show new path variables |
| `services/start-services.sh` | Updated with unified path configuration, added asset checks, PYTHONPATH for RVC |

### API

| File | Changes |
|------|---------|
| `apps/api/app/Services/TrainerService.php` | Changed `uploadTrainingAudio()` to use preprocessor service; fixed model path handling |

### Docker Compose

| File | Changes |
|------|---------|
| `infra/compose/docker-compose.yml` | Fixed preprocessor/trainer mounts; added RVC mount to preprocessor; added Docker socket to API |
| `infra/compose/docker-compose.prod.yml` | Same fixes for production |

### Scripts

| File | Changes |
|------|---------|
| `scripts/dev-up.sh` | NEW - Conditional asset download + Docker start script |
| `scripts/service-up.sh` | NEW - Individual service starter with dependencies |

### Documentation

| File | Changes |
|------|---------|
| `docs/DEVELOPMENT.md` | NEW - Comprehensive development guide |
| `README.md` | Updated with new scripts and workflow |

---

## Data Flow (After Fix)

```
1. Upload Training Audio
   User → API → Preprocessor POST /api/v1/preprocess/upload
   Files saved to: /data/uploads/{exp_name}/

2. Start Preprocessing  
   API → Preprocessor POST /api/v1/preprocess/start
   Reads from: /data/uploads/{exp_name}/
   Writes to:  /data/{exp_name}/0_gt_wavs/
               /data/{exp_name}/1_16k_wavs/
               /data/{exp_name}/2a_f0/
               /data/{exp_name}/2b_f0nsf/
               /data/{exp_name}/3_feature768/

3. Start Training
   API → Trainer POST /api/v1/trainer/start
   Validates: /data/{exp_name}/ (same volume as preprocessor!)
   Trains using: /app/rvc/infer/modules/train/train.py
   Saves to: /models/{exp_name}/{exp_name}.pth
             /models/{exp_name}/{exp_name}.index

4. Model Available for Inference
   Voice Engine reads from: /app/assets/models/{exp_name}/
   (Same directory as trainer's /models via bind mount)
```

---

## Testing Checklist

### Docker Dev
- [ ] `./scripts/dev-up.sh` completes successfully
- [ ] All services start: `docker compose ps`
- [ ] Upload training audio via web UI
- [ ] Start preprocessing → job completes
- [ ] Start training → job completes
- [ ] Model appears in voice models list
- [ ] Model can be used for inference

### Docker Prod
- [ ] `./scripts/dev-up.sh --prod` completes
- [ ] Same workflow as dev
- [ ] HTTPS routing works

### Local (No Docker)
- [ ] `./scripts/dev-up.sh --no-docker` downloads assets
- [ ] `./services/start-services.sh preprocess` starts
- [ ] `./services/start-services.sh trainer` starts
- [ ] Can run full training workflow

---

## Quick Validation Commands

```bash
# Check preprocessor health
curl http://localhost:8003/health

# Check trainer health
curl http://localhost:8002/health

# Upload test audio
curl -X POST http://localhost:8003/api/v1/preprocess/upload \
  -F "exp_name=test_model" \
  -F "files=@sample.wav"

# Start preprocessing
curl -X POST http://localhost:8003/api/v1/preprocess/start \
  -H "Content-Type: application/json" \
  -d '{"exp_name": "test_model"}'

# Check preprocessing status
curl http://localhost:8003/api/v1/preprocess/status/{job_id}

# Validate preprocessing
curl http://localhost:8003/api/v1/preprocess/validate/test_model

# Start training
curl -X POST http://localhost:8002/api/v1/trainer/start \
  -H "Content-Type: application/json" \
  -d '{"exp_name": "test_model", "epochs": 10}'

# Check training status
curl http://localhost:8002/api/v1/trainer/status/{job_id}
```
