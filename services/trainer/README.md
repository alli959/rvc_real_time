# Trainer Service

Training orchestration service for RVC voice models.

## Overview

This service handles the training workflow for RVC models:
- Training job management (create, monitor, stop)
- Model extraction from checkpoints
- Index building (FAISS)
- Progress tracking and logging

## Prerequisites

Before training, the preprocessor service must have created:
- `{models_dir}/{exp_name}/0_gt_wavs/`
- `{models_dir}/{exp_name}/1_16k_wavs/`
- `{models_dir}/{exp_name}/2a_f0/`
- `{models_dir}/{exp_name}/2b_f0nsf/`
- `{models_dir}/{exp_name}/3_feature768/` (or 3_feature256 for v1)

## API Endpoints

### Start Training
```
POST /api/v1/trainer/start
{
  "exp_name": "model_name",
  "sample_rate": 48000,
  "version": "v2",
  "epochs": 150,
  "batch_size": 4,
  "save_every_epoch": 10,
  "gpus": "0"
}
```

### Get Training Status
```
GET /api/v1/trainer/status/{job_id}
```

### Stop Training
```
POST /api/v1/trainer/stop/{job_id}
```

### Extract Model
```
POST /api/v1/trainer/extract-model
{
  "exp_name": "model_name",
  "checkpoint": "G_3640.pth"  // optional
}
```

### Build Index
```
POST /api/v1/trainer/build-index
{
  "exp_name": "model_name"
}
```

## Running Locally

```bash
cd services/trainer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Running with Docker

```bash
docker build -t morphvox-trainer .
docker run -p 8002:8002 \
  --gpus all \
  --shm-size=8gb \
  -v /path/to/assets:/app/assets \
  morphvox-trainer
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HTTP_PORT | 8002 | HTTP API port |
| MODELS_DIR | /app/assets/models | Models directory |
| DEVICE | cuda:0 | PyTorch device |
| LOG_LEVEL | INFO | Logging level |

## Training Process

1. Verify preprocessing outputs exist
2. Generate training configuration
3. Create filelist.txt
4. Launch training subprocess
5. Monitor progress via log parsing
6. Save checkpoints periodically
7. Extract final model
8. Build FAISS index
