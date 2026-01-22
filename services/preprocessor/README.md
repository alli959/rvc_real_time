# Preprocessor Service

Audio preprocessing service for RVC voice model training.

## Overview

This service handles all preprocessing steps required before training an RVC model:
- Audio slicing (silence detection)
- Normalization and filtering  
- Resampling (target SR + 16kHz for features)
- F0 extraction (RMVPE)
- Feature extraction (HuBERT)

## API Endpoints

### Start Preprocessing
```
POST /api/v1/preprocess/start
{
  "exp_name": "model_name",
  "input_dir": "/app/uploads/model_name",
  "sample_rate": 48000,
  "version": "v2",
  "n_threads": 4
}
```

### Get Status
```
GET /api/v1/preprocess/status/{job_id}
```

### Validate Preprocessing
```
GET /api/v1/preprocess/validate/{exp_name}
```

## Directory Structure

After preprocessing, the experiment directory will contain:
```
{models_dir}/{exp_name}/
├── 0_gt_wavs/          # Ground truth WAVs at target SR
├── 1_16k_wavs/         # Resampled to 16kHz for HuBERT
├── 2a_f0/              # Coarse F0 contours
├── 2b_f0nsf/           # Fine F0 contours
├── 3_feature768/       # HuBERT features (v2)
├── preprocess.log      # Preprocessing log
└── config.json         # Preprocessing config
```

## Running Locally

```bash
cd services/preprocessor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Running with Docker

```bash
docker build -t morphvox-preprocessor .
docker run -p 8003:8003 \
  -v /path/to/assets:/app/assets \
  -v /path/to/uploads:/app/uploads \
  morphvox-preprocessor
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HTTP_PORT | 8003 | HTTP API port |
| MODELS_DIR | /app/assets/models | Output directory for preprocessed data |
| UPLOADS_DIR | /app/uploads | Input directory for raw audio |
| HUBERT_PATH | /app/assets/hubert/hubert_base.pt | HuBERT model path |
| RMVPE_PATH | /app/assets/rmvpe | RMVPE model directory |
| DEVICE | cuda:0 | PyTorch device |
| LOG_LEVEL | INFO | Logging level |
