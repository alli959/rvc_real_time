# MorphVox Platform - Local & Docker Development Guide

## Quick Start

### Docker Development (Recommended)

```bash
# Complete setup + start development environment
./scripts/dev-up.sh

# Start production environment  
./scripts/dev-up.sh --prod

# Setup assets only (no Docker)
./scripts/dev-up.sh --no-docker

# Rebuild images before starting
./scripts/dev-up.sh --rebuild
```

### Start Individual Services

```bash
# Start specific service with dependencies
./scripts/service-up.sh trainer

# Start infrastructure only (db, redis, minio)
./scripts/service-up.sh infra -d

# Available services:
# api, web, voice-engine, trainer, preprocess, infra
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MorphVox Platform                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Web UI     │  │  Laravel API │  │    Voice Engine        │ │
│  │  (Next.js)   │──│   (:8080)    │──│   Inference (:8001)    │ │
│  │   (:3000)    │  │              │  │   WebSocket (:8765)    │ │
│  └──────────────┘  └──────┬───────┘  └────────────────────────┘ │
│                           │                       ▲              │
│                    ┌──────┴───────┐               │              │
│                    ▼              ▼               │              │
│  ┌────────────────────┐  ┌────────────────────┐  │              │
│  │   Preprocessor     │  │     Trainer        │  │              │
│  │   (:8003)          │  │     (:8002)        │──┘              │
│  │   - Audio slicing  │  │   - RVC Training   │                 │
│  │   - F0 extraction  │  │   - Model export   │                 │
│  │   - HuBERT features│  │   - FAISS index    │                 │
│  └─────────┬──────────┘  └─────────┬──────────┘                 │
│            │                       │                             │
│            └───────────┬───────────┘                             │
│                        ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Shared Volumes                             ││
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐   ││
│  │  │  training_data  │  │         assets/models            │   ││
│  │  │  /data          │  │  voice-engine/assets/models      │   ││
│  │  │  - uploads/     │  │  - Final .pth files              │   ││
│  │  │  - {exp}/...    │  │  - FAISS .index files            │   ││
│  │  └─────────────────┘  └─────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Path Configuration (Critical)

### Shared Volumes

| Path | Container Mount | Purpose |
|------|-----------------|---------|
| `training_data` volume | `/data` | Preprocessing outputs + training inputs |
| `voice-engine/assets/models` | `/models` (trainer), `/var/www/html/storage/models` (API) | Final trained models |
| `voice-engine/assets/` | `/app/assets` (read-only) | Shared assets (HuBERT, RMVPE, pretrained) |

### Preprocessor Paths

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATA_ROOT` | `/data` | Where preprocessing outputs go |
| `UPLOADS_DIR` | `/data/uploads` | Uploaded training audio |
| `ASSETS_ROOT` | `/app/assets` | Read-only shared assets |
| `HUBERT_PATH` | `/app/assets/hubert/hubert_base.pt` | HuBERT model |
| `RMVPE_PATH` | `/app/assets/rmvpe` | RMVPE model directory |

### Trainer Paths

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATA_ROOT` | `/data` | Reads preprocessing outputs from here |
| `MODELS_ROOT` | `/models` | Saves final models here |
| `RVC_ROOT` | `/app/rvc` | RVC training code |
| `ASSETS_ROOT` | `/app/assets` | Pretrained models |

### Data Flow

```
1. User uploads audio → API → Preprocessor (/data/uploads/{exp_name}/)
2. Preprocessing runs → Creates /data/{exp_name}/0_gt_wavs/, 1_16k_wavs/, etc.
3. Training starts → Reads from /data/{exp_name}/, validates preprocessing
4. Training completes → Model saved to /models/{exp_name}/{exp_name}.pth
5. Voice Engine → Loads model from /app/assets/models/{exp_name}/...
```

---

## Local Development (No Docker)

### Prerequisites

1. Python 3.10+
2. CUDA GPU (optional, for acceleration)
3. Redis running locally or via Docker
4. MinIO running locally or via Docker

### Setup

```bash
# 1. Download assets (conditional - skips existing)
./scripts/dev-up.sh --no-docker

# 2. Start infrastructure services via Docker
./scripts/service-up.sh infra -d

# 3. Create Python virtual environments for each service
cd services/voice-engine && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd ../preprocessor && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd ../trainer && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Running Services Locally

```bash
# Option 1: Use the start script
cd services
./start-services.sh preprocess   # Start preprocessor
./start-services.sh trainer      # Start trainer
./start-services.sh voice-engine # Start voice engine
./start-services.sh all          # Start all (background)

# Option 2: Run manually
cd services/preprocessor
source .venv/bin/activate
export DATA_ROOT=/path/to/rvc_real_time/data
export ASSETS_ROOT=/path/to/rvc_real_time/services/voice-engine/assets
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

### Local Environment Variables

```bash
# Required for all services
export S3_ENDPOINT=http://localhost:9000
export S3_ACCESS_KEY=minioadmin
export S3_SECRET_KEY=minioadmin
export S3_BUCKET=morphvox
export DEVICE=cuda  # or 'cpu'

# Service-specific
export DATA_ROOT=/absolute/path/to/rvc_real_time/data
export ASSETS_ROOT=/absolute/path/to/rvc_real_time/services/voice-engine/assets
```

---

## Training Workflow

### 1. Upload Training Audio

```bash
# Via API (Laravel forwards to preprocessor)
curl -X POST http://localhost:8080/api/v1/training/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "exp_name=my_voice" \
  -F "files[]=@sample1.wav" \
  -F "files[]=@sample2.wav"
```

### 2. Start Preprocessing

```bash
curl -X POST http://localhost:8003/api/v1/preprocess/start \
  -H "Content-Type: application/json" \
  -d '{"exp_name": "my_voice", "sample_rate": 48000, "version": "v2"}'
```

### 3. Start Training

```bash
curl -X POST http://localhost:8002/api/v1/trainer/start \
  -H "Content-Type: application/json" \
  -d '{"exp_name": "my_voice", "epochs": 100, "batch_size": 8}'
```

### 4. Check Status

```bash
# Preprocessing status
curl http://localhost:8003/api/v1/preprocess/status/{job_id}

# Training status
curl http://localhost:8002/api/v1/trainer/status/{job_id}

# Validate preprocessing
curl http://localhost:8003/api/v1/preprocess/validate/my_voice
```

---

## Troubleshooting

### Common Issues

#### "HuBERT model not found"
```bash
./scripts/dev-up.sh --no-docker  # Downloads missing assets
```

#### "RMVPE import error" (Preprocessor)
The preprocessor needs access to the RVC code for RMVPE. Ensure:
- Docker: `/app/rvc` is mounted from `voice-engine/rvc`
- Local: Add `PYTHONPATH=$PROJECT_ROOT/services/voice-engine`

#### "Experiment directory not found" (Trainer)
Preprocessing outputs go to `DATA_ROOT/{exp_name}/`. Check:
- Both services use the same `DATA_ROOT`
- Docker: Both use the `training_data` volume
- Local: Both use the same absolute path

#### "Permission denied" writing to assets
Assets are mounted read-only. Preprocessing should write to `DATA_ROOT`, not `ASSETS_ROOT`.

#### Training produces broken audio / very high loss values
If `loss_disc` shows values in the billions instead of 1-10, the audio normalization may be disabled. Check `data_utils.py`:
```python
# This line MUST be uncommented:
audio_norm = audio / self.max_wav_value
```
After fixing, delete all `.spec.pt` cache files and restart training.

#### File uploads getting stuck / timeout
Large multi-file uploads are batched (4 files at a time). If uploads still timeout:
- Check PHP memory_limit (should be 1024M+)
- Check upload_max_filesize and post_max_size in php.ini
- Verify PHP-FPM has enough workers (pm.max_children ≥ 20)

#### Site freezes during training
PHP-FPM may be exhausted. Check `apps/api/docker/www.conf` has sufficient workers:
```ini
pm.max_children = 50
pm.start_servers = 10
```

### Health Checks

```bash
# All services
curl http://localhost:8001/health  # Voice Engine
curl http://localhost:8002/health  # Trainer
curl http://localhost:8003/health  # Preprocessor
curl http://localhost:8080/api/health  # API
```

### Logs

```bash
# Docker
docker compose -f infra/compose/docker-compose.yml logs -f preprocess
docker compose -f infra/compose/docker-compose.yml logs -f trainer

# Local (when using start-services.sh all)
tail -f logs/preprocess.log
tail -f logs/trainer.log
```

---

## Service Ports

| Service | Port | Protocol |
|---------|------|----------|
| API | 8080 | HTTP |
| Web UI | 3000 | HTTP |
| Voice Engine | 8001 | HTTP |
| Voice Engine WS | 8765 | WebSocket |
| Trainer | 8002 | HTTP |
| Preprocessor | 8003 | HTTP |
| MinIO API | 9000 | HTTP |
| MinIO Console | 9001 | HTTP |
| MariaDB | 3306 | MySQL |
| Redis | 6379 | Redis |
