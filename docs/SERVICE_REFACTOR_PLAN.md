# MorphVox Service Refactor Plan

## Implementation Status: ✅ COMPLETED

This document describes the service refactoring that separates the monolithic voice-engine into three distinct services.

## Final Architecture

### Directory Structure
```
services/
├── preprocessor/        # ✅ CREATED - Audio preprocessing + feature extraction
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   ├── README.md
│   └── app/
│       ├── api.py           # FastAPI routes
│       ├── config.py        # Settings (WebUI-matching parameters)
│       ├── slicer.py        # WebUI-matching slicer
│       ├── audio.py         # Audio utilities
│       ├── preprocess.py    # Main preprocessing pipeline
│       ├── f0_extract.py    # RMVPE F0 extraction
│       └── feature_extract.py  # HuBERT feature extraction
│
├── trainer/             # ✅ CREATED - RVC model training
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   ├── README.md
│   └── app/
│       ├── api.py           # FastAPI routes
│       ├── config.py        # Training settings
│       ├── jobs.py          # Job management
│       └── training.py      # Training execution
│
└── voice-engine/        # ✅ UPDATED - Inference only
    ├── app/             # Inference code (stays)
    ├── rvc/             # RVC model code (shared)
    └── infer/           # Inference code from WebUI (shared)
```

### Ports & Services
| Service | HTTP Port | WS Port | GPU | shm_size |
|---------|-----------|---------|-----|----------|
| voice-engine | 8001 | 8765 | Yes | 4gb (dev) / 8gb (prod) |
| trainer | 8002 | - | Yes | 8gb |
| preprocess | 8003 | - | Yes | 4gb (dev) / 8gb (prod) |
| api | 8080 | - | No | - |
| web | 3000 | - | No | - |

---

## Service Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL                                       │
│                     (Users via Nginx/API)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Laravel API (Port 8080)                          │
│                    (Orchestration + Business Logic)                     │
│  - Calls preprocess service for dataset preparation                     │
│  - Calls trainer service for training jobs                              │
│  - Calls voice-engine for inference                                     │
└─────────────────────────────────────────────────────────────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐
│   PREPROCESS    │  │    TRAINER      │  │       VOICE-ENGINE          │
│   (Port 8003)   │  │   (Port 8002)   │  │   (HTTP 8001, WS 8765)      │
│                 │  │                 │  │                             │
│ - Audio slicing │  │ - Training jobs │  │ - Inference only            │
│ - Resampling    │  │ - Progress/logs │  │ - Real-time conversion (WS) │
│ - Normalization │  │ - Checkpoints   │  │ - Model loading             │
│ - F0 extraction │  │ - Model extract │  │ - TTS                       │
│ - Feature extr. │  │ - Index build   │  │                             │
│                 │  │                 │  │                             │
│ GPU: Optional   │  │ GPU: Required   │  │ GPU: Required               │
│ shm: 4gb        │  │ shm: 8gb        │  │ shm: 4gb                    │
└─────────────────┘  └─────────────────┘  └─────────────────────────────┘
         │                      │                      │
         └──────────────────────┴──────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Shared Volumes      │
                    │ - /assets/models      │
                    │ - /assets/hubert      │
                    │ - /assets/rmvpe       │
                    │ - /assets/pretrained_v2│
                    │ - /data (training)    │
                    └───────────────────────┘
```

---

## API Endpoints

### Preprocessor Service (Port 8003)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/preprocess/start` | Start preprocessing job |
| GET | `/api/v1/preprocess/status/{job_id}` | Get job status |
| GET | `/api/v1/preprocess/validate/{exp_name}` | Validate preprocessing outputs |
| GET | `/api/v1/preprocess/jobs` | List all jobs |

### Trainer Service (Port 8002)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/trainer/start` | Start training job |
| GET | `/api/v1/trainer/status/{job_id}` | Get job status |
| POST | `/api/v1/trainer/stop/{job_id}` | Stop training job |
| GET | `/api/v1/trainer/jobs` | List all jobs |
| POST | `/api/v1/trainer/extract-model` | Extract inference model |
| POST | `/api/v1/trainer/build-index` | Build FAISS index |
| GET | `/api/v1/trainer/validate/{exp_name}` | Validate preprocessing |

### Voice Engine Service (Port 8001, WS 8765)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/convert` | Single-shot voice conversion |
| POST | `/api/v1/tts/generate` | TTS generation |
| GET | `/api/v1/models` | List loaded models |
| POST | `/api/v1/models/load` | Load a model |
| WS | `ws://:8765` | Real-time voice conversion |

---

## Preprocessing Pipeline

### Output Directory Structure (per experiment)
```
{data_root}/{exp_name}/
├── 0_gt_wavs/          # Ground truth WAVs at target SR (48kHz)
├── 1_16k_wavs/         # Resampled to 16kHz for HuBERT
├── 2a_f0/              # Coarse F0 contours (quantized)
├── 2b_f0nsf/           # Fine F0 contours (for NSF vocoder)
├── 3_feature768/       # HuBERT features (v2 = 768-dim)
├── config.json         # Training config
├── filelist.txt        # Training file list
└── total_fea.npy       # Concatenated features (for index)
```

### Key Parameters (matching WebUI)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 48000 | Target sample rate |
| Slicer Threshold | -42 dB | Silence detection threshold |
| Min Length | 1500 ms | Minimum chunk length |
| Min Interval | 400 ms | Minimum silence between chunks |
| Hop Size | 15 ms | Slicer analysis hop |
| Max Silence Kept | 500 ms | Max silence in chunk |
| Chunk Length | 3.7 s | Audio chunk duration |
| Chunk Overlap | 0.3 s | Overlap between chunks |
| Normalization Max | 0.9 | Peak normalization target |
| Alpha | 0.75 | RMS smoothing alpha |
| High-pass Filter | 48 Hz | Butter N=5 high-pass |

---

## Laravel Integration

### New Service Classes
- `app/Services/PreprocessorService.php` - Preprocessor API client
- `app/Services/TrainerService.php` - Updated to use new trainer service

### Config (config/services.php)
```php
'trainer' => [
    'url' => env('TRAINER_URL', 'http://trainer:8002'),
    'timeout' => env('TRAINER_TIMEOUT', 600),
],

'preprocessor' => [
    'url' => env('PREPROCESSOR_URL', 'http://preprocess:8003'),
    'timeout' => env('PREPROCESSOR_TIMEOUT', 300),
],
```

---

## Startup Scripts

### Development (Docker Compose)
```bash
# Start all services
./scripts/run-local-fresh.sh

# Start with rebuild
./scripts/run-local-fresh.sh --rebuild

# Clean start (wipe volumes)
./scripts/run-local-fresh.sh --clean
```

### Individual Services (no Docker)
```bash
# Start individual services for debugging
./services/start-services.sh voice-engine
./services/start-services.sh trainer
./services/start-services.sh preprocess
./services/start-services.sh all
```

---

## Docker Compose Updates

Both `docker-compose.yml` (dev) and `docker-compose.prod.yml` (prod) have been updated to include:

1. **preprocess** service with:
   - Port 8003
   - Shared assets (read-only)
   - Training data volume
   - GPU access

2. **trainer** service with:
   - Port 8002
   - RVC code mounted from voice-engine (read-only)
   - Training data volume
   - Models output directory
   - GPU access

3. **training_data** volume for shared preprocessing outputs

4. Updated API environment variables:
   - `TRAINER_URL=http://trainer:8002`
   - `PREPROCESSOR_URL=http://preprocess:8003`

---

## Training Workflow

1. **Upload Audio** → Laravel API receives audio files
2. **Start Preprocessing** → API calls preprocessor service
   - Slicing by silence
   - Resampling to target SR + 16kHz
   - F0 extraction (RMVPE)
   - HuBERT feature extraction
3. **Validate** → API calls preprocessor to validate outputs
4. **Start Training** → API calls trainer service
   - Trainer validates preprocessing exists
   - Generates filelist.txt and config.json
   - Launches training subprocess
   - Monitors progress
5. **Extract Model** → Trainer extracts inference model from checkpoint
6. **Build Index** → Trainer builds FAISS index
7. **Model Ready** → API can load model in voice-engine for inference
