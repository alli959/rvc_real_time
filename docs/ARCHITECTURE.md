# MorphVox Platform Architecture

## Overview

MorphVox is a comprehensive AI voice conversion platform built as a monorepo with the following components:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              MorphVox Platform                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────────────────────────────┐ │
│  │   Next.js   │───▶│   Laravel   │───▶│         Python Services            │ │
│  │   WebUI     │    │    API      │    │                                    │ │
│  │  (port 3000)│◀───│ (port 8080) │◀───│  ┌──────────────────────────────┐ │ │
│  └─────────────┘    └──────┬──────┘    │  │   Voice Engine (Inference)   │ │ │
│                            │           │  │   HTTP: 8001  |  WS: 8765    │ │ │
│                            │           │  └──────────────────────────────┘ │ │
│                            ▼           │                                    │ │
│                     ┌──────────────┐   │  ┌──────────────────────────────┐ │ │
│                     │   MariaDB    │   │  │   Preprocessor Service       │ │ │
│                     │    (3306)    │   │  │   HTTP: 8003                 │ │ │
│                     └──────────────┘   │  │   (F0, HuBERT, slicing)      │ │ │
│                            │           │  └──────────────────────────────┘ │ │
│                     ┌──────────────┐   │                                    │ │
│                     │    Redis     │   │  ┌──────────────────────────────┐ │ │
│                     │    (6379)    │   │  │      Trainer Service         │ │ │
│                     └──────────────┘   │  │   HTTP: 8002                 │ │ │
│                            │           │  │   (RVC training, indexing)   │ │ │
│                     ┌──────────────┐   │  └──────────────────────────────┘ │ │
│                     │    MinIO     │◀──┴────────────────────────────────────┘ │
│                     │  S3 Storage  │                                          │
│                     │ (9000/9001)  │                                          │
│                     └──────────────┘                                          │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
morphvox/
├── apps/
│   ├── api/                    # Laravel 11 Backend
│   │   ├── app/
│   │   │   ├── Http/Controllers/Api/
│   │   │   ├── Models/
│   │   │   ├── Policies/
│   │   │   ├── Jobs/
│   │   │   └── Services/
│   │   ├── database/migrations/
│   │   ├── routes/api.php
│   │   └── docker/
│   │
│   └── web/                    # Next.js 14 Frontend
│       ├── src/
│       │   ├── app/
│       │   ├── components/
│       │   └── lib/
│       └── Dockerfile
│
├── services/
│   ├── voice-engine/           # RVC Inference Service (HTTP: 8001, WS: 8765)
│   │   ├── app/
│   │   │   ├── core/           # Config, logging, exceptions
│   │   │   ├── models/         # Pydantic request/response schemas
│   │   │   ├── services/       # Business logic layer
│   │   │   │   ├── voice_conversion/
│   │   │   │   ├── audio_analysis/
│   │   │   │   ├── tts/
│   │   │   │   └── youtube/
│   │   │   └── routers/        # FastAPI route handlers
│   │   ├── rvc/                # RVC pipeline (vendored)
│   │   ├── assets/             # Models, weights
│   │   └── main.py
│   │
│   ├── preprocessor/           # Audio Preprocessing Service (HTTP: 8003)
│   │   ├── app/
│   │   │   ├── api/            # FastAPI routers
│   │   │   ├── pipeline/       # Preprocessing stages
│   │   │   │   ├── audio_slicer.py
│   │   │   │   ├── f0_extractor.py
│   │   │   │   └── feature_extractor.py
│   │   │   └── core/           # Config, exceptions
│   │   └── main.py
│   │
│   └── trainer/                # Model Training Service (HTTP: 8002)
│       ├── app/
│       │   ├── api/            # FastAPI routers
│       │   ├── pipeline/       # Training pipeline
│       │   │   ├── training.py
│       │   │   ├── model_extractor.py
│       │   │   └── index_builder.py
│       │   └── core/           # Config, exceptions
│       └── main.py
│
├── packages/
│   ├── sdk-js/                 # JavaScript/TypeScript SDK
│   ├── sdk-python/             # Python SDK
│   └── shared/                 # OpenAPI schemas
│
├── infra/
│   └── compose/
│       ├── docker-compose.yml
│       └── .env.example
│
└── docs/
    └── ARCHITECTURE.md
```

## Components

### 1. Laravel API (`apps/api`)

The API serves as the central hub for:
- **Authentication**: User registration, login, token management (Laravel Sanctum)
- **Authorization**: Role-based permissions (Spatie Permission)
- **Model Registry**: CRUD operations for voice models
- **Job Queue**: Async processing with Redis queue
- **Usage Tracking**: Events for billing/analytics

**Key Endpoints:**
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/models` - List public models
- `POST /api/models` - Create new model
- `POST /api/jobs/inference` - Create voice conversion job

### 2. Next.js WebUI (`apps/web`)

Modern React frontend with:
- **Landing Page**: Marketing/info about the platform
- **Model Browser**: Search and discover voice models
- **Upload Interface**: Drag-and-drop model uploads
- **Voice Studio**: Audio upload and conversion UI
- **Dashboard**: User's models, jobs, usage stats

**Tech Stack:**
- Next.js 14 with App Router
- TailwindCSS for styling
- TanStack Query for data fetching
- Zustand for state management

### 3. Voice Engine (`services/voice-engine`)

Python service for voice conversion and audio processing (inference only):

**API Endpoints:**
- **HTTP API** (port 8001): File-based inference
- **WebSocket** (port 8765): Real-time streaming

**Modular Architecture:**
```
app/
├── core/           # Infrastructure (config, logging, exceptions)
├── models/         # Pydantic schemas (TTS, conversion, audio, youtube)
├── services/       # Business logic
│   ├── voice_conversion/  # RVC model management
│   ├── audio_analysis/    # Voice detection, vocal separation
│   ├── tts/               # Bark TTS, Edge TTS, audio effects
│   └── youtube/           # YouTube search/download with caching
└── routers/        # FastAPI HTTP handlers
```

**Supported Features:**
- RVC voice conversion with WebUI model compatibility
- Multiple F0 methods (rmvpe, pm, harvest, dio)
- Index-based similarity search
- Bark TTS (neural, emotion support)
- Edge TTS (50+ voices)
- UVR5 vocal separation (HP3, HP5 models)
- Voice detection (speaker count)
- Audio effects (reverb, echo, pitch shift)
- YouTube audio download

### 4. Preprocessor Service (`services/preprocessor`)

Dedicated Python service for audio preprocessing before training:

**API Endpoints:**
- **HTTP API** (port 8003): Preprocessing jobs

**Features:**
- Audio slicing with silence detection
- Audio normalization and filtering
- Resampling (target SR + 16kHz for HuBERT)
- F0 extraction using RMVPE
- HuBERT feature extraction
- Progress tracking via WebSocket

**Output Structure:**
```
{data_root}/{exp_name}/
├── 0_gt_wavs/        # Ground truth WAVs at target SR
├── 1_16k_wavs/       # Resampled to 16kHz for HuBERT
├── 2a_f0/            # Coarse F0 contours
├── 2b_f0nsf/         # Fine F0 contours
├── 3_feature768/     # HuBERT features (v2)
└── config.json       # Preprocessing config
```

### 5. Trainer Service (`services/trainer`)

Dedicated Python service for RVC model training:

**API Endpoints:**
- **HTTP API** (port 8002): Training job management

**Features:**
- Training job orchestration (start, stop, resume)
- Model extraction from checkpoints
- FAISS index building
- Progress tracking and logging
- GPU resource management

**Workflow:**
1. Verify preprocessor outputs exist
2. Generate training configuration
3. Create filelist.txt
4. Launch training subprocess
5. Extract final model and build index

### 6. Storage (MinIO)

S3-compatible object storage for:
- Voice model files (`.pth`, `.index`)
- Audio inputs and outputs
- User uploads
- Training data and checkpoints

**Bucket Structure:**
```
morphvox/
├── models/{user_id}/{model_id}/
│   ├── model.pth
│   ├── model.index
│   └── config.json
├── jobs/{user_id}/{job_id}/
│   ├── input.wav
│   └── output.wav
├── training/{exp_name}/
│   ├── checkpoints/
│   └── logs/
└── public/
    └── thumbnails/
```

## Data Flow

### Voice Conversion Flow

```
1. User uploads audio file
   ├── WebUI → S3 pre-signed URL upload
   └── Creates Job in Laravel

2. Job queued for processing
   ├── Laravel pushes to Redis queue
   └── Worker picks up job

3. Voice Engine processes audio
   ├── Downloads input from S3
   ├── Runs RVC inference
   └── Uploads output to S3

4. User downloads result
   └── Gets pre-signed URL from API
```

### Model Upload Flow

```
1. User initiates upload
   ├── API creates model record (status: pending)
   └── Returns pre-signed upload URLs

2. Direct S3 upload
   ├── Frontend uploads .pth to model URL
   └── Frontend uploads .index to index URL

3. Confirm upload
   ├── API validates files exist
   └── Updates status to 'ready'
```

## User Roles & Permissions

| Role | Permissions |
|------|-------------|
| guest | View public models |
| user | + Create jobs, view own jobs |
| premium | + Upload models, manage own models |
| creator | + Train models |
| moderator | + Manage all models/jobs, view users |
| admin | All permissions |

## Getting Started

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (optional, for faster inference)

### Quick Start

```bash
# 1. Clone the repository
cd /path/to/voiceforge

# 2. Copy environment file
cp infra/compose/.env.example infra/compose/.env

# 3. Start all services
cd infra/compose
docker compose up -d

# 4. Run Laravel migrations
docker compose exec api php artisan migrate --seed

# 5. Generate app key
docker compose exec api php artisan key:generate

# 6. Access the platform
# - WebUI: http://localhost:3000
# - API: http://localhost:8000
# - MinIO Console: http://localhost:9001
```

### Development

**Laravel API:**
```bash
cd apps/api
composer install
php artisan serve
```

**Next.js WebUI:**
```bash
cd apps/web
npm install
npm run dev
```

**Voice Engine:**
```bash
cd services/voice-engine
pip install -r requirements.txt
python main.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_DATABASE` | MariaDB database | morphvox |
| `DB_USERNAME` | Database user | morphvox |
| `DB_PASSWORD` | Database password | secret |
| `MINIO_ACCESS_KEY` | MinIO access key | minioadmin |
| `MINIO_SECRET_KEY` | MinIO secret key | minioadmin |
| `VOICE_ENGINE_DEVICE` | CPU or CUDA | cuda |

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| WebUI | 3000 | Next.js frontend |
| API | 8080 | Laravel backend |
| Voice Engine HTTP | 8001 | Inference API |
| Voice Engine WS | 8765 | Real-time streaming |
| Trainer | 8002 | Training service |
| Preprocessor | 8003 | Audio preprocessing |
| MariaDB | 3306 | Database |
| Redis | 6379 | Cache/Queue |
| MinIO API | 9000 | S3 storage |
| MinIO Console | 9001 | Storage admin |

## API Reference

See the OpenAPI specification in `packages/shared/openapi.yaml` for full API documentation.

## Future Enhancements

- [ ] Real-time WebRTC streaming
- [ ] Subscription/token billing
- [ ] Model marketplace
- [ ] Additional RVC model architectures
- [ ] Distributed training across multiple GPUs
