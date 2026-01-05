# MorphVox Platform Architecture

## Overview

MorphVox is a comprehensive AI voice conversion platform built as a monorepo with the following components:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              MorphVox Platform                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────────────┐ │
│  │   Next.js   │───▶│   Laravel   │───▶│      Python Services         │ │
│  │   WebUI     │    │    API      │    │                              │ │
│  │  (port 3000)│◀───│ (port 8080) │◀───│  ┌────────────────────────┐ │ │
│  └─────────────┘    └──────┬──────┘    │  │    Voice Engine        │ │ │
│                            │           │  │   HTTP: 8000           │ │ │
│                            │           │  │   WS:   8765           │ │ │
│                            ▼           │  └────────────────────────┘ │ │
│                     ┌──────────────┐   │                              │ │
│                     │  PostgreSQL  │   │  ┌────────────────────────┐ │ │
│                     │    (5432)    │   │  │    Trainer (future)    │ │ │
│                     └──────────────┘   │  └────────────────────────┘ │ │
│                            │           │                              │ │
│                     ┌──────────────┐   │  ┌────────────────────────┐ │ │
│                     │    Redis     │   │  │  Preprocessor (future) │ │ │
│                     │    (6379)    │   │  └────────────────────────┘ │ │
│                     └──────────────┘   └──────────────────────────────┘ │
│                            │                         │                   │
│                     ┌──────────────┐                 │                   │
│                     │    MinIO     │◀────────────────┘                   │
│                     │  S3 Storage  │                                     │
│                     │ (9000/9001)  │                                     │
│                     └──────────────┘                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
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
│   ├── voice-engine/           # RVC Inference Service
│   │   ├── app/
│   │   ├── rvc/
│   │   ├── infer/
│   │   └── main.py
│   │
│   ├── trainer/                # Model Training (future)
│   └── preprocessor/           # Audio Preprocessing (future)
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

Python service for RVC inference:
- **HTTP API** (port 8000): File-based inference
- **WebSocket** (port 8765): Real-time streaming

Supports:
- Multiple F0 methods (rmvpe, pm, harvest, dio)
- Index-based similarity search
- Batch processing

### 4. Storage (MinIO)

S3-compatible object storage for:
- Voice model files (`.pth`, `.index`)
- Audio inputs and outputs
- User uploads

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
| `DB_DATABASE` | PostgreSQL database | voiceforge |
| `DB_USERNAME` | Database user | voiceforge |
| `DB_PASSWORD` | Database password | secret |
| `MINIO_ACCESS_KEY` | MinIO access key | minioadmin |
| `MINIO_SECRET_KEY` | MinIO secret key | minioadmin |
| `VOICE_ENGINE_DEVICE` | CPU or CUDA | cpu |

## API Reference

See the OpenAPI specification in `packages/shared/openapi.yaml` for full API documentation.

## Future Enhancements

- [ ] Model training pipeline
- [ ] Real-time WebRTC streaming
- [ ] Subscription/token billing
- [ ] Model marketplace
- [ ] Multi-language TTS integration
