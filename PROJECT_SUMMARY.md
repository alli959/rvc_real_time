# MorphVox Platform - Project Summary

## Project Overview

MorphVox is a comprehensive AI voice conversion platform built as a full-stack monorepo. It evolved from a standalone RVC voice conversion tool into a complete platform featuring:

- **Web Application** - Modern Next.js frontend for model browsing and voice processing
- **Laravel API Backend** - User auth, model registry, job processing, and admin panel
- **Voice Engine** - Python service for RVC inference, TTS, and audio processing
- **Preprocessor Service** - Audio preprocessing for model training (F0, HuBERT features)
- **Trainer Service** - RVC model training orchestration
- **Infrastructure** - Docker Compose stack with MariaDB, Redis, MinIO S3 storage

## Implemented Features

### Platform Features
- ✅ User authentication with OAuth (Google, GitHub)
- ✅ Role-based permissions (user, premium, creator, admin)
- ✅ Voice model registry with public/private models
- ✅ Job queue with full history tracking
- ✅ Admin panel at admin.morphvox.net
- ✅ User role request system
- ✅ S3 storage with presigned URLs

### Voice Engine Features
- ✅ RVC voice conversion with WebUI model compatibility
- ✅ Text-to-Speech (Bark TTS + Edge TTS with 50+ voices)
- ✅ Emotion tags and audio effects (reverb, echo, pitch)
- ✅ Vocal separation using UVR5 (HP3, HP5 models)
- ✅ Voice swap in songs (song remix)
- ✅ Multi-voice TTS support
- ✅ Voice detection (count speakers in audio)
- ✅ YouTube audio download integration

### Training Features
- ✅ Dedicated preprocessor service for audio preparation
- ✅ F0 extraction (RMVPE method)
- ✅ HuBERT feature extraction
- ✅ Audio slicing with silence detection
- ✅ Dedicated trainer service for model training
- ✅ Checkpoint extraction and model packaging
- ✅ FAISS index building
- ✅ Training wizard with step-by-step workflow
- ✅ Batched file uploads (auto-splits large uploads)
- ✅ Upload progress tracking with visual feedback
- ✅ Audio normalization for proper training quality
- ✅ Continue training from checkpoint support

### API & Streaming
- ✅ RESTful HTTP API (port 8001)
- ✅ WebSocket server for real-time streaming (port 8765)
- ✅ Multiple concurrent client support
- ✅ GPU/CUDA automatic detection

### Deployment
- ✅ Docker containerization for all services
- ✅ Docker Compose for development and production
- ✅ Nginx reverse proxy with SSL/Let's Encrypt
- ✅ Health checks and environment configuration

## Project Structure

```
morphvox/
├── apps/
│   ├── api/                      # Laravel 11 Backend
│   │   ├── app/Http/Controllers/
│   │   ├── app/Services/
│   │   ├── routes/api.php
│   │   └── config/
│   │
│   └── web/                      # Next.js 14 Frontend
│       └── src/app/
│
├── services/
│   ├── voice-engine/             # Python Voice Engine (Inference)
│   │   ├── app/
│   │   │   ├── core/             # Config, logging, exceptions
│   │   │   ├── models/           # Pydantic request/response schemas
│   │   │   ├── services/         # Business logic layer
│   │   │   │   ├── voice_conversion/
│   │   │   │   ├── audio_analysis/
│   │   │   │   ├── tts/
│   │   │   │   └── youtube/
│   │   │   └── routers/          # FastAPI route handlers
│   │   ├── rvc/                  # RVC pipeline
│   │   ├── assets/               # Models, weights
│   │   └── main.py
│   │
│   ├── preprocessor/             # Audio Preprocessing Service
│   │   ├── app/
│   │   │   ├── api/              # FastAPI routers
│   │   │   ├── pipeline/         # Preprocessing stages
│   │   │   └── core/             # Config, exceptions
│   │   └── main.py
│   │
│   └── trainer/                  # Model Training Service
│       ├── app/
│       │   ├── api/              # FastAPI routers
│       │   ├── pipeline/         # Training pipeline
│       │   └── core/             # Config, exceptions
│       └── main.py
│
├── packages/
│   ├── sdk-js/                   # JavaScript/TypeScript SDK
│   ├── sdk-python/               # Python SDK
│   └── shared/                   # OpenAPI schemas
│
├── infra/
│   └── compose/                  # Docker Compose configs
│       ├── docker-compose.yml
│       └── docker-compose.prod.yml
│
├── docs/
│   └── ARCHITECTURE.md
│
└── scripts/                      # Setup and utility scripts
```

## Technology Stack

### Frontend (apps/web)
- **Next.js 14** - React framework with App Router
- **TailwindCSS** - Styling
- **TanStack Query** - Data fetching
- **Zustand** - State management

### Backend (apps/api)
- **Laravel 11** - PHP framework
- **Laravel Sanctum** - API authentication
- **Spatie Permission** - Role-based access control
- **Laravel Queue** - Background job processing

### Voice Engine (services/voice-engine)
- **FastAPI** - HTTP API framework
- **PyTorch** - Model inference and GPU support
- **Bark** - Neural text-to-speech
- **Edge TTS** - Microsoft TTS API
- **UVR5** - Vocal separation
- **yt-dlp** - YouTube audio download

### Preprocessor (services/preprocessor)
- **FastAPI** - HTTP API framework
- **librosa** - Audio processing
- **RMVPE** - F0 extraction
- **HuBERT** - Feature extraction

### Trainer (services/trainer)
- **FastAPI** - HTTP API framework
- **PyTorch** - Model training
- **FAISS** - Index building

### Infrastructure
- **Docker** - Containerization
- **MariaDB/PostgreSQL** - Database
- **Redis** - Cache and job queue
- **MinIO** - S3-compatible object storage
- **Nginx** - Reverse proxy with SSL

## Architecture

### System Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐
│   Next.js   │───▶│   Laravel   │───▶│       Python Services           │
│   WebUI     │    │    API      │    │                                 │
│  (port 3000)│◀───│ (port 8080) │◀───│  voice-engine  (8001, ws:8765) │
└─────────────┘    └──────┬──────┘    │  preprocessor  (8003)           │
                         │           │  trainer       (8002)           │
                  ┌──────┴──────┐    └─────────────────────────────────┘
                  │  MariaDB    │                    │
                  │  Redis      │                    │
                  │  MinIO S3   │◀───────────────────┘
                  └─────────────┘
```

### Voice Engine Architecture
```
HTTP Request → Router → Service → Model/Inference → Response
                 │
                 └── Thin handlers, business logic in services
```

### Key Services

1. **Voice Conversion** (`services/voice_conversion/`)
   - Model discovery and loading
   - RVC pipeline integration
   - Index-based similarity search

2. **Text-to-Speech** (`services/tts/`)
   - Bark TTS (neural, emotion support)
   - Edge TTS (50+ voices)
   - Audio effects (reverb, echo, pitch)
   - Multi-voice synthesis

3. **Audio Analysis** (`services/audio_analysis/`)
   - Voice detection (speaker count)
   - Vocal separation (UVR5)
   - Audio feature extraction

4. **YouTube Integration** (`services/youtube/`)
   - Audio search and download
   - Caching for repeated requests

## Usage

### Docker Compose (Recommended)
```bash
cd infra/compose
cp .env.example .env
docker compose up -d
```

### Voice Engine Only
```bash
cd services/voice-engine
pip install -r requirements.txt
python main.py --mode api
```

### Development
```bash
# API
cd apps/api && composer install && php artisan serve

# WebUI  
cd apps/web && npm install && npm run dev

# Voice Engine
cd services/voice-engine && python main.py --mode api
```

## Configuration

Environment variables are configured via `.env.example` files in:
- `infra/compose/.env.example` - Docker Compose stack
- `apps/api/.env.example` - Laravel API
- `services/voice-engine/.env.example` - Voice Engine

Key settings include database credentials, S3 storage, OAuth keys, and service URLs.

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

## User Roles & Permissions

| Role | Permissions |
|------|-------------|
| guest | View public models |
| user | + Use TTS, create jobs |
| premium | + Upload models, extended usage |
| creator | + Train models |
| admin | Full platform access |

## Future Enhancements

- [ ] Real-time WebRTC streaming
- [ ] Subscription billing
- [ ] Model marketplace
- [ ] Additional RVC model architectures
- [ ] Performance monitoring dashboard
- [ ] Distributed training across multiple GPUs

## Recent Updates (January 2026)

### Training Pipeline Improvements
- **Audio Normalization Fix**: Fixed critical bug where training received unnormalized int16 audio values instead of normalized floats (-1 to 1), causing loss values in billions instead of normal ranges (~1-10)
- **Batched File Uploads**: Large file uploads are now automatically batched (4 files at a time) to prevent timeout issues
- **Upload Progress Tracking**: Visual progress bar shows upload status across all batches
- **PHP-FPM Tuning**: Increased max workers from 5 to 50 for better concurrent request handling

### Infrastructure
- **Storage Layout**: Unified storage directory structure for all services
- **PHP Configuration**: Optimized memory limits (1GB) and upload sizes for large audio files

## Development Guidelines

### Voice Engine Structure
```
routers/   → Thin HTTP handlers (validation, response formatting)
services/  → Business logic (reusable, testable)
models/    → Pydantic schemas (request/response validation)
core/      → Infrastructure (config, logging, exceptions)
```

### Adding New Features
1. Create models in `app/models/`
2. Implement business logic in `app/services/`
3. Add HTTP handlers in `app/routers/`
4. Register router in `main.py`

### Code Style
- Follow existing patterns
- Use type hints (Pydantic models)
- Add docstrings to public functions
- Keep services stateless when possible

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Full system architecture
- [services/voice-engine/README.md](services/voice-engine/README.md) - Voice Engine docs
- [apps/api/README.md](apps/api/README.md) - Laravel API docs
- [infra/COMMANDS.md](infra/COMMANDS.md) - Deployment commands

---

**Project Status**: ✅ Active Development

The platform is deployed at [morphvox.net](https://morphvox.net) with ongoing feature development.
