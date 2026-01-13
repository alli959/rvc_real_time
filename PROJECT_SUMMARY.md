# MorphVox Platform - Project Summary

## Project Overview

MorphVox is a comprehensive AI voice conversion platform built as a full-stack monorepo. It evolved from a standalone RVC voice conversion tool into a complete platform featuring:

- **Web Application** - Modern Next.js frontend for model browsing and voice processing
- **Laravel API Backend** - User auth, model registry, job processing, and admin panel
- **Voice Engine** - Python service for RVC inference, TTS, and audio processing
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
│   └── voice-engine/             # Python Voice Engine
│       ├── app/
│       │   ├── core/             # Config, logging, exceptions
│       │   ├── models/           # Pydantic request/response schemas
│       │   ├── services/         # Business logic layer
│       │   │   ├── voice_conversion/
│       │   │   ├── audio_analysis/
│       │   │   ├── tts/
│       │   │   └── youtube/
│       │   └── routers/          # FastAPI route handlers
│       ├── rvc/                  # RVC pipeline
│       ├── assets/               # Models, weights
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

### Infrastructure
- **Docker** - Containerization
- **MariaDB/PostgreSQL** - Database
- **Redis** - Cache and job queue
- **MinIO** - S3-compatible object storage
- **Nginx** - Reverse proxy with SSL

## Architecture

### System Architecture
```
┌─────────────┐    ┌─────────────┐    ┌──────────────────────────┐
│   Next.js   │───▶│   Laravel   │───▶│     Voice Engine         │
│   WebUI     │    │    API      │    │  (FastAPI + RVC + TTS)   │
│  (port 3000)│◀───│ (port 8080) │◀───│  HTTP: 8001, WS: 8765    │
└─────────────┘    └──────┬──────┘    └────────────┬─────────────┘
                         │                         │
                  ┌──────┴──────┐                  │
                  │  MariaDB    │                  │
                  │  Redis      │                  │
                  │  MinIO S3   │◀─────────────────┘
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
| Voice Engine HTTP | 8001 | File-based API |
| Voice Engine WS | 8765 | Real-time streaming |
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

- [ ] Model training pipeline (in progress - see TRAINER_DESIGN.md)
- [ ] Real-time WebRTC streaming
- [ ] Subscription billing
- [ ] Model marketplace
- [ ] Additional RVC model architectures
- [ ] Performance monitoring dashboard

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
