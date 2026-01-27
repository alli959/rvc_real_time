# MorphVox Platform

> **Note:** This project has been restructured into a full-stack platform. The original RVC real-time voice conversion code is now in `services/voice-engine/`.

## 🎯 Live Demo

[https://morphvox.net](https://morphvox.net)

## 🎯 Overview

MorphVox is a comprehensive AI voice conversion platform featuring:

- **🌐 WebUI** - Modern Next.js frontend for model browsing, voice conversion, and TTS
- **🔧 API Backend** - Laravel API for user management, model registry, and job processing
- **🎤 Voice Engine** - Python RVC service for real-time voice conversion
- **📦 S3 Storage** - MinIO for scalable object storage with presigned URLs
- **🗣️ Text-to-Speech** - Bark TTS (neural) + Edge TTS (50+ voices) with emotion support
- **🎵 Audio Processing** - Voice conversion, vocal separation (UVR5), and voice swap
- **🎶 Song Remix** - Split vocals from instrumentals and swap voices in songs
- **🔊 Voice Training** - Train custom RVC voice models from audio samples
- **👤 Admin Panel** - Full administration dashboard at admin.morphvox.net
- **🔐 OAuth Login** - Google and GitHub OAuth authentication support

## 📁 Project Structure

```
morphvox/
├── apps/
│   ├── api/                    # Laravel 11 Backend
│   └── web/                    # Next.js 14 Frontend
│
├── services/
│   ├── voice-engine/           # RVC Inference (HTTP: 8001, WS: 8765)
│   ├── preprocessor/           # Audio Preprocessing (HTTP: 8003)
│   └── trainer/                # Model Training (HTTP: 8002)
│
├── infra/
│   └── compose/                # Docker Compose stack
│
├── scripts/
│   ├── dev-up.sh               # Complete dev setup & start
│   └── service-up.sh           # Start individual services
│
└── docs/
    ├── ARCHITECTURE.md         # Full architecture documentation
    └── DEVELOPMENT.md          # Development guide
```

## 🚀 Quick Start

### Using the Setup Script (Recommended)

```bash
# Complete setup: downloads assets + starts Docker services
./scripts/dev-up.sh

# Production mode
./scripts/dev-up.sh --prod

# Setup assets only (no Docker)
./scripts/dev-up.sh --no-docker
```

### Using Docker Compose (Manual)

```bash
# 1. Copy environment file
cp infra/compose/.env.example infra/compose/.env

# 2. Start all services
cd infra/compose
docker compose up -d

# 3. Initialize database
docker compose exec api php artisan migrate --seed
docker compose exec api php artisan key:generate

# 4. Access the platform
open http://localhost:3000      # WebUI
open http://localhost:8080      # API
open http://localhost:9001      # MinIO Console
```

### Start Individual Services

```bash
# Start specific service with its dependencies
./scripts/service-up.sh trainer       # Trainer + preprocess + minio
./scripts/service-up.sh voice-engine  # Voice engine + minio
./scripts/service-up.sh infra -d      # Infrastructure only (background)
```

### Voice Engine Only (Original Functionality)

If you just need the RVC voice conversion service:

```bash
cd services/voice-engine

# Using Docker
docker build -t voice-engine .
docker run -p 8765:8765 -v ./assets:/app/assets voice-engine

# Or locally
pip install -r requirements.txt
python main.py --mode api
```

## 🎤 Voice Engine Features

The voice engine supports:

- **Real-time Audio Processing** via WebSocket (port 8765)
- **WebUI model compatibility** (`.pth` + `.index` files)
- **Multiple F0 methods** (rmvpe, pm, harvest, dio)
- **GPU acceleration** (CUDA auto-detect)

### Quick Model Selection

```bash
cd services/voice-engine
./start-api.sh
# Select a model from the interactive menu
```

### WebSocket Client Example

```python
import asyncio
import websockets
import json

async def convert_voice():
    async with websockets.connect('ws://localhost:8765') as ws:
        # Send model config
        await ws.send(json.dumps({
            'type': 'config',
            'model': 'BillCipher',
            'pitch': 0
        }))
        
        # Stream audio chunks
        for chunk in audio_chunks:
            await ws.send(chunk)
            converted = await ws.recv()
```

## 🔐 User Roles

| Role | Capabilities |
|------|-------------|
| **user** | Use public models, TTS, create jobs |
| **premium** | Upload private models, extended usage |
| **creator** | Train custom models, monetization |
| **admin** | Full platform access, admin panel |

> Users can request role upgrades through the platform. Admins can approve/reject role requests.

## 📚 Documentation

- [Full Architecture](docs/ARCHITECTURE.md) - Detailed system design
- [Voice Engine](services/voice-engine/README.md) - RVC service docs
- [API Routes](apps/api/routes/api.php) - Backend endpoints

## 🛠️ Development

### Prerequisites

- Docker & Docker Compose
- Node.js 20+ (for WebUI development)
- PHP 8.3+ (for API development)
- Python 3.10+ (for Voice Engine)
- NVIDIA GPU (optional, for faster inference)

### Local Development

```bash
# API (Laravel)
cd apps/api
composer install
cp .env.example .env
php artisan serve

# WebUI (Next.js)
cd apps/web
npm install
npm run dev

# Voice Engine (Python)
cd services/voice-engine
pip install -r requirements.txt
python main.py --mode api
```

## 📊 Service Ports

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

## 🗺️ Roadmap

- [x] Laravel API with auth & permissions
- [x] Next.js WebUI scaffold
- [x] Docker Compose infrastructure
- [x] Text-to-Speech with Edge TTS (50+ voices)
- [x] Audio processing (vocal separation, voice swap)
- [x] Admin panel (admin.morphvox.net)
- [x] Role request system
- [x] Job queue tracking & history
- [x] YouTube audio download integration
- [x] OAuth login (Google & GitHub)
- [x] S3 storage with presigned URLs
- [x] Combined models page with tabs (Community/My Models)
- [x] Dedicated Song Remix page
- [x] Dedicated preprocessor service (audio slicing, F0, HuBERT)
- [x] Dedicated trainer service (RVC training, index building)
- [x] Training wizard with upload progress tracking
- [x] Batched file uploads for reliable large transfers
- [x] Audio normalization fix for training quality
- [ ] Real-time WebRTC streaming
- [ ] Subscription billing
- [ ] Model marketplace

## 📄 License

MIT License - See LICENSE file for details.

## 👨‍💻 Creator

**Alexander Guðmundsson**

- 🌐 Website: [morphvox.net](https://morphvox.net)
- 💼 LinkedIn: [Alexander Guðmundsson](https://www.linkedin.com/in/alexander-gu%C3%B0mundsson-053200189/)
- 🐙 GitHub: [alexanderg](https://github.com/alli959)

---

**Original RVC Real-Time Project** - For the original standalone voice conversion tool, see [services/voice-engine/](services/voice-engine/).
