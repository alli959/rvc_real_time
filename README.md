# MorphVox Platform

> **Note:** This project has been restructured into a full-stack platform. The original RVC real-time voice conversion code is now in `services/voice-engine/`.

## � Live Demo

**Production Site:** [https://morphvox.net](https://morphvox.net)

## �🎯 Overview

MorphVox is a comprehensive AI voice conversion platform featuring:

- **🌐 WebUI** - Modern Next.js frontend for model browsing and voice conversion
- **🔧 API Backend** - Laravel API for user management, model registry, and job processing
- **🎤 Voice Engine** - Python RVC service for real-time voice conversion
- **📦 S3 Storage** - MinIO for scalable object storage

## 📁 Project Structure

```
morphvox/
├── apps/
│   ├── api/                    # Laravel 11 Backend
│   └── web/                    # Next.js 14 Frontend
│
├── services/
│   └── voice-engine/           # RVC Inference (original code)
│
├── infra/
│   └── compose/                # Docker Compose stack
│
└── docs/
    └── ARCHITECTURE.md         # Full architecture documentation
```

## 🚀 Quick Start

### Using Docker Compose (Recommended)

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
open http://localhost:8000/api  # API
open http://localhost:9001      # MinIO Console
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
| **user** | Use public models, create jobs |
| **premium** | Upload private models |
| **creator** | Train custom models |
| **admin** | Full platform access |

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
| Voice Engine HTTP | 8000 | File-based API |
| Voice Engine WS | 8765 | Real-time streaming |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache/Queue |
| MinIO API | 9000 | S3 storage |
| MinIO Console | 9001 | Storage admin |

## 🗺️ Roadmap

- [x] Laravel API with auth & permissions
- [x] Next.js WebUI scaffold
- [x] Docker Compose infrastructure
- [ ] Model training pipeline
- [ ] Real-time WebRTC streaming
- [ ] Subscription billing
- [ ] Model marketplace

## 📄 License

MIT License - See LICENSE file for details.

## 👨‍💻 Creator

**Alexander Guðmundsson**

- 🌐 Website: [morphvox.net](https://morphvox.net)
- 💬 Discord: [alexanderg](https://discord.com/users/alexanderg)
- 💼 LinkedIn: [Alexander Guðmundsson](https://linkedin.com/in/alexander-gudmundsson)
- 🐙 GitHub: [alexanderg](https://github.com/alexanderg)

---

**Original RVC Real-Time Project** - For the original standalone voice conversion tool, see [services/voice-engine/](services/voice-engine/).
