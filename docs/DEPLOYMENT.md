# MorphVox Production Deployment Guide

## Overview

MorphVox runs on a **Vast.ai GPU instance** (RTX 3090) with all services running **natively** (no Docker). Services are managed by **supervisord**. The domain `morphvox.net` is routed through a **Cloudflare Tunnel** to the instance.

**Key Constraint:** Vast.ai does NOT support Docker-in-Docker. All services must run natively.

---

## Architecture

```
Internet → Cloudflare Tunnel → Nginx (port 80) → Services
                                  ├── / → Next.js (port 3000)
                                  ├── /api → PHP-FPM (unix socket)
                                  ├── /ws → Voice Engine WebSocket (port 8765)
                                  ├── /voice-engine/ → Voice Engine HTTP (port 8001)
                                  ├── /storage → Static files
                                  └── /sanctum → PHP-FPM (auth cookies)

admin.morphvox.net → Cloudflare Tunnel → Nginx → PHP-FPM (Laravel admin)
```

### Services (all managed by supervisor)

| Service | Port | Technology | Purpose |
|---------|------|-----------|---------|
| nginx | 80 | Nginx | Reverse proxy, static files |
| nextjs | 3000 | Node.js (standalone build) | Frontend SPA |
| php-fpm | unix socket | PHP 8.5 | Laravel API |
| voice-engine | 8001 (HTTP), 8765 (WS), 9876 (internal) | Python 3.10 + PyTorch | AI inference, YouTube, audio processing |
| mariadb | 3306 | MariaDB | Database |
| redis | 6379 | Redis | Sessions, queues, cache |
| minio | 9000/9001 | MinIO | Object storage (S3-compatible) |
| cloudflared | - | Cloudflare Tunnel | Exposes site to internet |
| laravel-worker | - | PHP | Queue processing |

---

## Connecting to the Instance

### SSH Access

```bash
ssh -p <PORT> root@ssh1.vast.ai
```

The port changes when instances are recreated. Find it via:
```bash
vastai show instances --raw | jq '.[0].ssh_port'
# Or check Vast.ai dashboard
```

### Current Instance Details

- **Instance ID:** 36076771
- **GPU:** RTX 3090 (24GB VRAM)
- **RAM:** 64GB
- **Storage:** ~750GB
- **Location:** Pennsylvania, US
- **SSH:** `ssh -p 36770 root@ssh1.vast.ai`

> ⚠️ The SSH port (36770) may change if the instance is recreated.

---

## Directory Structure

```
/workspace/rvc_real_time/          # Main project root
├── apps/
│   ├── api/                       # Laravel PHP API
│   │   ├── .env                   # Production environment config
│   │   ├── public/                # Web root for PHP-FPM
│   │   ├── storage/logs/          # Laravel logs
│   │   └── config/services.php    # Service URLs config
│   └── web/                       # Next.js frontend
│       ├── .env.local             # Contains NEXT_PUBLIC_API_URL
│       └── .next/standalone/      # Production build (served by Node)
├── services/
│   └── voice-engine/              # Python voice engine
│       ├── main.py                # Entry point
│       ├── app/
│       │   ├── http_api.py        # FastAPI routes
│       │   ├── youtube_service.py # YouTube download/search
│       │   ├── model_manager.py   # RVC model loading
│       │   └── services/youtube/  # YouTube service module
│       └── assets/                # Symlinks to storage/assets
│           ├── rmvpe/ → /workspace/rvc_real_time/storage/assets/rmvpe/
│           ├── hubert/ → /workspace/rvc_real_time/storage/assets/hubert/
│           ├── models/ → /workspace/rvc_real_time/storage/models/
│           └── uvr5_weights/      # UVR5 vocal separation models (182MB)
├── storage/
│   ├── models/                    # Voice model files (12GB, 67 folders)
│   ├── uploads/                   # User uploads (211MB)
│   ├── app/public/model-images/   # Model avatar images (served via /storage)
│   ├── assets/                    # AI model weights (hubert, rmvpe)
│   ├── training/                  # Training data (48GB) - not backed up yet
│   └── preprocess/                # Preprocessed data (44GB)
├── scripts/
│   └── download_uvr5_assets.sh    # Downloads UVR5 models from HuggingFace
└── docker-compose.yml             # Local dev only (NOT used in production)

/etc/supervisor/conf.d/morphvox.conf   # Supervisor service definitions
/etc/nginx/sites-available/morphvox    # Nginx virtual host config
/etc/nginx/sites-enabled/morphvox      # Symlink to above
/workspace/onstart.sh                  # Startup persistence script
/usr/local/bin/run-cloudflared.sh      # Cloudflare tunnel runner
/root/.config/yt-dlp/config            # yt-dlp global config (JS runtime)
```

---

## Key Configuration Files

### Laravel `.env` (apps/api/.env)

Critical settings for production:
```env
APP_URL=https://morphvox.net
APP_ENV=production
APP_DEBUG=false
DB_HOST=127.0.0.1
DB_DATABASE=morphvox
DB_USERNAME=morphvox
DB_PASSWORD=MorphV0x2026Prod
SESSION_DOMAIN=.morphvox.net
VOICE_ENGINE_URL=http://127.0.0.1:8001
TRAINER_URL=http://127.0.0.1:8001
SANCTUM_STATEFUL_DOMAINS=morphvox.net,www.morphvox.net,admin.morphvox.net
TRUSTED_PROXIES=*
CORS_ALLOWED_ORIGINS=https://morphvox.net,https://www.morphvox.net,https://admin.morphvox.net
```

### Next.js `.env.local` (apps/web/.env.local)

```env
NEXT_PUBLIC_API_URL=https://morphvox.net/api
NEXT_PUBLIC_VOICE_ENGINE_WS_URL=wss://morphvox.net/ws
```

> ⚠️ `NEXT_PUBLIC_` vars are baked into the client JS at **build time**. If you change them, you must rebuild: `cd apps/web && npm run build`

### PHP-FPM (php.ini)

Located at `/etc/php/8.5/fpm/php.ini`:
```ini
upload_max_filesize = 500M
post_max_size = 500M
memory_limit = 512M
```

### yt-dlp Config

Located at `/root/.config/yt-dlp/config`:
```
--js-runtimes node
--remote-components ejs:github
```

> This is needed because YouTube now requires JS runtime for challenge solving. The Python `yt_dlp` library does NOT read this config — it's set in code via `'js_runtimes': {'node': {}}` and `'remote_components': ['ejs:github']` in ydl_opts.

---

## Common Operations

### Restart a service

```bash
supervisorctl restart <service>
# e.g., supervisorctl restart voice-engine
# e.g., supervisorctl restart php-fpm
```

### Check service status

```bash
supervisorctl status
```

### View logs

```bash
# Supervisor logs
tail -f /var/log/supervisor/voice-engine_err.log
tail -f /var/log/supervisor/nextjs_err.log
tail -f /var/log/supervisor/php-fpm.log

# Laravel logs
tail -f /workspace/rvc_real_time/apps/api/storage/logs/laravel.log

# Nginx logs
tail -f /var/log/nginx/morphvox-access.log
tail -f /var/log/nginx/morphvox-error.log
```

### Deploy code changes

There is no CI/CD pipeline. Deployment is manual:

```bash
# From local machine, sync changes to the instance:
rsync -avz --exclude='.git' --exclude='node_modules' --exclude='.next' \
  -e "ssh -p 36770" \
  /path/to/changed/files root@ssh1.vast.ai:/workspace/rvc_real_time/

# Then restart the affected service:
ssh -p 36770 root@ssh1.vast.ai 'supervisorctl restart voice-engine'
```

For **PHP/Laravel changes**: Just rsync the files — PHP reloads on each request.  
For **Voice Engine changes**: Restart `voice-engine` after syncing.  
For **Frontend changes**: Must rebuild Next.js:
```bash
cd /workspace/rvc_real_time/apps/web
npm run build
supervisorctl restart nextjs
```

### Database access

```bash
mysql -u morphvox -pMorphV0x2026Prod morphvox
```

### Rebuild the frontend

```bash
cd /workspace/rvc_real_time/apps/web
NEXT_PUBLIC_API_URL=https://morphvox.net/api npm run build
supervisorctl restart nextjs
```

---

## Important Technical Notes

### Voice Model Path Resolution

The voice engine resolves model paths in this order:
1. Absolute path (if exists on disk)
2. `MODEL_DIR` + filename
3. `MODEL_DIR` + relative path
4. Recursive glob in `MODEL_DIR`

Database stores absolute paths like:
```
/workspace/rvc_real_time/storage/models/Biden/biden.pth
```

The `MODEL_DIR` env var is set to `/workspace/rvc_real_time/storage/models` in the supervisor config.

### Authentication Flow

- Frontend uses **Bearer tokens** stored in `localStorage`
- Tokens are Laravel Sanctum personal access tokens
- `SANCTUM_STATEFUL_DOMAINS` must include `morphvox.net` for session-based auth to work
- The `VoiceModelController::resolveUser()` method manually checks Bearer tokens for routes without `auth:sanctum` middleware

### Cloudflare Tunnel

The tunnel maps:
- `morphvox.net` → `http://localhost:80`
- `admin.morphvox.net` → `http://localhost:80`

It handles HTTPS termination. Nginx only serves HTTP (port 80). The `X-Forwarded-Proto: https` header is set by nginx for all proxied requests so Laravel generates correct HTTPS URLs.

### Startup Persistence

When the Vast.ai instance restarts, `/workspace/onstart.sh` runs automatically (via Vast.ai's onstart mechanism + symlink at `/workspace/entrypoint.sh`). It starts supervisord which starts all services.

### UVR5 (Vocal Separation)

Required for "Voice Swap" / "Song Remix" features. Models located at:
```
services/voice-engine/assets/uvr5_weights/
├── HP2-人声vocals+非人声instrumentals.pth (61MB)
├── HP3_all_vocals.pth (61MB)
└── HP5-主旋律人声vocals+其他able.pth (61MB)
```

If missing, download with:
```bash
cd /workspace/rvc_real_time/services/voice-engine
bash scripts/download_uvr5_assets.sh
```

### YouTube Downloads

yt-dlp requires a JavaScript runtime (Node.js) to solve YouTube's challenges. This is configured in the Python code via:
```python
ydl_opts = {
    'js_runtimes': {'node': {}},
    'remote_components': ['ejs:github'],
    # ... other options
}
```

Both `app/youtube_service.py` and `app/services/youtube/service.py` need these options in ALL `ydl_opts` dicts.

---

## Troubleshooting

### "Failed to load models" in frontend
- Check `SANCTUM_STATEFUL_DOMAINS` includes `morphvox.net` in `.env`
- Check `VoiceModelController::myModels()` uses `$this->resolveUser()` (not `$request->user()`)
- Verify API returns 200: `curl -s -H "Accept: application/json" -H "Host: morphvox.net" -H "X-Forwarded-Proto: https" http://localhost/api/voice-models`

### "This video is not available" (YouTube)
- yt-dlp needs JS runtime options in Python code
- Test CLI: `yt-dlp --js-runtimes node --remote-components ejs:github --print title "URL"`
- May also need `pip install --upgrade yt-dlp`

### 413 Request Entity Too Large
- Check `client_max_body_size` in nginx config (currently 2G)
- Check PHP `upload_max_filesize` and `post_max_size` (currently 500M)

### Voice conversion returns 500
- Check model paths in DB are absolute: `/workspace/rvc_real_time/storage/models/...`
- Check model files actually exist on disk
- Check `assets/rmvpe/rmvpe.pt` and `assets/hubert/hubert_base.pt` symlinks exist

### Services in FATAL state
- Check logs: `tail -50 /var/log/supervisor/<service>_err.log`
- PHP-FPM: ensure `/run/php/` directory exists (`mkdir -p /run/php`)
- laravel-worker: often fails if DB isn't ready yet; `supervisorctl start laravel-worker`

---

## Creating a New Instance (Full Setup)

If the Vast.ai instance is destroyed, here's the high-level recreation process:

1. **Rent a new RTX 3090 instance** with ≥150GB disk, Ubuntu 22.04, reliability ≥0.95
2. **Install base packages:** nginx, php8.x-fpm, mariadb-server, redis-server, node, python3, supervisor, ffmpeg
3. **Clone the repo** to `/workspace/rvc_real_time/`
4. **Restore database** from a dump (keep dumps in R2 or local)
5. **Transfer model files** (12GB) via rsync from local or R2
6. **Install Python deps:** `pip install -r services/voice-engine/requirements.txt`
7. **Install PHP deps:** `cd apps/api && composer install`
8. **Build frontend:** `cd apps/web && npm ci && NEXT_PUBLIC_API_URL=https://morphvox.net/api npm run build`
9. **Copy config files:** supervisor conf, nginx conf, .env files
10. **Install cloudflared** and configure tunnel token
11. **Download AI assets:** UVR5, HuBERT, RMVPE models
12. **Create symlinks** in voice-engine/assets/
13. **Start supervisord**

The scripts in `scripts/` and the Docker configs in `infra/` can guide what packages/configs are needed.

---

## Local Development vs Production

| Aspect | Local (Docker) | Production (Vast.ai) |
|--------|---------------|---------------------|
| Orchestration | docker-compose.prod.yml | supervisord |
| Database | Container (port 3307) | Native MariaDB (3306) |
| PHP | Container | Native php8.5-fpm |
| Model paths | `/var/www/html/storage/models/` | `/workspace/rvc_real_time/storage/models/` |
| Frontend | `npm run dev` (port 3000) | Standalone build (port 3000) |
| HTTPS | No (localhost) | Cloudflare Tunnel |
| Domain | localhost | morphvox.net |

---

## Credentials Reference

All credentials are stored in `/workspace/rvc_real_time/apps/api/.env` on the instance. Key ones:

- **DB:** morphvox / MorphV0x2026Prod @ localhost:3306
- **MinIO:** Access/secret keys in supervisor env + .env
- **Cloudflare Tunnel:** Token in `/usr/local/bin/run-cloudflared.sh`
- **Vast.ai API:** In .env as `VASTAI_API_KEY`
- **R2 (Cloudflare):** In .env as `R2_ACCESS_KEY` / `R2_SECRET_KEY`

> Never commit credentials to git. They live only on the instance and in the .env file.
