# MorphVox Unified Storage Layout

## Overview

This document describes the canonical storage layout for the MorphVox platform. All services mount and use the same `./storage` root directory for consistent data access.

## Storage Structure

```
storage/
├── logs/                          # Service-specific file logs
│   ├── api/                       # Laravel API logs
│   ├── api-worker/                # Laravel queue worker logs
│   ├── web/                       # Next.js frontend logs
│   ├── voice-engine/              # Voice inference service logs
│   ├── trainer/                   # RVC training service logs
│   ├── preprocess/                # Audio preprocessing logs
│   ├── nginx/                     # Reverse proxy access/error logs
│   ├── mariadb/                   # Database logs
│   ├── redis/                     # Cache logs
│   └── minio/                     # Object storage logs
│
├── data/                          # Runtime data
│   ├── uploads/                   # All uploaded source files
│   │   └── <model_slug>/          # Per-model upload directory
│   │
│   ├── preprocess/                # Preprocessing artifacts
│   │   └── <model_slug>/
│   │       ├── 0_gt_wavs/         # Ground truth wav slices
│   │       ├── 1_16k_wavs/        # Resampled 16kHz wavs
│   │       ├── 2a_f0/             # F0 pitch features
│   │       ├── 2b_f0nsf/          # F0 NSF features
│   │       └── 3_feature768/      # HuBERT features
│   │
│   ├── training/                  # Training artifacts
│   │   └── <model_slug>/
│   │       ├── G_*.pth            # Generator checkpoints
│   │       ├── D_*.pth            # Discriminator checkpoints
│   │       └── events.out.*       # TensorBoard logs
│   │
│   ├── outputs/                   # Generated outputs
│   │   └── <model_slug>/          # Per-model inference outputs
│   │
│   └── infra/                     # Infrastructure persistence
│       ├── mariadb/               # MySQL data (/var/lib/mysql)
│       ├── redis/                 # Redis AOF data
│       ├── minio/                 # S3-compatible storage
│       └── certbot/               # SSL certificates
│           ├── conf/              # Let's Encrypt config
│           └── www/               # ACME challenge files
│
├── assets/                        # Shared non-user assets (read-only for services)
│   ├── hubert/                    # HuBERT model (hubert_base.pt)
│   ├── rmvpe/                     # RMVPE pitch model (rmvpe.pt)
│   ├── pretrained_v2/             # RVC pretrained models (f0G48k.pth, f0D48k.pth)
│   ├── uvr5_weights/              # Vocal separation models
│   ├── bark/                      # Bark TTS models
│   ├── whisper/                   # Whisper transcription models
│   └── index/                     # FAISS index files
│
└── models/                        # User voice models
    ├── <model_name>.pth           # Model weights
    ├── <model_name>.index         # FAISS index (optional)
    └── <model_name>/              # Per-model metadata directory
        ├── config.json            # Model configuration (optional)
        └── image.<ext>            # Model avatar/cover image
```

## Environment Variables

All services use these environment variables with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_ROOT` | `/storage` | Root storage path inside containers |
| `STORAGE_LOGS` | `$STORAGE_ROOT/logs` | Service logs directory |
| `STORAGE_DATA` | `$STORAGE_ROOT/data` | Runtime data directory |
| `STORAGE_ASSETS` | `$STORAGE_ROOT/assets` | Shared assets directory |
| `STORAGE_MODELS` | `$STORAGE_ROOT/models` | User models directory |

### Derived Paths

Services derive specific paths from these variables:

- **Uploads**: `$STORAGE_DATA/uploads`
- **Preprocessing**: `$STORAGE_DATA/preprocess`
- **Training**: `$STORAGE_DATA/training`
- **Outputs**: `$STORAGE_DATA/outputs`
- **HuBERT**: `$STORAGE_ASSETS/hubert/hubert_base.pt`
- **RMVPE**: `$STORAGE_ASSETS/rmvpe`
- **Pretrained**: `$STORAGE_ASSETS/pretrained_v2`

## Container Mount Strategy

All containers mount the entire `./storage` directory to `/storage`:

```yaml
volumes:
  - ../../storage:/storage
```

This provides:
- Consistent paths across all services
- Single source of truth for all data
- Easy backup and migration
- Works with both dev and prod compose files

## Service-Specific Configurations

### API (Laravel)

```php
// config/storage_paths.php
return [
    'root' => env('STORAGE_ROOT', '/storage'),
    'logs' => env('STORAGE_ROOT', '/storage') . '/logs/api',
    'models' => env('STORAGE_ROOT', '/storage') . '/models',
];
```

### Voice Engine (Python)

```python
# app/storage_paths.py
STORAGE_ROOT = os.getenv('STORAGE_ROOT', '/storage')
STORAGE_LOGS = os.path.join(STORAGE_ROOT, 'logs/voice-engine')
STORAGE_MODELS = os.path.join(STORAGE_ROOT, 'models')
STORAGE_ASSETS = os.path.join(STORAGE_ROOT, 'assets')
```

### Trainer & Preprocessor (Python)

```python
# Uses same storage_paths.py pattern
DATA_ROOT = os.path.join(STORAGE_ROOT, 'data')
UPLOADS_DIR = os.path.join(DATA_ROOT, 'uploads')
PREPROCESS_DIR = os.path.join(DATA_ROOT, 'preprocess')
TRAINING_DIR = os.path.join(DATA_ROOT, 'training')
```

## Logging Strategy

Each service writes logs to its dedicated directory under `storage/logs/<service>/`:

| Service | Log File | Method |
|---------|----------|--------|
| api | `laravel.log` | Laravel file driver |
| api-worker | `worker.log` | Queue worker output |
| web | `web.log` | stdout tee'd to file |
| voice-engine | `voice_engine.log` | Python RotatingFileHandler |
| trainer | `trainer.log` | Python RotatingFileHandler |
| preprocess | `preprocessor.log` | Python RotatingFileHandler |
| nginx | `access.log`, `error.log` | Nginx config |
| mariadb | `error.log` | MariaDB config |
| redis | `redis.log` | Redis config |
| minio | `minio.log` | MinIO stdout redirect |

## Migration Guide

### From Old Structure

Run the migration script:

```bash
./scripts/migrate-storage.sh
```

This will:
1. Create the new storage structure
2. Move existing data to correct locations
3. Create compatibility symlinks (temporary)
4. Update environment files

### Manual Migration

1. Stop all services: `docker-compose down`
2. Create storage structure: `mkdir -p storage/{logs,data,assets,models}`
3. Move assets: `mv services/voice-engine/assets/* storage/assets/`
4. Move models: `mv services/voice-engine/assets/models/* storage/models/`
5. Move data: `mv data/* storage/data/uploads/`
6. Update `.env` with `STORAGE_ROOT=./storage`
7. Start services: `docker-compose up -d`

## Future S3 Migration

The storage path logic is centralized in `storage_paths.py` (Python) and `config/storage_paths.php` (PHP). To migrate to S3:

1. Create an S3 storage provider that implements the same interface
2. Update the storage path module to use S3 URLs when configured
3. Set `STORAGE_BACKEND=s3` in environment

## Verification

After setup, verify the storage layout:

```bash
# Check directory structure
tree -L 3 storage/

# Verify mounts in containers
docker exec morphvox-voice-engine ls -la /storage

# Check logs are being written
tail -f storage/logs/voice-engine/voice_engine.log

# Verify model access
docker exec morphvox-api ls -la /storage/models
```
