# MorphVox API

Laravel 11 backend API for the MorphVox voice conversion platform.

## Quick Start

```bash
# Install dependencies
composer install

# Configure environment
cp .env.example .env
php artisan key:generate

# Run migrations
php artisan migrate

# Sync voice models
php artisan voice-models:sync

# Start development server
php artisan serve --port=8000
```

## Features

- **User Authentication** - Laravel Sanctum with SPA support
- **Voice Models Management** - Unified system for local and S3 storage
- **Job Queue** - Background processing for voice conversion
- **Permissions System** - Spatie Laravel Permission
- **RESTful API** - JSON API with CORS support

## Documentation

- [Voice Models System](VOICE_MODELS.md) - Complete guide to managing voice models
- [API Routes](routes/api.php) - All available endpoints
- [Environment Variables](#environment-variables) - Configuration options

## Environment Variables

### Application

```bash
APP_NAME=MorphVox
APP_ENV=local
APP_DEBUG=true
APP_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
```

### Database

```bash
DB_CONNECTION=mysql
DB_HOST=localhost
DB_PORT=3306
DB_DATABASE=morphvox
DB_USERNAME=morphvox
DB_PASSWORD=master123
```

### Voice Models Storage

```bash
# Storage type: "local" or "s3"
VOICE_MODELS_STORAGE=local

# For local storage
VOICE_MODELS_LOCAL_PATH=../../services/voice-engine/assets/models

# For S3 storage
VOICE_MODELS_S3_DISK=s3
VOICE_MODELS_S3_PREFIX=models
VOICE_MODELS_S3_URL_EXPIRATION=60
```

See [VOICE_MODELS.md](VOICE_MODELS.md) for complete voice models documentation.

### Object Storage (S3/MinIO)

```bash
FILESYSTEM_DISK=s3
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_DEFAULT_REGION=us-east-1
AWS_BUCKET=morphvox
AWS_ENDPOINT=http://minio:9000
AWS_USE_PATH_STYLE_ENDPOINT=true
```

### Voice Engine Service

```bash
VOICE_ENGINE_URL=http://voice-engine:8765
VOICE_ENGINE_SOCKET_URL=tcp://voice-engine:9876
```

## API Endpoints

### Authentication

```http
POST   /api/auth/register
POST   /api/auth/login
POST   /api/auth/logout
GET    /api/auth/me
```

### Voice Models (System)

```http
GET    /api/voice-models              # List all models
GET    /api/voice-models/stats        # Get statistics
GET    /api/voice-models/config       # Get configuration
GET    /api/voice-models/{slug}       # Get single model
POST   /api/voice-models/sync         # Trigger sync (auth)
PATCH  /api/voice-models/{slug}       # Update metadata (auth)
```

### User Models

```http
GET    /api/models                    # List public models
GET    /api/models/{uuid}             # Get model details
GET    /api/models/my                 # My models (auth)
POST   /api/models                    # Create model (auth)
PUT    /api/models/{uuid}             # Update model (auth)
DELETE /api/models/{uuid}             # Delete model (auth)
```

### Jobs

```http
GET    /api/jobs                      # List jobs (auth)
GET    /api/jobs/{uuid}               # Job status (auth)
DELETE /api/jobs/{uuid}               # Cancel job (auth)
```

## Artisan Commands

### Voice Models

```bash
# Sync models from storage
php artisan voice-models:sync

# Sync with options
php artisan voice-models:sync --storage=s3 --prune --force

# View command help
php artisan voice-models:sync --help
```

### Database

```bash
# Run migrations
php artisan migrate

# Rollback migrations
php artisan migrate:rollback

# Fresh migration with seeding
php artisan migrate:fresh --seed
```

### Cache & Config

```bash
# Clear all caches
php artisan optimize:clear

# Cache configuration
php artisan config:cache
php artisan route:cache
php artisan view:cache
```

## Development

### Running Tests

```bash
php artisan test
```

### Code Style

```bash
# Fix code style
./vendor/bin/pint

# Check code style
./vendor/bin/pint --test
```

### Database Seeding

```bash
php artisan db:seed
```

## Project Structure

```
apps/api/
├── app/
│   ├── Console/Commands/          # Artisan commands
│   │   └── SyncVoiceModels.php    # Voice models sync
│   ├── Http/Controllers/Api/      # API controllers
│   │   ├── AuthController.php
│   │   ├── SystemVoiceModelController.php
│   │   ├── VoiceModelController.php
│   │   └── JobController.php
│   ├── Models/                    # Eloquent models
│   │   ├── User.php
│   │   ├── SystemVoiceModel.php
│   │   ├── VoiceModel.php
│   │   └── JobQueue.php
│   └── Services/                  # Business logic
│       └── VoiceModelScanner.php
├── config/
│   ├── voice_models.php           # Voice models config
│   ├── cors.php                   # CORS config
│   └── ...
├── database/migrations/           # Database migrations
├── routes/
│   └── api.php                    # API routes
├── .env                           # Environment config
└── composer.json                  # PHP dependencies
```

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
php artisan serve --port=8001
```

### Database Connection Failed

```bash
# Check MySQL is running
sudo systemctl status mysql

# Create database
mysql -u root -p
> CREATE DATABASE morphvox;
> GRANT ALL ON morphvox.* TO 'morphvox'@'localhost' IDENTIFIED BY 'master123';
> FLUSH PRIVILEGES;
```

### Voice Models Not Syncing

See [VOICE_MODELS.md - Troubleshooting](VOICE_MODELS.md#troubleshooting)

## License

MIT
