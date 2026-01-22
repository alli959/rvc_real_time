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

- **User Authentication** - Laravel Sanctum with Bearer token support
- **OAuth Integration** - Google and GitHub OAuth login/signup
- **Voice Models Management** - Unified system for local and S3 storage
- **Text-to-Speech** - Edge TTS with 50+ voices, emotion tags, and audio effects
- **Audio Processing** - Voice conversion, vocal separation, and voice swap
- **Job Queue** - Background processing with full history tracking
- **Admin Panel** - User management, model administration, job monitoring
- **Role Requests** - User role upgrade request system
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

### OAuth Configuration

```bash
# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=${APP_URL}/api/auth/google/callback

# GitHub OAuth
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
GITHUB_REDIRECT_URI=${APP_URL}/api/auth/github/callback
```

## API Endpoints

### Authentication

```http
POST   /api/auth/register
POST   /api/auth/login
POST   /api/auth/logout
GET    /api/auth/me
GET    /api/auth/invitation/{token}
POST   /api/auth/register-with-invite

# OAuth Routes
GET    /api/auth/{provider}/redirect   # Get OAuth redirect URL (google, github)
GET    /api/auth/{provider}/callback   # Handle OAuth callback
POST   /api/auth/{provider}/code       # Exchange code for token
```

### Voice Models (System)

```http
GET    /api/voice-models              # List all models
GET    /api/voice-models/stats        # Get statistics
GET    /api/voice-models/config       # Get configuration
GET    /api/voice-models/{slug}       # Get single model
GET    /api/voice-models/my           # My uploaded models (auth)
POST   /api/voice-models              # Create model (auth)
PUT    /api/voice-models/{slug}       # Update model (auth)
DELETE /api/voice-models/{slug}       # Delete model (auth)
POST   /api/voice-models/{slug}/image # Upload model image (auth)
```

### Text-to-Speech

```http
GET    /api/tts/voices                # List available TTS voices (public)
POST   /api/tts/generate              # Generate TTS audio (auth)
POST   /api/tts/stream                # Stream TTS audio (auth)
```

### Audio Processing

```http
POST   /api/audio/process             # Process audio (auth)
                                      # Modes: convert, split, swap
```

### YouTube Integration

```http
POST   /api/youtube/search            # Search for songs
POST   /api/youtube/download          # Download audio
GET    /api/youtube/info/{videoId}    # Get video info
```

### Voice Models

```http
GET    /api/voice-models              # List public models
GET    /api/voice-models/{slug}       # Get model details
GET    /api/voice-models/my           # My models (auth)
GET    /api/voice-models/stats        # Model statistics
POST   /api/voice-models              # Create model (auth)
PUT    /api/voice-models/{id}         # Update model (auth)
DELETE /api/voice-models/{id}         # Delete model (auth)
POST   /api/voice-models/upload       # Direct file upload (auth)
POST   /api/voice-models/{id}/image   # Upload model image (auth)
POST   /api/voice-models/{id}/upload-urls    # Get pre-signed URLs (auth)
POST   /api/voice-models/{id}/confirm-upload # Confirm upload (auth)
GET    /api/voice-models/{id}/download-urls  # Get download URLs (auth)
```

### Jobs

```http
GET    /api/jobs                      # List jobs (auth)
GET    /api/jobs/{uuid}               # Job status (auth)
POST   /api/jobs/inference            # Create inference job (auth)
POST   /api/jobs/{job}/upload-url     # Get upload URL (auth)
POST   /api/jobs/{job}/start          # Start processing (auth)
POST   /api/jobs/{job}/cancel         # Cancel job (auth)
GET    /api/jobs/{job}/output         # Get output download URL (auth)
```

Job types: `tts`, `voice_conversion`, `audio_convert`, `audio_split`, `audio_swap`

### Role Requests

```http
GET    /api/role-requests/available-roles  # Get available roles
GET    /api/role-requests/my               # My requests (auth)
POST   /api/role-requests                  # Submit request (auth)
DELETE /api/role-requests/{id}             # Cancel request (auth)
```

### Admin Routes

```http
GET    /api/admin/users               # List users
PUT    /api/admin/users/{user}        # Update user
DELETE /api/admin/users/{user}        # Delete user
POST   /api/admin/voice-models/sync   # Sync voice models
GET    /api/admin/voice-models/config # Get model config
GET    /api/admin/role-requests       # List role requests
POST   /api/admin/role-requests/{id}/approve  # Approve request
POST   /api/admin/role-requests/{id}/reject   # Reject request
GET    /api/admin/jobs                # All jobs
GET    /api/admin/stats               # System statistics
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
│   ├── Http/Controllers/
│   │   ├── Api/                   # API controllers
│   │   │   ├── AuthController.php
│   │   │   ├── OAuthController.php       # OAuth login
│   │   │   ├── VoiceModelController.php
│   │   │   ├── ModelUploadController.php
│   │   │   ├── TTSController.php
│   │   │   ├── AudioProcessingController.php
│   │   │   ├── JobController.php
│   │   │   ├── RoleRequestController.php
│   │   │   └── YouTubeController.php
│   │   └── Admin/                 # Admin controllers
│   │       ├── DashboardController.php
│   │       ├── UserController.php
│   │       ├── JobsAdminController.php
│   │       └── VoiceModelAdminController.php
│   ├── Models/                    # Eloquent models
│   │   ├── User.php
│   │   ├── VoiceModel.php
│   │   ├── JobQueue.php
│   │   ├── RoleRequest.php
│   │   ├── UsageEvent.php
│   │   └── UserInvitation.php
│   └── Services/                  # Business logic
│       ├── VoiceModelScanner.php
│       ├── VoiceEngineService.php
│       └── StorageService.php
├── config/
│   ├── voice_models.php           # Voice models config
│   ├── cors.php                   # CORS config
│   ├── admin.php                  # Admin panel config
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
