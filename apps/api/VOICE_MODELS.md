# Voice Models System

The VoxMorph voice models system provides a unified approach to managing server-side voice models, supporting both local directory storage and cloud storage (S3-compatible).

## Architecture

### Storage Options

The system supports two storage backends that can be configured via environment variables:

1. **Local Storage** - Models stored in a local directory (default)
2. **S3 Storage** - Models stored in S3-compatible cloud storage (AWS S3, MinIO, etc.)

### Key Components

| Component | Purpose |
|-----------|---------|
| `SystemVoiceModel` | Eloquent model representing a voice model |
| `VoiceModelScanner` | Service that scans storage and extracts model metadata |
| `SyncVoiceModels` | Artisan command to sync models to database |
| `SystemVoiceModelController` | REST API for accessing models |

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Storage Type
# Options: "local" or "s3"
VOICE_MODELS_STORAGE=local

# Local Storage Configuration
# Path can be absolute or relative to api root
VOICE_MODELS_LOCAL_PATH=../../services/voice-engine/assets/models

# S3 Storage Configuration
VOICE_MODELS_S3_DISK=s3
VOICE_MODELS_S3_PREFIX=models
VOICE_MODELS_S3_URL_EXPIRATION=60

# Default Engine
VOICE_MODELS_DEFAULT_ENGINE=rvc
```

### Config File

The configuration is centralized in `config/voice_models.php`:

```php
<?php

return [
    'storage' => env('VOICE_MODELS_STORAGE', 'local'),
    
    'local' => [
        'path' => env('VOICE_MODELS_LOCAL_PATH', base_path('../services/voice-engine/assets/models')),
    ],
    
    's3' => [
        'disk' => env('VOICE_MODELS_S3_DISK', 's3'),
        'prefix' => env('VOICE_MODELS_S3_PREFIX', 'models'),
        'url_expiration' => env('VOICE_MODELS_S3_URL_EXPIRATION', 60),
    ],
    
    'model_extensions' => ['pth', 'onnx'],
    'index_extensions' => ['index'],
    'default_engine' => env('VOICE_MODELS_DEFAULT_ENGINE', 'rvc'),
];
```

## Model Directory Structure

### Local Storage

Models should be organized in folders:

```
models/
├── BillCipher/
│   ├── BillCipher.pth          # Model file
│   └── BillCipher.index        # Index file (optional)
├── Donald-Trump/
│   ├── Trump_e160_s7520.pth
│   └── added_IVF1377_Flat_nprobe_1_Trump_v2.index
└── sigurgeir-0.5-model/
    ├── G_1360_infer.pth        # Preferred: *_infer.pth
    ├── added_IVF673_Flat_nprobe_1_v2.index
    └── config.json              # Optional metadata
```

### S3 Storage

Same structure in S3 bucket under the configured prefix:

```
s3://bucket-name/models/
├── BillCipher/
│   ├── BillCipher.pth
│   └── BillCipher.index
└── ...
```

## Syncing Models

### Initial Sync

Scan storage and populate the database:

```bash
cd apps/api
php artisan voice-models:sync
```

### Command Options

```bash
# Sync with specific storage type
php artisan voice-models:sync --storage=local
php artisan voice-models:sync --storage=s3

# Override local path
php artisan voice-models:sync --path=/custom/path/to/models

# Remove database entries for models no longer in storage
php artisan voice-models:sync --prune

# Force re-sync all models (even if unchanged)
php artisan voice-models:sync --force
```

### Example Output

```
Storage type: local
Scanning local directory: /path/to/models

Found 15 models
 15/15 [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 100%

Sync complete!
+-----------+-------+
| Action    | Count |
+-----------+-------+
| Created   | 15    |
| Updated   | 0     |
| Unchanged | 0     |
| Pruned    | 0     |
+-----------+-------+
```

### Scheduled Syncing

Add to `app/Console/Kernel.php` to sync automatically:

```php
protected function schedule(Schedule $schedule): void
{
    // Sync models every hour
    $schedule->command('voice-models:sync')->hourly();
    
    // Or daily with pruning
    $schedule->command('voice-models:sync --prune')->dailyAt('03:00');
}
```

## REST API

### Public Endpoints

#### List Models

```http
GET /api/voice-models
```

Query parameters:
- `search` - Search by name or slug
- `engine` - Filter by engine (e.g., "rvc")
- `storage_type` - Filter by storage ("local" or "s3")
- `has_index` - Filter by index file presence (true/false)
- `featured` - Show only featured models (true/false)
- `sort` - Sort field (name, size_bytes, usage_count, created_at)
- `direction` - Sort direction (asc, desc)
- `per_page` - Results per page (default: 50, max: 100)
- `all` - Return all results without pagination (true/false)

Example:
```bash
curl "http://localhost:8000/api/voice-models?search=Bill&has_index=true"
```

Response:
```json
{
  "data": [
    {
      "id": 1,
      "slug": "BillCipher",
      "name": "BillCipher",
      "description": null,
      "model_file": "BillCipher.pth",
      "model_path": "/path/to/BillCipher.pth",
      "index_file": "BillCipher.index",
      "has_index": true,
      "size": "53.67 MB",
      "size_bytes": 56271950,
      "storage_type": "local",
      "engine": "rvc",
      "is_active": true,
      "is_featured": false,
      "usage_count": 0,
      "download_url": "/path/to/BillCipher.pth",
      "index_download_url": "/path/to/BillCipher.index",
      "metadata": null,
      "created_at": "2026-01-02T21:58:15Z",
      "updated_at": "2026-01-02T21:58:15Z"
    }
  ],
  "total": 1,
  "per_page": 50,
  "current_page": 1,
  "last_page": 1
}
```

#### Get Single Model

```http
GET /api/voice-models/{slug}
```

Example:
```bash
curl "http://localhost:8000/api/voice-models/BillCipher"
```

#### Get Statistics

```http
GET /api/voice-models/stats
```

Response:
```json
{
  "total": 15,
  "active": 15,
  "featured": 2,
  "with_index": 14,
  "by_engine": {
    "rvc": 15
  },
  "by_storage": {
    "local": 15,
    "s3": 0
  },
  "total_size_bytes": 5368709120,
  "last_synced": "2026-01-02T21:58:15Z",
  "configured_storage": "local"
}
```

#### Get Configuration

```http
GET /api/voice-models/config
```

Response:
```json
{
  "storage": "local",
  "local_path": "/path/to/models",
  "s3_disk": "s3",
  "s3_prefix": "models",
  "default_engine": "rvc"
}
```

### Protected Endpoints (Require Authentication)

#### Trigger Sync

```http
POST /api/voice-models/sync
Authorization: Bearer {token}
Content-Type: application/json

{
  "prune": false,
  "storage": "local"
}
```

#### Update Model Metadata

```http
PATCH /api/voice-models/{slug}
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "Custom Name",
  "description": "This is a great voice model",
  "is_active": true,
  "is_featured": true
}
```

## Model Detection Rules

The scanner follows these rules when detecting models:

### Model Files (.pth, .onnx)

Priority order:
1. Files ending with `*_infer.pth` (inference-optimized)
2. Highest numbered `G_*.pth` (generator checkpoint)
3. Any `.pth` file except `D_*.pth` (discriminator)

### Index Files (.index)

Priority order:
1. Files starting with `added_*` (trained index)
2. Files starting with `trained_*`
3. Any `.index` file

### Metadata Extraction

The scanner automatically extracts:
- **Epochs and steps** from filename patterns (e.g., `model_e160_s7520.pth` → epochs: 160, steps: 7520)
- **Config** from `config.json` if present in model directory
- **File sizes** for storage tracking

## Switching Storage Backends

### From Local to S3

1. Update `.env`:
```bash
VOICE_MODELS_STORAGE=s3
VOICE_MODELS_S3_DISK=s3
VOICE_MODELS_S3_PREFIX=models
```

2. Upload models to S3:
```bash
aws s3 sync /local/models/ s3://your-bucket/models/
```

3. Sync database:
```bash
php artisan voice-models:sync --prune
```

### From S3 to Local

1. Download models from S3:
```bash
aws s3 sync s3://your-bucket/models/ /local/models/
```

2. Update `.env`:
```bash
VOICE_MODELS_STORAGE=local
VOICE_MODELS_LOCAL_PATH=../../services/voice-engine/assets/models
```

3. Sync database:
```bash
php artisan voice-models:sync --prune
```

## Database Schema

The `system_voice_models` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | bigint | Primary key |
| `slug` | string | Unique identifier (folder name) |
| `name` | string | Display name |
| `description` | text | Optional description |
| `model_file` | string | Model filename |
| `model_path` | string | Absolute path or URL |
| `index_file` | string | Index filename (nullable) |
| `index_path` | string | Index path or URL (nullable) |
| `has_index` | boolean | Has index file |
| `size_bytes` | bigint | File size in bytes |
| `storage_type` | string | "local" or "s3" |
| `storage_path` | string | Relative path in storage |
| `index_storage_path` | string | Index relative path |
| `engine` | string | Voice engine type |
| `metadata` | json | Additional metadata |
| `is_active` | boolean | Is model active |
| `is_featured` | boolean | Is model featured |
| `usage_count` | integer | Usage counter |
| `last_synced_at` | timestamp | Last sync time |
| `created_at` | timestamp | Creation time |
| `updated_at` | timestamp | Update time |

## Frontend Integration

### API Client (TypeScript)

```typescript
import { voiceModelsApi, SystemVoiceModel } from '@/lib/api';

// List models
const { data } = await voiceModelsApi.list({ 
  search: 'Bill',
  has_index: true 
});

// Get single model
const { model } = await voiceModelsApi.get('BillCipher');

// Get stats
const stats = await voiceModelsApi.stats();

// Trigger sync (authenticated)
await voiceModelsApi.sync({ prune: true });
```

### React Query Hook

```typescript
import { useQuery } from '@tanstack/react-query';
import { voiceModelsApi } from '@/lib/api';

function ModelsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['voice-models', search],
    queryFn: () => voiceModelsApi.list({ search }),
  });

  const models: SystemVoiceModel[] = data?.data || [];
  
  // Render models...
}
```

## Troubleshooting

### Models Not Appearing

1. Check storage configuration:
```bash
php artisan voice-models:config
```

2. Verify path exists:
```bash
# Local
ls -la /path/to/models

# S3
aws s3 ls s3://bucket/models/
```

3. Run sync with verbose output:
```bash
php artisan voice-models:sync -v
```

### Permission Issues

For local storage, ensure Laravel has read access:
```bash
chmod -R 755 /path/to/models
chown -R www-data:www-data /path/to/models
```

For S3, verify IAM permissions include:
- `s3:ListBucket`
- `s3:GetObject`
- `s3:GetObjectAttributes`

### Slow Sync

For large model collections:
- Use `--force` only when necessary
- Consider increasing PHP memory limit
- Index S3 buckets for faster listing

### Stale Data

Force a complete re-sync:
```bash
php artisan voice-models:sync --force --prune
```

## Best Practices

1. **Use symlinks** for models shared across multiple locations
2. **Run sync regularly** via scheduler or webhook
3. **Enable pruning** to remove deleted models from database
4. **Use featured flag** to highlight recommended models
5. **Add descriptions** via API to help users choose models
6. **Monitor storage costs** when using S3
7. **Cache model lists** on frontend to reduce API calls
8. **Use pagination** for large model collections

## Migration Guide

If you have existing `local_voice_models` or separate model tables:

```bash
# The migration automatically renames the table
php artisan migrate

# Clear and re-sync
php artisan tinker
>>> App\Models\SystemVoiceModel::truncate();
>>> exit

php artisan voice-models:sync
```

## See Also

- [Voice Engine Documentation](../../services/voice-engine/README.md)
- [API Routes](routes/api.php)
- [Model Scanner Service](app/Services/VoiceModelScanner.php)
- [System Voice Model](app/Models/SystemVoiceModel.php)
