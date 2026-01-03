<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\Storage;

class SystemVoiceModel extends Model
{
    use HasFactory;

    protected $fillable = [
        'slug',
        'name',
        'description',
        'model_file',
        'model_path',
        'index_file',
        'index_path',
        'has_index',
        'size_bytes',
        'storage_type',
        'storage_path',
        'index_storage_path',
        'engine',
        'metadata',
        'is_active',
        'is_featured',
        'usage_count',
        'last_synced_at',
    ];

    protected $casts = [
        'has_index' => 'boolean',
        'is_active' => 'boolean',
        'is_featured' => 'boolean',
        'metadata' => 'array',
        'last_synced_at' => 'datetime',
        'size_bytes' => 'integer',
        'usage_count' => 'integer',
    ];

    protected $appends = ['size', 'download_url', 'index_download_url'];

    /**
     * Human-readable file size
     */
    public function getSizeAttribute(): string
    {
        $bytes = $this->size_bytes;
        if ($bytes >= 1073741824) {
            return number_format($bytes / 1073741824, 2) . ' GB';
        } elseif ($bytes >= 1048576) {
            return number_format($bytes / 1048576, 2) . ' MB';
        } elseif ($bytes >= 1024) {
            return number_format($bytes / 1024, 2) . ' KB';
        }
        return $bytes . ' bytes';
    }

    /**
     * Get download URL for the model file
     */
    public function getDownloadUrlAttribute(): ?string
    {
        if ($this->storage_type === 's3') {
            if ($this->storage_path) {
                $disk = config('voice_models.s3.disk', 's3');
                $expiration = config('voice_models.s3.url_expiration', 60);
                try {
                    return Storage::disk($disk)->temporaryUrl($this->storage_path, now()->addMinutes($expiration));
                } catch (\Exception $e) {
                    // Fall back to regular URL if temporaryUrl not supported
                    return Storage::disk($disk)->url($this->storage_path);
                }
            }
        }
        
        // For local storage, return the absolute path
        return $this->model_path;
    }

    /**
     * Get download URL for the index file
     */
    public function getIndexDownloadUrlAttribute(): ?string
    {
        if (!$this->has_index) {
            return null;
        }

        if ($this->storage_type === 's3') {
            if ($this->index_storage_path) {
                $disk = config('voice_models.s3.disk', 's3');
                $expiration = config('voice_models.s3.url_expiration', 60);
                try {
                    return Storage::disk($disk)->temporaryUrl($this->index_storage_path, now()->addMinutes($expiration));
                } catch (\Exception $e) {
                    return Storage::disk($disk)->url($this->index_storage_path);
                }
            }
        }

        return $this->index_path;
    }

    /**
     * Check if model is stored locally
     */
    public function isLocal(): bool
    {
        return $this->storage_type === 'local';
    }

    /**
     * Check if model is stored on S3
     */
    public function isS3(): bool
    {
        return $this->storage_type === 's3';
    }

    /**
     * Scope for active models only
     */
    public function scopeActive($query)
    {
        return $query->where('is_active', true);
    }

    /**
     * Scope for featured models
     */
    public function scopeFeatured($query)
    {
        return $query->where('is_featured', true);
    }

    /**
     * Scope by engine type
     */
    public function scopeEngine($query, string $engine)
    {
        return $query->where('engine', $engine);
    }

    /**
     * Scope by storage type
     */
    public function scopeStorageType($query, string $type)
    {
        return $query->where('storage_type', $type);
    }

    /**
     * Increment usage counter
     */
    public function incrementUsage(): void
    {
        $this->increment('usage_count');
    }
}
