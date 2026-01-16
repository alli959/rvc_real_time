<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\BelongsToMany;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Support\Str;

class VoiceModel extends Model
{
    use HasFactory, SoftDeletes;

    protected $fillable = [
        'uuid',
        'user_id',      // NULL = system model, otherwise user-uploaded
        'name',
        'slug',
        'description',
        'image_path',   // path to model avatar/cover image
        'gender',       // optional: male, female (for TTS default selection)
        'avatar',
        'engine',
        'visibility',   // public, private, unlisted

        // Storage paths
        'model_path',   // path to .pth file
        'index_path',   // path to .index file
        'config_path',
        'has_index',
        'size_bytes',
        'storage_type', // local, s3

        // Admin flags
        'is_active',
        'is_featured',
        'last_synced_at',

        // Metadata
        'metadata',
        'tags',
        'status',       // pending, ready, failed
        'usage_count',
        'download_count',
        'has_consent',
        'consent_notes',

        // Language readiness scores
        'en_readiness_score',
        'en_phoneme_coverage',
        'en_missing_phonemes',
        'is_readiness_score',
        'is_phoneme_coverage',
        'is_missing_phonemes',
        'language_scan_results',
        'language_scanned_at',

        // Inference test results
        'inference_test_score',
        'inference_test_results',
        'inference_tested_at',
        'en_inference_score',
        'is_inference_score',
    ];

    protected $casts = [
        'metadata' => 'array',
        'tags' => 'array',
        'has_consent' => 'boolean',
        'has_index' => 'boolean',
        'is_active' => 'boolean',
        'is_featured' => 'boolean',
        'last_synced_at' => 'datetime',
        'size_bytes' => 'integer',
        'usage_count' => 'integer',
        'download_count' => 'integer',
        // Language readiness
        'en_readiness_score' => 'decimal:2',
        'en_phoneme_coverage' => 'decimal:2',
        'en_missing_phonemes' => 'array',
        'is_readiness_score' => 'decimal:2',
        'is_phoneme_coverage' => 'decimal:2',
        'is_missing_phonemes' => 'array',
        'language_scan_results' => 'array',
        'language_scanned_at' => 'datetime',
        // Inference test results
        'inference_test_score' => 'decimal:2',
        'inference_test_results' => 'array',
        'inference_tested_at' => 'datetime',
        'en_inference_score' => 'decimal:2',
        'is_inference_score' => 'decimal:2',
    ];

    protected $hidden = [
        'config_path',
    ];

    protected $appends = [
        'model_file',
        'index_file',
        'size',
        'image_url',
    ];

    /**
     * Get the model filename from model_path
     */
    public function getModelFileAttribute(): ?string
    {
        return $this->model_path ? basename($this->model_path) : null;
    }

    /**
     * Get the index filename from index_path
     */
    public function getIndexFileAttribute(): ?string
    {
        return $this->index_path ? basename($this->index_path) : null;
    }

    /**
     * Get the full URL to the model image
     */
    public function getImageUrlAttribute(): ?string
    {
        if (!$this->image_path) {
            return null;
        }

        // If it's already a full URL, return as-is
        if (str_starts_with($this->image_path, 'http')) {
            return $this->image_path;
        }

        // Return URL via public storage
        return url('storage/' . $this->image_path);
    }

    /**
     * Get human-readable size
     */
    public function getSizeAttribute(): string
    {
        if (!$this->size_bytes) {
            return 'Unknown';
        }

        $bytes = $this->size_bytes;
        $units = ['B', 'KB', 'MB', 'GB'];
        $i = 0;

        while ($bytes >= 1024 && $i < count($units) - 1) {
            $bytes /= 1024;
            $i++;
        }

        return round($bytes, 1) . ' ' . $units[$i];
    }

    protected static function boot()
    {
        parent::boot();

        static::creating(function ($model) {
            if (empty($model->uuid)) {
                $model->uuid = (string) Str::uuid();
            }
            if (empty($model->slug)) {
                // Generate clean slug from name (e.g., "Bjarni Ben" -> "bjarni-ben")
                $baseSlug = Str::slug($model->name);
                
                // Check if slug already exists (including soft-deleted records)
                $existingCount = static::withTrashed()
                    ->where('slug', $baseSlug)
                    ->orWhere('slug', 'like', $baseSlug . '-%')
                    ->count();
                
                if ($existingCount === 0) {
                    // No conflicts - use clean slug
                    $model->slug = $baseSlug;
                } else {
                    // Conflict exists - append version number
                    $model->slug = $baseSlug . '-' . ($existingCount + 1);
                }
            }
        });
    }

    /**
     * Check if this is a system model (not user-uploaded)
     */
    public function isSystemModel(): bool
    {
        return is_null($this->user_id);
    }

    // Relationships
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function jobs(): HasMany
    {
        return $this->hasMany(JobQueue::class, 'voice_model_id');
    }

    public function usageEvents(): HasMany
    {
        return $this->hasMany(UsageEvent::class);
    }

    public function permittedUsers(): BelongsToMany
    {
        return $this->belongsToMany(User::class, 'voice_model_user_access', 'voice_model_id', 'user_id')
            ->withPivot(['can_view', 'can_use'])
            ->withTimestamps();
    }

    // Scopes
    public function scopePublic($query)
    {
        return $query->where('visibility', 'public')
            ->where('status', 'ready')
            ->where('is_active', true);
    }

    public function scopeSystem($query)
    {
        return $query->whereNull('user_id');
    }

    public function scopeUserUploaded($query)
    {
        return $query->whereNotNull('user_id');
    }

    public function scopeActive($query)
    {
        return $query->where('is_active', true);
    }

    public function scopeOwnedBy($query, $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function scopeAccessibleBy($query, $userId)
    {
        // Check if user is admin - admins can see ALL models (including inactive)
        $user = User::find($userId);
        if ($user && $user->hasRole('admin')) {
            // Admin sees everything - no filters
            return $query;
        }

        return $query
            ->where('status', 'ready')
            ->where('is_active', true)
            ->where(function ($q) use ($userId) {
                $q->where('visibility', 'public')
                    ->orWhere('user_id', $userId)
                    ->orWhereHas('permittedUsers', function ($sub) use ($userId) {
                        $sub->where('users.id', $userId)
                            ->where('voice_model_user_access.can_view', true);
                    });
            });
    }

    public function scopeUsableBy($query, $userId)
    {
        // Check if user is admin - admins can use all models
        $user = User::find($userId);
        if ($user && $user->hasRole('admin')) {
            return $query->where('status', 'ready');
        }

        return $query
            ->where('status', 'ready')
            ->where('is_active', true)
            ->where(function ($q) use ($userId) {
                // If a model is public, it can be used.
                $q->where('visibility', 'public')
                    ->orWhere('user_id', $userId)
                    ->orWhereHas('permittedUsers', function ($sub) use ($userId) {
                        $sub->where('users.id', $userId)
                            ->where('voice_model_user_access.can_use', true);
                    });
            });
    }

    // Helpers
    public function isOwnedBy($user): bool
    {
        return $this->user_id === ($user->id ?? $user);
    }

    public function isPublic(): bool
    {
        return $this->visibility === 'public';
    }

    public function isReady(): bool
    {
        return $this->status === 'ready';
    }

    public function getStoragePrefix(): string
    {
        if ($this->user_id) {
            return "users/{$this->user_id}/models/{$this->uuid}";
        }
        return "public/models/{$this->uuid}";
    }

    public function incrementUsage(): void
    {
        $this->increment('usage_count');
    }

    public function incrementDownloads(): void
    {
        $this->increment('download_count');
    }
}
