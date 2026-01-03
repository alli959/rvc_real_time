<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Support\Str;

class VoiceModel extends Model
{
    use HasFactory, SoftDeletes;

    protected $fillable = [
        'uuid',
        'user_id',
        'name',
        'slug',
        'description',
        'avatar',
        'engine',
        'visibility',
        'model_path',
        'index_path',
        'config_path',
        'metadata',
        'tags',
        'status',
        'has_consent',
        'consent_notes',
    ];

    protected $casts = [
        'metadata' => 'array',
        'tags' => 'array',
        'has_consent' => 'boolean',
    ];

    protected $hidden = [
        'model_path',
        'index_path',
        'config_path',
    ];

    protected static function boot()
    {
        parent::boot();

        static::creating(function ($model) {
            if (empty($model->uuid)) {
                $model->uuid = (string) Str::uuid();
            }
            if (empty($model->slug)) {
                $model->slug = Str::slug($model->name) . '-' . Str::random(6);
            }
        });
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

    // Scopes
    public function scopePublic($query)
    {
        return $query->where('visibility', 'public')->where('status', 'ready');
    }

    public function scopeOwnedBy($query, $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function scopeAccessibleBy($query, $userId)
    {
        return $query->where(function ($q) use ($userId) {
            $q->where('visibility', 'public')
              ->orWhere('user_id', $userId);
        })->where('status', 'ready');
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
