<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Foundation\Auth\User as Authenticatable;
use Illuminate\Notifications\Notifiable;
use Laravel\Sanctum\HasApiTokens;
use Spatie\Permission\Traits\HasRoles;

class User extends Authenticatable
{
    use HasApiTokens, HasFactory, Notifiable, HasRoles;

    protected $fillable = [
        'name',
        'email',
        'password',
        'avatar',
    ];

    protected $hidden = [
        'password',
        'remember_token',
    ];

    protected function casts(): array
    {
        return [
            'email_verified_at' => 'datetime',
            'password' => 'hashed',
        ];
    }

    // Relationships
    public function voiceModels(): HasMany
    {
        return $this->hasMany(VoiceModel::class);
    }

    public function jobs(): HasMany
    {
        return $this->hasMany(JobQueue::class);
    }

    public function usageEvents(): HasMany
    {
        return $this->hasMany(UsageEvent::class);
    }

    // Helpers
    public function getPublicModels()
    {
        return $this->voiceModels()->where('visibility', 'public')->where('status', 'ready');
    }

    public function getActiveJobs()
    {
        return $this->jobs()->whereIn('status', [
            JobQueue::STATUS_PENDING,
            JobQueue::STATUS_QUEUED,
            JobQueue::STATUS_PROCESSING,
        ]);
    }

    public function getUsageForPeriod($start, $end)
    {
        return $this->usageEvents()
            ->whereBetween('created_at', [$start, $end])
            ->selectRaw('event_type, SUM(audio_seconds) as total_seconds, SUM(tokens_used) as total_tokens, COUNT(*) as count')
            ->groupBy('event_type')
            ->get();
    }

    public function canUploadModels(): bool
    {
        return $this->hasPermissionTo('upload_models');
    }

    public function canPublishModels(): bool
    {
        return $this->hasPermissionTo('publish_models');
    }

    public function canTrainModels(): bool
    {
        return $this->hasPermissionTo('train_models');
    }
}
