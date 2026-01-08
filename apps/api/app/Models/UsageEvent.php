<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class UsageEvent extends Model
{
    public $timestamps = false;

    protected $fillable = [
        'user_id',
        'voice_model_id',
        'job_id',
        'event_type',
        'audio_seconds',
        'tokens_used',
        'cost',
        'billing_period',
        'metadata',
        'created_at',
    ];

    protected $casts = [
        'metadata' => 'array',
        'cost' => 'decimal:6',
        'created_at' => 'datetime',
    ];

    // Event type constants
    const TYPE_INFERENCE = 'inference';
    const TYPE_TRAINING = 'training';
    const TYPE_DOWNLOAD = 'download';
    const TYPE_API_CALL = 'api_call';
    const TYPE_TTS = 'tts';

    protected static function boot()
    {
        parent::boot();

        static::creating(function ($model) {
            if (empty($model->created_at)) {
                $model->created_at = now();
            }
        });
    }

    // Relationships
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function voiceModel(): BelongsTo
    {
        return $this->belongsTo(VoiceModel::class);
    }

    public function job(): BelongsTo
    {
        return $this->belongsTo(JobQueue::class, 'job_id');
    }

    // Scopes
    public function scopeForUser($query, $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function scopeInPeriod($query, $start, $end)
    {
        return $query->whereBetween('created_at', [$start, $end]);
    }

    public function scopeOfType($query, $type)
    {
        return $query->where('event_type', $type);
    }

    // Factory methods
    public static function recordInference(int $userId, ?int $modelId, int $audioSeconds, ?int $jobId = null): self
    {
        return static::create([
            'user_id' => $userId,
            'voice_model_id' => $modelId,
            'job_id' => $jobId,
            'event_type' => self::TYPE_INFERENCE,
            'audio_seconds' => $audioSeconds,
        ]);
    }

    public static function recordDownload(int $userId, int $modelId): self
    {
        return static::create([
            'user_id' => $userId,
            'voice_model_id' => $modelId,
            'event_type' => self::TYPE_DOWNLOAD,
        ]);
    }

    public static function recordTTS(int $userId, ?int $modelId, int $textLength, bool $withConversion = false): self
    {
        return static::create([
            'user_id' => $userId,
            'voice_model_id' => $modelId,
            'event_type' => self::TYPE_TTS,
            'tokens_used' => $textLength,
            'metadata' => [
                'text_length' => $textLength,
                'with_conversion' => $withConversion,
            ],
        ]);
    }
}
