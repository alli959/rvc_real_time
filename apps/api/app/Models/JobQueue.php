<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Str;

class JobQueue extends Model
{
    use HasFactory;

    protected $table = 'jobs_queue';

    protected $fillable = [
        'uuid',
        'user_id',
        'voice_model_id',
        'type',
        'status',
        'input_path',
        'output_path',
        'parameters',
        'progress',
        'progress_message',
        'started_at',
        'completed_at',
        'error_message',
        'error_details',
        'worker_id',
    ];

    protected $casts = [
        'parameters' => 'array',
        'error_details' => 'array',
        'started_at' => 'datetime',
        'completed_at' => 'datetime',
    ];

    protected static function boot()
    {
        parent::boot();

        static::creating(function ($model) {
            if (empty($model->uuid)) {
                $model->uuid = (string) Str::uuid();
            }
        });
    }

    // Status constants
    const STATUS_PENDING = 'pending';
    const STATUS_QUEUED = 'queued';
    const STATUS_PROCESSING = 'processing';
    const STATUS_COMPLETED = 'completed';
    const STATUS_FAILED = 'failed';
    const STATUS_CANCELLED = 'cancelled';

    // Type constants
    const TYPE_INFERENCE = 'inference';
    const TYPE_TRAINING = 'training';
    const TYPE_PREPROCESSING = 'preprocessing';
    const TYPE_TTS = 'tts';
    const TYPE_AUDIO_CONVERT = 'audio_convert';
    const TYPE_AUDIO_SPLIT = 'audio_split';
    const TYPE_AUDIO_SWAP = 'audio_swap';

    // Relationships
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function voiceModel(): BelongsTo
    {
        return $this->belongsTo(VoiceModel::class);
    }

    // Scopes
    public function scopePending($query)
    {
        return $query->where('status', self::STATUS_PENDING);
    }

    public function scopeProcessing($query)
    {
        return $query->where('status', self::STATUS_PROCESSING);
    }

    public function scopeForUser($query, $userId)
    {
        return $query->where('user_id', $userId);
    }

    // Status helpers
    public function isPending(): bool
    {
        return $this->status === self::STATUS_PENDING;
    }

    public function isProcessing(): bool
    {
        return $this->status === self::STATUS_PROCESSING;
    }

    public function isCompleted(): bool
    {
        return $this->status === self::STATUS_COMPLETED;
    }

    public function isFailed(): bool
    {
        return $this->status === self::STATUS_FAILED;
    }

    public function isFinished(): bool
    {
        return in_array($this->status, [
            self::STATUS_COMPLETED,
            self::STATUS_FAILED,
            self::STATUS_CANCELLED
        ]);
    }

    // State transitions
    public function markAsQueued(): void
    {
        $this->update(['status' => self::STATUS_QUEUED]);
    }

    public function markAsProcessing(string $workerId = null): void
    {
        $this->update([
            'status' => self::STATUS_PROCESSING,
            'started_at' => now(),
            'worker_id' => $workerId,
        ]);
    }

    public function markAsCompleted(string $outputPath = null): void
    {
        $this->update([
            'status' => self::STATUS_COMPLETED,
            'completed_at' => now(),
            'progress' => 100,
            'output_path' => $outputPath ?? $this->output_path,
        ]);
    }

    public function markAsFailed(string $errorMessage, array $errorDetails = null): void
    {
        $this->update([
            'status' => self::STATUS_FAILED,
            'completed_at' => now(),
            'error_message' => $errorMessage,
            'error_details' => $errorDetails,
        ]);
    }

    public function updateProgress(int $progress, string $message = null): void
    {
        $this->update([
            'progress' => min(100, max(0, $progress)),
            'progress_message' => $message,
        ]);
    }

    public function getDuration(): ?int
    {
        if (!$this->started_at || !$this->completed_at) {
            return null;
        }
        return $this->started_at->diffInSeconds($this->completed_at);
    }
}
