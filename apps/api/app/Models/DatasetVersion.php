<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Support\Str;

/**
 * DatasetVersion Model
 * 
 * Tracks versions of training datasets for a voice model.
 * Each version represents a snapshot of audio files used for training.
 * 
 * @property int $id
 * @property string $uuid
 * @property int $voice_model_id
 * @property int $version_number
 * @property string $manifest_hash
 * @property int $audio_count
 * @property float $total_duration_seconds
 * @property int $segment_count
 * @property int $sample_rate
 * @property array $preprocessing_config
 * @property string $status
 * @property array|null $metadata
 * @property \Carbon\Carbon $created_at
 * @property \Carbon\Carbon $updated_at
 * 
 * @property-read VoiceModel $voiceModel
 * @property-read \Illuminate\Database\Eloquent\Collection|TrainingRun[] $trainingRuns
 */
class DatasetVersion extends Model
{
    protected $fillable = [
        'uuid',
        'voice_model_id',
        'version_number',
        'manifest_hash',
        'audio_count',
        'total_duration_seconds',
        'segment_count',
        'sample_rate',
        'preprocessing_config',
        'status',
        'metadata',
    ];

    protected $casts = [
        'version_number' => 'integer',
        'audio_count' => 'integer',
        'total_duration_seconds' => 'decimal:2',
        'segment_count' => 'integer',
        'sample_rate' => 'integer',
        'preprocessing_config' => 'array',
        'metadata' => 'array',
    ];

    // Status constants
    const STATUS_PENDING = 'pending';
    const STATUS_PROCESSING = 'processing';
    const STATUS_READY = 'ready';
    const STATUS_FAILED = 'failed';

    /**
     * Boot the model.
     */
    protected static function boot()
    {
        parent::boot();

        static::creating(function ($model) {
            if (empty($model->uuid)) {
                $model->uuid = (string) Str::uuid();
            }
        });
    }

    /**
     * Get the voice model this dataset version belongs to.
     */
    public function voiceModel(): BelongsTo
    {
        return $this->belongsTo(VoiceModel::class);
    }

    /**
     * Get all training runs that used this dataset version.
     */
    public function trainingRuns(): HasMany
    {
        return $this->hasMany(TrainingRun::class);
    }

    /**
     * Get the duration formatted as human readable string.
     */
    public function getFormattedDurationAttribute(): string
    {
        $seconds = (int) $this->total_duration_seconds;
        $hours = floor($seconds / 3600);
        $minutes = floor(($seconds % 3600) / 60);
        $secs = $seconds % 60;

        if ($hours > 0) {
            return sprintf('%d:%02d:%02d', $hours, $minutes, $secs);
        }
        return sprintf('%d:%02d', $minutes, $secs);
    }

    /**
     * Get the next version number for a voice model.
     */
    public static function getNextVersionNumber(int $voiceModelId): int
    {
        $maxVersion = static::where('voice_model_id', $voiceModelId)->max('version_number');
        return ($maxVersion ?? 0) + 1;
    }

    /**
     * Check if this is the latest version for the voice model.
     */
    public function isLatest(): bool
    {
        $latestVersion = static::where('voice_model_id', $this->voice_model_id)
            ->max('version_number');
        return $this->version_number === $latestVersion;
    }

    /**
     * Scope to get ready datasets only.
     */
    public function scopeReady($query)
    {
        return $query->where('status', self::STATUS_READY);
    }
}
