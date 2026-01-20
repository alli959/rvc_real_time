<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Support\Str;

/**
 * TrainingRun Model
 * 
 * Represents a single training session for a voice model.
 * Supports git-like branching: runs can branch from checkpoints of parent runs.
 * 
 * @property int $id
 * @property string $uuid
 * @property int $voice_model_id
 * @property int|null $dataset_version_id
 * @property int|null $parent_run_id
 * @property int|null $parent_checkpoint_id
 * @property string $run_number
 * @property string $mode
 * @property string $status
 * @property array $config_snapshot
 * @property int $target_epochs
 * @property int $completed_epochs
 * @property int $start_epoch
 * @property int|null $best_checkpoint_id
 * @property float|null $best_loss
 * @property int|null $duration_seconds
 * @property string|null $error_message
 * @property array|null $metadata
 * @property \Carbon\Carbon $created_at
 * @property \Carbon\Carbon $updated_at
 * @property \Carbon\Carbon|null $started_at
 * @property \Carbon\Carbon|null $completed_at
 * 
 * @property-read VoiceModel $voiceModel
 * @property-read DatasetVersion|null $datasetVersion
 * @property-read TrainingRun|null $parentRun
 * @property-read TrainingCheckpoint|null $parentCheckpoint
 * @property-read TrainingCheckpoint|null $bestCheckpoint
 * @property-read \Illuminate\Database\Eloquent\Collection|TrainingRun[] $childRuns
 * @property-read \Illuminate\Database\Eloquent\Collection|TrainingCheckpoint[] $checkpoints
 */
class TrainingRun extends Model
{
    protected $fillable = [
        'uuid',
        'voice_model_id',
        'dataset_version_id',
        'parent_run_id',
        'parent_checkpoint_id',
        'run_number',
        'mode',
        'status',
        'config_snapshot',
        'target_epochs',
        'completed_epochs',
        'start_epoch',
        'best_checkpoint_id',
        'best_loss',
        'duration_seconds',
        'error_message',
        'metadata',
        'started_at',
        'completed_at',
    ];

    protected $casts = [
        'config_snapshot' => 'array',
        'metadata' => 'array',
        'target_epochs' => 'integer',
        'completed_epochs' => 'integer',
        'start_epoch' => 'integer',
        'best_loss' => 'decimal:6',
        'duration_seconds' => 'integer',
        'started_at' => 'datetime',
        'completed_at' => 'datetime',
    ];

    // Mode constants
    const MODE_NEW = 'new';
    const MODE_RESUME = 'resume';       // Continue same dataset, same checkpoint
    const MODE_CONTINUE = 'continue';   // Same dataset, but fresh start (fine-tune)
    const MODE_BRANCH = 'branch';       // New dataset from existing checkpoint

    // Status constants
    const STATUS_PENDING = 'pending';
    const STATUS_PREPARING = 'preparing';
    const STATUS_TRAINING = 'training';
    const STATUS_PAUSED = 'paused';
    const STATUS_COMPLETED = 'completed';
    const STATUS_FAILED = 'failed';
    const STATUS_CANCELLED = 'cancelled';

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
            if (empty($model->run_number)) {
                $model->run_number = static::generateRunNumber($model->voice_model_id);
            }
        });
    }

    /**
     * Get the voice model this run belongs to.
     */
    public function voiceModel(): BelongsTo
    {
        return $this->belongsTo(VoiceModel::class);
    }

    /**
     * Get the dataset version used for this run.
     */
    public function datasetVersion(): BelongsTo
    {
        return $this->belongsTo(DatasetVersion::class);
    }

    /**
     * Get the parent run (if this is a branch/resume/continue).
     */
    public function parentRun(): BelongsTo
    {
        return $this->belongsTo(TrainingRun::class, 'parent_run_id');
    }

    /**
     * Get the checkpoint this run branched from.
     */
    public function parentCheckpoint(): BelongsTo
    {
        return $this->belongsTo(TrainingCheckpoint::class, 'parent_checkpoint_id');
    }

    /**
     * Get the best checkpoint from this run.
     */
    public function bestCheckpoint(): BelongsTo
    {
        return $this->belongsTo(TrainingCheckpoint::class, 'best_checkpoint_id');
    }

    /**
     * Get all child runs that branched from this run.
     */
    public function childRuns(): HasMany
    {
        return $this->hasMany(TrainingRun::class, 'parent_run_id');
    }

    /**
     * Get all checkpoints from this run.
     */
    public function checkpoints(): HasMany
    {
        return $this->hasMany(TrainingCheckpoint::class);
    }

    /**
     * Generate a unique run number for a voice model.
     * Format: run-001, run-002, etc.
     */
    public static function generateRunNumber(int $voiceModelId): string
    {
        $count = static::where('voice_model_id', $voiceModelId)->count();
        return sprintf('run-%03d', $count + 1);
    }

    /**
     * Get the display name for the run mode.
     */
    public function getModeDisplayAttribute(): string
    {
        return match($this->mode) {
            self::MODE_NEW => 'Fresh Training',
            self::MODE_RESUME => 'Resume Training',
            self::MODE_CONTINUE => 'Continue (Fine-tune)',
            self::MODE_BRANCH => 'Branch from Checkpoint',
            default => ucfirst($this->mode),
        };
    }

    /**
     * Get the effective epoch count (completed - start).
     */
    public function getEpochsTrainedAttribute(): int
    {
        return $this->completed_epochs - $this->start_epoch;
    }

    /**
     * Get formatted duration.
     */
    public function getFormattedDurationAttribute(): string
    {
        if (!$this->duration_seconds) {
            return '-';
        }

        $seconds = $this->duration_seconds;
        $hours = floor($seconds / 3600);
        $minutes = floor(($seconds % 3600) / 60);
        $secs = $seconds % 60;

        if ($hours > 0) {
            return sprintf('%dh %dm', $hours, $minutes);
        }
        return sprintf('%dm %ds', $minutes, $secs);
    }

    /**
     * Check if this run can be resumed.
     */
    public function canResume(): bool
    {
        return in_array($this->status, [self::STATUS_PAUSED, self::STATUS_FAILED])
            && $this->checkpoints()->exists();
    }

    /**
     * Check if this run is active.
     */
    public function isActive(): bool
    {
        return in_array($this->status, [
            self::STATUS_PENDING,
            self::STATUS_PREPARING,
            self::STATUS_TRAINING,
        ]);
    }

    /**
     * Get the latest checkpoint from this run.
     */
    public function getLatestCheckpoint(): ?TrainingCheckpoint
    {
        return $this->checkpoints()
            ->orderBy('epoch', 'desc')
            ->orderBy('global_step', 'desc')
            ->first();
    }

    /**
     * Scope to get active runs.
     */
    public function scopeActive($query)
    {
        return $query->whereIn('status', [
            self::STATUS_PENDING,
            self::STATUS_PREPARING,
            self::STATUS_TRAINING,
        ]);
    }

    /**
     * Scope to get completed runs.
     */
    public function scopeCompleted($query)
    {
        return $query->where('status', self::STATUS_COMPLETED);
    }

    /**
     * Get the ancestry chain (all parent runs up to root).
     */
    public function getAncestry(): array
    {
        $ancestry = [];
        $current = $this->parentRun;

        while ($current) {
            $ancestry[] = $current;
            $current = $current->parentRun;
        }

        return $ancestry;
    }
}
