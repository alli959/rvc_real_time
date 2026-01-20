<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Str;

/**
 * TrainingCheckpoint Model
 * 
 * Represents a saved state during training.
 * Checkpoints can be milestones (kept during cleanup), best (lowest loss), or regular.
 * 
 * @property int $id
 * @property string $uuid
 * @property int $training_run_id
 * @property int $epoch
 * @property int $global_step
 * @property string $checkpoint_name
 * @property string $short_name
 * @property string $generator_path
 * @property string $discriminator_path
 * @property string $checkpoint_directory
 * @property int $file_size_bytes
 * @property float|null $loss_g
 * @property float|null $loss_d
 * @property float|null $loss_mel
 * @property float|null $loss_kl
 * @property float|null $loss_fm
 * @property bool $is_milestone
 * @property bool $is_best
 * @property bool $is_final
 * @property bool $is_archived
 * @property bool $is_exported
 * @property array|null $metadata
 * @property string|null $notes
 * @property \Carbon\Carbon $created_at
 * @property \Carbon\Carbon|null $archived_at
 * @property \Carbon\Carbon|null $exported_at
 * 
 * @property-read TrainingRun $trainingRun
 */
class TrainingCheckpoint extends Model
{
    public $timestamps = false;

    protected $fillable = [
        'uuid',
        'training_run_id',
        'epoch',
        'global_step',
        'checkpoint_name',
        'short_name',
        'generator_path',
        'discriminator_path',
        'checkpoint_directory',
        'file_size_bytes',
        'loss_g',
        'loss_d',
        'loss_mel',
        'loss_kl',
        'loss_fm',
        'is_milestone',
        'is_best',
        'is_final',
        'is_archived',
        'is_exported',
        'metadata',
        'notes',
        'created_at',
        'archived_at',
        'exported_at',
    ];

    protected $casts = [
        'epoch' => 'integer',
        'global_step' => 'integer',
        'file_size_bytes' => 'integer',
        'loss_g' => 'decimal:6',
        'loss_d' => 'decimal:6',
        'loss_mel' => 'decimal:6',
        'loss_kl' => 'decimal:6',
        'loss_fm' => 'decimal:6',
        'is_milestone' => 'boolean',
        'is_best' => 'boolean',
        'is_final' => 'boolean',
        'is_archived' => 'boolean',
        'is_exported' => 'boolean',
        'metadata' => 'array',
        'created_at' => 'datetime',
        'archived_at' => 'datetime',
        'exported_at' => 'datetime',
    ];

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
            if (empty($model->created_at)) {
                $model->created_at = now();
            }
            if (empty($model->short_name)) {
                $model->short_name = sprintf('ep%04d', $model->epoch);
            }
        });
    }

    /**
     * Get the training run this checkpoint belongs to.
     */
    public function trainingRun(): BelongsTo
    {
        return $this->belongsTo(TrainingRun::class);
    }

    /**
     * Generate a descriptive checkpoint name.
     * Format: {model_name}_run-{run_number}_ep{epoch}_{date}_{quality}
     */
    public static function generateCheckpointName(
        string $modelName,
        string $runNumber,
        int $epoch,
        ?float $loss = null,
        bool $isBest = false
    ): string {
        $sanitizedName = preg_replace('/[^a-zA-Z0-9_-]/', '_', $modelName);
        $date = now()->format('Ymd');
        
        $parts = [
            $sanitizedName,
            $runNumber,
            sprintf('ep%04d', $epoch),
            $date,
        ];

        if ($isBest) {
            $parts[] = 'best';
        } elseif ($loss !== null) {
            $parts[] = sprintf('loss%.3f', $loss);
        }

        return implode('_', $parts);
    }

    /**
     * Get the total loss (sum of all loss components).
     */
    public function getTotalLossAttribute(): ?float
    {
        $losses = array_filter([
            $this->loss_g,
            $this->loss_d,
            $this->loss_mel,
            $this->loss_kl,
            $this->loss_fm,
        ], fn($v) => $v !== null);

        return empty($losses) ? null : array_sum($losses);
    }

    /**
     * Get the primary loss (generator loss or mel loss).
     */
    public function getPrimaryLossAttribute(): ?float
    {
        return $this->loss_g ?? $this->loss_mel;
    }

    /**
     * Get formatted file size.
     */
    public function getFormattedFileSizeAttribute(): string
    {
        $bytes = $this->file_size_bytes;
        
        if ($bytes < 1024) {
            return $bytes . ' B';
        } elseif ($bytes < 1048576) {
            return round($bytes / 1024, 1) . ' KB';
        } elseif ($bytes < 1073741824) {
            return round($bytes / 1048576, 1) . ' MB';
        }
        return round($bytes / 1073741824, 2) . ' GB';
    }

    /**
     * Get display flags as array.
     */
    public function getFlagsAttribute(): array
    {
        $flags = [];
        if ($this->is_milestone) $flags[] = 'milestone';
        if ($this->is_best) $flags[] = 'best';
        if ($this->is_final) $flags[] = 'final';
        if ($this->is_archived) $flags[] = 'archived';
        if ($this->is_exported) $flags[] = 'exported';
        return $flags;
    }

    /**
     * Check if checkpoint files exist on disk.
     */
    public function filesExist(): bool
    {
        // Base path would be configured in services
        // For now, just check if paths are set
        return !empty($this->generator_path) && !empty($this->discriminator_path);
    }

    /**
     * Archive this checkpoint.
     */
    public function archive(): void
    {
        $this->update([
            'is_archived' => true,
            'archived_at' => now(),
        ]);
    }

    /**
     * Mark as exported.
     */
    public function markAsExported(): void
    {
        $this->update([
            'is_exported' => true,
            'exported_at' => now(),
        ]);
    }

    /**
     * Scope to get non-archived checkpoints.
     */
    public function scopeActive($query)
    {
        return $query->where('is_archived', false);
    }

    /**
     * Scope to get milestone checkpoints.
     */
    public function scopeMilestones($query)
    {
        return $query->where('is_milestone', true);
    }

    /**
     * Scope to get best checkpoints.
     */
    public function scopeBest($query)
    {
        return $query->where('is_best', true);
    }
}
