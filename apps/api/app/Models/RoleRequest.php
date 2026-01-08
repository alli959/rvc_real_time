<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Str;

class RoleRequest extends Model
{
    use HasFactory;

    protected $fillable = [
        'uuid',
        'user_id',
        'requested_role',
        'message',
        'status',
        'admin_response',
        'reviewed_by',
        'reviewed_at',
    ];

    protected $casts = [
        'reviewed_at' => 'datetime',
    ];

    protected static function boot()
    {
        parent::boot();

        static::creating(function ($model) {
            $model->uuid = $model->uuid ?? (string) Str::uuid();
        });
    }

    // Status constants
    const STATUS_PENDING = 'pending';
    const STATUS_APPROVED = 'approved';
    const STATUS_REJECTED = 'rejected';

    // Available requestable roles (excluding admin)
    public static function getRequestableRoles(): array
    {
        return [
            'creator' => [
                'name' => 'Creator',
                'description' => 'Upload and manage voice models',
                'permissions' => ['upload_models'],
            ],
            'trainer' => [
                'name' => 'Trainer',
                'description' => 'Train custom voice models using the platform',
                'permissions' => ['upload_models', 'train_models'],
            ],
            'publisher' => [
                'name' => 'Publisher',
                'description' => 'Publish models publicly for community use',
                'permissions' => ['upload_models', 'publish_models'],
            ],
        ];
    }

    // Relationships
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function reviewer(): BelongsTo
    {
        return $this->belongsTo(User::class, 'reviewed_by');
    }

    // Scopes
    public function scopePending($query)
    {
        return $query->where('status', self::STATUS_PENDING);
    }

    public function scopeForUser($query, $userId)
    {
        return $query->where('user_id', $userId);
    }

    // Helpers
    public function isPending(): bool
    {
        return $this->status === self::STATUS_PENDING;
    }

    public function isApproved(): bool
    {
        return $this->status === self::STATUS_APPROVED;
    }

    public function isRejected(): bool
    {
        return $this->status === self::STATUS_REJECTED;
    }

    public function approve(User $admin, ?string $response = null): bool
    {
        $this->update([
            'status' => self::STATUS_APPROVED,
            'admin_response' => $response,
            'reviewed_by' => $admin->id,
            'reviewed_at' => now(),
        ]);

        // Assign the role to the user
        $this->user->assignRole($this->requested_role);

        return true;
    }

    public function reject(User $admin, ?string $response = null): bool
    {
        $this->update([
            'status' => self::STATUS_REJECTED,
            'admin_response' => $response,
            'reviewed_by' => $admin->id,
            'reviewed_at' => now(),
        ]);

        return true;
    }
}
