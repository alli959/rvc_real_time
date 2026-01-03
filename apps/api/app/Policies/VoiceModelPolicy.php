<?php

namespace App\Policies;

use App\Models\User;
use App\Models\VoiceModel;
use Illuminate\Auth\Access\HandlesAuthorization;

class VoiceModelPolicy
{
    use HandlesAuthorization;

    /**
     * Determine whether the user can view any models.
     */
    public function viewAny(?User $user): bool
    {
        // Anyone can browse public models
        return true;
    }

    /**
     * Determine whether the user can view the model.
     */
    public function view(?User $user, VoiceModel $model): bool
    {
        // Public models can be viewed by anyone
        if ($model->visibility === 'public') {
            return true;
        }

        // Private/unlisted models require authentication
        if (!$user) {
            return false;
        }

        // Owner can always view
        if ($model->user_id === $user->id) {
            return true;
        }

        // Admins can view any model
        if ($user->hasRole('admin')) {
            return true;
        }

        // Unlisted models: anyone with the link can view
        if ($model->visibility === 'unlisted') {
            return true;
        }

        return false;
    }

    /**
     * Determine whether the user can create models.
     */
    public function create(User $user): bool
    {
        // Must have upload permission
        return $user->hasPermissionTo('upload_models');
    }

    /**
     * Determine whether the user can update the model.
     */
    public function update(User $user, VoiceModel $model): bool
    {
        // Owner can update
        if ($model->user_id === $user->id) {
            return true;
        }

        // Admins can update any model
        if ($user->hasRole('admin')) {
            return true;
        }

        return false;
    }

    /**
     * Determine whether the user can delete the model.
     */
    public function delete(User $user, VoiceModel $model): bool
    {
        // Owner can delete
        if ($model->user_id === $user->id) {
            return true;
        }

        // Admins can delete any model
        if ($user->hasRole('admin')) {
            return true;
        }

        return false;
    }

    /**
     * Determine whether the user can use the model for inference.
     */
    public function use(User $user, VoiceModel $model): bool
    {
        // Model must be ready
        if ($model->status !== 'ready') {
            return false;
        }

        // Public models can be used by anyone
        if ($model->visibility === 'public') {
            return true;
        }

        // Owner can always use their models
        if ($model->user_id === $user->id) {
            return true;
        }

        // Admins can use any model
        if ($user->hasRole('admin')) {
            return true;
        }

        // Unlisted: anyone with the link
        if ($model->visibility === 'unlisted') {
            return true;
        }

        return false;
    }

    /**
     * Determine whether the user can download the model files.
     */
    public function download(User $user, VoiceModel $model): bool
    {
        // Owner can always download
        if ($model->user_id === $user->id) {
            return true;
        }

        // Admins can download any
        if ($user->hasRole('admin')) {
            return true;
        }

        // Check if model allows downloads
        // For now, only owners and admins can download
        return false;
    }

    /**
     * Determine whether the user can restore the model.
     */
    public function restore(User $user, VoiceModel $model): bool
    {
        // Only owner or admin can restore
        return $model->user_id === $user->id || $user->hasRole('admin');
    }

    /**
     * Determine whether the user can permanently delete the model.
     */
    public function forceDelete(User $user, VoiceModel $model): bool
    {
        // Only admins can permanently delete
        return $user->hasRole('admin');
    }
}
