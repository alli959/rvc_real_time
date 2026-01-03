<?php

namespace App\Policies;

use App\Models\User;
use App\Models\JobQueue;
use Illuminate\Auth\Access\HandlesAuthorization;

class JobQueuePolicy
{
    use HandlesAuthorization;

    /**
     * Determine whether the user can view any jobs.
     */
    public function viewAny(User $user): bool
    {
        // Any authenticated user can view their own jobs
        return true;
    }

    /**
     * Determine whether the user can view the job.
     */
    public function view(User $user, JobQueue $job): bool
    {
        // Owner can view
        if ($job->user_id === $user->id) {
            return true;
        }

        // Admins can view any job
        if ($user->hasRole('admin')) {
            return true;
        }

        return false;
    }

    /**
     * Determine whether the user can create jobs.
     */
    public function create(User $user): bool
    {
        // Any authenticated user can create jobs
        // Rate limiting / quota checking happens elsewhere
        return true;
    }

    /**
     * Determine whether the user can update the job.
     */
    public function update(User $user, JobQueue $job): bool
    {
        // Only owner can update (e.g., start processing)
        if ($job->user_id === $user->id) {
            return true;
        }

        // Admins can update any job
        if ($user->hasRole('admin')) {
            return true;
        }

        return false;
    }

    /**
     * Determine whether the user can cancel the job.
     */
    public function cancel(User $user, JobQueue $job): bool
    {
        // Can't cancel already completed/failed jobs
        if (in_array($job->status, ['completed', 'failed', 'cancelled'])) {
            return false;
        }

        // Owner can cancel
        if ($job->user_id === $user->id) {
            return true;
        }

        // Admins can cancel any job
        if ($user->hasRole('admin')) {
            return true;
        }

        return false;
    }

    /**
     * Determine whether the user can delete the job.
     */
    public function delete(User $user, JobQueue $job): bool
    {
        // Owner can delete their jobs
        if ($job->user_id === $user->id) {
            return true;
        }

        // Admins can delete any job
        if ($user->hasRole('admin')) {
            return true;
        }

        return false;
    }

    /**
     * Determine whether the user can download the job output.
     */
    public function downloadOutput(User $user, JobQueue $job): bool
    {
        // Job must be completed
        if ($job->status !== 'completed') {
            return false;
        }

        // Owner can download
        if ($job->user_id === $user->id) {
            return true;
        }

        // Admins can download any
        if ($user->hasRole('admin')) {
            return true;
        }

        return false;
    }
}
