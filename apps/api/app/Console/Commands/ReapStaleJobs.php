<?php

namespace App\Console\Commands;

use App\Models\JobQueue;
use Illuminate\Console\Command;

class ReapStaleJobs extends Command
{
    protected $signature = 'jobs:reap-stale';
    protected $description = 'Mark stale jobs as failed (no progress in 15min or never started in 5min)';

    public function handle(): int
    {
        $count = JobQueue::stale()->update([
            'status' => JobQueue::STATUS_FAILED,
            'error_message' => 'Job timed out (no progress received)',
            'completed_at' => now(),
        ]);

        if ($count > 0) {
            $this->info("Marked {$count} stale jobs as failed.");
        }

        return 0;
    }
}
