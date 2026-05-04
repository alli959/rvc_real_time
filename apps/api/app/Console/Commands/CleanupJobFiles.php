<?php

namespace App\Console\Commands;

use App\Models\JobQueue;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Storage;

class CleanupJobFiles extends Command
{
    protected $signature = 'jobs:cleanup';
    protected $description = 'Delete output files for unsaved jobs older than 24 hours';

    public function handle(): int
    {
        $disk = Storage::disk('s3');
        $count = 0;

        // Completed unsaved jobs older than 24h
        $jobs = JobQueue::where('saved', false)
            ->where('status', JobQueue::STATUS_COMPLETED)
            ->where('completed_at', '<', now()->subHours(24))
            ->whereNotNull('output_path')
            ->get();

        // Failed jobs with orphaned files older than 24h
        $failedJobs = JobQueue::where('status', JobQueue::STATUS_FAILED)
            ->where('updated_at', '<', now()->subHours(24))
            ->whereNotNull('output_path')
            ->get();

        $allJobs = $jobs->merge($failedJobs);

        foreach ($allJobs as $job) {
            // Delete primary output
            if ($job->output_path && $disk->exists($job->output_path)) {
                $disk->delete($job->output_path);
                $count++;
            }

            // Delete multi-output files
            $outputPaths = $job->parameters['output_paths'] ?? [];
            foreach ($outputPaths as $path) {
                if ($disk->exists($path)) {
                    $disk->delete($path);
                    $count++;
                }
            }

            $job->update(['output_path' => null]);
        }

        if ($count > 0) {
            $this->info("Deleted {$count} output files from {$allJobs->count()} jobs.");
        }

        return 0;
    }
}
