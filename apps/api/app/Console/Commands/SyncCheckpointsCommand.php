<?php

namespace App\Console\Commands;

use App\Models\TrainingCheckpoint;
use App\Models\TrainingRun;
use App\Models\VoiceModel;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\File;

class SyncCheckpointsCommand extends Command
{
    protected $signature = 'training:sync-checkpoints 
                            {--model= : Sync checkpoints for a specific model slug}
                            {--all : Sync checkpoints for all models}
                            {--dry-run : Show what would be done without making changes}';

    protected $description = 'Scan disk and sync checkpoints into the database for training runs';

    protected string $preprocessDir;
    protected string $trainingDir;

    public function handle(): int
    {
        $this->preprocessDir = storage_path('data/preprocess');
        $this->trainingDir = storage_path('data/training');
        
        if (!File::isDirectory($this->preprocessDir) && !File::isDirectory($this->trainingDir)) {
            $this->error("Neither preprocess nor training directory found");
            return 1;
        }

        $dryRun = $this->option('dry-run');
        if ($dryRun) {
            $this->info('DRY RUN - No changes will be made');
        }

        if ($this->option('all')) {
            $this->syncAllModels($dryRun);
        } elseif ($modelSlug = $this->option('model')) {
            $this->syncModel($modelSlug, $dryRun);
        } else {
            $this->error('Please specify --model=<slug> or --all');
            return 1;
        }

        return 0;
    }

    protected function syncAllModels(bool $dryRun): void
    {
        $models = VoiceModel::whereHas('trainingRuns')->get();
        
        foreach ($models as $model) {
            $this->syncModel($model->slug, $dryRun);
        }
    }

    protected function syncModel(string $slug, bool $dryRun): void
    {
        $model = VoiceModel::where('slug', $slug)->first();
        
        if (!$model) {
            $this->error("Model not found: {$slug}");
            return;
        }

        $this->info("Syncing checkpoints for: {$model->name} ({$slug})");

        // Find checkpoints on disk from multiple directories
        $diskCheckpoints = [];
        $stepsPerEpoch = 20; // Default, will be updated if we find existing checkpoints
        
        // Check preprocess directory
        $preprocessModelDir = "{$this->preprocessDir}/{$slug}";
        if (File::isDirectory($preprocessModelDir)) {
            $preprocessCheckpoints = $this->scanDiskCheckpoints($preprocessModelDir, $stepsPerEpoch);
            $this->info("  Found " . count($preprocessCheckpoints) . " checkpoints in preprocess/");
            $diskCheckpoints = array_merge($diskCheckpoints, $preprocessCheckpoints);
        }
        
        // Check training directory
        $trainingModelDir = "{$this->trainingDir}/{$slug}";
        if (File::isDirectory($trainingModelDir)) {
            $trainingCheckpoints = $this->scanDiskCheckpoints($trainingModelDir, $stepsPerEpoch);
            $this->info("  Found " . count($trainingCheckpoints) . " checkpoints in training/");
            // Merge, preferring training checkpoints if they have the same global_step
            foreach ($trainingCheckpoints as $step => $cp) {
                $diskCheckpoints[$step] = $cp;
            }
        }
        
        if (empty($diskCheckpoints)) {
            $this->warn("  No checkpoint files found on disk");
            return;
        }
        
        ksort($diskCheckpoints);
        $this->info("  Total unique checkpoints: " . count($diskCheckpoints));

        // Get completed training runs (most recent first)
        $runs = $model->trainingRuns()
            ->whereIn('status', ['completed', 'training'])
            ->orderBy('started_at', 'desc')
            ->get();

        if ($runs->isEmpty()) {
            $this->warn("  No completed training runs found");
            return;
        }

        // Find runs that have NO checkpoints (need to be backfilled)
        $runsWithoutCheckpoints = $runs->filter(fn($run) => $run->checkpoints()->count() === 0);
        
        if ($runsWithoutCheckpoints->isEmpty()) {
            $this->info("  All runs already have checkpoints");
            return;
        }

        // For simplicity, assign disk checkpoints to the most recently completed run
        // that has no checkpoints yet
        $targetRun = $runsWithoutCheckpoints->first();
        $this->syncRunCheckpoints($targetRun, $diskCheckpoints, $dryRun);
    }

    protected function scanDiskCheckpoints(string $dir, int $stepsPerEpoch = 20): array
    {
        $checkpoints = [];
        
        $files = File::glob("{$dir}/G_*.pth");
        foreach ($files as $genPath) {
            // Extract global step from filename like G_240.pth
            $filename = basename($genPath);
            if (preg_match('/G_(\d+)\.pth$/', $filename, $matches)) {
                $globalStep = (int) $matches[1];
                $discPath = str_replace('/G_', '/D_', $genPath);
                
                // Calculate epoch from global step
                $epoch = (int) round($globalStep / $stepsPerEpoch);
                
                $checkpoints[$globalStep] = [
                    'global_step' => $globalStep,
                    'epoch' => $epoch,
                    'generator_path' => $genPath,
                    'discriminator_path' => File::exists($discPath) ? $discPath : null,
                    'file_size' => File::size($genPath),
                    'modified_at' => File::lastModified($genPath),
                ];
            }
        }

        ksort($checkpoints);
        return $checkpoints;
    }

    protected function syncRunCheckpoints(TrainingRun $run, array $diskCheckpoints, bool $dryRun): void
    {
        $this->info("  Run #{$run->run_number} ({$run->status}): target {$run->target_epochs} epochs");

        // Get existing checkpoints for this run
        $existingSteps = $run->checkpoints()->pluck('global_step')->toArray();
        $this->info("    Existing checkpoints: " . (count($existingSteps) > 0 ? count($existingSteps) . " checkpoints" : 'none'));

        // Estimate steps per epoch based on existing checkpoints or use default
        $stepsPerEpoch = $this->estimateStepsPerEpoch($run);
        $this->info("    Estimated steps per epoch: {$stepsPerEpoch}");

        // Calculate max global step for this run based on target epochs
        $maxStep = $run->target_epochs * $stepsPerEpoch;

        // Filter disk checkpoints that don't exist yet and are within this run's range
        $newCheckpoints = [];
        foreach ($diskCheckpoints as $globalStep => $cpData) {
            // Skip if already exists
            if (in_array($globalStep, $existingSteps)) {
                continue;
            }
            
            $epoch = $cpData['epoch'];
            
            // For completed runs, only include checkpoints <= max step
            if ($run->status === 'completed' && $globalStep > $maxStep) {
                continue;
            }
            
            $newCheckpoints[$globalStep] = $cpData;
        }

        if (empty($newCheckpoints)) {
            $this->info("    No new checkpoints to import");
            return;
        }

        $epochs = array_map(fn($cp) => $cp['epoch'], $newCheckpoints);
        $this->info("    Found " . count($newCheckpoints) . " new checkpoints to import: epochs " . implode(', ', $epochs));

        if ($dryRun) {
            return;
        }

        // Import checkpoints
        $importedCount = 0;
        foreach ($newCheckpoints as $globalStep => $cpData) {
            $this->importCheckpoint($run, $cpData);
            $importedCount++;
        }

        // Update run's checkpoint_count
        $totalCheckpoints = $run->checkpoints()->count();
        $run->update(['checkpoint_count' => $totalCheckpoints]);

        $this->info("    Imported {$importedCount} checkpoints (total: {$totalCheckpoints})");
    }

    protected function estimateStepsPerEpoch(TrainingRun $run): int
    {
        // Try to calculate from existing checkpoints
        $existing = $run->checkpoints()
            ->whereNotNull('global_step')
            ->where('global_step', '>', 0)
            ->orderBy('epoch')
            ->first();
            
        if ($existing && $existing->epoch > 0) {
            return (int) round($existing->global_step / $existing->epoch);
        }
        
        // Default based on typical batch sizes
        // With batch_size=6 and typical dataset size, usually ~20 steps/epoch
        return 20;
    }

    protected function importCheckpoint(TrainingRun $run, array $cpData): void
    {
        $model = $run->voiceModel;
        $epoch = $cpData['epoch'];
        $globalStep = $cpData['global_step'];
        
        // Generate checkpoint name
        $checkpointName = TrainingCheckpoint::generateCheckpointName(
            $model->name,
            $run->run_number,
            $epoch
        );

        // Determine if this is a milestone or final
        $isMilestone = $epoch > 0 && ($epoch % 50 === 0 || $epoch % 100 === 0);
        $isFinal = $epoch === $run->target_epochs || $epoch >= ($run->completed_epochs ?? 0);

        TrainingCheckpoint::create([
            'training_run_id' => $run->id,
            'epoch' => $epoch,
            'global_step' => $globalStep,
            'checkpoint_name' => $checkpointName,
            'short_name' => sprintf('ep%d', $epoch),
            'generator_path' => $cpData['generator_path'],
            'discriminator_path' => $cpData['discriminator_path'] ?? '',
            'checkpoint_directory' => dirname($cpData['generator_path']),
            'file_size_bytes' => $cpData['file_size'] ?? 0,
            'is_milestone' => $isMilestone,
            'is_best' => false,
            'is_final' => $isFinal,
        ]);

        $this->line("      + Imported epoch {$epoch} (step {$globalStep})");
    }
}
