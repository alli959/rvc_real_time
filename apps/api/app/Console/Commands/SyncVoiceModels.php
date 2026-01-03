<?php

namespace App\Console\Commands;

use App\Models\SystemVoiceModel;
use App\Services\VoiceModelScanner;
use Illuminate\Console\Command;

class SyncVoiceModels extends Command
{
    protected $signature = 'voice-models:sync 
                            {--storage= : Override storage type (local, s3)}
                            {--path= : Override local path (only for local storage)}
                            {--prune : Remove database entries for models no longer in storage}
                            {--force : Force update all models even if unchanged}';

    protected $description = 'Synchronize voice models from configured storage (local directory or S3) to database';

    public function handle(VoiceModelScanner $scanner): int
    {
        $storageType = $this->option('storage') ?? config('voice_models.storage', 'local');
        
        $this->info("Storage type: {$storageType}");
        
        if ($storageType === 'local') {
            $path = $this->option('path') ?? $scanner->getLocalPath();
            $this->info("Scanning local directory: {$path}");
        } else {
            $disk = config('voice_models.s3.disk', 's3');
            $prefix = config('voice_models.s3.prefix', 'models');
            $this->info("Scanning S3 bucket: {$disk}/{$prefix}");
        }
        
        $this->newLine();

        // Override config if custom options provided
        if ($this->option('storage')) {
            config(['voice_models.storage' => $storageType]);
        }
        if ($this->option('path') && $storageType === 'local') {
            config(['voice_models.local.path' => $this->option('path')]);
        }

        try {
            $scannedModels = $scanner->scan();
        } catch (\Exception $e) {
            $this->error("Failed to scan storage: " . $e->getMessage());
            return Command::FAILURE;
        }

        $this->info("Found " . count($scannedModels) . " models");

        if (count($scannedModels) === 0) {
            $this->warn("No models found in storage");
            return Command::SUCCESS;
        }

        $stats = ['created' => 0, 'updated' => 0, 'unchanged' => 0, 'pruned' => 0];

        $bar = $this->output->createProgressBar(count($scannedModels));
        $bar->start();

        $syncedSlugs = [];

        foreach ($scannedModels as $modelData) {
            $result = $this->syncModel($modelData);
            $stats[$result]++;
            $syncedSlugs[] = $modelData['slug'];
            $bar->advance();
        }

        $bar->finish();
        $this->newLine(2);

        // Prune removed models (only for current storage type)
        if ($this->option('prune')) {
            $pruned = SystemVoiceModel::where('storage_type', $storageType)
                ->whereNotIn('slug', $syncedSlugs)
                ->delete();
            $stats['pruned'] = $pruned;
            if ($pruned > 0) {
                $this->warn("Pruned {$pruned} models no longer in storage");
            }
        }

        // Summary
        $this->info("Sync complete!");
        $this->table(
            ['Action', 'Count'],
            [
                ['Created', $stats['created']],
                ['Updated', $stats['updated']],
                ['Unchanged', $stats['unchanged']],
                ['Pruned', $stats['pruned']],
            ]
        );

        return Command::SUCCESS;
    }

    protected function syncModel(array $modelData): string
    {
        $existing = SystemVoiceModel::where('slug', $modelData['slug'])
            ->where('storage_type', $modelData['storage_type'])
            ->first();

        if (!$existing) {
            SystemVoiceModel::create($modelData);
            return 'created';
        }

        // Check if update needed
        $needsUpdate = $this->option('force')
            || $existing->model_path !== $modelData['model_path']
            || $existing->size_bytes !== $modelData['size_bytes']
            || $existing->has_index !== $modelData['has_index']
            || $existing->storage_path !== $modelData['storage_path'];

        if ($needsUpdate) {
            $existing->update(array_merge($modelData, [
                'last_synced_at' => now(),
            ]));
            return 'updated';
        }

        // Just update last_synced_at
        $existing->update(['last_synced_at' => now()]);
        return 'unchanged';
    }
}
