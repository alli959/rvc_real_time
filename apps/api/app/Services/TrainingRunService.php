<?php

namespace App\Services;

use App\Models\DatasetVersion;
use App\Models\JobQueue;
use App\Models\TrainingCheckpoint;
use App\Models\TrainingRun;
use App\Models\VoiceModel;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;

/**
 * TrainingRunService
 * 
 * Orchestrates training runs with git-like branching model:
 * - NEW: Fresh training from scratch
 * - RESUME: Continue interrupted training from checkpoint
 * - CONTINUE: Fine-tune with same dataset from checkpoint  
 * - BRANCH: Train with new dataset starting from checkpoint
 * 
 * Manages the lifecycle of training runs, checkpoints, and dataset versions.
 */
class TrainingRunService
{
    protected string $baseUrl;
    protected int $timeout;
    protected TrainerService $trainerService;

    public function __construct(TrainerService $trainerService)
    {
        $this->trainerService = $trainerService;
        $this->baseUrl = config('services.voice_engine.trainer_url', 
            config('services.voice_engine.url', 'http://voice-engine:8001') . '/api/v1/trainer'
        );
        $this->timeout = config('services.voice_engine.timeout', 120);
    }

    // =========================================================================
    // Dataset Version Management
    // =========================================================================

    /**
     * Create or get the current dataset version for a model.
     * 
     * Computes a hash of the audio files to detect changes.
     * Returns existing version if hash matches, creates new version if different.
     */
    public function getOrCreateDatasetVersion(
        VoiceModel $model,
        array $audioFiles,
        array $preprocessingConfig = []
    ): DatasetVersion {
        // Compute manifest hash from file list
        $manifestHash = $this->computeManifestHash($audioFiles);
        
        // Check for existing version with same hash
        $existingVersion = DatasetVersion::where('voice_model_id', $model->id)
            ->where('manifest_hash', $manifestHash)
            ->where('status', DatasetVersion::STATUS_READY)
            ->first();
            
        if ($existingVersion) {
            Log::info('Using existing dataset version', [
                'model_id' => $model->id,
                'version' => $existingVersion->version_number,
            ]);
            return $existingVersion;
        }
        
        // Create new version
        $versionNumber = DatasetVersion::getNextVersionNumber($model->id);
        
        // Analyze audio files for metadata
        $analysis = $this->analyzeAudioFiles($audioFiles);
        
        $version = DatasetVersion::create([
            'voice_model_id' => $model->id,
            'version_number' => $versionNumber,
            'manifest_hash' => $manifestHash,
            'audio_count' => $analysis['audio_count'],
            'total_duration_seconds' => $analysis['total_duration'],
            'segment_count' => $analysis['segment_count'] ?? 0,
            'sample_rate' => $analysis['sample_rate'] ?? 40000,
            'preprocessing_config' => $preprocessingConfig,
            'status' => DatasetVersion::STATUS_READY,
            'metadata' => [
                'files' => array_map(fn($f) => $f['name'] ?? basename($f), $audioFiles),
                'created_from' => 'upload',
            ],
        ]);
        
        Log::info('Created new dataset version', [
            'model_id' => $model->id,
            'version' => $versionNumber,
            'audio_count' => $analysis['audio_count'],
        ]);
        
        return $version;
    }

    /**
     * Compute a hash for the audio file manifest.
     */
    protected function computeManifestHash(array $audioFiles): string
    {
        $manifest = [];
        foreach ($audioFiles as $file) {
            $name = $file['name'] ?? basename($file);
            $size = $file['size'] ?? (isset($file['content']) ? strlen($file['content']) : 0);
            $manifest[] = "{$name}:{$size}";
        }
        sort($manifest);
        return hash('sha256', implode('|', $manifest));
    }

    /**
     * Analyze audio files for metadata.
     */
    protected function analyzeAudioFiles(array $audioFiles): array
    {
        // Basic analysis - could be enhanced with FFprobe or voice-engine analysis
        return [
            'audio_count' => count($audioFiles),
            'total_duration' => 0, // Would need actual audio analysis
            'segment_count' => 0,
            'sample_rate' => 40000, // Default
        ];
    }

    // =========================================================================
    // Training Run Creation
    // =========================================================================

    /**
     * Start a new training run from scratch.
     */
    public function startNewTraining(
        VoiceModel $model,
        ?DatasetVersion $dataset,
        array $config,
        ?int $userId = null
    ): TrainingRun {
        return $this->createTrainingRun(
            model: $model,
            mode: TrainingRun::MODE_NEW,
            dataset: $dataset,
            config: $config,
            userId: $userId
        );
    }

    /**
     * Resume training from the latest checkpoint.
     * Uses same dataset and continues where left off.
     */
    public function resumeTraining(
        TrainingRun $previousRun,
        ?int $additionalEpochs = null,
        ?int $userId = null
    ): TrainingRun {
        if (!$previousRun->canResume()) {
            throw new \Exception("Cannot resume: run has no checkpoints or is still active");
        }
        
        $checkpoint = $previousRun->getLatestCheckpoint();
        if (!$checkpoint) {
            throw new \Exception("No checkpoint found to resume from");
        }
        
        $config = $previousRun->config_snapshot;
        if ($additionalEpochs) {
            $config['epochs'] = $checkpoint->epoch + $additionalEpochs;
        }
        
        return $this->createTrainingRun(
            model: $previousRun->voiceModel,
            mode: TrainingRun::MODE_RESUME,
            dataset: $previousRun->datasetVersion,
            config: $config,
            parentRun: $previousRun,
            parentCheckpoint: $checkpoint,
            startEpoch: $checkpoint->epoch,
            userId: $userId
        );
    }

    /**
     * Continue training from a specific checkpoint with same dataset.
     * Similar to resume but allows choosing checkpoint.
     */
    public function continueFromCheckpoint(
        TrainingCheckpoint $checkpoint,
        array $config,
        ?int $userId = null
    ): TrainingRun {
        $run = $checkpoint->trainingRun;
        
        return $this->createTrainingRun(
            model: $run->voiceModel,
            mode: TrainingRun::MODE_CONTINUE,
            dataset: $run->datasetVersion,
            config: $config,
            parentRun: $run,
            parentCheckpoint: $checkpoint,
            startEpoch: $checkpoint->epoch,
            userId: $userId
        );
    }

    /**
     * Branch from a checkpoint with a new/different dataset.
     * Like git branching - starts from checkpoint state but diverges.
     */
    public function branchFromCheckpoint(
        TrainingCheckpoint $checkpoint,
        DatasetVersion $newDataset,
        array $config,
        ?int $userId = null
    ): TrainingRun {
        $run = $checkpoint->trainingRun;
        
        return $this->createTrainingRun(
            model: $run->voiceModel,
            mode: TrainingRun::MODE_BRANCH,
            dataset: $newDataset,
            config: $config,
            parentRun: $run,
            parentCheckpoint: $checkpoint,
            startEpoch: $checkpoint->epoch,
            userId: $userId
        );
    }

    /**
     * Core method to create and start a training run.
     */
    protected function createTrainingRun(
        VoiceModel $model,
        string $mode,
        ?DatasetVersion $dataset,
        array $config,
        ?TrainingRun $parentRun = null,
        ?TrainingCheckpoint $parentCheckpoint = null,
        int $startEpoch = 0,
        ?int $userId = null
    ): TrainingRun {
        // Validate no active training on this model
        $activeRun = TrainingRun::where('voice_model_id', $model->id)
            ->active()
            ->first();
            
        if ($activeRun) {
            throw new \Exception("Model already has active training run: {$activeRun->run_number}");
        }
        
        // Set target epochs
        $targetEpochs = $config['epochs'] ?? 100;
        
        return DB::transaction(function () use (
            $model, $mode, $dataset, $config, $parentRun, 
            $parentCheckpoint, $startEpoch, $targetEpochs, $userId
        ) {
            // Create the training run record
            $run = TrainingRun::create([
                'voice_model_id' => $model->id,
                'dataset_version_id' => $dataset?->id,
                'parent_run_id' => $parentRun?->id,
                'parent_checkpoint_id' => $parentCheckpoint?->id,
                'mode' => $mode,
                'status' => TrainingRun::STATUS_PENDING,
                'config_snapshot' => $config,
                'target_epochs' => $targetEpochs,
                'completed_epochs' => $startEpoch,
                'start_epoch' => $startEpoch,
            ]);
            
            // Update model's current run reference
            $model->update([
                'current_run_id' => $run->id,
                'current_dataset_version_id' => $dataset?->id,
                'status' => 'training',
            ]);
            
            // Create JobQueue record for tracking
            $job = JobQueue::create([
                'user_id' => $userId,
                'voice_model_id' => $model->id,
                'training_run_id' => $run->id,
                'type' => JobQueue::TYPE_TRAINING,
                'status' => JobQueue::STATUS_PENDING,
                'is_resume' => in_array($mode, [
                    TrainingRun::MODE_RESUME, 
                    TrainingRun::MODE_CONTINUE, 
                    TrainingRun::MODE_BRANCH
                ]),
                'resume_from_checkpoint_id' => $parentCheckpoint?->id,
                'parameters' => [
                    'exp_name' => $model->slug,
                    'mode' => $mode,
                    'run_uuid' => $run->uuid,
                    'run_number' => $run->run_number,
                    'config' => $config,
                    'start_epoch' => $startEpoch,
                    'target_epochs' => $targetEpochs,
                ],
                'progress' => 0,
                'progress_message' => 'Starting training...',
            ]);
            
            // Start training on voice-engine
            $result = $this->startTrainingOnEngine(
                model: $model,
                run: $run,
                parentCheckpoint: $parentCheckpoint,
                config: $config
            );
            
            if (!$result) {
                // Mark as failed
                $run->update(['status' => TrainingRun::STATUS_FAILED]);
                $job->update([
                    'status' => JobQueue::STATUS_FAILED,
                    'error_message' => 'Failed to start training on voice engine',
                ]);
                $model->update(['status' => 'pending']);
                
                throw new \Exception("Failed to start training on voice engine");
            }
            
            // Update with voice-engine job ID
            $run->update([
                'status' => TrainingRun::STATUS_TRAINING,
                'started_at' => now(),
                'metadata' => [
                    'voice_engine_job_id' => $result['job_id'] ?? null,
                ],
            ]);
            
            $job->update([
                'status' => JobQueue::STATUS_PROCESSING,
                'started_at' => now(),
                'parameters->voice_engine_job_id' => $result['job_id'] ?? null,
            ]);
            
            Log::info('Training run started', [
                'run_id' => $run->id,
                'run_number' => $run->run_number,
                'mode' => $mode,
                'model' => $model->slug,
            ]);
            
            return $run;
        });
    }

    /**
     * Start training on the voice-engine service.
     */
    protected function startTrainingOnEngine(
        VoiceModel $model,
        TrainingRun $run,
        ?TrainingCheckpoint $parentCheckpoint,
        array $config
    ): ?array {
        try {
            $payload = [
                'exp_name' => $model->slug,
                'config' => $config,
            ];
            
            // Add resume info if branching/resuming from checkpoint
            if ($parentCheckpoint) {
                $payload['resume_from'] = [
                    'checkpoint_path' => $parentCheckpoint->checkpoint_directory,
                    'generator_path' => $parentCheckpoint->generator_path,
                    'discriminator_path' => $parentCheckpoint->discriminator_path,
                    'epoch' => $parentCheckpoint->epoch,
                ];
            }
            
            // Map training mode for voice-engine
            $payload['training_mode'] = match($run->mode) {
                TrainingRun::MODE_NEW => 'new_model',
                TrainingRun::MODE_RESUME => 'resume',
                TrainingRun::MODE_CONTINUE, TrainingRun::MODE_BRANCH => 'fine_tune',
                default => 'new_model',
            };
            
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/start", $payload);
            
            if ($response->successful()) {
                return $response->json();
            }
            
            Log::error('Start training failed', [
                'exp_name' => $model->slug,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);
            
        } catch (\Exception $e) {
            Log::error('Start training exception', [
                'exp_name' => $model->slug,
                'error' => $e->getMessage(),
            ]);
        }
        
        return null;
    }

    // =========================================================================
    // Checkpoint Management
    // =========================================================================

    /**
     * Record a checkpoint from training.
     */
    public function recordCheckpoint(
        TrainingRun $run,
        int $epoch,
        int $globalStep,
        string $generatorPath,
        string $discriminatorPath,
        array $losses = [],
        array $flags = []
    ): TrainingCheckpoint {
        $model = $run->voiceModel;
        
        // Determine if this is a milestone (e.g., every 50 epochs)
        $isMilestone = $flags['is_milestone'] ?? ($epoch % 50 === 0);
        
        // Determine if this is the best checkpoint
        $isBest = $flags['is_best'] ?? false;
        if (!$isBest && isset($losses['loss_g'])) {
            $currentBest = $run->bestCheckpoint;
            if (!$currentBest || $losses['loss_g'] < $currentBest->loss_g) {
                $isBest = true;
            }
        }
        
        // Generate checkpoint name
        $checkpointName = TrainingCheckpoint::generateCheckpointName(
            $model->name,
            $run->run_number,
            $epoch,
            $losses['loss_g'] ?? null,
            $isBest
        );
        
        // Get checkpoint directory from generator path
        $checkpointDir = dirname($generatorPath);
        
        // Calculate file size
        $fileSize = 0;
        // Note: Would need actual file access to calculate size
        
        $checkpoint = TrainingCheckpoint::create([
            'training_run_id' => $run->id,
            'epoch' => $epoch,
            'global_step' => $globalStep,
            'checkpoint_name' => $checkpointName,
            'short_name' => sprintf('ep%04d', $epoch),
            'generator_path' => $generatorPath,
            'discriminator_path' => $discriminatorPath,
            'checkpoint_directory' => $checkpointDir,
            'file_size_bytes' => $fileSize,
            'loss_g' => $losses['loss_g'] ?? null,
            'loss_d' => $losses['loss_d'] ?? null,
            'loss_mel' => $losses['loss_mel'] ?? null,
            'loss_kl' => $losses['loss_kl'] ?? null,
            'loss_fm' => $losses['loss_fm'] ?? null,
            'is_milestone' => $isMilestone,
            'is_best' => $isBest,
            'is_final' => $flags['is_final'] ?? false,
            'metadata' => $flags['metadata'] ?? null,
        ]);
        
        // Update run's best checkpoint if this is the best
        if ($isBest) {
            // Clear previous best flag
            TrainingCheckpoint::where('training_run_id', $run->id)
                ->where('id', '!=', $checkpoint->id)
                ->update(['is_best' => false]);
            
            $run->update([
                'best_checkpoint_id' => $checkpoint->id,
                'best_loss' => $losses['loss_g'] ?? null,
            ]);
        }
        
        // Update completed epochs
        $run->update(['completed_epochs' => $epoch]);
        
        Log::info('Checkpoint recorded', [
            'run_id' => $run->id,
            'epoch' => $epoch,
            'is_best' => $isBest,
            'is_milestone' => $isMilestone,
        ]);
        
        return $checkpoint;
    }

    /**
     * Get all checkpoints for a run, optionally filtered.
     */
    public function getCheckpoints(
        TrainingRun $run,
        bool $activeOnly = true,
        bool $milestonesOnly = false
    ) {
        $query = $run->checkpoints();
        
        if ($activeOnly) {
            $query->active();
        }
        
        if ($milestonesOnly) {
            $query->milestones();
        }
        
        return $query->orderBy('epoch', 'desc')->get();
    }

    /**
     * Archive old checkpoints, keeping milestones and best.
     */
    public function cleanupCheckpoints(TrainingRun $run, int $keepLatest = 5): int
    {
        $checkpoints = $run->checkpoints()
            ->where('is_milestone', false)
            ->where('is_best', false)
            ->where('is_final', false)
            ->where('is_archived', false)
            ->orderBy('epoch', 'desc')
            ->skip($keepLatest)
            ->get();
        
        foreach ($checkpoints as $checkpoint) {
            $checkpoint->archive();
        }
        
        return $checkpoints->count();
    }

    // =========================================================================
    // Training Run Status
    // =========================================================================

    /**
     * Sync training status from voice-engine.
     */
    public function syncRunStatus(TrainingRun $run): array
    {
        $voiceEngineJobId = $run->metadata['voice_engine_job_id'] ?? null;
        
        if (!$voiceEngineJobId) {
            return ['error' => 'No voice engine job ID'];
        }
        
        $status = $this->trainerService->getTrainingStatus($voiceEngineJobId);
        
        if (!$status) {
            return ['error' => 'Could not fetch status from voice engine'];
        }
        
        // Update run based on status
        $this->updateRunFromEngineStatus($run, $status);
        
        return $status;
    }

    /**
     * Update a training run based on voice-engine status.
     */
    protected function updateRunFromEngineStatus(TrainingRun $run, array $status): void
    {
        $engineStatus = $status['status'] ?? '';
        $updates = [];
        
        // Map engine status to run status
        $statusMap = [
            'completed' => TrainingRun::STATUS_COMPLETED,
            'failed' => TrainingRun::STATUS_FAILED,
            'cancelled' => TrainingRun::STATUS_CANCELLED,
            'training' => TrainingRun::STATUS_TRAINING,
            'preprocessing' => TrainingRun::STATUS_PREPARING,
        ];
        
        if (isset($statusMap[$engineStatus])) {
            $updates['status'] = $statusMap[$engineStatus];
            
            if (in_array($engineStatus, ['completed', 'failed', 'cancelled'])) {
                $updates['completed_at'] = now();
                
                // Calculate duration
                if ($run->started_at) {
                    $updates['duration_seconds'] = now()->diffInSeconds($run->started_at);
                }
                
                // Update model status
                $model = $run->voiceModel;
                if ($engineStatus === 'completed') {
                    $model->update([
                        'status' => 'ready',
                        'last_trained_at' => now(),
                        'total_training_epochs' => $model->total_training_epochs + $run->epochs_trained,
                        'total_training_runs' => $model->total_training_runs + 1,
                        'total_training_seconds' => $model->total_training_seconds + ($updates['duration_seconds'] ?? 0),
                    ]);
                } elseif ($engineStatus === 'failed') {
                    $updates['error_message'] = $status['error'] ?? 'Training failed';
                    $model->update(['status' => 'failed']);
                } else {
                    $model->update(['status' => 'pending']);
                }
            }
        }
        
        // Update epoch progress
        if (isset($status['current_epoch'])) {
            $updates['completed_epochs'] = (int) $status['current_epoch'];
        }
        
        if ($updates) {
            $run->update($updates);
        }
        
        // Record checkpoint if one was saved
        if (isset($status['checkpoint']) && isset($status['checkpoint']['epoch'])) {
            $this->handleCheckpointFromStatus($run, $status['checkpoint']);
        }
    }

    /**
     * Handle checkpoint info from status update.
     */
    protected function handleCheckpointFromStatus(TrainingRun $run, array $checkpointInfo): void
    {
        // Check if we already have this checkpoint
        $existing = TrainingCheckpoint::where('training_run_id', $run->id)
            ->where('epoch', $checkpointInfo['epoch'])
            ->where('global_step', $checkpointInfo['step'] ?? 0)
            ->first();
            
        if ($existing) {
            return;
        }
        
        // Record the new checkpoint
        $this->recordCheckpoint(
            run: $run,
            epoch: $checkpointInfo['epoch'],
            globalStep: $checkpointInfo['step'] ?? 0,
            generatorPath: $checkpointInfo['generator_path'] ?? '',
            discriminatorPath: $checkpointInfo['discriminator_path'] ?? '',
            losses: [
                'loss_g' => $checkpointInfo['loss_g'] ?? null,
                'loss_d' => $checkpointInfo['loss_d'] ?? null,
            ],
            flags: [
                'is_milestone' => $checkpointInfo['is_milestone'] ?? false,
                'is_best' => $checkpointInfo['is_best'] ?? false,
            ]
        );
    }

    /**
     * Complete a training run.
     */
    public function completeRun(TrainingRun $run, ?array $result = null): void
    {
        $run->update([
            'status' => TrainingRun::STATUS_COMPLETED,
            'completed_at' => now(),
            'duration_seconds' => $run->started_at ? now()->diffInSeconds($run->started_at) : null,
        ]);
        
        // Mark latest checkpoint as final
        $latestCheckpoint = $run->getLatestCheckpoint();
        if ($latestCheckpoint) {
            $latestCheckpoint->update(['is_final' => true]);
        }
        
        // Update model
        $model = $run->voiceModel;
        $model->update([
            'status' => 'ready',
            'current_run_id' => null, // Clear active run
            'last_trained_at' => now(),
            'total_training_epochs' => $model->total_training_epochs + $run->epochs_trained,
            'total_training_runs' => $model->total_training_runs + 1,
            'total_training_seconds' => $model->total_training_seconds + ($run->duration_seconds ?? 0),
            'best_checkpoint_id' => $run->best_checkpoint_id ?? $model->best_checkpoint_id,
        ]);
        
        Log::info('Training run completed', [
            'run_id' => $run->id,
            'run_number' => $run->run_number,
            'epochs_trained' => $run->epochs_trained,
            'duration' => $run->formatted_duration,
        ]);
    }

    /**
     * Cancel a training run.
     */
    public function cancelRun(TrainingRun $run): bool
    {
        // Cancel on voice-engine
        $voiceEngineJobId = $run->metadata['voice_engine_job_id'] ?? null;
        if ($voiceEngineJobId) {
            $this->trainerService->cancelTraining($voiceEngineJobId);
        }
        
        $run->update([
            'status' => TrainingRun::STATUS_CANCELLED,
            'completed_at' => now(),
            'duration_seconds' => $run->started_at ? now()->diffInSeconds($run->started_at) : null,
        ]);
        
        // Reset model status
        $run->voiceModel->update([
            'status' => $run->voiceModel->model_path ? 'ready' : 'pending',
            'current_run_id' => null,
        ]);
        
        // Update job queue
        JobQueue::where('training_run_id', $run->id)
            ->whereNotIn('status', [JobQueue::STATUS_COMPLETED, JobQueue::STATUS_FAILED])
            ->update([
                'status' => JobQueue::STATUS_CANCELLED,
                'completed_at' => now(),
            ]);
        
        return true;
    }

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * Get all runs for a model.
     */
    public function getRunsForModel(VoiceModel $model, bool $includeChildren = false)
    {
        $query = TrainingRun::where('voice_model_id', $model->id);
        
        if ($includeChildren) {
            $query->with(['checkpoints', 'childRuns', 'parentCheckpoint']);
        }
        
        return $query->orderBy('created_at', 'desc')->get();
    }

    /**
     * Get the training history tree for a model.
     * Returns runs organized by parent-child relationships.
     */
    public function getTrainingTree(VoiceModel $model): array
    {
        $allRuns = TrainingRun::where('voice_model_id', $model->id)
            ->with(['checkpoints', 'parentCheckpoint', 'datasetVersion'])
            ->get();
        
        // Build tree structure
        $rootRuns = $allRuns->filter(fn($r) => $r->parent_run_id === null);
        $tree = [];
        
        foreach ($rootRuns as $root) {
            $tree[] = $this->buildRunNode($root, $allRuns);
        }
        
        return $tree;
    }

    /**
     * Build a run node with children for the tree.
     */
    protected function buildRunNode(TrainingRun $run, $allRuns): array
    {
        $children = $allRuns->filter(fn($r) => $r->parent_run_id === $run->id);
        
        return [
            'run' => $run,
            'checkpoints' => $run->checkpoints->sortBy('epoch'),
            'children' => $children->map(fn($child) => $this->buildRunNode($child, $allRuns))->values(),
        ];
    }

    /**
     * Get resumable runs for a model (paused or failed with checkpoints).
     */
    public function getResumableRuns(VoiceModel $model)
    {
        return TrainingRun::where('voice_model_id', $model->id)
            ->whereIn('status', [TrainingRun::STATUS_PAUSED, TrainingRun::STATUS_FAILED])
            ->whereHas('checkpoints')
            ->with('checkpoints')
            ->orderBy('updated_at', 'desc')
            ->get();
    }
}
