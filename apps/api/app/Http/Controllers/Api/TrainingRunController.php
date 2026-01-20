<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\DatasetVersion;
use App\Models\TrainingCheckpoint;
use App\Models\TrainingRun;
use App\Models\VoiceModel;
use App\Services\TrainingRunService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;

/**
 * Training Run Controller
 * 
 * Manages training runs with git-like branching model:
 * - List runs and their checkpoints for a model
 * - Start new training, resume, continue, or branch
 * - View training history tree
 * - Manage checkpoints
 */
class TrainingRunController extends Controller
{
    protected TrainingRunService $runService;

    public function __construct(TrainingRunService $runService)
    {
        $this->runService = $runService;
    }

    // =========================================================================
    // Training Runs
    // =========================================================================

    /**
     * List all training runs for a voice model.
     */
    public function index(Request $request, VoiceModel $voiceModel): JsonResponse
    {
        // Check permissions
        if (!$this->canAccessModel($request->user(), $voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $runs = $this->runService->getRunsForModel($voiceModel, includeChildren: true);

        return response()->json([
            'runs' => $runs->map(fn($run) => $this->formatRun($run)),
            'active_run' => $voiceModel->currentRun ? $this->formatRun($voiceModel->currentRun) : null,
        ]);
    }

    /**
     * Get the training history tree for a model.
     */
    public function tree(Request $request, VoiceModel $voiceModel): JsonResponse
    {
        if (!$this->canAccessModel($request->user(), $voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $tree = $this->runService->getTrainingTree($voiceModel);

        return response()->json([
            'tree' => $tree,
            'total_runs' => $voiceModel->total_training_runs,
            'total_epochs' => $voiceModel->total_training_epochs,
            'total_time' => $this->formatDuration($voiceModel->total_training_seconds),
        ]);
    }

    /**
     * Get details of a specific training run.
     */
    public function show(Request $request, TrainingRun $run): JsonResponse
    {
        if (!$this->canAccessModel($request->user(), $run->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $run->load(['checkpoints', 'datasetVersion', 'parentRun', 'parentCheckpoint']);

        return response()->json([
            'run' => $this->formatRun($run, detailed: true),
            'checkpoints' => $run->checkpoints->sortByDesc('epoch')->map(fn($c) => $this->formatCheckpoint($c)),
            'ancestry' => $run->getAncestry(),
        ]);
    }

    /**
     * Get runs that can be resumed.
     */
    public function resumable(Request $request, VoiceModel $voiceModel): JsonResponse
    {
        if (!$this->canAccessModel($request->user(), $voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $runs = $this->runService->getResumableRuns($voiceModel);

        return response()->json([
            'resumable_runs' => $runs->map(fn($run) => [
                'run' => $this->formatRun($run),
                'latest_checkpoint' => $run->getLatestCheckpoint() 
                    ? $this->formatCheckpoint($run->getLatestCheckpoint())
                    : null,
            ]),
        ]);
    }

    // =========================================================================
    // Start Training
    // =========================================================================

    /**
     * Start a new training run.
     */
    public function startNew(Request $request, VoiceModel $voiceModel): JsonResponse
    {
        $validated = $request->validate([
            'config' => ['nullable', 'array'],
            'config.epochs' => ['nullable', 'integer', 'min:1', 'max:2000'],
            'config.sample_rate' => ['nullable', 'integer', 'in:32000,40000,48000'],
            'config.f0_method' => ['nullable', 'string', 'in:rmvpe,pm,harvest'],
            'config.batch_size' => ['nullable', 'integer', 'min:1', 'max:32'],
        ]);

        if (!$this->canTrainModel($request->user(), $voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        if ($voiceModel->isTraining()) {
            return response()->json([
                'error' => 'Model is already training',
                'active_run' => $voiceModel->currentRun ? $this->formatRun($voiceModel->currentRun) : null,
            ], 409);
        }

        try {
            $run = $this->runService->startNewTraining(
                model: $voiceModel,
                dataset: $voiceModel->currentDatasetVersion,
                config: $validated['config'] ?? [],
                userId: $request->user()->id
            );

            return response()->json([
                'success' => true,
                'run' => $this->formatRun($run),
            ], 201);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Failed to start training',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Resume training from the latest checkpoint.
     */
    public function resume(Request $request, TrainingRun $run): JsonResponse
    {
        $validated = $request->validate([
            'additional_epochs' => ['nullable', 'integer', 'min:1', 'max:1000'],
        ]);

        if (!$this->canTrainModel($request->user(), $run->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        if (!$run->canResume()) {
            return response()->json([
                'error' => 'Cannot resume this run',
                'reason' => $run->isActive() ? 'Run is still active' : 'No checkpoints available',
            ], 400);
        }

        try {
            $newRun = $this->runService->resumeTraining(
                previousRun: $run,
                additionalEpochs: $validated['additional_epochs'] ?? null,
                userId: $request->user()->id
            );

            return response()->json([
                'success' => true,
                'run' => $this->formatRun($newRun),
                'resumed_from' => $this->formatRun($run),
            ], 201);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Failed to resume training',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Continue training from a specific checkpoint.
     */
    public function continueFromCheckpoint(Request $request, TrainingCheckpoint $checkpoint): JsonResponse
    {
        $validated = $request->validate([
            'config' => ['required', 'array'],
            'config.epochs' => ['required', 'integer', 'min:1', 'max:2000'],
        ]);

        $run = $checkpoint->trainingRun;
        if (!$this->canTrainModel($request->user(), $run->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        try {
            $newRun = $this->runService->continueFromCheckpoint(
                checkpoint: $checkpoint,
                config: $validated['config'],
                userId: $request->user()->id
            );

            return response()->json([
                'success' => true,
                'run' => $this->formatRun($newRun),
                'continued_from_checkpoint' => $this->formatCheckpoint($checkpoint),
            ], 201);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Failed to continue training',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Branch from a checkpoint with a new dataset.
     */
    public function branch(Request $request, TrainingCheckpoint $checkpoint): JsonResponse
    {
        $validated = $request->validate([
            'dataset_version_id' => ['required', 'exists:dataset_versions,id'],
            'config' => ['required', 'array'],
            'config.epochs' => ['required', 'integer', 'min:1', 'max:2000'],
        ]);

        $run = $checkpoint->trainingRun;
        if (!$this->canTrainModel($request->user(), $run->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $newDataset = DatasetVersion::findOrFail($validated['dataset_version_id']);

        // Verify dataset belongs to the same model
        if ($newDataset->voice_model_id !== $run->voice_model_id) {
            return response()->json([
                'error' => 'Dataset version does not belong to this model',
            ], 400);
        }

        try {
            $newRun = $this->runService->branchFromCheckpoint(
                checkpoint: $checkpoint,
                newDataset: $newDataset,
                config: $validated['config'],
                userId: $request->user()->id
            );

            return response()->json([
                'success' => true,
                'run' => $this->formatRun($newRun),
                'branched_from_checkpoint' => $this->formatCheckpoint($checkpoint),
                'new_dataset' => $this->formatDatasetVersion($newDataset),
            ], 201);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Failed to branch training',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Cancel an active training run.
     */
    public function cancel(Request $request, TrainingRun $run): JsonResponse
    {
        if (!$this->canTrainModel($request->user(), $run->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        if (!$run->isActive()) {
            return response()->json([
                'error' => 'Run is not active',
                'status' => $run->status,
            ], 400);
        }

        $success = $this->runService->cancelRun($run);

        return response()->json([
            'success' => $success,
            'run' => $this->formatRun($run->fresh()),
        ]);
    }

    /**
     * Get current status of a training run.
     */
    public function status(Request $request, TrainingRun $run): JsonResponse
    {
        if (!$this->canAccessModel($request->user(), $run->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        // Sync status from voice-engine if active
        if ($run->isActive()) {
            $engineStatus = $this->runService->syncRunStatus($run);
            $run->refresh();
        }

        return response()->json([
            'run' => $this->formatRun($run, detailed: true),
            'engine_status' => $engineStatus ?? null,
        ]);
    }

    // =========================================================================
    // Checkpoints
    // =========================================================================

    /**
     * List checkpoints for a training run.
     */
    public function checkpoints(Request $request, TrainingRun $run): JsonResponse
    {
        if (!$this->canAccessModel($request->user(), $run->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $checkpoints = $this->runService->getCheckpoints(
            run: $run,
            activeOnly: $request->boolean('active_only', true),
            milestonesOnly: $request->boolean('milestones_only', false)
        );

        return response()->json([
            'checkpoints' => $checkpoints->map(fn($c) => $this->formatCheckpoint($c)),
            'best_checkpoint' => $run->bestCheckpoint ? $this->formatCheckpoint($run->bestCheckpoint) : null,
        ]);
    }

    /**
     * Get a specific checkpoint.
     */
    public function showCheckpoint(Request $request, TrainingCheckpoint $checkpoint): JsonResponse
    {
        if (!$this->canAccessModel($request->user(), $checkpoint->trainingRun->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        return response()->json([
            'checkpoint' => $this->formatCheckpoint($checkpoint, detailed: true),
            'run' => $this->formatRun($checkpoint->trainingRun),
        ]);
    }

    /**
     * Archive a checkpoint.
     */
    public function archiveCheckpoint(Request $request, TrainingCheckpoint $checkpoint): JsonResponse
    {
        if (!$this->canTrainModel($request->user(), $checkpoint->trainingRun->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        if ($checkpoint->is_milestone || $checkpoint->is_best) {
            return response()->json([
                'error' => 'Cannot archive milestone or best checkpoints',
            ], 400);
        }

        $checkpoint->archive();

        return response()->json([
            'success' => true,
            'checkpoint' => $this->formatCheckpoint($checkpoint->fresh()),
        ]);
    }

    /**
     * Add a note to a checkpoint.
     */
    public function addCheckpointNote(Request $request, TrainingCheckpoint $checkpoint): JsonResponse
    {
        $validated = $request->validate([
            'notes' => ['required', 'string', 'max:500'],
        ]);

        if (!$this->canTrainModel($request->user(), $checkpoint->trainingRun->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $checkpoint->update(['notes' => $validated['notes']]);

        return response()->json([
            'success' => true,
            'checkpoint' => $this->formatCheckpoint($checkpoint->fresh()),
        ]);
    }

    // =========================================================================
    // Dataset Versions
    // =========================================================================

    /**
     * List dataset versions for a model.
     */
    public function datasetVersions(Request $request, VoiceModel $voiceModel): JsonResponse
    {
        if (!$this->canAccessModel($request->user(), $voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $versions = $voiceModel->datasetVersions()
            ->orderBy('version_number', 'desc')
            ->get();

        return response()->json([
            'versions' => $versions->map(fn($v) => $this->formatDatasetVersion($v)),
            'current_version' => $voiceModel->currentDatasetVersion 
                ? $this->formatDatasetVersion($voiceModel->currentDatasetVersion)
                : null,
        ]);
    }

    /**
     * Get a specific dataset version.
     */
    public function showDatasetVersion(Request $request, DatasetVersion $version): JsonResponse
    {
        if (!$this->canAccessModel($request->user(), $version->voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $version->load('trainingRuns');

        return response()->json([
            'version' => $this->formatDatasetVersion($version, detailed: true),
            'runs_using_this' => $version->trainingRuns->count(),
        ]);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    protected function canAccessModel($user, VoiceModel $model): bool
    {
        if ($user->hasRole('admin')) {
            return true;
        }
        return $model->user_id === $user->id || $model->visibility === 'public';
    }

    protected function canTrainModel($user, VoiceModel $model): bool
    {
        if ($user->hasRole('admin')) {
            return true;
        }
        return $model->user_id === $user->id;
    }

    protected function formatRun(TrainingRun $run, bool $detailed = false): array
    {
        $data = [
            'id' => $run->id,
            'uuid' => $run->uuid,
            'run_number' => $run->run_number,
            'mode' => $run->mode,
            'mode_display' => $run->mode_display,
            'status' => $run->status,
            'target_epochs' => $run->target_epochs,
            'completed_epochs' => $run->completed_epochs,
            'start_epoch' => $run->start_epoch,
            'epochs_trained' => $run->epochs_trained,
            'best_loss' => $run->best_loss,
            'duration' => $run->formatted_duration,
            'can_resume' => $run->canResume(),
            'is_active' => $run->isActive(),
            'started_at' => $run->started_at?->toIso8601String(),
            'completed_at' => $run->completed_at?->toIso8601String(),
            'created_at' => $run->created_at->toIso8601String(),
        ];

        if ($detailed) {
            $data['config_snapshot'] = $run->config_snapshot;
            $data['error_message'] = $run->error_message;
            $data['metadata'] = $run->metadata;
            $data['duration_seconds'] = $run->duration_seconds;
            
            if ($run->parentRun) {
                $data['parent_run'] = [
                    'id' => $run->parentRun->id,
                    'run_number' => $run->parentRun->run_number,
                ];
            }
            
            if ($run->parentCheckpoint) {
                $data['parent_checkpoint'] = $this->formatCheckpoint($run->parentCheckpoint);
            }
            
            if ($run->datasetVersion) {
                $data['dataset_version'] = $this->formatDatasetVersion($run->datasetVersion);
            }
        }

        return $data;
    }

    protected function formatCheckpoint(TrainingCheckpoint $checkpoint, bool $detailed = false): array
    {
        $data = [
            'id' => $checkpoint->id,
            'uuid' => $checkpoint->uuid,
            'epoch' => $checkpoint->epoch,
            'global_step' => $checkpoint->global_step,
            'checkpoint_name' => $checkpoint->checkpoint_name,
            'short_name' => $checkpoint->short_name,
            'loss_g' => $checkpoint->loss_g,
            'primary_loss' => $checkpoint->primary_loss,
            'flags' => $checkpoint->flags,
            'is_milestone' => $checkpoint->is_milestone,
            'is_best' => $checkpoint->is_best,
            'is_final' => $checkpoint->is_final,
            'is_archived' => $checkpoint->is_archived,
            'file_size' => $checkpoint->formatted_file_size,
            'notes' => $checkpoint->notes,
            'created_at' => $checkpoint->created_at->toIso8601String(),
        ];

        if ($detailed) {
            $data['generator_path'] = $checkpoint->generator_path;
            $data['discriminator_path'] = $checkpoint->discriminator_path;
            $data['checkpoint_directory'] = $checkpoint->checkpoint_directory;
            $data['loss_d'] = $checkpoint->loss_d;
            $data['loss_mel'] = $checkpoint->loss_mel;
            $data['loss_kl'] = $checkpoint->loss_kl;
            $data['loss_fm'] = $checkpoint->loss_fm;
            $data['total_loss'] = $checkpoint->total_loss;
            $data['metadata'] = $checkpoint->metadata;
        }

        return $data;
    }

    protected function formatDatasetVersion(DatasetVersion $version, bool $detailed = false): array
    {
        $data = [
            'id' => $version->id,
            'uuid' => $version->uuid,
            'version_number' => $version->version_number,
            'audio_count' => $version->audio_count,
            'duration' => $version->formatted_duration,
            'segment_count' => $version->segment_count,
            'sample_rate' => $version->sample_rate,
            'status' => $version->status,
            'is_latest' => $version->isLatest(),
            'created_at' => $version->created_at->toIso8601String(),
        ];

        if ($detailed) {
            $data['manifest_hash'] = $version->manifest_hash;
            $data['preprocessing_config'] = $version->preprocessing_config;
            $data['metadata'] = $version->metadata;
            $data['total_duration_seconds'] = $version->total_duration_seconds;
        }

        return $data;
    }

    protected function formatDuration(int $seconds): string
    {
        $hours = floor($seconds / 3600);
        $minutes = floor(($seconds % 3600) / 60);

        if ($hours > 0) {
            return sprintf('%dh %dm', $hours, $minutes);
        }
        return sprintf('%dm', $minutes);
    }
}
