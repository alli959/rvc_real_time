<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\VoiceModel;
use App\Services\TrainerService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;

/**
 * Trainer Controller
 * 
 * Handles model scanning for language readiness, training jobs,
 * and recording wizard sessions.
 */
class TrainerController extends Controller
{
    protected TrainerService $trainer;

    public function __construct(TrainerService $trainer)
    {
        $this->trainer = $trainer;
    }

    /**
     * Check if trainer API is available
     */
    public function health(): JsonResponse
    {
        return response()->json([
            'available' => $this->trainer->isAvailable(),
            'languages' => $this->trainer->getAvailableLanguages(),
        ]);
    }

    /**
     * Get available languages for training
     */
    public function languages(): JsonResponse
    {
        return response()->json([
            'languages' => $this->trainer->getAvailableLanguages(),
        ]);
    }

    // =========================================================================
    // Model Scanning
    // =========================================================================

    /**
     * Scan a single model for language readiness
     */
    public function scanModel(Request $request, VoiceModel $voiceModel): JsonResponse
    {
        $request->validate([
            'languages' => ['nullable', 'array'],
            'languages.*' => ['string', 'in:en,is'],
        ]);

        $languages = $request->input('languages', ['en', 'is']);

        // Check permissions
        if (!$this->canScanModel($request->user(), $voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        // Scan and update
        $success = $this->trainer->scanAndUpdateModel($voiceModel, $languages);

        if (!$success) {
            return response()->json([
                'error' => 'Scan failed',
                'message' => 'Could not scan model. Trainer API may be unavailable.',
            ], 500);
        }

        // Refresh model
        $voiceModel->refresh();

        return response()->json([
            'success' => true,
            'model' => [
                'id' => $voiceModel->id,
                'name' => $voiceModel->name,
                'en_readiness_score' => $voiceModel->en_readiness_score,
                'en_phoneme_coverage' => $voiceModel->en_phoneme_coverage,
                'en_missing_phonemes' => $voiceModel->en_missing_phonemes,
                'is_readiness_score' => $voiceModel->is_readiness_score,
                'is_phoneme_coverage' => $voiceModel->is_phoneme_coverage,
                'is_missing_phonemes' => $voiceModel->is_missing_phonemes,
                'language_scanned_at' => $voiceModel->language_scanned_at?->toIso8601String(),
            ],
        ]);
    }

    /**
     * Scan all models for language readiness (admin only)
     */
    public function scanAllModels(Request $request): JsonResponse
    {
        $request->validate([
            'languages' => ['nullable', 'array'],
            'languages.*' => ['string', 'in:en,is'],
        ]);

        $languages = $request->input('languages', ['en', 'is']);

        $results = $this->trainer->scanAllModels($languages);

        return response()->json([
            'success' => true,
            'results' => $results,
        ]);
    }

    /**
     * Get language readiness for a model (without rescanning)
     */
    public function getModelReadiness(VoiceModel $voiceModel): JsonResponse
    {
        return response()->json([
            'model_id' => $voiceModel->id,
            'name' => $voiceModel->name,
            'languages' => [
                'en' => [
                    'readiness_score' => $voiceModel->en_readiness_score,
                    'phoneme_coverage' => $voiceModel->en_phoneme_coverage,
                    'missing_phonemes' => $voiceModel->en_missing_phonemes,
                ],
                'is' => [
                    'readiness_score' => $voiceModel->is_readiness_score,
                    'phoneme_coverage' => $voiceModel->is_phoneme_coverage,
                    'missing_phonemes' => $voiceModel->is_missing_phonemes,
                ],
            ],
            'full_results' => $voiceModel->language_scan_results,
            'scanned_at' => $voiceModel->language_scanned_at?->toIso8601String(),
        ]);
    }

    /**
     * Analyze gaps for a model
     */
    public function analyzeGaps(Request $request, VoiceModel $voiceModel): JsonResponse
    {
        $request->validate([
            'language' => ['required', 'string', 'in:en,is'],
        ]);

        $gaps = $this->trainer->analyzeGaps($voiceModel, $request->input('language'));

        if (!$gaps) {
            return response()->json([
                'error' => 'Gap analysis failed',
            ], 500);
        }

        return response()->json($gaps);
    }

    // =========================================================================
    // Phonemes & Prompts
    // =========================================================================

    /**
     * Get phoneme set for a language
     */
    public function phonemes(string $language): JsonResponse
    {
        if (!in_array($language, ['en', 'is'])) {
            return response()->json(['error' => 'Unsupported language'], 400);
        }

        $phonemes = $this->trainer->getPhonemes($language);

        if (!$phonemes) {
            return response()->json(['error' => 'Failed to get phonemes'], 500);
        }

        return response()->json($phonemes);
    }

    /**
     * Get prompts for a language
     */
    public function prompts(string $language): JsonResponse
    {
        if (!in_array($language, ['en', 'is'])) {
            return response()->json(['error' => 'Unsupported language'], 400);
        }

        $prompts = $this->trainer->getPrompts($language);

        if (!$prompts) {
            return response()->json(['error' => 'Failed to get prompts'], 500);
        }

        return response()->json($prompts);
    }

    /**
     * Get prompts for specific phonemes
     */
    public function promptsForPhonemes(Request $request, string $language): JsonResponse
    {
        $request->validate([
            'phonemes' => ['required', 'array'],
            'phonemes.*' => ['string'],
        ]);

        if (!in_array($language, ['en', 'is'])) {
            return response()->json(['error' => 'Unsupported language'], 400);
        }

        $prompts = $this->trainer->getPromptsForPhonemes(
            $language,
            $request->input('phonemes')
        );

        if (!$prompts) {
            return response()->json(['error' => 'Failed to get prompts'], 500);
        }

        return response()->json($prompts);
    }

    // =========================================================================
    // Inference Testing
    // =========================================================================

    /**
     * Test a model using inference
     * 
     * Runs test sentences through the model to assess quality.
     * Useful for models without training data.
     */
    public function testModelInference(Request $request, VoiceModel $voiceModel): JsonResponse
    {
        $request->validate([
            'languages' => ['nullable', 'array'],
            'languages.*' => ['string', 'in:en,is'],
            'test_sentences' => ['nullable', 'array'],
            'test_sentences.*' => ['string', 'max:500'],
            'voice' => ['nullable', 'string', 'max:100'],
        ]);

        // Check permissions
        if (!$this->canScanModel($request->user(), $voiceModel)) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $languages = $request->input('languages', ['en']);
        $testSentences = $request->input('test_sentences');
        $voice = $request->input('voice', 'en-US-GuyNeural');

        $result = $this->trainer->testModelInference(
            $voiceModel,
            $languages,
            $testSentences,
            $voice
        );

        if (!$result) {
            return response()->json([
                'error' => 'Inference test failed',
                'message' => 'Could not test model. Make sure the model file is valid.',
            ], 500);
        }

        return response()->json([
            'success' => true,
            'model_id' => $voiceModel->id,
            'model_name' => $voiceModel->name,
            'results' => $result,
        ]);
    }

    // =========================================================================
    // Training Jobs
    // =========================================================================

    /**
     * Upload training audio
     */
    public function uploadAudio(Request $request): JsonResponse
    {
        // Normalize files to always be an array (handles both single file and multiple files)
        // Check both 'files' and 'files[]' field names - different clients may use different conventions
        $uploadedFiles = $request->file('files');
        
        \Log::info('Upload audio request received', [
            'exp_name' => $request->input('exp_name'),
            'has_files_field' => $request->hasFile('files'),
            'files_field_type' => $uploadedFiles ? gettype($uploadedFiles) : 'null',
            'all_inputs' => array_keys($request->all()),
            'all_files' => array_keys($request->allFiles()),
        ]);
        
        if ($uploadedFiles && !is_array($uploadedFiles)) {
            $uploadedFiles = [$uploadedFiles];
        }
        
        $request->validate([
            'exp_name' => ['required', 'string', 'max:100'],
            'files' => ['required'],
        ]);
        
        if (!$uploadedFiles || count($uploadedFiles) === 0) {
            return response()->json(['error' => 'No files uploaded'], 422);
        }

        $files = [];
        foreach ($uploadedFiles as $file) {
            // Validate each file
            if (!in_array($file->getClientOriginalExtension(), ['wav', 'mp3', 'flac', 'ogg'])) {
                return response()->json(['error' => 'Invalid file type: ' . $file->getClientOriginalName()], 422);
            }
            if ($file->getSize() > 102400 * 1024) { // 100MB
                return response()->json(['error' => 'File too large: ' . $file->getClientOriginalName()], 422);
            }
            
            \Log::info('Processing uploaded file', [
                'name' => $file->getClientOriginalName(),
                'size' => $file->getSize(),
            ]);
            $files[] = [
                'name' => $file->getClientOriginalName(),
                'content' => file_get_contents($file->getRealPath()),
            ];
        }

        $result = $this->trainer->uploadTrainingAudio(
            $request->input('exp_name'),
            $files
        );

        if (!$result) {
            return response()->json(['error' => 'Upload failed'], 500);
        }

        return response()->json($result);
    }

    /**
     * Start a training job
     */
    public function startTraining(Request $request): JsonResponse
    {
        $request->validate([
            'exp_name' => ['required', 'string', 'max:100'],
            'config' => ['nullable', 'array'],
            'config.sample_rate' => ['nullable', 'integer', 'in:32000,40000,48000'],
            'config.f0_method' => ['nullable', 'string', 'in:rmvpe,pm,harvest'],
            'config.epochs' => ['nullable', 'integer', 'min:1', 'max:2000'],
            'config.batch_size' => ['nullable', 'integer', 'min:1', 'max:32'],
            'audio_paths' => ['nullable', 'array'],
        ]);

        $result = $this->trainer->startTraining(
            $request->input('exp_name'),
            $request->input('config', []),
            $request->input('audio_paths')
        );

        if (!$result) {
            return response()->json(['error' => 'Failed to start training'], 500);
        }

        return response()->json($result);
    }

    /**
     * Get training job status
     */
    public function trainingStatus(string $jobId): JsonResponse
    {
        $status = $this->trainer->getTrainingStatus($jobId);

        if (!$status) {
            return response()->json(['error' => 'Job not found'], 404);
        }

        return response()->json($status);
    }

    /**
     * Cancel a training job
     */
    public function cancelTraining(string $jobId): JsonResponse
    {
        $success = $this->trainer->cancelTraining($jobId);

        return response()->json([
            'success' => $success,
            'job_id' => $jobId,
        ]);
    }

    /**
     * Request a checkpoint save for a training job
     */
    public function requestCheckpoint(Request $request, string $jobId): JsonResponse
    {
        $stopAfter = $request->boolean('stop_after', false);
        
        $result = $this->trainer->requestCheckpoint($jobId, $stopAfter);

        if (!$result) {
            return response()->json([
                'error' => 'Failed to request checkpoint',
                'job_id' => $jobId,
            ], 500);
        }

        return response()->json($result);
    }

    /**
     * Get checkpoint request status
     */
    public function getCheckpointStatus(string $jobId): JsonResponse
    {
        $status = $this->trainer->getCheckpointStatus($jobId);

        if (!$status) {
            return response()->json(['error' => 'Failed to get checkpoint status'], 500);
        }

        return response()->json($status);
    }

    /**
     * List training jobs
     */
    public function listTrainingJobs(): JsonResponse
    {
        $jobs = $this->trainer->listTrainingJobs();

        return response()->json([
            'jobs' => $jobs ?? [],
        ]);
    }

    // =========================================================================
    // Recording Wizard
    // =========================================================================

    /**
     * Create a wizard session
     */
    public function createWizardSession(Request $request): JsonResponse
    {
        $request->validate([
            'language' => ['required', 'string', 'in:en,is'],
            'exp_name' => ['required', 'string', 'max:100'],
            'prompt_count' => ['nullable', 'integer', 'min:10', 'max:200'],
            'target_phonemes' => ['nullable', 'array'],
        ]);

        $session = $this->trainer->createWizardSession(
            $request->input('language'),
            $request->input('exp_name'),
            $request->input('prompt_count', 50),
            $request->input('target_phonemes')
        );

        if (!$session) {
            return response()->json(['error' => 'Failed to create session'], 500);
        }

        return response()->json($session);
    }

    /**
     * Get wizard session
     */
    public function getWizardSession(string $sessionId): JsonResponse
    {
        $session = $this->trainer->getWizardSession($sessionId);

        if (!$session) {
            return response()->json(['error' => 'Session not found'], 404);
        }

        return response()->json($session);
    }

    /**
     * Start wizard session
     */
    public function startWizardSession(string $sessionId): JsonResponse
    {
        $session = $this->trainer->startWizardSession($sessionId);

        if (!$session) {
            return response()->json(['error' => 'Failed to start session'], 500);
        }

        return response()->json($session);
    }

    /**
     * Get current prompt
     */
    public function getWizardPrompt(string $sessionId): JsonResponse
    {
        $prompt = $this->trainer->getWizardCurrentPrompt($sessionId);

        if (!$prompt) {
            return response()->json(['error' => 'No current prompt'], 404);
        }

        return response()->json($prompt);
    }

    /**
     * Submit recording
     */
    public function submitWizardRecording(Request $request, string $sessionId): JsonResponse
    {
        $request->validate([
            'audio' => ['required', 'string'], // Base64 encoded
            'sample_rate' => ['nullable', 'integer'],
            'auto_advance' => ['nullable', 'boolean'],
        ]);

        $result = $this->trainer->submitWizardRecording(
            $sessionId,
            $request->input('audio'),
            $request->input('sample_rate', 16000),
            $request->boolean('auto_advance', false)
        );

        if (!$result) {
            return response()->json(['error' => 'Failed to submit recording'], 500);
        }

        return response()->json($result);
    }

    /**
     * Wizard navigation
     */
    public function wizardNext(string $sessionId): JsonResponse
    {
        $result = $this->trainer->wizardNext($sessionId);
        return $result 
            ? response()->json($result) 
            : response()->json(['error' => 'Navigation failed'], 400);
    }

    public function wizardPrevious(string $sessionId): JsonResponse
    {
        $result = $this->trainer->wizardPrevious($sessionId);
        return $result 
            ? response()->json($result) 
            : response()->json(['error' => 'Navigation failed'], 400);
    }

    public function wizardSkip(string $sessionId): JsonResponse
    {
        $result = $this->trainer->wizardSkip($sessionId);
        return $result 
            ? response()->json($result) 
            : response()->json(['error' => 'Navigation failed'], 400);
    }

    /**
     * Pause wizard session
     */
    public function pauseWizardSession(string $sessionId): JsonResponse
    {
        $session = $this->trainer->pauseWizardSession($sessionId);

        if (!$session) {
            return response()->json(['error' => 'Failed to pause session'], 500);
        }

        return response()->json($session);
    }

    /**
     * Complete wizard session
     */
    public function completeWizardSession(string $sessionId): JsonResponse
    {
        $result = $this->trainer->completeWizardSession($sessionId);

        if (!$result) {
            return response()->json(['error' => 'Failed to complete session'], 500);
        }

        return response()->json($result);
    }

    /**
     * Cancel wizard session
     */
    public function cancelWizardSession(string $sessionId): JsonResponse
    {
        $success = $this->trainer->cancelWizardSession($sessionId);

        return response()->json([
            'success' => $success,
            'session_id' => $sessionId,
        ]);
    }

    // =========================================================================
    // Model Training Data
    // =========================================================================

    /**
     * Get all recordings for a model
     */
    public function getModelRecordings(string $modelSlug): JsonResponse
    {
        $model = VoiceModel::where('slug', $modelSlug)
            ->orWhere('name', $modelSlug)
            ->first();

        if (!$model) {
            return response()->json(['error' => 'Model not found'], 404);
        }

        $expName = $model->slug ?: $model->name;
        $recordings = $this->trainer->getModelRecordings($expName);

        if (!$recordings) {
            return response()->json(['error' => 'Failed to get recordings'], 500);
        }

        return response()->json($recordings);
    }

    /**
     * Get category recording status for a model
     */
    public function getCategoryStatus(Request $request, string $modelSlug): JsonResponse
    {
        $language = $request->query('language', 'en');
        
        $model = VoiceModel::where('slug', $modelSlug)
            ->orWhere('name', $modelSlug)
            ->first();

        if (!$model) {
            return response()->json(['error' => 'Model not found'], 404);
        }

        $expName = $model->slug ?: $model->name;
        $status = $this->trainer->getCategoryStatus($expName, $language);

        if (!$status) {
            return response()->json(['error' => 'Failed to get category status'], 500);
        }

        // Merge with model's existing phoneme data
        $status['model'] = [
            'id' => $model->id,
            'name' => $model->name,
            'en_phoneme_coverage' => $model->en_phoneme_coverage,
            'en_missing_phonemes' => $model->en_missing_phonemes ?? [],
            'is_phoneme_coverage' => $model->is_phoneme_coverage,
            'is_missing_phonemes' => $model->is_missing_phonemes ?? [],
            'language_scanned_at' => $model->language_scanned_at,
        ];

        return response()->json($status);
    }

    /**
     * Start training using all collected recordings
     */
    public function trainModel(Request $request, string $modelSlug): JsonResponse
    {
        $model = VoiceModel::where('slug', $modelSlug)
            ->orWhere('name', $modelSlug)
            ->first();

        if (!$model) {
            return response()->json(['error' => 'Model not found'], 404);
        }

        // Check authorization
        $user = $request->user();
        if ($model->user_id !== $user->id && !$user->hasRole('admin')) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $expName = $model->slug ?: $model->name;
        $config = $request->input('config', []);
        
        $result = $this->trainer->trainModel($expName, $config);

        if (!$result) {
            return response()->json(['error' => 'Failed to start training'], 500);
        }

        return response()->json($result);
    }

    /**
     * Get comprehensive model training info including checkpoint status
     * 
     * Returns information about recordings, preprocessed data,
     * training status, and checkpoint info for resuming training.
     */
    public function getModelTrainingInfo(Request $request, string $modelSlug): JsonResponse
    {
        $model = VoiceModel::where('slug', $modelSlug)
            ->orWhere('name', $modelSlug)
            ->first();

        if (!$model) {
            return response()->json(['error' => 'Model not found'], 404);
        }

        // Check authorization
        $user = $request->user();
        if ($model->user_id !== $user->id && !$user->hasRole('admin')) {
            return response()->json(['error' => 'Unauthorized'], 403);
        }

        $expName = $model->slug ?: $model->name;
        $info = $this->trainer->getModelTrainingInfo($expName);

        if (!$info) {
            return response()->json(['error' => 'Failed to get training info'], 500);
        }

        // Add model database info
        $info['model'] = [
            'id' => $model->id,
            'name' => $model->name,
            'slug' => $model->slug,
            'status' => $model->status,
            'en_readiness_score' => $model->en_readiness_score,
            'language_scanned_at' => $model->language_scanned_at,
        ];

        return response()->json($info);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * Check if user can scan a model
     */
    protected function canScanModel($user, VoiceModel $model): bool
    {
        // Admins can scan any model
        if ($user->hasRole('admin')) {
            return true;
        }

        // Users with trainer role can scan their own models
        if ($user->hasRole('trainer') && $model->user_id === $user->id) {
            return true;
        }

        // Model owners can scan their models
        if ($model->user_id === $user->id) {
            return true;
        }

        return false;
    }
}
