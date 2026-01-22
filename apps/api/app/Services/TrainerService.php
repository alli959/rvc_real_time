<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use App\Models\VoiceModel;

/**
 * Service for interacting with the Trainer Service API.
 * 
 * Handles:
 * - Model scanning for language readiness
 * - Gap analysis for missing phonemes
 * - Training job management
 * - Recording wizard sessions
 * 
 * NOTE: Training is now handled by a separate trainer service (http://trainer:8002).
 * Preprocessing is handled by the preprocessor service (http://preprocess:8003).
 */
class TrainerService
{
    protected string $baseUrl;
    protected int $timeout;

    public function __construct()
    {
        // Use the new separate trainer service URL
        $this->baseUrl = config('services.trainer.url', 'http://trainer:8002') . '/api/v1/trainer';
        $this->timeout = config('services.trainer.timeout', 600);
    }

    /**
     * Check if trainer API is available
     */
    public function isAvailable(): bool
    {
        try {
            // Health check is at the root, not under /api/v1/trainer
            $healthUrl = config('services.trainer.url', 'http://trainer:8002') . '/health';
            $response = Http::timeout(5)->get($healthUrl);
            return $response->successful() && ($response->json('status') === 'healthy');
        } catch (\Exception $e) {
            Log::warning('Trainer API health check failed', ['error' => $e->getMessage()]);
            return false;
        }
    }

    /**
     * Get available languages for training/scanning
     */
    public function getAvailableLanguages(): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/prompts/languages");
            
            if ($response->successful()) {
                return $response->json('languages', ['en', 'is']);
            }
        } catch (\Exception $e) {
            Log::error('Failed to get available languages', ['error' => $e->getMessage()]);
        }
        
        return ['en', 'is']; // Default fallback
    }

    /**
     * Scan a voice model for language readiness
     * 
     * @param VoiceModel $model The model to scan
     * @param array $languages Languages to check (default: ['en', 'is'])
     * @return array|null Scan results or null on failure
     */
    public function scanModel(VoiceModel $model, array $languages = ['en', 'is']): ?array
    {
        try {
            // Get the full path to the model file
            $modelPath = $this->getModelPath($model);
            
            if (!$modelPath) {
                Log::warning('Cannot scan model: no model path', ['model_id' => $model->id]);
                return null;
            }

            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/scan", [
                    'model_path' => $modelPath,
                    'languages' => $languages,
                ]);

            if ($response->successful()) {
                return $response->json();
            }

            Log::error('Model scan failed', [
                'model_id' => $model->id,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);

        } catch (\Exception $e) {
            Log::error('Model scan exception', [
                'model_id' => $model->id,
                'error' => $e->getMessage(),
            ]);
        }

        return null;
    }

    /**
     * Analyze gaps for a specific language
     */
    public function analyzeGaps(VoiceModel $model, string $language = 'en'): ?array
    {
        try {
            $modelPath = $this->getModelPath($model);
            
            if (!$modelPath) {
                return null;
            }

            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/analyze-gaps", [
                    'model_path' => $modelPath,
                    'language' => $language,
                ]);

            if ($response->successful()) {
                return $response->json();
            }

        } catch (\Exception $e) {
            Log::error('Gap analysis failed', [
                'model_id' => $model->id,
                'language' => $language,
                'error' => $e->getMessage(),
            ]);
        }

        return null;
    }

    /**
     * Get phoneme set for a language
     */
    public function getPhonemes(string $language): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/phonemes/{$language}");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Failed to get phonemes', ['language' => $language, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Get prompts for a language
     */
    public function getPrompts(string $language): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/prompts/{$language}");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Failed to get prompts', ['language' => $language, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Get prompts for specific missing phonemes
     */
    public function getPromptsForPhonemes(string $language, array $phonemes): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/prompts/{$language}/for-phonemes", $phonemes);

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Failed to get prompts for phonemes', [
                'language' => $language,
                'phonemes' => $phonemes,
                'error' => $e->getMessage(),
            ]);
        }

        return null;
    }

    // =========================================================================
    // Training Jobs
    // =========================================================================

    /**
     * Upload training audio files to the preprocessor service
     * 
     * NOTE: Training audio is uploaded to the PREPROCESSOR service, which manages
     * the shared training_data volume. The preprocessor stores files at:
     * /data/uploads/{exp_name}/ which is then processed during preprocessing.
     * 
     * The trainer service reads from the same /data volume after preprocessing
     * is complete (from /data/{exp_name}/ directories).
     */
    public function uploadTrainingAudio(string $expName, array $files): ?array
    {
        try {
            // Use preprocessor service for uploads (it manages the shared /data volume)
            $preprocessorUrl = config('services.preprocessor.url', 'http://preprocess:8003');
            
            $request = Http::timeout($this->timeout)
                ->asMultipart();

            // Add exp_name as a form field
            $request->attach('exp_name', $expName);
            
            foreach ($files as $file) {
                $request->attach('files', $file['content'], $file['name']);
            }

            $response = $request->post("{$preprocessorUrl}/api/v1/preprocess/upload");

            if ($response->successful()) {
                Log::info('Training audio uploaded successfully', [
                    'exp_name' => $expName,
                    'files_count' => count($files),
                    'response' => $response->json()
                ]);
                return $response->json();
            }
            
            Log::error('Training upload failed', [
                'exp_name' => $expName,
                'status' => $response->status(),
                'error' => $response->body()
            ]);

        } catch (\Exception $e) {
            Log::error('Training upload exception', ['exp_name' => $expName, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Start a training job
     */
    public function startTraining(string $expName, array $config = [], ?array $audioPaths = null): ?array
    {
        try {
            $payload = ['exp_name' => $expName];
            
            if (!empty($config)) {
                $payload['config'] = $config;
            }
            
            if ($audioPaths) {
                $payload['audio_paths'] = $audioPaths;
            }

            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/start", $payload);

            if ($response->successful()) {
                return $response->json();
            }

            Log::error('Start training failed', [
                'exp_name' => $expName,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);

        } catch (\Exception $e) {
            Log::error('Start training exception', ['exp_name' => $expName, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Get training job status and update model when completed
     */
    public function getTrainingStatus(string $jobId): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/status/{$jobId}");

            if ($response->successful()) {
                $status = $response->json();
                
                // If training completed, update the model record
                if (isset($status['status']) && $status['status'] === 'completed') {
                    $this->handleTrainingCompleted($status);
                }
                
                return $status;
            }
        } catch (\Exception $e) {
            Log::error('Get training status failed', ['job_id' => $jobId, 'error' => $e->getMessage()]);
        }

        return null;
    }
    
    /**
     * Handle training completion - update the model record
     */
    protected function handleTrainingCompleted(array $status): void
    {
        $expName = $status['exp_name'] ?? null;
        if (!$expName) {
            Log::warning('Training completed but no exp_name in status');
            return;
        }
        
        // Find the model by slug or name
        $model = \App\Models\VoiceModel::where('slug', $expName)
            ->orWhere('name', $expName)
            ->first();
            
        if (!$model) {
            Log::warning('Training completed but model not found', ['exp_name' => $expName]);
            return;
        }
        
        // Skip if already processed
        if ($model->status === 'ready' && $model->model_path) {
            return;
        }
        
        // Build the model path - models are stored at /models/{exp_name}/{exp_name}.pth
        // which maps to voice-engine/assets/models/{exp_name}/{exp_name}.pth
        // The API scans models from VOICE_MODELS_PATH which is mounted to the same directory
        $modelPath = "{$expName}/{$expName}.pth";
        $indexPath = null;
        
        // Try to find the index file name from result
        if (isset($status['result']['index_path'])) {
            // Convert absolute path to relative path for storage
            $resultIndex = $status['result']['index_path'];
            // Extract just the relative path part: {exp_name}/{exp_name}.index
            $indexPath = "{$expName}/" . basename($resultIndex);
        }
        
        // Update the model
        $model->update([
            'status' => 'ready',
            'model_path' => $modelPath,
            'index_path' => $indexPath,
        ]);
        
        Log::info('Model updated after training completed', [
            'model_id' => $model->id,
            'exp_name' => $expName,
            'model_path' => $modelPath,
        ]);
    }

    /**
     * Cancel a training job
     */
    public function cancelTraining(string $jobId): bool
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/cancel/{$jobId}");

            return $response->successful();
        } catch (\Exception $e) {
            Log::error('Cancel training failed', ['job_id' => $jobId, 'error' => $e->getMessage()]);
        }

        return false;
    }

    /**
     * Request a checkpoint save for a training job
     * 
     * @param string $jobId Training job ID
     * @param bool $stopAfter If true, stop training after saving checkpoint
     */
    public function requestCheckpoint(string $jobId, bool $stopAfter = false): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/checkpoint/{$jobId}", [
                    'stop_after' => $stopAfter,
                ]);

            if ($response->successful()) {
                return $response->json();
            }
            
            Log::warning('Checkpoint request failed', [
                'job_id' => $jobId,
                'status' => $response->status(),
                'body' => $response->body(),
            ]);
        } catch (\Exception $e) {
            Log::error('Checkpoint request exception', ['job_id' => $jobId, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Get checkpoint request status for a training job
     */
    public function getCheckpointStatus(string $jobId): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/checkpoint/{$jobId}/status");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Get checkpoint status failed', ['job_id' => $jobId, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * List all training jobs
     */
    public function listTrainingJobs(): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/jobs");

            if ($response->successful()) {
                return $response->json('jobs', []);
            }
        } catch (\Exception $e) {
            Log::error('List training jobs failed', ['error' => $e->getMessage()]);
        }

        return null;
    }

    // =========================================================================
    // Recording Wizard
    // =========================================================================

    /**
     * Create a new recording wizard session
     */
    public function createWizardSession(string $language, string $expName, int $promptCount = 50, ?array $targetPhonemes = null): ?array
    {
        try {
            $payload = [
                'language' => $language,
                'exp_name' => $expName,
                'prompt_count' => $promptCount,
            ];

            if ($targetPhonemes) {
                $payload['target_phonemes'] = $targetPhonemes;
            }

            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/wizard/sessions", $payload);

            if ($response->successful()) {
                return $response->json();
            }

        } catch (\Exception $e) {
            Log::error('Create wizard session failed', ['error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Get wizard session details
     */
    public function getWizardSession(string $sessionId): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/wizard/sessions/{$sessionId}");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Get wizard session failed', ['session_id' => $sessionId, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Start a wizard session
     */
    public function startWizardSession(string $sessionId): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/wizard/sessions/{$sessionId}/start");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Start wizard session failed', ['session_id' => $sessionId, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Get current prompt in wizard session
     */
    public function getWizardCurrentPrompt(string $sessionId): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/wizard/sessions/{$sessionId}/current");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Get wizard prompt failed', ['session_id' => $sessionId, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Submit a recording for current prompt
     */
    public function submitWizardRecording(string $sessionId, string $audioBase64, int $sampleRate = 16000, bool $autoAdvance = false): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/wizard/sessions/{$sessionId}/submit", [
                    'audio' => $audioBase64,
                    'sample_rate' => $sampleRate,
                    'auto_advance' => $autoAdvance,
                ]);

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Submit wizard recording failed', ['session_id' => $sessionId, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Navigate wizard session
     */
    public function wizardNext(string $sessionId): ?array
    {
        return $this->wizardNavigate($sessionId, 'next');
    }

    public function wizardPrevious(string $sessionId): ?array
    {
        return $this->wizardNavigate($sessionId, 'previous');
    }

    public function wizardSkip(string $sessionId): ?array
    {
        return $this->wizardNavigate($sessionId, 'skip');
    }

    protected function wizardNavigate(string $sessionId, string $action): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/wizard/sessions/{$sessionId}/{$action}");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error("Wizard {$action} failed", ['session_id' => $sessionId, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Pause wizard session
     */
    public function pauseWizardSession(string $sessionId): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/wizard/sessions/{$sessionId}/pause");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Pause wizard session failed', ['session_id' => $sessionId, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Complete wizard session
     */
    public function completeWizardSession(string $sessionId): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/wizard/sessions/{$sessionId}/complete");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Complete wizard session failed', ['session_id' => $sessionId, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Cancel wizard session
     */
    public function cancelWizardSession(string $sessionId): bool
    {
        try {
            $response = Http::timeout($this->timeout)
                ->delete("{$this->baseUrl}/wizard/sessions/{$sessionId}");

            return $response->successful();
        } catch (\Exception $e) {
            Log::error('Cancel wizard session failed', ['session_id' => $sessionId, 'error' => $e->getMessage()]);
        }

        return false;
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * Get the full filesystem path to a model's .pth file
     * 
     * Translates API container paths to voice-engine accessible paths.
     * The API stores models at /var/www/html/storage/models/
     * The voice-engine accesses them at /app/assets/models/ (bind mount)
     */
    protected function getModelPath(VoiceModel $model): ?string
    {
        if (!$model->model_path) {
            return null;
        }

        $path = $model->model_path;

        // Translate API container path to voice-engine path
        // API: /var/www/html/storage/models/... -> Voice-engine: /app/assets/models/...
        if (str_starts_with($path, '/var/www/html/storage/models/')) {
            $path = str_replace('/var/www/html/storage/models/', '/app/assets/models/', $path);
            Log::debug('Translated model path', ['original' => $model->model_path, 'translated' => $path]);
            return $path;
        }

        // If it's already an absolute path pointing to assets/models, return as-is
        if (str_starts_with($path, '/app/assets/models/')) {
            return $path;
        }

        // If it's already an absolute path (not in API storage), return as-is
        if (str_starts_with($path, '/')) {
            return $path;
        }

        // For local storage with relative path, use voice-engine assets path
        if ($model->storage_type === 'local') {
            return '/app/assets/models/' . ltrim($path, '/');
        }

        // For S3, we'd need to download first - not supported for scanning yet
        return null;
    }

    /**
     * Scan and update a voice model's language scores
     */
    public function scanAndUpdateModel(VoiceModel $model, array $languages = ['en', 'is']): bool
    {
        $scanResults = $this->scanModel($model, $languages);
        
        if (!$scanResults) {
            return false;
        }

        $updateData = [
            'language_scan_results' => $scanResults,
            'language_scanned_at' => now(),
        ];

        // Extract EN scores from language_scores
        if (isset($scanResults['language_scores']['en'])) {
            $en = $scanResults['language_scores']['en'];
            $updateData['en_readiness_score'] = $en['overall_score'] ?? null;
            $updateData['en_phoneme_coverage'] = $en['component_scores']['phoneme_coverage'] ?? null;
            // Missing phonemes from phoneme_report if available
            if (isset($en['phoneme_report']['missing_phonemes'])) {
                $updateData['en_missing_phonemes'] = $en['phoneme_report']['missing_phonemes'];
            }
        }

        // Extract IS scores from language_scores
        if (isset($scanResults['language_scores']['is'])) {
            $is = $scanResults['language_scores']['is'];
            $updateData['is_readiness_score'] = $is['overall_score'] ?? null;
            $updateData['is_phoneme_coverage'] = $is['component_scores']['phoneme_coverage'] ?? null;
            // Missing phonemes from phoneme_report if available
            if (isset($is['phoneme_report']['missing_phonemes'])) {
                $updateData['is_missing_phonemes'] = $is['phoneme_report']['missing_phonemes'];
            }
        }

        $model->update($updateData);
        
        return true;
    }

    /**
     * Scan all models for language readiness (batch operation)
     */
    public function scanAllModels(array $languages = ['en', 'is']): array
    {
        $results = [
            'total' => 0,
            'scanned' => 0,
            'failed' => 0,
            'skipped' => 0,
        ];

        $models = VoiceModel::where('status', 'ready')
            ->where('is_active', true)
            ->whereNotNull('model_path')
            ->get();

        $results['total'] = $models->count();

        foreach ($models as $model) {
            // Skip S3 models for now
            if ($model->storage_type === 's3') {
                $results['skipped']++;
                continue;
            }

            if ($this->scanAndUpdateModel($model, $languages)) {
                $results['scanned']++;
            } else {
                $results['failed']++;
            }
        }

        return $results;
    }

    /**
     * Test a model using inference
     * 
     * Runs test sentences through the model to assess quality without training data.
     * 
     * @param VoiceModel $model The model to test
     * @param array $languages Languages to test (default: ['en'])
     * @param array|null $testSentences Custom test sentences
     * @param string $voice TTS voice to use for test audio
     * @return array|null Test results or null on failure
     */
    public function testModelInference(
        VoiceModel $model, 
        array $languages = ['en'], 
        ?array $testSentences = null,
        string $voice = 'en-US-GuyNeural'
    ): ?array
    {
        try {
            $modelPath = $this->getModelPath($model);
            $indexPath = $this->getIndexPath($model);
            
            if (!$modelPath) {
                Log::warning('Cannot test model: no model path', ['model_id' => $model->id]);
                return null;
            }

            $payload = [
                'model_path' => $modelPath,
                'languages' => $languages,
                'voice' => $voice,
            ];

            if ($indexPath) {
                $payload['index_path'] = $indexPath;
            }

            if ($testSentences) {
                $payload['test_sentences'] = $testSentences;
            }

            $response = Http::timeout($this->timeout * 2) // Double timeout for inference tests
                ->post("{$this->baseUrl}/test", $payload);

            if ($response->successful()) {
                $results = $response->json();
                
                // Save results to the database
                $this->saveInferenceTestResults($model, $results);
                
                return $results;
            }

            Log::error('Model inference test failed', [
                'model_id' => $model->id,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);

        } catch (\Exception $e) {
            Log::error('Model inference test exception', [
                'model_id' => $model->id,
                'error' => $e->getMessage(),
            ]);
        }

        return null;
    }

    /**
     * Get the index path for a model (if available)
     */
    protected function getIndexPath(VoiceModel $model): ?string
    {
        if (!$model->index_path) {
            return null;
        }

        // For local models, return the container path
        if ($model->storage_type === 'local') {
            $basePath = '/app/assets/models';
            $indexPath = $model->index_path;
            
            // If path is already absolute (starts with /), use as-is
            if (str_starts_with($indexPath, '/')) {
                return $indexPath;
            }
            
            // Otherwise prepend base path
            return "{$basePath}/{$indexPath}";
        }

        return null;
    }

    /**
     * Save inference test results to the voice model
     */
    protected function saveInferenceTestResults(VoiceModel $model, array $results): void
    {
        $overallScore = $results['overall_score'] ?? 0;
        $languageScores = $results['language_scores'] ?? [];
        
        $model->update([
            'inference_test_score' => $overallScore,
            'inference_test_results' => $results,
            'inference_tested_at' => now(),
            'en_inference_score' => $languageScores['en']['overall_score'] ?? null,
            'is_inference_score' => $languageScores['is']['overall_score'] ?? null,
        ]);
        
        Log::info('Saved inference test results', [
            'model_id' => $model->id,
            'overall_score' => $overallScore,
            'en_score' => $languageScores['en']['overall_score'] ?? null,
            'is_score' => $languageScores['is']['overall_score'] ?? null,
        ]);
    }

    // =========================================================================
    // Model Training Data
    // =========================================================================

    /**
     * Get all recordings for a model across all wizard sessions
     */
    public function getModelRecordings(string $expName): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/model/{$expName}/recordings");

            if ($response->successful()) {
                return $response->json();
            }

            Log::error('Failed to get model recordings', [
                'exp_name' => $expName,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);
        } catch (\Exception $e) {
            Log::error('Get model recordings exception', ['exp_name' => $expName, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Get category recording status for a model
     */
    public function getCategoryStatus(string $expName, string $language): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/model/{$expName}/category-status/{$language}");

            if ($response->successful()) {
                return $response->json();
            }

            Log::error('Failed to get category status', [
                'exp_name' => $expName,
                'language' => $language,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);
        } catch (\Exception $e) {
            Log::error('Get category status exception', ['exp_name' => $expName, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Start training using all collected recordings
     */
    public function trainModel(string $expName, array $config = []): ?array
    {
        try {
            $payload = [];
            
            if (!empty($config)) {
                $payload['config'] = $config;
            }

            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/model/{$expName}/train", $payload);

            if ($response->successful()) {
                return $response->json();
            }

            Log::error('Train model failed', [
                'exp_name' => $expName,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);
        } catch (\Exception $e) {
            Log::error('Train model exception', ['exp_name' => $expName, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Get comprehensive model training info including checkpoint status
     * 
     * Returns information about:
     * - Recording count and duration
     * - Preprocessed data status
     * - Training status (has model, has index, epochs trained)
     * - Checkpoint information for resuming training
     * - Active training job info (if training is in progress)
     */
    public function getModelTrainingInfo(string $expName): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/model/{$expName}/info");

            if ($response->successful()) {
                $info = $response->json();
                
                // Also check for active training
                try {
                    $activeResponse = Http::timeout($this->timeout)
                        ->get("{$this->baseUrl}/model/{$expName}/active");
                    
                    if ($activeResponse->successful() && $activeResponse->json('active')) {
                        $job = $activeResponse->json('job');
                        $info['training']['status'] = 'training';
                        $info['training']['job_id'] = $job['job_id'] ?? null;
                        $info['training']['current_epoch'] = $job['current_epoch'] ?? 0;
                        $info['training']['total_epochs'] = $job['total_epochs'] ?? 0;
                        $info['training']['progress'] = $job['progress'] ?? 0;
                    } else {
                        $info['training']['status'] = 'idle';
                    }
                } catch (\Exception $e) {
                    $info['training']['status'] = 'unknown';
                    Log::warning('Failed to check active training', ['exp_name' => $expName, 'error' => $e->getMessage()]);
                }
                
                return $info;
            }

            Log::error('Failed to get model training info', [
                'exp_name' => $expName,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);
        } catch (\Exception $e) {
            Log::error('Get model training info exception', ['exp_name' => $expName, 'error' => $e->getMessage()]);
        }

        return null;
    }

    /**
     * Get model configuration (sample rate, version) from training config.
     * 
     * Auto-detects settings from config.json if present in the model directory.
     * Useful for pre-populating the Extract & Index form.
     * 
     * @param string $modelDir Model directory name (relative to models dir)
     * @return array|null Config data or null on failure
     */
    public function getModelConfig(string $modelDir): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/model/{$modelDir}/config");

            if ($response->successful()) {
                return $response->json();
            }

            Log::warning('Get model config failed', [
                'model_dir' => $modelDir,
                'status' => $response->status(),
            ]);

            return null;
        } catch (\Exception $e) {
            Log::warning('Get model config exception', [
                'model_dir' => $modelDir,
                'error' => $e->getMessage(),
            ]);

            return null;
        }
    }

    /**
     * Extract model from checkpoint and/or build FAISS index.
     * 
     * This is used when:
     * - Training completed but final model wasn't extracted (has G_*.pth but no {name}.pth)
     * - Index file is missing and needs to be rebuilt
     * 
     * @param string $modelDir Model directory name (relative to models dir)
     * @param bool $extractModel Whether to extract .pth from G_*.pth
     * @param bool $buildIndex Whether to build FAISS index
     * @param string|null $sampleRate Sample rate: 32k, 40k, or 48k (null = auto-detect)
     * @param string|null $version RVC version: v1 or v2 (null = auto-detect)
     * @param string|null $modelName Custom model name (defaults to directory name)
     * @return array|null Result or null on failure
     */
    public function extractModelAndBuildIndex(
        string $modelDir,
        bool $extractModel = true,
        bool $buildIndex = true,
        ?string $sampleRate = null,
        ?string $version = null,
        ?string $modelName = null
    ): ?array {
        try {
            $payload = [
                'model_dir' => $modelDir,
                'extract_model' => $extractModel,
                'build_index' => $buildIndex,
                'model_name' => $modelName,
            ];
            
            // Only include sample_rate and version if explicitly provided
            // Otherwise let the API auto-detect from config.json
            if ($sampleRate !== null) {
                $payload['sample_rate'] = $sampleRate;
            }
            if ($version !== null) {
                $payload['version'] = $version;
            }

            $response = Http::timeout($this->timeout * 2) // Double timeout for potentially long operations
                ->post("{$this->baseUrl}/model/extract", $payload);

            if ($response->successful()) {
                return $response->json();
            }

            Log::error('Extract model failed', [
                'model_dir' => $modelDir,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);

            return [
                'success' => false,
                'message' => $response->json('detail', 'Unknown error'),
            ];
        } catch (\Exception $e) {
            Log::error('Extract model exception', [
                'model_dir' => $modelDir,
                'error' => $e->getMessage(),
            ]);

            return [
                'success' => false,
                'message' => $e->getMessage(),
            ];
        }
    }
}
