<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use App\Models\VoiceModel;

/**
 * Service for interacting with the Voice Engine Trainer API.
 * 
 * Handles:
 * - Model scanning for language readiness
 * - Gap analysis for missing phonemes
 * - Training job management
 * - Recording wizard sessions
 */
class TrainerService
{
    protected string $baseUrl;
    protected int $timeout;

    public function __construct()
    {
        // Voice engine trainer API URL (same host as voice engine, different port or path)
        $this->baseUrl = config('services.voice_engine.trainer_url', 
            config('services.voice_engine.url', 'http://voice-engine:8001') . '/api/v1/trainer'
        );
        $this->timeout = config('services.voice_engine.timeout', 120);
    }

    /**
     * Check if trainer API is available
     */
    public function isAvailable(): bool
    {
        try {
            $response = Http::timeout(5)->get("{$this->baseUrl}/health");
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
     * Upload training audio files
     */
    public function uploadTrainingAudio(string $expName, array $files): ?array
    {
        try {
            $request = Http::timeout($this->timeout)
                ->asMultipart();

            foreach ($files as $file) {
                $request->attach('files[]', $file['content'], $file['name']);
            }

            $response = $request->post("{$this->baseUrl}/upload", [
                'exp_name' => $expName,
            ]);

            if ($response->successful()) {
                return $response->json();
            }

        } catch (\Exception $e) {
            Log::error('Training upload failed', ['exp_name' => $expName, 'error' => $e->getMessage()]);
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
     * Get training job status
     */
    public function getTrainingStatus(string $jobId): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/status/{$jobId}");

            if ($response->successful()) {
                return $response->json();
            }
        } catch (\Exception $e) {
            Log::error('Get training status failed', ['job_id' => $jobId, 'error' => $e->getMessage()]);
        }

        return null;
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
}
