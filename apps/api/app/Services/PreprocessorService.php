<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use App\Models\VoiceModel;

/**
 * Service for interacting with the Preprocessor API.
 * 
 * Handles:
 * - Audio preprocessing (slicing, resampling, normalization)
 * - F0 (pitch) extraction
 * - HuBERT feature extraction
 * - Preprocessing job management
 */
class PreprocessorService
{
    protected string $baseUrl;
    protected int $timeout;

    public function __construct()
    {
        $this->baseUrl = config('services.preprocessor.url', 'http://preprocess:8003');
        $this->timeout = config('services.preprocessor.timeout', 300);
    }

    /**
     * Check if preprocessor API is available
     */
    public function isAvailable(): bool
    {
        try {
            $response = Http::timeout(5)->get("{$this->baseUrl}/health");
            return $response->successful() && ($response->json('status') === 'healthy');
        } catch (\Exception $e) {
            Log::warning('Preprocessor API health check failed', ['error' => $e->getMessage()]);
            return false;
        }
    }

    /**
     * Start preprocessing for an experiment
     * 
     * @param string $expName Experiment/model name
     * @param array $options Preprocessing options
     * @return array|null Job info or null on failure
     */
    public function startPreprocessing(string $expName, array $options = []): ?array
    {
        try {
            $payload = array_merge([
                'exp_name' => $expName,
                'sample_rate' => 48000,
                'version' => 'v2',
                'n_threads' => 4,
            ], $options);

            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/api/v1/preprocess/start", $payload);

            if ($response->successful()) {
                Log::info('Preprocessing started', [
                    'exp_name' => $expName,
                    'job_id' => $response->json('job_id'),
                ]);
                return $response->json();
            }

            Log::error('Failed to start preprocessing', [
                'exp_name' => $expName,
                'status' => $response->status(),
                'error' => $response->json('detail'),
            ]);

        } catch (\Exception $e) {
            Log::error('Preprocessing start exception', [
                'exp_name' => $expName,
                'error' => $e->getMessage(),
            ]);
        }

        return null;
    }

    /**
     * Get preprocessing job status
     * 
     * @param string $jobId Job ID
     * @return array|null Job status or null on failure
     */
    public function getJobStatus(string $jobId): ?array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/api/v1/preprocess/status/{$jobId}");

            if ($response->successful()) {
                return $response->json();
            }

            if ($response->status() === 404) {
                Log::warning('Preprocessing job not found', ['job_id' => $jobId]);
                return null;
            }

        } catch (\Exception $e) {
            Log::error('Failed to get preprocessing status', [
                'job_id' => $jobId,
                'error' => $e->getMessage(),
            ]);
        }

        return null;
    }

    /**
     * Validate preprocessing outputs for an experiment
     * 
     * @param string $expName Experiment/model name
     * @return array Validation result with 'valid' boolean and 'message'
     */
    public function validatePreprocessing(string $expName): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/api/v1/preprocess/validate/{$expName}");

            if ($response->successful()) {
                return $response->json();
            }

            return [
                'valid' => false,
                'message' => $response->json('detail', 'Validation failed'),
            ];

        } catch (\Exception $e) {
            Log::error('Preprocessing validation exception', [
                'exp_name' => $expName,
                'error' => $e->getMessage(),
            ]);
            
            return [
                'valid' => false,
                'message' => 'Failed to validate: ' . $e->getMessage(),
            ];
        }
    }

    /**
     * List all preprocessing jobs
     * 
     * @return array List of jobs
     */
    public function listJobs(): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/api/v1/preprocess/jobs");

            if ($response->successful()) {
                return $response->json('jobs', []);
            }

        } catch (\Exception $e) {
            Log::error('Failed to list preprocessing jobs', ['error' => $e->getMessage()]);
        }

        return [];
    }

    /**
     * Upload audio files for preprocessing
     * 
     * @param string $expName Experiment/model name
     * @param array $files Array of files with 'content' and 'name'
     * @return array|null Upload result or null on failure
     */
    public function uploadAudio(string $expName, array $files): ?array
    {
        try {
            // Build multipart form data
            // NOTE: exp_name must be sent as a form field, not a file attachment
            // Using the multipart array format for proper Form(...) handling in FastAPI
            $multipart = [
                [
                    'name' => 'exp_name',
                    'contents' => $expName,
                ],
            ];
            
            foreach ($files as $file) {
                $multipart[] = [
                    'name' => 'files',
                    'contents' => $file['content'],
                    'filename' => $file['name'],
                ];
            }
            
            $request = Http::timeout($this->timeout)
                ->withOptions(['multipart' => $multipart]);

            $response = $request->post("{$this->baseUrl}/api/v1/preprocess/upload");

            if ($response->successful()) {
                Log::info('Audio uploaded for preprocessing', [
                    'exp_name' => $expName,
                    'files_count' => count($files),
                ]);
                return $response->json();
            }

            Log::error('Audio upload failed', [
                'exp_name' => $expName,
                'status' => $response->status(),
                'error' => $response->body(),
            ]);

        } catch (\Exception $e) {
            Log::error('Audio upload exception', [
                'exp_name' => $expName,
                'error' => $e->getMessage(),
            ]);
        }

        return null;
    }
}
