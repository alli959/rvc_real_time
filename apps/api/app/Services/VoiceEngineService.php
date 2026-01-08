<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class VoiceEngineService
{
    protected string $baseUrl;
    protected string $wsUrl;
    protected int $timeout;

    public function __construct()
    {
        $this->baseUrl = config('services.voice_engine.base_url', 'http://voice-engine:8001');
        $this->wsUrl = config('services.voice_engine.ws_url', 'ws://voice-engine:8765');
        $this->timeout = config('services.voice_engine.timeout', 300);
    }

    /**
     * Check if the voice engine is healthy.
     */
    public function isHealthy(): bool
    {
        try {
            $response = Http::timeout(5)->get("{$this->baseUrl}/health");
            return $response->successful();
        } catch (\Exception $e) {
            Log::warning('Voice engine health check failed', ['error' => $e->getMessage()]);
            return false;
        }
    }

    /**
     * List available models on the voice engine.
     */
    public function listModels(): array
    {
        try {
            $response = Http::timeout(30)->get("{$this->baseUrl}/models");
            
            if ($response->successful()) {
                return $response->json('models', []);
            }
            
            Log::error('Failed to list voice engine models', [
                'status' => $response->status(),
                'body' => $response->body(),
            ]);
            
            return [];
        } catch (\Exception $e) {
            Log::error('Exception listing voice engine models', ['error' => $e->getMessage()]);
            return [];
        }
    }

    /**
     * Load a model into the voice engine.
     */
    public function loadModel(string $modelPath, ?string $indexPath = null): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/model/load", [
                    'model_path' => $modelPath,
                    'index_path' => $indexPath,
                ]);
            
            if ($response->successful()) {
                return [
                    'success' => true,
                    'data' => $response->json(),
                ];
            }
            
            return [
                'success' => false,
                'error' => $response->json('error', 'Failed to load model'),
            ];
        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage(),
            ];
        }
    }

    /**
     * Process audio file through the voice engine.
     */
    public function processAudio(array $params): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/inference", [
                    'input_path' => $params['input_path'],
                    'output_path' => $params['output_path'],
                    'model_path' => $params['model_path'],
                    'index_path' => $params['index_path'] ?? null,
                    'pitch' => $params['pitch'] ?? 0,
                    'index_rate' => $params['index_rate'] ?? 0.75,
                    'filter_radius' => $params['filter_radius'] ?? 3,
                    'resample_sr' => $params['resample_sr'] ?? 0,
                    'rms_mix_rate' => $params['rms_mix_rate'] ?? 0.25,
                    'protect' => $params['protect'] ?? 0.33,
                    'f0_method' => $params['f0_method'] ?? 'rmvpe',
                ]);
            
            if ($response->successful()) {
                return [
                    'success' => true,
                    'data' => $response->json(),
                ];
            }
            
            return [
                'success' => false,
                'error' => $response->json('error', 'Inference failed'),
            ];
        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage(),
            ];
        }
    }

    /**
     * Get processing progress for a job.
     */
    public function getProgress(string $jobId): ?array
    {
        try {
            $response = Http::timeout(10)->get("{$this->baseUrl}/jobs/{$jobId}/progress");
            
            if ($response->successful()) {
                return $response->json();
            }
            
            return null;
        } catch (\Exception $e) {
            return null;
        }
    }

    /**
     * Cancel a running job.
     */
    public function cancelJob(string $jobId): bool
    {
        try {
            $response = Http::timeout(30)->post("{$this->baseUrl}/jobs/{$jobId}/cancel");
            return $response->successful();
        } catch (\Exception $e) {
            return false;
        }
    }

    /**
     * Get WebSocket URL for real-time streaming.
     */
    public function getWebSocketUrl(): string
    {
        return $this->wsUrl;
    }

    /**
     * Get engine statistics.
     */
    public function getStats(): array
    {
        try {
            $response = Http::timeout(10)->get("{$this->baseUrl}/stats");
            
            if ($response->successful()) {
                return $response->json();
            }
            
            return [];
        } catch (\Exception $e) {
            return [];
        }
    }
}
