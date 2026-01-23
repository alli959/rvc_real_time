<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

/**
 * Proxy controller for voice-engine API endpoints.
 * 
 * The browser cannot access internal Docker hostnames like voice-engine:8001,
 * so we proxy these requests through Laravel which CAN reach the internal network.
 */
class VoiceEngineProxyController extends Controller
{
    protected string $baseUrl;
    protected string $trainerBaseUrl;
    protected int $timeout;

    public function __construct()
    {
        $this->baseUrl = config('services.voice_engine.base_url', 'http://voice-engine:8001');
        $this->trainerBaseUrl = config('services.trainer.base_url', 'http://trainer:8002');
        $this->timeout = config('services.voice_engine.timeout', 300);
    }

    /**
     * Proxy GET request to voice engine metrics endpoint.
     */
    public function metrics()
    {
        return $this->proxyGet('/api/v1/admin/metrics');
    }

    /**
     * Proxy GET request to voice engine assets endpoint.
     */
    public function assets()
    {
        return $this->proxyGet('/api/v1/admin/assets');
    }

    /**
     * Proxy GET request to voice engine assets by category endpoint.
     */
    public function assetsByCategory()
    {
        return $this->proxyGet('/api/v1/admin/assets/by-category');
    }

    /**
     * Proxy POST request to voice engine asset action (start/stop).
     */
    public function assetAction(Request $request, string $assetId, string $action)
    {
        if (!in_array($action, ['start', 'stop'])) {
            return response()->json(['error' => 'Invalid action'], 400);
        }

        return $this->proxyPost("/api/v1/admin/assets/{$assetId}/{$action}", $request->all());
    }

    /**
     * Proxy GET request to voice engine logs services endpoint.
     */
    public function logsServices()
    {
        return $this->proxyGet('/api/v1/admin/logs/services');
    }

    /**
     * Proxy GET request to voice engine logs by service endpoint.
     */
    public function logsByService(Request $request, string $service)
    {
        $lines = $request->query('lines', 100);
        $source = $request->query('source', '');
        
        // Build the correct URL based on whether source is specified
        if ($source) {
            return $this->proxyGet("/api/v1/admin/logs/tail/{$service}/{$source}?lines={$lines}");
        }
        return $this->proxyGet("/api/v1/admin/logs/{$service}?lines={$lines}");
    }

    /**
     * Proxy GET request to trainer logs endpoint.
     */
    public function trainerLogs(Request $request)
    {
        $lines = $request->query('lines', 100);
        return $this->proxyTrainerGet("/api/v1/trainer/logs?lines={$lines}");
    }

    /**
     * Proxy GET request to trainer job logs endpoint.
     */
    public function trainerJobLogs(Request $request, string $jobId)
    {
        $lines = $request->query('lines', 100);
        return $this->proxyTrainerGet("/api/v1/trainer/logs/{$jobId}?lines={$lines}");
    }

    /**
     * Helper to proxy GET requests.
     */
    protected function proxyGet(string $path)
    {
        try {
            $response = Http::timeout($this->timeout)->get($this->baseUrl . $path);
            
            return response()->json(
                $response->json(),
                $response->status()
            );
        } catch (\Exception $e) {
            Log::error('Voice engine proxy error', [
                'path' => $path,
                'error' => $e->getMessage(),
            ]);
            
            return response()->json([
                'error' => 'Failed to connect to voice engine',
                'details' => $e->getMessage(),
            ], 503);
        }
    }

    /**
     * Helper to proxy POST requests.
     */
    protected function proxyPost(string $path, array $data = [])
    {
        try {
            $response = Http::timeout($this->timeout)->post($this->baseUrl . $path, $data);
            
            return response()->json(
                $response->json(),
                $response->status()
            );
        } catch (\Exception $e) {
            Log::error('Voice engine proxy error', [
                'path' => $path,
                'error' => $e->getMessage(),
            ]);
            
            return response()->json([
                'error' => 'Failed to connect to voice engine',
                'details' => $e->getMessage(),
            ], 503);
        }
    }

    /**
     * Helper to proxy GET requests to the trainer service.
     */
    protected function proxyTrainerGet(string $path)
    {
        try {
            $response = Http::timeout($this->timeout)->get($this->trainerBaseUrl . $path);
            
            return response()->json(
                $response->json(),
                $response->status()
            );
        } catch (\Exception $e) {
            Log::error('Trainer proxy error', [
                'path' => $path,
                'error' => $e->getMessage(),
            ]);
            
            return response()->json([
                'error' => 'Failed to connect to trainer service',
                'details' => $e->getMessage(),
            ], 503);
        }
    }
}
