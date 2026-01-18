<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Services\DockerLogsService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;

class LogsAdminController extends Controller
{
    private DockerLogsService $dockerLogs;

    public function __construct(DockerLogsService $dockerLogs)
    {
        $this->dockerLogs = $dockerLogs;
    }

    /**
     * Display the logs page.
     */
    public function index()
    {
        return view('admin.logs.index');
    }

    /**
     * Get all services with their log sources.
     */
    public function services(): JsonResponse
    {
        $services = $this->dockerLogs->getServicesWithLogSources();
        
        return response()->json([
            'services' => $services,
        ]);
    }

    /**
     * Get logs for a specific service and source.
     */
    public function serviceLogs(Request $request, string $service): JsonResponse
    {
        $lines = (int) $request->query('lines', 200);
        $source = $request->query('source', 'stdout');
        
        // Map service name to container name
        $containerMap = [
            'nginx' => 'morphvox-nginx',
            'api' => 'morphvox-api',
            'api-worker' => 'morphvox-api-worker',
            'web' => 'morphvox-web',
            'voice-engine' => 'morphvox-voice-engine',
            'db' => 'morphvox-db',
            'redis' => 'morphvox-redis',
            'minio' => 'morphvox-minio',
        ];
        
        $containerName = $containerMap[$service] ?? $service;
        $logLines = [];
        
        // For API/Worker services, try to read local files directly first
        if (in_array($service, ['api', 'api-worker'])) {
            $logLines = $this->getLocalServiceLogs($service, $source, $lines);
        }
        
        // If we got local logs, return them
        if (!empty($logLines) && !str_starts_with($logLines[0] ?? '', 'No ') && !str_starts_with($logLines[0] ?? '', 'Log source')) {
            return response()->json([
                'lines' => $logLines,
                'service' => $service,
                'source' => $source,
            ]);
        }
        
        // Fall back to Docker-based log retrieval
        if (str_contains($source, '_stdout') || $source === 'stdout') {
            // Container stdout/stderr logs
            $logLines = $this->dockerLogs->getContainerLogs($containerName, $lines);
        } else {
            // File-based log - get the path from the source ID
            $filePath = $this->getFilePathFromSource($source);
            
            if ($filePath) {
                $logLines = $this->dockerLogs->getContainerFileLog($containerName, $filePath, $lines);
            } else {
                $logLines = ["Unknown log source: $source"];
            }
        }
        
        return response()->json([
            'lines' => $logLines,
            'service' => $service,
            'source' => $source,
        ]);
    }
    
    /**
     * Get logs for local API/Worker services directly from mounted files.
     */
    private function getLocalServiceLogs(string $service, string $source, int $lines): array
    {
        // Map source IDs to local file paths (accessible within this container)
        $localFiles = [
            // API service logs
            'api_laravel' => storage_path('logs/laravel.log'),
            'api_stdout' => null, // stdout not available locally - will fall through to Docker
            'api_dpkg' => '/var/log/dpkg.log',
            'api_alternatives' => '/var/log/alternatives.log',
            'api_bootstrap' => '/var/log/bootstrap.log',
            // Worker service logs (shares the same laravel.log)
            'worker_laravel' => storage_path('logs/laravel.log'),
            'worker_stdout' => null,
            'worker_dpkg' => '/var/log/dpkg.log',
            'worker_alternatives' => '/var/log/alternatives.log',
            'worker_bootstrap' => '/var/log/bootstrap.log',
        ];
        
        $filePath = $localFiles[$source] ?? null;
        
        // If source is stdout, we can't read it locally
        if ($filePath === null) {
            return [];
        }
        
        if (!file_exists($filePath)) {
            return ["Log file not found: {$source}"];
        }
        
        return $this->tailFile($filePath, $lines);
    }

    /**
     * Get file path from source ID.
     */
    private function getFilePathFromSource(string $source): ?string
    {
        $pathMap = [
            // Nginx
            'nginx_access' => '/var/log/nginx/access.log',
            'nginx_error' => '/var/log/nginx/error.log',
            // API
            'api_laravel' => '/var/www/html/storage/logs/laravel.log',
            'api_dpkg' => '/var/log/dpkg.log',
            'api_alternatives' => '/var/log/alternatives.log',
            'api_bootstrap' => '/var/log/bootstrap.log',
            // Worker
            'worker_laravel' => '/var/www/html/storage/logs/laravel.log',
            'worker_dpkg' => '/var/log/dpkg.log',
            'worker_alternatives' => '/var/log/alternatives.log',
            'worker_bootstrap' => '/var/log/bootstrap.log',
            // Voice Engine
            'voice_engine' => '/app/logs/voice_engine.log',
            'voice_trainer' => '/app/logs/trainer.log',
            'voice_training' => '/app/logs/training.log',
            'voice_inference' => '/app/logs/inference.log',
        ];
        
        return $pathMap[$source] ?? null;
    }

    /**
     * Get Laravel application logs (local shortcut).
     */
    public function laravelLogs(Request $request): JsonResponse
    {
        $lines = $request->query('lines', 200);
        $logFile = storage_path('logs/laravel.log');
        
        if (!file_exists($logFile)) {
            return response()->json([
                'lines' => ['No Laravel logs available'],
                'service' => 'api',
                'source' => 'laravel',
            ]);
        }
        
        // Read last N lines efficiently
        $logLines = $this->tailFile($logFile, $lines);
        
        return response()->json([
            'lines' => $logLines,
            'service' => 'api',
            'source' => 'laravel',
        ]);
    }

    /**
     * Get worker/queue logs (local shortcut).
     */
    public function workerLogs(Request $request): JsonResponse
    {
        $lines = $request->query('lines', 200);
        $logFile = storage_path('logs/worker.log');
        
        // Also check for queue-specific log
        if (!file_exists($logFile)) {
            $logFile = storage_path('logs/laravel.log');
        }
        
        if (!file_exists($logFile)) {
            return response()->json([
                'lines' => ['No worker logs available'],
                'service' => 'api-worker',
                'source' => 'worker',
            ]);
        }
        
        $logLines = $this->tailFile($logFile, $lines);
        
        return response()->json([
            'lines' => $logLines,
            'service' => 'api-worker',
            'source' => 'worker',
        ]);
    }

    /**
     * Efficiently read the last N lines of a file.
     */
    private function tailFile(string $filePath, int $lines): array
    {
        $result = [];
        $fp = fopen($filePath, 'r');
        
        if (!$fp) {
            return ['Unable to read log file'];
        }
        
        // Go to end of file
        fseek($fp, 0, SEEK_END);
        $pos = ftell($fp);
        $lastLine = '';
        $lineCount = 0;
        
        // Read backwards
        while ($pos > 0 && $lineCount < $lines) {
            $pos--;
            fseek($fp, $pos);
            $char = fgetc($fp);
            
            if ($char === "\n") {
                if ($lastLine !== '') {
                    array_unshift($result, $lastLine);
                    $lineCount++;
                }
                $lastLine = '';
            } else {
                $lastLine = $char . $lastLine;
            }
        }
        
        // Don't forget the first line
        if ($lastLine !== '' && $lineCount < $lines) {
            array_unshift($result, $lastLine);
        }
        
        fclose($fp);
        
        return $result;
    }
}
