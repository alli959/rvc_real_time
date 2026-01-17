<?php

namespace App\Services;

use Illuminate\Support\Facades\Log;

/**
 * Service for reading Docker container logs via the Docker socket.
 * 
 * This service connects to the Docker daemon through the mounted socket
 * to fetch logs from any container in the Docker network.
 */
class DockerLogsService
{
    /**
     * Container name to ID mapping (cached)
     */
    private array $containerIds = [];

    /**
     * Get list of all running containers
     */
    public function getContainers(): array
    {
        try {
            $response = $this->dockerApiRequest('/containers/json?all=true');
            
            if (!$response) {
                return [];
            }
            
            $containers = [];
            foreach ($response as $container) {
                $name = ltrim($container['Names'][0] ?? '', '/');
                $containers[] = [
                    'id' => $container['Id'],
                    'name' => $name,
                    'image' => $container['Image'],
                    'state' => $container['State'],
                    'status' => $container['Status'] ?? '',
                ];
                
                // Cache the container ID
                $this->containerIds[$name] = $container['Id'];
            }
            
            return $containers;
        } catch (\Exception $e) {
            Log::error('Failed to get Docker containers: ' . $e->getMessage());
            return [];
        }
    }

    /**
     * Get logs for a specific container
     * 
     * @param string $containerName Name of the container (e.g., 'morphvox-api')
     * @param int $lines Number of lines to retrieve (tail)
     * @param bool $timestamps Include timestamps in output
     * @return array Array of log lines
     */
    public function getContainerLogs(string $containerName, int $lines = 200, bool $timestamps = false): array
    {
        try {
            $containerId = $this->getContainerId($containerName);
            
            if (!$containerId) {
                return ["Container '$containerName' not found"];
            }
            
            $query = http_build_query([
                'stdout' => 'true',
                'stderr' => 'true',
                'tail' => (string)$lines,
                'timestamps' => $timestamps ? 'true' : 'false',
            ]);
            
            $response = $this->dockerApiRequest("/containers/{$containerId}/logs?{$query}", true);
            
            if (!$response) {
                return [];
            }
            
            return $this->parseDockerLogStream($response);
        } catch (\Exception $e) {
            Log::error("Failed to get logs for container '$containerName': " . $e->getMessage());
            return ['Error fetching logs: ' . $e->getMessage()];
        }
    }

    /**
     * Get container ID by name
     */
    private function getContainerId(string $containerName): ?string
    {
        // Check cache first
        if (isset($this->containerIds[$containerName])) {
            return $this->containerIds[$containerName];
        }
        
        // Fetch all containers to populate cache
        $this->getContainers();
        
        return $this->containerIds[$containerName] ?? null;
    }

    /**
     * Make a request to the Docker API via Unix socket
     * 
     * @param string $endpoint API endpoint (e.g., '/containers/json')
     * @param bool $raw Return raw response instead of JSON-decoded
     * @return mixed
     */
    private function dockerApiRequest(string $endpoint, bool $raw = false): mixed
    {
        $socketPath = '/var/run/docker.sock';
        
        if (!file_exists($socketPath)) {
            Log::warning('Docker socket not found at ' . $socketPath);
            return null;
        }
        
        $socket = @stream_socket_client("unix://{$socketPath}", $errno, $errstr, 5);
        
        if (!$socket) {
            Log::error("Failed to connect to Docker socket: $errstr ($errno)");
            return null;
        }
        
        // Send HTTP request
        $request = "GET {$endpoint} HTTP/1.1\r\n";
        $request .= "Host: localhost\r\n";
        $request .= "Connection: close\r\n";
        $request .= "\r\n";
        
        fwrite($socket, $request);
        
        // Read response
        $response = '';
        while (!feof($socket)) {
            $response .= fread($socket, 8192);
        }
        
        fclose($socket);
        
        // Parse HTTP response - handle both regular and chunked transfer encoding
        $parts = explode("\r\n\r\n", $response, 2);
        $headers = $parts[0] ?? '';
        $body = $parts[1] ?? '';
        
        // Check if chunked transfer encoding
        if (stripos($headers, 'Transfer-Encoding: chunked') !== false) {
            $body = $this->decodeChunkedBody($body);
        }
        
        if ($raw) {
            return $body;
        }
        
        return json_decode($body, true);
    }

    /**
     * Decode a chunked HTTP response body
     */
    private function decodeChunkedBody(string $body): string
    {
        $decoded = '';
        $offset = 0;
        
        while ($offset < strlen($body)) {
            // Find the chunk size line
            $lineEnd = strpos($body, "\r\n", $offset);
            if ($lineEnd === false) {
                break;
            }
            
            $chunkSizeHex = substr($body, $offset, $lineEnd - $offset);
            $chunkSize = hexdec($chunkSizeHex);
            
            if ($chunkSize === 0) {
                break;
            }
            
            $offset = $lineEnd + 2; // Skip past \r\n
            $decoded .= substr($body, $offset, $chunkSize);
            $offset += $chunkSize + 2; // Skip past chunk data and \r\n
        }
        
        return $decoded;
    }

    /**
     * Parse Docker log stream format.
     * 
     * Docker multiplexes stdout/stderr with an 8-byte header:
     * [stream_type (1 byte)][0][0][0][size (4 bytes big-endian)][payload]
     * 
     * But when streaming via HTTP, we might just get plain text lines.
     */
    private function parseDockerLogStream(string $data): array
    {
        $lines = [];
        $offset = 0;
        $length = strlen($data);
        
        // Try to detect if this is multiplexed stream format
        // Byte 0 should be 0, 1, or 2 (stdin/stdout/stderr)
        // Bytes 1-3 should be 0
        $isMultiplexed = $length >= 8 && 
            ord($data[0]) <= 2 && 
            ord($data[1]) === 0 && 
            ord($data[2]) === 0 && 
            ord($data[3]) === 0;
        
        if ($isMultiplexed) {
            while ($offset + 8 <= $length) {
                // Read 8-byte header
                $header = substr($data, $offset, 8);
                $streamType = ord($header[0]);
                $frameSize = unpack('N', substr($header, 4, 4))[1];
                
                $offset += 8;
                
                if ($offset + $frameSize > $length) {
                    break;
                }
                
                $payload = substr($data, $offset, $frameSize);
                $offset += $frameSize;
                
                // Split payload into lines
                foreach (explode("\n", $payload) as $line) {
                    $trimmed = rtrim($line);
                    if ($trimmed !== '') {
                        $lines[] = $trimmed;
                    }
                }
            }
        } else {
            // Plain text format - just split by newlines
            foreach (explode("\n", $data) as $line) {
                $trimmed = rtrim($line);
                if ($trimmed !== '') {
                    $lines[] = $trimmed;
                }
            }
        }
        
        return $lines;
    }

    /**
     * Get log file contents from within a container
     * This is useful for log files inside containers that aren't stdout/stderr
     * 
     * @param string $containerName Container name
     * @param string $filePath Path to the log file inside the container
     * @param int $lines Number of lines (tail)
     * @return array
     */
    public function getContainerFileLog(string $containerName, string $filePath, int $lines = 200): array
    {
        try {
            $containerId = $this->getContainerId($containerName);
            
            if (!$containerId) {
                return ["Container '$containerName' not found"];
            }
            
            // Execute tail command inside container
            $command = [
                'Cmd' => ['tail', '-n', (string)$lines, $filePath],
                'AttachStdout' => true,
                'AttachStderr' => true,
            ];
            
            // Create exec instance
            $createResp = $this->dockerApiPost("/containers/{$containerId}/exec", $command);
            
            if (!$createResp || !isset($createResp['Id'])) {
                return ["Failed to create exec instance"];
            }
            
            // Start exec
            $execId = $createResp['Id'];
            $startResp = $this->dockerApiPost("/exec/{$execId}/start", ['Detach' => false, 'Tty' => false], true);
            
            if (!$startResp) {
                return [];
            }
            
            return $this->parseDockerLogStream($startResp);
        } catch (\Exception $e) {
            Log::error("Failed to read file from container '$containerName': " . $e->getMessage());
            return ['Error reading file: ' . $e->getMessage()];
        }
    }

    /**
     * Make a POST request to Docker API
     */
    private function dockerApiPost(string $endpoint, array $data, bool $raw = false): mixed
    {
        $socketPath = '/var/run/docker.sock';
        
        if (!file_exists($socketPath)) {
            return null;
        }
        
        $socket = @stream_socket_client("unix://{$socketPath}", $errno, $errstr, 5);
        
        if (!$socket) {
            return null;
        }
        
        $body = json_encode($data);
        
        $request = "POST {$endpoint} HTTP/1.1\r\n";
        $request .= "Host: localhost\r\n";
        $request .= "Content-Type: application/json\r\n";
        $request .= "Content-Length: " . strlen($body) . "\r\n";
        $request .= "Connection: close\r\n";
        $request .= "\r\n";
        $request .= $body;
        
        fwrite($socket, $request);
        
        $response = '';
        while (!feof($socket)) {
            $response .= fread($socket, 8192);
        }
        
        fclose($socket);
        
        $parts = explode("\r\n\r\n", $response, 2);
        $headers = $parts[0] ?? '';
        $responseBody = $parts[1] ?? '';
        
        if (stripos($headers, 'Transfer-Encoding: chunked') !== false) {
            $responseBody = $this->decodeChunkedBody($responseBody);
        }
        
        if ($raw) {
            return $responseBody;
        }
        
        return json_decode($responseBody, true);
    }

    /**
     * Get all services with their log sources
     */
    public function getServicesWithLogSources(): array
    {
        $containers = $this->getContainers();
        $services = [];
        
        // Define known log sources for each service
        $logSourceConfigs = [
            'morphvox-nginx' => [
                ['id' => 'nginx_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
                ['id' => 'nginx_access', 'name' => 'access.log', 'type' => 'file', 'path' => '/var/log/nginx/access.log'],
                ['id' => 'nginx_error', 'name' => 'error.log', 'type' => 'file', 'path' => '/var/log/nginx/error.log'],
            ],
            'morphvox-api' => [
                ['id' => 'api_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
                ['id' => 'api_laravel', 'name' => 'laravel.log', 'type' => 'file', 'path' => '/var/www/html/storage/logs/laravel.log'],
                ['id' => 'api_dpkg', 'name' => 'dpkg.log', 'type' => 'file', 'path' => '/var/log/dpkg.log'],
                ['id' => 'api_alternatives', 'name' => 'alternatives.log', 'type' => 'file', 'path' => '/var/log/alternatives.log'],
                ['id' => 'api_bootstrap', 'name' => 'bootstrap.log', 'type' => 'file', 'path' => '/var/log/bootstrap.log'],
            ],
            'morphvox-api-worker' => [
                ['id' => 'worker_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
                ['id' => 'worker_laravel', 'name' => 'laravel.log', 'type' => 'file', 'path' => '/var/www/html/storage/logs/laravel.log'],
                ['id' => 'worker_dpkg', 'name' => 'dpkg.log', 'type' => 'file', 'path' => '/var/log/dpkg.log'],
                ['id' => 'worker_alternatives', 'name' => 'alternatives.log', 'type' => 'file', 'path' => '/var/log/alternatives.log'],
                ['id' => 'worker_bootstrap', 'name' => 'bootstrap.log', 'type' => 'file', 'path' => '/var/log/bootstrap.log'],
            ],
            'morphvox-web' => [
                ['id' => 'web_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
            ],
            'morphvox-voice-engine' => [
                ['id' => 'voice_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
                ['id' => 'voice_engine', 'name' => 'voice_engine.log', 'type' => 'file', 'path' => '/app/logs/voice_engine.log'],
                ['id' => 'voice_trainer', 'name' => 'trainer.log', 'type' => 'file', 'path' => '/app/logs/trainer.log'],
                ['id' => 'voice_training', 'name' => 'training.log', 'type' => 'file', 'path' => '/app/logs/training.log'],
                ['id' => 'voice_inference', 'name' => 'inference.log', 'type' => 'file', 'path' => '/app/logs/inference.log'],
            ],
            'morphvox-db' => [
                ['id' => 'db_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
            ],
            'morphvox-redis' => [
                ['id' => 'redis_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
            ],
            'morphvox-minio' => [
                ['id' => 'minio_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
            ],
        ];
        
        // Map container names to friendly service names
        $serviceNameMap = [
            'morphvox-nginx' => 'nginx',
            'morphvox-api' => 'api',
            'morphvox-api-worker' => 'api-worker',
            'morphvox-web' => 'web',
            'morphvox-voice-engine' => 'voice-engine',
            'morphvox-db' => 'db',
            'morphvox-redis' => 'redis',
            'morphvox-minio' => 'minio',
        ];
        
        // Add training pseudo-service first
        $services[] = [
            'name' => 'training',
            'container_name' => 'training-jobs',
            'status' => 'running',
            'log_sources' => [
                ['id' => 'training_all', 'name' => 'All Training Jobs', 'type' => 'training'],
            ],
        ];
        
        foreach ($containers as $container) {
            $containerName = $container['name'];
            $serviceName = $serviceNameMap[$containerName] ?? null;
            
            if (!$serviceName) {
                continue; // Skip unknown containers
            }
            
            $logSources = $logSourceConfigs[$containerName] ?? [
                ['id' => "{$serviceName}_stdout", 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
            ];
            
            $services[] = [
                'name' => $serviceName,
                'container_name' => $containerName,
                'status' => $container['state'] === 'running' ? 'running' : 'stopped',
                'log_sources' => $logSources,
            ];
        }
        
        // Sort services in a specific order
        $order = ['training', 'nginx', 'api', 'api-worker', 'web', 'voice-engine', 'db', 'redis', 'minio'];
        usort($services, function($a, $b) use ($order) {
            $aIdx = array_search($a['name'], $order);
            $bIdx = array_search($b['name'], $order);
            if ($aIdx === false) $aIdx = 999;
            if ($bIdx === false) $bIdx = 999;
            return $aIdx - $bIdx;
        });
        
        return $services;
    }
}
