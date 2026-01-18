<?php

namespace App\Services;

use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Cache;

/**
 * Service for reading Docker container logs.
 * 
 * This service uses Docker API via TCP proxy (docker-proxy container) or Unix socket
 * to fetch container logs. Falls back to reading from mounted log directories when possible.
 * 
 * In Docker Desktop WSL2 environments, the Unix socket proxy doesn't work from inside containers,
 * so we use a socat-based TCP proxy (docker-proxy service) on port 2375.
 */
class DockerLogsService
{
    /**
     * Container name to ID mapping (cached)
     */
    private array $containerIds = [];
    
    /**
     * Whether to use CLI-based approach (fallback when socket doesn't work)
     */
    private bool $useCliMode = false;
    
    /**
     * Docker API endpoint (TCP proxy or Unix socket)
     */
    private ?string $dockerEndpoint = null;
    
    /**
     * Cache key for Docker accessibility status
     */
    private const DOCKER_ACCESS_CACHE_KEY = 'docker_logs_service:docker_accessible';
    
    /**
     * Cache duration for Docker accessibility check (5 minutes)
     */
    private const DOCKER_ACCESS_CACHE_TTL = 300;
    
    /**
     * Docker TCP proxy host (socat container)
     */
    private const DOCKER_PROXY_HOST = 'docker-proxy';
    private const DOCKER_PROXY_PORT = 2375;

    /**
     * Get list of all running containers
     */
    public function getContainers(): array
    {
        // Check cached Docker accessibility first
        if ($this->isDockerAccessible()) {
            $containers = $this->getContainersViaSocket();
            if (!empty($containers)) {
                return $containers;
            }
        }
        
        // Fallback to CLI mode (also checks cache)
        $this->useCliMode = true;
        return $this->getContainersViaCli();
    }
    
    /**
     * Get containers via Docker socket API
     */
    private function getContainersViaSocket(): array
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
            // Only log once via cache check, not every request
            Log::debug('Docker socket not accessible: ' . $e->getMessage());
            return [];
        }
    }
    
    /**
     * Get containers via Docker CLI (fallback method)
     */
    private function getContainersViaCli(): array
    {
        // If we already know Docker is not accessible, skip CLI attempts
        if (Cache::get(self::DOCKER_ACCESS_CACHE_KEY) === false) {
            return $this->getStaticContainerList();
        }
        
        try {
            // Use docker ps with JSON format
            $output = shell_exec('docker ps -a --format "{{json .}}" 2>&1');
            
            if (empty($output) || str_contains($output, 'Cannot connect')) {
                // Docker CLI also not working - cache this result and return static list
                Cache::put(self::DOCKER_ACCESS_CACHE_KEY, false, self::DOCKER_ACCESS_CACHE_TTL);
                return $this->getStaticContainerList();
            }
            
            $containers = [];
            $lines = explode("\n", trim($output));
            
            foreach ($lines as $line) {
                if (empty($line)) continue;
                
                $container = json_decode($line, true);
                if (!$container) continue;
                
                $name = $container['Names'] ?? '';
                $containers[] = [
                    'id' => $container['ID'] ?? '',
                    'name' => $name,
                    'image' => $container['Image'] ?? '',
                    'state' => strtolower($container['State'] ?? ''),
                    'status' => $container['Status'] ?? '',
                ];
                
                $this->containerIds[$name] = $container['ID'] ?? '';
            }
            
            return $containers;
        } catch (\Exception $e) {
            Log::debug('Failed to get Docker containers via CLI: ' . $e->getMessage());
            return $this->getStaticContainerList();
        }
    }
    
    /**
     * Get static container list when Docker access is not available
     */
    private function getStaticContainerList(): array
    {
        // Return known MorphVox containers with assumed running state
        return [
            ['id' => 'morphvox-nginx', 'name' => 'morphvox-nginx', 'image' => 'nginx:alpine', 'state' => 'running', 'status' => 'Up'],
            ['id' => 'morphvox-api', 'name' => 'morphvox-api', 'image' => 'morphvox-api', 'state' => 'running', 'status' => 'Up'],
            ['id' => 'morphvox-api-worker', 'name' => 'morphvox-api-worker', 'image' => 'morphvox-api', 'state' => 'running', 'status' => 'Up'],
            ['id' => 'morphvox-web', 'name' => 'morphvox-web', 'image' => 'morphvox-web', 'state' => 'running', 'status' => 'Up'],
            ['id' => 'morphvox-voice-engine', 'name' => 'morphvox-voice-engine', 'image' => 'morphvox-voice-engine', 'state' => 'running', 'status' => 'Up'],
            ['id' => 'morphvox-db', 'name' => 'morphvox-db', 'image' => 'mariadb:10.11', 'state' => 'running', 'status' => 'Up'],
            ['id' => 'morphvox-redis', 'name' => 'morphvox-redis', 'image' => 'redis:7-alpine', 'state' => 'running', 'status' => 'Up'],
            ['id' => 'morphvox-minio', 'name' => 'morphvox-minio', 'image' => 'minio/minio', 'state' => 'running', 'status' => 'Up'],
        ];
    }

    /**
     * Get contextual unavailable message based on container type
     */
    private function getUnavailableMessage(string $containerName): array
    {
        // API and Worker have local laravel.log available
        if (in_array($containerName, ['morphvox-api', 'morphvox-api-worker'])) {
            return [
                "Container stdout logs are not available (Docker access restricted).",
                "",
                "Select 'laravel.log' from the log sources above to view application logs.",
            ];
        }
        
        // Voice engine suggestion
        if ($containerName === 'morphvox-voice-engine') {
            return [
                "Voice Engine container logs are not available.",
                "",
                "Docker socket access is restricted in this environment.",
                "Check the voice-engine service directly or view logs via 'docker logs' on the host.",
            ];
        }
        
        // Database services
        if (in_array($containerName, ['morphvox-db', 'morphvox-redis'])) {
            return [
                ucfirst(str_replace('morphvox-', '', $containerName)) . " logs are not available.",
                "",
                "Docker socket access is restricted in this environment.",
                "Database services typically don't require frequent log monitoring.",
            ];
        }
        
        // Generic message for nginx, web, minio, etc.
        return [
            "Container logs are not available.",
            "",
            "Docker socket access is restricted in this environment.",
            "To view these logs, use 'docker logs {$containerName}' on the host machine.",
        ];
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
        // First try reading from mounted Docker log files (most reliable, works without socket)
        $logsFromFile = $this->getLogsFromMountedDir($containerName, $lines);
        if (!empty($logsFromFile) && $logsFromFile[0] !== "Container log file not found") {
            return $logsFromFile;
        }
        
        // Try Docker socket API (only if socket is accessible)
        if ($this->isDockerAccessible()) {
            $logsFromSocket = $this->getLogsViaSocket($containerName, $lines, $timestamps);
            if (!empty($logsFromSocket)) {
                return $logsFromSocket;
            }
        }
        
        // Fallback: Return contextual message
        return $this->getUnavailableMessage($containerName);
    }
    
    /**
     * Get logs from mounted /var/lib/docker/containers directory
     */
    private function getLogsFromMountedDir(string $containerName, int $lines = 200): array
    {
        try {
            $containersDir = '/var/lib/docker/containers';
            
            if (!is_dir($containersDir) || !is_readable($containersDir)) {
                return ["Container log file not found"];
            }
            
            // Try to find container by scanning directories and matching config
            $containerId = $this->findContainerIdFromMountedDir($containerName, $containersDir);
            
            if (!$containerId) {
                // Fall back to cached ID if available
                $containerId = $this->containerIds[$containerName] ?? null;
            }
            
            if (!$containerId || $containerId === $containerName) {
                // If the ID equals the name, we're using static list - no real container ID
                return ["Container log file not found"];
            }
            
            // Full container ID might be truncated, find the matching directory
            $fullId = null;
            $dirs = @scandir($containersDir);
            if ($dirs === false) {
                return ["Container log file not found"];
            }
            
            foreach ($dirs as $dir) {
                if ($dir === '.' || $dir === '..') continue;
                if (str_starts_with($dir, substr($containerId, 0, 12))) {
                    $fullId = $dir;
                    break;
                }
            }
            
            if (!$fullId) {
                return ["Container log file not found"];
            }
            
            $logFile = "{$containersDir}/{$fullId}/{$fullId}-json.log";
            
            if (!file_exists($logFile)) {
                return ["Container log file not found"];
            }
            
            // Read last N lines from the log file
            $output = shell_exec("tail -n {$lines} " . escapeshellarg($logFile) . " 2>&1");
            
            if (empty($output)) {
                return [];
            }
            
            // Parse JSON log format
            $logLines = [];
            foreach (explode("\n", trim($output)) as $line) {
                if (empty($line)) continue;
                
                $entry = json_decode($line, true);
                if ($entry && isset($entry['log'])) {
                    $logLines[] = rtrim($entry['log']);
                } else {
                    $logLines[] = $line;
                }
            }
            
            return $logLines;
        } catch (\Exception $e) {
            // Don't log this as error - it's expected when Docker access is unavailable
            Log::debug("Failed to read logs from mounted dir for '$containerName': " . $e->getMessage());
            return ["Container log file not found"];
        }
    }
    
    /**
     * Find container ID by scanning mounted /var/lib/docker/containers directory
     * This works even when Docker socket is not accessible
     */
    private function findContainerIdFromMountedDir(string $containerName, string $containersDir): ?string
    {
        // Check cache first
        if (isset($this->containerIds[$containerName]) && $this->containerIds[$containerName] !== $containerName) {
            return $this->containerIds[$containerName];
        }
        
        $dirs = @scandir($containersDir);
        if ($dirs === false) {
            return null;
        }
        
        foreach ($dirs as $dir) {
            if ($dir === '.' || $dir === '..') continue;
            
            // Check the config file for container name
            $configPath = "{$containersDir}/{$dir}/config.v2.json";
            if (file_exists($configPath)) {
                $configContent = @file_get_contents($configPath);
                if ($configContent) {
                    $config = json_decode($configContent, true);
                    $name = ltrim($config['Name'] ?? '', '/');
                    if ($name === $containerName) {
                        $this->containerIds[$containerName] = $dir;
                        return $dir;
                    }
                }
            }
        }
        
        return null;
    }
    
    /**
     * Check if Docker socket/CLI is accessible (cached for performance)
     */
    public function isDockerAccessible(): bool
    {
        // Check if we already determined we need CLI mode (which failed)
        if ($this->useCliMode) {
            return false;
        }
        
        // Check cache first
        $cached = Cache::get(self::DOCKER_ACCESS_CACHE_KEY);
        if ($cached !== null) {
            return $cached;
        }
        
        // Try TCP proxy first (docker-proxy container) - works in Docker Desktop WSL2
        $tcpSocket = @stream_socket_client(
            "tcp://" . self::DOCKER_PROXY_HOST . ":" . self::DOCKER_PROXY_PORT, 
            $errno, $errstr, 2
        );
        
        if ($tcpSocket) {
            fclose($tcpSocket);
            Log::info('Docker API accessible via TCP proxy (docker-proxy container)');
            Cache::put(self::DOCKER_ACCESS_CACHE_KEY, true, self::DOCKER_ACCESS_CACHE_TTL);
            return true;
        }
        
        // Try Unix socket as fallback
        $socketPath = '/var/run/docker.sock';
        
        if (!file_exists($socketPath)) {
            Cache::put(self::DOCKER_ACCESS_CACHE_KEY, false, self::DOCKER_ACCESS_CACHE_TTL);
            return false;
        }
        
        $socket = @stream_socket_client("unix://{$socketPath}", $errno, $errstr, 1);
        
        if (!$socket) {
            // Log once that Docker is not accessible, then cache the result
            Log::info('Docker socket not accessible - start docker-proxy container or check socket permissions');
            Cache::put(self::DOCKER_ACCESS_CACHE_KEY, false, self::DOCKER_ACCESS_CACHE_TTL);
            return false;
        }
        
        fclose($socket);
        Cache::put(self::DOCKER_ACCESS_CACHE_KEY, true, self::DOCKER_ACCESS_CACHE_TTL);
        return true;
    }
    
    /**
     * Get logs via Docker socket API
     */
    private function getLogsViaSocket(string $containerName, int $lines = 200, bool $timestamps = false): array
    {
        try {
            $containerId = $this->getContainerId($containerName);
            
            if (!$containerId || $containerId === $containerName) {
                // No real container ID available
                return [];
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
            Log::debug("Failed to get logs for container '$containerName': " . $e->getMessage());
            return [];
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
        // Try TCP proxy first (works in Docker Desktop WSL2)
        $response = $this->dockerApiRequestViaTcp($endpoint, $raw);
        if ($response !== null) {
            return $response;
        }
        
        // Fall back to Unix socket
        return $this->dockerApiRequestViaSocket($endpoint, $raw);
    }
    
    /**
     * Make a request to Docker API via TCP proxy (docker-proxy container)
     */
    private function dockerApiRequestViaTcp(string $endpoint, bool $raw = false): mixed
    {
        $host = self::DOCKER_PROXY_HOST;
        $port = self::DOCKER_PROXY_PORT;
        
        $socket = @stream_socket_client("tcp://{$host}:{$port}", $errno, $errstr, 3);
        
        if (!$socket) {
            return null;
        }
        
        // Send HTTP request
        $request = "GET {$endpoint} HTTP/1.1\r\n";
        $request .= "Host: {$host}\r\n";
        $request .= "Connection: close\r\n";
        $request .= "\r\n";
        
        fwrite($socket, $request);
        
        // Read response
        $response = '';
        stream_set_timeout($socket, 5);
        while (!feof($socket)) {
            $response .= fread($socket, 8192);
        }
        
        fclose($socket);
        
        return $this->parseHttpResponse($response, $raw);
    }
    
    /**
     * Make a request to Docker API via Unix socket
     */
    private function dockerApiRequestViaSocket(string $endpoint, bool $raw = false): mixed
    {
        $socketPath = '/var/run/docker.sock';
        
        if (!file_exists($socketPath)) {
            return null;
        }
        
        $socket = @stream_socket_client("unix://{$socketPath}", $errno, $errstr, 5);
        
        if (!$socket) {
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
        
        return $this->parseHttpResponse($response, $raw);
    }
    
    /**
     * Parse HTTP response body
     */
    private function parseHttpResponse(string $response, bool $raw = false): mixed
    {
        if (empty($response)) {
            return null;
        }
        
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
        // Docker socket is required for exec into container
        if (!$this->isDockerAccessible()) {
            return $this->getUnavailableMessage($containerName);
        }
        
        try {
            // Try to get container ID from mounted dir first, then fall back to API
            $containersDir = '/var/lib/docker/containers';
            $containerId = null;
            
            if (is_dir($containersDir) && is_readable($containersDir)) {
                $containerId = $this->findContainerIdFromMountedDir($containerName, $containersDir);
            }
            
            if (!$containerId) {
                $containerId = $this->getContainerId($containerName);
            }
            
            if (!$containerId || $containerId === $containerName) {
                return $this->getUnavailableMessage($containerName);
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
            Log::debug("Failed to read file from container '$containerName': " . $e->getMessage());
            return ['Container log file not available'];
        }
    }

    /**
     * Make a POST request to Docker API
     */
    private function dockerApiPost(string $endpoint, array $data, bool $raw = false): mixed
    {
        // Try TCP proxy first
        $response = $this->dockerApiPostViaTcp($endpoint, $data, $raw);
        if ($response !== null) {
            return $response;
        }
        
        // Fall back to Unix socket
        return $this->dockerApiPostViaSocket($endpoint, $data, $raw);
    }
    
    /**
     * Make a POST request via TCP proxy
     */
    private function dockerApiPostViaTcp(string $endpoint, array $data, bool $raw = false): mixed
    {
        $host = self::DOCKER_PROXY_HOST;
        $port = self::DOCKER_PROXY_PORT;
        
        $socket = @stream_socket_client("tcp://{$host}:{$port}", $errno, $errstr, 3);
        
        if (!$socket) {
            return null;
        }
        
        $body = json_encode($data);
        
        $request = "POST {$endpoint} HTTP/1.1\r\n";
        $request .= "Host: {$host}\r\n";
        $request .= "Content-Type: application/json\r\n";
        $request .= "Content-Length: " . strlen($body) . "\r\n";
        $request .= "Connection: close\r\n";
        $request .= "\r\n";
        $request .= $body;
        
        fwrite($socket, $request);
        
        $response = '';
        stream_set_timeout($socket, 5);
        while (!feof($socket)) {
            $response .= fread($socket, 8192);
        }
        
        fclose($socket);
        
        return $this->parseHttpResponse($response, $raw);
    }
    
    /**
     * Make a POST request via Unix socket
     */
    private function dockerApiPostViaSocket(string $endpoint, array $data, bool $raw = false): mixed
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
        
        return $this->parseHttpResponse($response, $raw);
    }

    /**
     * Get all services with their log sources
     */
    public function getServicesWithLogSources(): array
    {
        $containers = $this->getContainers();
        $services = [];
        
        // Check if Docker socket/CLI is accessible
        $dockerAccessible = $this->isDockerAccessible();
        
        // Define known log sources for each service
        // Note: When Docker is not accessible, we put file-based logs first (they're readable locally)
        $logSourceConfigs = [
            'morphvox-nginx' => $dockerAccessible ? [
                ['id' => 'nginx_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
                ['id' => 'nginx_access', 'name' => 'access.log', 'type' => 'file', 'path' => '/var/log/nginx/access.log'],
                ['id' => 'nginx_error', 'name' => 'error.log', 'type' => 'file', 'path' => '/var/log/nginx/error.log'],
            ] : [
                ['id' => 'nginx_stdout', 'name' => 'Container Logs (requires Docker access)', 'type' => 'stdout', 'disabled' => true],
            ],
            'morphvox-api' => $dockerAccessible ? [
                ['id' => 'api_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
                ['id' => 'api_laravel', 'name' => 'laravel.log', 'type' => 'file', 'path' => '/var/www/html/storage/logs/laravel.log'],
            ] : [
                // Put laravel.log first when Docker is not accessible (it's readable locally)
                ['id' => 'api_laravel', 'name' => 'laravel.log ✓', 'type' => 'file', 'path' => '/var/www/html/storage/logs/laravel.log'],
                ['id' => 'api_stdout', 'name' => 'Container Logs (requires Docker access)', 'type' => 'stdout', 'disabled' => true],
            ],
            'morphvox-api-worker' => $dockerAccessible ? [
                ['id' => 'worker_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
                ['id' => 'worker_laravel', 'name' => 'laravel.log', 'type' => 'file', 'path' => '/var/www/html/storage/logs/laravel.log'],
            ] : [
                ['id' => 'worker_laravel', 'name' => 'laravel.log ✓', 'type' => 'file', 'path' => '/var/www/html/storage/logs/laravel.log'],
                ['id' => 'worker_stdout', 'name' => 'Container Logs (requires Docker access)', 'type' => 'stdout', 'disabled' => true],
            ],
            'morphvox-web' => [
                ['id' => 'web_stdout', 'name' => $dockerAccessible ? 'Container Logs (stdout/stderr)' : 'Container Logs (requires Docker access)', 'type' => 'stdout', 'disabled' => !$dockerAccessible],
            ],
            'morphvox-voice-engine' => $dockerAccessible ? [
                ['id' => 'voice_stdout', 'name' => 'Container Logs (stdout/stderr)', 'type' => 'stdout'],
                ['id' => 'voice_engine', 'name' => 'voice_engine.log', 'type' => 'file', 'path' => '/app/logs/voice_engine.log'],
                ['id' => 'voice_trainer', 'name' => 'trainer.log', 'type' => 'file', 'path' => '/app/logs/trainer.log'],
                ['id' => 'voice_training', 'name' => 'training.log', 'type' => 'file', 'path' => '/app/logs/training.log'],
                ['id' => 'voice_inference', 'name' => 'inference.log', 'type' => 'file', 'path' => '/app/logs/inference.log'],
            ] : [
                ['id' => 'voice_stdout', 'name' => 'Container Logs (requires Docker access)', 'type' => 'stdout', 'disabled' => true],
            ],
            'morphvox-db' => [
                ['id' => 'db_stdout', 'name' => $dockerAccessible ? 'Container Logs (stdout/stderr)' : 'Container Logs (requires Docker access)', 'type' => 'stdout', 'disabled' => !$dockerAccessible],
            ],
            'morphvox-redis' => [
                ['id' => 'redis_stdout', 'name' => $dockerAccessible ? 'Container Logs (stdout/stderr)' : 'Container Logs (requires Docker access)', 'type' => 'stdout', 'disabled' => !$dockerAccessible],
            ],
            'morphvox-minio' => [
                ['id' => 'minio_stdout', 'name' => $dockerAccessible ? 'Container Logs (stdout/stderr)' : 'Container Logs (requires Docker access)', 'type' => 'stdout', 'disabled' => !$dockerAccessible],
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
