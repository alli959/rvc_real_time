<?php

namespace App\Services;

use Aws\S3\S3Client;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

class StorageService
{
    protected S3Client $client;
    protected string $bucket;
    protected int $presignedUrlExpiry;

    public function __construct()
    {
        $this->bucket = config('filesystems.disks.s3.bucket');
        $this->presignedUrlExpiry = config('filesystems.disks.s3.presigned_url_expiry', 3600);
        
        $this->client = new S3Client([
            'version' => 'latest',
            'region' => config('filesystems.disks.s3.region', 'us-east-1'),
            'endpoint' => config('filesystems.disks.s3.endpoint'),
            'use_path_style_endpoint' => config('filesystems.disks.s3.use_path_style_endpoint', true),
            'credentials' => [
                'key' => config('filesystems.disks.s3.key'),
                'secret' => config('filesystems.disks.s3.secret'),
            ],
        ]);
    }

    /**
     * Generate a pre-signed URL for uploading a file.
     */
    public function getUploadUrl(string $path, string $contentType = 'application/octet-stream', int $expiresIn = null): array
    {
        $expiresIn = $expiresIn ?? $this->presignedUrlExpiry;
        
        $command = $this->client->getCommand('PutObject', [
            'Bucket' => $this->bucket,
            'Key' => $path,
            'ContentType' => $contentType,
        ]);
        
        $request = $this->client->createPresignedRequest($command, "+{$expiresIn} seconds");
        
        return [
            'url' => (string) $request->getUri(),
            'method' => 'PUT',
            'headers' => [
                'Content-Type' => $contentType,
            ],
            'expires_at' => now()->addSeconds($expiresIn)->toIso8601String(),
        ];
    }

    /**
     * Generate a pre-signed URL for downloading a file.
     */
    public function getDownloadUrl(string $path, int $expiresIn = null, string $filename = null): array
    {
        $expiresIn = $expiresIn ?? $this->presignedUrlExpiry;
        
        $params = [
            'Bucket' => $this->bucket,
            'Key' => $path,
        ];
        
        if ($filename) {
            $params['ResponseContentDisposition'] = "attachment; filename=\"{$filename}\"";
        }
        
        $command = $this->client->getCommand('GetObject', $params);
        $request = $this->client->createPresignedRequest($command, "+{$expiresIn} seconds");
        
        return [
            'url' => (string) $request->getUri(),
            'method' => 'GET',
            'expires_at' => now()->addSeconds($expiresIn)->toIso8601String(),
        ];
    }

    /**
     * Check if a file exists.
     */
    public function exists(string $path): bool
    {
        return Storage::disk('s3')->exists($path);
    }

    /**
     * Get file metadata.
     */
    public function getMetadata(string $path): ?array
    {
        try {
            $result = $this->client->headObject([
                'Bucket' => $this->bucket,
                'Key' => $path,
            ]);
            
            return [
                'size' => $result['ContentLength'],
                'content_type' => $result['ContentType'],
                'last_modified' => $result['LastModified']->format('Y-m-d H:i:s'),
                'etag' => trim($result['ETag'], '"'),
            ];
        } catch (\Exception $e) {
            return null;
        }
    }

    /**
     * Delete a file.
     */
    public function delete(string $path): bool
    {
        return Storage::disk('s3')->delete($path);
    }

    /**
     * Delete multiple files with a prefix.
     */
    public function deletePrefix(string $prefix): bool
    {
        $files = Storage::disk('s3')->files($prefix);
        
        if (empty($files)) {
            return true;
        }
        
        return Storage::disk('s3')->delete($files);
    }

    /**
     * Copy a file.
     */
    public function copy(string $source, string $destination): bool
    {
        return Storage::disk('s3')->copy($source, $destination);
    }

    /**
     * Get the internal path for voice engine access.
     * This returns the path that the voice engine service can use.
     */
    public function getInternalPath(string $path): string
    {
        // For MinIO/S3, the voice engine accesses via the internal endpoint
        $endpoint = config('services.voice_engine.storage_endpoint', 'http://minio:9000');
        return "{$endpoint}/{$this->bucket}/{$path}";
    }

    /**
     * Generate paths for model storage.
     */
    public function generateModelPaths(string $userId, string $modelId): array
    {
        $prefix = "models/{$userId}/{$modelId}";
        
        return [
            'prefix' => $prefix,
            'model' => "{$prefix}/model.pth",
            'index' => "{$prefix}/model.index",
            'config' => "{$prefix}/config.json",
            'thumbnail' => "{$prefix}/thumbnail.jpg",
        ];
    }

    /**
     * Generate paths for job storage.
     */
    public function generateJobPaths(string $userId, string $jobId): array
    {
        $prefix = "jobs/{$userId}/{$jobId}";
        
        return [
            'prefix' => $prefix,
            'input' => "{$prefix}/input.wav",
            'output' => "{$prefix}/output.wav",
        ];
    }

    /**
     * Get content type for audio files.
     */
    public function getAudioContentType(string $extension): string
    {
        return match (strtolower($extension)) {
            'wav' => 'audio/wav',
            'mp3' => 'audio/mpeg',
            'flac' => 'audio/flac',
            'ogg' => 'audio/ogg',
            'm4a' => 'audio/mp4',
            default => 'application/octet-stream',
        };
    }
}
