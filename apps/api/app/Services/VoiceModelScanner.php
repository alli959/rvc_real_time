<?php

namespace App\Services;

use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

class VoiceModelScanner
{
    protected string $storageType;
    protected array $modelExtensions;
    protected array $indexExtensions;

    public function __construct()
    {
        $this->storageType = config('voice_models.storage', 'local');
        $this->modelExtensions = config('voice_models.model_extensions', ['pth', 'onnx']);
        $this->indexExtensions = config('voice_models.index_extensions', ['index']);
    }

    /**
     * Scan models from configured storage
     */
    public function scan(): array
    {
        return match ($this->storageType) {
            's3' => $this->scanS3(),
            default => $this->scanLocal(),
        };
    }

    /**
     * Scan local directory for models
     */
    protected function scanLocal(): array
    {
        $models = [];
        $basePath = $this->getLocalPath();

        if (!$basePath || !is_dir($basePath)) {
            return $models;
        }

        // Scan directories
        foreach (File::directories($basePath) as $dir) {
            $modelName = basename($dir);
            $modelInfo = $this->analyzeLocalDirectory($dir, $modelName);
            if ($modelInfo) {
                $models[] = $modelInfo;
            }
        }

        // Scan symlinked .pth files directly in models dir
        foreach (glob("$basePath/*.pth") as $pthFile) {
            if (is_link($pthFile)) {
                $modelInfo = $this->analyzeLocalSymlink($pthFile);
                if ($modelInfo) {
                    $models[] = $modelInfo;
                }
            }
        }

        return $models;
    }

    /**
     * Scan S3 bucket for models
     */
    protected function scanS3(): array
    {
        $models = [];
        $disk = config('voice_models.s3.disk', 's3');
        $prefix = config('voice_models.s3.prefix', 'models');

        try {
            $directories = Storage::disk($disk)->directories($prefix);

            foreach ($directories as $dir) {
                $modelName = basename($dir);
                $modelInfo = $this->analyzeS3Directory($disk, $dir, $modelName);
                if ($modelInfo) {
                    $models[] = $modelInfo;
                }
            }
        } catch (\Exception $e) {
            \Log::error("Failed to scan S3 for models: " . $e->getMessage());
        }

        return $models;
    }

    /**
     * Analyze a local model directory
     */
    protected function analyzeLocalDirectory(string $dir, string $modelName): ?array
    {
        if (is_link($dir)) {
            $dir = realpath($dir);
            if (!$dir) {
                return null;
            }
        }

        $modelFile = $this->findLocalModelFile($dir);
        if (!$modelFile) {
            // Check subdirectories
            foreach (File::directories($dir) as $subdir) {
                $modelFile = $this->findLocalModelFile($subdir);
                if ($modelFile) {
                    $dir = $subdir;
                    break;
                }
            }
        }

        if (!$modelFile) {
            return null;
        }

        // Note: We now support both exported models AND raw G_*.pth checkpoints
        // The voice-engine handles both formats

        $indexFile = $this->findLocalIndexFile($dir);
        $modelSize = file_exists($modelFile) ? filesize($modelFile) : 0;

        return [
            'slug' => $modelName,
            'name' => $this->formatModelName($modelName),
            'model_file' => basename($modelFile),
            'model_path' => $modelFile,
            'index_file' => $indexFile ? basename($indexFile) : null,
            'index_path' => $indexFile,
            'has_index' => (bool) $indexFile,
            'size_bytes' => $modelSize,
            'storage_type' => 'local',
            'storage_path' => null,
            'index_storage_path' => null,
            'engine' => config('voice_models.default_engine', 'rvc'),
            'metadata' => $this->extractLocalMetadata($dir, $modelFile),
            'last_synced_at' => now(),
        ];
    }

    /**
     * Analyze a local symlink
     */
    protected function analyzeLocalSymlink(string $pthFile): ?array
    {
        $realPath = realpath($pthFile);
        if (!$realPath) {
            return null;
        }

        $modelName = pathinfo($pthFile, PATHINFO_FILENAME);

        return [
            'slug' => $modelName,
            'name' => $this->formatModelName($modelName),
            'model_file' => basename($realPath),
            'model_path' => $realPath,
            'index_file' => null,
            'index_path' => null,
            'has_index' => false,
            'size_bytes' => file_exists($realPath) ? filesize($realPath) : 0,
            'storage_type' => 'local',
            'storage_path' => null,
            'index_storage_path' => null,
            'engine' => config('voice_models.default_engine', 'rvc'),
            'metadata' => null,
            'last_synced_at' => now(),
        ];
    }

    /**
     * Analyze an S3 model directory
     */
    protected function analyzeS3Directory(string $disk, string $dir, string $modelName): ?array
    {
        $files = Storage::disk($disk)->files($dir);
        
        $modelFile = $this->findS3ModelFile($files);
        if (!$modelFile) {
            // Check subdirectories
            $subdirs = Storage::disk($disk)->directories($dir);
            foreach ($subdirs as $subdir) {
                $subFiles = Storage::disk($disk)->files($subdir);
                $modelFile = $this->findS3ModelFile($subFiles);
                if ($modelFile) {
                    $files = $subFiles;
                    $dir = $subdir;
                    break;
                }
            }
        }

        if (!$modelFile) {
            return null;
        }

        $indexFile = $this->findS3IndexFile($files);
        $modelSize = Storage::disk($disk)->size($modelFile);

        return [
            'slug' => $modelName,
            'name' => $this->formatModelName($modelName),
            'model_file' => basename($modelFile),
            'model_path' => Storage::disk($disk)->url($modelFile),
            'index_file' => $indexFile ? basename($indexFile) : null,
            'index_path' => $indexFile ? Storage::disk($disk)->url($indexFile) : null,
            'has_index' => (bool) $indexFile,
            'size_bytes' => $modelSize,
            'storage_type' => 's3',
            'storage_path' => $modelFile,
            'index_storage_path' => $indexFile,
            'engine' => config('voice_models.default_engine', 'rvc'),
            'metadata' => $this->extractS3Metadata($disk, $dir, $modelFile),
            'last_synced_at' => now(),
        ];
    }

    /**
     * Find model file in local directory
     * Supports both exported models and raw training checkpoints (G_*.pth)
     */
    protected function findLocalModelFile(string $dir): ?string
    {
        // Priority 1: *_infer.pth (explicitly marked inference model)
        $inferFiles = glob("$dir/*_infer.pth");
        if (!empty($inferFiles)) {
            return $inferFiles[0];
        }

        // Priority 2: Named model files (not G_*.pth or D_*.pth patterns)
        // These are typically properly exported models from RVC WebUI
        $pthFiles = glob("$dir/*.pth");
        foreach ($pthFiles as $file) {
            $basename = basename($file);
            // Skip training checkpoints for now (will check in priority 3)
            if (preg_match('/^[GD]_\d+\.pth$/', $basename)) {
                continue;
            }
            // Skip discriminator models
            if (str_starts_with($basename, 'D_')) {
                continue;
            }
            return $file;
        }

        // Priority 3: G_<number>.pth - find the highest numbered checkpoint
        // These are raw training checkpoints that can still be used for inference
        $generatorFiles = glob("$dir/G_*.pth");
        if (!empty($generatorFiles)) {
            // Sort by the number in the filename to get the highest
            usort($generatorFiles, function ($a, $b) {
                preg_match('/G_(\d+)\.pth$/', $a, $matchA);
                preg_match('/G_(\d+)\.pth$/', $b, $matchB);
                $numA = isset($matchA[1]) ? (int)$matchA[1] : 0;
                $numB = isset($matchB[1]) ? (int)$matchB[1] : 0;
                return $numB - $numA; // Descending order
            });
            return $generatorFiles[0]; // Return highest numbered
        }

        return null;
    }

    /**
     * Find index file in local directory
     */
    protected function findLocalIndexFile(string $dir): ?string
    {
        $patterns = ['added_*.index', 'trained_*.index', '*.index'];

        foreach ($patterns as $pattern) {
            $files = glob("$dir/$pattern");
            if (!empty($files)) {
                return $files[0];
            }
        }

        return null;
    }

    /**
     * Find model file in S3 file list
     * Only selects properly exported inference models
     */
    protected function findS3ModelFile(array $files): ?string
    {
        // Priority 1: *_infer.pth (explicitly marked inference model)
        foreach ($files as $file) {
            if (str_contains($file, '_infer.pth')) {
                return $file;
            }
        }

        // Priority 2: Named model files (not G_*.pth or D_*.pth patterns)
        foreach ($files as $file) {
            if (!str_ends_with($file, '.pth')) {
                continue;
            }
            $basename = basename($file);
            // Skip training checkpoints (G_*.pth, D_*.pth)
            if (preg_match('/^[GD]_\d+\.pth$/', $basename)) {
                continue;
            }
            // Skip discriminator models
            if (str_starts_with($basename, 'D_')) {
                continue;
            }
            return $file;
        }

        // Do NOT use G_*.pth files - they are raw training checkpoints
        return null;
    }

    /**
     * Find index file in S3 file list
     */
    protected function findS3IndexFile(array $files): ?string
    {
        // Priority: added_*, trained_*, any .index
        foreach ($files as $file) {
            if (str_ends_with($file, '.index') && str_contains(basename($file), 'added_')) {
                return $file;
            }
        }
        foreach ($files as $file) {
            if (str_ends_with($file, '.index') && str_contains(basename($file), 'trained_')) {
                return $file;
            }
        }
        foreach ($files as $file) {
            if (str_ends_with($file, '.index')) {
                return $file;
            }
        }

        return null;
    }

    /**
     * Extract metadata from local model
     */
    protected function extractLocalMetadata(string $dir, string $modelFile): ?array
    {
        $metadata = [];

        // Extract epochs from filename
        if (preg_match('/[_-]e?(\d+)e?[_-]s?(\d+)/', basename($modelFile), $matches)) {
            $metadata['epochs'] = (int) $matches[1];
            $metadata['steps'] = (int) $matches[2];
        }

        // Check for config.json
        $configFile = "$dir/config.json";
        if (file_exists($configFile)) {
            $config = json_decode(file_get_contents($configFile), true);
            if ($config) {
                $metadata['config'] = $config;
            }
        }

        return !empty($metadata) ? $metadata : null;
    }

    /**
     * Extract metadata from S3 model
     */
    protected function extractS3Metadata(string $disk, string $dir, string $modelFile): ?array
    {
        $metadata = [];

        // Extract epochs from filename
        if (preg_match('/[_-]e?(\d+)e?[_-]s?(\d+)/', basename($modelFile), $matches)) {
            $metadata['epochs'] = (int) $matches[1];
            $metadata['steps'] = (int) $matches[2];
        }

        // Check for config.json
        $configPath = "$dir/config.json";
        if (Storage::disk($disk)->exists($configPath)) {
            try {
                $config = json_decode(Storage::disk($disk)->get($configPath), true);
                if ($config) {
                    $metadata['config'] = $config;
                }
            } catch (\Exception $e) {
                // Ignore
            }
        }

        return !empty($metadata) ? $metadata : null;
    }

    /**
     * Format model name for display
     */
    protected function formatModelName(string $name): string
    {
        return Str::of($name)
            ->replace(['_', '-'], ' ')
            ->title()
            ->toString();
    }

    /**
     * Get local models path
     */
    public function getLocalPath(): ?string
    {
        $path = config('voice_models.local.path');
        
        // If relative path, resolve from base
        if ($path && !str_starts_with($path, '/')) {
            $path = base_path($path);
        }
        
        return $path ? realpath($path) : null;
    }

    /**
     * Get current storage type
     */
    public function getStorageType(): string
    {
        return $this->storageType;
    }

    /**
     * Validate that a model file is a proper inference model (not a raw training checkpoint)
     * Uses Python script to check the checkpoint format
     */
    protected function validateModelFormat(string $modelPath): bool
    {
        // Quick checks first
        $basename = basename($modelPath);
        
        // Skip D_*.pth discriminator files
        if (str_starts_with($basename, 'D_')) {
            return false;
        }
        
        // Skip G_*.pth training checkpoints (unless *_infer.pth)
        if (preg_match('/^G_\d+\.pth$/', $basename)) {
            return false;
        }
        
        // Run Python validation script
        $scriptPath = base_path('../../services/voice-engine/scripts/validate_model.py');
        if (!file_exists($scriptPath)) {
            // Try alternative path (in case base_path resolution differs)
            $scriptPath = realpath(__DIR__ . '/../../../../services/voice-engine/scripts/validate_model.py');
        }
        if (!$scriptPath || !file_exists($scriptPath)) {
            \Log::warning("Model validation script not found");
            // Fall back to accepting the model (user may fix manually)
            return true;
        }
        
        $command = sprintf(
            'python3 %s %s 2>/dev/null',
            escapeshellarg($scriptPath),
            escapeshellarg($modelPath)
        );
        
        $output = [];
        $returnCode = 0;
        exec($command, $output, $returnCode);
        
        if ($returnCode === 0) {
            return true;
        }
        
        // Parse the JSON output for logging
        if (!empty($output)) {
            try {
                $result = json_decode(implode('', $output), true);
                if (isset($result['reason'])) {
                    \Log::info("Model validation failed for {$modelPath}: " . $result['reason']);
                }
            } catch (\Exception $e) {
                // Ignore JSON parse errors
            }
        }
        
        return false;
    }
}
