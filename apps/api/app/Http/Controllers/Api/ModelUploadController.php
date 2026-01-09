<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\VoiceModel;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use Illuminate\Validation\Rules\File;

/**
 * Controller for direct model file uploads.
 * 
 * Handles uploading .pth model files, .index files, and config.json files
 * to either S3/MinIO or local storage based on configuration.
 */
class ModelUploadController extends Controller
{
    /**
     * Get the configured storage disk and type for model uploads
     */
    protected function getStorageDisk(): array
    {
        $defaultDisk = config('filesystems.default', 'local');
        $preferS3 = env('MODEL_STORAGE', $defaultDisk) === 's3';
        
        // Check if S3 is configured and accessible
        if ($preferS3) {
            $s3Config = config('filesystems.disks.s3');
            if (!empty($s3Config['key']) && !empty($s3Config['secret']) && !empty($s3Config['bucket'])) {
                return [
                    'disk' => Storage::disk('s3'),
                    'type' => 's3'
                ];
            }
            \Log::warning('S3 preferred but not configured, falling back to local storage');
        }
        
        return [
            'disk' => Storage::disk('local'),
            'type' => 'local'
        ];
    }

    /**
     * Upload model files directly (model.pth, model.index, config.json)
     * 
     * This endpoint accepts multipart form data with:
     * - model_file: Required .pth file
     * - index_file: Optional .index file  
     * - config_file: Optional config.json file
     * - name: Model name
     * - description: Model description
     * - visibility: public/private/unlisted
     * - tags: Array of tags
     */
    public function upload(Request $request)
    {
        $user = $request->user();
        
        // Check permission
        if (!$user->canUploadModels() && !$user->hasRole('admin')) {
            return response()->json([
                'error' => 'You do not have permission to upload models',
            ], 403);
        }

        $validated = $request->validate([
            'name' => 'required|string|max:255',
            'description' => 'nullable|string|max:5000',
            'visibility' => 'required|in:public,private,unlisted',
            'tags' => 'nullable|array',
            'tags.*' => 'string|max:50',
            'has_consent' => 'boolean',
            'consent_notes' => 'nullable|string|max:1000',
            'model_file' => [
                'required',
                'file',
                'max:204800', // 200MB max (increased to match PHP upload_max_filesize)
                function ($attribute, $value, $fail) {
                    $ext = strtolower($value->getClientOriginalExtension());
                    if ($ext !== 'pth') {
                        $fail('The model file must be a .pth file.');
                    }
                },
            ],
            'index_file' => [
                'nullable',
                'file',
                'max:204800', // 200MB max
                function ($attribute, $value, $fail) {
                    if ($value) {
                        $ext = strtolower($value->getClientOriginalExtension());
                        if ($ext !== 'index') {
                            $fail('The index file must be a .index file.');
                        }
                    }
                },
            ],
            'config_file' => [
                'nullable',
                'file',
                'max:5120', // 5MB max
                function ($attribute, $value, $fail) {
                    if ($value) {
                        $ext = strtolower($value->getClientOriginalExtension());
                        if ($ext !== 'json') {
                            $fail('The config file must be a .json file.');
                        }
                    }
                },
            ],
        ]);

        // Get storage disk (S3 or local)
        $storage = $this->getStorageDisk();
        $disk = $storage['disk'];
        $storageType = $storage['type'];

        // Create the model record first
        $voiceModel = VoiceModel::create([
            'uuid' => (string) Str::uuid(),
            'user_id' => $user->id,
            'name' => $validated['name'],
            'description' => $validated['description'] ?? null,
            'engine' => 'rvc',
            'visibility' => $validated['visibility'],
            'tags' => $validated['tags'] ?? [],
            'has_consent' => $validated['has_consent'] ?? false,
            'consent_notes' => $validated['consent_notes'] ?? null,
            'storage_type' => $storageType,
            'status' => 'pending',
        ]);

        // For local storage, use models directory; for S3 use uuid prefix
        $prefix = $storageType === 'local' 
            ? 'models/' . $voiceModel->uuid
            : $voiceModel->getStoragePrefix();

        try {
            // Upload model file using streaming to handle large files
            $modelFile = $request->file('model_file');
            $modelPath = "{$prefix}/model.pth";
            
            \Log::info("Starting model upload: {$modelPath}, size: " . $modelFile->getSize());
            
            // Use stream for better memory handling with large files
            $stream = fopen($modelFile->getRealPath(), 'r');
            $disk->put($modelPath, $stream);
            if (is_resource($stream)) {
                fclose($stream);
            }
            
            \Log::info("Model file uploaded successfully: {$modelPath}");
            
            $modelSize = $modelFile->getSize();

            // Upload index file if provided
            $indexPath = null;
            if ($request->hasFile('index_file')) {
                $indexFile = $request->file('index_file');
                $indexPath = "{$prefix}/model.index";
                
                \Log::info("Uploading index file: {$indexPath}");
                
                $indexStream = fopen($indexFile->getRealPath(), 'r');
                $disk->put($indexPath, $indexStream);
                if (is_resource($indexStream)) {
                    fclose($indexStream);
                }
                
                $modelSize += $indexFile->getSize();
                \Log::info("Index file uploaded successfully");
            }

            // Upload config file if provided
            $configPath = null;
            if ($request->hasFile('config_file')) {
                $configFile = $request->file('config_file');
                $configPath = "{$prefix}/config.json";
                
                \Log::info("Uploading config file: {$configPath}");
                
                $configStream = fopen($configFile->getRealPath(), 'r');
                $disk->put($configPath, $configStream);
                if (is_resource($configStream)) {
                    fclose($configStream);
                }
                
                $modelSize += $configFile->getSize();
                \Log::info("Config file uploaded successfully");
            }

            // Update model record with paths
            $voiceModel->update([
                'model_path' => $modelPath,
                'index_path' => $indexPath,
                'config_path' => $configPath,
                'has_index' => $indexPath !== null,
                'size_bytes' => $modelSize,
                'status' => 'ready',
            ]);
            
            \Log::info("Model upload complete: {$voiceModel->uuid}, total size: {$modelSize}");

            return response()->json([
                'message' => 'Model uploaded successfully',
                'model' => $voiceModel->fresh(),
            ], 201);

        } catch (\Exception $e) {
            \Log::error("Model upload failed: " . $e->getMessage(), [
                'model_uuid' => $voiceModel->uuid ?? null,
                'exception' => $e,
            ]);
            
            // Clean up on failure
            $voiceModel->delete();
            $disk->deleteDirectory($prefix);

            return response()->json([
                'error' => 'Upload failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Upload additional files to an existing model
     */
    public function uploadFiles(Request $request, VoiceModel $voiceModel)
    {
        $user = $request->user();

        if (!$voiceModel->isOwnedBy($user) && !$user->hasRole('admin')) {
            return response()->json(['error' => 'Forbidden'], 403);
        }

        $validated = $request->validate([
            'index_file' => [
                'nullable',
                'file',
                'max:204800', // 200MB max
                function ($attribute, $value, $fail) {
                    if ($value) {
                        $ext = strtolower($value->getClientOriginalExtension());
                        if ($ext !== 'index') {
                            $fail('The index file must be a .index file.');
                        }
                    }
                },
            ],
            'config_file' => [
                'nullable',
                'file',
                'max:5120', // 5MB max
                function ($attribute, $value, $fail) {
                    if ($value) {
                        $ext = strtolower($value->getClientOriginalExtension());
                        if ($ext !== 'json') {
                            $fail('The config file must be a .json file.');
                        }
                    }
                },
            ],
        ]);

        if (!$request->hasFile('index_file') && !$request->hasFile('config_file')) {
            return response()->json([
                'error' => 'No files provided',
            ], 422);
        }

        // Use the model's storage type to determine which disk to use
        $disk = $voiceModel->storage_type === 's3' 
            ? Storage::disk('s3') 
            : Storage::disk('local');
        $prefix = $voiceModel->storage_type === 'local'
            ? 'models/' . $voiceModel->uuid
            : $voiceModel->getStoragePrefix();
        $updates = [];

        try {
            if ($request->hasFile('index_file')) {
                $indexFile = $request->file('index_file');
                $indexPath = "{$prefix}/model.index";
                $disk->putFileAs(
                    dirname($indexPath),
                    $indexFile,
                    basename($indexPath)
                );
                $updates['index_path'] = $indexPath;
                $updates['has_index'] = true;
            }

            if ($request->hasFile('config_file')) {
                $configFile = $request->file('config_file');
                $configPath = "{$prefix}/config.json";
                $disk->putFileAs(
                    dirname($configPath),
                    $configFile,
                    basename($configPath)
                );
                $updates['config_path'] = $configPath;
            }

            $voiceModel->update($updates);

            return response()->json([
                'message' => 'Files uploaded successfully',
                'model' => $voiceModel->fresh(),
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Upload failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Replace the model file for an existing model
     */
    public function replaceModel(Request $request, VoiceModel $voiceModel)
    {
        $user = $request->user();

        if (!$voiceModel->isOwnedBy($user) && !$user->hasRole('admin')) {
            return response()->json(['error' => 'Forbidden'], 403);
        }

        $validated = $request->validate([
            'model_file' => [
                'required',
                'file',
                'max:204800', // 200MB max
                function ($attribute, $value, $fail) {
                    $ext = strtolower($value->getClientOriginalExtension());
                    if ($ext !== 'pth') {
                        $fail('The model file must be a .pth file.');
                    }
                },
            ],
        ]);

        // Use the model's storage type to determine which disk to use
        $disk = $voiceModel->storage_type === 's3' 
            ? Storage::disk('s3') 
            : Storage::disk('local');
        $prefix = $voiceModel->storage_type === 'local'
            ? 'models/' . $voiceModel->uuid
            : $voiceModel->getStoragePrefix();

        try {
            // Delete old model file
            if ($voiceModel->model_path && $disk->exists($voiceModel->model_path)) {
                $disk->delete($voiceModel->model_path);
            }

            // Upload new model file
            $modelFile = $request->file('model_file');
            $modelPath = "{$prefix}/model.pth";
            $disk->putFileAs(
                dirname($modelPath),
                $modelFile,
                basename($modelPath)
            );

            $voiceModel->update([
                'model_path' => $modelPath,
                'size_bytes' => $modelFile->getSize() + 
                    ($voiceModel->index_path ? $disk->size($voiceModel->index_path) : 0),
            ]);

            return response()->json([
                'message' => 'Model file replaced successfully',
                'model' => $voiceModel->fresh(),
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Upload failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }
}
