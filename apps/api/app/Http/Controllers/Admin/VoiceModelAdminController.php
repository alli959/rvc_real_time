<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\User;
use App\Models\VoiceModel;
use App\Services\TrainerService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Storage;

class VoiceModelAdminController extends Controller
{
    public function index(Request $request)
    {
        $q = VoiceModel::query()->with('user:id,name,email');

        // Filter by type (system = user_id IS NULL, user = user_id IS NOT NULL)
        if ($request->filled('type')) {
            if ($request->string('type') === 'system') {
                $q->whereNull('user_id');
            } elseif ($request->string('type') === 'user') {
                $q->whereNotNull('user_id');
            }
        }

        if ($request->filled('visibility')) {
            $q->where('visibility', $request->string('visibility'));
        }

        if ($request->filled('search')) {
            $search = trim($request->string('search'));
            $q->where(function ($sub) use ($search) {
                $sub->where('name', 'like', "%{$search}%")
                    ->orWhere('slug', 'like', "%{$search}%");
            });
        }

        $models = $q->orderBy('created_at', 'desc')
            ->paginate(25)
            ->withQueryString();

        return view('admin.models.index', [
            'models' => $models,
        ]);
    }

    public function sync(Request $request)
    {
        $request->validate([
            'prune' => ['nullable', 'boolean'],
            'storage' => ['nullable', 'in:local,s3'],
        ]);

        $options = ['--prune' => (bool) $request->boolean('prune', true)];

        if ($request->filled('storage')) {
            $options['--storage'] = $request->string('storage');
        }

        Artisan::call('voice-models:sync', $options);
        $output = Artisan::output();

        return redirect()->route('admin.models.index')->with('status', "Sync completed.\n" . $output);
    }

    public function edit(VoiceModel $voiceModel)
    {
        $voiceModel->load('user');
        
        // Get training info if model has a model_dir
        $trainingInfo = null;
        if ($voiceModel->model_dir) {
            $trainer = app(TrainerService::class);
            $trainingInfo = $trainer->getModelTrainingInfo($voiceModel->model_dir);
        }

        return view('admin.models.edit', [
            'voiceModel' => $voiceModel,
            'trainingInfo' => $trainingInfo,
        ]);
    }

    /**
     * Request checkpoint save for a training job (and optionally stop training)
     */
    public function requestCheckpoint(Request $request, VoiceModel $voiceModel)
    {
        $stopAfter = $request->boolean('stop_after', false);
        $jobId = $request->input('job_id');
        
        if (!$jobId) {
            return back()->with('error', 'No job ID provided');
        }
        
        $trainer = app(TrainerService::class);
        $result = $trainer->requestCheckpoint($jobId, $stopAfter);
        
        if (!$result) {
            return back()->with('error', 'Failed to request checkpoint');
        }
        
        $message = $stopAfter 
            ? 'Checkpoint save requested. Training will stop after saving.'
            : 'Checkpoint save requested. Training will continue.';
            
        return back()->with('status', $message);
    }

    public function update(Request $request, VoiceModel $voiceModel)
    {
        $validated = $request->validate([
            'name' => ['required', 'string', 'max:255'],
            'description' => ['nullable', 'string', 'max:2000'],
            'gender' => ['nullable', 'in:Male,Female'],
            'visibility' => ['required', 'in:public,private,unlisted'],
            'is_active' => ['nullable', 'boolean'],
            'is_featured' => ['nullable', 'boolean'],
            'tags' => ['nullable', 'string', 'max:500'],
            'image' => ['nullable', 'image', 'mimes:jpeg,png,jpg,gif,webp', 'max:5120'], // 5MB max
        ]);

        $updateData = [
            'name' => $validated['name'],
            'description' => $validated['description'],
            'gender' => $validated['gender'] ?? null,
            'visibility' => $validated['visibility'],
            'is_active' => $request->boolean('is_active'),
            'is_featured' => $request->boolean('is_featured'),
            'tags' => $validated['tags'] ? array_map('trim', explode(',', $validated['tags'])) : null,
        ];

        // Handle image upload
        if ($request->hasFile('image')) {
            // Delete old image if exists
            if ($voiceModel->image_path && Storage::disk('public')->exists($voiceModel->image_path)) {
                Storage::disk('public')->delete($voiceModel->image_path);
            }

            // Store new image
            $path = $request->file('image')->store('model-images', 'public');
            $updateData['image_path'] = $path;
        }

        // Handle image removal
        if ($request->boolean('remove_image') && $voiceModel->image_path) {
            if (Storage::disk('public')->exists($voiceModel->image_path)) {
                Storage::disk('public')->delete($voiceModel->image_path);
            }
            $updateData['image_path'] = null;
        }

        $voiceModel->update($updateData);

        return redirect()->route('admin.models.edit', $voiceModel)->with('status', 'Model updated.');
    }

    public function editAccess(VoiceModel $voiceModel)
    {
        $users = User::query()->with('roles')->orderBy('created_at', 'desc')->get();
        $voiceModel->load('permittedUsers');

        $existing = $voiceModel->permittedUsers
            ->keyBy('id')
            ->map(fn ($u) => [
                'can_view' => (bool) $u->pivot?->can_view,
                'can_use' => (bool) $u->pivot?->can_use,
            ])
            ->toArray();

        return view('admin.models.access', [
            'voiceModel' => $voiceModel,
            'users' => $users,
            'existing' => $existing,
        ]);
    }

    public function updateAccess(Request $request, VoiceModel $voiceModel)
    {
        $validated = $request->validate([
            'access' => ['nullable', 'array'],
            'access.*.can_view' => ['nullable', 'boolean'],
            'access.*.can_use' => ['nullable', 'boolean'],
        ]);

        $sync = [];
        foreach (($validated['access'] ?? []) as $userId => $row) {
            $sync[(int) $userId] = [
                'can_view' => !empty($row['can_view']),
                'can_use' => !empty($row['can_use']),
            ];
        }

        $voiceModel->permittedUsers()->sync($sync);

        return redirect()->route('admin.models.access.edit', $voiceModel)->with('status', 'Access updated.');
    }

    public function scanAllLanguages(Request $request, TrainerService $trainerService)
    {
        try {
            // Get all models with paths
            $models = VoiceModel::whereNotNull('model_path')
                ->where('status', 'ready')
                ->get();

            $scannedCount = 0;
            $failedCount = 0;

            foreach ($models as $model) {
                try {
                    $result = $trainerService->scanAndUpdateModel($model);
                    if ($result) {
                        $scannedCount++;
                    } else {
                        $failedCount++;
                    }
                } catch (\Exception $e) {
                    $failedCount++;
                }
            }

            return redirect()->route('admin.models.index')->with('status', 
                "Language scan completed. Scanned: {$scannedCount}, Failed: {$failedCount}"
            );
        } catch (\Exception $e) {
            return redirect()->route('admin.models.index')->with('error', 
                'Failed to scan models: ' . $e->getMessage()
            );
        }
    }

    public function scanModelLanguages(Request $request, VoiceModel $voiceModel, TrainerService $trainerService)
    {
        try {
            $result = $trainerService->scanAndUpdateModel($voiceModel);
            
            // Refresh model to get updated values
            $voiceModel->refresh();

            if ($result) {
                return redirect()->route('admin.models.edit', $voiceModel)->with('status', 
                    "Language scan completed. EN: " . number_format($voiceModel->en_readiness_score ?? 0, 1) . "%, IS: " . number_format($voiceModel->is_readiness_score ?? 0, 1) . "%"
                );
            }

            return redirect()->route('admin.models.edit', $voiceModel)->with('error', 
                'Failed to scan model languages.'
            );
        } catch (\Exception $e) {
            return redirect()->route('admin.models.edit', $voiceModel)->with('error', 
                'Scan failed: ' . $e->getMessage()
            );
        }
    }

    /**
     * Transfer ownership of a model to a user.
     * 
     * This grants the user full ownership of the model,
     * allowing them to edit, manage, and delete it.
     */
    public function transferOwnership(Request $request, VoiceModel $voiceModel)
    {
        $validated = $request->validate([
            'user_id' => ['nullable', 'exists:users,id'],
        ]);

        $newUserId = $validated['user_id'];

        // If clearing ownership (setting to system model)
        if (!$newUserId) {
            $voiceModel->update(['user_id' => null]);
            return redirect()->route('admin.models.edit', $voiceModel)->with('status', 
                'Model ownership cleared. Model is now a system model.'
            );
        }

        $newOwner = User::find($newUserId);
        
        if (!$newOwner) {
            return redirect()->route('admin.models.edit', $voiceModel)->with('error', 
                'User not found.'
            );
        }

        $voiceModel->update(['user_id' => $newOwner->id]);

        return redirect()->route('admin.models.edit', $voiceModel)->with('status', 
            "Ownership transferred to {$newOwner->name} ({$newOwner->email})."
        );
    }

    /**
     * Test model using inference (without training data).
     */
    public function testModelInference(Request $request, VoiceModel $voiceModel, TrainerService $trainerService)
    {
        try {
            $languages = $request->input('languages', ['en']);
            
            $result = $trainerService->testModelInference($voiceModel, $languages);

            if ($result) {
                $overallScore = $result['overall_score'] ?? 0;
                return redirect()->route('admin.models.edit', $voiceModel)->with('status', 
                    "Inference test completed. Overall score: " . number_format($overallScore, 1) . "%"
                );
            }

            return redirect()->route('admin.models.edit', $voiceModel)->with('error', 
                'Inference test failed. Model may not be compatible.'
            );
        } catch (\Exception $e) {
            return redirect()->route('admin.models.edit', $voiceModel)->with('error', 
                'Inference test failed: ' . $e->getMessage()
            );
        }
    }

    /**
     * Extract model from checkpoint and/or build FAISS index.
     * 
     * This handles cases where:
     * - Training completed but final model wasn't extracted (has G_*.pth but no {name}.pth)
     * - Index file is missing and needs to be rebuilt
     */
    public function extractModel(Request $request, VoiceModel $voiceModel, TrainerService $trainerService)
    {
        try {
            $validated = $request->validate([
                'extract_model' => ['nullable', 'boolean'],
                'build_index' => ['nullable', 'boolean'],
                'sample_rate' => ['nullable', 'in:32k,40k,48k'],
                'version' => ['nullable', 'in:v1,v2'],
            ]);

            // Determine model directory from model_path or slug
            $modelDir = null;
            if ($voiceModel->model_path) {
                // Extract directory from model_path (e.g., "bjarni-ben-10/bjarni-ben-10.pth" -> "bjarni-ben-10")
                $modelDir = dirname($voiceModel->model_path);
                if ($modelDir === '.' || empty($modelDir)) {
                    $modelDir = $voiceModel->slug;
                }
            } else {
                $modelDir = $voiceModel->slug;
            }

            $result = $trainerService->extractModelAndBuildIndex(
                $modelDir,
                $request->boolean('extract_model', true),
                $request->boolean('build_index', true),
                $validated['sample_rate'] ?? '48k',
                $validated['version'] ?? 'v2',
                $voiceModel->slug
            );

            if ($result && $result['success']) {
                // Update voice model record with new paths
                $updates = [];
                
                if (!empty($result['model_path'])) {
                    // Convert absolute path to relative path
                    $relativePath = $this->getRelativeModelPath($result['model_path']);
                    if ($relativePath) {
                        $updates['model_path'] = $relativePath;
                    }
                }
                
                if (!empty($result['index_path'])) {
                    $relativePath = $this->getRelativeModelPath($result['index_path']);
                    if ($relativePath) {
                        $updates['index_path'] = $relativePath;
                        $updates['has_index'] = true;
                    }
                }
                
                if (!empty($updates)) {
                    $updates['status'] = 'ready';
                    $voiceModel->update($updates);
                }

                return redirect()->route('admin.models.edit', $voiceModel)->with('status', 
                    "Model extraction completed: " . ($result['message'] ?? 'Success')
                );
            }

            return redirect()->route('admin.models.edit', $voiceModel)->with('error', 
                'Extraction failed: ' . ($result['message'] ?? 'Unknown error')
            );
        } catch (\Exception $e) {
            return redirect()->route('admin.models.edit', $voiceModel)->with('error', 
                'Extraction failed: ' . $e->getMessage()
            );
        }
    }

    /**
     * Convert absolute path to relative model path.
     */
    private function getRelativeModelPath(string $absolutePath): ?string
    {
        // Remove common prefixes to get relative path
        $prefixes = [
            '/app/assets/models/',
            '/var/www/html/storage/models/',
            storage_path('models') . '/',
        ];
        
        foreach ($prefixes as $prefix) {
            if (str_starts_with($absolutePath, $prefix)) {
                return substr($absolutePath, strlen($prefix));
            }
        }
        
        // Try to extract just the model_dir/filename part
        if (preg_match('#/([^/]+/[^/]+\.(?:pth|index))$#', $absolutePath, $matches)) {
            return $matches[1];
        }
        
        return null;
    }
}
