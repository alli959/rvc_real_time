<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\VoiceModel;
use App\Models\UsageEvent;
use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use Laravel\Sanctum\PersonalAccessToken;

class VoiceModelController extends Controller
{
    /**
     * Resolve the authenticated user from the request.
     * This handles both routes with auth:sanctum middleware and public routes with bearer tokens.
     */
    protected function resolveUser(Request $request): ?User
    {
        // First check if middleware already authenticated the user
        if ($user = $request->user()) {
            return $user;
        }
        
        // For public routes, manually check for bearer token
        $bearerToken = $request->bearerToken();
        if (!$bearerToken) {
            return null;
        }
        
        // Resolve user from Sanctum personal access token
        $token = PersonalAccessToken::findToken($bearerToken);
        if (!$token) {
            return null;
        }
        
        return $token->tokenable;
    }
    
    /**
     * List models (public system models + user's own + shared)
     * 
     * Query params:
     * - public_only=true: Only show public models (for public listing pages)
     */
    public function index(Request $request)
    {
        $user = $this->resolveUser($request);
        
        // If public_only is requested, always use public scope regardless of auth
        // This is useful for public-facing model browsing pages
        if ($request->boolean('public_only')) {
            $query = VoiceModel::public();
        } elseif ($user) {
            // Logged in: show accessible models based on permissions
            $query = VoiceModel::accessibleBy($user->id);
        } else {
            // Guests only see public models
            $query = VoiceModel::public();
        }

        // Filter by type (system vs user-uploaded)
        if ($request->filled('type')) {
            if ($request->type === 'system') {
                $query->system();
            } elseif ($request->type === 'user') {
                $query->userUploaded();
            }
        }

        // Filter by engine
        if ($request->filled('engine')) {
            $query->where('engine', $request->engine);
        }

        // Filter by storage type
        if ($request->filled('storage_type')) {
            $query->where('storage_type', $request->storage_type);
        }

        // Filter by tags
        if ($request->has('tags')) {
            $tags = is_array($request->tags) ? $request->tags : [$request->tags];
            $query->whereJsonContains('tags', $tags);
        }

        // Search by name
        if ($request->filled('search')) {
            $search = $request->search;
            $query->where(function ($q) use ($search) {
                $q->where('name', 'like', "%{$search}%")
                  ->orWhere('slug', 'like', "%{$search}%");
            });
        }

        // Filter by has_index
        if ($request->has('has_index')) {
            $query->where('has_index', $request->boolean('has_index'));
        }

        // Filter featured
        if ($request->boolean('featured')) {
            $query->where('is_featured', true);
        }

        // Only ready models by default
        if (!$request->boolean('include_pending')) {
            $query->where('status', 'ready');
        }

        // Only active models by default
        if (!$request->boolean('include_inactive')) {
            $query->where('is_active', true);
        }

        // Sort
        $sortBy = $request->get('sort', 'name');
        $sortDir = $request->get('direction', 'asc');
        $allowedSorts = ['name', 'size_bytes', 'usage_count', 'created_at', 'updated_at'];
        
        if (in_array($sortBy, $allowedSorts)) {
            $query->orderBy($sortBy, $sortDir === 'desc' ? 'desc' : 'asc');
        }

        // Pagination
        $perPage = min($request->get('per_page', 50), 100);
        
        if ($request->boolean('all')) {
            $models = $query->with('user:id,name')->get();
            return response()->json([
                'data' => $models,
                'total' => $models->count(),
            ]);
        }

        $paginated = $query->with('user:id,name')->paginate($perPage);

        return response()->json([
            'data' => $paginated->items(),
            'total' => $paginated->total(),
            'per_page' => $paginated->perPage(),
            'current_page' => $paginated->currentPage(),
            'last_page' => $paginated->lastPage(),
        ]);
    }

    /**
     * Get user's own models
     */
    public function myModels(Request $request)
    {
        $query = VoiceModel::ownedBy($request->user()->id);

        // Include pending models for owner
        $sortBy = $request->get('sort', 'created_at');
        $sortDir = $request->get('direction', 'desc');
        $query->orderBy($sortBy, $sortDir);

        $models = $query->paginate($request->get('per_page', 20));

        return response()->json($models);
    }

    /**
     * Get single model by slug or route model binding
     */
    public function show(Request $request, $voiceModel)
    {
        // Support both route model binding and slug lookup
        if (is_string($voiceModel)) {
            $model = VoiceModel::where('slug', $voiceModel)->first();
            if (!$model) {
                return response()->json(['error' => 'Model not found'], 404);
            }
        } else {
            $model = $voiceModel;
        }

        $user = $this->resolveUser($request);
        
        // Check access - admins can view all models
        if (!$model->isPublic()) {
            if (!$user) {
                return response()->json(['error' => 'Access denied'], 403);
            }
            
            // Admin can see all models
            $isAdmin = $user->hasRole('admin');
            $isOwner = $model->isOwnedBy($user);
            $hasPermission = $model->permittedUsers()->where('users.id', $user->id)->exists();
            
            if (!$isAdmin && !$isOwner && !$hasPermission) {
                return response()->json(['error' => 'Access denied'], 403);
            }
        }

        // Increment usage counter
        $model->incrementUsage();

        return response()->json([
            'model' => $model->load('user:id,name'),
        ]);
    }

    /**
     * Create new model (user-uploaded, metadata only - files uploaded separately)
     */
    public function store(Request $request)
    {
        $this->authorize('create', VoiceModel::class);

        $validated = $request->validate([
            'name' => 'required|string|max:255',
            'description' => 'nullable|string|max:5000',
            'engine' => 'required|in:rvc,tts',
            'visibility' => 'required|in:public,private,unlisted',
            'tags' => 'nullable|array',
            'tags.*' => 'string|max:50',
            'has_consent' => 'boolean',
            'consent_notes' => 'nullable|string|max:1000',
            'metadata' => 'nullable|array',
        ]);

        $model = VoiceModel::create([
            ...$validated,
            'user_id' => $request->user()->id,
            'storage_type' => 's3', // User uploads go to S3
            'status' => 'pending',
        ]);

        return response()->json([
            'model' => $model,
            'upload_urls' => $this->generateUploadUrls($model),
        ], 201);
    }

    /**
     * Update model metadata
     */
    public function update(Request $request, VoiceModel $voiceModel)
    {
        $user = $request->user();

        // Check permission: owner or admin
        if (!$voiceModel->isOwnedBy($user) && !$user->hasRole('admin')) {
            return response()->json(['error' => 'Forbidden'], 403);
        }

        $validated = $request->validate([
            'name' => 'sometimes|string|max:255',
            'description' => 'nullable|string|max:5000',
            'visibility' => 'sometimes|in:public,private,unlisted',
            'tags' => 'nullable|array',
            'tags.*' => 'string|max:50',
            'has_consent' => 'boolean',
            'consent_notes' => 'nullable|string|max:1000',
            'metadata' => 'nullable|array',
            'is_active' => 'sometimes|boolean',
            'is_featured' => 'sometimes|boolean',
        ]);

        // Only admins can make models public or set featured
        if (!$user->hasRole('admin')) {
            if (isset($validated['visibility']) && $validated['visibility'] === 'public') {
                unset($validated['visibility']);
            }
            unset($validated['is_featured']);
        }

        $voiceModel->update($validated);

        return response()->json([
            'model' => $voiceModel->fresh(),
        ]);
    }

    /**
     * Delete model (user-uploaded only)
     */
    public function destroy(Request $request, VoiceModel $voiceModel)
    {
        $user = $request->user();

        // Can't delete system models via API
        if ($voiceModel->isSystemModel()) {
            return response()->json(['error' => 'Cannot delete system models'], 403);
        }

        // Check permission: owner or admin
        if (!$voiceModel->isOwnedBy($user) && !$user->hasRole('admin')) {
            return response()->json(['error' => 'Forbidden'], 403);
        }

        // Delete files from storage
        if ($voiceModel->storage_type === 's3') {
            $prefix = $voiceModel->getStoragePrefix();
            Storage::disk('s3')->deleteDirectory($prefix);
        }

        $voiceModel->delete();

        return response()->json([
            'message' => 'Model deleted successfully',
        ]);
    }

    /**
     * Get presigned upload URLs for model files
     */
    public function getUploadUrls(Request $request, VoiceModel $voiceModel)
    {
        $user = $request->user();

        if (!$voiceModel->isOwnedBy($user) && !$user->hasRole('admin')) {
            return response()->json(['error' => 'Forbidden'], 403);
        }

        return response()->json([
            'upload_urls' => $this->generateUploadUrls($voiceModel),
        ]);
    }

    /**
     * Get presigned download URLs for model files
     */
    public function getDownloadUrls(Request $request, VoiceModel $voiceModel)
    {
        $user = $request->user();

        // Check access
        if (!$voiceModel->isPublic() && (!$user || !$voiceModel->isOwnedBy($user))) {
            return response()->json(['error' => 'Access denied'], 403);
        }

        $urls = [];

        if ($voiceModel->storage_type === 's3') {
            $disk = Storage::disk('s3');

            if ($voiceModel->model_path && $disk->exists($voiceModel->model_path)) {
                $urls['model'] = $disk->temporaryUrl($voiceModel->model_path, now()->addHour());
            }

            if ($voiceModel->index_path && $disk->exists($voiceModel->index_path)) {
                $urls['index'] = $disk->temporaryUrl($voiceModel->index_path, now()->addHour());
            }

            if ($voiceModel->config_path && $disk->exists($voiceModel->config_path)) {
                $urls['config'] = $disk->temporaryUrl($voiceModel->config_path, now()->addHour());
            }
        } else {
            // Local storage - return paths directly (they'll be served via different mechanism)
            $urls['model'] = $voiceModel->model_path;
            $urls['index'] = $voiceModel->index_path;
            $urls['config'] = $voiceModel->config_path;
        }

        // Record download event
        if ($user) {
            UsageEvent::recordDownload($user->id, $voiceModel->id);
        }
        $voiceModel->incrementDownloads();

        return response()->json(['download_urls' => $urls]);
    }

    /**
     * Confirm upload completed (marks model as ready)
     */
    public function confirmUpload(Request $request, VoiceModel $voiceModel)
    {
        $user = $request->user();

        if (!$voiceModel->isOwnedBy($user) && !$user->hasRole('admin')) {
            return response()->json(['error' => 'Forbidden'], 403);
        }

        $validated = $request->validate([
            'model_uploaded' => 'required|boolean',
            'index_uploaded' => 'boolean',
        ]);

        $disk = Storage::disk('s3');
        $prefix = $voiceModel->getStoragePrefix();

        // Verify model file exists
        if (!$disk->exists("{$prefix}/model.pth")) {
            return response()->json(['error' => 'Model file not found'], 422);
        }

        $voiceModel->update([
            'model_path' => "{$prefix}/model.pth",
            'index_path' => $disk->exists("{$prefix}/model.index") ? "{$prefix}/model.index" : null,
            'config_path' => $disk->exists("{$prefix}/config.json") ? "{$prefix}/config.json" : null,
            'has_index' => $disk->exists("{$prefix}/model.index"),
            'status' => 'ready',
        ]);

        return response()->json([
            'model' => $voiceModel->fresh(),
            'message' => 'Model is now ready to use',
        ]);
    }

    /**
     * Trigger sync of system models from storage (admin only)
     */
    public function sync(Request $request)
    {
        $prune = $request->boolean('prune', false);
        $storage = $request->get('storage'); // optional override

        try {
            $options = ['--prune' => $prune];
            
            if ($storage) {
                $options['--storage'] = $storage;
            }

            Artisan::call('voice-models:sync', $options);
            $output = Artisan::output();

            return response()->json([
                'message' => 'Sync completed successfully',
                'storage' => $storage ?? config('voice_models.storage'),
                'output' => $output,
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Sync failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Get stats
     */
    public function stats(Request $request)
    {
        $systemOnly = $request->boolean('system', false);
        $query = $systemOnly ? VoiceModel::system() : VoiceModel::query();

        return response()->json([
            'total' => (clone $query)->count(),
            'active' => (clone $query)->where('is_active', true)->count(),
            'featured' => (clone $query)->where('is_featured', true)->count(),
            'with_index' => (clone $query)->where('has_index', true)->count(),
            'system_models' => VoiceModel::system()->count(),
            'user_models' => VoiceModel::userUploaded()->count(),
            'by_engine' => (clone $query)->selectRaw('engine, count(*) as count')
                ->groupBy('engine')
                ->pluck('count', 'engine'),
            'by_storage' => (clone $query)->selectRaw('storage_type, count(*) as count')
                ->groupBy('storage_type')
                ->pluck('count', 'storage_type'),
            'total_size_bytes' => (clone $query)->sum('size_bytes'),
            'last_synced' => VoiceModel::system()->max('last_synced_at'),
            'configured_storage' => config('voice_models.storage'),
        ]);
    }

    /**
     * Get current configuration (admin only)
     */
    public function config()
    {
        return response()->json([
            'storage' => config('voice_models.storage'),
            'local_path' => config('voice_models.local.path'),
            's3_disk' => config('voice_models.s3.disk'),
            's3_prefix' => config('voice_models.s3.prefix'),
            'default_engine' => config('voice_models.default_engine'),
        ]);
    }

    /**
     * Generate presigned upload URLs
     */
    protected function generateUploadUrls(VoiceModel $model): array
    {
        $prefix = $model->getStoragePrefix();
        $disk = Storage::disk('s3');
        $expiry = now()->addHour();

        return [
            'model' => $disk->temporaryUploadUrl("{$prefix}/model.pth", $expiry),
            'index' => $disk->temporaryUploadUrl("{$prefix}/model.index", $expiry),
            'config' => $disk->temporaryUploadUrl("{$prefix}/config.json", $expiry),
        ];
    }
}
