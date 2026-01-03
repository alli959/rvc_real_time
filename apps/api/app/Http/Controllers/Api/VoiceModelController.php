<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\VoiceModel;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

class VoiceModelController extends Controller
{
    /**
     * List models (public + user's own)
     */
    public function index(Request $request)
    {
        $query = VoiceModel::query();

        // Filter by visibility
        if ($request->user()) {
            $query->accessibleBy($request->user()->id);
        } else {
            $query->public();
        }

        // Filter by engine
        if ($request->has('engine')) {
            $query->where('engine', $request->engine);
        }

        // Filter by tags
        if ($request->has('tags')) {
            $tags = is_array($request->tags) ? $request->tags : [$request->tags];
            $query->whereJsonContains('tags', $tags);
        }

        // Search by name
        if ($request->has('search')) {
            $query->where('name', 'ilike', '%' . $request->search . '%');
        }

        // Sort
        $sortBy = $request->get('sort', 'created_at');
        $sortDir = $request->get('direction', 'desc');
        $query->orderBy($sortBy, $sortDir);

        $models = $query->with('user:id,name')->paginate($request->get('per_page', 20));

        return response()->json($models);
    }

    /**
     * Get single model
     */
    public function show(Request $request, VoiceModel $voiceModel)
    {
        // Check access
        if (!$voiceModel->isPublic() && !$voiceModel->isOwnedBy($request->user())) {
            abort(403, 'Access denied');
        }

        return response()->json([
            'model' => $voiceModel->load('user:id,name'),
        ]);
    }

    /**
     * Create new model (metadata only, files uploaded separately)
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
        $this->authorize('update', $voiceModel);

        $validated = $request->validate([
            'name' => 'sometimes|string|max:255',
            'description' => 'nullable|string|max:5000',
            'visibility' => 'sometimes|in:public,private,unlisted',
            'tags' => 'nullable|array',
            'tags.*' => 'string|max:50',
            'has_consent' => 'boolean',
            'consent_notes' => 'nullable|string|max:1000',
            'metadata' => 'nullable|array',
        ]);

        // Only admins can make models public
        if (isset($validated['visibility']) && $validated['visibility'] === 'public') {
            if (!$request->user()->canPublishModels()) {
                unset($validated['visibility']);
            }
        }

        $voiceModel->update($validated);

        return response()->json([
            'model' => $voiceModel->fresh(),
        ]);
    }

    /**
     * Delete model
     */
    public function destroy(Request $request, VoiceModel $voiceModel)
    {
        $this->authorize('delete', $voiceModel);

        // Delete files from storage
        $prefix = $voiceModel->getStoragePrefix();
        Storage::disk('s3')->deleteDirectory($prefix);

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
        $this->authorize('update', $voiceModel);

        return response()->json([
            'upload_urls' => $this->generateUploadUrls($voiceModel),
        ]);
    }

    /**
     * Get presigned download URLs for model files
     */
    public function getDownloadUrls(Request $request, VoiceModel $voiceModel)
    {
        if (!$voiceModel->isPublic() && !$voiceModel->isOwnedBy($request->user())) {
            abort(403, 'Access denied');
        }

        $urls = [];
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

        // Record download event
        if ($request->user()) {
            \App\Models\UsageEvent::recordDownload($request->user()->id, $voiceModel->id);
        }
        $voiceModel->incrementDownloads();

        return response()->json(['download_urls' => $urls]);
    }

    /**
     * Confirm upload completed (marks model as ready)
     */
    public function confirmUpload(Request $request, VoiceModel $voiceModel)
    {
        $this->authorize('update', $voiceModel);

        $validated = $request->validate([
            'model_uploaded' => 'required|boolean',
            'index_uploaded' => 'boolean',
        ]);

        $disk = Storage::disk('s3');
        $prefix = $voiceModel->getStoragePrefix();

        // Verify files exist
        if (!$disk->exists("{$prefix}/model.pth")) {
            return response()->json(['error' => 'Model file not found'], 422);
        }

        $voiceModel->update([
            'model_path' => "{$prefix}/model.pth",
            'index_path' => $disk->exists("{$prefix}/model.index") ? "{$prefix}/model.index" : null,
            'config_path' => $disk->exists("{$prefix}/config.json") ? "{$prefix}/config.json" : null,
            'status' => 'ready',
        ]);

        return response()->json([
            'model' => $voiceModel->fresh(),
            'message' => 'Model is now ready to use',
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
