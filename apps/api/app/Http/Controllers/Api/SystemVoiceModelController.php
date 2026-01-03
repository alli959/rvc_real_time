<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\SystemVoiceModel;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Artisan;

class SystemVoiceModelController extends Controller
{
    /**
     * List all system voice models from database
     */
    public function index(Request $request)
    {
        $query = SystemVoiceModel::active();

        // Search by name
        if ($request->filled('search')) {
            $search = $request->search;
            $query->where(function ($q) use ($search) {
                $q->where('name', 'like', "%{$search}%")
                  ->orWhere('slug', 'like', "%{$search}%");
            });
        }

        // Filter by engine
        if ($request->filled('engine')) {
            $query->engine($request->engine);
        }

        // Filter by storage type
        if ($request->filled('storage_type')) {
            $query->storageType($request->storage_type);
        }

        // Filter by has_index
        if ($request->has('has_index')) {
            $query->where('has_index', $request->boolean('has_index'));
        }

        // Filter featured
        if ($request->boolean('featured')) {
            $query->featured();
        }

        // Sort
        $sortBy = $request->get('sort', 'name');
        $sortDir = $request->get('direction', 'asc');
        $allowedSorts = ['name', 'size_bytes', 'usage_count', 'created_at'];
        
        if (in_array($sortBy, $allowedSorts)) {
            $query->orderBy($sortBy, $sortDir === 'desc' ? 'desc' : 'asc');
        }

        // Pagination
        $perPage = min($request->get('per_page', 50), 100);
        
        if ($request->boolean('all')) {
            $models = $query->get();
            return response()->json([
                'data' => $models,
                'total' => $models->count(),
            ]);
        }

        $paginated = $query->paginate($perPage);

        return response()->json([
            'data' => $paginated->items(),
            'total' => $paginated->total(),
            'per_page' => $paginated->perPage(),
            'current_page' => $paginated->currentPage(),
            'last_page' => $paginated->lastPage(),
        ]);
    }

    /**
     * Get a single model by slug
     */
    public function show(string $slug)
    {
        $model = SystemVoiceModel::where('slug', $slug)->first();

        if (!$model) {
            return response()->json(['error' => 'Model not found'], 404);
        }

        // Increment usage counter
        $model->incrementUsage();

        return response()->json(['model' => $model]);
    }

    /**
     * Update model metadata (admin only)
     */
    public function update(Request $request, string $slug)
    {
        $model = SystemVoiceModel::where('slug', $slug)->firstOrFail();

        $validated = $request->validate([
            'name' => 'sometimes|string|max:255',
            'description' => 'sometimes|nullable|string',
            'is_active' => 'sometimes|boolean',
            'is_featured' => 'sometimes|boolean',
        ]);

        $model->update($validated);

        return response()->json(['model' => $model->fresh()]);
    }

    /**
     * Trigger a sync of models from storage
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
     * Get sync status and stats
     */
    public function stats()
    {
        return response()->json([
            'total' => SystemVoiceModel::count(),
            'active' => SystemVoiceModel::active()->count(),
            'featured' => SystemVoiceModel::featured()->count(),
            'with_index' => SystemVoiceModel::where('has_index', true)->count(),
            'by_engine' => SystemVoiceModel::selectRaw('engine, count(*) as count')
                ->groupBy('engine')
                ->pluck('count', 'engine'),
            'by_storage' => SystemVoiceModel::selectRaw('storage_type, count(*) as count')
                ->groupBy('storage_type')
                ->pluck('count', 'storage_type'),
            'total_size_bytes' => SystemVoiceModel::sum('size_bytes'),
            'last_synced' => SystemVoiceModel::max('last_synced_at'),
            'configured_storage' => config('voice_models.storage'),
        ]);
    }

    /**
     * Get current configuration
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
}
