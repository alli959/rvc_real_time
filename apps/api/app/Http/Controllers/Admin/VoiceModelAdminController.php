<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\User;
use App\Models\VoiceModel;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Artisan;

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

        return view('admin.models.edit', [
            'voiceModel' => $voiceModel,
        ]);
    }

    public function update(Request $request, VoiceModel $voiceModel)
    {
        $validated = $request->validate([
            'name' => ['required', 'string', 'max:255'],
            'description' => ['nullable', 'string', 'max:2000'],
            'visibility' => ['required', 'in:public,private,unlisted'],
            'is_active' => ['nullable', 'boolean'],
            'is_featured' => ['nullable', 'boolean'],
            'tags' => ['nullable', 'string', 'max:500'],
        ]);

        $voiceModel->update([
            'name' => $validated['name'],
            'description' => $validated['description'],
            'visibility' => $validated['visibility'],
            'is_active' => $request->boolean('is_active'),
            'is_featured' => $request->boolean('is_featured'),
            'tags' => $validated['tags'] ? array_map('trim', explode(',', $validated['tags'])) : null,
        ]);

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
}
