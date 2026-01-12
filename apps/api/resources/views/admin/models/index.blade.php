@extends('admin.layout')

@section('title', 'Voice Models')
@section('header', 'Voice Models')

@section('header-actions')
<div class="flex items-center gap-2">
  <form method="POST" action="{{ route('admin.models.scan-languages') }}" class="inline" id="scan-all-form">
    @csrf
    <button type="submit" class="inline-flex items-center gap-2 px-4 py-2 bg-accent-600 hover:bg-accent-700 rounded-lg text-sm font-medium transition-colors" id="scan-all-btn">
      <i data-lucide="languages" class="w-4 h-4"></i>
      Scan All Languages
    </button>
  </form>
  <form method="POST" action="{{ route('admin.models.sync') }}" class="inline">
    @csrf
    <input type="hidden" name="prune" value="1">
    <button type="submit" class="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm font-medium transition-colors">
      <i data-lucide="refresh-cw" class="w-4 h-4"></i>
      Sync Storage Models
    </button>
  </form>
</div>
@endsection

@section('content')
<!-- Search/Filter -->
<div class="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-6">
  <form method="GET" action="{{ route('admin.models.index') }}" class="flex flex-wrap gap-4">
    <div class="flex-1 min-w-[200px]">
      <div class="relative">
        <div class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">
          <i data-lucide="search" class="w-4 h-4"></i>
        </div>
        <input
          name="search"
          value="{{ request('search') }}"
          placeholder="Search by name or slug..."
          class="w-full bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
      </div>
    </div>
    <div class="w-40">
      <select name="type" class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500">
        <option value="">All Types</option>
        <option value="system" @selected(request('type') === 'system')>System</option>
        <option value="user" @selected(request('type') === 'user')>User-uploaded</option>
      </select>
    </div>
    <div class="w-40">
      <select name="visibility" class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500">
        <option value="">All Visibility</option>
        <option value="public" @selected(request('visibility') === 'public')>Public</option>
        <option value="private" @selected(request('visibility') === 'private')>Private</option>
        <option value="unlisted" @selected(request('visibility') === 'unlisted')>Unlisted</option>
      </select>
    </div>
    <div class="flex gap-2">
      <button type="submit" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
        Filter
      </button>
      <a href="{{ route('admin.models.index') }}" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
        Reset
      </a>
    </div>
  </form>
</div>

<!-- Models Table -->
<div class="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
  <div class="overflow-x-auto">
    <table class="w-full">
      <thead>
        <tr class="border-b border-gray-800">
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">ID</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Model</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Type</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Owner</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Lang Scores</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Visibility</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Status</th>
          <th class="text-right px-6 py-4 text-sm font-medium text-gray-400">Actions</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-gray-800">
        @forelse($models as $model)
        <tr class="hover:bg-gray-800/50 transition-colors">
          <td class="px-6 py-4 text-sm text-gray-500">{{ $model->id }}</td>
          <td class="px-6 py-4">
            <div class="flex items-center gap-3">
              <div class="w-12 h-12 bg-gray-800 rounded-lg flex items-center justify-center overflow-hidden flex-shrink-0">
                @if($model->image_path)
                  <img src="{{ $model->image_url }}" alt="{{ $model->name }}" class="w-full h-full object-cover" />
                @else
                  <i data-lucide="audio-waveform" class="w-5 h-5 text-gray-500"></i>
                @endif
              </div>
              <div>
                <p class="text-sm font-medium">{{ $model->name }}</p>
                <p class="text-xs text-gray-500">{{ $model->slug }}</p>
              </div>
            </div>
          </td>
          <td class="px-6 py-4">
            <span class="px-2 py-1 text-xs font-medium rounded-full {{ $model->isSystemModel() ? 'bg-purple-500/10 text-purple-400' : 'bg-blue-500/10 text-blue-400' }}">
              {{ $model->isSystemModel() ? 'System' : 'User' }}
            </span>
          </td>
          <td class="px-6 py-4 text-sm text-gray-400">
            @if($model->user)
              {{ $model->user->name }}
            @else
              <span class="text-gray-600">—</span>
            @endif
          </td>
          <td class="px-6 py-4">
            @if($model->language_scanned_at)
              <div class="flex items-center gap-2 text-xs">
                <div class="flex items-center gap-1">
                  <span class="font-medium">EN:</span>
                  <span class="px-1.5 py-0.5 rounded 
                    @if($model->en_readiness_score >= 80) bg-green-500/10 text-green-400
                    @elseif($model->en_readiness_score >= 50) bg-yellow-500/10 text-yellow-400
                    @else bg-red-500/10 text-red-400
                    @endif">
                    {{ number_format($model->en_readiness_score ?? 0, 0) }}%
                  </span>
                </div>
                <div class="flex items-center gap-1">
                  <span class="font-medium">IS:</span>
                  <span class="px-1.5 py-0.5 rounded 
                    @if($model->is_readiness_score >= 80) bg-green-500/10 text-green-400
                    @elseif($model->is_readiness_score >= 50) bg-yellow-500/10 text-yellow-400
                    @else bg-red-500/10 text-red-400
                    @endif">
                    {{ number_format($model->is_readiness_score ?? 0, 0) }}%
                  </span>
                </div>
              </div>
              <p class="text-[10px] text-gray-600 mt-0.5">{{ $model->language_scanned_at->diffForHumans() }}</p>
            @else
              <span class="text-gray-600 text-xs">Not scanned</span>
            @endif
          </td>
          <td class="px-6 py-4">
            <span class="px-2 py-1 text-xs font-medium rounded-full 
              @if($model->visibility === 'public') bg-green-500/10 text-green-400
              @elseif($model->visibility === 'private') bg-yellow-500/10 text-yellow-400
              @else bg-gray-500/10 text-gray-400
              @endif">
              {{ $model->visibility }}
            </span>
          </td>
          <td class="px-6 py-4">
            <span class="px-2 py-1 text-xs font-medium rounded-full 
              @if($model->status === 'ready') bg-green-500/10 text-green-400
              @elseif($model->status === 'pending') bg-yellow-500/10 text-yellow-400
              @else bg-red-500/10 text-red-400
              @endif">
              {{ $model->status }}
            </span>
          </td>
          <td class="px-6 py-4 text-sm text-gray-500">
            {{ $model->last_synced_at?->format('M d, Y H:i') ?? '—' }}
          </td>
          <td class="px-6 py-4">
            <div class="flex items-center justify-end gap-2">
              <a href="{{ route('admin.models.edit', $model) }}" class="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors" title="Edit Model">
                <i data-lucide="pencil" class="w-4 h-4"></i>
              </a>
              <a href="{{ route('admin.models.access.edit', $model) }}" class="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors" title="Manage Access">
                <i data-lucide="shield" class="w-4 h-4"></i>
              </a>
            </div>
          </td>
        </tr>
        @empty
        <tr>
          <td colspan="8" class="px-6 py-12 text-center text-gray-500">
            <i data-lucide="audio-waveform" class="w-12 h-12 mx-auto mb-4 opacity-50"></i>
            <p>No voice models found</p>
          </td>
        </tr>
        @endforelse
      </tbody>
    </table>
  </div>

  @if($models->hasPages())
  <div class="px-6 py-4 border-t border-gray-800">
    {{ $models->links() }}
  </div>
  @endif
</div>
@endsection
