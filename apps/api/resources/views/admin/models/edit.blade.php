@extends('admin.layout')

@section('title', 'Edit Model')
@section('header')
<div>
  <p class="text-sm text-gray-400 mb-1">Edit Voice Model</p>
  <h1 class="text-2xl font-bold">{{ $voiceModel->name }}</h1>
</div>
@endsection

@section('header-actions')
<div class="flex gap-2">
  <a href="{{ route('admin.models.access.edit', $voiceModel) }}" class="inline-flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
    <i data-lucide="shield" class="w-4 h-4"></i>
    Manage Access
  </a>
  <a href="{{ route('admin.models.index') }}" class="inline-flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
    <i data-lucide="arrow-left" class="w-4 h-4"></i>
    Back to Models
  </a>
</div>
@endsection

@section('content')
<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <!-- Main Form -->
  <div class="lg:col-span-2">
    <form method="POST" action="{{ route('admin.models.update', $voiceModel) }}" class="bg-gray-900 border border-gray-800 rounded-xl p-6">
      @csrf
      @method('PUT')

      <div class="mb-6">
        <label for="name" class="block text-sm font-medium text-gray-400 mb-2">Model Name</label>
        <input
          type="text"
          id="name"
          name="name"
          value="{{ old('name', $voiceModel->name) }}"
          required
          class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
        @error('name')
          <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
        @enderror
      </div>

      <div class="mb-6">
        <label for="description" class="block text-sm font-medium text-gray-400 mb-2">Description</label>
        <textarea
          id="description"
          name="description"
          rows="4"
          placeholder="Describe this voice model..."
          class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
        >{{ old('description', $voiceModel->description) }}</textarea>
        @error('description')
          <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
        @enderror
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div>
          <label for="visibility" class="block text-sm font-medium text-gray-400 mb-2">Visibility</label>
          <select
            id="visibility"
            name="visibility"
            required
            class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="public" @selected(old('visibility', $voiceModel->visibility) === 'public')>
              🌍 Public — Anyone can see and use
            </option>
            <option value="unlisted" @selected(old('visibility', $voiceModel->visibility) === 'unlisted')>
              🔗 Unlisted — Only with direct link
            </option>
            <option value="private" @selected(old('visibility', $voiceModel->visibility) === 'private')>
              🔒 Private — Only owner and granted users
            </option>
          </select>
          @error('visibility')
            <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
          @enderror
        </div>

        <div>
          <label for="tags" class="block text-sm font-medium text-gray-400 mb-2">Tags</label>
          <input
            type="text"
            id="tags"
            name="tags"
            value="{{ old('tags', is_array($voiceModel->tags) ? implode(', ', $voiceModel->tags) : $voiceModel->tags) }}"
            placeholder="male, english, deep (comma separated)"
            class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          @error('tags')
            <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
          @enderror
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <label class="flex items-center gap-3 cursor-pointer bg-gray-800 border border-gray-700 rounded-lg p-4 hover:border-gray-600 transition-colors">
          <input
            type="checkbox"
            name="is_active"
            value="1"
            @checked(old('is_active', $voiceModel->is_active))
            class="w-4 h-4 rounded bg-gray-700 border-gray-600 text-primary-600 focus:ring-primary-500 focus:ring-offset-gray-800"
          />
          <div>
            <span class="text-sm font-medium">Active</span>
            <p class="text-xs text-gray-500">Model is available for use</p>
          </div>
        </label>

        <label class="flex items-center gap-3 cursor-pointer bg-gray-800 border border-gray-700 rounded-lg p-4 hover:border-gray-600 transition-colors">
          <input
            type="checkbox"
            name="is_featured"
            value="1"
            @checked(old('is_featured', $voiceModel->is_featured))
            class="w-4 h-4 rounded bg-gray-700 border-gray-600 text-primary-600 focus:ring-primary-500 focus:ring-offset-gray-800"
          />
          <div>
            <span class="text-sm font-medium">Featured</span>
            <p class="text-xs text-gray-500">Show in featured section</p>
          </div>
        </label>
      </div>

      <div class="flex justify-end gap-3 pt-4 border-t border-gray-800">
        <a href="{{ route('admin.models.index') }}" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
          Cancel
        </a>
        <button type="submit" class="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm font-medium transition-colors">
          <i data-lucide="save" class="w-4 h-4"></i>
          Save Changes
        </button>
      </div>
    </form>
  </div>

  <!-- Sidebar Info -->
  <div class="space-y-6">
    <!-- Model Info Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 class="text-sm font-medium text-gray-400 mb-4">Model Information</h3>

      <div class="space-y-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gray-800 rounded-lg flex items-center justify-center">
            <i data-lucide="audio-waveform" class="w-5 h-5 text-gray-500"></i>
          </div>
          <div>
            <p class="text-sm font-medium">{{ $voiceModel->name }}</p>
            <p class="text-xs text-gray-500">{{ $voiceModel->slug }}</p>
          </div>
        </div>

        <div class="pt-4 border-t border-gray-800 space-y-3">
          <div class="flex justify-between text-sm">
            <span class="text-gray-500">Type</span>
            <span class="px-2 py-1 text-xs font-medium rounded-full {{ $voiceModel->isSystemModel() ? 'bg-purple-500/10 text-purple-400' : 'bg-blue-500/10 text-blue-400' }}">
              {{ $voiceModel->isSystemModel() ? 'System' : 'User Uploaded' }}
            </span>
          </div>

          <div class="flex justify-between text-sm">
            <span class="text-gray-500">Status</span>
            <span class="px-2 py-1 text-xs font-medium rounded-full 
              @if($voiceModel->status === 'ready') bg-green-500/10 text-green-400
              @elseif($voiceModel->status === 'pending') bg-yellow-500/10 text-yellow-400
              @else bg-red-500/10 text-red-400
              @endif">
              {{ ucfirst($voiceModel->status) }}
            </span>
          </div>

          <div class="flex justify-between text-sm">
            <span class="text-gray-500">Engine</span>
            <span class="text-gray-300">{{ $voiceModel->engine ?? 'RVC' }}</span>
          </div>

          <div class="flex justify-between text-sm">
            <span class="text-gray-500">Has Index</span>
            <span class="text-gray-300">{{ $voiceModel->has_index ? 'Yes' : 'No' }}</span>
          </div>

          @if($voiceModel->size_bytes)
          <div class="flex justify-between text-sm">
            <span class="text-gray-500">Size</span>
            <span class="text-gray-300">{{ number_format($voiceModel->size_bytes / 1024 / 1024, 1) }} MB</span>
          </div>
          @endif

          <div class="flex justify-between text-sm">
            <span class="text-gray-500">Storage</span>
            <span class="text-gray-300">{{ ucfirst($voiceModel->storage_type ?? 'local') }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Owner / Creator Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 class="text-sm font-medium text-gray-400 mb-4">Owner & Creator</h3>

      <div class="space-y-4">
        @if($voiceModel->user)
          <div>
            <p class="text-xs text-gray-500 mb-2">Uploaded By</p>
            <div class="flex items-center gap-3">
              <div class="w-8 h-8 bg-primary-600/20 rounded-full flex items-center justify-center">
                <span class="text-xs font-bold text-primary-400">{{ substr($voiceModel->user->name, 0, 1) }}</span>
              </div>
              <div>
                <p class="text-sm font-medium">{{ $voiceModel->user->name }}</p>
                <p class="text-xs text-gray-500">{{ $voiceModel->user->email }}</p>
              </div>
            </div>
          </div>
        @else
          <div>
            <p class="text-xs text-gray-500 mb-2">Uploaded By</p>
            <div class="flex items-center gap-3 text-gray-400">
              <div class="w-8 h-8 bg-purple-600/20 rounded-full flex items-center justify-center">
                <i data-lucide="server" class="w-4 h-4 text-purple-400"></i>
              </div>
              <div>
                <p class="text-sm font-medium">System</p>
                <p class="text-xs text-gray-500">Auto-synced from storage</p>
              </div>
            </div>
          </div>
        @endif

        @if($voiceModel->metadata && isset($voiceModel->metadata['creator']))
          <div class="pt-4 border-t border-gray-800">
            <p class="text-xs text-gray-500 mb-2">Original Creator</p>
            <p class="text-sm text-gray-300">{{ $voiceModel->metadata['creator'] }}</p>
          </div>
        @endif

        @if($voiceModel->metadata && isset($voiceModel->metadata['credit']))
          <div class="pt-4 border-t border-gray-800">
            <p class="text-xs text-gray-500 mb-2">Credit / Source</p>
            <p class="text-sm text-gray-300">{{ $voiceModel->metadata['credit'] }}</p>
          </div>
        @endif
      </div>
    </div>

    <!-- Stats Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 class="text-sm font-medium text-gray-400 mb-4">Statistics</h3>

      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-800 rounded-lg p-3 text-center">
          <p class="text-2xl font-bold text-primary-400">{{ number_format($voiceModel->usage_count ?? 0) }}</p>
          <p class="text-xs text-gray-500">Times Used</p>
        </div>
        <div class="bg-gray-800 rounded-lg p-3 text-center">
          <p class="text-2xl font-bold text-green-400">{{ number_format($voiceModel->download_count ?? 0) }}</p>
          <p class="text-xs text-gray-500">Downloads</p>
        </div>
      </div>

      <div class="mt-4 pt-4 border-t border-gray-800 space-y-2 text-sm">
        <div class="flex justify-between">
          <span class="text-gray-500">Created</span>
          <span class="text-gray-400">{{ $voiceModel->created_at->format('M d, Y') }}</span>
        </div>
        @if($voiceModel->last_synced_at)
        <div class="flex justify-between">
          <span class="text-gray-500">Last Synced</span>
          <span class="text-gray-400">{{ $voiceModel->last_synced_at->format('M d, Y H:i') }}</span>
        </div>
        @endif
      </div>
    </div>
  </div>
</div>
@endsection
