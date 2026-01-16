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
    <form method="POST" action="{{ route('admin.models.update', $voiceModel) }}" enctype="multipart/form-data" class="bg-gray-900 border border-gray-800 rounded-xl p-6">
      @csrf
      @method('PUT')

      <!-- Model Image Upload -->
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-400 mb-2">Model Image</label>
        <div class="flex items-start gap-4">
          <!-- Current Image Preview -->
          <div class="w-32 h-32 bg-gray-800 rounded-lg flex items-center justify-center overflow-hidden flex-shrink-0 border border-gray-700" id="image-preview-container">
            @if($voiceModel->image_path)
              <img src="{{ $voiceModel->image_url }}" alt="{{ $voiceModel->name }}" class="w-full h-full object-cover" id="image-preview" />
            @else
              <i data-lucide="image" class="w-8 h-8 text-gray-600" id="image-placeholder"></i>
              <img src="" alt="" class="w-full h-full object-cover hidden" id="image-preview" />
            @endif
          </div>
          
          <div class="flex-1 space-y-3">
            <div>
              <input
                type="file"
                id="image"
                name="image"
                accept="image/jpeg,image/png,image/jpg,image/gif,image/webp"
                class="hidden"
                onchange="previewImage(this)"
              />
              <label for="image" class="inline-flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium cursor-pointer transition-colors">
                <i data-lucide="upload" class="w-4 h-4"></i>
                Choose Image
              </label>
              <p class="mt-2 text-xs text-gray-500">JPEG, PNG, GIF, or WebP. Max 5MB.</p>
            </div>
            
            @if($voiceModel->image_path)
            <label class="flex items-center gap-2 text-sm text-red-400 hover:text-red-300 cursor-pointer">
              <input type="checkbox" name="remove_image" value="1" class="rounded bg-gray-700 border-gray-600 text-red-600 focus:ring-red-500" />
              Remove current image
            </label>
            @endif
          </div>
        </div>
        @error('image')
          <p class="mt-2 text-sm text-red-400">{{ $message }}</p>
        @enderror
      </div>

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
          <label for="gender" class="block text-sm font-medium text-gray-400 mb-2">Gender (for TTS)</label>
          <select
            id="gender"
            name="gender"
            class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="" @selected(old('gender', $voiceModel->gender) === null || old('gender', $voiceModel->gender) === '')>
              — Not Set —
            </option>
            <option value="Male" @selected(old('gender', $voiceModel->gender) === 'Male')>
              ♂ Male
            </option>
            <option value="Female" @selected(old('gender', $voiceModel->gender) === 'Female')>
              ♀ Female
            </option>
          </select>
          <p class="mt-1 text-xs text-gray-500">When set, TTS will auto-select this gender</p>
          @error('gender')
            <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
          @enderror
        </div>

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
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
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
          <div class="w-16 h-16 bg-gray-800 rounded-lg flex items-center justify-center overflow-hidden">
            @if($voiceModel->image_path)
              <img src="{{ $voiceModel->image_url }}" alt="{{ $voiceModel->name }}" class="w-full h-full object-cover" />
            @else
              <i data-lucide="audio-waveform" class="w-8 h-8 text-gray-500"></i>
            @endif
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

    <!-- Training Status Card (if training is active) -->
    @if(isset($trainingInfo) && isset($trainingInfo['training']) && $trainingInfo['training']['status'] === 'training')
    <div class="bg-gray-900 border border-blue-500/50 rounded-xl p-6">
      <h3 class="text-sm font-medium text-blue-400 mb-4 flex items-center gap-2">
        <i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i>
        Training In Progress
      </h3>
      
      <div class="space-y-3">
        <div class="flex justify-between text-sm">
          <span class="text-gray-500">Progress</span>
          <span class="text-gray-300">{{ $trainingInfo['training']['epochs_trained'] ?? 0 }} / {{ $trainingInfo['training']['total_epochs'] ?? '?' }} epochs</span>
        </div>
        
        @if(isset($trainingInfo['training']['job_id']))
        <div class="flex justify-between text-sm">
          <span class="text-gray-500">Job ID</span>
          <span class="text-gray-400 font-mono text-xs">{{ $trainingInfo['training']['job_id'] }}</span>
        </div>
        
        <!-- Checkpoint Controls -->
        <div class="pt-3 border-t border-gray-800 space-y-2">
          <p class="text-xs text-gray-500">Training Controls</p>
          
          <form method="POST" action="{{ route('admin.models.checkpoint', $voiceModel) }}" class="flex gap-2">
            @csrf
            <input type="hidden" name="job_id" value="{{ $trainingInfo['training']['job_id'] }}">
            
            <button type="submit" name="stop_after" value="0" 
              class="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
              onclick="return confirm('Save a checkpoint now and continue training?')">
              <i data-lucide="save" class="w-4 h-4"></i>
              Save Checkpoint
            </button>
            
            <button type="submit" name="stop_after" value="1" 
              class="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 bg-orange-600 hover:bg-orange-700 rounded-lg text-sm font-medium transition-colors"
              onclick="return confirm('Save a checkpoint and STOP training? You can resume later.')">
              <i data-lucide="square" class="w-4 h-4"></i>
              Save & Stop
            </button>
          </form>
        </div>
        @endif
      </div>
    </div>
    @endif

    <!-- Model Tools Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 class="text-sm font-medium text-gray-400 mb-4">Model Tools</h3>
      
      <div class="space-y-4">
        <!-- Extract Model / Build Index -->
        <div>
          <p class="text-xs text-gray-500 mb-2">Extract & Index</p>
          <p class="text-xs text-gray-600 mb-3">
            Extract final model from G_*.pth checkpoint and/or build FAISS index for voice matching.
          </p>
          <form method="POST" action="{{ route('admin.models.extract-model', $voiceModel) }}" class="space-y-3">
            @csrf
            <div class="grid grid-cols-2 gap-2">
              <label class="flex items-center gap-2 text-sm cursor-pointer">
                <input type="checkbox" name="extract_model" value="1" checked 
                  class="rounded bg-gray-700 border-gray-600 text-primary-600 focus:ring-primary-500" />
                <span class="text-gray-300">Extract Model</span>
              </label>
              <label class="flex items-center gap-2 text-sm cursor-pointer">
                <input type="checkbox" name="build_index" value="1" checked 
                  class="rounded bg-gray-700 border-gray-600 text-primary-600 focus:ring-primary-500" />
                <span class="text-gray-300">Build Index</span>
              </label>
            </div>
            
            <div class="grid grid-cols-2 gap-2">
              <div>
                <label class="text-xs text-gray-500">Sample Rate</label>
                <select name="sample_rate" class="w-full mt-1 bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary-500">
                  <option value="48k" selected>48kHz</option>
                  <option value="40k">40kHz</option>
                  <option value="32k">32kHz</option>
                </select>
              </div>
              <div>
                <label class="text-xs text-gray-500">Version</label>
                <select name="version" class="w-full mt-1 bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary-500">
                  <option value="v2" selected>v2</option>
                  <option value="v1">v1</option>
                </select>
              </div>
            </div>
            
            <button type="submit" class="w-full inline-flex items-center justify-center gap-2 px-3 py-2 bg-orange-600 hover:bg-orange-700 rounded-lg text-sm font-medium transition-colors"
              onclick="return confirm('This will extract the model from checkpoint and/or build the FAISS index. Continue?')">
              <i data-lucide="package" class="w-4 h-4"></i>
              Extract & Build Index
            </button>
          </form>
        </div>

        <!-- File Paths Info -->
        @if($voiceModel->model_path || $voiceModel->index_path)
        <div class="pt-4 border-t border-gray-800">
          <p class="text-xs text-gray-500 mb-2">Current File Paths</p>
          @if($voiceModel->model_path)
            <div class="text-xs text-gray-400 mb-1 break-all">
              <span class="text-gray-500">Model:</span> {{ $voiceModel->model_path }}
            </div>
          @endif
          @if($voiceModel->index_path)
            <div class="text-xs text-gray-400 break-all">
              <span class="text-gray-500">Index:</span> {{ $voiceModel->index_path }}
            </div>
          @endif
        </div>
        @endif
      </div>
    </div>

    <!-- Owner / Creator Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h3 class="text-sm font-medium text-gray-400 mb-4">Owner & Creator</h3>

      <div class="space-y-4">
        @if($voiceModel->user)
          <div>
            <p class="text-xs text-gray-500 mb-2">Current Owner</p>
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
            <p class="text-xs text-gray-500 mb-2">Current Owner</p>
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

        <!-- Transfer Ownership Form -->
        <form method="POST" action="{{ route('admin.models.transfer-ownership', $voiceModel) }}" class="pt-4 border-t border-gray-800">
          @csrf
          <p class="text-xs text-gray-500 mb-2">Transfer Ownership</p>
          <div class="flex gap-2">
            <select 
              name="user_id" 
              class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="">— System (No Owner) —</option>
              @foreach(\App\Models\User::orderBy('name')->get() as $user)
                <option value="{{ $user->id }}" @selected($voiceModel->user_id === $user->id)>
                  {{ $user->name }} ({{ $user->email }})
                </option>
              @endforeach
            </select>
            <button 
              type="submit"
              class="px-3 py-2 bg-primary-600 hover:bg-primary-500 rounded-lg text-sm font-medium transition-colors"
              onclick="return confirm('Transfer ownership of this model?')"
            >
              Transfer
            </button>
          </div>
          <p class="mt-2 text-xs text-gray-500">The new owner will be able to edit and delete this model.</p>
        </form>

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

    <!-- Language Readiness Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-sm font-medium text-gray-400">Language Readiness</h3>
        <div class="flex gap-2">
          <form method="POST" action="{{ route('admin.models.scan-model-languages', $voiceModel) }}" class="inline">
            @csrf
            <button type="submit" class="inline-flex items-center gap-1 px-2 py-1 bg-accent-600 hover:bg-accent-700 rounded text-xs font-medium transition-colors" title="Scan training data for phoneme coverage">
              <i data-lucide="scan" class="w-3 h-3"></i>
              Scan
            </button>
          </form>
        </div>
      </div>

      @if($voiceModel->language_scanned_at)
        <div class="space-y-4">
          <!-- English Score -->
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm font-medium">English (EN)</span>
              <span class="text-sm font-bold 
                @if(($voiceModel->en_readiness_score ?? 0) >= 80) text-green-400
                @elseif(($voiceModel->en_readiness_score ?? 0) >= 50) text-yellow-400
                @else text-red-400
                @endif">
                {{ number_format($voiceModel->en_readiness_score ?? 0, 0) }}%
              </span>
            </div>
            <div class="w-full bg-gray-800 rounded-full h-2">
              <div class="h-2 rounded-full transition-all duration-300 
                @if(($voiceModel->en_readiness_score ?? 0) >= 80) bg-green-500
                @elseif(($voiceModel->en_readiness_score ?? 0) >= 50) bg-yellow-500
                @else bg-red-500
                @endif" 
                style="width: {{ min($voiceModel->en_readiness_score ?? 0, 100) }}%"></div>
            </div>
            <p class="text-xs text-gray-500 mt-1">Phoneme coverage: {{ number_format($voiceModel->en_phoneme_coverage ?? 0, 1) }}%</p>
            @if($voiceModel->en_missing_phonemes && count($voiceModel->en_missing_phonemes) > 0)
              <p class="text-xs text-gray-600 mt-1">Missing: {{ implode(', ', array_slice($voiceModel->en_missing_phonemes, 0, 10)) }}{{ count($voiceModel->en_missing_phonemes) > 10 ? '...' : '' }}</p>
            @endif
          </div>

          <!-- Icelandic Score -->
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm font-medium">Icelandic (IS)</span>
              <span class="text-sm font-bold 
                @if(($voiceModel->is_readiness_score ?? 0) >= 80) text-green-400
                @elseif(($voiceModel->is_readiness_score ?? 0) >= 50) text-yellow-400
                @else text-red-400
                @endif">
                {{ number_format($voiceModel->is_readiness_score ?? 0, 0) }}%
              </span>
            </div>
            <div class="w-full bg-gray-800 rounded-full h-2">
              <div class="h-2 rounded-full transition-all duration-300 
                @if(($voiceModel->is_readiness_score ?? 0) >= 80) bg-green-500
                @elseif(($voiceModel->is_readiness_score ?? 0) >= 50) bg-yellow-500
                @else bg-red-500
                @endif" 
                style="width: {{ min($voiceModel->is_readiness_score ?? 0, 100) }}%"></div>
            </div>
            <p class="text-xs text-gray-500 mt-1">Phoneme coverage: {{ number_format($voiceModel->is_phoneme_coverage ?? 0, 1) }}%</p>
            @if($voiceModel->is_missing_phonemes && count($voiceModel->is_missing_phonemes) > 0)
              <p class="text-xs text-gray-600 mt-1">Missing: {{ implode(', ', array_slice($voiceModel->is_missing_phonemes, 0, 10)) }}{{ count($voiceModel->is_missing_phonemes) > 10 ? '...' : '' }}</p>
            @endif
          </div>

          <div class="pt-3 border-t border-gray-800">
            <p class="text-xs text-gray-500">Last scanned {{ $voiceModel->language_scanned_at->diffForHumans() }}</p>
          </div>
        </div>
      @else
        <div class="text-center py-4">
          <i data-lucide="languages" class="w-8 h-8 mx-auto mb-2 text-gray-600"></i>
          <p class="text-sm text-gray-500">Not scanned yet</p>
          <p class="text-xs text-gray-600 mt-1">Click "Scan" to analyze language readiness</p>
        </div>
      @endif
    </div>

    <!-- Inference Test Results Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-sm font-medium text-gray-400">Inference Test Results</h3>
        <form method="POST" action="{{ route('admin.models.test-inference', $voiceModel) }}" class="inline">
          @csrf
          <button type="submit" class="inline-flex items-center gap-1 px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs font-medium transition-colors" title="Run inference test">
            <i data-lucide="play" class="w-3 h-3"></i>
            Test
          </button>
        </form>
      </div>

      @if($voiceModel->inference_tested_at)
        <div class="space-y-4">
          <!-- Overall Score -->
          <div class="bg-gray-800 rounded-lg p-4 text-center">
            <p class="text-3xl font-bold 
              @if(($voiceModel->inference_test_score ?? 0) >= 70) text-green-400
              @elseif(($voiceModel->inference_test_score ?? 0) >= 40) text-yellow-400
              @else text-red-400
              @endif">
              {{ number_format($voiceModel->inference_test_score ?? 0, 1) }}%
            </p>
            <p class="text-xs text-gray-500 mt-1">Overall Quality Score</p>
          </div>

          <!-- Per-Language Scores -->
          <div class="grid grid-cols-2 gap-3">
            <div class="bg-gray-800 rounded-lg p-3 text-center">
              <p class="text-lg font-bold 
                @if(($voiceModel->en_inference_score ?? 0) >= 70) text-green-400
                @elseif(($voiceModel->en_inference_score ?? 0) >= 40) text-yellow-400
                @else text-red-400
                @endif">
                {{ $voiceModel->en_inference_score ? number_format($voiceModel->en_inference_score, 1) . '%' : 'N/A' }}
              </p>
              <p class="text-xs text-gray-500">English</p>
            </div>
            <div class="bg-gray-800 rounded-lg p-3 text-center">
              <p class="text-lg font-bold 
                @if(($voiceModel->is_inference_score ?? 0) >= 70) text-green-400
                @elseif(($voiceModel->is_inference_score ?? 0) >= 40) text-yellow-400
                @else text-red-400
                @endif">
                {{ $voiceModel->is_inference_score ? number_format($voiceModel->is_inference_score, 1) . '%' : 'N/A' }}
              </p>
              <p class="text-xs text-gray-500">Icelandic</p>
            </div>
          </div>

          <!-- Detailed Metrics -->
          @if($voiceModel->inference_test_results)
            @php
              $results = $voiceModel->inference_test_results;
              $enResults = $results['language_scores']['en'] ?? null;
            @endphp
            @if($enResults)
              <div class="space-y-2">
                <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider">English Metrics</h4>
                <div class="grid grid-cols-2 gap-2 text-xs">
                  <div class="flex justify-between p-2 bg-gray-800/50 rounded">
                    <span class="text-gray-500">Pitch Stability</span>
                    <span class="text-gray-300">{{ number_format($enResults['pitch_stability'] ?? 0, 1) }}%</span>
                  </div>
                  <div class="flex justify-between p-2 bg-gray-800/50 rounded">
                    <span class="text-gray-500">Audio Clarity</span>
                    <span class="text-gray-300">{{ number_format($enResults['audio_clarity'] ?? 0, 1) }}%</span>
                  </div>
                  <div class="flex justify-between p-2 bg-gray-800/50 rounded">
                    <span class="text-gray-500">Voice Consistency</span>
                    <span class="text-gray-300">{{ number_format($enResults['voice_consistency'] ?? 0, 1) }}%</span>
                  </div>
                  <div class="flex justify-between p-2 bg-gray-800/50 rounded">
                    <span class="text-gray-500">Naturalness</span>
                    <span class="text-gray-300">{{ number_format($enResults['naturalness'] ?? 0, 1) }}%</span>
                  </div>
                </div>
              </div>
            @endif
          @endif

          <div class="pt-3 border-t border-gray-800">
            <p class="text-xs text-gray-500">Last tested {{ $voiceModel->inference_tested_at->diffForHumans() }}</p>
          </div>
        </div>
      @else
        <div class="text-center py-4">
          <i data-lucide="test-tube" class="w-8 h-8 mx-auto mb-2 text-gray-600"></i>
          <p class="text-sm text-gray-500">No test results yet</p>
          <p class="text-xs text-gray-600 mt-1">Click "Test" to run inference quality test</p>
        </div>
      @endif
    </div>
  </div>
</div>

<script>
function previewImage(input) {
  const preview = document.getElementById('image-preview');
  const placeholder = document.getElementById('image-placeholder');
  
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
      preview.src = e.target.result;
      preview.classList.remove('hidden');
      if (placeholder) {
        placeholder.classList.add('hidden');
      }
    };
    
    reader.readAsDataURL(input.files[0]);
  }
}
</script>
@endsection
