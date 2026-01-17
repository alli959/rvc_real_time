@extends('admin.layout')

@section('title', 'System Assets')

@section('content')
<div class="space-y-6" x-data="assetsPage()" x-init="init()">
  <!-- Header -->
  <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
    <div>
      <h1 class="text-2xl font-bold text-white flex items-center gap-3">
        <i data-lucide="box" class="w-7 h-7"></i>
        System Assets
      </h1>
      <p class="text-gray-400 mt-1">Manage heavy components and GPU resources</p>
    </div>
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 text-sm" x-show="!error">
        <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
        <span class="text-gray-400">Live (3s)</span>
      </div>
      <div class="flex items-center gap-2 text-sm" x-show="error">
        <span class="w-2 h-2 rounded-full bg-red-500"></span>
        <span class="text-red-400">Error</span>
      </div>
      <button @click="refreshAssets()" class="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors">
        <i data-lucide="refresh-cw" class="w-4 h-4" :class="{'animate-spin': refreshing}"></i>
        Refresh
      </button>
    </div>
  </div>

  <!-- Error Banner -->
  <div x-show="error" x-cloak class="bg-red-900/50 border border-red-700 rounded-lg p-4">
    <div class="flex items-center gap-3">
      <i data-lucide="alert-circle" class="w-5 h-5 text-red-400"></i>
      <div>
        <p class="text-red-400 font-medium">Failed to load assets</p>
        <p class="text-red-300 text-sm" x-text="error"></p>
      </div>
    </div>
  </div>

  <!-- Resource Summary -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div class="flex items-center gap-3">
        <div class="p-3 bg-green-500/20 rounded-lg">
          <i data-lucide="check-circle" class="w-6 h-6 text-green-400"></i>
        </div>
        <div>
          <p class="text-2xl font-bold text-white" x-text="summary.total_running">0</p>
          <p class="text-sm text-gray-400">Running Assets</p>
        </div>
      </div>
    </div>
    
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div class="flex items-center gap-3">
        <div class="p-3 bg-purple-500/20 rounded-lg">
          <i data-lucide="cpu" class="w-6 h-6 text-purple-400"></i>
        </div>
        <div>
          <p class="text-2xl font-bold text-white" x-text="formatMB(summary.total_vram_mb)">0 MB</p>
          <p class="text-sm text-gray-400">Estimated VRAM Used</p>
        </div>
      </div>
    </div>
    
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div class="flex items-center gap-3">
        <div class="p-3 bg-blue-500/20 rounded-lg">
          <i data-lucide="memory-stick" class="w-6 h-6 text-blue-400"></i>
        </div>
        <div>
          <p class="text-2xl font-bold text-white" x-text="formatMB(summary.total_ram_mb)">0 MB</p>
          <p class="text-sm text-gray-400">Estimated RAM Used</p>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Tabs: Assets vs Models -->
  <div class="border-b border-gray-800">
    <div class="flex gap-4">
      <button
        @click="mainTab = 'assets'"
        class="px-6 py-3 text-sm font-medium transition-colors border-b-2"
        :class="mainTab === 'assets' ? 'border-primary-500 text-primary-400' : 'border-transparent text-gray-400 hover:text-white'"
      >
        <i data-lucide="cpu" class="w-4 h-4 inline mr-2"></i>
        System Assets
        <span class="ml-2 px-2 py-0.5 text-xs rounded bg-gray-700" x-text="systemAssetCount">0</span>
      </button>
      <button
        @click="mainTab = 'models'"
        class="px-6 py-3 text-sm font-medium transition-colors border-b-2"
        :class="mainTab === 'models' ? 'border-primary-500 text-primary-400' : 'border-transparent text-gray-400 hover:text-white'"
      >
        <i data-lucide="audio-waveform" class="w-4 h-4 inline mr-2"></i>
        Voice Models
        <span class="ml-2 px-2 py-0.5 text-xs rounded bg-gray-700" x-text="modelAssetCount">0</span>
      </button>
    </div>
  </div>

  <!-- Loading State -->
  <template x-if="loading">
    <div class="flex items-center justify-center py-12">
      <div class="flex items-center gap-3 text-gray-400">
        <i data-lucide="loader-2" class="w-6 h-6 animate-spin"></i>
        <span>Loading assets...</span>
      </div>
    </div>
  </template>

  <!-- System Assets Tab -->
  <template x-if="!loading && mainTab === 'assets'">
    <div class="space-y-6">
      <!-- System Asset Category Tabs -->
      <div class="flex flex-wrap gap-2">
        <button
          @click="selectedSystemCategory = null"
          class="px-3 py-1.5 text-sm rounded-lg transition-colors"
          :class="selectedSystemCategory === null ? 'bg-primary-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'"
        >
          All
        </button>
        <template x-for="(cat, key) in systemCategories" :key="key">
          <button
            @click="selectedSystemCategory = key"
            class="px-3 py-1.5 text-sm rounded-lg transition-colors"
            :class="selectedSystemCategory === key ? 'bg-primary-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'"
          >
            <span x-text="cat.name"></span>
            <span class="ml-1 px-1.5 py-0.5 text-xs rounded" 
                  :class="cat.running > 0 ? 'bg-green-500/20 text-green-400' : 'bg-gray-700/50'"
                  x-text="cat.running + '/' + cat.count"></span>
          </button>
        </template>
      </div>

      <!-- System Asset Cards -->
      <template x-for="(cat, catKey) in filteredSystemCategories" :key="catKey">
        <div class="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          <!-- Category Header -->
          <div 
            class="p-4 border-b border-gray-800 cursor-pointer hover:bg-gray-800/50 transition-colors flex items-center justify-between"
            @click="toggleCategory(catKey)"
          >
            <div class="flex items-center gap-3">
              <i 
                :data-lucide="getCategoryIcon(catKey)" 
                class="w-5 h-5"
                :class="getCategoryColor(catKey)"
              ></i>
              <div>
                <h2 class="font-semibold text-white" x-text="cat.name"></h2>
                <p class="text-xs text-gray-500">
                  <span x-text="cat.running"></span> running / <span x-text="cat.count"></span> total
                  <span x-show="cat.vram_mb > 0" class="ml-2">
                    • <span x-text="formatMB(cat.vram_mb)"></span> VRAM
                  </span>
                </p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <span 
                x-show="cat.running > 0"
                class="px-2 py-1 text-xs font-medium bg-green-500/20 text-green-400 rounded-full"
              >
                <span x-text="cat.running"></span> active
              </span>
              <i 
                data-lucide="chevron-down" 
                class="w-5 h-5 text-gray-400 transition-transform"
                :class="expandedCategories[catKey] ? 'rotate-180' : ''"
              ></i>
            </div>
          </div>

          <!-- Category Content -->
          <div x-show="expandedCategories[catKey]" x-collapse>
            <div class="p-4 grid gap-4 grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
              <template x-for="asset in cat.assets" :key="asset.id">
                <div class="asset-card bg-gray-800/50 border border-gray-700 rounded-lg p-4"
                     :class="{'ring-2 ring-green-500/50': asset.status === 'running'}">
                  <div class="flex items-start justify-between mb-3">
                    <div class="flex items-center gap-2">
                      <div class="p-2 rounded-lg" :class="asset.resource_type === 'gpu' ? 'bg-purple-500/20' : 'bg-blue-500/20'">
                        <i :data-lucide="getAssetIcon(asset)" class="w-4 h-4" :class="asset.resource_type === 'gpu' ? 'text-purple-400' : 'text-blue-400'"></i>
                      </div>
                      <div>
                        <h3 class="font-medium text-white text-sm" x-text="asset.name"></h3>
                        <p class="text-xs text-gray-500 line-clamp-1" x-text="asset.description"></p>
                      </div>
                    </div>
                    <span class="px-2 py-0.5 text-xs font-medium rounded-full" :class="getStatusClass(asset.status)" x-text="formatStatus(asset.status)"></span>
                  </div>
                  
                  <div class="flex items-center justify-between text-xs text-gray-400 mb-3">
                    <span x-show="asset.estimated_vram_mb > 0">
                      <i data-lucide="cpu" class="w-3 h-3 inline"></i>
                      <span x-text="formatMB(asset.estimated_vram_mb)"></span> VRAM
                    </span>
                    <span x-show="asset.estimated_ram_mb > 0">
                      <i data-lucide="memory-stick" class="w-3 h-3 inline"></i>
                      <span x-text="formatMB(asset.estimated_ram_mb)"></span> RAM
                    </span>
                  </div>
                  
                  <div class="flex gap-2">
                    <template x-if="asset.status === 'stopped' || asset.status === 'on_disk'">
                      <button 
                        @click="startAsset(asset.id)"
                        :disabled="actionLoading === asset.id"
                        class="flex-1 px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 text-white text-xs rounded transition-colors flex items-center justify-center gap-1"
                      >
                        <i data-lucide="play" class="w-3 h-3" x-show="actionLoading !== asset.id"></i>
                        <i data-lucide="loader-2" class="w-3 h-3 animate-spin" x-show="actionLoading === asset.id"></i>
                        Load
                      </button>
                    </template>
                    <template x-if="asset.status === 'running' || asset.status === 'in_use'">
                      <button 
                        @click="stopAsset(asset.id)"
                        :disabled="actionLoading === asset.id"
                        class="flex-1 px-3 py-1.5 bg-red-600 hover:bg-red-700 disabled:bg-gray-700 text-white text-xs rounded transition-colors flex items-center justify-center gap-1"
                      >
                        <i data-lucide="square" class="w-3 h-3" x-show="actionLoading !== asset.id"></i>
                        <i data-lucide="loader-2" class="w-3 h-3 animate-spin" x-show="actionLoading === asset.id"></i>
                        Unload
                      </button>
                    </template>
                  </div>
                </div>
              </template>
            </div>
          </div>
        </div>
      </template>

      <template x-if="Object.keys(filteredSystemCategories).length === 0">
        <div class="text-center py-12 text-gray-400">
          <i data-lucide="inbox" class="w-12 h-12 mx-auto mb-4 opacity-50"></i>
          <p>No system assets found</p>
        </div>
      </template>
    </div>
  </template>

  <!-- Voice Models Tab -->
  <template x-if="!loading && mainTab === 'models'">
    <div class="space-y-6">
      <!-- Model Category Tabs -->
      <div class="flex flex-wrap gap-2">
        <button
          @click="selectedModelCategory = null"
          class="px-3 py-1.5 text-sm rounded-lg transition-colors"
          :class="selectedModelCategory === null ? 'bg-primary-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'"
        >
          All Models
        </button>
        <template x-for="(cat, key) in modelCategories" :key="key">
          <button
            @click="selectedModelCategory = key"
            class="px-3 py-1.5 text-sm rounded-lg transition-colors"
            :class="selectedModelCategory === key ? 'bg-primary-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'"
          >
            <span x-text="cat.name"></span>
            <span class="ml-1 px-1.5 py-0.5 text-xs rounded bg-gray-700/50" x-text="cat.count"></span>
          </button>
        </template>
      </div>

      <!-- Model Cards -->
      <template x-for="(cat, catKey) in filteredModelCategories" :key="catKey">
        <div class="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          <!-- Category Header -->
          <div 
            class="p-4 border-b border-gray-800 cursor-pointer hover:bg-gray-800/50 transition-colors flex items-center justify-between"
            @click="toggleCategory(catKey)"
          >
            <div class="flex items-center gap-3">
              <i 
                :data-lucide="getCategoryIcon(catKey)" 
                class="w-5 h-5"
                :class="getCategoryColor(catKey)"
              ></i>
              <div>
                <h2 class="font-semibold text-white" x-text="cat.name"></h2>
                <p class="text-xs text-gray-500">
                  <span x-text="cat.count"></span> models
                  <span x-show="cat.running > 0" class="ml-2 text-green-400">
                    • <span x-text="cat.running"></span> loaded
                  </span>
                </p>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <span 
                x-show="cat.running > 0"
                class="px-2 py-1 text-xs font-medium bg-green-500/20 text-green-400 rounded-full"
              >
                <span x-text="cat.running"></span> loaded
              </span>
              <i 
                data-lucide="chevron-down" 
                class="w-5 h-5 text-gray-400 transition-transform"
                :class="expandedCategories[catKey] ? 'rotate-180' : ''"
              ></i>
            </div>
          </div>

          <!-- Models Grid -->
          <div x-show="expandedCategories[catKey]" x-collapse>
            <div class="p-4 grid gap-3 grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              <template x-for="asset in cat.assets" :key="asset.id">
                <div class="bg-gray-800/50 border border-gray-700 rounded-lg p-3 hover:border-gray-600 transition-colors"
                     :class="{'ring-1 ring-green-500/50': asset.status === 'running'}">
                  <div class="flex items-start gap-2">
                    <div class="p-1.5 rounded bg-pink-500/20">
                      <i :data-lucide="getAssetIcon(asset)" class="w-3 h-3 text-pink-400"></i>
                    </div>
                    <div class="flex-1 min-w-0">
                      <h3 class="font-medium text-white text-sm truncate" x-text="asset.name" :title="asset.name"></h3>
                      <p class="text-xs text-gray-500" x-show="asset.metrics?.file_size_mb">
                        <span x-text="formatMB(asset.metrics?.file_size_mb)"></span>
                      </p>
                    </div>
                    <span class="px-1.5 py-0.5 text-xs rounded" :class="getStatusClass(asset.status)" x-text="asset.status === 'on_disk' ? 'ready' : asset.status"></span>
                  </div>
                </div>
              </template>
            </div>
          </div>
        </div>
      </template>

      <template x-if="Object.keys(filteredModelCategories).length === 0">
        <div class="text-center py-12 text-gray-400">
          <i data-lucide="audio-waveform" class="w-12 h-12 mx-auto mb-4 opacity-50"></i>
          <p>No voice models found</p>
          <p class="text-sm mt-2">Models will appear here once discovered by the voice engine</p>
        </div>
      </template>
    </div>
  </template>
</div>

<!-- Force Stop Modal -->
<div x-show="showForceModal" x-cloak class="fixed inset-0 z-50 flex items-center justify-center" style="display: none;">
  <div class="absolute inset-0 bg-black/50" @click="showForceModal = false"></div>
  <div class="relative bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-md mx-4">
    <h3 class="text-lg font-semibold text-white mb-2">Force Stop Asset?</h3>
    <p class="text-gray-400 text-sm mb-4" x-text="forceModalMessage"></p>
    <div class="flex gap-3">
      <button @click="showForceModal = false" class="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors">
        Cancel
      </button>
      <button @click="confirmForceStop()" class="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors">
        Force Stop
      </button>
    </div>
  </div>
</div>

<script>
function assetsPage() {
  return {
    loading: true,
    refreshing: false,
    error: null,
    actionLoading: null,
    categories: {},
    summary: { total_running: 0, total_vram_mb: 0, total_ram_mb: 0 },
    mainTab: 'assets', // 'assets' or 'models'
    selectedSystemCategory: null,
    selectedModelCategory: null,
    expandedCategories: {
      core: true,
      tts: true,
      pipelines: true,
      voice_models: true,
      pretrained: true,
      uvr_weights: true,
    },
    pollInterval: null,
    showForceModal: false,
    forceModalMessage: '',
    forceStopAssetId: null,
    
    // System asset categories (core, tts, pipelines)
    get systemCategories() {
      const systemKeys = ['core', 'tts', 'pipelines'];
      const result = {};
      for (const key of systemKeys) {
        if (this.categories[key]) {
          result[key] = this.categories[key];
        }
      }
      return result;
    },
    
    // Model asset categories (voice_models, pretrained, uvr_weights)
    get modelCategories() {
      const modelKeys = ['voice_models', 'pretrained', 'uvr_weights'];
      const result = {};
      for (const key of modelKeys) {
        if (this.categories[key]) {
          result[key] = this.categories[key];
        }
      }
      return result;
    },
    
    get systemAssetCount() {
      return Object.values(this.systemCategories).reduce((sum, cat) => sum + (cat.count || 0), 0);
    },
    
    get modelAssetCount() {
      return Object.values(this.modelCategories).reduce((sum, cat) => sum + (cat.count || 0), 0);
    },
    
    get filteredSystemCategories() {
      if (!this.selectedSystemCategory) {
        return this.systemCategories;
      }
      const result = {};
      if (this.systemCategories[this.selectedSystemCategory]) {
        result[this.selectedSystemCategory] = this.systemCategories[this.selectedSystemCategory];
      }
      return result;
    },
    
    get filteredModelCategories() {
      if (!this.selectedModelCategory) {
        return this.modelCategories;
      }
      const result = {};
      if (this.modelCategories[this.selectedModelCategory]) {
        result[this.selectedModelCategory] = this.modelCategories[this.selectedModelCategory];
      }
      return result;
    },
    
    async init() {
      await this.loadAssets();
      this.startPolling();
      this.$nextTick(() => {
        if (window.lucide) lucide.createIcons();
      });
    },
    
    async loadAssets() {
      try {
        const response = await fetch('{{ route("admin.proxy.assets.byCategory") }}', {
          headers: {
            'Accept': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
          }
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Assets response:', data);
        
        if (data.error) {
          throw new Error(data.error);
        }
        
        if (data.categories) {
          this.categories = data.categories;
        }
        if (data.summary) {
          this.summary = data.summary;
        }
        
        this.error = null;
      } catch (err) {
        console.error('Failed to load assets:', err);
        this.error = err.message || 'Unknown error';
      }
      this.loading = false;
      
      this.$nextTick(() => {
        if (window.lucide) lucide.createIcons();
      });
    },
    
    startPolling() {
      this.pollInterval = setInterval(async () => {
        if (!this.actionLoading) {
          await this.loadAssets();
        }
      }, 3000);
    },
    
    stopPolling() {
      if (this.pollInterval) {
        clearInterval(this.pollInterval);
        this.pollInterval = null;
      }
    },
    
    async refreshAssets() {
      this.refreshing = true;
      await this.loadAssets();
      this.refreshing = false;
    },
    
    toggleCategory(catKey) {
      this.expandedCategories[catKey] = !this.expandedCategories[catKey];
      this.$nextTick(() => {
        if (window.lucide) lucide.createIcons();
      });
    },
    
    async startAsset(assetId) {
      this.actionLoading = assetId;
      try {
        const response = await fetch(`{{ route("admin.proxy.assets.action", ["assetId" => "__assetId__", "action" => "start"]) }}`.replace('__assetId__', assetId), {
          method: 'POST',
          headers: {
            'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content,
            'Accept': 'application/json'
          }
        });
        const result = await response.json();
        
        if (result.success) {
          await this.loadAssets();
        } else {
          this.setAssetError(assetId, result.error);
        }
      } catch (error) {
        console.error('Failed to start asset:', error);
        this.setAssetError(assetId, error.message);
      }
      this.actionLoading = null;
    },
    
    async stopAsset(assetId, force = false) {
      this.actionLoading = assetId;
      try {
        const response = await fetch(`{{ route("admin.proxy.assets.action", ["assetId" => "__assetId__", "action" => "stop"]) }}`.replace('__assetId__', assetId) + `?force=${force}`, {
          method: 'POST',
          headers: {
            'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content,
            'Accept': 'application/json'
          }
        });
        const result = await response.json();
        
        if (result.success) {
          await this.loadAssets();
        } else if (result.error?.includes('in use') && !force) {
          this.forceStopAssetId = assetId;
          this.forceModalMessage = result.error + ' Do you want to force stop it?';
          this.showForceModal = true;
        } else {
          this.setAssetError(assetId, result.error);
        }
      } catch (error) {
        console.error('Failed to stop asset:', error);
        this.setAssetError(assetId, error.message);
      }
      this.actionLoading = null;
    },
    
    async confirmForceStop() {
      this.showForceModal = false;
      if (this.forceStopAssetId) {
        await this.stopAsset(this.forceStopAssetId, true);
        this.forceStopAssetId = null;
      }
    },
    
    setAssetError(assetId, error) {
      for (const cat of Object.values(this.categories)) {
        const asset = cat.assets?.find(a => a.id === assetId);
        if (asset) {
          asset.error = error;
          setTimeout(() => { asset.error = null; }, 5000);
          break;
        }
      }
    },
    
    formatMB(mb) {
      if (!mb || mb === 0) return '0 MB';
      if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
      return Math.round(mb) + ' MB';
    },
    
    formatStatus(status) {
      const map = {
        'running': 'running',
        'stopped': 'stopped',
        'loading': 'loading',
        'unloading': 'unloading',
        'in_use': 'in use',
        'on_disk': 'on disk',
        'error': 'error',
        'unknown': 'unknown'
      };
      return map[status] || status;
    },
    
    getStatusClass(status) {
      const map = {
        'running': 'bg-green-500/20 text-green-400',
        'in_use': 'bg-blue-500/20 text-blue-400',
        'stopped': 'bg-gray-500/20 text-gray-400',
        'on_disk': 'bg-gray-500/20 text-gray-400',
        'loading': 'bg-yellow-500/20 text-yellow-400',
        'unloading': 'bg-yellow-500/20 text-yellow-400',
        'error': 'bg-red-500/20 text-red-400',
        'unknown': 'bg-gray-500/20 text-gray-400',
      };
      return map[status] || 'bg-gray-500/20 text-gray-400';
    },
    
    getAssetIcon(asset) {
      if (asset.type === 'voice_model') return 'audio-waveform';
      if (asset.type === 'pretrained') return 'brain';
      if (asset.type === 'uvr_weight') return 'split';
      if (asset.type === 'service') return 'cloud';
      if (asset.type === 'component') return 'box';
      return 'cpu';
    },
    
    getCategoryIcon(catKey) {
      const map = {
        'core': 'cpu',
        'tts': 'volume-2',
        'pipelines': 'git-branch',
        'voice_models': 'audio-waveform',
        'pretrained': 'brain',
        'uvr_weights': 'split',
      };
      return map[catKey] || 'folder';
    },
    
    getCategoryColor(catKey) {
      const map = {
        'core': 'text-purple-400',
        'tts': 'text-blue-400',
        'pipelines': 'text-green-400',
        'voice_models': 'text-pink-400',
        'pretrained': 'text-yellow-400',
        'uvr_weights': 'text-cyan-400',
      };
      return map[catKey] || 'text-gray-400';
    },
  }
}
</script>
@endsection
