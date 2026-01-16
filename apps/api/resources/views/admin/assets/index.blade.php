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
    <div class="flex items-center gap-2">
      <button @click="refreshAssets()" class="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors">
        <i data-lucide="refresh-cw" class="w-4 h-4" :class="{'animate-spin': refreshing}"></i>
        Refresh
      </button>
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
          <p class="text-2xl font-bold text-white" x-text="runningCount">0</p>
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
          <p class="text-2xl font-bold text-white" x-text="formatMB(totalVramUsed)">0 MB</p>
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
          <p class="text-2xl font-bold text-white" x-text="formatMB(totalRamUsed)">0 MB</p>
          <p class="text-sm text-gray-400">Estimated RAM Used</p>
        </div>
      </div>
    </div>
  </div>

  <!-- Assets Grid -->
  <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
    <template x-for="asset in assets" :key="asset.id">
      <div class="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
        <!-- Asset Header -->
        <div class="p-4 border-b border-gray-800">
          <div class="flex items-start justify-between">
            <div class="flex items-center gap-3">
              <div 
                class="p-3 rounded-lg"
                :class="{
                  'bg-purple-500/20': asset.resource_type === 'gpu',
                  'bg-blue-500/20': asset.resource_type === 'cpu',
                  'bg-green-500/20': asset.resource_type === 'ram'
                }"
              >
                <i 
                  :data-lucide="getAssetIcon(asset)" 
                  class="w-6 h-6"
                  :class="{
                    'text-purple-400': asset.resource_type === 'gpu',
                    'text-blue-400': asset.resource_type === 'cpu',
                    'text-green-400': asset.resource_type === 'ram'
                  }"
                ></i>
              </div>
              <div>
                <h3 class="font-semibold text-white" x-text="asset.name">Asset Name</h3>
                <p class="text-xs text-gray-500" x-text="asset.type">Type</p>
              </div>
            </div>
            
            <!-- Status Badge -->
            <span 
              class="px-2 py-1 text-xs font-medium rounded-full"
              :class="{
                'bg-green-500/20 text-green-400': asset.status === 'running',
                'bg-gray-500/20 text-gray-400': asset.status === 'stopped',
                'bg-yellow-500/20 text-yellow-400': asset.status === 'loading' || asset.status === 'unloading',
                'bg-red-500/20 text-red-400': asset.status === 'error'
              }"
              x-text="asset.status"
            >Status</span>
          </div>
        </div>
        
        <!-- Asset Body -->
        <div class="p-4 space-y-4">
          <p class="text-sm text-gray-400" x-text="asset.description">Description</p>
          
          <!-- Resource Estimates -->
          <div class="grid grid-cols-2 gap-3">
            <div class="bg-gray-800/50 rounded-lg p-3">
              <div class="flex items-center gap-2 mb-1">
                <i data-lucide="memory-stick" class="w-4 h-4 text-gray-500"></i>
                <span class="text-xs text-gray-500">Est. RAM</span>
              </div>
              <p class="text-sm font-medium text-white" x-text="formatMB(asset.estimated_ram_mb)">0 MB</p>
            </div>
            <div class="bg-gray-800/50 rounded-lg p-3">
              <div class="flex items-center gap-2 mb-1">
                <i data-lucide="monitor" class="w-4 h-4 text-gray-500"></i>
                <span class="text-xs text-gray-500">Est. VRAM</span>
              </div>
              <p class="text-sm font-medium text-white" x-text="formatMB(asset.estimated_vram_mb)">0 MB</p>
            </div>
          </div>
          
          <!-- Dependencies -->
          <div x-show="asset.dependencies && asset.dependencies.length > 0">
            <p class="text-xs text-gray-500 mb-2">Dependencies:</p>
            <div class="flex flex-wrap gap-1">
              <template x-for="dep in asset.dependencies" :key="dep">
                <span class="px-2 py-0.5 text-xs bg-gray-800 text-gray-400 rounded" x-text="dep"></span>
              </template>
            </div>
          </div>
          
          <!-- Actual Metrics (when running) -->
          <div x-show="asset.status === 'running' && (asset.metrics.ram_usage_mb || asset.metrics.vram_usage_mb)" class="pt-3 border-t border-gray-800">
            <p class="text-xs text-gray-500 mb-2">Actual Usage:</p>
            <div class="grid grid-cols-2 gap-2 text-sm">
              <div x-show="asset.metrics.ram_usage_mb">
                <span class="text-gray-400">RAM:</span>
                <span class="text-white ml-1" x-text="formatMB(asset.metrics.ram_usage_mb)"></span>
              </div>
              <div x-show="asset.metrics.vram_usage_mb">
                <span class="text-gray-400">VRAM:</span>
                <span class="text-white ml-1" x-text="formatMB(asset.metrics.vram_usage_mb)"></span>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Asset Footer - Actions -->
        <div class="p-4 bg-gray-800/30 border-t border-gray-800">
          <div class="flex items-center gap-2">
            <button 
              x-show="asset.status === 'stopped' || asset.status === 'error'"
              @click="startAsset(asset.id)"
              :disabled="actionLoading === asset.id"
              class="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white text-sm rounded-lg transition-colors"
            >
              <i data-lucide="play" class="w-4 h-4" x-show="actionLoading !== asset.id"></i>
              <i data-lucide="loader-2" class="w-4 h-4 animate-spin" x-show="actionLoading === asset.id"></i>
              <span>Start</span>
            </button>
            
            <button 
              x-show="asset.status === 'running'"
              @click="stopAsset(asset.id)"
              :disabled="actionLoading === asset.id"
              class="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white text-sm rounded-lg transition-colors"
            >
              <i data-lucide="square" class="w-4 h-4" x-show="actionLoading !== asset.id"></i>
              <i data-lucide="loader-2" class="w-4 h-4 animate-spin" x-show="actionLoading === asset.id"></i>
              <span>Stop</span>
            </button>
            
            <button 
              x-show="asset.status === 'running'"
              @click="restartAsset(asset.id)"
              :disabled="actionLoading === asset.id"
              class="flex items-center justify-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-700 disabled:cursor-not-allowed text-white text-sm rounded-lg transition-colors"
            >
              <i data-lucide="refresh-cw" class="w-4 h-4" x-show="actionLoading !== asset.id"></i>
              <i data-lucide="loader-2" class="w-4 h-4 animate-spin" x-show="actionLoading === asset.id"></i>
            </button>
          </div>
          
          <!-- Error Message -->
          <div x-show="asset.error" class="mt-2">
            <p class="text-xs text-red-400" x-text="asset.error"></p>
          </div>
        </div>
      </div>
    </template>
  </div>
  
  <!-- Empty State -->
  <div x-show="assets.length === 0 && !loading" class="bg-gray-900 border border-gray-800 rounded-lg p-12 text-center">
    <i data-lucide="box" class="w-12 h-12 mx-auto mb-4 text-gray-600"></i>
    <p class="text-gray-400">No assets registered</p>
  </div>
  
  <!-- Loading State -->
  <div x-show="loading" class="bg-gray-900 border border-gray-800 rounded-lg p-12 text-center">
    <i data-lucide="loader-2" class="w-8 h-8 mx-auto mb-4 text-gray-600 animate-spin"></i>
    <p class="text-gray-400">Loading assets...</p>
  </div>
</div>

<!-- Force Stop Confirmation Modal -->
<div 
  x-show="showForceModal" 
  x-cloak
  class="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
  @click.self="showForceModal = false"
>
  <div class="bg-gray-900 border border-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
    <div class="flex items-center gap-3 mb-4">
      <div class="p-2 bg-yellow-500/20 rounded-lg">
        <i data-lucide="alert-triangle" class="w-6 h-6 text-yellow-400"></i>
      </div>
      <h3 class="text-lg font-semibold text-white">Confirm Force Stop</h3>
    </div>
    
    <p class="text-gray-400 mb-4" x-text="forceModalMessage">Are you sure you want to force stop this asset?</p>
    
    <div class="flex gap-3">
      <button 
        @click="showForceModal = false"
        class="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
      >
        Cancel
      </button>
      <button 
        @click="confirmForceStop()"
        class="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
      >
        Force Stop
      </button>
    </div>
  </div>
</div>

<script>
function assetsPage() {
  return {
    assets: [],
    loading: true,
    refreshing: false,
    actionLoading: null,
    showForceModal: false,
    forceModalMessage: '',
    forceStopAssetId: null,
    
    async init() {
      await this.loadAssets();
      
      this.$nextTick(() => {
        if (window.lucide) lucide.createIcons();
      });
    },
    
    async loadAssets() {
      this.loading = true;
      try {
        const response = await fetch('{{ config("services.voice_engine.url") }}/api/v1/admin/assets');
        const data = await response.json();
        this.assets = data.assets || [];
      } catch (error) {
        console.error('Failed to load assets:', error);
        // Use fallback data
        this.assets = [
          { id: 'bark_tts', name: 'Bark TTS', type: 'model', resource_type: 'gpu', status: 'stopped', description: 'Suno Bark text-to-speech model', estimated_ram_mb: 2000, estimated_vram_mb: 4000, dependencies: [], metrics: {} },
          { id: 'hubert', name: 'HuBERT Model', type: 'model', resource_type: 'gpu', status: 'stopped', description: 'HuBERT base model for voice feature extraction', estimated_ram_mb: 500, estimated_vram_mb: 1000, dependencies: [], metrics: {} },
          { id: 'rmvpe', name: 'RMVPE F0 Extractor', type: 'model', resource_type: 'gpu', status: 'stopped', description: 'RMVPE model for pitch extraction', estimated_ram_mb: 200, estimated_vram_mb: 500, dependencies: [], metrics: {} },
          { id: 'uvr5', name: 'UVR5 Vocal Separator', type: 'model', resource_type: 'gpu', status: 'stopped', description: 'Ultimate Vocal Remover for vocal/instrumental separation', estimated_ram_mb: 2000, estimated_vram_mb: 2000, dependencies: [], metrics: {} },
        ];
      }
      this.loading = false;
      
      this.$nextTick(() => {
        if (window.lucide) lucide.createIcons();
      });
    },
    
    async refreshAssets() {
      this.refreshing = true;
      await this.loadAssets();
      this.refreshing = false;
    },
    
    async startAsset(assetId) {
      this.actionLoading = assetId;
      try {
        const response = await fetch(`{{ config("services.voice_engine.url") }}/api/v1/admin/assets/${assetId}/start`, {
          method: 'POST'
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
        const response = await fetch(`{{ config("services.voice_engine.url") }}/api/v1/admin/assets/${assetId}/stop?force=${force}`, {
          method: 'POST'
        });
        const result = await response.json();
        
        if (result.success) {
          await this.loadAssets();
        } else if (result.requires_force) {
          // Show confirmation modal
          this.forceStopAssetId = assetId;
          this.forceModalMessage = result.error + (result.jobs ? ` (Jobs: ${result.jobs.join(', ')})` : '');
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
    
    async restartAsset(assetId) {
      this.actionLoading = assetId;
      try {
        const response = await fetch(`{{ config("services.voice_engine.url") }}/api/v1/admin/assets/${assetId}/restart`, {
          method: 'POST'
        });
        const result = await response.json();
        
        if (result.success) {
          await this.loadAssets();
        } else {
          this.setAssetError(assetId, result.error);
        }
      } catch (error) {
        console.error('Failed to restart asset:', error);
        this.setAssetError(assetId, error.message);
      }
      this.actionLoading = null;
    },
    
    setAssetError(assetId, error) {
      const asset = this.assets.find(a => a.id === assetId);
      if (asset) {
        asset.error = error;
        setTimeout(() => { asset.error = null; }, 5000);
      }
    },
    
    getAssetIcon(asset) {
      const icons = {
        'bark_tts': 'mic-2',
        'hubert': 'audio-waveform',
        'rmvpe': 'waves',
        'uvr5': 'split',
        'rvc_inference': 'zap',
        'training_pipeline': 'graduation-cap',
      };
      return icons[asset.id] || 'box';
    },
    
    formatMB(mb) {
      if (!mb) return '0 MB';
      if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
      return Math.round(mb) + ' MB';
    },
    
    get runningCount() {
      return this.assets.filter(a => a.status === 'running').length;
    },
    
    get totalVramUsed() {
      return this.assets
        .filter(a => a.status === 'running')
        .reduce((sum, a) => sum + (a.metrics.vram_usage_mb || a.estimated_vram_mb || 0), 0);
    },
    
    get totalRamUsed() {
      return this.assets
        .filter(a => a.status === 'running')
        .reduce((sum, a) => sum + (a.metrics.ram_usage_mb || a.estimated_ram_mb || 0), 0);
    }
  };
}
</script>

<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
@endsection
