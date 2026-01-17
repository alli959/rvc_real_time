@extends('admin.layout')

@section('title', 'System Logs')

@section('content')
<div class="space-y-6" x-data="logsPage()" x-init="init()">
  <!-- Header -->
  <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
    <div>
      <h1 class="text-2xl font-bold text-white flex items-center gap-3">
        <i data-lucide="scroll-text" class="w-7 h-7"></i>
        System Logs
      </h1>
      <p class="text-gray-400 mt-1">View logs from all services in real-time</p>
    </div>
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 text-sm" x-show="!paused">
        <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
        <span class="text-gray-400">Live (2s)</span>
      </div>
      <div class="flex items-center gap-2 text-sm" x-show="paused">
        <span class="w-2 h-2 rounded-full bg-yellow-500"></span>
        <span class="text-gray-400">Paused</span>
      </div>
      <button @click="refreshServices()" class="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors">
        <i data-lucide="refresh-cw" class="w-4 h-4" :class="{'animate-spin': refreshing}"></i>
        Refresh
      </button>
    </div>
  </div>

  <!-- Service Tabs -->
  <div class="bg-gray-900 border border-gray-800 rounded-lg">
    <div class="border-b border-gray-800">
      <nav class="flex overflow-x-auto" aria-label="Tabs">
        <template x-for="service in services" :key="service.name">
          <button 
            @click="selectService(service.name)"
            :class="selectedService === service.name 
              ? 'border-primary-500 text-primary-400 bg-gray-800' 
              : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-700'"
            class="flex-shrink-0 px-6 py-4 border-b-2 font-medium text-sm transition-colors flex items-center gap-2"
          >
            <span class="w-2 h-2 rounded-full" :class="service.status === 'running' ? 'bg-green-500' : 'bg-gray-500'"></span>
            <span x-text="service.name"></span>
          </button>
        </template>
      </nav>
    </div>

    <!-- Log Panels -->
    <div class="p-6 space-y-6">
      <!-- Log Source Selector -->
      <div class="flex flex-wrap gap-2" x-show="currentSources.length > 0">
        <template x-for="source in currentSources" :key="source.id">
          <button 
            @click="selectSource(source.id)"
            :class="selectedSource === source.id 
              ? 'bg-primary-600 text-white border-primary-500' 
              : 'bg-gray-800 text-gray-300 border-gray-700 hover:bg-gray-700'"
            class="px-3 py-1.5 text-sm rounded-lg border transition-colors flex items-center gap-2"
          >
            <span class="w-1.5 h-1.5 rounded-full" :class="source.type === 'stdout' ? 'bg-blue-400' : 'bg-green-400'"></span>
            <span x-text="source.name"></span>
          </button>
        </template>
      </div>

      <!-- Log Panel Controls -->
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button 
            @click="togglePause()"
            :class="paused ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-gray-700 hover:bg-gray-600'"
            class="px-3 py-1.5 text-sm text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <i :data-lucide="paused ? 'play' : 'pause'" class="w-4 h-4"></i>
            <span x-text="paused ? 'Resume' : 'Pause'"></span>
          </button>
          
          <div class="flex items-center gap-2">
            <input 
              type="text" 
              x-model="filterText" 
              placeholder="Filter logs..." 
              class="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent w-64"
            >
          </div>
        </div>
        
        <div class="flex items-center gap-2">
          <span class="text-sm text-gray-400">
            <span x-text="filteredLines.length"></span> lines
          </span>
          <button 
            @click="downloadLogs()"
            class="px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <i data-lucide="download" class="w-4 h-4"></i>
            Download
          </button>
          <button 
            @click="clearLogs()"
            class="px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <i data-lucide="trash-2" class="w-4 h-4"></i>
            Clear
          </button>
        </div>
      </div>

      <!-- Log Output -->
      <div 
        x-ref="logOutput"
        class="bg-black rounded-lg p-4 h-[500px] overflow-y-auto font-mono text-sm"
        @scroll="handleScroll()"
      >
        <template x-if="loading">
          <div class="flex items-center justify-center h-full">
            <div class="flex items-center gap-3 text-gray-400">
              <i data-lucide="loader-2" class="w-6 h-6 animate-spin"></i>
              <span>Loading logs...</span>
            </div>
          </div>
        </template>
        
        <template x-if="!loading && filteredLines.length === 0">
          <div class="flex items-center justify-center h-full">
            <div class="text-center text-gray-500">
              <i data-lucide="file-text" class="w-12 h-12 mx-auto mb-3 opacity-50"></i>
              <p>No logs to display</p>
              <p class="text-sm mt-1" x-show="filterText">Try clearing the filter</p>
            </div>
          </div>
        </template>
        
        <template x-if="!loading && filteredLines.length > 0">
          <div>
            <template x-for="(line, index) in filteredLines" :key="index">
              <div 
                class="py-0.5 hover:bg-gray-900/50"
                :class="{
                  'text-red-400': line.includes('ERROR') || line.includes('error') || line.includes('Exception'),
                  'text-yellow-400': line.includes('WARN') || line.includes('warning'),
                  'text-green-400': line.includes('INFO') || line.includes('info'),
                  'text-blue-400': line.includes('DEBUG') || line.includes('debug'),
                  'text-gray-300': !line.includes('ERROR') && !line.includes('error') && !line.includes('WARN') && !line.includes('INFO') && !line.includes('DEBUG')
                }"
              >
                <span x-html="highlightSearch(line)"></span>
              </div>
            </template>
          </div>
        </template>
        
        <!-- Auto-scroll indicator -->
        <div 
          x-show="!autoScroll && !paused" 
          @click="scrollToBottom()"
          class="fixed bottom-24 right-12 px-3 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg cursor-pointer transition-colors flex items-center gap-2 shadow-lg"
        >
          <i data-lucide="arrow-down" class="w-4 h-4"></i>
          <span class="text-sm">New logs</span>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Alpine.js Data -->
<script>
function logsPage() {
  return {
    services: [],
    selectedService: null,
    currentSources: [],
    selectedSource: null,
    logLines: [],
    filterText: '',
    paused: false,
    loading: true,
    refreshing: false,
    autoScroll: true,
    ws: null,
    pollInterval: null,
    lastLineCount: 0,
    isTrainingService: false,
    
    async init() {
      await this.loadServices();
      
      // Reinitialize Lucide icons after Alpine renders
      this.$nextTick(() => {
        if (window.lucide) lucide.createIcons();
      });
    },
    
    async loadServices() {
      try {
        // Try to load services from Docker logs endpoint
        const response = await fetch('{{ route("admin.logs.services") }}');
        const data = await response.json();
        
        this.services = data.services || [];
        
        if (this.services.length > 0) {
          this.selectService(this.services[0].name);
        }
      } catch (error) {
        console.error('Failed to load services:', error);
        // Fallback services
        this.services = [
          { name: 'training', container_name: 'training-jobs', status: 'running', log_sources: [
            { id: 'training_all', name: 'All Training Jobs', type: 'training' }
          ]},
          { name: 'api', container_name: 'morphvox-api', status: 'running', log_sources: [
            { id: 'api_stdout', name: 'Container Logs', type: 'stdout' },
            { id: 'api_laravel', name: 'laravel.log', type: 'file' }
          ]},
          { name: 'voice-engine', container_name: 'morphvox-voice-engine', status: 'running', log_sources: [
            { id: 'voice_stdout', name: 'Container Logs', type: 'stdout' }
          ]},
        ];
        if (this.services.length > 0) {
          this.selectService(this.services[0].name);
        }
      }
    },
    
    async refreshServices() {
      this.refreshing = true;
      await this.loadServices();
      this.refreshing = false;
    },
    
    selectService(serviceName) {
      this.selectedService = serviceName;
      this.isTrainingService = serviceName === 'training';
      this.stopPolling();
      
      const service = this.services.find(s => s.name === serviceName);
      this.currentSources = service?.log_sources || [];
      
      if (this.currentSources.length > 0) {
        this.selectSource(this.currentSources[0].id);
      } else {
        this.logLines = ['No log sources available for this service'];
        this.loading = false;
      }
      
      this.$nextTick(() => {
        if (window.lucide) lucide.createIcons();
      });
    },
    
    async selectSource(sourceId) {
      this.selectedSource = sourceId;
      this.loading = true;
      this.logLines = [];
      
      // Load initial logs
      await this.fetchLogs();
      
      this.loading = false;
      this.scrollToBottom();
      
      // Start polling for live updates
      this.startPolling();
    },
    
    async fetchLogs() {
      try {
        let response;
        
        if (this.isTrainingService) {
          // Fetch from trainer API via proxy
          response = await fetch('{{ route("admin.proxy.trainer.logs") }}?lines=500');
        } else {
          // Fetch from Docker logs service (handles all containers)
          response = await fetch(`{{ route("admin.logs.service", ["service" => "__service__"]) }}`.replace('__service__', this.selectedService) + `?lines=500&source=${this.selectedSource}`);
        }
        
        const data = await response.json();
        const newLines = data.lines || [];
        
        // Only update if we have new content
        if (newLines.length !== this.lastLineCount || (newLines.length > 0 && newLines[newLines.length - 1] !== this.logLines[this.logLines.length - 1])) {
          this.logLines = newLines;
          this.lastLineCount = newLines.length;
          
          if (this.autoScroll && !this.paused) {
            this.scrollToBottom();
          }
        }
      } catch (error) {
        console.error('Failed to load logs:', error);
        if (this.logLines.length === 0) {
          this.logLines = ['Error loading logs: ' + error.message];
        }
      }
    },
    
    startPolling() {
      this.stopPolling();
      
      // Poll every 2 seconds for new logs
      this.pollInterval = setInterval(async () => {
        if (!this.paused && this.selectedService && this.selectedSource) {
          await this.fetchLogs();
        }
      }, 2000);
    },
    
    stopPolling() {
      if (this.pollInterval) {
        clearInterval(this.pollInterval);
        this.pollInterval = null;
      }
    },
    
    togglePause() {
      this.paused = !this.paused;
      
      if (!this.paused) {
        this.fetchLogs();
        this.scrollToBottom();
      }
    },
    
    get filteredLines() {
      if (!this.filterText) return this.logLines;
      const filter = this.filterText.toLowerCase();
      return this.logLines.filter(line => line.toLowerCase().includes(filter));
    },
    
    highlightSearch(line) {
      if (!this.filterText) return this.escapeHtml(line);
      
      const escaped = this.escapeHtml(line);
      const searchText = this.escapeHtml(this.filterText);
      const regex = new RegExp(`(${searchText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
      return escaped.replace(regex, '<mark class="bg-yellow-500 text-black px-0.5 rounded">$1</mark>');
    },
    
    escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    },
    
    handleScroll() {
      const el = this.$refs.logOutput;
      if (!el) return;
      
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
      this.autoScroll = atBottom;
    },
    
    scrollToBottom() {
      this.$nextTick(() => {
        const el = this.$refs.logOutput;
        if (el) {
          el.scrollTop = el.scrollHeight;
          this.autoScroll = true;
        }
      });
    },
    
    clearLogs() {
      this.logLines = [];
    },
    
    downloadLogs() {
      const content = this.filteredLines.join('\n');
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${this.selectedService}_${this.selectedSource}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.log`;
      a.click();
      URL.revokeObjectURL(url);
    },
    
    destroy() {
      this.stopPolling();
    }
  };
}
</script>
@endsection
