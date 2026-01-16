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
    <div class="flex items-center gap-2">
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
                <span x-text="line"></span>
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
    
    async init() {
      await this.loadServices();
      
      // Reinitialize Lucide icons after Alpine renders
      this.$nextTick(() => {
        if (window.lucide) lucide.createIcons();
      });
    },
    
    async loadServices() {
      try {
        const response = await fetch('{{ config("services.voice_engine.url") }}/api/v1/admin/logs/services');
        const data = await response.json();
        this.services = data.services || [];
        
        if (this.services.length > 0) {
          this.selectService(this.services[0].name);
        }
      } catch (error) {
        console.error('Failed to load services:', error);
        // Fallback services
        this.services = [
          { name: 'nginx', container_name: 'morphvox-nginx', status: 'unknown' },
          { name: 'api', container_name: 'morphvox-api', status: 'unknown' },
          { name: 'voice-engine', container_name: 'morphvox-voice-engine', status: 'unknown' },
        ];
        this.selectService(this.services[0].name);
      }
    },
    
    async refreshServices() {
      this.refreshing = true;
      await this.loadServices();
      this.refreshing = false;
    },
    
    selectService(serviceName) {
      this.selectedService = serviceName;
      this.disconnectWs();
      
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
      try {
        const response = await fetch(`{{ config("services.voice_engine.url") }}/api/v1/admin/logs/tail/${this.selectedService}/${sourceId}?lines=200`);
        const data = await response.json();
        this.logLines = data.lines || [];
      } catch (error) {
        console.error('Failed to load logs:', error);
        this.logLines = ['Error loading logs: ' + error.message];
      }
      
      this.loading = false;
      this.scrollToBottom();
      
      // Connect WebSocket for live updates
      this.connectWs();
    },
    
    connectWs() {
      this.disconnectWs();
      
      const wsUrl = '{{ config("services.voice_engine.url") }}'.replace('http', 'ws');
      const fullUrl = `${wsUrl}/api/v1/admin/logs/stream/${this.selectedService}/${this.selectedSource}`;
      
      try {
        this.ws = new WebSocket(fullUrl);
        
        this.ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          if (data.type === 'line') {
            this.logLines.push(data.line);
            // Keep only last 5000 lines
            if (this.logLines.length > 5000) {
              this.logLines = this.logLines.slice(-5000);
            }
            if (this.autoScroll && !this.paused) {
              this.scrollToBottom();
            }
          } else if (data.type === 'initial') {
            // Already loaded via REST
          }
        };
        
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
          // Reconnect after delay if not paused
          if (!this.paused) {
            setTimeout(() => {
              if (this.selectedSource && !this.paused) {
                this.connectWs();
              }
            }, 5000);
          }
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
      }
    },
    
    disconnectWs() {
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
    },
    
    togglePause() {
      this.paused = !this.paused;
      
      if (this.ws) {
        this.ws.send(JSON.stringify({ action: this.paused ? 'pause' : 'resume' }));
      }
      
      if (!this.paused) {
        this.scrollToBottom();
      }
    },
    
    get filteredLines() {
      if (!this.filterText) return this.logLines;
      const filter = this.filterText.toLowerCase();
      return this.logLines.filter(line => line.toLowerCase().includes(filter));
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
    }
  };
}
</script>

<!-- Alpine.js CDN -->
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
@endsection
