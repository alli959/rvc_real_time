@extends('admin.layout')

@section('title', 'System Metrics')

@section('content')
<div class="space-y-6" x-data="metricsPage()" x-init="init()">
  <!-- Header -->
  <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
    <div>
      <h1 class="text-2xl font-bold text-white flex items-center gap-3">
        <i data-lucide="activity" class="w-7 h-7"></i>
        System Metrics
      </h1>
      <p class="text-gray-400 mt-1">Real-time system resource monitoring</p>
    </div>
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 text-sm">
        <span class="w-2 h-2 rounded-full" :class="connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'"></span>
        <span class="text-gray-400" x-text="connected ? 'Live' : 'Disconnected'"></span>
      </div>
      <div class="text-sm text-gray-500">
        Last update: <span x-text="lastUpdate"></span>
      </div>
    </div>
  </div>

  <!-- Main Metrics Grid -->
  <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
    <!-- CPU Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-3">
          <div class="p-3 bg-blue-500/20 rounded-lg">
            <i data-lucide="cpu" class="w-6 h-6 text-blue-400"></i>
          </div>
          <div>
            <h3 class="text-lg font-semibold text-white">CPU</h3>
            <p class="text-xs text-gray-500">Processor Usage</p>
          </div>
        </div>
      </div>
      
      <div class="space-y-4">
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span class="text-gray-400">Usage</span>
            <span class="text-white font-medium" x-text="metrics.cpu.usage_percent.toFixed(1) + '%'">0%</span>
          </div>
          <div class="h-3 bg-gray-800 rounded-full overflow-hidden">
            <div 
              class="h-full transition-all duration-500 rounded-full"
              :class="metrics.cpu.usage_percent > 80 ? 'bg-red-500' : metrics.cpu.usage_percent > 60 ? 'bg-yellow-500' : 'bg-blue-500'"
              :style="`width: ${metrics.cpu.usage_percent}%`"
            ></div>
          </div>
        </div>
        
        <div class="grid grid-cols-3 gap-4 pt-2 border-t border-gray-800">
          <div class="text-center">
            <p class="text-lg font-semibold text-white" x-text="metrics.cpu.load_avg[0]?.toFixed(2) || '0.00'">0.00</p>
            <p class="text-xs text-gray-500">1m Load</p>
          </div>
          <div class="text-center">
            <p class="text-lg font-semibold text-white" x-text="metrics.cpu.load_avg[1]?.toFixed(2) || '0.00'">0.00</p>
            <p class="text-xs text-gray-500">5m Load</p>
          </div>
          <div class="text-center">
            <p class="text-lg font-semibold text-white" x-text="metrics.cpu.load_avg[2]?.toFixed(2) || '0.00'">0.00</p>
            <p class="text-xs text-gray-500">15m Load</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Memory Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-3">
          <div class="p-3 bg-green-500/20 rounded-lg">
            <i data-lucide="memory-stick" class="w-6 h-6 text-green-400"></i>
          </div>
          <div>
            <h3 class="text-lg font-semibold text-white">Memory</h3>
            <p class="text-xs text-gray-500">RAM Usage</p>
          </div>
        </div>
      </div>
      
      <div class="space-y-4">
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span class="text-gray-400">Used</span>
            <span class="text-white font-medium">
              <span x-text="formatBytes(metrics.memory.used_bytes)">0 GB</span>
              / <span x-text="formatBytes(metrics.memory.total_bytes)">0 GB</span>
            </span>
          </div>
          <div class="h-3 bg-gray-800 rounded-full overflow-hidden">
            <div 
              class="h-full transition-all duration-500 rounded-full"
              :class="metrics.memory.used_percent > 90 ? 'bg-red-500' : metrics.memory.used_percent > 70 ? 'bg-yellow-500' : 'bg-green-500'"
              :style="`width: ${metrics.memory.used_percent}%`"
            ></div>
          </div>
        </div>
        
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span class="text-gray-400">Swap</span>
            <span class="text-white font-medium">
              <span x-text="formatBytes(metrics.memory.swap_used_bytes)">0 GB</span>
              / <span x-text="formatBytes(metrics.memory.swap_total_bytes)">0 GB</span>
            </span>
          </div>
          <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div 
              class="h-full bg-purple-500 transition-all duration-500 rounded-full"
              :style="`width: ${metrics.memory.swap_used_percent || 0}%`"
            ></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Disk Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-3">
          <div class="p-3 bg-yellow-500/20 rounded-lg">
            <i data-lucide="hard-drive" class="w-6 h-6 text-yellow-400"></i>
          </div>
          <div>
            <h3 class="text-lg font-semibold text-white">Disk</h3>
            <p class="text-xs text-gray-500">Storage Usage</p>
          </div>
        </div>
      </div>
      
      <div class="space-y-3">
        <template x-for="disk in metrics.disks" :key="disk.mount">
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-400" x-text="disk.mount">/</span>
              <span class="text-white font-medium">
                <span x-text="formatBytes(disk.used_bytes)">0 GB</span>
                / <span x-text="formatBytes(disk.total_bytes)">0 GB</span>
              </span>
            </div>
            <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div 
                class="h-full transition-all duration-500 rounded-full"
                :class="disk.used_percent > 90 ? 'bg-red-500' : disk.used_percent > 70 ? 'bg-yellow-500' : 'bg-yellow-500'"
                :style="`width: ${disk.used_percent}%`"
              ></div>
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- Network Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-3">
          <div class="p-3 bg-cyan-500/20 rounded-lg">
            <i data-lucide="network" class="w-6 h-6 text-cyan-400"></i>
          </div>
          <div>
            <h3 class="text-lg font-semibold text-white">Network</h3>
            <p class="text-xs text-gray-500">Throughput</p>
          </div>
        </div>
      </div>
      
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-800/50 rounded-lg p-4 text-center">
          <div class="flex items-center justify-center gap-2 text-green-400 mb-2">
            <i data-lucide="arrow-down" class="w-4 h-4"></i>
            <span class="text-xs uppercase tracking-wider">Download</span>
          </div>
          <p class="text-2xl font-bold text-white" x-text="metrics.network.rx_mbps?.toFixed(2) + ' Mbps' || '0 Mbps'">0 Mbps</p>
          <p class="text-xs text-gray-500 mt-1" x-text="formatBytes(metrics.network.rx_bytes_per_sec) + '/s'">0 B/s</p>
        </div>
        <div class="bg-gray-800/50 rounded-lg p-4 text-center">
          <div class="flex items-center justify-center gap-2 text-blue-400 mb-2">
            <i data-lucide="arrow-up" class="w-4 h-4"></i>
            <span class="text-xs uppercase tracking-wider">Upload</span>
          </div>
          <p class="text-2xl font-bold text-white" x-text="metrics.network.tx_mbps?.toFixed(2) + ' Mbps' || '0 Mbps'">0 Mbps</p>
          <p class="text-xs text-gray-500 mt-1" x-text="formatBytes(metrics.network.tx_bytes_per_sec) + '/s'">0 B/s</p>
        </div>
      </div>
    </div>
  </div>

  <!-- GPU Section -->
  <div class="bg-gray-900 border border-gray-800 rounded-lg p-6" x-show="metrics.gpu_available">
    <div class="flex items-center justify-between mb-6">
      <div class="flex items-center gap-3">
        <div class="p-3 bg-purple-500/20 rounded-lg">
          <i data-lucide="monitor" class="w-6 h-6 text-purple-400"></i>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-white">GPU</h3>
          <p class="text-xs text-gray-500">Graphics Processing Units</p>
        </div>
      </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
      <template x-for="gpu in metrics.gpus" :key="gpu.index">
        <div class="bg-gray-800/50 rounded-lg p-4">
          <div class="flex items-center justify-between mb-4">
            <div>
              <p class="text-sm font-medium text-white" x-text="gpu.name">GPU</p>
              <p class="text-xs text-gray-500">GPU <span x-text="gpu.index">0</span></p>
            </div>
            <div class="flex items-center gap-2">
              <i data-lucide="thermometer" class="w-4 h-4 text-gray-400"></i>
              <span class="text-sm" :class="gpu.temperature_c > 80 ? 'text-red-400' : gpu.temperature_c > 60 ? 'text-yellow-400' : 'text-green-400'" x-text="gpu.temperature_c + '°C'">0°C</span>
            </div>
          </div>
          
          <div class="space-y-3">
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="text-gray-400">Utilization</span>
                <span class="text-white" x-text="gpu.utilization_percent + '%'">0%</span>
              </div>
              <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  class="h-full bg-purple-500 transition-all duration-500 rounded-full"
                  :style="`width: ${gpu.utilization_percent}%`"
                ></div>
              </div>
            </div>
            
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="text-gray-400">VRAM</span>
                <span class="text-white">
                  <span x-text="Math.round(gpu.vram_used_mb)">0</span> /
                  <span x-text="Math.round(gpu.vram_total_mb)">0</span> MB
                </span>
              </div>
              <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  class="h-full transition-all duration-500 rounded-full"
                  :class="gpu.vram_used_percent > 90 ? 'bg-red-500' : gpu.vram_used_percent > 70 ? 'bg-yellow-500' : 'bg-purple-500'"
                  :style="`width: ${gpu.vram_used_percent}%`"
                ></div>
              </div>
            </div>
            
            <div class="flex justify-between text-sm pt-2 border-t border-gray-700" x-show="gpu.power_draw_w">
              <span class="text-gray-400">Power Draw</span>
              <span class="text-white" x-text="gpu.power_draw_w + ' W'">0 W</span>
            </div>
          </div>
        </div>
      </template>
    </div>
  </div>
  
  <!-- No GPU Message -->
  <div class="bg-gray-900 border border-gray-800 rounded-lg p-6 text-center" x-show="!metrics.gpu_available">
    <div class="flex flex-col items-center gap-3 text-gray-500">
      <i data-lucide="monitor-off" class="w-12 h-12 opacity-50"></i>
      <p>GPU metrics unavailable</p>
      <p class="text-sm">nvidia-smi not found or no NVIDIA GPU detected</p>
    </div>
  </div>
</div>

<script>
function metricsPage() {
  return {
    metrics: {
      cpu: { usage_percent: 0, load_avg: [0, 0, 0], cores: [] },
      memory: { used_bytes: 0, total_bytes: 0, used_percent: 0, swap_used_bytes: 0, swap_total_bytes: 0, swap_used_percent: 0 },
      disks: [],
      network: { rx_bytes_per_sec: 0, tx_bytes_per_sec: 0, rx_mbps: 0, tx_mbps: 0 },
      gpus: [],
      gpu_available: false,
    },
    connected: false,
    lastUpdate: 'Never',
    ws: null,
    
    init() {
      this.connectWs();
      
      this.$nextTick(() => {
        if (window.lucide) lucide.createIcons();
      });
    },
    
    connectWs() {
      const wsUrl = '{{ config("services.voice_engine.url") }}'.replace('http', 'ws');
      const fullUrl = `${wsUrl}/api/v1/admin/metrics/stream`;
      
      try {
        this.ws = new WebSocket(fullUrl);
        
        this.ws.onopen = () => {
          this.connected = true;
        };
        
        this.ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          if (data.type === 'metrics') {
            this.metrics = data.data;
            this.lastUpdate = new Date().toLocaleTimeString();
          }
        };
        
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.connected = false;
        };
        
        this.ws.onclose = () => {
          this.connected = false;
          // Reconnect after delay
          setTimeout(() => this.connectWs(), 3000);
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        // Fallback to REST polling
        this.pollMetrics();
      }
    },
    
    async pollMetrics() {
      try {
        const response = await fetch('{{ config("services.voice_engine.url") }}/api/v1/admin/metrics');
        this.metrics = await response.json();
        this.lastUpdate = new Date().toLocaleTimeString();
        this.connected = true;
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
        this.connected = false;
      }
      
      // Continue polling
      setTimeout(() => this.pollMetrics(), 2000);
    },
    
    formatBytes(bytes) {
      if (bytes === 0 || bytes === undefined) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
  };
}
</script>

<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
@endsection
