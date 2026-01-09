@extends('admin.layout')

@section('title', 'Jobs Queue')

@section('content')
<div class="space-y-6">
  <!-- Header -->
  <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
    <div>
      <h1 class="text-2xl font-bold text-white flex items-center gap-3">
        <i data-lucide="list-music" class="w-7 h-7"></i>
        Jobs Queue
      </h1>
      <p class="text-gray-400 mt-1">Monitor all generation jobs across the platform</p>
    </div>
  </div>

  <!-- Stats Cards -->
  <div class="grid grid-cols-2 md:grid-cols-6 gap-4">
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-gray-700 rounded-lg">
          <i data-lucide="layers" class="w-5 h-5 text-gray-400"></i>
        </div>
        <div>
          <p class="text-2xl font-bold text-white">{{ number_format($stats['total']) }}</p>
          <p class="text-xs text-gray-500">Total Jobs</p>
        </div>
      </div>
    </div>
    
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-yellow-500/20 rounded-lg">
          <i data-lucide="clock" class="w-5 h-5 text-yellow-400"></i>
        </div>
        <div>
          <p class="text-2xl font-bold text-white">{{ number_format($stats['pending']) }}</p>
          <p class="text-xs text-gray-500">Pending</p>
        </div>
      </div>
    </div>
    
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-blue-500/20 rounded-lg">
          <i data-lucide="loader-2" class="w-5 h-5 text-blue-400"></i>
        </div>
        <div>
          <p class="text-2xl font-bold text-white">{{ number_format($stats['processing']) }}</p>
          <p class="text-xs text-gray-500">Processing</p>
        </div>
      </div>
    </div>
    
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-green-500/20 rounded-lg">
          <i data-lucide="check-circle-2" class="w-5 h-5 text-green-400"></i>
        </div>
        <div>
          <p class="text-2xl font-bold text-white">{{ number_format($stats['completed']) }}</p>
          <p class="text-xs text-gray-500">Completed</p>
        </div>
      </div>
    </div>
    
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-red-500/20 rounded-lg">
          <i data-lucide="x-circle" class="w-5 h-5 text-red-400"></i>
        </div>
        <div>
          <p class="text-2xl font-bold text-white">{{ number_format($stats['failed']) }}</p>
          <p class="text-xs text-gray-500">Failed</p>
        </div>
      </div>
    </div>
    
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-primary-500/20 rounded-lg">
          <i data-lucide="calendar" class="w-5 h-5 text-primary-400"></i>
        </div>
        <div>
          <p class="text-2xl font-bold text-white">{{ number_format($stats['today']) }}</p>
          <p class="text-xs text-gray-500">Today</p>
        </div>
      </div>
    </div>
  </div>

  <!-- Filters -->
  <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
    <form method="GET" class="flex flex-wrap gap-4">
      <div class="flex-1 min-w-[200px]">
        <input type="text" name="search" value="{{ request('search') }}" placeholder="Search by user..." 
               class="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent">
      </div>
      
      <select name="status" class="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500">
        <option value="">All Statuses</option>
        <option value="pending" {{ request('status') == 'pending' ? 'selected' : '' }}>Pending</option>
        <option value="queued" {{ request('status') == 'queued' ? 'selected' : '' }}>Queued</option>
        <option value="processing" {{ request('status') == 'processing' ? 'selected' : '' }}>Processing</option>
        <option value="completed" {{ request('status') == 'completed' ? 'selected' : '' }}>Completed</option>
        <option value="failed" {{ request('status') == 'failed' ? 'selected' : '' }}>Failed</option>
        <option value="cancelled" {{ request('status') == 'cancelled' ? 'selected' : '' }}>Cancelled</option>
      </select>
      
      <select name="type" class="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500">
        <option value="">All Types</option>
        @foreach($jobTypes as $type)
          <option value="{{ $type }}" {{ request('type') == $type ? 'selected' : '' }}>{{ ucwords(str_replace('_', ' ', $type)) }}</option>
        @endforeach
      </select>
      
      <select name="user_id" class="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500">
        <option value="">All Users</option>
        @foreach($users as $user)
          <option value="{{ $user->id }}" {{ request('user_id') == $user->id ? 'selected' : '' }}>{{ $user->name }}</option>
        @endforeach
      </select>
      
      <button type="submit" class="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors">
        Filter
      </button>
      
      @if(request()->hasAny(['search', 'status', 'type', 'user_id']))
        <a href="{{ route('admin.jobs.index') }}" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors">
          Clear
        </a>
      @endif
    </form>
  </div>

  <!-- Jobs Table -->
  <div class="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
    <div class="overflow-x-auto">
      <table class="w-full">
        <thead>
          <tr class="border-b border-gray-800 text-left">
            <th class="px-6 py-4 text-xs font-medium text-gray-400 uppercase tracking-wider w-8"></th>
            <th class="px-6 py-4 text-xs font-medium text-gray-400 uppercase tracking-wider">ID</th>
            <th class="px-6 py-4 text-xs font-medium text-gray-400 uppercase tracking-wider">User</th>
            <th class="px-6 py-4 text-xs font-medium text-gray-400 uppercase tracking-wider">Type</th>
            <th class="px-6 py-4 text-xs font-medium text-gray-400 uppercase tracking-wider">Model</th>
            <th class="px-6 py-4 text-xs font-medium text-gray-400 uppercase tracking-wider">Status</th>
            <th class="px-6 py-4 text-xs font-medium text-gray-400 uppercase tracking-wider">Created</th>
            <th class="px-6 py-4 text-xs font-medium text-gray-400 uppercase tracking-wider">Duration</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-800">
          @forelse($jobs as $job)
            <tr class="hover:bg-gray-800/50 transition-colors cursor-pointer job-row" data-job-id="{{ $job->uuid }}" onclick="toggleJobDetails('{{ $job->uuid }}')">
              <td class="px-6 py-4">
                <i data-lucide="chevron-right" class="w-4 h-4 text-gray-500 transition-transform chevron-{{ $job->uuid }}"></i>
              </td>
              <td class="px-6 py-4">
                <span class="text-sm font-mono text-gray-400">{{ Str::limit($job->uuid, 8, '') }}</span>
              </td>
              <td class="px-6 py-4">
                @if($job->user)
                  <div>
                    <p class="text-sm font-medium text-white">{{ $job->user->name }}</p>
                    <p class="text-xs text-gray-500">{{ $job->user->email }}</p>
                  </div>
                @else
                  <span class="text-gray-500">-</span>
                @endif
              </td>
              <td class="px-6 py-4">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium 
                  @switch($job->type)
                    @case('tts')
                      bg-purple-500/20 text-purple-400
                      @break
                    @case('audio_convert')
                      bg-blue-500/20 text-blue-400
                      @break
                    @case('audio_split')
                      bg-cyan-500/20 text-cyan-400
                      @break
                    @case('audio_swap')
                      bg-pink-500/20 text-pink-400
                      @break
                    @case('inference')
                      bg-green-500/20 text-green-400
                      @break
                    @default
                      bg-gray-500/20 text-gray-400
                  @endswitch
                ">
                  {{ ucwords(str_replace('_', ' ', $job->type ?? 'unknown')) }}
                </span>
              </td>
              <td class="px-6 py-4">
                @if($job->voiceModel)
                  <span class="text-sm text-white">{{ Str::limit($job->voiceModel->name, 20) }}</span>
                @else
                  <span class="text-gray-500">-</span>
                @endif
              </td>
              <td class="px-6 py-4">
                <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium
                  @switch($job->status)
                    @case('pending')
                      bg-gray-500/20 text-gray-400
                      @break
                    @case('queued')
                      bg-yellow-500/20 text-yellow-400
                      @break
                    @case('processing')
                      bg-blue-500/20 text-blue-400
                      @break
                    @case('completed')
                      bg-green-500/20 text-green-400
                      @break
                    @case('failed')
                      bg-red-500/20 text-red-400
                      @break
                    @case('cancelled')
                      bg-gray-500/20 text-gray-400
                      @break
                    @default
                      bg-gray-500/20 text-gray-400
                  @endswitch
                ">
                  @switch($job->status)
                    @case('pending')
                      <i data-lucide="clock" class="w-3 h-3"></i>
                      @break
                    @case('queued')
                      <i data-lucide="list" class="w-3 h-3"></i>
                      @break
                    @case('processing')
                      <i data-lucide="loader-2" class="w-3 h-3 animate-spin"></i>
                      @break
                    @case('completed')
                      <i data-lucide="check" class="w-3 h-3"></i>
                      @break
                    @case('failed')
                      <i data-lucide="x" class="w-3 h-3"></i>
                      @break
                    @case('cancelled')
                      <i data-lucide="ban" class="w-3 h-3"></i>
                      @break
                  @endswitch
                  {{ ucfirst($job->status) }}
                </span>
              </td>
              <td class="px-6 py-4">
                <div>
                  <p class="text-sm text-white">{{ $job->created_at->format('M d, Y') }}</p>
                  <p class="text-xs text-gray-500">{{ $job->created_at->format('H:i:s') }}</p>
                </div>
              </td>
              <td class="px-6 py-4">
                @if($job->started_at && $job->completed_at)
                  <span class="text-sm text-gray-400">{{ $job->started_at->diffForHumans($job->completed_at, true) }}</span>
                @elseif($job->started_at)
                  <span class="text-sm text-blue-400">{{ $job->started_at->diffForHumans(null, true) }} ago</span>
                @else
                  <span class="text-gray-500">-</span>
                @endif
              </td>
            </tr>
            <!-- Expandable Details Row -->
            <tr class="hidden job-details-{{ $job->uuid }} bg-gray-800/30">
              <td colspan="8" class="px-6 py-4">
                <div class="space-y-4">
                  <!-- Error Section (if failed) -->
                  @if($job->status === 'failed')
                    <div class="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                      <h4 class="text-sm font-medium text-red-400 mb-2 flex items-center gap-2">
                        <i data-lucide="alert-circle" class="w-4 h-4"></i>
                        Error Details
                      </h4>
                      <p class="text-sm text-red-300 mb-3">{{ $job->error_message ?? 'Unknown error' }}</p>
                      @if($job->error_details)
                        <div class="bg-gray-900/50 rounded p-3 overflow-x-auto">
                          <pre class="text-xs text-gray-400 whitespace-pre-wrap">{{ json_encode($job->error_details, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) }}</pre>
                        </div>
                      @endif
                    </div>
                  @endif

                  <!-- Parameters Section -->
                  <div class="bg-gray-800/50 rounded-lg p-4">
                    <h4 class="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
                      <i data-lucide="settings" class="w-4 h-4"></i>
                      Generation Parameters
                    </h4>
                    @if($job->parameters && count($job->parameters) > 0)
                      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        @foreach($job->parameters as $key => $value)
                          <div class="bg-gray-900/50 rounded p-3">
                            <p class="text-xs text-gray-500 mb-1">{{ ucwords(str_replace('_', ' ', $key)) }}</p>
                            <p class="text-sm text-white font-mono">
                              @if(is_array($value))
                                {{ json_encode($value) }}
                              @elseif(is_bool($value))
                                {{ $value ? 'true' : 'false' }}
                              @else
                                {{ $value ?? '-' }}
                              @endif
                            </p>
                          </div>
                        @endforeach
                      </div>
                    @else
                      <p class="text-sm text-gray-500">No parameters recorded</p>
                    @endif
                  </div>

                  <!-- Input/Output Transcript Section (for TTS) -->
                  @if($job->type === 'tts' && isset($job->parameters['text']))
                    <div class="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
                      <h4 class="text-sm font-medium text-purple-400 mb-2 flex items-center gap-2">
                        <i data-lucide="message-square" class="w-4 h-4"></i>
                        Input Text
                      </h4>
                      <p class="text-sm text-gray-300 whitespace-pre-wrap bg-gray-900/50 rounded p-3">{{ $job->parameters['text'] ?? 'No text recorded' }}</p>
                    </div>
                  @endif

                  <!-- File Paths Section -->
                  <div class="bg-gray-800/50 rounded-lg p-4">
                    <h4 class="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
                      <i data-lucide="file" class="w-4 h-4"></i>
                      File Information
                    </h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p class="text-xs text-gray-500 mb-1">Input Path</p>
                        <p class="text-sm text-gray-400 font-mono truncate">{{ $job->input_path ?? '-' }}</p>
                      </div>
                      <div>
                        <p class="text-xs text-gray-500 mb-1">Output Path</p>
                        <p class="text-sm text-gray-400 font-mono truncate">{{ $job->output_path ?? '-' }}</p>
                      </div>
                    </div>
                  </div>

                  <!-- Timing Section -->
                  <div class="bg-gray-800/50 rounded-lg p-4">
                    <h4 class="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
                      <i data-lucide="clock" class="w-4 h-4"></i>
                      Timing Information
                    </h4>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p class="text-xs text-gray-500 mb-1">Created At</p>
                        <p class="text-sm text-white">{{ $job->created_at->format('Y-m-d H:i:s') }}</p>
                      </div>
                      <div>
                        <p class="text-xs text-gray-500 mb-1">Started At</p>
                        <p class="text-sm text-white">{{ $job->started_at ? $job->started_at->format('Y-m-d H:i:s') : '-' }}</p>
                      </div>
                      <div>
                        <p class="text-xs text-gray-500 mb-1">Completed At</p>
                        <p class="text-sm text-white">{{ $job->completed_at ? $job->completed_at->format('Y-m-d H:i:s') : '-' }}</p>
                      </div>
                      <div>
                        <p class="text-xs text-gray-500 mb-1">Duration</p>
                        <p class="text-sm text-white">{{ $job->getDuration() ? $job->getDuration() . ' seconds' : '-' }}</p>
                      </div>
                    </div>
                  </div>

                  <!-- Worker Info -->
                  @if($job->worker_id)
                    <div class="text-xs text-gray-500">
                      Worker ID: <span class="font-mono text-gray-400">{{ $job->worker_id }}</span>
                    </div>
                  @endif
                </div>
              </td>
            </tr>
          @empty
            <tr>
              <td colspan="8" class="px-6 py-12 text-center">
                <i data-lucide="inbox" class="w-12 h-12 text-gray-600 mx-auto mb-4"></i>
                <p class="text-gray-400">No jobs found</p>
              </td>
            </tr>
          @endforelse
        </tbody>
      </table>
    </div>
    
    @if($jobs->hasPages())
      <div class="px-6 py-4 border-t border-gray-800">
        {{ $jobs->links() }}
      </div>
    @endif
  </div>
</div>

<script>
  function toggleJobDetails(uuid) {
    const detailsRow = document.querySelector(`.job-details-${uuid}`);
    const chevron = document.querySelector(`.chevron-${uuid}`);
    
    if (detailsRow) {
      detailsRow.classList.toggle('hidden');
      
      if (chevron) {
        if (detailsRow.classList.contains('hidden')) {
          chevron.style.transform = 'rotate(0deg)';
        } else {
          chevron.style.transform = 'rotate(90deg)';
        }
      }
      
      // Re-initialize Lucide icons in the expanded section
      if (typeof lucide !== 'undefined') {
        lucide.createIcons();
      }
    }
  }

  // Add smooth transition for chevron rotation
  document.addEventListener('DOMContentLoaded', function() {
    const chevrons = document.querySelectorAll('[class*="chevron-"]');
    chevrons.forEach(chevron => {
      chevron.style.transition = 'transform 0.2s ease';
    });
  });
</script>
@endsection
