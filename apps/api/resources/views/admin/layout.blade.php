<!doctype html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="csrf-token" content="{{ csrf_token() }}">
  <title>@yield('title', 'Admin') - MorphVox</title>
  
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            primary: {
              50: '#eff6ff',
              100: '#dbeafe',
              200: '#bfdbfe',
              300: '#93c5fd',
              400: '#60a5fa',
              500: '#3b82f6',
              600: '#2563eb',
              700: '#1d4ed8',
              800: '#1e40af',
              900: '#1e3a8a',
            }
          }
        }
      }
    }
  </script>
  
  <!-- Lucide Icons -->
  <script src="https://unpkg.com/lucide@latest"></script>
  
  <!-- Alpine.js -->
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
  
  <style>
    [x-cloak] { display: none !important; }
  </style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen">
  <div class="flex min-h-screen">
    <!-- Sidebar -->
    <aside class="w-64 bg-gray-900 border-r border-gray-800 flex flex-col">
      <!-- Logo -->
      <div class="p-6 border-b border-gray-800">
        <a href="{{ route('admin.dashboard') }}" class="flex items-center gap-3">
          <div class="w-10 h-10 bg-primary-600 rounded-xl flex items-center justify-center">
            <i data-lucide="mic-2" class="w-6 h-6 text-white"></i>
          </div>
          <span class="text-xl font-bold">MorphVox</span>
        </a>
      </div>

      <!-- Navigation -->
      <nav class="flex-1 p-4 space-y-1">
        <a href="{{ route('admin.dashboard') }}" 
           class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors {{ request()->routeIs('admin.dashboard') ? 'bg-primary-600 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white' }}">
          <i data-lucide="layout-dashboard" class="w-5 h-5"></i>
          <span>Dashboard</span>
        </a>
        
        <a href="{{ route('admin.users.index') }}" 
           class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors {{ request()->routeIs('admin.users.*') ? 'bg-primary-600 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white' }}">
          <i data-lucide="users" class="w-5 h-5"></i>
          <span>Users</span>
        </a>
        
        <a href="{{ route('admin.models.index') }}" 
           class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors {{ request()->routeIs('admin.models.*') ? 'bg-primary-600 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white' }}">
          <i data-lucide="audio-waveform" class="w-5 h-5"></i>
          <span>Voice Models</span>
        </a>
        
        <a href="{{ route('admin.jobs.index') }}" 
           class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors {{ request()->routeIs('admin.jobs.*') ? 'bg-primary-600 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white' }}">
          <i data-lucide="list-music" class="w-5 h-5"></i>
          <span>Jobs Queue</span>
        </a>
        
        <!-- Divider -->
        <div class="my-4 border-t border-gray-800"></div>
        <p class="px-4 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">System</p>
        
        <a href="{{ route('admin.logs.index') }}" 
           class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors {{ request()->routeIs('admin.logs.*') ? 'bg-primary-600 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white' }}">
          <i data-lucide="scroll-text" class="w-5 h-5"></i>
          <span>Logs</span>
        </a>
        
        <a href="{{ route('admin.metrics.index') }}" 
           class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors {{ request()->routeIs('admin.metrics.*') ? 'bg-primary-600 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white' }}">
          <i data-lucide="activity" class="w-5 h-5"></i>
          <span>System Metrics</span>
        </a>
        
        <a href="{{ route('admin.assets.index') }}" 
           class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors {{ request()->routeIs('admin.assets.*') ? 'bg-primary-600 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white' }}">
          <i data-lucide="box" class="w-5 h-5"></i>
          <span>Assets</span>
        </a>
      </nav>

      <!-- User Menu -->
      @auth
      <div class="p-4 border-t border-gray-800">
        <div class="flex items-center gap-3 mb-3">
          <div class="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center">
            <i data-lucide="user" class="w-5 h-5 text-gray-400"></i>
          </div>
          <div class="flex-1 min-w-0">
            <p class="text-sm font-medium truncate">{{ auth()->user()->name }}</p>
            <p class="text-xs text-gray-500 truncate">{{ auth()->user()->email }}</p>
          </div>
        </div>
        <form method="POST" action="{{ route('admin.logout') }}">
          @csrf
          <button type="submit" class="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-gray-300 transition-colors">
            <i data-lucide="log-out" class="w-4 h-4"></i>
            <span>Sign out</span>
          </button>
        </form>
      </div>
      @endauth
    </aside>

    <!-- Main Content -->
    <main class="flex-1 overflow-auto">
      <!-- Header -->
      <header class="sticky top-0 z-10 bg-gray-950/80 backdrop-blur border-b border-gray-800 px-8 py-4">
        <div class="flex items-center justify-between">
          <h1 class="text-2xl font-bold">@yield('header', 'Dashboard')</h1>
          @yield('header-actions')
        </div>
      </header>

      <!-- Page Content -->
      <div class="p-8 relative">
        @if(session('success'))
          <div class="mb-6 p-4 bg-green-500/10 border border-green-500/50 rounded-lg flex items-center gap-3">
            <i data-lucide="check-circle" class="w-5 h-5 text-green-500"></i>
            <p class="text-green-400">{{ session('success') }}</p>
          </div>
        @endif

        @if(session('error'))
          <div class="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex items-center gap-3">
            <i data-lucide="alert-circle" class="w-5 h-5 text-red-500"></i>
            <p class="text-red-400">{{ session('error') }}</p>
          </div>
        @endif

        @yield('content')
      </div>
    </main>
  </div>

  <!-- Initialize Lucide Icons -->
  <script>
    lucide.createIcons();
  </script>
</body>
</html>
