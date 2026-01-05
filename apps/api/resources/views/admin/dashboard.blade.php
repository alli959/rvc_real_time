@extends('admin.layout')

@section('title', 'Dashboard')
@section('header', 'Dashboard')

@section('content')
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
  <!-- Users Card -->
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
    <div class="flex items-center gap-4">
      <div class="w-12 h-12 bg-blue-500/10 rounded-lg flex items-center justify-center">
        <i data-lucide="users" class="w-6 h-6 text-blue-500"></i>
      </div>
      <div>
        <p class="text-sm text-gray-400">Total Users</p>
        <p class="text-3xl font-bold">{{ number_format($usersCount) }}</p>
      </div>
    </div>
    <div class="mt-4 pt-4 border-t border-gray-800">
      <a href="{{ route('admin.users.index') }}" class="text-sm text-primary-400 hover:text-primary-300 flex items-center gap-1">
        Manage users
        <i data-lucide="arrow-right" class="w-4 h-4"></i>
      </a>
    </div>
  </div>

  <!-- Voice Models Card -->
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
    <div class="flex items-center gap-4">
      <div class="w-12 h-12 bg-purple-500/10 rounded-lg flex items-center justify-center">
        <i data-lucide="audio-waveform" class="w-6 h-6 text-purple-500"></i>
      </div>
      <div>
        <p class="text-sm text-gray-400">Voice Models</p>
        <p class="text-3xl font-bold">{{ number_format($modelsCount) }}</p>
      </div>
    </div>
    <div class="mt-4 pt-4 border-t border-gray-800">
      <a href="{{ route('admin.models.index') }}" class="text-sm text-primary-400 hover:text-primary-300 flex items-center gap-1">
        Manage models
        <i data-lucide="arrow-right" class="w-4 h-4"></i>
      </a>
    </div>
  </div>

  <!-- Jobs Card -->
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
    <div class="flex items-center gap-4">
      <div class="w-12 h-12 bg-green-500/10 rounded-lg flex items-center justify-center">
        <i data-lucide="zap" class="w-6 h-6 text-green-500"></i>
      </div>
      <div>
        <p class="text-sm text-gray-400">Queue Jobs</p>
        <p class="text-3xl font-bold">{{ number_format($jobsCount) }}</p>
      </div>
    </div>
    <div class="mt-4 pt-4 border-t border-gray-800">
      <span class="text-sm text-gray-500">Processing workload</span>
    </div>
  </div>

  <!-- Usage Events Card -->
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
    <div class="flex items-center gap-4">
      <div class="w-12 h-12 bg-orange-500/10 rounded-lg flex items-center justify-center">
        <i data-lucide="activity" class="w-6 h-6 text-orange-500"></i>
      </div>
      <div>
        <p class="text-sm text-gray-400">Usage Events</p>
        <p class="text-3xl font-bold">{{ number_format($usageEventsCount) }}</p>
      </div>
    </div>
    <div class="mt-4 pt-4 border-t border-gray-800">
      <span class="text-sm text-gray-500">Analytics tracking</span>
    </div>
  </div>
</div>

<!-- Quick Actions -->
<div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
  <h2 class="text-lg font-semibold mb-4">Quick Actions</h2>
  <div class="flex flex-wrap gap-3">
    <a href="{{ route('admin.users.create') }}" class="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm font-medium transition-colors">
      <i data-lucide="user-plus" class="w-4 h-4"></i>
      Add User
    </a>
    <form method="POST" action="{{ route('admin.models.sync') }}" class="inline">
      @csrf
      <button type="submit" class="inline-flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
        <i data-lucide="refresh-cw" class="w-4 h-4"></i>
        Sync Models
      </button>
    </form>
  </div>
</div>
@endsection
