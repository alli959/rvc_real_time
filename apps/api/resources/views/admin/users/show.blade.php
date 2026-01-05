@extends('admin.layout')

@section('title', 'User Details')
@section('header')
<div class="flex items-center gap-4">
  <div class="w-12 h-12 bg-primary-600/20 rounded-full flex items-center justify-center">
    <span class="text-lg font-bold text-primary-400">{{ substr($user->name, 0, 1) }}</span>
  </div>
  <div>
    <h1 class="text-2xl font-bold">{{ $user->name }}</h1>
    <p class="text-gray-400">{{ $user->email }}</p>
  </div>
</div>
@endsection

@section('header-actions')
<div class="flex items-center gap-3">
  <a href="{{ route('admin.users.edit', $user) }}" class="inline-flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
    <i data-lucide="edit" class="w-4 h-4"></i>
    Edit
  </a>
  <form method="POST" action="{{ route('admin.users.destroy', $user) }}" onsubmit="return confirm('Are you sure you want to delete this user?');">
    @csrf
    @method('DELETE')
    <button type="submit" class="inline-flex items-center gap-2 px-4 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg text-sm font-medium transition-colors">
      <i data-lucide="trash-2" class="w-4 h-4"></i>
      Delete
    </button>
  </form>
</div>
@endsection

@section('content')
<!-- User Info Card -->
<div class="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-6">
  <h2 class="text-sm font-medium text-gray-400 mb-4">User Information</h2>
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
    <div>
      <p class="text-xs text-gray-500 mb-1">User ID</p>
      <p class="font-mono text-sm">{{ $user->id }}</p>
    </div>
    <div>
      <p class="text-xs text-gray-500 mb-1">Roles</p>
      <div class="flex flex-wrap gap-2">
        @forelse($user->roles as $role)
          <span class="px-2 py-1 text-xs font-medium rounded-full bg-primary-600/20 text-primary-400">
            {{ ucfirst($role->name) }}
          </span>
        @empty
          <span class="text-gray-500 text-sm">No roles assigned</span>
        @endforelse
      </div>
    </div>
    <div>
      <p class="text-xs text-gray-500 mb-1">Created</p>
      <p class="text-sm">{{ $user->created_at?->format('M d, Y \a\t H:i') }}</p>
    </div>
  </div>
</div>

<!-- Quick Stats -->
<div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center">
        <i data-lucide="audio-waveform" class="w-5 h-5 text-blue-400"></i>
      </div>
      <div>
        <p class="text-2xl font-bold">{{ $user->voiceModels()->count() }}</p>
        <p class="text-xs text-gray-500">Voice Models</p>
      </div>
    </div>
  </div>

  <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 bg-green-500/10 rounded-lg flex items-center justify-center">
        <i data-lucide="clock" class="w-5 h-5 text-green-400"></i>
      </div>
      <div>
        <p class="text-2xl font-bold">{{ $user->last_login_at?->diffForHumans() ?? '—' }}</p>
        <p class="text-xs text-gray-500">Last Login</p>
      </div>
    </div>
  </div>

  <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center">
        <i data-lucide="mail" class="w-5 h-5 text-purple-400"></i>
      </div>
      <div>
        <p class="text-2xl font-bold">{{ $user->email_verified_at ? 'Yes' : 'No' }}</p>
        <p class="text-xs text-gray-500">Email Verified</p>
      </div>
    </div>
  </div>
</div>

<!-- Back Button -->
<a href="{{ route('admin.users.index') }}" class="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
  <i data-lucide="arrow-left" class="w-4 h-4"></i>
  Back to Users
</a>
@endsection
