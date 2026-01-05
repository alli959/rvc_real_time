@extends('admin.auth.layout')

@section('title', 'Admin Login')

@section('content')
<div class="text-center mb-8">
  <a href="/" class="inline-flex items-center gap-3 mb-6">
    <div class="w-12 h-12 bg-primary-600 rounded-xl flex items-center justify-center">
      <i data-lucide="mic-2" class="w-7 h-7 text-white"></i>
    </div>
    <span class="text-2xl font-bold">MorphVox</span>
  </a>
  <h1 class="text-3xl font-bold mb-2">Admin Login</h1>
  <p class="text-gray-400">Sign in to manage MorphVox</p>
</div>

<div class="bg-gray-900 border border-gray-800 rounded-2xl p-8">
  @if ($errors->any())
    <div class="mb-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex items-center gap-3">
      <i data-lucide="alert-circle" class="w-5 h-5 text-red-500 flex-shrink-0"></i>
      <p class="text-red-400 text-sm">{{ $errors->first() }}</p>
    </div>
  @endif

  <form method="POST" action="{{ route('admin.login.post') }}" class="space-y-6">
    @csrf

    <div>
      <label for="email" class="block text-sm font-medium text-gray-300 mb-2">Email</label>
      <div class="relative">
        <div class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">
          <i data-lucide="mail" class="w-5 h-5"></i>
        </div>
        <input
          id="email"
          name="email"
          type="email"
          value="{{ old('email') }}"
          required
          autocomplete="email"
          placeholder="admin@morphvox.net"
          class="w-full bg-gray-800/50 border border-gray-700 rounded-lg pl-10 pr-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-colors"
        />
      </div>
    </div>

    <div>
      <label for="password" class="block text-sm font-medium text-gray-300 mb-2">Password</label>
      <div class="relative">
        <div class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">
          <i data-lucide="lock" class="w-5 h-5"></i>
        </div>
        <input
          id="password"
          name="password"
          type="password"
          required
          autocomplete="current-password"
          placeholder="••••••••"
          class="w-full bg-gray-800/50 border border-gray-700 rounded-lg pl-10 pr-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-colors"
        />
      </div>
    </div>

    <div class="flex items-center justify-between">
      <label class="flex items-center gap-2 cursor-pointer">
        <input 
          type="checkbox" 
          name="remember" 
          value="1"
          class="w-4 h-4 rounded border-gray-600 bg-gray-800 text-primary-600 focus:ring-primary-500 focus:ring-offset-gray-900"
        />
        <span class="text-sm text-gray-400">Remember me</span>
      </label>
    </div>

    <button
      type="submit"
      class="w-full bg-primary-600 hover:bg-primary-700 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
    >
      <i data-lucide="log-in" class="w-5 h-5"></i>
      <span>Sign in</span>
    </button>
  </form>
</div>

<p class="text-center text-sm text-gray-500 mt-6">
  <a href="/" class="text-primary-400 hover:text-primary-300">← Back to MorphVox</a>
</p>
@endsection
