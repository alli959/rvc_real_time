@extends('admin.layout')

@section('title', 'Edit User')
@section('header', 'Edit User')

@section('header-actions')
<a href="{{ route('admin.users.show', $user) }}" class="inline-flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
  <i data-lucide="arrow-left" class="w-4 h-4"></i>
  Back to User
</a>
@endsection

@section('content')
<div class="max-w-2xl">
  <!-- User Info Header -->
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-6">
    <div class="flex items-center gap-4">
      <div class="w-12 h-12 bg-primary-600/20 rounded-full flex items-center justify-center">
        <span class="text-lg font-bold text-primary-400">{{ substr($user->name, 0, 1) }}</span>
      </div>
      <div>
        <p class="font-medium">{{ $user->name }}</p>
        <p class="text-sm text-gray-500">{{ $user->email }}</p>
      </div>
    </div>
  </div>

  <form method="POST" action="{{ route('admin.users.update', $user) }}" class="bg-gray-900 border border-gray-800 rounded-xl p-6">
    @csrf
    @method('PUT')

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
      <div>
        <label for="name" class="block text-sm font-medium text-gray-400 mb-2">Name</label>
        <input
          type="text"
          id="name"
          name="name"
          value="{{ old('name', $user->name) }}"
          required
          class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
        @error('name')
          <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
        @enderror
      </div>

      <div>
        <label for="email" class="block text-sm font-medium text-gray-400 mb-2">Email</label>
        <input
          type="email"
          id="email"
          name="email"
          value="{{ old('email', $user->email) }}"
          required
          class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
        @error('email')
          <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
        @enderror
      </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
      <div>
        <label for="password" class="block text-sm font-medium text-gray-400 mb-2">New Password</label>
        <input
          type="password"
          id="password"
          name="password"
          class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
        <p class="mt-1 text-xs text-gray-500">Leave blank to keep current password.</p>
        @error('password')
          <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
        @enderror
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-400 mb-2">Roles</label>
        @php $currentRoles = $user->roles->pluck('name')->toArray(); @endphp
        <div class="space-y-2 bg-gray-800 border border-gray-700 rounded-lg p-3 max-h-48 overflow-y-auto">
          @foreach($roles as $role)
            <label class="flex items-center gap-3 cursor-pointer hover:bg-gray-700/50 p-2 rounded-lg transition-colors">
              <input
                type="checkbox"
                name="roles[]"
                value="{{ $role->name }}"
                @checked(in_array($role->name, old('roles', $currentRoles)))
                class="w-4 h-4 rounded bg-gray-700 border-gray-600 text-primary-600 focus:ring-primary-500 focus:ring-offset-gray-800"
              />
              <div>
                <span class="text-sm font-medium">{{ ucfirst($role->name) }}</span>
                @if($role->name === 'admin')
                  <span class="ml-2 text-xs text-yellow-400">Full access</span>
                @endif
              </div>
            </label>
          @endforeach
        </div>
        @error('roles')
          <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
        @enderror
        @error('roles.*')
          <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
        @enderror
      </div>
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-800">
      <a href="{{ route('admin.users.show', $user) }}" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
        Cancel
      </a>
      <button type="submit" class="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm font-medium transition-colors">
        <i data-lucide="save" class="w-4 h-4"></i>
        Save Changes
      </button>
    </div>
  </form>
</div>
@endsection
