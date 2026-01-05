@extends('admin.layout')

@section('title', 'Users')
@section('header', 'Users')

@section('header-actions')
<div class="flex gap-2">
  <a href="{{ route('admin.users.invite') }}" class="inline-flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
    <i data-lucide="mail" class="w-4 h-4"></i>
    Invite User
  </a>
  <a href="{{ route('admin.users.create') }}" class="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm font-medium transition-colors">
    <i data-lucide="user-plus" class="w-4 h-4"></i>
    New User
  </a>
</div>
@endsection

@section('content')
<!-- Search/Filter -->
<div class="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-6">
  <form method="GET" action="{{ route('admin.users.index') }}" class="flex flex-wrap gap-4">
    <div class="flex-1 min-w-[240px]">
      <div class="relative">
        <div class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">
          <i data-lucide="search" class="w-4 h-4"></i>
        </div>
        <input
          name="search"
          value="{{ request('search') }}"
          placeholder="Search by name or email..."
          class="w-full bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
      </div>
    </div>
    <div class="flex gap-2">
      <button type="submit" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
        Filter
      </button>
      <a href="{{ route('admin.users.index') }}" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
        Reset
      </a>
    </div>
  </form>
</div>

<!-- Users Table -->
<div class="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
  <div class="overflow-x-auto">
    <table class="w-full">
      <thead>
        <tr class="border-b border-gray-800">
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">ID</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Name</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Email</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Roles</th>
          <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Created</th>
          <th class="text-right px-6 py-4 text-sm font-medium text-gray-400">Actions</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-gray-800">
        @forelse($users as $user)
        <tr class="hover:bg-gray-800/50 transition-colors">
          <td class="px-6 py-4 text-sm text-gray-500">{{ $user->id }}</td>
          <td class="px-6 py-4">
            <a href="{{ route('admin.users.show', $user) }}" class="text-sm font-medium text-white hover:text-primary-400">
              {{ $user->name }}
            </a>
          </td>
          <td class="px-6 py-4 text-sm text-gray-400">{{ $user->email }}</td>
          <td class="px-6 py-4">
            <div class="flex flex-wrap gap-1">
              @foreach($user->roles as $role)
                <span class="px-2 py-1 text-xs font-medium rounded-full {{ $role->name === 'admin' ? 'bg-red-500/10 text-red-400' : 'bg-primary-500/10 text-primary-400' }}">
                  {{ $role->name }}
                </span>
              @endforeach
            </div>
          </td>
          <td class="px-6 py-4 text-sm text-gray-500">{{ $user->created_at?->format('M d, Y') }}</td>
          <td class="px-6 py-4">
            <div class="flex items-center justify-end gap-2">
              <a href="{{ route('admin.users.edit', $user) }}" class="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors" title="Edit">
                <i data-lucide="pencil" class="w-4 h-4"></i>
              </a>
              <form method="POST" action="{{ route('admin.users.destroy', $user) }}" onsubmit="return confirm('Are you sure you want to delete this user?');">
                @csrf
                @method('DELETE')
                <button type="submit" class="p-2 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors" title="Delete">
                  <i data-lucide="trash-2" class="w-4 h-4"></i>
                </button>
              </form>
            </div>
          </td>
        </tr>
        @empty
        <tr>
          <td colspan="6" class="px-6 py-12 text-center text-gray-500">
            <i data-lucide="users" class="w-12 h-12 mx-auto mb-4 opacity-50"></i>
            <p>No users found</p>
          </td>
        </tr>
        @endforelse
      </tbody>
    </table>
  </div>

  @if($users->hasPages())
  <div class="px-6 py-4 border-t border-gray-800">
    {{ $users->links() }}
  </div>
  @endif
</div>
@endsection
