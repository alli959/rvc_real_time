@extends('admin.layout')

@section('title', 'Model Access')
@section('header')
<div>
  <p class="text-sm text-gray-400 mb-1">Voice Model Access</p>
  <h1 class="text-2xl font-bold">{{ $voiceModel->name }}</h1>
</div>
@endsection

@section('header-actions')
<a href="{{ route('admin.models.index') }}" class="inline-flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
  <i data-lucide="arrow-left" class="w-4 h-4"></i>
  Back to Models
</a>
@endsection

@section('content')
<div class="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-6">
  <div class="flex items-center gap-4">
    <div class="w-12 h-12 bg-gray-800 rounded-lg flex items-center justify-center">
      <i data-lucide="audio-waveform" class="w-6 h-6 text-gray-500"></i>
    </div>
    <div>
      <p class="font-medium">{{ $voiceModel->name }}</p>
      <p class="text-sm text-gray-500">
        {{ $voiceModel->isSystemModel() ? 'System Model' : 'User Model' }} •
        {{ ucfirst($voiceModel->visibility) }}
      </p>
    </div>
  </div>
  <p class="mt-4 text-sm text-gray-400">
    Use this to grant per-user access for non-public models (or to explicitly allow use).
  </p>
</div>

<form method="POST" action="{{ route('admin.models.access.update', $voiceModel) }}">
  @csrf
  @method('PUT')

  <div class="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
    <div class="overflow-x-auto">
      <table class="w-full">
        <thead>
          <tr class="border-b border-gray-800">
            <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">User</th>
            <th class="text-left px-6 py-4 text-sm font-medium text-gray-400">Role</th>
            <th class="text-center px-6 py-4 text-sm font-medium text-gray-400">Can View</th>
            <th class="text-center px-6 py-4 text-sm font-medium text-gray-400">Can Use</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-800">
          @forelse($users as $user)
            @php $row = $existing[$user->id] ?? ['can_view' => false, 'can_use' => false]; @endphp
            <tr class="hover:bg-gray-800/50 transition-colors">
              <td class="px-6 py-4">
                <p class="text-sm font-medium">{{ $user->name }}</p>
                <p class="text-xs text-gray-500">{{ $user->email }}</p>
              </td>
              <td class="px-6 py-4">
                <span class="px-2 py-1 text-xs font-medium rounded-full bg-gray-800 text-gray-400">
                  {{ $user->roles->first()?->name ?? '—' }}
                </span>
              </td>
              <td class="px-6 py-4 text-center">
                <label class="inline-flex items-center justify-center cursor-pointer">
                  <input
                    type="checkbox"
                    name="access[{{ $user->id }}][can_view]"
                    value="1"
                    class="sr-only peer"
                    @checked($row['can_view'])
                  >
                  <div class="w-10 h-6 bg-gray-700 rounded-full peer-checked:bg-primary-600 peer-focus:ring-2 peer-focus:ring-primary-500/50 transition-colors relative">
                    <div class="absolute left-1 top-1 w-4 h-4 bg-white rounded-full peer-checked:translate-x-4 transition-transform"></div>
                  </div>
                </label>
              </td>
              <td class="px-6 py-4 text-center">
                <label class="inline-flex items-center justify-center cursor-pointer">
                  <input
                    type="checkbox"
                    name="access[{{ $user->id }}][can_use]"
                    value="1"
                    class="sr-only peer"
                    @checked($row['can_use'])
                  >
                  <div class="w-10 h-6 bg-gray-700 rounded-full peer-checked:bg-primary-600 peer-focus:ring-2 peer-focus:ring-primary-500/50 transition-colors relative">
                    <div class="absolute left-1 top-1 w-4 h-4 bg-white rounded-full peer-checked:translate-x-4 transition-transform"></div>
                  </div>
                </label>
              </td>
            </tr>
          @empty
            <tr>
              <td colspan="4" class="px-6 py-12 text-center text-gray-500">
                <i data-lucide="users" class="w-12 h-12 mx-auto mb-4 opacity-50"></i>
                <p>No users to display</p>
              </td>
            </tr>
          @endforelse
        </tbody>
      </table>
    </div>

    @if(count($users) > 0)
    <div class="px-6 py-4 border-t border-gray-800 flex justify-end gap-3">
      <a href="{{ route('admin.models.index') }}" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
        Cancel
      </a>
      <button type="submit" class="px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm font-medium transition-colors">
        Save Access
      </button>
    </div>
    @endif
  </div>
</form>
@endsection
