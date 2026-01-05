@extends('admin.layout')

@section('title', 'Invite User')
@section('header', 'Invite User')

@section('header-actions')
<a href="{{ route('admin.users.index') }}" class="inline-flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
  <i data-lucide="arrow-left" class="w-4 h-4"></i>
  Back to Users
</a>
@endsection

@section('content')
<div class="max-w-2xl">
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-6">
    <div class="flex items-center gap-4">
      <div class="w-12 h-12 bg-primary-600/20 rounded-full flex items-center justify-center">
        <i data-lucide="mail" class="w-6 h-6 text-primary-400"></i>
      </div>
      <div>
        <p class="font-medium">Send Invitation</p>
        <p class="text-sm text-gray-500">The user will receive an email with a link to register on the frontend.</p>
      </div>
    </div>
  </div>

  <form method="POST" action="{{ route('admin.users.invite.send') }}" class="bg-gray-900 border border-gray-800 rounded-xl p-6">
    @csrf

    <div class="mb-6">
      <label for="email" class="block text-sm font-medium text-gray-400 mb-2">Email Address</label>
      <input
        type="email"
        id="email"
        name="email"
        value="{{ old('email') }}"
        required
        placeholder="user@example.com"
        class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
      />
      @error('email')
        <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
      @enderror
    </div>

    <div class="mb-6">
      <label class="block text-sm font-medium text-gray-400 mb-2">Assign Roles</label>
      <p class="text-xs text-gray-500 mb-3">Select which roles the user will have after they register.</p>
      <div class="space-y-2 bg-gray-800 border border-gray-700 rounded-lg p-3 max-h-48 overflow-y-auto">
        @foreach($roles as $role)
          <label class="flex items-center gap-3 cursor-pointer hover:bg-gray-700/50 p-2 rounded-lg transition-colors">
            <input
              type="checkbox"
              name="roles[]"
              value="{{ $role->name }}"
              @checked(in_array($role->name, old('roles', ['user'])))
              class="w-4 h-4 rounded bg-gray-700 border-gray-600 text-primary-600 focus:ring-primary-500 focus:ring-offset-gray-800"
            />
            <div>
              <span class="text-sm font-medium">{{ ucfirst($role->name) }}</span>
              @if($role->name === 'admin')
                <span class="ml-2 text-xs text-yellow-400">Full access</span>
              @elseif($role->name === 'premium')
                <span class="ml-2 text-xs text-purple-400">Can upload models</span>
              @elseif($role->name === 'creator')
                <span class="ml-2 text-xs text-blue-400">Can train models</span>
              @endif
            </div>
          </label>
        @endforeach
      </div>
      @error('roles')
        <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
      @enderror
    </div>

    <div class="mb-6">
      <label for="message" class="block text-sm font-medium text-gray-400 mb-2">Personal Message (optional)</label>
      <textarea
        id="message"
        name="message"
        rows="3"
        placeholder="Add a personal note to include in the invitation email..."
        class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
      >{{ old('message') }}</textarea>
      @error('message')
        <p class="mt-1 text-sm text-red-400">{{ $message }}</p>
      @enderror
    </div>

    <div class="bg-gray-800/50 rounded-lg p-4 mb-6">
      <div class="flex items-start gap-3">
        <i data-lucide="info" class="w-5 h-5 text-blue-400 mt-0.5"></i>
        <div class="text-sm text-gray-400">
          <p class="font-medium text-gray-300 mb-1">How it works:</p>
          <ul class="list-disc list-inside space-y-1">
            <li>The user receives an email with a unique registration link</li>
            <li>The link expires in 7 days</li>
            <li>They set their name and password during registration</li>
            <li>The selected roles are automatically assigned</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-800">
      <a href="{{ route('admin.users.index') }}" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors">
        Cancel
      </a>
      <button type="submit" class="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm font-medium transition-colors">
        <i data-lucide="send" class="w-4 h-4"></i>
        Send Invitation
      </button>
    </div>
  </form>
</div>
@endsection
