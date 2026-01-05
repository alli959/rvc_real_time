<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Mail\UserInvitation as UserInvitationMail;
use App\Models\User;
use App\Models\UserInvitation;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Mail;
use Illuminate\Support\Str;
use Spatie\Permission\Models\Role;

class UserController extends Controller
{
    public function index(Request $request)
    {
        $q = User::query();

        if ($request->filled('search')) {
            $search = trim($request->string('search'));
            $q->where(function ($sub) use ($search) {
                $sub->where('name', 'like', "%{$search}%")
                    ->orWhere('email', 'like', "%{$search}%");
            });
        }

        $users = $q->with('roles')
            ->orderBy('created_at', 'desc')
            ->paginate(25)
            ->withQueryString();

        return view('admin.users.index', [
            'users' => $users,
        ]);
    }

    public function create()
    {
        return view('admin.users.create', [
            'roles' => Role::query()->orderBy('name')->get(),
        ]);
    }

    public function store(Request $request)
    {
        $validated = $request->validate([
            'name' => ['required', 'string', 'max:255'],
            'email' => ['required', 'email', 'max:255', 'unique:users,email'],
            'password' => ['required', 'string', 'min:8'],
            'roles' => ['required', 'array', 'min:1'],
            'roles.*' => ['string', 'exists:roles,name'],
        ]);

        $user = User::create([
            'name' => $validated['name'],
            'email' => $validated['email'],
            'password' => Hash::make($validated['password']),
        ]);

        $user->syncRoles($validated['roles']);

        return redirect()->route('admin.users.show', $user)->with('status', 'User created.');
    }

    public function show(User $user)
    {
        $user->load('roles', 'permissions');
        return view('admin.users.show', [
            'user' => $user,
        ]);
    }

    public function edit(User $user)
    {
        $user->load('roles');
        return view('admin.users.edit', [
            'user' => $user,
            'roles' => Role::query()->orderBy('name')->get(),
        ]);
    }

    public function update(Request $request, User $user)
    {
        $validated = $request->validate([
            'name' => ['required', 'string', 'max:255'],
            'email' => ['required', 'email', 'max:255', 'unique:users,email,' . $user->id],
            'password' => ['nullable', 'string', 'min:8'],
            'roles' => ['required', 'array', 'min:1'],
            'roles.*' => ['string', 'exists:roles,name'],
        ]);

        $data = [
            'name' => $validated['name'],
            'email' => $validated['email'],
        ];

        if (!empty($validated['password'])) {
            $data['password'] = Hash::make($validated['password']);
        }

        $user->update($data);
        $user->syncRoles($validated['roles']);

        return redirect()->route('admin.users.show', $user)->with('status', 'User updated.');
    }

    public function destroy(Request $request, User $user)
    {
        // Prevent deleting yourself from the admin UI.
        if (Auth::id() === $user->id) {
            return back()->withErrors(['user' => 'You cannot delete your own account.']);
        }

        $user->delete();

        return redirect()->route('admin.users.index')->with('status', 'User deleted.');
    }

    /**
     * Show invite user form
     */
    public function showInvite()
    {
        return view('admin.users.invite', [
            'roles' => Role::query()->orderBy('name')->get(),
        ]);
    }

    /**
     * Send invitation email to a new user
     */
    public function sendInvite(Request $request)
    {
        $validated = $request->validate([
            'email' => ['required', 'email', 'max:255', 'unique:users,email'],
            'roles' => ['required', 'array', 'min:1'],
            'roles.*' => ['string', 'exists:roles,name'],
            'message' => ['nullable', 'string', 'max:500'],
        ]);

        // Create invitation token
        $token = Str::random(64);

        $invitation = UserInvitation::create([
            'email' => $validated['email'],
            'token' => hash('sha256', $token),
            'roles' => $validated['roles'],
            'message' => $validated['message'] ?? null,
            'invited_by' => Auth::id(),
            'expires_at' => now()->addDays(7),
        ]);

        // Build the invite URL (frontend registration with token)
        $inviteUrl = config('app.frontend_url', 'https://morphvox.net') . '/register?invite=' . $token;

        // Send email using Mailable
        Mail::to($validated['email'])->send(new UserInvitationMail(
            $inviteUrl,
            $validated['message'] ?? null,
            Auth::user()->name
        ));

        return redirect()->route('admin.users.index')->with('status', "Invitation sent to {$validated['email']}");
    }
}
