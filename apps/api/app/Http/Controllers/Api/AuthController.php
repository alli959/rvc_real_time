<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\User;
use App\Models\UserInvitation;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use Illuminate\Validation\ValidationException;

class AuthController extends Controller
{
    /**
     * Register a new user
     */
    public function register(Request $request)
    {
        $validated = $request->validate([
            'name' => 'required|string|max:255',
            'email' => 'required|string|email|max:255|unique:users',
            'password' => 'required|string|min:8|confirmed',
        ]);

        $user = User::create([
            'name' => $validated['name'],
            'email' => $validated['email'],
            'password' => Hash::make($validated['password']),
        ]);

        // Assign default role
        $user->assignRole('user');

        $token = $user->createToken('auth-token')->plainTextToken;

        return response()->json([
            'user' => $this->formatUserWithRoles($user),
            'token' => $token,
        ], 201);
    }

    /**
     * Login user and create token
     */
    public function login(Request $request)
    {
        $request->validate([
            'email' => 'required|email',
            'password' => 'required',
        ]);

        $user = User::where('email', $request->email)->first();

        if (!$user || !Hash::check($request->password, $user->password)) {
            throw ValidationException::withMessages([
                'email' => ['The provided credentials are incorrect.'],
            ]);
        }

        // Revoke old tokens if needed
        // $user->tokens()->delete();

        $token = $user->createToken('auth-token')->plainTextToken;

        return response()->json([
            'user' => $this->formatUserWithRoles($user),
            'token' => $token,
        ]);
    }

    /**
     * Get current user
     */
    public function me(Request $request)
    {
        return response()->json([
            'user' => $this->formatUserWithRoles($request->user()),
        ]);
    }

    /**
     * Format user with roles and permissions as arrays of names
     */
    protected function formatUserWithRoles(User $user): array
    {
        return [
            'id' => $user->id,
            'name' => $user->name,
            'email' => $user->email,
            'avatar' => $user->avatar,
            'email_verified_at' => $user->email_verified_at,
            'created_at' => $user->created_at,
            'updated_at' => $user->updated_at,
            'roles' => $user->getRoleNames()->toArray(),
            'permissions' => $user->getAllPermissions()->pluck('name')->toArray(),
        ];
    }

    /**
     * Logout user (revoke token)
     */
    public function logout(Request $request)
    {
        $request->user()->currentAccessToken()->delete();

        return response()->json([
            'message' => 'Logged out successfully',
        ]);
    }

    /**
     * Refresh token
     */
    public function refresh(Request $request)
    {
        $user = $request->user();
        
        // Delete current token
        $user->currentAccessToken()->delete();
        
        // Create new token
        $token = $user->createToken('auth-token')->plainTextToken;

        return response()->json([
            'token' => $token,
        ]);
    }

    /**
     * Check if invitation token is valid
     */
    public function checkInvitation(string $token)
    {
        $hashedToken = hash('sha256', $token);
        
        $invitation = UserInvitation::where('token', $hashedToken)->first();

        if (!$invitation) {
            return response()->json([
                'valid' => false,
                'message' => 'Invalid invitation link.',
            ], 404);
        }

        if ($invitation->isAccepted()) {
            return response()->json([
                'valid' => false,
                'message' => 'This invitation has already been used.',
            ], 410);
        }

        if ($invitation->isExpired()) {
            return response()->json([
                'valid' => false,
                'message' => 'This invitation has expired.',
            ], 410);
        }

        return response()->json([
            'valid' => true,
            'email' => $invitation->email,
            'roles' => $invitation->roles,
            'invitedBy' => $invitation->inviter?->name,
            'expiresAt' => $invitation->expires_at->toIso8601String(),
        ]);
    }

    /**
     * Register a user with an invitation token
     */
    public function registerWithInvitation(Request $request)
    {
        $validated = $request->validate([
            'token' => 'required|string',
            'name' => 'required|string|max:255',
            'password' => 'required|string|min:8|confirmed',
        ]);

        $hashedToken = hash('sha256', $validated['token']);
        $invitation = UserInvitation::where('token', $hashedToken)->first();

        if (!$invitation || !$invitation->isValid()) {
            throw ValidationException::withMessages([
                'token' => ['Invalid or expired invitation.'],
            ]);
        }

        // Check if user with this email already exists
        if (User::where('email', $invitation->email)->exists()) {
            throw ValidationException::withMessages([
                'email' => ['A user with this email already exists.'],
            ]);
        }

        $user = User::create([
            'name' => $validated['name'],
            'email' => $invitation->email,
            'password' => Hash::make($validated['password']),
        ]);

        // Assign roles from invitation
        $user->syncRoles($invitation->roles);

        // Mark invitation as accepted
        $invitation->update(['accepted_at' => now()]);

        $token = $user->createToken('auth-token')->plainTextToken;

        return response()->json([
            'user' => $this->formatUserWithRoles($user),
            'token' => $token,
        ], 201);
    }
}
