<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Str;
use Laravel\Socialite\Facades\Socialite;
use Laravel\Socialite\Two\AbstractProvider;

class OAuthController extends Controller
{
    /**
     * Redirect to OAuth provider
     */
    public function redirect(string $provider)
    {
        $this->validateProvider($provider);

        // For SPA, we return the redirect URL instead of redirecting
        /** @var AbstractProvider $driver */
        $driver = Socialite::driver($provider);
        $url = $driver
            ->stateless()
            ->redirect()
            ->getTargetUrl();

        return response()->json(['url' => $url]);
    }

    /**
     * Handle OAuth callback - redirects to frontend with token
     */
    public function callback(string $provider, Request $request)
    {
        $this->validateProvider($provider);
        
        $frontendUrl = config('app.frontend_url', 'https://morphvox.net');

        try {
            // Get the code from the request
            $code = $request->input('code');
            
            if (!$code) {
                return redirect("{$frontendUrl}/login?error=no_code");
            }

            // Get user from provider
            /** @var AbstractProvider $driver */
            $driver = Socialite::driver($provider);
            $socialUser = $driver
                ->stateless()
                ->user();

            // Find or create user
            $user = $this->findOrCreateUser($socialUser, $provider);

            // Create token
            $token = $user->createToken('auth-token')->plainTextToken;

            // Redirect to frontend with token
            return redirect("{$frontendUrl}/auth/callback?token={$token}&provider={$provider}");
        } catch (\Exception $e) {
            return redirect("{$frontendUrl}/login?error=" . urlencode($e->getMessage()));
        }
    }

    /**
     * Handle callback with code from frontend (for SPA flow)
     */
    public function handleCode(string $provider, Request $request)
    {
        $this->validateProvider($provider);

        $request->validate([
            'code' => 'required|string',
        ]);

        try {
            // Exchange code for user
            /** @var AbstractProvider $driver */
            $driver = Socialite::driver($provider);
            $socialUser = $driver
                ->stateless()
                ->user();

            // Find or create user
            $user = $this->findOrCreateUser($socialUser, $provider);

            // Create token
            $token = $user->createToken('auth-token')->plainTextToken;

            return response()->json([
                'user' => $this->formatUserWithRoles($user),
                'token' => $token,
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Authentication failed',
                'message' => $e->getMessage(),
            ], 401);
        }
    }

    /**
     * Find or create user from OAuth provider data
     */
    protected function findOrCreateUser($socialUser, string $provider): User
    {
        // First, check if user exists with this email
        $user = User::where('email', $socialUser->getEmail())->first();

        if ($user) {
            // Update OAuth info if user exists
            $user->update([
                "{$provider}_id" => $socialUser->getId(),
                'avatar' => $socialUser->getAvatar() ?? $user->avatar,
            ]);

            return $user;
        }

        // Create new user
        $user = User::create([
            'name' => $socialUser->getName() ?? $socialUser->getNickname() ?? 'User',
            'email' => $socialUser->getEmail(),
            'password' => Hash::make(Str::random(32)), // Random password since they'll use OAuth
            "{$provider}_id" => $socialUser->getId(),
            'avatar' => $socialUser->getAvatar(),
            'email_verified_at' => now(), // OAuth emails are verified
        ]);

        // Assign default role
        $user->assignRole('user');

        return $user;
    }

    /**
     * Format user with roles and permissions
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
     * Validate provider
     */
    protected function validateProvider(string $provider): void
    {
        if (!in_array($provider, ['google', 'github'])) {
            abort(404, 'Provider not supported');
        }
    }
}
