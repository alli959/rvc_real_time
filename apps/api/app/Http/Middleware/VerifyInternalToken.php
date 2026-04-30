<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class VerifyInternalToken
{
    /**
     * Verify that the request contains a valid internal service token.
     * Used for service-to-service communication within the Docker network.
     */
    public function handle(Request $request, Closure $next): Response
    {
        $expectedToken = config('services.internal_token');

        if (empty($expectedToken)) {
            // If no token configured, reject all requests (fail-closed)
            abort(403, 'Internal token not configured');
        }

        $providedToken = $request->header('X-Internal-Token')
            ?? $request->bearerToken();

        if (!$providedToken || !hash_equals($expectedToken, $providedToken)) {
            abort(403, 'Invalid internal token');
        }

        return $next($request);
    }
}
