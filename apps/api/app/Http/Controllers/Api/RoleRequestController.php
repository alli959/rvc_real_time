<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\RoleRequest;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class RoleRequestController extends Controller
{
    /**
     * Get available roles for request
     */
    public function getAvailableRoles()
    {
        $roles = RoleRequest::getRequestableRoles();
        $user = Auth::user();

        // Mark roles the user already has
        $userRoles = $user->getRoleNames()->toArray();
        
        $rolesWithStatus = [];
        foreach ($roles as $key => $role) {
            $rolesWithStatus[$key] = [
                ...$role,
                'has_role' => in_array($key, $userRoles),
            ];
        }

        return response()->json([
            'roles' => $rolesWithStatus,
        ]);
    }

    /**
     * Get user's role requests
     */
    public function myRequests(Request $request)
    {
        $requests = RoleRequest::forUser($request->user()->id)
            ->with('reviewer:id,name')
            ->orderBy('created_at', 'desc')
            ->get();

        return response()->json([
            'requests' => $requests,
        ]);
    }

    /**
     * Create a new role request
     */
    public function store(Request $request)
    {
        $validated = $request->validate([
            'role' => 'required|string|in:' . implode(',', array_keys(RoleRequest::getRequestableRoles())),
            'message' => 'required|string|max:1000',
        ]);

        $user = $request->user();

        // Check if user already has the role
        if ($user->hasRole($validated['role'])) {
            return response()->json([
                'error' => 'You already have this role',
            ], 422);
        }

        // Check if there's already a pending request for this role
        $existingRequest = RoleRequest::where('user_id', $user->id)
            ->where('requested_role', $validated['role'])
            ->where('status', RoleRequest::STATUS_PENDING)
            ->first();

        if ($existingRequest) {
            return response()->json([
                'error' => 'You already have a pending request for this role',
            ], 422);
        }

        $roleRequest = RoleRequest::create([
            'user_id' => $user->id,
            'requested_role' => $validated['role'],
            'message' => $validated['message'],
            'status' => RoleRequest::STATUS_PENDING,
        ]);

        return response()->json([
            'message' => 'Role request submitted successfully',
            'request' => $roleRequest,
        ], 201);
    }

    /**
     * Cancel a pending request
     */
    public function cancel(Request $request, RoleRequest $roleRequest)
    {
        if ($roleRequest->user_id !== $request->user()->id) {
            return response()->json(['error' => 'Forbidden'], 403);
        }

        if (!$roleRequest->isPending()) {
            return response()->json([
                'error' => 'Only pending requests can be cancelled',
            ], 422);
        }

        $roleRequest->delete();

        return response()->json([
            'message' => 'Request cancelled successfully',
        ]);
    }

    /**
     * Admin: List all role requests
     */
    public function adminIndex(Request $request)
    {
        $query = RoleRequest::with(['user:id,name,email', 'reviewer:id,name']);

        if ($request->has('status')) {
            $query->where('status', $request->status);
        }

        $requests = $query->orderBy('created_at', 'desc')
            ->paginate($request->get('per_page', 20));

        return response()->json($requests);
    }

    /**
     * Admin: Approve a role request
     */
    public function approve(Request $request, RoleRequest $roleRequest)
    {
        $validated = $request->validate([
            'response' => 'nullable|string|max:500',
        ]);

        if (!$roleRequest->isPending()) {
            return response()->json([
                'error' => 'This request has already been reviewed',
            ], 422);
        }

        $roleRequest->approve($request->user(), $validated['response'] ?? null);

        return response()->json([
            'message' => 'Role request approved',
            'request' => $roleRequest->fresh()->load(['user:id,name,email', 'reviewer:id,name']),
        ]);
    }

    /**
     * Admin: Reject a role request
     */
    public function reject(Request $request, RoleRequest $roleRequest)
    {
        $validated = $request->validate([
            'response' => 'nullable|string|max:500',
        ]);

        if (!$roleRequest->isPending()) {
            return response()->json([
                'error' => 'This request has already been reviewed',
            ], 422);
        }

        $roleRequest->reject($request->user(), $validated['response'] ?? null);

        return response()->json([
            'message' => 'Role request rejected',
            'request' => $roleRequest->fresh()->load(['user:id,name,email', 'reviewer:id,name']),
        ]);
    }
}
