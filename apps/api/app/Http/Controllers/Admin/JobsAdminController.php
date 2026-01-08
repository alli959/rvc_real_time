<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\JobQueue;
use App\Models\User;
use Illuminate\Http\Request;

class JobsAdminController extends Controller
{
    /**
     * Display a listing of all jobs.
     */
    public function index(Request $request)
    {
        $query = JobQueue::with(['user:id,name,email', 'voiceModel:id,uuid,name,slug']);

        // Filter by status
        if ($request->filled('status')) {
            $query->where('status', $request->status);
        }

        // Filter by type
        if ($request->filled('type')) {
            $query->where('type', $request->type);
        }

        // Filter by user
        if ($request->filled('user_id')) {
            $query->where('user_id', $request->user_id);
        }

        // Search by user email or name
        if ($request->filled('search')) {
            $search = $request->search;
            $query->whereHas('user', function ($q) use ($search) {
                $q->where('name', 'like', "%{$search}%")
                  ->orWhere('email', 'like', "%{$search}%");
            });
        }

        $jobs = $query->orderBy('created_at', 'desc')->paginate(20);

        // Get stats
        $stats = [
            'total' => JobQueue::count(),
            'pending' => JobQueue::where('status', JobQueue::STATUS_PENDING)->count(),
            'processing' => JobQueue::where('status', JobQueue::STATUS_PROCESSING)->count(),
            'completed' => JobQueue::where('status', JobQueue::STATUS_COMPLETED)->count(),
            'failed' => JobQueue::where('status', JobQueue::STATUS_FAILED)->count(),
            'today' => JobQueue::whereDate('created_at', today())->count(),
        ];

        // Get unique job types for filter
        $jobTypes = JobQueue::distinct()->pluck('type')->filter()->toArray();

        // Get users for filter
        $users = User::orderBy('name')->get(['id', 'name', 'email']);

        return view('admin.jobs.index', compact('jobs', 'stats', 'jobTypes', 'users'));
    }

    /**
     * Display the specified job.
     */
    public function show(JobQueue $job)
    {
        $job->load(['user', 'voiceModel']);

        return view('admin.jobs.show', compact('job'));
    }
}
