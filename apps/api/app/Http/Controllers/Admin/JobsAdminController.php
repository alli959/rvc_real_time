<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\JobQueue;
use App\Models\User;
use App\Models\TrainingRun;
use App\Services\TrainerService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;

class JobsAdminController extends Controller
{
    protected TrainerService $trainerService;

    public function __construct(TrainerService $trainerService)
    {
        $this->trainerService = $trainerService;
    }

    /**
     * Display a listing of all jobs.
     */
    public function index(Request $request)
    {
        $validated = $request->validate([
            'status' => 'nullable|string|in:pending,queued,processing,completed,failed,cancelled',
            'type' => 'nullable|string|in:inference,training',
            'per_page' => 'nullable|integer|min:1|max:100',
            'user_id' => 'nullable|integer|exists:users,id',
            'search' => 'nullable|string|max:255',
        ]);

        $query = JobQueue::with(['user:id,name,email', 'voiceModel:id,uuid,name,slug']);

        // Filter by status
        if ($validated['status'] ?? null) {
            $query->where('status', $validated['status']);
        }

        // Filter by type
        if ($validated['type'] ?? null) {
            $query->where('type', $validated['type']);
        }

        // Filter by user
        if ($validated['user_id'] ?? null) {
            $query->where('user_id', $validated['user_id']);
        }

        // Search by user email or name
        if ($validated['search'] ?? null) {
            $search = $validated['search'];
            $query->whereHas('user', function ($q) use ($search) {
                $q->where('name', 'like', "%{$search}%")
                  ->orWhere('email', 'like', "%{$search}%");
            });
        }

        $jobs = $query->orderBy('created_at', 'desc')->paginate($validated['per_page'] ?? 20);

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

    /**
     * Force cancel a stuck job (admin only).
     */
    public function forceCancel(Request $request, JobQueue $job): JsonResponse
    {
        // For training jobs, try to cancel in voice-engine first
        if ($job->type === JobQueue::TYPE_TRAINING) {
            $voiceEngineJobId = $job->parameters['voice_engine_job_id'] ?? null;
            if ($voiceEngineJobId) {
                try {
                    $this->trainerService->cancelTraining($voiceEngineJobId);
                } catch (\Exception $e) {
                    // Log but continue - we still want to mark the job as cancelled in DB
                    \Log::warning('Failed to cancel in voice-engine', [
                        'job_id' => $job->uuid,
                        'voice_engine_job_id' => $voiceEngineJobId,
                        'error' => $e->getMessage(),
                    ]);
                }
            }

            // Also update any stuck TrainingRuns for this model
            TrainingRun::where('voice_model_id', $job->voice_model_id)
                ->whereIn('status', [
                    TrainingRun::STATUS_PENDING, 
                    TrainingRun::STATUS_PREPARING, 
                    TrainingRun::STATUS_TRAINING
                ])
                ->update([
                    'status' => TrainingRun::STATUS_FAILED,
                    'error_message' => 'Force reset by admin',
                ]);
        }

        // Update the job status to cancelled/failed
        if (in_array($job->status, [JobQueue::STATUS_PENDING, JobQueue::STATUS_QUEUED, JobQueue::STATUS_PROCESSING])) {
            $job->update([
                'status' => JobQueue::STATUS_CANCELLED,
                'error_message' => 'Force cancelled by admin',
                'completed_at' => now(),
            ]);
        }

        // Update voice model status if needed - reset to pending so user can retry
        if ($job->voiceModel && in_array($job->voiceModel->status, ['training', 'failed'])) {
            $job->voiceModel->update(['status' => 'pending']);
        }

        return response()->json([
            'success' => true,
            'message' => 'Job reset successfully - model can now be retrained',
        ]);
    }
}
