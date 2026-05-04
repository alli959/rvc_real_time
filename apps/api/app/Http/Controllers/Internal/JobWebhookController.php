<?php

namespace App\Http\Controllers\Internal;

use App\Http\Controllers\Controller;
use App\Models\JobQueue;
use App\Models\UsageEvent;
use Illuminate\Http\Request;

class JobWebhookController extends Controller
{
    /**
     * Receive progress update from voice engine.
     * POST /api/internal/jobs/{job}/progress
     */
    public function progress(Request $request, JobQueue $job)
    {
        // Terminal state guard
        if ($job->isTerminal()) {
            return response()->json(['status' => 'ignored', 'reason' => 'terminal_state']);
        }

        $validated = $request->validate([
            'progress' => 'required|integer|min:0|max:100',
            'message' => 'required|string|max:255',
            'step' => 'required|string|max:50',
            'step_number' => 'required|integer|min:0',
            'total_steps' => 'required|integer|min:1',
        ]);

        // Ordering guard: only update if step_number >= current
        if ($validated['step_number'] < $job->step_number) {
            return response()->json(['status' => 'ignored', 'reason' => 'stale_update']);
        }

        $job->update([
            'status' => JobQueue::STATUS_PROCESSING,
            'progress' => $validated['progress'],
            'progress_message' => $validated['message'],
            'step_number' => $validated['step_number'],
            'total_steps' => $validated['total_steps'],
        ]);

        return response()->json(['status' => 'ok']);
    }

    /**
     * Receive completion notification from voice engine.
     * POST /api/internal/jobs/{job}/complete
     */
    public function complete(Request $request, JobQueue $job)
    {
        // Idempotency: ignore if already completed or failed
        if (in_array($job->status, [JobQueue::STATUS_COMPLETED, JobQueue::STATUS_FAILED])) {
            return response()->json(['status' => 'ignored', 'reason' => 'already_terminal']);
        }

        $validated = $request->validate([
            'status' => 'required|in:completed,failed',
            'output_path' => 'nullable|string|max:500',
            'output_paths' => 'nullable|array',
            'output_paths.*' => 'string|max:500',
            'sample_rate' => 'nullable|integer',
            'duration' => 'nullable|numeric',
            'error' => 'nullable|string|max:1000',
        ]);

        if ($validated['status'] === 'completed') {
            // Cancel race: accept completion even if cancelled
            $params = $job->parameters ?? [];
            if ($validated['sample_rate'] ?? null) {
                $params['sample_rate'] = $validated['sample_rate'];
            }
            if ($validated['duration'] ?? null) {
                $params['duration'] = $validated['duration'];
            }
            if ($validated['output_paths'] ?? null) {
                $params['output_paths'] = $validated['output_paths'];
            }

            $job->update([
                'status' => JobQueue::STATUS_COMPLETED,
                'output_path' => $validated['output_path'],
                'parameters' => $params,
                'progress' => 100,
                'progress_message' => 'Complete',
                'completed_at' => now(),
            ]);

            // Record usage event
            UsageEvent::create([
                'user_id' => $job->user_id,
                'event_type' => 'audio_processing',
                'metadata' => [
                    'job_uuid' => $job->uuid,
                    'type' => $job->type,
                    'duration' => $validated['duration'] ?? null,
                ],
            ]);
        } else {
            $job->update([
                'status' => JobQueue::STATUS_FAILED,
                'error_message' => $validated['error'] ?? 'Unknown error',
                'completed_at' => now(),
            ]);
        }

        return response()->json(['status' => 'ok']);
    }

    /**
     * Return job status for cancellation polling.
     * GET /api/internal/jobs/{job}/status
     */
    public function status(JobQueue $job)
    {
        return response()->json([
            'status' => $job->status,
        ]);
    }
}
