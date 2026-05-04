<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Jobs\ProcessVoiceJob;
use App\Models\JobQueue;
use App\Models\VoiceModel;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;

class JobController extends Controller
{
    /**
     * List user's jobs
     */
    public function index(Request $request)
    {
        $validated = $request->validate([
            'status' => 'nullable|string|in:pending,queued,processing,completed,failed,cancelled',
            'type' => 'nullable|string|in:inference,training',
            'per_page' => 'nullable|integer|min:1|max:100',
        ]);

        $query = JobQueue::forUser($request->user()->id)
            ->with('voiceModel:id,uuid,name,slug');

        // Filter by status
        if ($validated['status'] ?? null) {
            $query->where('status', $validated['status']);
        }

        // Filter by type
        if ($validated['type'] ?? null) {
            $query->where('type', $validated['type']);
        }

        $jobs = $query->orderBy('created_at', 'desc')
            ->paginate($validated['per_page'] ?? 20);

        return response()->json($jobs);
    }

    /**
     * Get single job
     */
    public function show(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id && !$request->user()->is_admin) {
            abort(403, 'Access denied');
        }

        $data = $job->load('voiceModel:id,uuid,name,slug')->toArray();
        $data['output_url'] = null;
        $data['output_urls'] = null;

        if ($job->isCompleted() && $job->output_path) {
            $data['output_url'] = "/api/jobs/{$job->uuid}/stream";

            if ($job->type === JobQueue::TYPE_AUDIO_SPLIT && !empty($job->parameters['output_paths'])) {
                $data['output_urls'] = [
                    'vocals' => "/api/jobs/{$job->uuid}/stream?track=vocals",
                    'instrumental' => "/api/jobs/{$job->uuid}/stream?track=instrumental",
                ];
            }
        }

        return response()->json($data);
    }

    /**
     * Create inference job
     */
    public function createInference(Request $request)
    {
        $validated = $request->validate([
            'model_id' => 'required|exists:voice_models,id',
            'input_type' => 'required|in:upload,url,text',
            'parameters' => 'nullable|array',
            'parameters.f0_up_key' => 'nullable|integer|between:-24,24',
            'parameters.f0_method' => 'nullable|string|in:rmvpe,dio,harvest,crepe',
            'parameters.index_rate' => 'nullable|numeric|between:0,1',
            'parameters.protect' => 'nullable|numeric|between:0,1',
        ]);

        // Check model access
        $model = VoiceModel::findOrFail($validated['model_id']);
        if (!$model->isPublic() && !$model->isOwnedBy($request->user())) {
            abort(403, 'Access denied to this model');
        }

        if (!$model->isReady()) {
            abort(422, 'Model is not ready for inference');
        }

        // Create job
        $job = JobQueue::create([
            'user_id' => $request->user()->id,
            'voice_model_id' => $model->id,
            'type' => JobQueue::TYPE_INFERENCE,
            'status' => JobQueue::STATUS_PENDING,
            'parameters' => $validated['parameters'] ?? [],
        ]);

        // Generate upload URL for input audio
        $inputPath = "users/{$request->user()->id}/jobs/{$job->uuid}/input.wav";
        $job->update(['input_path' => $inputPath]);

        $uploadUrl = Storage::disk('s3')->temporaryUploadUrl($inputPath, now()->addHour());

        return response()->json([
            'job' => $job,
            'upload_url' => $uploadUrl,
        ], 201);
    }

    /**
     * Start job processing (after upload confirmed)
     */
    public function start(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id) {
            abort(403, 'Access denied');
        }

        if (!$job->isPending()) {
            abort(422, 'Job has already been started');
        }

        // Verify input file exists
        if (!Storage::disk('s3')->exists($job->input_path)) {
            abort(422, 'Input file not uploaded');
        }

        // Dispatch to queue
        ProcessVoiceJob::dispatch($job);

        $job->markAsQueued();

        return response()->json([
            'job' => $job->fresh(),
            'message' => 'Job queued for processing',
        ]);
    }

    /**
     * Cancel a job. For queued jobs, marks immediately as cancelled.
     * For in-progress jobs, marks cancelled — voice engine checks status and stops.
     */
    public function cancel(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id) {
            abort(403, 'Access denied');
        }

        if (in_array($job->status, [JobQueue::STATUS_COMPLETED, JobQueue::STATUS_FAILED, JobQueue::STATUS_CANCELLED])) {
            abort(422, 'Cannot cancel a finished job');
        }

        $wasProcessing = $job->status === JobQueue::STATUS_PROCESSING;

        $job->update([
            'status' => JobQueue::STATUS_CANCELLED,
            'completed_at' => now(),
        ]);

        return response()->json([
            'status' => 'cancelled',
            'message' => $wasProcessing
                ? 'Cancellation requested. May take up to 60 seconds to stop.'
                : 'Job cancelled',
        ]);
    }

    /**
     * Get job output (download URL)
     */
    public function getOutput(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id) {
            abort(403, 'Access denied');
        }

        if (!$job->isCompleted()) {
            abort(422, 'Job is not completed');
        }

        if (!$job->output_path || !Storage::disk('s3')->exists($job->output_path)) {
            abort(404, 'Output file not found');
        }

        $downloadUrl = Storage::disk('s3')->temporaryUrl($job->output_path, now()->addHour());

        return response()->json([
            'download_url' => $downloadUrl,
        ]);
    }

    /**
     * Get upload URL for job input
     */
    public function getUploadUrl(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id) {
            abort(403, 'Access denied');
        }

        if (!$job->isPending()) {
            abort(422, 'Can only upload to pending jobs');
        }

        $uploadUrl = Storage::disk('s3')->temporaryUploadUrl($job->input_path, now()->addHour());

        return response()->json([
            'upload_url' => $uploadUrl,
        ]);
    }

    /**
     * Stream job output file from MinIO.
     */
    public function stream(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id && !$request->user()->is_admin) {
            abort(403, 'Access denied');
        }

        if (!$job->isCompleted()) {
            abort(422, 'Job is not completed');
        }

        $track = $request->query('track');
        $path = $job->output_path;

        if ($track && $job->type === JobQueue::TYPE_AUDIO_SPLIT) {
            $outputPaths = $job->parameters['output_paths'] ?? [];
            $pathMap = [];
            foreach ($outputPaths as $p) {
                if (str_contains($p, '_vocals.')) $pathMap['vocals'] = $p;
                if (str_contains($p, '_instrumental.')) $pathMap['instrumental'] = $p;
            }
            $path = $pathMap[$track] ?? $path;
        }

        if (!$path || !Storage::disk('s3')->exists($path)) {
            abort(404, 'Output file not found');
        }

        $size = Storage::disk('s3')->size($path);
        $download = $request->query('download') === '1';
        $filename = basename($path);
        $disposition = $download ? "attachment; filename=\"{$filename}\"" : "inline; filename=\"{$filename}\"";

        $headers = [
            'Content-Type' => 'audio/wav',
            'Content-Length' => $size,
            'Content-Disposition' => $disposition,
            'Accept-Ranges' => 'bytes',
        ];

        $range = $request->header('Range');
        if ($range && preg_match('/bytes=(\d+)-(\d*)/', $range, $matches)) {
            $start = (int) $matches[1];
            $end = $matches[2] !== '' ? (int) $matches[2] : $size - 1;
            $length = $end - $start + 1;

            return response()->stream(function () use ($path, $start, $length) {
                $stream = Storage::disk('s3')->readStream($path);
                fseek($stream, $start);
                $remaining = $length;
                while ($remaining > 0 && !feof($stream)) {
                    $chunk = fread($stream, min(8192, $remaining));
                    echo $chunk;
                    $remaining -= strlen($chunk);
                    flush();
                }
                fclose($stream);
            }, 206, array_merge($headers, [
                'Content-Length' => $length,
                'Content-Range' => "bytes {$start}-{$end}/{$size}",
            ]));
        }

        return response()->stream(function () use ($path) {
            $stream = Storage::disk('s3')->readStream($path);
            while (!feof($stream)) {
                echo fread($stream, 8192);
                flush();
            }
            fclose($stream);
        }, 200, $headers);
    }

    /**
     * Save job output (prevents auto-deletion).
     */
    public function save(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id) {
            abort(403, 'Access denied');
        }
        $job->update(['saved' => true]);
        return response()->json(['status' => 'saved']);
    }

    /**
     * Unsave job output (subject to 24h cleanup).
     */
    public function unsave(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id) {
            abort(403, 'Access denied');
        }
        $job->update(['saved' => false]);
        return response()->json(['status' => 'unsaved']);
    }

    /**
     * Admin: List all jobs across all users
     */
    public function adminIndex(Request $request)
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

        $jobs = $query->orderBy('created_at', 'desc')
            ->paginate($validated['per_page'] ?? 20);

        return response()->json($jobs);
    }

    /**
     * Admin: Get system statistics
     */
    public function systemStats(Request $request)
    {
        $stats = [
            'total_jobs' => JobQueue::count(),
            'jobs_by_status' => [
                'pending' => JobQueue::where('status', JobQueue::STATUS_PENDING)->count(),
                'queued' => JobQueue::where('status', JobQueue::STATUS_QUEUED)->count(),
                'processing' => JobQueue::where('status', JobQueue::STATUS_PROCESSING)->count(),
                'completed' => JobQueue::where('status', JobQueue::STATUS_COMPLETED)->count(),
                'failed' => JobQueue::where('status', JobQueue::STATUS_FAILED)->count(),
                'cancelled' => JobQueue::where('status', JobQueue::STATUS_CANCELLED)->count(),
            ],
            'jobs_by_type' => JobQueue::selectRaw('type, COUNT(*) as count')
                ->groupBy('type')
                ->pluck('count', 'type'),
            'jobs_today' => JobQueue::whereDate('created_at', today())->count(),
            'jobs_this_week' => JobQueue::whereBetween('created_at', [now()->startOfWeek(), now()])->count(),
            'recent_jobs' => JobQueue::with(['user:id,name,email', 'voiceModel:id,name'])
                ->orderBy('created_at', 'desc')
                ->limit(10)
                ->get(),
        ];

        return response()->json($stats);
    }
}
