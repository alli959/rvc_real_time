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
        $query = JobQueue::forUser($request->user()->id)
            ->with('voiceModel:id,uuid,name,slug');

        // Filter by status
        if ($request->has('status')) {
            $query->where('status', $request->status);
        }

        // Filter by type
        if ($request->has('type')) {
            $query->where('type', $request->type);
        }

        $jobs = $query->orderBy('created_at', 'desc')
            ->paginate($request->get('per_page', 20));

        return response()->json($jobs);
    }

    /**
     * Get single job
     */
    public function show(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id) {
            abort(403, 'Access denied');
        }

        return response()->json([
            'job' => $job->load('voiceModel:id,uuid,name,slug'),
        ]);
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
     * Cancel a pending/queued job
     */
    public function cancel(Request $request, JobQueue $job)
    {
        if ($job->user_id !== $request->user()->id) {
            abort(403, 'Access denied');
        }

        if ($job->isFinished()) {
            abort(422, 'Cannot cancel a finished job');
        }

        $job->update([
            'status' => JobQueue::STATUS_CANCELLED,
            'completed_at' => now(),
        ]);

        return response()->json([
            'job' => $job->fresh(),
            'message' => 'Job cancelled',
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
}
