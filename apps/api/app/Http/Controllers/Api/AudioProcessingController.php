<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\VoiceModel;
use App\Models\UsageEvent;
use App\Models\JobQueue;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Str;

/**
 * Controller for Audio Processing functionality.
 * 
 * Handles audio processing operations including:
 * - Voice conversion (convert mode)
 * - Vocal/instrumental separation (split mode)
 * - Voice swap (split + convert + merge)
 */
class AudioProcessingController extends Controller
{
    /**
     * Process audio with various modes
     * 
     * Modes:
     * - convert: Apply voice conversion to audio
     * - split: Separate vocals from instrumentals using UVR5
     * - swap: Separate vocals, convert them, and merge back with instrumental
     */
    public function process(Request $request)
    {
        $user = $request->user();

        $validated = $request->validate([
            'audio' => 'required|string', // Base64 encoded audio
            'sample_rate' => 'nullable|integer|min:8000|max:96000',
            'mode' => 'required|in:convert,split,swap',
            'model_id' => 'nullable|integer|exists:voice_models,id',
            'checkpoint' => 'nullable|string|max:255', // Optional checkpoint filename for model versioning
            'f0_up_key' => 'nullable|integer|min:-24|max:24',
            'index_rate' => 'nullable|numeric|min:0|max:1',
            'pitch_shift_all' => 'nullable|integer|min:-24|max:24',
            'instrumental_pitch' => 'nullable|integer|min:-24|max:24',
            'extract_all_vocals' => 'nullable|boolean',
            // Multi-voice swap parameters
            'voice_count' => 'nullable|integer|min:1|max:4',
            'voice_configs' => 'nullable|array|max:4',
            'voice_configs.*.model_id' => 'required_with:voice_configs|integer|exists:voice_models,id',
            'voice_configs.*.f0_up_key' => 'nullable|integer|min:-24|max:24',
            'voice_configs.*.extraction_mode' => 'nullable|string|in:main,all',
        ]);

        $mode = $validated['mode'];
        $sampleRate = $validated['sample_rate'] ?? 44100;
        $voiceCount = $validated['voice_count'] ?? 1;
        
        // Get voice model if specified (for single voice or first voice)
        $voiceModel = null;
        $modelPath = null;
        $indexPath = null;
        
        if (!empty($validated['model_id']) && in_array($mode, ['convert', 'swap'])) {
            $voiceModel = VoiceModel::find($validated['model_id']);
            
            if ($voiceModel) {
                // Check access
                if (!$voiceModel->isPublic() && !$voiceModel->isOwnedBy($user)) {
                    $hasPermission = $voiceModel->permittedUsers()
                        ->where('users.id', $user->id)
                        ->where('voice_model_user_access.can_use', true)
                        ->exists();
                    
                    if (!$hasPermission && !$user->hasRole('admin')) {
                        return response()->json(['error' => 'Access denied to voice model'], 403);
                    }
                }
                
                $modelPath = $voiceModel->model_path;
                $indexPath = $voiceModel->index_path;
            }
        }

        // Build voice configs for multi-voice swap
        $voiceConfigs = null;
        if ($voiceCount > 1 && !empty($validated['voice_configs'])) {
            $voiceConfigs = [];
            foreach ($validated['voice_configs'] as $vc) {
                $vcModel = VoiceModel::find($vc['model_id']);
                if (!$vcModel) {
                    return response()->json(['error' => 'Invalid voice model in voice_configs'], 400);
                }
                
                // Check access for each model
                if (!$vcModel->isPublic() && !$vcModel->isOwnedBy($user)) {
                    $hasPermission = $vcModel->permittedUsers()
                        ->where('users.id', $user->id)
                        ->where('voice_model_user_access.can_use', true)
                        ->exists();
                    
                    if (!$hasPermission && !$user->hasRole('admin')) {
                        return response()->json(['error' => "Access denied to voice model: {$vcModel->name}"], 403);
                    }
                }
                
                $voiceConfigs[] = [
                    'model_path' => $vcModel->model_path,
                    'index_path' => $vcModel->index_path,
                    'f0_up_key' => $vc['f0_up_key'] ?? 0,
                    'extraction_mode' => $vc['extraction_mode'] ?? 'main', // 'main' = HP5, 'all' = HP3
                ];
            }
        }

        // Concurrency guard: one active job per user (admins exempt)
        $activeJob = null;
        $job = \DB::transaction(function () use ($user, $validated, $voiceModel, $voiceCount, $mode, $sampleRate, $voiceConfigs, &$activeJob) {
            $activeJob = JobQueue::forUser($user->id)
                ->active()
                ->lockForUpdate()
                ->first();

            if ($activeJob && !$user->is_admin) {
                return null;
            }

            $typeMap = [
                'swap' => JobQueue::TYPE_AUDIO_SWAP,
                'split' => JobQueue::TYPE_AUDIO_SPLIT,
                'convert' => JobQueue::TYPE_AUDIO_CONVERT,
            ];

            return JobQueue::create([
                'user_id' => $user->id,
                'voice_model_id' => $voiceModel?->id,
                'type' => $typeMap[$mode] ?? JobQueue::TYPE_AUDIO_SWAP,
                'status' => JobQueue::STATUS_QUEUED,
                'parameters' => [
                    'mode' => $mode,
                    'sample_rate' => $sampleRate,
                    'f0_up_key' => $validated['f0_up_key'] ?? 0,
                    'index_rate' => $validated['index_rate'] ?? 0.75,
                    'pitch_shift_all' => $validated['pitch_shift_all'] ?? 0,
                    'model_name' => $voiceModel?->name,
                    'voice_count' => $voiceCount,
                ],
            ]);
        });

        if (!$job) {
            return response()->json([
                'error' => 'You have an active job in progress',
                'active_job_id' => $activeJob->uuid,
            ], 429);
        }

        // Build webhook URLs for voice engine callbacks (use internal URL to avoid Cloudflare)
        $baseUrl = config('services.internal_base_url', 'http://localhost:8000');
        $webhookUrls = [
            'progress_url' => "{$baseUrl}/api/internal/jobs/{$job->uuid}/progress",
            'complete_url' => "{$baseUrl}/api/internal/jobs/{$job->uuid}/complete",
            'status_url' => "{$baseUrl}/api/internal/jobs/{$job->uuid}/status",
        ];

        // Build voice engine payload
        $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
        $payload = [
            'job_uuid' => $job->uuid,
            'user_id' => $user->id,
            ...$webhookUrls,
            'audio' => $validated['audio'],
            'sample_rate' => $sampleRate,
            'mode' => $mode,
            'f0_up_key' => $validated['f0_up_key'] ?? 0,
            'index_rate' => $validated['index_rate'] ?? 0.75,
            'pitch_shift_all' => $validated['pitch_shift_all'] ?? 0,
            'instrumental_pitch' => $validated['instrumental_pitch'] ?? null,
            'extract_all_vocals' => $validated['extract_all_vocals'] ?? false,
            'voice_count' => $voiceCount,
        ];

        if ($modelPath) {
            $payload['model_path'] = $modelPath;
            $payload['index_path'] = $indexPath;
        }

        if (!empty($validated['checkpoint'])) {
            $payload['checkpoint'] = $validated['checkpoint'];
        }

        if ($voiceConfigs) {
            $payload['voice_configs'] = $voiceConfigs;
        }

        // Dispatch to voice engine (5s timeout — only waiting for 202 ack)
        try {
            $response = Http::timeout(5)->post("{$voiceEngineUrl}/audio/process", $payload);
        } catch (\Exception $e) {
            $job->update([
                'status' => JobQueue::STATUS_FAILED,
                'error_message' => 'Voice engine unavailable',
                'completed_at' => now(),
            ]);
            return response()->json([
                'error' => 'Voice engine unavailable',
                'job_id' => $job->uuid,
            ], 503);
        }

        if (!$response->successful()) {
            $detail = $response->json('error') ?? $response->json('detail') ?? 'Unknown error';
            $job->update([
                'status' => JobQueue::STATUS_FAILED,
                'error_message' => is_string($detail) ? $detail : json_encode($detail),
                'completed_at' => now(),
            ]);
            return response()->json([
                'error' => 'Voice engine rejected job',
                'message' => $detail,
                'job_id' => $job->uuid,
            ], $response->status());
        }

        return response()->json([
            'job_id' => $job->uuid,
            'status' => 'queued',
        ], 202);
    }

    /**
     * Detect number of distinct voices/singers in audio
     * 
     * Useful for:
     * - Determining if a song has backup vocals
     * - Knowing how many voice models to assign for multi-voice swap
     * - Detecting duets, harmonies, or group vocals
     */
    public function detectVoices(Request $request)
    {
        $validated = $request->validate([
            'audio' => 'required|string', // Base64 encoded audio
            'sample_rate' => 'nullable|integer|min:8000|max:96000',
            'use_vocals_only' => 'nullable|boolean',
            'max_voices' => 'nullable|integer|min:1|max:10',
        ]);

        try {
            // Call voice engine
            $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
            
            $payload = [
                'audio' => $validated['audio'],
                'sample_rate' => $validated['sample_rate'] ?? 44100,
                'use_vocals_only' => $validated['use_vocals_only'] ?? true,
                'max_voices' => $validated['max_voices'] ?? 5,
            ];

            $response = Http::timeout(120)->post("{$voiceEngineUrl}/voice-count/detect", $payload);

            if (!$response->successful()) {
                return response()->json([
                    'error' => 'Voice detection failed',
                    'message' => $response->json('detail') ?? 'Unknown error',
                ], 500);
            }

            return response()->json($response->json());

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Voice detection failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }
}
