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
            'f0_up_key' => 'nullable|integer|min:-24|max:24',
            'index_rate' => 'nullable|numeric|min:0|max:1',
            'pitch_shift_all' => 'nullable|integer|min:-24|max:24',
            'instrumental_pitch' => 'nullable|integer|min:-24|max:24',
        ]);

        $mode = $validated['mode'];
        $sampleRate = $validated['sample_rate'] ?? 44100;
        
        // Get voice model if specified
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

        // Determine job type based on mode
        $jobType = match($mode) {
            'convert' => 'audio_convert',
            'split' => 'audio_split',
            'swap' => 'audio_swap',
            default => 'audio_process',
        };

        // Create job record
        $job = JobQueue::create([
            'user_id' => $user->id,
            'voice_model_id' => $voiceModel?->id,
            'type' => $jobType,
            'status' => JobQueue::STATUS_PROCESSING,
            'parameters' => [
                'mode' => $mode,
                'sample_rate' => $sampleRate,
                'f0_up_key' => $validated['f0_up_key'] ?? 0,
                'index_rate' => $validated['index_rate'] ?? 0.75,
                'pitch_shift_all' => $validated['pitch_shift_all'] ?? 0,
                'model_name' => $voiceModel?->name,
            ],
            'started_at' => now(),
        ]);

        try {
            // Call voice engine
            $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
            
            $payload = [
                'audio' => $validated['audio'],
                'sample_rate' => $sampleRate,
                'mode' => $mode,
                'f0_up_key' => $validated['f0_up_key'] ?? 0,
                'index_rate' => $validated['index_rate'] ?? 0.75,
                'pitch_shift_all' => $validated['pitch_shift_all'] ?? 0,
                'instrumental_pitch' => $validated['instrumental_pitch'] ?? null,
            ];

            if ($modelPath) {
                $payload['model_path'] = $modelPath;
                $payload['index_path'] = $indexPath;
            }

            $response = Http::timeout(300)->post("{$voiceEngineUrl}/audio/process", $payload);

            if (!$response->successful()) {
                $job->update([
                    'status' => JobQueue::STATUS_FAILED,
                    'error_message' => 'Audio processing failed',
                    'error_details' => ['response' => $response->json()],
                    'completed_at' => now(),
                ]);

                return response()->json([
                    'error' => 'Audio processing failed',
                    'message' => $response->json('detail') ?? 'Unknown error',
                    'job_id' => $job->uuid,
                ], 500);
            }

            $result = $response->json();

            // Record usage if model was used
            if ($voiceModel && in_array($mode, ['convert', 'swap'])) {
                UsageEvent::create([
                    'user_id' => $user->id,
                    'event_type' => 'audio_conversion',
                    'voice_model_id' => $voiceModel->id,
                    'metadata' => [
                        'mode' => $mode,
                        'f0_up_key' => $validated['f0_up_key'] ?? 0,
                    ],
                ]);
                $voiceModel->incrementUsage();
            }

            // Mark job as completed
            $job->update([
                'status' => JobQueue::STATUS_COMPLETED,
                'completed_at' => now(),
                'progress' => 100,
            ]);

            return response()->json([
                ...$result,
                'job_id' => $job->uuid,
            ]);

        } catch (\Exception $e) {
            $job->update([
                'status' => JobQueue::STATUS_FAILED,
                'error_message' => $e->getMessage(),
                'completed_at' => now(),
            ]);

            return response()->json([
                'error' => 'Audio processing failed',
                'message' => $e->getMessage(),
                'job_id' => $job->uuid,
            ], 500);
        }
    }
}
