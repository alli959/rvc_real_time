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
                'voice_count' => $voiceCount,
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
                'extract_all_vocals' => $validated['extract_all_vocals'] ?? false, // Default to main vocal only
                'voice_count' => $voiceCount,
            ];

            if ($modelPath) {
                $payload['model_path'] = $modelPath;
                $payload['index_path'] = $indexPath;
            }
            
            if ($voiceConfigs) {
                $payload['voice_configs'] = $voiceConfigs;
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
