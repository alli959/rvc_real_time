<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\VoiceModel;
use App\Models\UsageEvent;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

/**
 * Controller for Text-to-Speech functionality.
 * 
 * Generates speech from text using Edge TTS or other TTS engines,
 * then optionally converts the voice using an RVC model.
 */
class TTSController extends Controller
{
    /**
     * Available voices for Edge TTS with enhanced metadata
     */
    protected array $voices = [
        // English - US
        'en-US-GuyNeural' => ['name' => 'Guy', 'language' => 'English (US)', 'gender' => 'male', 'supports_styles' => true],
        'en-US-JennyNeural' => ['name' => 'Jenny', 'language' => 'English (US)', 'gender' => 'female', 'supports_styles' => true],
        'en-US-AriaNeural' => ['name' => 'Aria', 'language' => 'English (US)', 'gender' => 'female', 'supports_styles' => true],
        'en-US-DavisNeural' => ['name' => 'Davis', 'language' => 'English (US)', 'gender' => 'male', 'supports_styles' => true],
        'en-US-TonyNeural' => ['name' => 'Tony', 'language' => 'English (US)', 'gender' => 'male', 'supports_styles' => true],
        'en-US-SaraNeural' => ['name' => 'Sara', 'language' => 'English (US)', 'gender' => 'female', 'supports_styles' => true],
        // English - UK
        'en-GB-RyanNeural' => ['name' => 'Ryan', 'language' => 'English (UK)', 'gender' => 'male', 'supports_styles' => false],
        'en-GB-SoniaNeural' => ['name' => 'Sonia', 'language' => 'English (UK)', 'gender' => 'female', 'supports_styles' => false],
        'en-GB-LibbyNeural' => ['name' => 'Libby', 'language' => 'English (UK)', 'gender' => 'female', 'supports_styles' => false],
        // English - Australia
        'en-AU-NatashaNeural' => ['name' => 'Natasha', 'language' => 'English (AU)', 'gender' => 'female', 'supports_styles' => false],
        'en-AU-WilliamNeural' => ['name' => 'William', 'language' => 'English (AU)', 'gender' => 'male', 'supports_styles' => false],
        // Spanish
        'es-ES-AlvaroNeural' => ['name' => 'Alvaro', 'language' => 'Spanish (Spain)', 'gender' => 'male', 'supports_styles' => false],
        'es-ES-ElviraNeural' => ['name' => 'Elvira', 'language' => 'Spanish (Spain)', 'gender' => 'female', 'supports_styles' => false],
        'es-MX-DaliaNeural' => ['name' => 'Dalia', 'language' => 'Spanish (Mexico)', 'gender' => 'female', 'supports_styles' => false],
        'es-MX-JorgeNeural' => ['name' => 'Jorge', 'language' => 'Spanish (Mexico)', 'gender' => 'male', 'supports_styles' => false],
        // French
        'fr-FR-HenriNeural' => ['name' => 'Henri', 'language' => 'French (France)', 'gender' => 'male', 'supports_styles' => false],
        'fr-FR-DeniseNeural' => ['name' => 'Denise', 'language' => 'French (France)', 'gender' => 'female', 'supports_styles' => false],
        'fr-CA-SylvieNeural' => ['name' => 'Sylvie', 'language' => 'French (Canada)', 'gender' => 'female', 'supports_styles' => false],
        'fr-CA-JeanNeural' => ['name' => 'Jean', 'language' => 'French (Canada)', 'gender' => 'male', 'supports_styles' => false],
        // German
        'de-DE-ConradNeural' => ['name' => 'Conrad', 'language' => 'German', 'gender' => 'male', 'supports_styles' => false],
        'de-DE-KatjaNeural' => ['name' => 'Katja', 'language' => 'German', 'gender' => 'female', 'supports_styles' => false],
        // Japanese
        'ja-JP-KeitaNeural' => ['name' => 'Keita', 'language' => 'Japanese', 'gender' => 'male', 'supports_styles' => false],
        'ja-JP-NanamiNeural' => ['name' => 'Nanami', 'language' => 'Japanese', 'gender' => 'female', 'supports_styles' => false],
        // Chinese
        'zh-CN-YunxiNeural' => ['name' => 'Yunxi', 'language' => 'Chinese (Mandarin)', 'gender' => 'male', 'supports_styles' => true],
        'zh-CN-XiaoxiaoNeural' => ['name' => 'Xiaoxiao', 'language' => 'Chinese (Mandarin)', 'gender' => 'female', 'supports_styles' => true],
        // Korean
        'ko-KR-InJoonNeural' => ['name' => 'InJoon', 'language' => 'Korean', 'gender' => 'male', 'supports_styles' => false],
        'ko-KR-SunHiNeural' => ['name' => 'SunHi', 'language' => 'Korean', 'gender' => 'female', 'supports_styles' => false],
        // Russian
        'ru-RU-DmitryNeural' => ['name' => 'Dmitry', 'language' => 'Russian', 'gender' => 'male', 'supports_styles' => false],
        'ru-RU-SvetlanaNeural' => ['name' => 'Svetlana', 'language' => 'Russian', 'gender' => 'female', 'supports_styles' => false],
        // Portuguese
        'pt-BR-AntonioNeural' => ['name' => 'Antonio', 'language' => 'Portuguese (Brazil)', 'gender' => 'male', 'supports_styles' => false],
        'pt-BR-FranciscaNeural' => ['name' => 'Francisca', 'language' => 'Portuguese (Brazil)', 'gender' => 'female', 'supports_styles' => false],
        'pt-PT-DuarteNeural' => ['name' => 'Duarte', 'language' => 'Portuguese (Portugal)', 'gender' => 'male', 'supports_styles' => false],
        'pt-PT-RaquelNeural' => ['name' => 'Raquel', 'language' => 'Portuguese (Portugal)', 'gender' => 'female', 'supports_styles' => false],
        // Italian
        'it-IT-DiegoNeural' => ['name' => 'Diego', 'language' => 'Italian', 'gender' => 'male', 'supports_styles' => false],
        'it-IT-ElsaNeural' => ['name' => 'Elsa', 'language' => 'Italian', 'gender' => 'female', 'supports_styles' => false],
        // Dutch
        'nl-NL-MaartenNeural' => ['name' => 'Maarten', 'language' => 'Dutch', 'gender' => 'male', 'supports_styles' => false],
        'nl-NL-ColetteNeural' => ['name' => 'Colette', 'language' => 'Dutch', 'gender' => 'female', 'supports_styles' => false],
        // Polish
        'pl-PL-MarekNeural' => ['name' => 'Marek', 'language' => 'Polish', 'gender' => 'male', 'supports_styles' => false],
        'pl-PL-ZofiaNeural' => ['name' => 'Zofia', 'language' => 'Polish', 'gender' => 'female', 'supports_styles' => false],
        // Arabic
        'ar-SA-HamedNeural' => ['name' => 'Hamed', 'language' => 'Arabic (Saudi)', 'gender' => 'male', 'supports_styles' => false],
        'ar-SA-ZariyahNeural' => ['name' => 'Zariyah', 'language' => 'Arabic (Saudi)', 'gender' => 'female', 'supports_styles' => false],
        // Hindi
        'hi-IN-MadhurNeural' => ['name' => 'Madhur', 'language' => 'Hindi', 'gender' => 'male', 'supports_styles' => false],
        'hi-IN-SwaraNeural' => ['name' => 'Swara', 'language' => 'Hindi', 'gender' => 'female', 'supports_styles' => false],
        // Icelandic
        'is-IS-GudrunNeural' => ['name' => 'Guðrún', 'language' => 'Icelandic', 'gender' => 'female', 'supports_styles' => false],
        'is-IS-GunnarNeural' => ['name' => 'Gunnar', 'language' => 'Icelandic', 'gender' => 'male', 'supports_styles' => false],
        // Swedish
        'sv-SE-SofieNeural' => ['name' => 'Sofie', 'language' => 'Swedish', 'gender' => 'female', 'supports_styles' => false],
        'sv-SE-MattiasNeural' => ['name' => 'Mattias', 'language' => 'Swedish', 'gender' => 'male', 'supports_styles' => false],
        // Norwegian
        'nb-NO-IselinNeural' => ['name' => 'Iselin', 'language' => 'Norwegian', 'gender' => 'female', 'supports_styles' => false],
        'nb-NO-FinnNeural' => ['name' => 'Finn', 'language' => 'Norwegian', 'gender' => 'male', 'supports_styles' => false],
        // Danish
        'da-DK-ChristelNeural' => ['name' => 'Christel', 'language' => 'Danish', 'gender' => 'female', 'supports_styles' => false],
        'da-DK-JeppeNeural' => ['name' => 'Jeppe', 'language' => 'Danish', 'gender' => 'male', 'supports_styles' => false],
        // Finnish
        'fi-FI-SelmaNeural' => ['name' => 'Selma', 'language' => 'Finnish', 'gender' => 'female', 'supports_styles' => false],
        'fi-FI-HarriNeural' => ['name' => 'Harri', 'language' => 'Finnish', 'gender' => 'male', 'supports_styles' => false],
        // Turkish
        'tr-TR-EmelNeural' => ['name' => 'Emel', 'language' => 'Turkish', 'gender' => 'female', 'supports_styles' => false],
        'tr-TR-AhmetNeural' => ['name' => 'Ahmet', 'language' => 'Turkish', 'gender' => 'male', 'supports_styles' => false],
        // Greek
        'el-GR-AthinaNeural' => ['name' => 'Athina', 'language' => 'Greek', 'gender' => 'female', 'supports_styles' => false],
        'el-GR-NestorasNeural' => ['name' => 'Nestoras', 'language' => 'Greek', 'gender' => 'male', 'supports_styles' => false],
    ];

    /**
     * Available speech styles (emotions) for certain voices
     */
    protected array $styles = [
        'default' => 'Normal speaking voice',
        'cheerful' => 'Expresses a positive and happy tone',
        'sad' => 'Expresses a sorrowful tone',
        'angry' => 'Expresses an angry and annoyed tone',
        'fearful' => 'Expresses a scared and nervous tone',
        'friendly' => 'Expresses a warm and pleasant tone',
        'whispering' => 'Speaks softly with a whisper',
        'shouting' => 'Speaks loudly with emphasis',
        'excited' => 'Expresses an upbeat and enthusiastic tone',
        'unfriendly' => 'Expresses a cold and indifferent tone',
        'terrified' => 'Expresses a very scared tone',
        'hopeful' => 'Expresses a warm and hoping tone',
        'narration-professional' => 'Neutral narration style',
        'newscast-casual' => 'Casual news reading',
        'newscast-formal' => 'Formal news reading',
        'documentary-narration' => 'Documentary style narration',
        'customerservice' => 'Friendly customer service voice',
        'chat' => 'Casual conversational style',
        'assistant' => 'Warm and helpful assistant voice',
        'empathetic' => 'Understanding and caring tone',
    ];

    /**
     * Get available TTS voices, styles, and languages
     */
    public function getVoices()
    {
        $voices = collect($this->voices)->map(fn($info, $id) => [
            'id' => $id,
            ...$info,
        ])->values();
        
        $styles = collect($this->styles)->map(fn($desc, $id) => [
            'id' => $id,
            'description' => $desc,
        ])->values();
        
        // Extract unique languages from voices
        $languages = collect($this->voices)
            ->pluck('language')
            ->unique()
            ->sort()
            ->values();
        
        return response()->json([
            'voices' => $voices,
            'styles' => $styles,
            'languages' => $languages,
        ]);
    }

    /**
     * Generate TTS audio, optionally with voice conversion
     * 
     * Request body:
     * - text: The text to convert to speech (required)
     * - voice: Edge TTS voice ID (default: en-US-GuyNeural)
     * - style: Speaking style/emotion (default: default)
     * - rate: Speech rate adjustment (-50% to +50%, default: 0%)
     * - pitch: Pitch adjustment (-50Hz to +50Hz, default: 0Hz)
     * - voice_model_id: Optional voice model ID for RVC conversion
     * - f0_up_key: Pitch shift for RVC (-12 to 12, default: 0)
     */
    public function generate(Request $request)
    {
        $user = $request->user();

        $validated = $request->validate([
            'text' => 'required|string|max:5000',
            'voice' => 'nullable|string',
            'style' => 'nullable|string',
            'rate' => 'nullable|string', // e.g., "+10%", "-20%"
            'pitch' => 'nullable|string', // e.g., "+5Hz", "-10Hz"
            'voice_model_id' => 'nullable|integer|exists:voice_models,id',
            'f0_up_key' => 'nullable|integer|min:-12|max:12',
            'index_rate' => 'nullable|numeric|min:0|max:1',
        ]);

        $text = $validated['text'];
        $voice = $validated['voice'] ?? 'en-US-GuyNeural';
        $style = $validated['style'] ?? 'default';
        $rate = $validated['rate'] ?? '+0%';
        $pitch = $validated['pitch'] ?? '+0Hz';

        // Validate voice
        if (!isset($this->voices[$voice])) {
            return response()->json([
                'error' => 'Invalid voice',
                'available_voices' => array_keys($this->voices),
            ], 422);
        }

        try {
            // Generate TTS audio using voice engine
            $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
            
            $ttsResponse = Http::timeout(60)->post("{$voiceEngineUrl}/tts", [
                'text' => $text,
                'voice' => $voice,
                'style' => $style,
                'rate' => $rate,
                'pitch' => $pitch,
            ]);

            if (!$ttsResponse->successful()) {
                return response()->json([
                    'error' => 'TTS generation failed',
                    'message' => $ttsResponse->json('error') ?? 'Unknown error',
                ], 500);
            }

            $audioBase64 = $ttsResponse->json('audio');
            $sampleRate = $ttsResponse->json('sample_rate', 24000);

            // If voice model specified, apply RVC conversion
            if (!empty($validated['voice_model_id'])) {
                $voiceModel = VoiceModel::findOrFail($validated['voice_model_id']);
                
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

                // Send to voice engine for conversion
                $convertResponse = Http::timeout(120)->post("{$voiceEngineUrl}/convert", [
                    'audio' => $audioBase64,
                    'sample_rate' => $sampleRate,
                    'model_path' => $voiceModel->model_path,
                    'index_path' => $voiceModel->index_path,
                    'f0_up_key' => $validated['f0_up_key'] ?? 0,
                    'index_rate' => $validated['index_rate'] ?? 0.75,
                ]);

                if (!$convertResponse->successful()) {
                    return response()->json([
                        'error' => 'Voice conversion failed',
                        'message' => $convertResponse->json('error') ?? 'Unknown error',
                    ], 500);
                }

                $audioBase64 = $convertResponse->json('audio');
                $sampleRate = $convertResponse->json('sample_rate', 16000);

                // Record usage
                UsageEvent::recordTTS($user->id, $voiceModel->id, strlen($text), true);
                $voiceModel->incrementUsage();
            } else {
                // Record TTS-only usage
                UsageEvent::recordTTS($user->id, null, strlen($text), false);
            }

            return response()->json([
                'audio' => $audioBase64,
                'sample_rate' => $sampleRate,
                'format' => 'wav',
                'text_length' => strlen($text),
                'voice' => $voice,
                'style' => $style,
                'converted' => !empty($validated['voice_model_id']),
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'TTS generation failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Stream TTS audio (for longer texts)
     */
    public function stream(Request $request)
    {
        $validated = $request->validate([
            'text' => 'required|string|max:50000',
            'voice' => 'nullable|string',
            'style' => 'nullable|string',
            'rate' => 'nullable|string',
            'pitch' => 'nullable|string',
        ]);

        $text = $validated['text'];
        $voice = $validated['voice'] ?? 'en-US-GuyNeural';

        // Split text into sentences for streaming
        $sentences = preg_split('/(?<=[.!?])\s+/', $text, -1, PREG_SPLIT_NO_EMPTY);

        return response()->stream(function () use ($sentences, $voice, $validated) {
            $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
            
            foreach ($sentences as $sentence) {
                if (empty(trim($sentence))) continue;
                
                try {
                    $response = Http::timeout(30)->post("{$voiceEngineUrl}/tts", [
                        'text' => $sentence,
                        'voice' => $voice,
                        'style' => $validated['style'] ?? 'default',
                        'rate' => $validated['rate'] ?? '+0%',
                        'pitch' => $validated['pitch'] ?? '+0Hz',
                    ]);

                    if ($response->successful()) {
                        echo "data: " . json_encode([
                            'audio' => $response->json('audio'),
                            'text' => $sentence,
                        ]) . "\n\n";
                        ob_flush();
                        flush();
                    }
                } catch (\Exception $e) {
                    // Continue with next sentence
                    continue;
                }
            }

            echo "data: [DONE]\n\n";
            ob_flush();
            flush();
        }, 200, [
            'Content-Type' => 'text/event-stream',
            'Cache-Control' => 'no-cache',
            'Connection' => 'keep-alive',
            'X-Accel-Buffering' => 'no',
        ]);
    }
}
