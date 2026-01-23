<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\VoiceModel;
use App\Models\UsageEvent;
use App\Models\JobQueue;
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
     * Available emotion tags for inline text emotions
     */
    protected array $emotions = [
        'positive' => [
            'happy' => 'Cheerful, upbeat tone',
            'excited' => 'Very enthusiastic',
            'cheerful' => 'Light and positive',
            'joyful' => 'Full of joy',
        ],
        'negative' => [
            'sad' => 'Melancholic, slow',
            'melancholy' => 'Deep sadness',
            'depressed' => 'Very low energy',
            'disappointed' => 'Let down feeling',
        ],
        'angry' => [
            'angry' => 'Frustrated, intense',
            'furious' => 'Very angry',
            'annoyed' => 'Mildly irritated',
            'frustrated' => 'Exasperated',
        ],
        'calm' => [
            'calm' => 'Relaxed, peaceful',
            'peaceful' => 'Very serene',
            'relaxed' => 'At ease',
            'neutral' => 'Standard tone',
        ],
        'surprised' => [
            'surprised' => 'Caught off guard',
            'shocked' => 'Very surprised',
            'amazed' => 'In awe',
        ],
        'fear' => [
            'scared' => 'Frightened',
            'terrified' => 'Extremely scared',
            'anxious' => 'Nervous, worried',
            'nervous' => 'Slightly on edge',
        ],
        'special' => [
            'whisper' => 'Quiet, secretive',
            'shouting' => 'Loud, emphatic',
            'sarcastic' => 'Ironic tone',
            'romantic' => 'Soft, loving',
            'serious' => 'Grave, important',
            'playful' => 'Fun, teasing',
            'dramatic' => 'Theatrical',
            'mysterious' => 'Enigmatic',
        ],
        'voice_effects' => [
            'robot' => 'Robotic/electronic voice (bitcrush + lowpass)',
            'spooky' => 'Spooky/haunted voice (reverb + lowpass)',
            'ethereal' => 'Ethereal/heavenly voice (reverb + highpass)',
            'phone' => 'Phone call quality (bandpass filter)',
            'radio' => 'Radio broadcast quality (bandpass + saturation)',
            'megaphone' => 'Megaphone/PA system (bandpass + distortion)',
            'echo' => 'Echoey room (reverb)',
            'underwater' => 'Underwater/muffled (heavy lowpass)',
        ],
        'sounds' => [
            // Laughs
            'laugh' => 'Laughing (haha)',
            'giggle' => 'Light giggle (hehe)',
            'chuckle' => 'Soft laugh (heh)',
            'snicker' => 'Suppressed laugh',
            'cackle' => 'Witch-like laugh',
            // Crying
            'cry' => 'Crying sound',
            'sob' => 'Deep sobbing',
            'sniff' => 'Sniffling',
            // Surprise/Fear
            'gasp' => 'Gasp (aah!)',
            'scream' => 'Screaming',
            'shriek' => 'High scream',
            // Pain/Discomfort
            'groan' => 'Groaning (ugh)',
            'moan' => 'Moaning',
            'sigh' => 'Sighing (haaah)',
            'yawn' => 'Yawning',
            // Body sounds
            'cough' => 'Coughing',
            'sneeze' => 'Sneezing (achoo)',
            'hiccup' => 'Hiccup',
            'burp' => 'Burping',
            'gulp' => 'Swallowing',
            'slurp' => 'Slurping',
            // Vocalizations
            'growl' => 'Growling (grrr)',
            'hiss' => 'Hissing (ssss)',
            'hum' => 'Humming',
            'whistle' => 'Whistling',
            'shush' => 'Shushing (shhh)',
            'kiss' => 'Kiss (mwah)',
            'blow' => 'Blowing air',
            // Breathing
            'pant' => 'Heavy panting',
            'breathe' => 'Deep breath',
            'inhale' => 'Inhaling',
            'exhale' => 'Exhaling',
            // Speech patterns
            'stutter' => 'Stuttering',
            'mumble' => 'Mumbling',
            'stammer' => 'Stammering',
            // Thinking
            'hmm' => 'Thinking (hmm)',
            'uhh' => 'Hesitation (uhh)',
            'umm' => 'Filler (umm)',
            // Reactions
            'wow' => 'Amazement (wow)',
            'ooh' => 'Interest (ooh)',
            'ahh' => 'Realization (ahh)',
            'ugh' => 'Disgust (ugh)',
            'eww' => 'Grossed out',
            'yay' => 'Celebration (yay!)',
            'boo' => 'Disapproval',
            'woohoo' => 'Excitement (woohoo!)',
            'ow' => 'Pain (ow)',
            'ouch' => 'Pain (ouch)',
            'phew' => 'Relief (phew)',
            'tsk' => 'Disapproval (tsk tsk)',
            'psst' => 'Getting attention (psst)',
        ],
    ];

    /**
     * Get available TTS voices, styles, languages, and emotions
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
            'emotions' => $this->emotions,
            'emotionUsage' => [
                'tagged' => '[emotion]Your text here[/emotion]',
                'soundBracket' => '[laugh]',
                'soundAsterisk' => '*laugh*',
                'soundParen' => '(sigh)',
            ],
            'emotionExamples' => [
                '[happy]I am so excited to see you![/happy]',
                '[sad]I miss you so much[/sad]',
                'Hello! *laugh* That was funny!',
                '[whisper]This is a secret[/whisper]',
                'Oh no! (gasp) What happened?',
                '[excited]We won the game![/excited] [laugh]',
            ],
        ]);
    }

    /**
     * Get TTS capabilities including whether Bark (native emotions) is available
     */
    public function getCapabilities()
    {
        $voiceEngineUrl = config('services.voice_engine.url', 'http://voice-engine:8000');
        
        try {
            // Forward to voice engine to get actual capabilities
            $response = Http::timeout(10)->get("{$voiceEngineUrl}/tts/capabilities");
            
            if ($response->successful()) {
                return response()->json($response->json());
            }
        } catch (\Exception $e) {
            \Log::warning("Failed to fetch TTS capabilities from voice engine: " . $e->getMessage());
        }
        
        // Fallback response if voice engine is unreachable
        return response()->json([
            'bark_available' => false,
            'edge_tts_available' => true,
            'supported_emotions' => [
                'happy', 'excited', 'cheerful', 'joyful',
                'sad', 'melancholy', 'depressed', 'disappointed',
                'angry', 'furious', 'annoyed', 'frustrated',
                'calm', 'peaceful', 'relaxed', 'neutral',
                'surprised', 'shocked', 'amazed',
                'scared', 'terrified', 'anxious', 'nervous',
                'whisper', 'shouting', 'sarcastic', 'romantic',
                'serious', 'playful', 'dramatic', 'mysterious',
                'robot', 'spooky', 'ethereal', 'phone', 'radio',
                'megaphone', 'echo', 'underwater'
            ],
            'supported_sounds' => [
                'laugh', 'laughing', 'giggle', 'chuckle', 'snicker', 'cackle',
                'cry', 'crying', 'sob', 'sniff',
                'gasp', 'scream', 'shriek',
                'groan', 'moan', 'sigh', 'yawn',
                'cough', 'sneeze', 'hiccup', 'burp', 'gulp', 'slurp',
                'growl', 'hiss', 'hum', 'whistle', 'shush', 'kiss', 'blow',
                'pant', 'breathe', 'inhale', 'exhale',
                'stutter', 'mumble', 'stammer',
                'hmm', 'thinking', 'uhh', 'umm'
            ],
            'bark_speakers' => ['default', 'male1', 'male2', 'female1', 'female2', 'dramatic', 'calm'],
            'recommendation' => 'Using Edge TTS with audio processing. Emotions are simulated via pitch/rate changes and audio effects.'
        ]);
    }

    /**
     * Generate TTS audio, optionally with voice conversion
     * 
     * Request body:
     * - text: The text to convert to speech (required, supports emotion tags)
     * - voice: Edge TTS voice ID (default: en-US-GuyNeural)
     * - style: Speaking style/emotion (default: default)
     * - rate: Speech rate adjustment (-50% to +50%, default: 0%)
     * - pitch: Pitch adjustment (-50Hz to +50Hz, default: 0Hz)
     * - voice_model_id: Optional voice model ID for RVC conversion
     * - f0_up_key: Pitch shift for RVC (-12 to 12, default: 0)
     * - include_segments: Optional array of {voice_model_id, text} for multi-voice
     * 
     * Emotion tags in text:
     * - [happy]Hello![/happy] - Tagged sections with emotions
     * - [laugh] or *laugh* or (laugh) - Sound effects
     * - [whisper]Secret[/whisper] - Special effects
     * - <speed rate="-30%">Slow text</speed> - Speed control
     * - <include voice_model_id="5">Other voice</include> - Multi-voice
     */
    public function generate(Request $request)
    {
        $user = $request->user();

        $validated = $request->validate([
            'text' => 'required|string|max:10000',
            'voice' => 'nullable|string',
            'style' => 'nullable|string',
            'rate' => 'nullable|string', // e.g., "+10%", "-20%"
            'pitch' => 'nullable|string', // e.g., "+5Hz", "-10Hz"
            'voice_model_id' => 'nullable|integer|exists:voice_models,id',
            'f0_up_key' => 'nullable|integer|min:-12|max:12',
            'index_rate' => 'nullable|numeric|min:0|max:1',
            'apply_effects' => 'nullable|string|max:50', // Audio effect to apply after conversion
            // Bark TTS options (native emotion support)
            'use_bark' => 'nullable|boolean', // Use Bark TTS for native emotions (default: true)
            'bark_speaker' => 'nullable|string|in:default,male1,male2,female1,female2,dramatic,calm',
            // Multi-voice support
            'include_segments' => 'nullable|array',
            'include_segments.*.voice_model_id' => 'nullable|integer|exists:voice_models,id',
            'include_segments.*.text' => 'nullable|string',
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

        // Create job record
        $voiceModel = !empty($validated['voice_model_id']) 
            ? VoiceModel::find($validated['voice_model_id']) 
            : null;

        $job = JobQueue::create([
            'user_id' => $user->id,
            'voice_model_id' => $voiceModel?->id,
            'type' => JobQueue::TYPE_TTS,
            'status' => JobQueue::STATUS_PROCESSING,
            'parameters' => [
                'text' => $text,
                'voice' => $voice,
                'style' => $style,
                'rate' => $rate,
                'pitch' => $pitch,
                'f0_up_key' => $validated['f0_up_key'] ?? 0,
                'index_rate' => $validated['index_rate'] ?? 0.75,
                'with_conversion' => !empty($validated['voice_model_id']),
            ],
            'started_at' => now(),
        ]);

        try {
            // Check if this is a multi-voice request (has <include> tags or include_segments)
            $hasIncludeTags = preg_match('/<include\s+[^>]+>/', $text);
            $hasIncludeSegments = !empty($validated['include_segments']);
            
            if ($hasIncludeTags || $hasIncludeSegments) {
                // Multi-voice TTS generation
                return $this->generateMultiVoice($request, $validated, $voiceModel, $job, $user);
            }
            
            // Generate TTS audio using voice engine
            $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
            
            // Use longer timeout for Bark TTS (it needs time to download models on first run and generate)
            // Bark generation can take 15-30 seconds, first run can take 2-3 minutes for model download
            $useBark = $validated['use_bark'] ?? true;
            $ttsTimeout = $useBark ? 180 : 60;  // 3 minutes for Bark, 1 minute for Edge TTS
            
            $ttsResponse = Http::timeout($ttsTimeout)->post("{$voiceEngineUrl}/tts", [
                'text' => $text,
                'voice' => $voice,
                'style' => $style,
                'rate' => $rate,
                'pitch' => $pitch,
                'use_bark' => $useBark,
                'bark_speaker' => $validated['bark_speaker'] ?? 'default',
            ]);

            if (!$ttsResponse->successful()) {
                $job->update([
                    'status' => JobQueue::STATUS_FAILED,
                    'error_message' => 'TTS generation failed',
                    'error_details' => ['response' => $ttsResponse->json()],
                    'completed_at' => now(),
                ]);

                return response()->json([
                    'error' => 'TTS generation failed',
                    'message' => $ttsResponse->json('error') ?? 'Unknown error',
                    'job_id' => $job->uuid,
                ], 500);
            }

            $audioBase64 = $ttsResponse->json('audio');
            $sampleRate = $ttsResponse->json('sample_rate', 24000);

            // If voice model specified, apply RVC conversion
            if ($voiceModel) {
                // Check access
                if (!$voiceModel->isPublic() && !$voiceModel->isOwnedBy($user)) {
                    $hasPermission = $voiceModel->permittedUsers()
                        ->where('users.id', $user->id)
                        ->where('voice_model_user_access.can_use', true)
                        ->exists();
                    
                    if (!$hasPermission && !$user->hasRole('admin')) {
                        $job->update([
                            'status' => JobQueue::STATUS_FAILED,
                            'error_message' => 'Access denied to voice model',
                            'completed_at' => now(),
                        ]);
                        return response()->json(['error' => 'Access denied to voice model', 'job_id' => $job->uuid], 403);
                    }
                }

                // Send to voice engine for conversion
                $convertPayload = [
                    'audio' => $audioBase64,
                    'sample_rate' => $sampleRate,
                    'model_path' => $voiceModel->getVoiceEngineModelPath(),
                    'index_path' => $voiceModel->getVoiceEngineIndexPath(),
                    'f0_up_key' => $validated['f0_up_key'] ?? 0,
                    'index_rate' => $validated['index_rate'] ?? 0.75,
                ];
                
                // Add audio effects if specified (applies after voice conversion)
                if (!empty($validated['apply_effects'])) {
                    $convertPayload['apply_effects'] = $validated['apply_effects'];
                }
                
                $convertResponse = Http::timeout(120)->post("{$voiceEngineUrl}/convert", $convertPayload);

                if (!$convertResponse->successful()) {
                    $job->update([
                        'status' => JobQueue::STATUS_FAILED,
                        'error_message' => 'Voice conversion failed',
                        'error_details' => ['response' => $convertResponse->json()],
                        'completed_at' => now(),
                    ]);

                    return response()->json([
                        'error' => 'Voice conversion failed',
                        'message' => $convertResponse->json('error') ?? 'Unknown error',
                        'job_id' => $job->uuid,
                    ], 500);
                }

                $audioBase64 = $convertResponse->json('audio');
                // IMPORTANT: Use sample_rate from voice engine - it reflects the model's actual output rate (32k/40k/48k)
                // Fallback to 48000 only if missing (most common RVC model rate)
                $sampleRate = $convertResponse->json('sample_rate');
                if (!$sampleRate) {
                    Log::warning('Voice engine /convert response missing sample_rate', [
                        'model' => $voiceModel->model_path,
                        'response_keys' => array_keys($convertResponse->json() ?? []),
                    ]);
                    $sampleRate = 48000; // Safe fallback - most RVC models are 48kHz
                }

                // Record usage
                UsageEvent::recordTTS($user->id, $voiceModel->id, strlen($text), true);
                $voiceModel->incrementUsage();
            } else {
                // Record TTS-only usage
                UsageEvent::recordTTS($user->id, null, strlen($text), false);
            }

            // Mark job as completed
            $job->update([
                'status' => JobQueue::STATUS_COMPLETED,
                'completed_at' => now(),
                'progress' => 100,
            ]);

            return response()->json([
                'audio' => $audioBase64,
                'sample_rate' => $sampleRate,
                'format' => 'wav',
                'text_length' => strlen($text),
                'voice' => $voice,
                'style' => $style,
                'converted' => !empty($validated['voice_model_id']),
                'job_id' => $job->uuid,
            ]);

        } catch (\Exception $e) {
            $job->update([
                'status' => JobQueue::STATUS_FAILED,
                'error_message' => $e->getMessage(),
                'completed_at' => now(),
            ]);

            return response()->json([
                'error' => 'TTS generation failed',
                'message' => $e->getMessage(),
                'job_id' => $job->uuid,
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
    
    /**
     * Generate multi-voice TTS with include segment support
     */
    protected function generateMultiVoice(Request $request, array $validated, ?VoiceModel $defaultModel, JobQueue $job, $user)
    {
        $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
        
        // Build voice model mappings for include segments
        $voiceModelMappings = [];
        $missingModelIds = [];
        
        // Extract voice_model_ids from include tags in text
        preg_match_all('/<include\s+voice_model_id=["\']?(\d+)["\']?[^>]*>/', $validated['text'], $matches);
        $includeModelIds = array_unique($matches[1] ?? []);
        
        // Also check include_segments array
        if (!empty($validated['include_segments'])) {
            foreach ($validated['include_segments'] as $segment) {
                if (!empty($segment['voice_model_id'])) {
                    $includeModelIds[] = $segment['voice_model_id'];
                }
            }
            $includeModelIds = array_unique($includeModelIds);
        }
        
        // Load all required voice models and check permissions
        foreach ($includeModelIds as $modelId) {
            $model = VoiceModel::find($modelId);
            if (!$model) {
                $missingModelIds[] = $modelId;
                continue;
            }
            
            // Check access
            if (!$model->isPublic() && !$model->isOwnedBy($user)) {
                $hasPermission = $model->permittedUsers()
                    ->where('users.id', $user->id)
                    ->where('voice_model_user_access.can_use', true)
                    ->exists();
                
                if (!$hasPermission && !$user->hasRole('admin')) {
                    continue; // Skip this model silently
                }
            }
            
            $voiceModelMappings[(string)$modelId] = [
                'model_path' => $model->model_path,
                'index_path' => $model->index_path,
            ];
        }
        
        // Return error if any voice models referenced in text don't exist
        if (!empty($missingModelIds)) {
            $job->update([
                'status' => JobQueue::STATUS_FAILED,
                'error_message' => 'Voice model(s) not found',
                'error_details' => ['missing_ids' => $missingModelIds],
                'completed_at' => now(),
            ]);
            
            return response()->json([
                'error' => 'Voice model(s) not found',
                'message' => 'Voice model ID(s) ' . implode(', ', $missingModelIds) . ' do not exist. Please use valid voice model IDs in your <include> tags.',
                'missing_ids' => $missingModelIds,
                'job_id' => $job->uuid,
            ], 422);
        }
        
        // Prepare request to voice engine multi-voice endpoint
        $multiVoicePayload = [
            'text' => $validated['text'],
            'voice' => $validated['voice'] ?? 'en-US-GuyNeural',
            'style' => $validated['style'] ?? 'default',
            'rate' => $validated['rate'] ?? '+0%',
            'pitch' => $validated['pitch'] ?? '+0Hz',
            'voice_model_mappings' => $voiceModelMappings,
        ];
        
        // Add default model if specified
        if ($defaultModel) {
            $multiVoicePayload['default_model_path'] = $defaultModel->model_path;
            $multiVoicePayload['default_index_path'] = $defaultModel->index_path;
            $multiVoicePayload['default_f0_up_key'] = $validated['f0_up_key'] ?? 0;
            $multiVoicePayload['default_index_rate'] = $validated['index_rate'] ?? 0.75;
        }
        
        try {
            $response = Http::timeout(180)->post("{$voiceEngineUrl}/tts/multi-voice", $multiVoicePayload);
            
            if (!$response->successful()) {
                $job->update([
                    'status' => JobQueue::STATUS_FAILED,
                    'error_message' => 'Multi-voice TTS generation failed',
                    'error_details' => ['response' => $response->json()],
                    'completed_at' => now(),
                ]);

                return response()->json([
                    'error' => 'Multi-voice TTS generation failed',
                    'message' => $response->json('error') ?? 'Unknown error',
                    'job_id' => $job->uuid,
                ], 500);
            }
            
            $audioBase64 = $response->json('audio');
            $sampleRate = $response->json('sample_rate', 24000);
            $segmentsProcessed = $response->json('segments_processed', 1);
            $includeSegmentsUsed = $response->json('include_segments_used', 0);
            
            // Apply post-conversion effects if specified
            if (!empty($validated['apply_effects']) && $audioBase64) {
                $effectsResponse = Http::timeout(60)->post("{$voiceEngineUrl}/apply-effects", [
                    'audio' => $audioBase64,
                    'sample_rate' => $sampleRate,
                    'effect' => $validated['apply_effects'],
                ]);
                
                if ($effectsResponse->successful()) {
                    $audioBase64 = $effectsResponse->json('audio');
                }
            }
            
            // Record usage for all models used
            foreach ($includeModelIds as $modelId) {
                if (isset($voiceModelMappings[(string)$modelId])) {
                    $model = VoiceModel::find($modelId);
                    if ($model) {
                        UsageEvent::recordTTS($user->id, $model->id, strlen($validated['text']) / count($includeModelIds), true);
                        $model->incrementUsage();
                    }
                }
            }
            
            if ($defaultModel) {
                UsageEvent::recordTTS($user->id, $defaultModel->id, strlen($validated['text']), true);
                $defaultModel->incrementUsage();
            }
            
            // Mark job as completed
            $job->update([
                'status' => JobQueue::STATUS_COMPLETED,
                'completed_at' => now(),
                'progress' => 100,
            ]);

            return response()->json([
                'audio' => $audioBase64,
                'sample_rate' => $sampleRate,
                'format' => 'wav',
                'text_length' => strlen($validated['text']),
                'voice' => $validated['voice'] ?? 'en-US-GuyNeural',
                'style' => $validated['style'] ?? 'default',
                'converted' => !empty($defaultModel) || !empty($voiceModelMappings),
                'multi_voice' => true,
                'segments_processed' => $segmentsProcessed,
                'include_segments_used' => $includeSegmentsUsed,
                'job_id' => $job->uuid,
            ]);
            
        } catch (\Exception $e) {
            $job->update([
                'status' => JobQueue::STATUS_FAILED,
                'error_message' => $e->getMessage(),
                'completed_at' => now(),
            ]);

            return response()->json([
                'error' => 'Multi-voice TTS generation failed',
                'message' => $e->getMessage(),
                'job_id' => $job->uuid,
            ], 500);
        }
    }
}
