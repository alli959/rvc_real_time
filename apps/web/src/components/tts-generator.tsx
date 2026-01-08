'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { ttsApi, voiceModelsApi, TTSVoice, TTSStyle, VoiceModel } from '@/lib/api';
import {
  Volume2 as SpeakerWaveIcon,
  Square as StopIcon,
  Play as PlayIcon,
  Download as ArrowDownTrayIcon,
  Sparkles as SparklesIcon,
  Globe,
  User,
  AlertCircle,
  Info,
} from 'lucide-react';

interface TTSGeneratorProps {
  className?: string;
}

interface EnhancedTTSVoice extends TTSVoice {
  supports_styles?: boolean;
}

export function TTSGenerator({ className }: TTSGeneratorProps) {
  // TTS settings
  const [text, setText] = useState('');
  const [language, setLanguage] = useState('English (US)');
  const [gender, setGender] = useState<'male' | 'female'>('male');
  const [voice, setVoice] = useState('en-US-GuyNeural');
  const [style, setStyle] = useState('default');
  const [rate, setRate] = useState(0); // -50 to 50
  const [pitch, setPitch] = useState(0); // -50 to 50

  // Voice conversion settings
  const [useVoiceModel, setUseVoiceModel] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [f0UpKey, setF0UpKey] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);

  // Data
  const [voices, setVoices] = useState<EnhancedTTSVoice[]>([]);
  const [styles, setStyles] = useState<TTSStyle[]>([]);
  const [languages, setLanguages] = useState<string[]>([]);
  const [voiceModels, setVoiceModels] = useState<VoiceModel[]>([]);

  // State
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Load voices and models on mount
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const [voicesData, modelsData] = await Promise.all([
          ttsApi.getVoices(),
          voiceModelsApi.list({ all: true }),
        ]);
        setVoices(voicesData.voices || []);
        setStyles(voicesData.styles || []);
        setLanguages(voicesData.languages || []);
        setVoiceModels(modelsData.data || []);
      } catch (err) {
        console.error('Failed to load TTS data:', err);
        setError('Failed to load voice data');
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  // Cleanup audio URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  // Filter voices by language and gender
  const filteredVoices = useMemo(() => {
    return voices.filter(v => v.language === language && v.gender === gender);
  }, [voices, language, gender]);

  // Select first voice when filters change
  useEffect(() => {
    if (filteredVoices.length > 0 && !filteredVoices.find(v => v.id === voice)) {
      setVoice(filteredVoices[0].id);
    }
  }, [filteredVoices, voice]);

  // Get current voice info
  const currentVoice = useMemo(() => {
    return voices.find(v => v.id === voice);
  }, [voices, voice]);

  // Check if current voice supports styles
  const supportsStyles = currentVoice?.supports_styles ?? false;

  // Reset style if voice doesn't support it
  useEffect(() => {
    if (!supportsStyles && style !== 'default') {
      setStyle('default');
    }
  }, [supportsStyles, style]);

  const handleGenerate = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setGenerating(true);
    setError(null);
    setAudioUrl(null);

    try {
      const response = await ttsApi.generate({
        text: text.trim(),
        voice,
        style,
        rate: rate > 0 ? `+${rate}%` : `${rate}%`,
        pitch: pitch > 0 ? `+${pitch}Hz` : `${pitch}Hz`,
        voice_model_id: useVoiceModel && selectedModelId ? selectedModelId : undefined,
        f0_up_key: useVoiceModel ? f0UpKey : undefined,
        index_rate: useVoiceModel ? indexRate : undefined,
      });

      // Convert base64 to blob URL
      const audioBlob = base64ToBlob(response.audio, 'audio/wav');
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);

      // Auto-play
      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.play();
        setIsPlaying(true);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || err.response?.data?.error || 'Generation failed');
    } finally {
      setGenerating(false);
    }
  };

  const handlePlayPause = () => {
    if (!audioRef.current || !audioUrl) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleDownload = () => {
    if (!audioUrl) return;
    
    const a = document.createElement('a');
    a.href = audioUrl;
    a.download = `tts_${Date.now()}.wav`;
    a.click();
  };

  const base64ToBlob = (base64: string, mimeType: string): Blob => {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Text Input */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Text to Speak
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          maxLength={5000}
          className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
          placeholder="Enter the text you want to convert to speech..."
        />
        <p className="mt-1 text-xs text-gray-500 text-right">
          {text.length} / 5000 characters
        </p>
      </div>

      {/* Language and Voice Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Language */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            <div className="flex items-center gap-2">
              <Globe className="h-4 w-4" />
              Language
            </div>
          </label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            {languages.map((lang) => (
              <option key={lang} value={lang}>
                {lang}
              </option>
            ))}
          </select>
        </div>

        {/* Gender */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            <div className="flex items-center gap-2">
              <User className="h-4 w-4" />
              Voice Type
            </div>
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => setGender('male')}
              className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
                gender === 'male'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Male
            </button>
            <button
              onClick={() => setGender('female')}
              className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
                gender === 'female'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Female
            </button>
          </div>
        </div>

        {/* Voice */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Voice
          </label>
          <select
            value={voice}
            onChange={(e) => setVoice(e.target.value)}
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            {filteredVoices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name} {v.supports_styles ? '✨' : ''}
              </option>
            ))}
          </select>
          {filteredVoices.length === 0 && (
            <p className="mt-1 text-xs text-yellow-400">
              No voices available for this language/gender combination
            </p>
          )}
        </div>
      </div>

      {/* Speaking Style */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Speaking Style
          {!supportsStyles && (
            <span className="ml-2 text-xs text-gray-500">
              (Not available for this voice)
            </span>
          )}
        </label>
        <div className="flex flex-wrap gap-2">
          {styles.slice(0, 10).map((s) => (
            <button
              key={s.id}
              onClick={() => supportsStyles && setStyle(s.id)}
              disabled={!supportsStyles && s.id !== 'default'}
              className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                style === s.id
                  ? 'bg-primary-600 text-white'
                  : supportsStyles || s.id === 'default'
                  ? 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  : 'bg-gray-800/50 text-gray-600 cursor-not-allowed'
              }`}
              title={s.description}
            >
              {s.name || s.id.charAt(0).toUpperCase() + s.id.slice(1).replace(/-/g, ' ')}
            </button>
          ))}
        </div>
        {supportsStyles && style !== 'default' && (
          <p className="mt-2 text-xs text-yellow-400 flex items-center gap-1">
            <Info className="h-3 w-3" />
            Note: Styles require Azure Speech SDK. Edge TTS may not apply all styles.
          </p>
        )}
      </div>

      {/* Rate & Pitch Sliders */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Speed: {rate > 0 ? `+${rate}%` : `${rate}%`}
          </label>
          <input
            type="range"
            min="-50"
            max="50"
            value={rate}
            onChange={(e) => setRate(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Slow</span>
            <span>Normal</span>
            <span>Fast</span>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Pitch: {pitch > 0 ? `+${pitch}Hz` : `${pitch}Hz`}
          </label>
          <input
            type="range"
            min="-50"
            max="50"
            value={pitch}
            onChange={(e) => setPitch(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Low</span>
            <span>Normal</span>
            <span>High</span>
          </div>
        </div>
      </div>

      {/* Voice Conversion Toggle */}
      <div className="border-t border-gray-700 pt-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <SparklesIcon className="h-5 w-5 text-primary-400" />
            <span className="font-medium text-white">Voice Conversion</span>
            <span className="text-xs text-gray-500">(Apply custom voice model)</span>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={useVoiceModel}
              onChange={(e) => setUseVoiceModel(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-500/20 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>

        {useVoiceModel && (
          <div className="space-y-4 bg-gray-800/50 rounded-lg p-4">
            <p className="text-sm text-gray-400">
              Transform the generated speech using a custom voice model (RVC). Great for character voices!
            </p>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Voice Model
              </label>
              <select
                value={selectedModelId || ''}
                onChange={(e) => setSelectedModelId(e.target.value ? parseInt(e.target.value) : null)}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="">Select a voice model...</option>
                {voiceModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Pitch Shift: {f0UpKey > 0 ? `+${f0UpKey}` : f0UpKey}
                </label>
                <input
                  type="range"
                  min="-12"
                  max="12"
                  value={f0UpKey}
                  onChange={(e) => setF0UpKey(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>-12</span>
                  <span>0</span>
                  <span>+12</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Index Rate: {indexRate.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={indexRate * 100}
                  onChange={(e) => setIndexRate(parseInt(e.target.value) / 100)}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0</span>
                  <span>0.5</span>
                  <span>1.0</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-400/10 border border-red-400/20 text-red-400 px-4 py-3 rounded-lg text-sm flex items-center gap-2">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Audio Player */}
      {audioUrl && (
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-4">
            <button
              onClick={handlePlayPause}
              className="p-3 bg-primary-600 rounded-full hover:bg-primary-700 transition-colors"
            >
              {isPlaying ? (
                <StopIcon className="h-6 w-6 text-white" />
              ) : (
                <PlayIcon className="h-6 w-6 text-white" />
              )}
            </button>
            <div className="flex-1">
              <audio
                ref={audioRef}
                src={audioUrl}
                onEnded={() => setIsPlaying(false)}
                onPause={() => setIsPlaying(false)}
                onPlay={() => setIsPlaying(true)}
              />
              <div className="h-1 bg-gray-700 rounded-full">
                <div className="h-full bg-primary-500 rounded-full w-0" />
              </div>
            </div>
            <button
              onClick={handleDownload}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Download"
            >
              <ArrowDownTrayIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={generating || !text.trim() || filteredVoices.length === 0}
        className="w-full py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
      >
        {generating ? (
          <>
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Generating...
          </>
        ) : (
          <>
            <SpeakerWaveIcon className="h-5 w-5" />
            Generate Speech
          </>
        )}
      </button>
    </div>
  );
}
