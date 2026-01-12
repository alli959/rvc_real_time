'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useAuthStore } from '@/lib/store';
import { ModelSelector } from '@/components/model-selector';
import { VoiceModel } from '@/lib/api';
import { 
  Loader2, 
  Download, 
  Play, 
  Pause,
  Smile,
  Frown,
  Angry,
  Heart,
  AlertTriangle,
  Sparkles,
  Wand2,
  Music,
  Mic,
  Volume2,
  X,
  Gauge,
  Users,
  Plus
} from 'lucide-react';

// Emotion/Style categories with their tags
const EMOTION_CATEGORIES = {
  positive: {
    name: 'Positive',
    icon: Smile,
    color: 'bg-green-500',
    items: [
      { tag: 'cheerful', label: 'Cheerful', icon: '😊' },
      { tag: 'happy', label: 'Happy', icon: '😃' },
      { tag: 'excited', label: 'Excited', icon: '🎉' },
      { tag: 'friendly', label: 'Friendly', icon: '🤗' },
      { tag: 'hopeful', label: 'Hopeful', icon: '🌟' },
      { tag: 'affectionate', label: 'Affectionate', icon: '💕' },
      { tag: 'gentle', label: 'Gentle', icon: '🌸' },
      { tag: 'lyrical', label: 'Lyrical', icon: '🎵' },
    ]
  },
  negative: {
    name: 'Sad / Serious',
    icon: Frown,
    color: 'bg-blue-500',
    items: [
      { tag: 'sad', label: 'Sad', icon: '😢' },
      { tag: 'depressed', label: 'Depressed', icon: '😞' },
      { tag: 'disgruntled', label: 'Disgruntled', icon: '😒' },
      { tag: 'serious', label: 'Serious', icon: '😐' },
      { tag: 'embarrassed', label: 'Embarrassed', icon: '😳' },
      { tag: 'envious', label: 'Envious', icon: '😤' },
    ]
  },
  angry: {
    name: 'Angry / Intense',
    icon: Angry,
    color: 'bg-red-500',
    items: [
      { tag: 'angry', label: 'Angry', icon: '😠' },
      { tag: 'shouting', label: 'Shouting', icon: '🗣️' },
      { tag: 'unfriendly', label: 'Unfriendly', icon: '😑' },
    ]
  },
  calm: {
    name: 'Calm / Professional',
    icon: Heart,
    color: 'bg-purple-500',
    items: [
      { tag: 'calm', label: 'Calm', icon: '😌' },
      { tag: 'empathetic', label: 'Empathetic', icon: '💗' },
      { tag: 'assistant', label: 'Assistant', icon: '🤖' },
      { tag: 'customerservice', label: 'Customer Service', icon: '📞' },
      { tag: 'newscast', label: 'Newscast', icon: '📺' },
      { tag: 'narration', label: 'Narration', icon: '📖' },
      { tag: 'documentary', label: 'Documentary', icon: '🎬' },
      { tag: 'advertisement', label: 'Advertisement', icon: '📢' },
      { tag: 'poetry', label: 'Poetry', icon: '✨' },
    ]
  },
  fear: {
    name: 'Fear / Surprise',
    icon: AlertTriangle,
    color: 'bg-yellow-500',
    items: [
      { tag: 'fearful', label: 'Fearful', icon: '😨' },
      { tag: 'terrified', label: 'Terrified', icon: '😱' },
      { tag: 'scared', label: 'Scared', icon: '😰' },
      { tag: 'worried', label: 'Worried', icon: '😟' },
      { tag: 'surprised', label: 'Surprised', icon: '😲' },
    ]
  },
  special: {
    name: 'Special Effects',
    icon: Sparkles,
    color: 'bg-cyan-500',
    items: [
      { tag: 'whispering', label: 'Whispering', icon: '🤫' },
      { tag: 'chat', label: 'Chat', icon: '💬' },
    ]
  },
  effects: {
    name: 'Voice Effects',
    icon: Wand2,
    color: 'bg-orange-500',
    items: [
      { tag: 'robot', label: 'Robot', icon: '🤖', isEffect: true },
      { tag: 'spooky', label: 'Spooky', icon: '👻', isEffect: true },
      { tag: 'phone', label: 'Phone', icon: '📱', isEffect: true },
      { tag: 'radio', label: 'Radio', icon: '📻', isEffect: true },
      { tag: 'underwater', label: 'Underwater', icon: '🌊', isEffect: true },
      { tag: 'megaphone', label: 'Megaphone', icon: '📣', isEffect: true },
      { tag: 'evil', label: 'Evil', icon: '😈', isEffect: true },
      { tag: 'dramatic', label: 'Dramatic', icon: '🎭', isEffect: true },
      { tag: 'dreamy', label: 'Dreamy', icon: '💭', isEffect: true },
    ]
  },
  sounds: {
    name: 'Sound Effects',
    icon: Music,
    color: 'bg-pink-500',
    items: [
      { tag: 'laugh', label: 'Laugh', icon: '😂', isSound: true },
      { tag: 'giggle', label: 'Giggle', icon: '🤭', isSound: true },
      { tag: 'evil_laugh', label: 'Evil Laugh', icon: '😈', isSound: true },
      { tag: 'cry', label: 'Cry', icon: '😭', isSound: true },
      { tag: 'sob', label: 'Sob', icon: '😿', isSound: true },
      { tag: 'sigh', label: 'Sigh', icon: '😮‍💨', isSound: true },
      { tag: 'gasp', label: 'Gasp', icon: '😯', isSound: true },
      { tag: 'scream', label: 'Scream', icon: '😱', isSound: true },
      { tag: 'yawn', label: 'Yawn', icon: '🥱', isSound: true },
      { tag: 'cough', label: 'Cough', icon: '😷', isSound: true },
      { tag: 'sneeze', label: 'Sneeze', icon: '🤧', isSound: true },
      { tag: 'growl', label: 'Growl', icon: '😤', isSound: true },
      { tag: 'groan', label: 'Groan', icon: '😩', isSound: true },
      { tag: 'moan', label: 'Moan', icon: '😫', isSound: true },
      { tag: 'hum', label: 'Hum', icon: '🎵', isSound: true },
      { tag: 'chuckle', label: 'Chuckle', icon: '😏', isSound: true },
      { tag: 'whisper', label: 'Whisper', icon: '🤫', isSound: true },
      { tag: 'shush', label: 'Shush', icon: '🤐', isSound: true },
      { tag: 'hiccup', label: 'Hiccup', icon: '🫢', isSound: true },
      { tag: 'burp', label: 'Burp', icon: '😋', isSound: true },
      { tag: 'clearing_throat', label: 'Clear Throat', icon: '🗣️', isSound: true },
      { tag: 'sniff', label: 'Sniff', icon: '👃', isSound: true },
      { tag: 'snore', label: 'Snore', icon: '😴', isSound: true },
      { tag: 'pant', label: 'Pant', icon: '🐕', isSound: true },
      { tag: 'huff', label: 'Huff', icon: '😤', isSound: true },
      { tag: 'gulp', label: 'Gulp', icon: '😰', isSound: true },
      { tag: 'whimper', label: 'Whimper', icon: '🥺', isSound: true },
      { tag: 'wail', label: 'Wail', icon: '😭', isSound: true },
      { tag: 'howl', label: 'Howl', icon: '🐺', isSound: true },
      { tag: 'shriek', label: 'Shriek', icon: '😱', isSound: true },
      { tag: 'yelp', label: 'Yelp', icon: '😣', isSound: true },
      { tag: 'grunt', label: 'Grunt', icon: '💪', isSound: true },
      { tag: 'exclaim', label: 'Exclaim', icon: '❗', isSound: true },
      { tag: 'murmur', label: 'Murmur', icon: '💭', isSound: true },
      { tag: 'mutter', label: 'Mutter', icon: '🙄', isSound: true },
      { tag: 'stammer', label: 'Stammer', icon: '😣', isSound: true },
      { tag: 'stutter', label: 'Stutter', icon: '🔄', isSound: true },
      { tag: 'slur', label: 'Slur', icon: '🍺', isSound: true },
      { tag: 'babble', label: 'Babble', icon: '👶', isSound: true },
      { tag: 'ramble', label: 'Ramble', icon: '🗣️', isSound: true },
      { tag: 'applause', label: 'Applause', icon: '👏', isSound: true },
      { tag: 'cheering', label: 'Cheering', icon: '🎉', isSound: true },
      { tag: 'clap', label: 'Clap', icon: '👐', isSound: true },
      { tag: 'snap', label: 'Snap', icon: '🫰', isSound: true },
      { tag: 'whistle', label: 'Whistle', icon: '😗', isSound: true },
      { tag: 'boo', label: 'Boo', icon: '👎', isSound: true },
      { tag: 'hiss', label: 'Hiss', icon: '🐍', isSound: true },
      { tag: 'shout', label: 'Shout', icon: '📢', isSound: true },
      { tag: 'yell', label: 'Yell', icon: '🗣️', isSound: true },
      { tag: 'cheer', label: 'Cheer', icon: '🙌', isSound: true },
      { tag: 'chant', label: 'Chant', icon: '🎵', isSound: true },
      { tag: 'sing', label: 'Sing', icon: '🎤', isSound: true },
      { tag: 'hum_song', label: 'Hum Song', icon: '🎶', isSound: true },
      { tag: 'beatbox', label: 'Beatbox', icon: '🥁', isSound: true },
      { tag: 'kiss', label: 'Kiss', icon: '💋', isSound: true },
    ]
  }
};

// Effects that can be applied after voice conversion
const VOICE_CONVERSION_EFFECTS = [
  { value: '', label: 'None' },
  { value: 'robot', label: '🤖 Robot' },
  { value: 'spooky', label: '👻 Spooky' },
  { value: 'phone', label: '📱 Phone' },
  { value: 'radio', label: '📻 Radio' },
  { value: 'underwater', label: '🌊 Underwater' },
  { value: 'megaphone', label: '📣 Megaphone' },
  { value: 'evil', label: '😈 Evil' },
  { value: 'dramatic', label: '🎭 Dramatic' },
  { value: 'dreamy', label: '💭 Dreamy' },
  { value: 'scared', label: '😰 Scared/Trembling' },
  { value: 'angry', label: '😠 Angry/Distorted' },
  { value: 'sad', label: '😢 Sad/Muffled' },
  { value: 'excited', label: '🎉 Excited/Bright' },
  { value: 'serious', label: '😐 Serious/Deep' },
  { value: 'whisper', label: '🤫 Whisper' },
];

// Speed presets for quick selection
const SPEED_PRESETS = [
  { value: '-50%', label: 'Very Slow', icon: '🐢' },
  { value: '-30%', label: 'Slow', icon: '🐌' },
  { value: '-15%', label: 'Slightly Slow', icon: '🚶' },
  { value: '+0%', label: 'Normal', icon: '🏃' },
  { value: '+15%', label: 'Slightly Fast', icon: '🚴' },
  { value: '+30%', label: 'Fast', icon: '🏎️' },
  { value: '+50%', label: 'Very Fast', icon: '⚡' },
];

// Interface for include segments (multi-voice)
interface IncludeSegment {
  id: string;
  text: string;
  voiceModelId: number | null;
  voiceModelName: string;
  f0UpKey: number;
  indexRate: number;
}

interface Voice {
  id: string;
  name: string;
  language: string;
  gender: string;
  supports_styles?: boolean;
}

interface TTSGeneratorProps {
  preSelectedModelId?: number;
  hideModelSelector?: boolean;
}

export function TTSGenerator({ preSelectedModelId, hideModelSelector = false }: TTSGeneratorProps = {}) {
  const { token, isHydrated } = useAuthStore();
  
  // Voice and text state
  const [voices, setVoices] = useState<Voice[]>([]);
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  // Language and gender for auto-selection
  const [selectedLanguage, setSelectedLanguage] = useState('');
  const [selectedGender, setSelectedGender] = useState('Male'); // Default to Male
  
  // Speech rate/speed control
  const [speechRate, setSpeechRate] = useState('+0%');
  const [showSpeedModal, setShowSpeedModal] = useState(false);
  const [speedTagRate, setSpeedTagRate] = useState('-30%');
  
  // Bark TTS toggle (native emotions vs faster Edge TTS)
  const [useBark, setUseBark] = useState(true);
  
  // Multi-voice/include segments
  const [includeSegments, setIncludeSegments] = useState<IncludeSegment[]>([]);
  const [showIncludeModal, setShowIncludeModal] = useState(false);
  const [editingIncludeSegment, setEditingIncludeSegment] = useState<IncludeSegment | null>(null);
  
  // Voice conversion options (always enabled)
  // Defaults optimized for maximum voice model similarity
  const [selectedVoiceModel, setSelectedVoiceModel] = useState<number | null>(preSelectedModelId || null);
  const [selectedVoiceModelData, setSelectedVoiceModelData] = useState<VoiceModel | null>(null);
  const [indexRatio, setIndexRatio] = useState(0.75); // Higher = more like target voice model
  const [pitchShift, setPitchShift] = useState(0); // Default 0 (no pitch change)
  const [convertEffect, setConvertEffect] = useState('');
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [showEmotionPicker, setShowEmotionPicker] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('positive');
  const audioRef = useRef<HTMLAudioElement | null>(null);
  
  // Get unique languages - use language field from API
  const languages = useMemo(() => {
    const langSet = new Set<string>();
    voices.forEach(v => {
      if (v.language) langSet.add(v.language);
    });
    return Array.from(langSet).sort();
  }, [voices]);
  
  // Filter voices by language and gender
  const filteredVoices = useMemo(() => {
    let filtered = voices.filter(v => v.language); // Filter out voices without language
    if (selectedLanguage) {
      filtered = filtered.filter(v => v.language === selectedLanguage);
    }
    if (selectedGender) {
      filtered = filtered.filter(v => v.gender?.toLowerCase() === selectedGender.toLowerCase());
    }
    return filtered;
  }, [voices, selectedLanguage, selectedGender]);
  
  // Auto-select best voice based on filters
  const selectedVoice = useMemo(() => {
    if (filteredVoices.length === 0) return null;
    // Prefer voices with style support
    const styled = filteredVoices.find(v => v.supports_styles);
    return styled || filteredVoices[0];
  }, [filteredVoices]);
  
  // Fetch voices on mount - wait for hydration and token
  useEffect(() => {
    if (!isHydrated || !token) return;
    fetchVoices();
    fetchTTSCapabilities();
  }, [isHydrated, token]);
  
  // TTS capabilities state
  const [ttsCapabilities, setTTSCapabilities] = useState<{
    bark_available: boolean;
    recommendation: string;
  } | null>(null);
  
  // Set default language
  useEffect(() => {
    if (languages.length > 0 && !selectedLanguage) {
      // Default to English (US)
      const english = languages.find(lang => lang.includes('English (US)'));
      if (english) setSelectedLanguage(english);
      else setSelectedLanguage(languages[0]);
    }
  }, [languages, selectedLanguage]);

  const fetchVoices = async () => {
    if (!token) return;
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/tts/voices`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setVoices(data.voices || []);
      }
    } catch (err) {
      console.error('Failed to fetch voices:', err);
    }
  };

  const fetchTTSCapabilities = async () => {
    if (!token) return;
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/tts/capabilities`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setTTSCapabilities(data);
      }
    } catch (err) {
      console.error('Failed to fetch TTS capabilities:', err);
    }
  };

  // Insert emotion tag at cursor
  const insertTag = useCallback((tag: string, isSound?: boolean) => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const currentText = text;
    
    let insertText: string;
    let cursorOffset: number;
    
    if (isSound) {
      // Sound effects are self-closing
      insertText = `[${tag}]`;
      cursorOffset = insertText.length;
    } else {
      // Emotions wrap text
      insertText = `[${tag}][/${tag}]`;
      cursorOffset = tag.length + 2; // Position cursor between tags
    }
    
    const newText = currentText.substring(0, start) + insertText + currentText.substring(end);
    setText(newText);
    
    // Set cursor position after React re-render
    setTimeout(() => {
      textarea.focus();
      const newPos = start + cursorOffset;
      textarea.setSelectionRange(newPos, newPos);
    }, 0);
    
    setShowEmotionPicker(false);
  }, [text]);
  
  // Insert speed tag at cursor
  const insertSpeedTag = useCallback((rate: string) => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const currentText = text;
    const selectedText = currentText.substring(start, end);
    
    // Wrap selected text in speed tag, or insert empty tag
    const insertText = selectedText 
      ? `<speed rate="${rate}">${selectedText}</speed>`
      : `<speed rate="${rate}"></speed>`;
    
    const newText = currentText.substring(0, start) + insertText + currentText.substring(end);
    setText(newText);
    
    // Position cursor inside the tag if no text was selected
    setTimeout(() => {
      textarea.focus();
      if (!selectedText) {
        const newPos = start + `<speed rate="${rate}">`.length;
        textarea.setSelectionRange(newPos, newPos);
      } else {
        const newPos = start + insertText.length;
        textarea.setSelectionRange(newPos, newPos);
      }
    }, 0);
    
    setShowSpeedModal(false);
  }, [text]);
  
  // Insert include tag for multi-voice
  const insertIncludeTag = useCallback((segment: IncludeSegment) => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const currentText = text;
    
    // Build the include tag with all parameters
    let attrs = `voice_model_id="${segment.voiceModelId}"`;
    if (segment.f0UpKey !== 0) {
      attrs += ` f0_up_key="${segment.f0UpKey}"`;
    }
    if (segment.indexRate !== 0.75) {
      attrs += ` index_rate="${segment.indexRate}"`;
    }
    
    const insertText = `<include ${attrs}>${segment.text}</include>`;
    
    const newText = currentText.substring(0, start) + insertText + currentText.substring(end);
    setText(newText);
    
    // Track the segment
    setIncludeSegments(prev => [...prev, { ...segment, id: Date.now().toString() }]);
    
    setTimeout(() => {
      textarea.focus();
      const newPos = start + insertText.length;
      textarea.setSelectionRange(newPos, newPos);
    }, 0);
    
    setShowIncludeModal(false);
    setEditingIncludeSegment(null);
  }, [text]);

  const handleGenerate = async () => {
    if (!text.trim() || !selectedVoice) {
      setError('Please enter text and ensure a voice is available');
      return;
    }

    if (!selectedVoiceModel) {
      setError('Please select a voice model for conversion');
      return;
    }

    setLoading(true);
    setError('');
    setAudioUrl('');

    try {
      const payload: any = {
        text: text.trim(),
        voice: selectedVoice.id,
        rate: speechRate, // Include speech rate
        convert_voice: true,
        voice_model_id: selectedVoiceModel,
        index_rate: indexRatio,
        f0_up_key: pitchShift,
        use_bark: useBark, // Use Bark TTS for native emotions (slower but better)
      };
      
      // Add effect to apply after conversion
      if (convertEffect) {
        payload.apply_effects = convertEffect;
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/tts/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        // Try to parse as JSON, but handle HTML error pages
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const data = await response.json();
          throw new Error(data.message || 'Generation failed');
        } else {
          throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
      }

      // Parse JSON response with base64 audio
      const data = await response.json();
      
      if (!data.audio) {
        throw new Error('No audio data received');
      }
      
      // Decode base64 audio to blob
      const binaryString = atob(data.audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const togglePlayback = () => {
    if (!audioRef.current) return;
    
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
    a.download = 'tts-output.mp3';
    a.click();
  };

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.onended = () => setIsPlaying(false);
    }
  }, [audioUrl]);

  // Check if gender should be locked to model's gender
  const modelHasGender = selectedVoiceModelData?.gender ? true : false;

  return (
    <div className="space-y-6">
      {/* TTS Engine Status */}
      {ttsCapabilities && (
        <div className={`flex items-center gap-3 p-3 rounded-lg border ${
          ttsCapabilities.bark_available 
            ? (useBark ? 'bg-green-500/10 border-green-500/30' : 'bg-blue-500/10 border-blue-500/30')
            : 'bg-yellow-500/10 border-yellow-500/30'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            ttsCapabilities.bark_available 
              ? (useBark ? 'bg-green-500' : 'bg-blue-500')
              : 'bg-yellow-500'
          } animate-pulse`} />
          <div className="flex-1">
            <p className={`text-sm font-medium ${
              ttsCapabilities.bark_available 
                ? (useBark ? 'text-green-400' : 'text-blue-400')
                : 'text-yellow-400'
            }`}>
              {ttsCapabilities.bark_available 
                ? (useBark 
                    ? '🎭 Bark TTS - Native Emotions (Slower, ~15-30s)' 
                    : '⚡ Edge TTS - Fast Mode (Audio Effects)')
                : '⚙️ Edge TTS with Audio Processing'}
            </p>
            <p className="text-xs text-gray-400 mt-0.5">
              {ttsCapabilities.bark_available 
                ? (useBark 
                    ? 'Sound effects like [laugh], [sigh], [gasp] will be naturally rendered. First generation may take longer.' 
                    : 'Faster generation with simulated emotions via audio processing.')
                : 'Emotions simulated via pitch/rate changes and audio effects.'}
            </p>
          </div>
          {/* Bark Toggle Switch */}
          {ttsCapabilities.bark_available && (
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={useBark}
                onChange={(e) => setUseBark(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-purple-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
              <span className="ml-2 text-xs text-gray-400">{useBark ? 'Bark' : 'Fast'}</span>
            </label>
          )}
        </div>
      )}

      {/* Language & Gender Selection */}
      <div className={`grid gap-4 ${modelHasGender ? 'grid-cols-1' : 'grid-cols-2'}`}>
        <div>
          <label className="block text-sm font-medium mb-2 text-gray-200">Language</label>
          <select
            className="w-full rounded-lg border border-gray-600 bg-gray-800 px-4 py-2.5 text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
          >
            {languages.length === 0 && (
              <option value="">Loading languages...</option>
            )}
            {languages.map((lang) => (
              <option key={lang} value={lang}>{lang}</option>
            ))}
          </select>
        </div>
        {!modelHasGender && (
          <div>
            <label className="block text-sm font-medium mb-2 text-gray-200">Gender</label>
            <select
              className="w-full rounded-lg border border-gray-600 bg-gray-800 px-4 py-2.5 text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              value={selectedGender}
              onChange={(e) => setSelectedGender(e.target.value)}
            >
              <option value="">Any</option>
              <option value="Female">Female</option>
              <option value="Male">Male</option>
            </select>
          </div>
        )}
      </div>
      
      {/* Auto-selected voice display */}
      {selectedVoice && (
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Mic className="h-4 w-4" />
          <span>Using: <strong className="text-white">{selectedVoice.name}</strong></span>
          {selectedVoice.supports_styles && (
            <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded">Supports Emotions</span>
          )}
        </div>
      )}

      {/* Text Input with Emotion Button */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium text-gray-200">Text to Speak</label>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSpeedModal(true)}
              className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg border border-gray-600 bg-gray-800 hover:bg-gray-700 text-white transition-colors"
            >
              <Gauge className="h-4 w-4" />
              Add Speed Tag
            </button>
            <button
              onClick={() => {
                setEditingIncludeSegment({
                  id: '',
                  text: '',
                  voiceModelId: null,
                  voiceModelName: '',
                  f0UpKey: 0,
                  indexRate: 0.75,
                });
                setShowIncludeModal(true);
              }}
              className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg border border-gray-600 bg-gray-800 hover:bg-gray-700 text-white transition-colors"
            >
              <Users className="h-4 w-4" />
              Add Voice
            </button>
            <button
              onClick={() => setShowEmotionPicker(true)}
              className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg border border-gray-600 bg-gray-800 hover:bg-gray-700 text-white transition-colors"
            >
              <Sparkles className="h-4 w-4" />
              Add Emotion / Effect
            </button>
          </div>
        </div>
        
        <textarea
          ref={textareaRef}
          className="w-full rounded-lg border border-gray-600 bg-gray-800 px-4 py-3 min-h-[150px] font-mono text-sm text-white placeholder-gray-500 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          placeholder="Enter text... Use [emotion]text[/emotion] tags, <speed rate='-30%'>slow text</speed>, or <include voice_model_id='123'>other voice</include>"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        
        <p className="text-xs text-gray-500 mt-1">
          Example: Hello! [cheerful]I&apos;m so happy![/cheerful] [laugh] &lt;speed rate=&quot;-30%&quot;&gt;This is slower.&lt;/speed&gt;
        </p>
      </div>

      {/* Emotion Picker Modal */}
      {showEmotionPicker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
          <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-3xl max-h-[80vh] flex flex-col overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <h2 className="flex items-center gap-2 text-lg font-semibold text-white">
                <Sparkles className="h-5 w-5 text-purple-500" />
                Add Emotion, Style, or Sound Effect
              </h2>
              <button
                onClick={() => setShowEmotionPicker(false)}
                className="p-1 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            {/* Category Tabs */}
            <div className="flex flex-wrap gap-2 p-4 border-b border-gray-700">
              {Object.entries(EMOTION_CATEGORIES).map(([key, category]) => {
                const Icon = category.icon;
                return (
                  <button
                    key={key}
                    onClick={() => setSelectedCategory(key)}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all ${
                      selectedCategory === key
                        ? `${category.color} text-white`
                        : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {category.name}
                  </button>
                );
              })}
            </div>
            
            {/* Items Grid */}
            <div className="flex-1 overflow-y-auto p-4">
              <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-6 gap-2">
                {EMOTION_CATEGORIES[selectedCategory as keyof typeof EMOTION_CATEGORIES]?.items.map((item) => (
                  <button
                    key={item.tag}
                    onClick={() => insertTag(item.tag, (item as any).isSound)}
                    className="flex flex-col items-center gap-1 p-3 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-gray-500 transition-all group"
                  >
                    <span className="text-2xl group-hover:scale-110 transition-transform">
                      {item.icon}
                    </span>
                    <span className="text-xs text-gray-400 group-hover:text-white">
                      {item.label}
                    </span>
                  </button>
                ))}
              </div>
            </div>
            
            {/* Help Text */}
            <div className="border-t border-gray-700 p-4 text-sm text-gray-400">
              {selectedCategory === 'sounds' ? (
                <p>Sound effects are inserted as <code className="bg-gray-800 px-1 rounded">[laugh]</code> - they play as standalone sounds.</p>
              ) : selectedCategory === 'effects' ? (
                <p>Voice effects wrap your text: <code className="bg-gray-800 px-1 rounded">[robot]your text here[/robot]</code></p>
              ) : (
                <p>Emotions wrap your text: <code className="bg-gray-800 px-1 rounded">[happy]your text here[/happy]</code></p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Speed Tag Modal */}
      {showSpeedModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
          <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-md overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <h2 className="flex items-center gap-2 text-lg font-semibold text-white">
                <Gauge className="h-5 w-5 text-cyan-500" />
                Add Speed Control
              </h2>
              <button
                onClick={() => setShowSpeedModal(false)}
                className="p-1 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            {/* Speed Slider */}
            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-200">
                  Speed: {speedTagRate}
                </label>
                <input
                  type="range"
                  value={parseInt(speedTagRate)}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    setSpeedTagRate(val >= 0 ? `+${val}%` : `${val}%`);
                  }}
                  min={-50}
                  max={50}
                  step={5}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>🐢 Very Slow</span>
                  <span>Normal</span>
                  <span>Very Fast ⚡</span>
                </div>
              </div>
              
              {/* Quick Presets */}
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-200">Quick Presets</label>
                <div className="grid grid-cols-4 gap-2">
                  {SPEED_PRESETS.map((preset) => (
                    <button
                      key={preset.value}
                      onClick={() => setSpeedTagRate(preset.value)}
                      className={`flex flex-col items-center gap-1 p-2 rounded-lg border transition-all ${
                        speedTagRate === preset.value
                          ? 'border-cyan-500 bg-cyan-500/20 text-white'
                          : 'border-gray-700 bg-gray-800 hover:bg-gray-700 text-gray-300'
                      }`}
                    >
                      <span className="text-lg">{preset.icon}</span>
                      <span className="text-xs">{preset.label}</span>
                    </button>
                  ))}
                </div>
              </div>
              
              <p className="text-xs text-gray-500">
                Select text first, then add speed tag to wrap it. Or insert an empty tag and type inside.
              </p>
            </div>
            
            {/* Footer */}
            <div className="flex justify-end gap-2 p-4 border-t border-gray-700">
              <button
                onClick={() => setShowSpeedModal(false)}
                className="px-4 py-2 rounded-lg border border-gray-600 bg-gray-800 hover:bg-gray-700 text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => insertSpeedTag(speedTagRate)}
                className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-700 text-white transition-colors"
              >
                Insert Speed Tag
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Include Voice Modal (Multi-voice) */}
      {showIncludeModal && editingIncludeSegment && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
          <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-lg overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <h2 className="flex items-center gap-2 text-lg font-semibold text-white">
                <Users className="h-5 w-5 text-pink-500" />
                Add Another Voice
              </h2>
              <button
                onClick={() => {
                  setShowIncludeModal(false);
                  setEditingIncludeSegment(null);
                }}
                className="p-1 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <div className="p-4 space-y-4">
              {/* Voice Model Selection */}
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-200">
                  Voice Model for this segment
                </label>
                <ModelSelector
                  value={editingIncludeSegment.voiceModelId}
                  onChange={(id, model) => {
                    setEditingIncludeSegment(prev => prev ? {
                      ...prev,
                      voiceModelId: id,
                      voiceModelName: model?.name || ''
                    } : null);
                  }}
                  placeholder="Select a different voice model..."
                  accentColor="primary"
                />
              </div>
              
              {/* Text Input */}
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-200">
                  Text for this voice
                </label>
                <textarea
                  className="w-full rounded-lg border border-gray-600 bg-gray-800 px-4 py-3 min-h-[100px] text-sm text-white placeholder-gray-500 focus:ring-2 focus:ring-pink-500 focus:border-transparent"
                  placeholder="Enter the text this voice should say..."
                  value={editingIncludeSegment.text}
                  onChange={(e) => setEditingIncludeSegment(prev => prev ? { ...prev, text: e.target.value } : null)}
                />
              </div>
              
              {/* Advanced Options */}
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium mb-2 text-gray-200">
                    Pitch Shift: {editingIncludeSegment.f0UpKey > 0 ? '+' : ''}{editingIncludeSegment.f0UpKey} semitones
                  </label>
                  <input
                    type="range"
                    value={editingIncludeSegment.f0UpKey}
                    onChange={(e) => setEditingIncludeSegment(prev => prev ? { ...prev, f0UpKey: parseInt(e.target.value) } : null)}
                    min={-12}
                    max={12}
                    step={1}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2 text-gray-200">
                    Index Ratio: {editingIncludeSegment.indexRate.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    value={editingIncludeSegment.indexRate}
                    onChange={(e) => setEditingIncludeSegment(prev => prev ? { ...prev, indexRate: parseFloat(e.target.value) } : null)}
                    min={0}
                    max={1}
                    step={0.05}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                  />
                </div>
              </div>
              
              <p className="text-xs text-gray-500">
                This will insert: <code className="bg-gray-800 px-1 rounded">&lt;include voice_model_id=&quot;...&quot;&gt;text&lt;/include&gt;</code>
              </p>
            </div>
            
            {/* Footer */}
            <div className="flex justify-end gap-2 p-4 border-t border-gray-700">
              <button
                onClick={() => {
                  setShowIncludeModal(false);
                  setEditingIncludeSegment(null);
                }}
                className="px-4 py-2 rounded-lg border border-gray-600 bg-gray-800 hover:bg-gray-700 text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (editingIncludeSegment.voiceModelId && editingIncludeSegment.text.trim()) {
                    insertIncludeTag(editingIncludeSegment);
                  }
                }}
                disabled={!editingIncludeSegment.voiceModelId || !editingIncludeSegment.text.trim()}
                className="px-4 py-2 rounded-lg bg-pink-600 hover:bg-pink-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white transition-colors"
              >
                Insert Voice Segment
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Speech Speed Control */}
      <div className="border border-gray-700 rounded-lg p-4 space-y-3">
        <h3 className="font-medium text-white flex items-center gap-2">
          <Gauge className="h-5 w-5 text-cyan-400" />
          Speech Speed
        </h3>
        
        <div>
          <label className="block text-sm font-medium mb-2 text-gray-200">
            Base Speed: {speechRate}
          </label>
          <input
            type="range"
            value={parseInt(speechRate)}
            onChange={(e) => {
              const val = parseInt(e.target.value);
              setSpeechRate(val >= 0 ? `+${val}%` : `${val}%`);
            }}
            min={-50}
            max={50}
            step={5}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>🐢 Slower (-50%)</span>
            <span>Normal</span>
            <span>Faster (+50%) ⚡</span>
          </div>
        </div>
        
        <p className="text-xs text-gray-500">
          This affects all text. Use &lt;speed rate=&quot;...&quot;&gt; tags for per-section speed control.
        </p>
      </div>

      {/* Voice Conversion Options - Always visible */}
      <div className="border border-gray-700 rounded-lg p-4 space-y-4">
        <h3 className="font-medium text-white flex items-center gap-2">
          <Wand2 className="h-5 w-5 text-purple-400" />
          Voice Conversion (RVC)
        </h3>
        
        {!hideModelSelector && (
          <ModelSelector
            value={selectedVoiceModel}
            onChange={(id, model) => {
              setSelectedVoiceModel(id);
              setSelectedVoiceModelData(model || null);
              // Auto-set gender from model if it has one
              if (model?.gender) {
                setSelectedGender(model.gender);
              }
            }}
            placeholder="Select a voice model..."
            accentColor="primary"
          />
        )}
        
        <div>
          <label className="block text-sm font-medium mb-2 text-gray-200">
            Voice Effect (applied after conversion)
          </label>
          <select
            className="w-full rounded-lg border border-gray-600 bg-gray-800 px-4 py-2.5 text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            value={convertEffect}
            onChange={(e) => setConvertEffect(e.target.value)}
          >
            {VOICE_CONVERSION_EFFECTS.map((effect) => (
              <option key={effect.value} value={effect.value}>
                {effect.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            These effects are applied AFTER voice conversion for better results
          </p>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-2 text-gray-200">
            Index Ratio: {indexRatio.toFixed(2)}
          </label>
          <input
            type="range"
            value={indexRatio}
            onChange={(e) => setIndexRatio(parseFloat(e.target.value))}
            min={0}
            max={1}
            step={0.05}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Higher = more like voice model personality (0.7-0.85 recommended)
          </p>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-2 text-gray-200">
            Pitch Shift: {pitchShift > 0 ? '+' : ''}{pitchShift} semitones
          </label>
          <input
            type="range"
            value={pitchShift}
            onChange={(e) => setPitchShift(parseInt(e.target.value))}
            min={-12}
            max={12}
            step={1}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Match pitch to voice model: +6 to +12 for higher voices, -6 to -12 for deeper
          </p>
        </div>
      </div>

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={loading || !text.trim() || !selectedVoice}
        className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium transition-colors"
      >
        {loading ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            Generating...
          </>
        ) : (
          <>
            <Volume2 className="h-5 w-5" />
            Generate Speech
          </>
        )}
      </button>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400">
          {error}
        </div>
      )}

      {/* Audio Player */}
      {audioUrl && (
        <div className="p-4 bg-gray-800 rounded-lg space-y-3">
          <audio ref={audioRef} src={audioUrl} />
          <div className="flex items-center gap-3">
            <button
              onClick={togglePlayback}
              className="p-2 rounded-lg border border-gray-600 bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-600 bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            >
              <Download className="h-4 w-4" />
              Download
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
