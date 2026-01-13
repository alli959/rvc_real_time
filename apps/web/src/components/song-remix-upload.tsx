'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { audioProcessingApi, youtubeApi, voiceDetectionApi, YouTubeSearchResult, VoiceModelConfig } from '@/lib/api';
import { AudioPlayer } from '@/components/audio-player';
import {
  Upload,
  FileAudio,
  Play,
  Square,
  Download,
  Split,
  Merge,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Volume2,
  Guitar,
  Sparkles,
  Search,
  Trash2,
  Clock,
  Users,
  User,
  Info,
  Plus,
  Minus,
  ChevronDown,
} from 'lucide-react';

type Tab = 'upload' | 'search';
type ProcessingMode = 'split' | 'swap';

interface AudioFile {
  file: File;
  url: string;
  name: string;
}

interface YouTubeAudio {
  videoId: string;
  title: string;
  artist: string;
  duration: number;
  thumbnail: string;
  audioBase64: string;
  sampleRate: number;
}

interface ProcessedResult {
  type: 'vocals' | 'instrumental' | 'swapped';
  url: string;
  name: string;
}

interface VoiceDetectionInfo {
  voiceCount: number;
  confidence: number;
  method: string;
}

// Model info for multi-voice selection
interface AvailableModel {
  id: number;
  name: string;
}

interface SongRemixUploadProps {
  selectedModelId: number;
  modelName: string;
  availableModels?: AvailableModel[]; // List of available models for multi-voice selection
  onProcessComplete?: (results: ProcessedResult[]) => void;
}

export function SongRemixUpload({ selectedModelId, modelName, availableModels, onProcessComplete }: SongRemixUploadProps) {
  const [activeTab, setActiveTab] = useState<Tab>('upload');
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('split');
  
  // Audio state
  const [audioFile, setAudioFile] = useState<AudioFile | null>(null);
  const [youtubeAudio, setYoutubeAudio] = useState<YouTubeAudio | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playingUrl, setPlayingUrl] = useState<string | null>(null);
  
  // YouTube search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<YouTubeSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadingVideoId, setDownloadingVideoId] = useState<string | null>(null);
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState<string>('');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [results, setResults] = useState<ProcessedResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // Voice detection state
  const [voiceDetection, setVoiceDetection] = useState<VoiceDetectionInfo | null>(null);
  const [isDetectingVoices, setIsDetectingVoices] = useState(false);
  
  // Multi-voice options (for Splitter)
  const [removeAllVocals, setRemoveAllVocals] = useState(false); // false = lead only (default), true = all vocals
  
  // Multi-voice swap settings
  const [voiceCount, setVoiceCount] = useState(1); // Number of voice layers to extract (1-4)
  const [additionalVoices, setAdditionalVoices] = useState<{ modelId: number; f0UpKey: number }[]>([]); // Voice 2, 3, 4
  
  // Conversion settings
  const [f0UpKey, setF0UpKey] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Cleanup URLs on unmount
  useEffect(() => {
    return () => {
      if (audioFile?.url) URL.revokeObjectURL(audioFile.url);
      results.forEach(r => URL.revokeObjectURL(r.url));
    };
  }, [audioFile, results]);

  // File drop handler
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      if (audioFile?.url) URL.revokeObjectURL(audioFile.url);
      const url = URL.createObjectURL(file);
      setAudioFile({ file, url, name: file.name });
      setYoutubeAudio(null);
      setResults([]);
      setError(null);
      setVoiceDetection(null); // Reset voice detection for new audio
    }
  }, [audioFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'],
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024, // 100MB
  });

  // Play/pause audio
  const togglePlay = (url: string) => {
    if (!audioRef.current) return;

    if (playingUrl === url && isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
      setPlayingUrl(null);
    } else {
      audioRef.current.src = url;
      audioRef.current.play();
      setIsPlaying(true);
      setPlayingUrl(url);
    }
  };

  // YouTube search
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    setError(null);
    
    try {
      const response = await youtubeApi.search(searchQuery, 10);
      setSearchResults(response.results);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Search failed');
    } finally {
      setIsSearching(false);
    }
  };

  // YouTube download
  const handleYouTubeDownload = async (result: YouTubeSearchResult) => {
    setIsDownloading(true);
    setDownloadingVideoId(result.id);
    setError(null);
    
    try {
      const response = await youtubeApi.download(result.id, true);
      
      setYoutubeAudio({
        videoId: response.video_id,
        title: response.title || result.title,
        artist: response.artist || result.artist,
        duration: response.duration || result.duration,
        thumbnail: result.thumbnail,
        audioBase64: response.audio,
        sampleRate: response.sample_rate,
      });
      
      setAudioFile(null);
      setSearchResults([]);
      setSearchQuery('');
      setResults([]);
      setVoiceDetection(null); // Reset voice detection for new audio
      
    } catch (err: any) {
      setError(err.response?.data?.message || 'Download failed');
    } finally {
      setIsDownloading(false);
      setDownloadingVideoId(null);
    }
  };

  // Voice detection function
  const detectVoices = useCallback(async (audioBase64: string) => {
    setIsDetectingVoices(true);
    
    try {
      const result = await voiceDetectionApi.detect({
        audio: audioBase64,
        use_vocals_only: true,
        max_voices: 6,
      });

      setVoiceDetection({
        voiceCount: result.voice_count,
        confidence: result.confidence,
        method: result.method,
      });
    } catch (err: any) {
      console.error('Voice detection failed:', err);
      // Don't show error to user, just silently fail
      setVoiceDetection(null);
    } finally {
      setIsDetectingVoices(false);
    }
  }, []);

  // Voice detection disabled - API is not reliable enough yet
  // Users can manually choose "All Vocals" if they know the song has harmonies
  // useEffect(() => {
  //   const runDetection = async () => {
  //     if (youtubeAudio?.audioBase64 && !voiceDetection && !isDetectingVoices) {
  //       detectVoices(youtubeAudio.audioBase64);
  //     } else if (audioFile && !voiceDetection && !isDetectingVoices) {
  //       const reader = new FileReader();
  //       reader.onload = () => {
  //         const result = reader.result as string;
  //         const base64 = result.split(',')[1] || result;
  //         detectVoices(base64);
  //       };
  //       reader.readAsDataURL(audioFile.file);
  //     }
  //   };
  //   runDetection();
  // }, [audioFile, youtubeAudio, voiceDetection, isDetectingVoices, detectVoices]);

  // Format duration
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Process audio
  const handleProcess = async () => {
    let base64Audio: string;
    let audioName: string;
    
    // Get audio based on source
    if (youtubeAudio) {
      base64Audio = youtubeAudio.audioBase64;
      audioName = `${youtubeAudio.artist} - ${youtubeAudio.title}`;
    } else if (audioFile) {
      audioName = audioFile.name;
      base64Audio = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result as string;
          const base64 = result.split(',')[1] || result;
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(audioFile.file);
      });
    } else {
      setError('Please select or search for an audio file');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResults([]);
    setProcessingProgress(0);

    try {
      setProcessingStep('Preparing audio...');
      setProcessingProgress(10);

      // Convert file to base64
      const steps = {
        'split': ['Analyzing frequencies...', 'Separating vocals...', 'Extracting instrumental...'],
        'swap': voiceCount > 1 
          ? ['Extracting clean instrumental...', 'Cascading voice extraction...', 'Converting voices...', 'Merging tracks...']
          : ['Separating vocals...', 'Converting vocals...', 'Merging tracks...'],
      };

      const currentSteps = steps[processingMode];
      setProcessingStep(currentSteps[0]);
      setProcessingProgress(30);

      // Build voice configs for multi-voice swap
      let voiceConfigs: VoiceModelConfig[] | undefined = undefined;
      if (processingMode === 'swap' && voiceCount > 1) {
        voiceConfigs = [
          { model_id: selectedModelId, f0_up_key: f0UpKey }, // Voice 1 (main)
          ...additionalVoices.slice(0, voiceCount - 1).map(v => ({
            model_id: v.modelId,
            f0_up_key: v.f0UpKey,
          })),
        ];
      }

      const response = await audioProcessingApi.process({
        audio: base64Audio,
        mode: processingMode,
        model_id: processingMode === 'swap' ? selectedModelId : undefined,
        f0_up_key: f0UpKey,
        index_rate: indexRate,
        pitch_shift_all: f0UpKey,
        instrumental_pitch: f0UpKey,
        // Only apply extract_all_vocals for split mode - swap mode uses default behavior
        extract_all_vocals: processingMode === 'split' ? removeAllVocals : undefined,
        // Multi-voice swap settings
        voice_count: processingMode === 'swap' ? voiceCount : undefined,
        voice_configs: voiceConfigs,
      });

      setProcessingStep('Processing complete!');
      setProcessingProgress(90);

      // Helper to convert base64 to blob URL
      const base64ToUrl = (base64: string, mimeType: string = 'audio/wav'): string => {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: mimeType });
        return URL.createObjectURL(blob);
      };

      // Build results based on mode
      const newResults: ProcessedResult[] = [];
      
      if (processingMode === 'split') {
        if (response.vocals) {
          newResults.push({
            type: 'vocals',
            url: base64ToUrl(response.vocals),
            name: `vocals_${audioName.replace(/\.[^/.]+$/, '')}.wav`,
          });
        }
        if (response.instrumental) {
          newResults.push({
            type: 'instrumental',
            url: base64ToUrl(response.instrumental),
            name: `instrumental_${audioName.replace(/\.[^/.]+$/, '')}.wav`,
          });
        }
      } else if (response.converted) {
        newResults.push({
          type: 'swapped',
          url: base64ToUrl(response.converted),
          name: `swapped_${audioName.replace(/\.[^/.]+$/, '')}.wav`,
        });
      }

      setResults(newResults);
      setProcessingProgress(100);
      setProcessingStep('Complete!');
      
      if (onProcessComplete) {
        onProcessComplete(newResults);
      }
    } catch (err: any) {
      setError(err.message || 'Processing failed');
    } finally {
      setIsProcessing(false);
    }
  };

  // Download result
  const downloadResult = (result: ProcessedResult) => {
    const a = document.createElement('a');
    a.href = result.url;
    a.download = result.name;
    a.click();
  };

  // Check if we have audio
  const hasAudio = !!audioFile || !!youtubeAudio;

  return (
    <div className="space-y-4">
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        onEnded={() => {
          setIsPlaying(false);
          setPlayingUrl(null);
        }}
      />

      {/* Processing Mode Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Remix Mode
        </label>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setProcessingMode('split')}
            className={`p-3 rounded-lg border-2 transition-all ${
              processingMode === 'split'
                ? 'border-accent-500 bg-accent-500/10'
                : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
            }`}
          >
            <Split className={`h-5 w-5 mx-auto mb-1 ${
              processingMode === 'split' ? 'text-accent-400' : 'text-gray-400'
            }`} />
            <div className={`text-sm font-medium ${processingMode === 'split' ? 'text-white' : 'text-gray-300'}`}>
              Vocal Splitter
            </div>
            <p className="text-xs text-gray-500 mt-0.5 hidden sm:block">
              Separate vocals & instrumentals
            </p>
          </button>

          <button
            onClick={() => setProcessingMode('swap')}
            className={`p-3 rounded-lg border-2 transition-all ${
              processingMode === 'swap'
                ? 'border-accent-500 bg-accent-500/10'
                : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
            }`}
          >
            <Merge className={`h-5 w-5 mx-auto mb-1 ${
              processingMode === 'swap' ? 'text-accent-400' : 'text-gray-400'
            }`} />
            <div className={`text-sm font-medium ${processingMode === 'swap' ? 'text-white' : 'text-gray-300'}`}>
              Voice Swap
            </div>
            <p className="text-xs text-gray-500 mt-0.5 hidden sm:block">
              Replace with {modelName}
            </p>
          </button>
        </div>
      </div>

      {/* Input Source Tabs */}
      <div className="border border-gray-700 rounded-lg overflow-hidden">
        <div className="flex border-b border-gray-700">
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex-1 px-3 py-2 text-sm font-medium transition-colors flex items-center justify-center gap-1.5 ${
              activeTab === 'upload'
                ? 'bg-gray-800 text-white border-b-2 border-accent-500'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <Upload className="h-4 w-4" />
            <span>Upload File</span>
          </button>
          <button
            onClick={() => setActiveTab('search')}
            className={`flex-1 px-3 py-2 text-sm font-medium transition-colors flex items-center justify-center gap-1.5 ${
              activeTab === 'search'
                ? 'bg-gray-800 text-white border-b-2 border-accent-500'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <Search className="h-4 w-4" />
            <span>Search Songs</span>
          </button>
        </div>

        <div className="p-4">
          {activeTab === 'upload' ? (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-accent-500 bg-accent-500/10'
                  : audioFile
                  ? 'border-green-500 bg-green-500/10'
                  : 'border-gray-700 hover:border-gray-600'
              }`}
            >
              <input {...getInputProps()} />
              {audioFile ? (
                <div className="space-y-2">
                  <FileAudio className="h-10 w-10 mx-auto text-green-400" />
                  <p className="text-white font-medium text-sm truncate">{audioFile.name}</p>
                  <p className="text-xs text-gray-400">Click or drag to replace</p>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      togglePlay(audioFile.url);
                    }}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
                  >
                    {playingUrl === audioFile.url && isPlaying ? (
                      <Square className="h-3 w-3" />
                    ) : (
                      <Play className="h-3 w-3" />
                    )}
                    {playingUrl === audioFile.url && isPlaying ? 'Stop' : 'Preview'}
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload className="h-10 w-10 mx-auto text-gray-500" />
                  <p className="text-gray-300 text-sm">
                    {isDragActive ? 'Drop the file here' : 'Drag & drop an audio file here'}
                  </p>
                  <p className="text-xs text-gray-500">
                    or click to select (MP3, WAV, FLAC, up to 100MB)
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {/* Search Input */}
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Search for songs..."
                    className="w-full pl-9 pr-3 py-2 text-sm bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-accent-500 focus:border-transparent"
                  />
                </div>
                <button
                  onClick={handleSearch}
                  disabled={isSearching || !searchQuery.trim()}
                  className="px-4 py-2 text-sm bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
                >
                  {isSearching ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Search className="h-4 w-4" />
                  )}
                  Search
                </button>
              </div>

              {/* Selected Song Display */}
              {youtubeAudio && (
                <div className="p-3 bg-gray-800 rounded-lg">
                  <div className="flex items-center gap-3">
                    <img 
                      src={youtubeAudio.thumbnail} 
                      alt={youtubeAudio.title}
                      referrerPolicy="no-referrer"
                      className="w-14 h-14 rounded object-cover"
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-white font-medium text-sm truncate">{youtubeAudio.title}</p>
                      <p className="text-gray-400 text-xs truncate">{youtubeAudio.artist}</p>
                      <p className="text-gray-500 text-xs mt-0.5">
                        {formatDuration(youtubeAudio.duration)}
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <CheckCircle2 className="h-4 w-4 text-green-400" />
                      <button
                        onClick={() => setYoutubeAudio(null)}
                        className="p-1.5 text-gray-400 hover:text-red-400 rounded hover:bg-gray-700"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Search Results */}
              {searchResults.length > 0 && (
                <div className="space-y-1.5 max-h-48 overflow-y-auto">
                  <p className="text-xs text-gray-400">Select a song:</p>
                  {searchResults.map((result) => (
                    <button
                      key={result.id}
                      onClick={() => handleYouTubeDownload(result)}
                      disabled={isDownloading}
                      className="w-full p-2 bg-gray-800 hover:bg-gray-700 rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50 text-left"
                    >
                      <img 
                        src={result.thumbnail} 
                        alt={result.title}
                        referrerPolicy="no-referrer"
                        className="w-10 h-10 rounded object-cover flex-shrink-0"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-white text-sm font-medium truncate">{result.title}</p>
                        <p className="text-gray-400 text-xs truncate">{result.artist}</p>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className="text-xs text-gray-500 flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {formatDuration(result.duration)}
                        </span>
                        {downloadingVideoId === result.id ? (
                          <Loader2 className="h-4 w-4 animate-spin text-accent-400" />
                        ) : (
                          <Download className="h-4 w-4 text-gray-400" />
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {/* Empty state */}
              {!youtubeAudio && searchResults.length === 0 && (
                <div className="text-center py-4 text-gray-500 text-sm">
                  Search for a song to get started
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Vocal Extraction Options - Always show when audio is loaded */}
      {(audioFile || youtubeAudio) && processingMode === 'split' && (
        <div className="bg-gray-800/50 rounded-lg p-3 space-y-3">
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-accent-400" />
            <span className="text-sm font-medium text-white">Vocal Extraction Mode</span>
          </div>

          <p className="text-xs text-gray-400">
            Choose what to separate from the instrumentals:
          </p>
          
          <div className="flex gap-2">
            <button
              onClick={() => setRemoveAllVocals(true)}
              className={`flex-1 px-3 py-2 text-sm rounded-lg border transition-colors ${
                removeAllVocals
                  ? 'bg-accent-600/20 border-accent-500 text-accent-400'
                  : 'border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              <span className="font-medium">All Vocals</span>
              <p className="text-[10px] opacity-70 mt-0.5">
                Remove lead + backup vocals
              </p>
            </button>
            <button
              onClick={() => setRemoveAllVocals(false)}
              className={`flex-1 px-3 py-2 text-sm rounded-lg border transition-colors ${
                !removeAllVocals
                  ? 'bg-accent-600/20 border-accent-500 text-accent-400'
                  : 'border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              <span className="font-medium">Lead Vocal Only</span>
              <p className="text-[10px] opacity-70 mt-0.5">
                Keep backup/harmony vocals
              </p>
            </button>
          </div>

          {/* Voice detection info if available */}
          {isDetectingVoices && (
            <div className="flex items-center gap-2 text-gray-400 text-xs">
              <Loader2 className="h-3 w-3 animate-spin text-accent-400" />
              <span>Analyzing for multiple voices...</span>
            </div>
          )}
          {voiceDetection && (
            <div className="flex items-center gap-2 text-xs">
              {voiceDetection.voiceCount === 1 ? (
                <>
                  <User className="h-3 w-3 text-blue-400" />
                  <span className="text-gray-400">Single voice detected</span>
                </>
              ) : (
                <>
                  <Users className="h-3 w-3 text-purple-400" />
                  <span className="text-purple-400">{voiceDetection.voiceCount} voices detected - &quot;All Vocals&quot; recommended</span>
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* Voice Swap Info */}
      {(audioFile || youtubeAudio) && processingMode === 'swap' && (
        <div className="bg-gray-800/50 rounded-lg p-3 space-y-2">
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-accent-400" />
            <span className="text-sm font-medium text-white">Voice Conversion</span>
          </div>
          
          {isDetectingVoices && (
            <div className="flex items-center gap-2 text-gray-400 text-xs">
              <Loader2 className="h-3 w-3 animate-spin text-accent-400" />
              <span>Analyzing voices...</span>
            </div>
          )}
          
          {voiceDetection && voiceDetection.voiceCount > 1 && (
            <p className="text-xs text-amber-400/80 flex items-center gap-1">
              <Info className="h-3 w-3" />
              Multiple voices detected - all vocals will be converted to {modelName}
            </p>
          )}
          
          {(!voiceDetection || voiceDetection.voiceCount === 1) && !isDetectingVoices && (
            <p className="text-xs text-gray-400">
              All vocals will be converted to the selected voice model
            </p>
          )}
        </div>
      )}

      {/* Multi-Voice Configuration (for swap mode) */}
      {(audioFile || youtubeAudio) && processingMode === 'swap' && availableModels && availableModels.length > 1 && (
        <div className="bg-gray-800/50 rounded-lg p-3 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4 text-purple-400" />
              <span className="text-sm font-medium text-white">Voice Layers</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  if (voiceCount > 1) {
                    setVoiceCount(voiceCount - 1);
                    setAdditionalVoices(additionalVoices.slice(0, voiceCount - 2));
                  }
                }}
                disabled={voiceCount <= 1}
                className="p-1 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Minus className="h-3 w-3" />
              </button>
              <span className="text-sm font-medium text-white w-6 text-center">{voiceCount}</span>
              <button
                onClick={() => {
                  if (voiceCount < 4 && additionalVoices.length < 3) {
                    setVoiceCount(voiceCount + 1);
                    // Add a new voice with a default model (not the main one)
                    const otherModels = availableModels.filter(m => m.id !== selectedModelId);
                    const defaultModel = otherModels[additionalVoices.length % otherModels.length] || availableModels[0];
                    setAdditionalVoices([...additionalVoices, { modelId: defaultModel.id, f0UpKey: 0 }]);
                  }
                }}
                disabled={voiceCount >= 4 || !availableModels || availableModels.length <= 1}
                className="p-1 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Plus className="h-3 w-3" />
              </button>
            </div>
          </div>

          {voiceCount === 1 ? (
            <p className="text-xs text-gray-400">
              Single voice mode: Main vocal → Convert to <span className="text-accent-400">{modelName}</span> → Mix with instrumental
            </p>
          ) : (
            <div className="space-y-2">
              <p className="text-xs text-gray-500">
                Multi-voice cascading: Extract each voice layer separately, convert with different models, mix with clean instrumental
              </p>
              
              {/* Voice 1 (always the selected model) */}
              <div className="flex items-center gap-2 bg-gray-700/50 rounded p-2">
                <span className="text-xs text-gray-400 w-16">Voice 1:</span>
                <span className="text-xs text-accent-400 flex-1">{modelName}</span>
                <span className="text-xs text-gray-500">Lead Vocal</span>
              </div>
              
              {/* Additional voices */}
              {additionalVoices.slice(0, voiceCount - 1).map((voice, index) => (
                <div key={index} className="flex items-center gap-2 bg-gray-700/50 rounded p-2">
                  <span className="text-xs text-gray-400 w-16">Voice {index + 2}:</span>
                  <select
                    value={voice.modelId}
                    onChange={(e) => {
                      const newVoices = [...additionalVoices];
                      newVoices[index] = { ...newVoices[index], modelId: parseInt(e.target.value) };
                      setAdditionalVoices(newVoices);
                    }}
                    className="flex-1 text-xs bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white"
                  >
                    {availableModels.map(m => (
                      <option key={m.id} value={m.id}>{m.name}</option>
                    ))}
                  </select>
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-gray-500">Pitch:</span>
                    <input
                      type="number"
                      min="-12"
                      max="12"
                      value={voice.f0UpKey}
                      onChange={(e) => {
                        const newVoices = [...additionalVoices];
                        newVoices[index] = { ...newVoices[index], f0UpKey: parseInt(e.target.value) || 0 };
                        setAdditionalVoices(newVoices);
                      }}
                      className="w-12 text-xs bg-gray-700 border border-gray-600 rounded px-1 py-1 text-white text-center"
                    />
                  </div>
                </div>
              ))}
              
              <p className="text-xs text-amber-400/70 flex items-center gap-1">
                <Info className="h-3 w-3" />
                Processing time increases with each additional voice
              </p>
            </div>
          )}
        </div>
      )}

      {/* Conversion Settings (for swap mode) */}
      {processingMode === 'swap' && (
        <div className="bg-gray-800/50 rounded-lg p-3 space-y-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-accent-400" />
            <span className="text-sm font-medium text-white">Conversion Settings</span>
          </div>
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-400">Pitch Shift: {f0UpKey}</span>
              </div>
              <input
                type="range"
                min="-12"
                max="12"
                value={f0UpKey}
                onChange={(e) => setF0UpKey(parseInt(e.target.value))}
                className="w-full accent-accent-500"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>-12</span>
                <span>0</span>
                <span>+12</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-400">Index Rate: {indexRate}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={indexRate}
                onChange={(e) => setIndexRate(parseFloat(e.target.value))}
                className="w-full accent-accent-500"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>0.0</span>
                <span>0.5</span>
                <span>1.0</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Process Button */}
      <button
        onClick={handleProcess}
        disabled={!hasAudio || isProcessing}
        className="w-full py-2.5 bg-accent-600 text-white rounded-lg font-medium hover:bg-accent-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-sm"
      >
        {isProcessing ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            {processingStep}
          </>
        ) : (
          <>
            <Split className="h-4 w-4" />
            {processingMode === 'split' 
              ? 'Split Vocals & Instrumentals' 
              : `Swap with ${modelName}`}
          </>
        )}
      </button>

      {/* Progress Bar */}
      {isProcessing && (
        <div className="space-y-1">
          <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-accent-500 transition-all duration-300"
              style={{ width: `${processingProgress}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 text-center">{processingStep}</p>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-green-400">
            <CheckCircle2 className="h-4 w-4" />
            <span className="text-sm font-medium">Processing Complete!</span>
          </div>

          <div className="space-y-2">
            {results.map((result, index) => (
              <AudioPlayer
                key={index}
                url={result.url}
                name={result.name}
                subtitle={result.type.charAt(0).toUpperCase() + result.type.slice(1)}
                icon={
                  result.type === 'vocals' ? <Volume2 className="h-5 w-5 text-purple-400" /> :
                  result.type === 'instrumental' ? <Guitar className="h-5 w-5 text-blue-400" /> :
                  <Volume2 className="h-5 w-5 text-accent-400" />
                }
                onDownload={() => downloadResult(result)}
                showDownload={true}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
