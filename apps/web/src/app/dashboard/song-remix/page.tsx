'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { audioProcessingApi, youtubeApi, VoiceModel, YouTubeSearchResult } from '@/lib/api';
import { ModelSelector } from '@/components/model-selector';
import { AudioPlayer } from '@/components/audio-player';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  Music,
  FileAudio,
  Play,
  Square,
  Download,
  Sparkles,
  Split,
  Merge,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Volume2,
  Guitar,
  Trash2,
  Search,
  Clock,
} from 'lucide-react';

type Tab = 'upload' | 'youtube';
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

export default function SongRemixPage() {
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
  
  // Voice model selection (for swap mode)
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
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
      setResults([]);
      setError(null);
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
      
      // Convert base64 to blob URL for preview
      const byteCharacters = atob(response.audio);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'audio/wav' });
      
      setYoutubeAudio({
        videoId: response.video_id,
        title: response.title || result.title,
        artist: response.artist || result.artist,
        duration: response.duration || result.duration,
        thumbnail: result.thumbnail,
        audioBase64: response.audio,
        sampleRate: response.sample_rate,
      });
      
      setSearchResults([]);
      setSearchQuery('');
      
    } catch (err: any) {
      setError(err.response?.data?.message || 'Download failed');
    } finally {
      setIsDownloading(false);
      setDownloadingVideoId(null);
    }
  };

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
    
    // Get audio based on active tab
    if (activeTab === 'youtube' && youtubeAudio) {
      base64Audio = youtubeAudio.audioBase64;
      audioName = `${youtubeAudio.artist} - ${youtubeAudio.title}`;
    } else if (activeTab === 'upload' && audioFile) {
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

    if (processingMode === 'swap' && !selectedModelId) {
      setError('Please select a voice model');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResults([]);
    setProcessingProgress(0);

    try {
      setProcessingStep('Preparing audio...');
      setProcessingProgress(10);

      const steps = {
        'split': ['Analyzing frequencies...', 'Separating vocals...', 'Extracting instrumental...'],
        'swap': ['Separating vocals...', 'Converting vocals...', 'Merging tracks...'],
      };

      const currentSteps = steps[processingMode];
      setProcessingStep(currentSteps[0]);
      setProcessingProgress(30);

      const response = await audioProcessingApi.process({
        audio: base64Audio,
        mode: processingMode,
        model_id: selectedModelId || undefined,
        f0_up_key: f0UpKey,
        index_rate: indexRate,
        pitch_shift_all: f0UpKey,
        instrumental_pitch: f0UpKey,
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

  // Check if we have audio based on active tab
  const hasAudio = activeTab === 'upload' ? !!audioFile : !!youtubeAudio;

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-accent-600/20 rounded-lg">
            <Music className="h-6 w-6 text-accent-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Song Remix</h1>
            <p className="text-gray-400">
              Split vocals from instrumentals or swap vocals with AI voice models
            </p>
          </div>
        </div>

        {/* Processing Mode Selection */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <label className="block text-sm font-medium text-gray-300 mb-3">
            Remix Mode
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <button
              onClick={() => setProcessingMode('split')}
              className={`p-4 rounded-lg border-2 transition-all ${
                processingMode === 'split'
                  ? 'border-accent-500 bg-accent-500/10'
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
              }`}
            >
              <Split className={`h-6 w-6 mx-auto mb-2 ${
                processingMode === 'split' ? 'text-accent-400' : 'text-gray-400'
              }`} />
              <div className={`font-medium ${processingMode === 'split' ? 'text-white' : 'text-gray-300'}`}>
                Vocal Splitter
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Separate vocals & instrumentals with UVR5
              </p>
            </button>

            <button
              onClick={() => setProcessingMode('swap')}
              className={`p-4 rounded-lg border-2 transition-all ${
                processingMode === 'swap'
                  ? 'border-accent-500 bg-accent-500/10'
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
              }`}
            >
              <Merge className={`h-6 w-6 mx-auto mb-2 ${
                processingMode === 'swap' ? 'text-accent-400' : 'text-gray-400'
              }`} />
              <div className={`font-medium ${processingMode === 'swap' ? 'text-white' : 'text-gray-300'}`}>
                Voice Swap
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Replace vocals with converted AI voice
              </p>
            </button>
          </div>
        </div>

        {/* Input Source Tabs */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          <div className="flex border-b border-gray-800">
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'upload'
                  ? 'bg-gray-800 text-white border-b-2 border-accent-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <Upload className="h-4 w-4 inline mr-2" />
              Upload File
            </button>
            <button
              onClick={() => setActiveTab('youtube')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'youtube'
                  ? 'bg-gray-800 text-white border-b-2 border-accent-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <Search className="h-4 w-4 inline mr-2" />
              Search Songs
            </button>
          </div>

          <div className="p-6">
            {activeTab === 'upload' ? (
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? 'border-accent-500 bg-accent-500/10'
                    : audioFile
                    ? 'border-green-500 bg-green-500/10'
                    : 'border-gray-700 hover:border-gray-600'
                }`}
              >
                <input {...getInputProps()} />
                {audioFile ? (
                  <div className="space-y-3">
                    <FileAudio className="h-12 w-12 mx-auto text-green-400" />
                    <p className="text-white font-medium">{audioFile.name}</p>
                    <p className="text-sm text-gray-400">Click or drag to replace</p>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        togglePlay(audioFile.url);
                      }}
                      className="inline-flex items-center gap-2 px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
                    >
                      {playingUrl === audioFile.url && isPlaying ? (
                        <Square className="h-4 w-4" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                      {playingUrl === audioFile.url && isPlaying ? 'Stop' : 'Preview'}
                    </button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <Upload className="h-12 w-12 mx-auto text-gray-500" />
                    <p className="text-gray-300">
                      {isDragActive ? 'Drop the file here' : 'Drag & drop an audio file here'}
                    </p>
                    <p className="text-sm text-gray-500">
                      or click to select (MP3, WAV, FLAC, up to 100MB)
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                {/* Search Input */}
                <div className="flex gap-2">
                  <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                      placeholder="Search for songs, artists, or albums..."
                      className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-accent-500 focus:border-transparent"
                    />
                  </div>
                  <button
                    onClick={handleSearch}
                    disabled={isSearching || !searchQuery.trim()}
                    className="px-6 py-3 bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    {isSearching ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Search className="h-5 w-5" />
                    )}
                    Search
                  </button>
                </div>

                {/* Selected Song Display */}
                {youtubeAudio && (
                  <div className="p-4 bg-gray-800 rounded-lg">
                    <div className="flex items-center gap-4">
                      <img 
                        src={youtubeAudio.thumbnail} 
                        alt={youtubeAudio.title}
                        referrerPolicy="no-referrer"
                        className="w-20 h-20 rounded-lg object-cover"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-white font-medium truncate">{youtubeAudio.title}</p>
                        <p className="text-gray-400 text-sm truncate">{youtubeAudio.artist}</p>
                        <p className="text-gray-500 text-xs mt-1">
                          {formatDuration(youtubeAudio.duration)}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="h-5 w-5 text-green-400" />
                        <button
                          onClick={() => setYoutubeAudio(null)}
                          className="p-2 text-gray-400 hover:text-red-400 rounded-lg hover:bg-gray-700"
                        >
                          <Trash2 className="h-5 w-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Search Results */}
                {searchResults.length > 0 && (
                  <div className="space-y-2 max-h-80 overflow-y-auto">
                    <p className="text-sm text-gray-400">Select a song:</p>
                    {searchResults.map((result) => (
                      <button
                        key={result.id}
                        onClick={() => handleYouTubeDownload(result)}
                        disabled={isDownloading}
                        className="w-full p-3 bg-gray-800 hover:bg-gray-700 rounded-lg flex items-center gap-3 transition-colors disabled:opacity-50"
                      >
                        <img 
                          src={result.thumbnail} 
                          alt={result.title}
                          referrerPolicy="no-referrer"
                          className="w-16 h-12 rounded object-cover"
                        />
                        <div className="flex-1 text-left min-w-0">
                          <p className="text-white font-medium truncate text-sm">{result.title}</p>
                          <p className="text-gray-400 text-xs truncate">{result.artist}</p>
                        </div>
                        <div className="flex items-center gap-3 text-gray-500 text-xs">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {formatDuration(result.duration)}
                          </span>
                          {result.is_cached && (
                            <span className="px-2 py-0.5 bg-green-500/20 text-green-400 rounded text-xs">
                              Cached
                            </span>
                          )}
                        </div>
                        {downloadingVideoId === result.id ? (
                          <Loader2 className="h-5 w-5 text-accent-400 animate-spin" />
                        ) : (
                          <Download className="h-5 w-5 text-gray-400" />
                        )}
                      </button>
                    ))}
                  </div>
                )}

                {/* Empty state */}
                {!youtubeAudio && searchResults.length === 0 && (
                  <div className="text-center py-8 text-gray-400">
                    <Music className="h-12 w-12 mx-auto mb-3 text-gray-600" />
                    <p>Search for a song to get started</p>
                    <p className="text-sm text-gray-500 mt-1">
                      Search by artist, song name, or album
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Voice Model Selection (for swap mode) */}
        {processingMode === 'swap' && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 space-y-4">
            <h3 className="font-medium text-white flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-accent-400" />
              Voice Model
            </h3>

            <ModelSelector
              value={selectedModelId}
              onChange={(id) => setSelectedModelId(id)}
              placeholder="Select a voice model..."
              accentColor="accent"
            />

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
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-accent-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Index Rate: {indexRate.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={indexRate}
                  onChange={(e) => setIndexRate(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-accent-500"
                />
              </div>
            </div>
          </div>
        )}

        {/* Process Button */}
        <button
          onClick={handleProcess}
          disabled={isProcessing || !hasAudio || (processingMode === 'swap' && !selectedModelId)}
          className="w-full py-4 bg-accent-600 text-white rounded-lg hover:bg-accent-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              {processingStep} ({processingProgress}%)
            </>
          ) : processingMode === 'split' ? (
            <>
              <Split className="h-5 w-5" />
              Split Vocals & Instrumentals
            </>
          ) : (
            <>
              <Merge className="h-5 w-5" />
              Swap Vocals
            </>
          )}
        </button>

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
            <AlertCircle className="h-5 w-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 space-y-4">
            <h3 className="font-medium text-white flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-400" />
              Results
            </h3>

            <div className="grid gap-4">
              {results.map((result, idx) => (
                <AudioPlayer
                  key={idx}
                  url={result.url}
                  name={result.name}
                  subtitle={result.type.charAt(0).toUpperCase() + result.type.slice(1)}
                  icon={
                    result.type === 'vocals' ? <Volume2 className="h-5 w-5 text-blue-400" /> :
                    result.type === 'instrumental' ? <Guitar className="h-5 w-5 text-purple-400" /> :
                    <Merge className="h-5 w-5 text-green-400" />
                  }
                  onDownload={() => downloadResult(result)}
                  showDownload={true}
                />
              ))}
            </div>
          </div>
        )}

        {/* Hidden audio element */}
        <audio
          ref={audioRef}
          onEnded={() => {
            setIsPlaying(false);
            setPlayingUrl(null);
          }}
        />
      </div>
    </DashboardLayout>
  );
}
