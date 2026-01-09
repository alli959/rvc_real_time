'use client';

import { useState, useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { youtubeApi, audioProcessingApi, YouTubeSearchResult } from '@/lib/api';
import {
  Upload,
  FileAudio,
  Search,
  Square,
  Play,
  Trash2,
  Loader2,
  CheckCircle2,
  Download,
  AlertCircle,
  Split,
  Merge,
  Volume2,
  Guitar,
  Music,
  Sparkles,
  Clock,
  Eye,
} from 'lucide-react';

type ProcessingMode = 'split' | 'swap';

interface ProcessedResult {
  url: string;
  name: string;
  type: 'vocals' | 'instrumental' | 'converted' | 'mixed';
}

interface VocalSplitterSwapProps {
  selectedModelId?: number;
  modelName?: string;
  onProcessComplete?: (results: ProcessedResult[]) => void;
}

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

export function VocalSplitterSwap({ selectedModelId, modelName, onProcessComplete }: VocalSplitterSwapProps) {
  // Mode state (splitter vs swap)
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('split');
  
  // Input tab state - now 'upload' or 'youtube' (no record for splitter/swap)
  const [activeTab, setActiveTab] = useState<'upload' | 'youtube'>('upload');
  
  // Audio state
  const [audioFile, setAudioFile] = useState<AudioFile | null>(null);
  const [youtubeAudio, setYoutubeAudio] = useState<YouTubeAudio | null>(null);
  
  // YouTube search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<YouTubeSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadingVideoId, setDownloadingVideoId] = useState<string | null>(null);
  
  // Settings
  const [pitchShift, setPitchShift] = useState(0); // Unified pitch shift for both vocals and instrumental
  const [indexRate, setIndexRate] = useState(0.75);
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStep, setProcessingStep] = useState('');
  const [error, setError] = useState('');
  const [results, setResults] = useState<ProcessedResult[]>([]);
  
  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [playingUrl, setPlayingUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  
  // File drop handler
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      if (audioFile?.url) {
        URL.revokeObjectURL(audioFile.url);
      }
      setAudioFile({
        file,
        url: URL.createObjectURL(file),
        name: file.name,
      });
      setError('');
      setResults([]);
    }
  }, [audioFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.flac', '.ogg', '.m4a'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: false,
  });

  // YouTube search function
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    setError('');
    try {
      const response = await youtubeApi.search(searchQuery);
      setSearchResults(response.results);
    } catch (err: any) {
      setError(err.message || 'Search failed');
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  // YouTube download function
  const handleDownload = async (video: YouTubeSearchResult) => {
    setIsDownloading(true);
    setDownloadingVideoId(video.id);
    setError('');
    
    try {
      const data = await youtubeApi.download(video.id);
      setYoutubeAudio({
        videoId: video.id,
        title: video.title,
        artist: video.artist,
        duration: video.duration,
        thumbnail: video.thumbnail,
        audioBase64: data.audio,
        sampleRate: data.sample_rate,
      });
      setSearchResults([]);
      setResults([]);
    } catch (err: any) {
      setError(err.message || 'Failed to download audio');
    } finally {
      setIsDownloading(false);
      setDownloadingVideoId(null);
    }
  };

  // Format duration helper
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Format view count helper
  const formatViews = (views: number) => {
    if (views >= 1000000) return `${(views / 1000000).toFixed(1)}M views`;
    if (views >= 1000) return `${(views / 1000).toFixed(1)}K views`;
    return `${views} views`;
  };

  // Playback functions
  const togglePlay = (url: string) => {
    if (!audioRef.current) {
      audioRef.current = new Audio();
      audioRef.current.onended = () => {
        setIsPlaying(false);
        setPlayingUrl(null);
      };
    }

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

  // Process audio
  const handleProcess = async () => {
    const hasSource = activeTab === 'upload' ? !!audioFile : !!youtubeAudio;
    if (!hasSource) {
      setError('Please select an audio file or search for a song first');
      return;
    }

    if (processingMode === 'swap' && !selectedModelId) {
      setError('Voice model is required for vocal swap');
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);
    setProcessingStep('Preparing audio...');
    setError('');
    setResults([]);

    try {
      // Get base64 audio
      let base64Audio: string;
      let audioName: string;
      
      if (activeTab === 'youtube' && youtubeAudio) {
        base64Audio = youtubeAudio.audioBase64;
        audioName = youtubeAudio.title;
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
        throw new Error('No audio source available');
      }

      const newResults: ProcessedResult[] = [];

      // Helper to convert base64 to blob URL
      const base64ToUrl = (base64: string): string => {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'audio/wav' });
        return URL.createObjectURL(blob);
      };

      if (processingMode === 'split') {
        // Just split vocals and instrumental
        setProcessingStep('Separating vocals and instrumentals...');
        setProcessingProgress(30);

        const response = await audioProcessingApi.process({
          audio: base64Audio,
          mode: 'split',
          pitch_shift_all: pitchShift,
          instrumental_pitch: pitchShift,
        });

        setProcessingProgress(90);
        setProcessingStep('Finalizing...');
        
        if (response.vocals) {
          newResults.push({
            url: base64ToUrl(response.vocals),
            name: `vocals_${audioName.replace(/\.[^/.]+$/, '')}.wav`,
            type: 'vocals',
          });
        }
        if (response.instrumental) {
          newResults.push({
            url: base64ToUrl(response.instrumental),
            name: `instrumental_${audioName.replace(/\.[^/.]+$/, '')}.wav`,
            type: 'instrumental',
          });
        }
      } else {
        // Swap: split, convert vocals, mix back
        setProcessingStep('Processing vocals...');
        setProcessingProgress(20);

        const response = await audioProcessingApi.process({
          audio: base64Audio,
          mode: 'swap',
          model_id: selectedModelId!,
          f0_up_key: pitchShift,
          index_rate: indexRate,
          pitch_shift_all: pitchShift,
          instrumental_pitch: pitchShift,
        });

        setProcessingProgress(90);
        setProcessingStep('Finalizing...');
        
        // Return converted/swapped result
        if (response.converted) {
          newResults.push({
            url: base64ToUrl(response.converted),
            name: `swapped_${modelName?.replace(/\s+/g, '_') || 'model'}_${audioName.replace(/\.[^/.]+$/, '')}.wav`,
            type: 'mixed',
          });
        }
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

  // Check if we have audio based on active tab
  const hasAudio = activeTab === 'upload' ? !!audioFile : !!youtubeAudio;

  const getResultIcon = (type: string) => {
    switch (type) {
      case 'vocals':
        return <Volume2 className="h-5 w-5 text-blue-400" />;
      case 'instrumental':
        return <Guitar className="h-5 w-5 text-purple-400" />;
      case 'converted':
        return <Sparkles className="h-5 w-5 text-primary-400" />;
      case 'mixed':
        return <Music className="h-5 w-5 text-green-400" />;
      default:
        return <Music className="h-5 w-5 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Mode Selection Tabs */}
      <div className="flex rounded-lg bg-gray-800/50 p-1">
        <button
          onClick={() => setProcessingMode('split')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-colors flex items-center justify-center gap-2 ${
            processingMode === 'split'
              ? 'bg-primary-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <Split className="h-4 w-4" />
          Vocal Splitter
        </button>
        <button
          onClick={() => setProcessingMode('swap')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded-md transition-colors flex items-center justify-center gap-2 ${
            processingMode === 'swap'
              ? 'bg-primary-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <Merge className="h-4 w-4" />
          Vocal Swap
        </button>
      </div>

      {/* Mode Description */}
      <p className="text-sm text-gray-400">
        {processingMode === 'split'
          ? 'Separate vocals and instrumentals from any audio file'
          : `Replace vocals with ${modelName || 'selected voice model'} and mix back with instrumental`}
      </p>

      {/* Input Source Tabs */}
      <div className="flex border-b border-gray-700">
        <button
          onClick={() => setActiveTab('upload')}
          className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activeTab === 'upload'
              ? 'text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <Upload className="h-4 w-4 inline mr-2" />
          Upload File
        </button>
        <button
          onClick={() => setActiveTab('youtube')}
          className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activeTab === 'youtube'
              ? 'text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <Search className="h-4 w-4 inline mr-2" />
          Search Songs
        </button>
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === 'upload' ? (
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-primary-500 bg-primary-500/10'
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
                  className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
              <button
                onClick={handleSearch}
                disabled={isSearching || !searchQuery.trim()}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
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
                    className="w-20 h-20 rounded-lg object-cover"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium truncate">{youtubeAudio.title}</p>
                    <p className="text-gray-400 text-sm truncate">{youtubeAudio.artist}</p>
                    <p className="text-gray-500 text-xs mt-1">
                      {formatDuration(youtubeAudio.duration)} • Ready to process
                    </p>
                  </div>
                  <button
                    onClick={() => setYoutubeAudio(null)}
                    className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                  >
                    <Trash2 className="h-5 w-5" />
                  </button>
                </div>
              </div>
            )}

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {searchResults.map((video) => (
                  <div
                    key={video.id}
                    className="flex items-center gap-4 p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-colors cursor-pointer"
                    onClick={() => handleDownload(video)}
                  >
                    <img
                      src={video.thumbnail}
                      alt={video.title}
                      className="w-16 h-12 rounded object-cover"
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-white text-sm font-medium truncate">{video.title}</p>
                      <p className="text-gray-500 text-xs truncate">{video.artist}</p>
                      <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {formatDuration(video.duration)}
                        </span>
                        <span className="flex items-center gap-1">
                          <Eye className="h-3 w-3" />
                          {formatViews(video.view_count)}
                        </span>
                      </div>
                    </div>
                    {isDownloading && downloadingVideoId === video.id ? (
                      <Loader2 className="h-5 w-5 text-primary-400 animate-spin" />
                    ) : (
                      <Download className="h-5 w-5 text-gray-400" />
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* No results state */}
            {searchResults.length === 0 && !youtubeAudio && !isSearching && (
              <div className="text-center py-8 text-gray-500">
                <Search className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>Search for a song to get started</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Pitch Settings */}
      <div className="bg-gray-800/50 rounded-lg p-4 space-y-4">
        <h4 className="font-medium text-white flex items-center gap-2">
          <Music className="h-4 w-4 text-primary-400" />
          Output Pitch Adjustment
        </h4>
        <p className="text-sm text-gray-400">
          Shift the pitch of both vocals and instrumental
        </p>
        
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Pitch Shift: {pitchShift > 0 ? `+${pitchShift}` : pitchShift} semitones
          </label>
          <input
            type="range"
            min="-12"
            max="12"
            value={pitchShift}
            onChange={(e) => setPitchShift(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>-12 (lower)</span>
            <span>0 (original)</span>
            <span>+12 (higher)</span>
          </div>
        </div>

        {/* Additional settings for swap mode */}
        {processingMode === 'swap' && (
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
            <p className="text-xs text-gray-500 mt-1">
              Higher values make the output sound more like the target voice model
            </p>
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

      {/* Processing Progress */}
      {isProcessing && (
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-3 mb-3">
            <Loader2 className="h-5 w-5 text-primary-400 animate-spin" />
            <span className="text-white font-medium">{processingStep}</span>
          </div>
          <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-500 transition-all duration-500"
              style={{ width: `${processingProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-3">
          <h4 className="font-medium text-white flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-green-400" />
            Results
          </h4>

          {results.map((result, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-4 bg-gray-800 rounded-lg"
            >
              <div className="flex items-center gap-3">
                {getResultIcon(result.type)}
                <div>
                  <p className="text-white font-medium">{result.name}</p>
                  <p className="text-xs text-gray-400 capitalize">{result.type}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => togglePlay(result.url)}
                  className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                >
                  {playingUrl === result.url && isPlaying ? (
                    <Square className="h-5 w-5" />
                  ) : (
                    <Play className="h-5 w-5" />
                  )}
                </button>
                <button
                  onClick={() => downloadResult(result)}
                  className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <Download className="h-5 w-5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Process Button */}
      <button
        onClick={handleProcess}
        disabled={isProcessing || !hasAudio || (processingMode === 'swap' && !selectedModelId)}
        className="w-full py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
      >
        {isProcessing ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            Processing...
          </>
        ) : processingMode === 'split' ? (
          <>
            <Split className="h-5 w-5" />
            Split Audio
          </>
        ) : (
          <>
            <Merge className="h-5 w-5" />
            Swap Vocals
          </>
        )}
      </button>
    </div>
  );
}
