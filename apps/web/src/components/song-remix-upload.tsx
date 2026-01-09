'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { audioProcessingApi, youtubeApi, YouTubeSearchResult } from '@/lib/api';
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

interface SongRemixUploadProps {
  selectedModelId: number;
  modelName: string;
  onProcessComplete?: (results: ProcessedResult[]) => void;
}

export function SongRemixUpload({ selectedModelId, modelName, onProcessComplete }: SongRemixUploadProps) {
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
        'swap': ['Separating vocals...', 'Converting vocals...', 'Merging tracks...'],
      };

      const currentSteps = steps[processingMode];
      setProcessingStep(currentSteps[0]);
      setProcessingProgress(30);

      const response = await audioProcessingApi.process({
        audio: base64Audio,
        mode: processingMode,
        model_id: processingMode === 'swap' ? selectedModelId : undefined,
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
              <div
                key={index}
                className="p-3 bg-gray-800 rounded-lg flex items-center gap-3"
              >
                <div className="p-1.5 rounded bg-gray-700">
                  {result.type === 'vocals' && <Volume2 className="h-4 w-4 text-purple-400" />}
                  {result.type === 'instrumental' && <Guitar className="h-4 w-4 text-blue-400" />}
                  {result.type === 'swapped' && <Volume2 className="h-4 w-4 text-accent-400" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white truncate">{result.name}</p>
                  <p className="text-xs text-gray-400 capitalize">{result.type}</p>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => togglePlay(result.url)}
                    className="p-1.5 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
                  >
                    {playingUrl === result.url && isPlaying ? (
                      <Square className="h-4 w-4" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                  </button>
                  <button
                    onClick={() => downloadResult(result)}
                    className="p-1.5 bg-accent-600 rounded hover:bg-accent-700 transition-colors"
                  >
                    <Download className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
