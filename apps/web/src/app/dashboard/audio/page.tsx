'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { voiceModelsApi, audioProcessingApi, VoiceModel } from '@/lib/api';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  Mic,
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
} from 'lucide-react';

type Tab = 'upload' | 'record';
type ProcessingMode = 'convert' | 'split' | 'swap';

interface AudioFile {
  file: File;
  url: string;
  name: string;
}

interface ProcessedResult {
  type: 'converted' | 'vocals' | 'instrumental' | 'swapped';
  url: string;
  name: string;
}

export default function AudioProcessingPage() {
  const [activeTab, setActiveTab] = useState<Tab>('upload');
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('convert');
  
  // Audio state
  const [audioFile, setAudioFile] = useState<AudioFile | null>(null);
  const [recordedAudio, setRecordedAudio] = useState<AudioFile | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playingUrl, setPlayingUrl] = useState<string | null>(null);
  
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState<string>('');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [results, setResults] = useState<ProcessedResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // Voice model selection
  const [voiceModels, setVoiceModels] = useState<VoiceModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [f0UpKey, setF0UpKey] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [pitchShiftAll, setPitchShiftAll] = useState(0);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Load voice models
  useEffect(() => {
    const loadModels = async () => {
      setModelsLoading(true);
      setModelsError(null);
      try {
        // Use per_page instead of all parameter for more reliable pagination
        const data = await voiceModelsApi.list({ per_page: 100 });
        const models = data.data || [];
        setVoiceModels(models);
        
        // Auto-select first model if available
        if (models.length > 0 && !selectedModelId) {
          setSelectedModelId(models[0].id);
        }
      } catch (err: any) {
        console.error('Failed to load models:', err);
        setModelsError('Failed to load voice models. Please refresh the page.');
      } finally {
        setModelsLoading(false);
      }
    };
    loadModels();
  }, []);

  // Cleanup URLs on unmount
  useEffect(() => {
    return () => {
      if (audioFile?.url) URL.revokeObjectURL(audioFile.url);
      if (recordedAudio?.url) URL.revokeObjectURL(recordedAudio.url);
      results.forEach(r => URL.revokeObjectURL(r.url));
    };
  }, [audioFile, recordedAudio, results]);

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

  // Recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(audioBlob);
        const file = new File([audioBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        setRecordedAudio({ file, url, name: file.name });
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime(t => t + 1);
      }, 1000);
    } catch (err) {
      setError('Failed to access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

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

  // Process audio
  const handleProcess = async () => {
    const sourceAudio = activeTab === 'upload' ? audioFile : recordedAudio;
    if (!sourceAudio) {
      setError('Please select or record an audio file');
      return;
    }

    if ((processingMode === 'convert' || processingMode === 'swap') && !selectedModelId) {
      setError('Please select a voice model');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResults([]);
    setProcessingProgress(0);

    try {
      // Step 1: Read file as base64
      setProcessingStep('Reading audio file...');
      setProcessingProgress(10);
      
      const base64Audio = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result as string;
          // Remove data URL prefix (e.g., "data:audio/wav;base64,")
          const base64 = result.split(',')[1] || result;
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(sourceAudio.file);
      });

      // Step 2: Call API
      const steps = {
        'convert': ['Converting voice...', 'Finalizing...'],
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
        pitch_shift_all: (processingMode === 'split' || processingMode === 'swap') ? pitchShiftAll : undefined,
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
            name: `vocals_${sourceAudio.name.replace(/\.[^/.]+$/, '')}.wav`,
          });
        }
        if (response.instrumental) {
          newResults.push({
            type: 'instrumental',
            url: base64ToUrl(response.instrumental),
            name: `instrumental_${sourceAudio.name.replace(/\.[^/.]+$/, '')}.wav`,
          });
        }
      } else if (response.converted) {
        newResults.push({
          type: processingMode === 'swap' ? 'swapped' : 'converted',
          url: base64ToUrl(response.converted),
          name: `${processingMode === 'swap' ? 'swapped' : 'converted'}_${sourceAudio.name.replace(/\.[^/.]+$/, '')}.wav`,
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

  // Format recording time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const currentAudio = activeTab === 'upload' ? audioFile : recordedAudio;

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary-600/20 rounded-lg">
            <Music className="h-6 w-6 text-primary-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Audio Processing</h1>
            <p className="text-gray-400">
              Convert voice, split vocals/instruments, or swap vocals in your audio files
            </p>
          </div>
        </div>

        {/* Processing Mode Selection */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <label className="block text-sm font-medium text-gray-300 mb-3">
            Processing Mode
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <button
              onClick={() => setProcessingMode('convert')}
              className={`p-4 rounded-lg border-2 transition-all ${
                processingMode === 'convert'
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
              }`}
            >
              <Sparkles className={`h-6 w-6 mx-auto mb-2 ${
                processingMode === 'convert' ? 'text-primary-400' : 'text-gray-400'
              }`} />
              <div className={`font-medium ${processingMode === 'convert' ? 'text-white' : 'text-gray-300'}`}>
                Voice Convert
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Apply a voice model to the audio
              </p>
            </button>

            <button
              onClick={() => setProcessingMode('split')}
              className={`p-4 rounded-lg border-2 transition-all ${
                processingMode === 'split'
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
              }`}
            >
              <Split className={`h-6 w-6 mx-auto mb-2 ${
                processingMode === 'split' ? 'text-primary-400' : 'text-gray-400'
              }`} />
              <div className={`font-medium ${processingMode === 'split' ? 'text-white' : 'text-gray-300'}`}>
                Vocal Splitter
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Separate vocals & instrumentals
              </p>
            </button>

            <button
              onClick={() => setProcessingMode('swap')}
              className={`p-4 rounded-lg border-2 transition-all ${
                processingMode === 'swap'
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600'
              }`}
            >
              <Merge className={`h-6 w-6 mx-auto mb-2 ${
                processingMode === 'swap' ? 'text-primary-400' : 'text-gray-400'
              }`} />
              <div className={`font-medium ${processingMode === 'swap' ? 'text-white' : 'text-gray-300'}`}>
                Vocal Swap
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Replace vocals with converted voice
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
                  ? 'bg-gray-800 text-white border-b-2 border-primary-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <Upload className="h-4 w-4 inline mr-2" />
              Upload File
            </button>
            <button
              onClick={() => setActiveTab('record')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'record'
                  ? 'bg-gray-800 text-white border-b-2 border-primary-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <Mic className="h-4 w-4 inline mr-2" />
              Record Audio
            </button>
          </div>

          <div className="p-6">
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
              <div className="text-center space-y-6">
                {recordedAudio ? (
                  <div className="space-y-4">
                    <div className="p-4 bg-gray-800 rounded-lg">
                      <FileAudio className="h-8 w-8 mx-auto text-green-400 mb-2" />
                      <p className="text-white font-medium">{recordedAudio.name}</p>
                      <div className="flex justify-center gap-2 mt-3">
                        <button
                          onClick={() => togglePlay(recordedAudio.url)}
                          className="flex items-center gap-2 px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
                        >
                          {playingUrl === recordedAudio.url && isPlaying ? (
                            <Square className="h-4 w-4" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                          {playingUrl === recordedAudio.url && isPlaying ? 'Stop' : 'Play'}
                        </button>
                        <button
                          onClick={() => {
                            URL.revokeObjectURL(recordedAudio.url);
                            setRecordedAudio(null);
                          }}
                          className="flex items-center gap-2 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                        >
                          <Trash2 className="h-4 w-4" />
                          Delete
                        </button>
                      </div>
                    </div>
                    <p className="text-sm text-gray-400">
                      Record again to replace
                    </p>
                  </div>
                ) : null}

                <div className="flex flex-col items-center gap-4">
                  {isRecording && (
                    <div className="text-2xl font-mono text-red-400">
                      {formatTime(recordingTime)}
                    </div>
                  )}
                  
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`p-6 rounded-full transition-all ${
                      isRecording
                        ? 'bg-red-500 hover:bg-red-600 animate-pulse'
                        : 'bg-primary-600 hover:bg-primary-700'
                    }`}
                  >
                    {isRecording ? (
                      <Square className="h-8 w-8 text-white" />
                    ) : (
                      <Mic className="h-8 w-8 text-white" />
                    )}
                  </button>
                  
                  <p className="text-sm text-gray-400">
                    {isRecording ? 'Click to stop recording' : 'Click to start recording'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Voice Model Selection (for convert and swap modes) */}
        {(processingMode === 'convert' || processingMode === 'swap') && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 space-y-4">
            <h3 className="font-medium text-white flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary-400" />
              Voice Model
            </h3>

            {modelsLoading ? (
              <div className="flex items-center gap-2 text-gray-400">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading voice models...
              </div>
            ) : modelsError ? (
              <div className="flex items-center gap-2 text-red-400">
                <AlertCircle className="h-4 w-4" />
                {modelsError}
              </div>
            ) : voiceModels.length === 0 ? (
              <div className="text-gray-400 text-sm">
                No voice models available. Please upload or enable some models first.
              </div>
            ) : (
              <>
                <div>
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
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* Pitch Shift Options (for split and swap modes) */}
        {(processingMode === 'split' || processingMode === 'swap') && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 space-y-4">
            <h3 className="font-medium text-white flex items-center gap-2">
              <Music className="h-5 w-5 text-primary-400" />
              {processingMode === 'split' ? 'Output Pitch Adjustment' : 'Instrumental Pitch Adjustment'}
            </h3>
            <p className="text-sm text-gray-400">
              {processingMode === 'split' 
                ? 'Shift the pitch of both vocals and instrumental in the output'
                : 'Shift the pitch of the instrumental track (vocals are adjusted separately via "Pitch Shift" above)'
              }
            </p>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                {processingMode === 'split' ? 'Pitch Shift (Both)' : 'Instrumental Pitch'}: {pitchShiftAll > 0 ? `+${pitchShiftAll}` : pitchShiftAll} semitones
              </label>
              <input
                type="range"
                min="-12"
                max="12"
                value={pitchShiftAll}
                onChange={(e) => setPitchShiftAll(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>-12 (lower)</span>
                <span>0 (original)</span>
                <span>+12 (higher)</span>
              </div>
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="bg-red-400/10 border border-red-400/20 text-red-400 px-4 py-3 rounded-lg text-sm flex items-center gap-2">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Processing Progress */}
        {isProcessing && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center gap-3 mb-4">
              <Loader2 className="h-5 w-5 text-primary-400 animate-spin" />
              <span className="text-white font-medium">{processingStep}</span>
            </div>
            <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 transition-all duration-500"
                style={{ width: `${processingProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 space-y-4">
            <h3 className="font-medium text-white flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-400" />
              Results
            </h3>

            <div className="space-y-3">
              {results.map((result, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 bg-gray-800 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    {result.type === 'vocals' ? (
                      <Volume2 className="h-5 w-5 text-blue-400" />
                    ) : result.type === 'instrumental' ? (
                      <Guitar className="h-5 w-5 text-purple-400" />
                    ) : (
                      <Sparkles className="h-5 w-5 text-primary-400" />
                    )}
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
          </div>
        )}

        {/* Process Button */}
        <button
          onClick={handleProcess}
          disabled={isProcessing || !currentAudio || ((processingMode === 'convert' || processingMode === 'swap') && !selectedModelId)}
          className="w-full py-4 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium text-lg"
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              {processingMode === 'convert' ? (
                <>
                  <Sparkles className="h-5 w-5" />
                  Convert Voice
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
            </>
          )}
        </button>

        {/* Hidden audio element for playback */}
        <audio
          ref={audioRef}
          onEnded={() => {
            setIsPlaying(false);
            setPlayingUrl(null);
          }}
          className="hidden"
        />
      </div>
    </DashboardLayout>
  );
}
