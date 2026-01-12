'use client';

import { useState, useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useAuthStore } from '@/lib/store';
import { audioProcessingApi } from '@/lib/api';
import { AudioPlayer } from '@/components/audio-player';
import {
  Upload,
  FileAudio,
  Mic,
  Square,
  Play,
  Trash2,
  Loader2,
  CheckCircle2,
  Download,
  AlertCircle,
  Sparkles,
} from 'lucide-react';

interface ProcessedResult {
  url: string;
  name: string;
  type: 'converted';
}

interface VoiceConvertUploadProps {
  selectedModelId: number;
  modelName: string;
  onConversionComplete?: (result: ProcessedResult) => void;
}

interface AudioFile {
  file: File;
  url: string;
  name: string;
}

interface RecordedAudio {
  blob: Blob;
  url: string;
  name: string;
}

export function VoiceConvertUpload({ selectedModelId, modelName, onConversionComplete }: VoiceConvertUploadProps) {
  const { token } = useAuthStore();
  
  // Tab state
  const [activeTab, setActiveTab] = useState<'upload' | 'record'>('upload');
  
  // Audio state
  const [audioFile, setAudioFile] = useState<AudioFile | null>(null);
  const [recordedAudio, setRecordedAudio] = useState<RecordedAudio | null>(null);
  
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Conversion settings
  const [pitchShift, setPitchShift] = useState(0);
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

  // Recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        setRecordedAudio({
          blob,
          url,
          name: `recording-${Date.now()}.webm`,
        });
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      timerRef.current = setInterval(() => {
        setRecordingTime(t => t + 1);
      }, 1000);
    } catch (err) {
      setError('Microphone access denied. Please allow microphone access to record.');
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

  // Format recording time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Process audio
  const handleProcess = async () => {
    const audioSource = activeTab === 'upload' ? audioFile : recordedAudio;
    if (!audioSource) {
      setError('Please select or record audio first');
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);
    setProcessingStep('Preparing audio...');
    setError('');
    setResults([]);

    try {
      // Convert audio to base64
      const audioBlob = activeTab === 'upload' ? audioFile!.file : recordedAudio!.blob;
      const base64Audio = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result as string;
          const base64 = result.split(',')[1] || result;
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(audioBlob);
      });

      setProcessingStep('Converting voice...');
      setProcessingProgress(30);

      // Call the API
      const response = await audioProcessingApi.process({
        audio: base64Audio,
        mode: 'convert',
        model_id: selectedModelId,
        f0_up_key: pitchShift,
        index_rate: indexRate,
      });

      setProcessingProgress(90);
      setProcessingStep('Finalizing...');

      // Convert response base64 to blob URL
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

      if (!response.converted) {
        throw new Error('No converted audio in response');
      }
      
      const convertedResult: ProcessedResult = {
        url: base64ToUrl(response.converted),
        name: `converted_${modelName.replace(/\s+/g, '_')}.wav`,
        type: 'converted',
      };

      setResults([convertedResult]);
      setProcessingProgress(100);
      setProcessingStep('Complete!');
      
      if (onConversionComplete) {
        onConversionComplete(convertedResult);
      }
    } catch (err: any) {
      setError(err.message || 'Conversion failed');
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
  const hasAudio = activeTab === 'upload' ? !!audioFile : !!recordedAudio;

  return (
    <div className="space-y-6">
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
          onClick={() => setActiveTab('record')}
          className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activeTab === 'record'
              ? 'text-white border-b-2 border-primary-500'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          <Mic className="h-4 w-4 inline mr-2" />
          Record Audio
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
          <div className="text-center space-y-6">
            {recordedAudio && (
              <div className="space-y-4">
                <AudioPlayer
                  url={recordedAudio.url}
                  name={recordedAudio.name}
                  subtitle="Recorded audio"
                  icon={<FileAudio className="h-5 w-5" />}
                  showDownload={false}
                />
                <div className="flex justify-center">
                  <button
                    onClick={() => {
                      URL.revokeObjectURL(recordedAudio.url);
                      setRecordedAudio(null);
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete Recording
                  </button>
                </div>
                <p className="text-sm text-gray-400">
                  Record again to replace
                </p>
              </div>
            )}

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

      {/* Conversion Settings */}
      <div className="bg-gray-800/50 rounded-lg p-4 space-y-4">
        <h4 className="font-medium text-white flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary-400" />
          Conversion Settings
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Pitch Shift: {pitchShift > 0 ? `+${pitchShift}` : pitchShift}
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
              <span>0.0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
          </div>
        </div>
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
            <AudioPlayer
              key={index}
              url={result.url}
              name={result.name}
              subtitle="Converted"
              icon={<Sparkles className="h-5 w-5" />}
              onDownload={() => downloadResult(result)}
              showDownload={true}
            />
          ))}
        </div>
      )}

      {/* Process Button */}
      <button
        onClick={handleProcess}
        disabled={isProcessing || !hasAudio}
        className="w-full py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
      >
        {isProcessing ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <Sparkles className="h-5 w-5" />
            Convert with {modelName}
          </>
        )}
      </button>
    </div>
  );
}
