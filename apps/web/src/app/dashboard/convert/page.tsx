'use client';

import { useState, useCallback, useRef, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { useDropzone } from 'react-dropzone';
import { useQuery } from '@tanstack/react-query';
import {
  Mic2,
  Upload,
  Download,
  ChevronLeft,
  Sliders,
  Loader2,
  FileAudio,
  MessageSquare,
  Radio,
  Square,
  AlertCircle,
  CheckCircle2,
  Volume2,
  X,
} from 'lucide-react';
import Link from 'next/link';
import { voiceModelsApi, SystemVoiceModel, ttsApi, TTSVoice } from '@/lib/api';

// Tab types
type TabType = 'file' | 'tts' | 'speech';

interface ConversionSettings {
  pitch: number;
  indexRate: number;
  filterRadius: number;
  rmsMixRate: number;
  protect: number;
}

const defaultSettings: ConversionSettings = {
  pitch: 0,
  indexRate: 0.75,
  filterRadius: 3,
  rmsMixRate: 0.25,
  protect: 0.33,
};

// Voice Engine WebSocket URL
const VOICE_ENGINE_URL = process.env.NEXT_PUBLIC_VOICE_ENGINE_WS_URL || 'ws://localhost:8765';

function ConvertPageContent() {
  const searchParams = useSearchParams();
  const modelSlug = searchParams.get('model');

  // State
  const [activeTab, setActiveTab] = useState<TabType>('file');
  const [settings, setSettings] = useState<ConversionSettings>(defaultSettings);
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // File upload state
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);

  // TTS state
  const [ttsText, setTtsText] = useState('');
  const [ttsVoice, setTtsVoice] = useState('en-US-GuyNeural');
  const [ttsRate, setTtsRate] = useState(0);
  const [ttsPitch, setTtsPitch] = useState(0);
  const [ttsVoices, setTtsVoices] = useState<TTSVoice[]>([]);
  const [ttsLanguages, setTtsLanguages] = useState<string[]>([]);
  const [ttsLanguage, setTtsLanguage] = useState('English (US)');
  const [ttsGender, setTtsGender] = useState<'male' | 'female'>('male');

  // Speech-to-speech state
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState<Blob | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);

  // Load TTS voices
  useEffect(() => {
    const loadVoices = async () => {
      try {
        const data = await ttsApi.getVoices();
        setTtsVoices(data.voices || []);
        setTtsLanguages(data.languages || []);
      } catch (err) {
        console.error('Failed to load TTS voices:', err);
      }
    };
    loadVoices();
  }, []);

  // Filter TTS voices by language and gender
  const filteredTtsVoices = ttsVoices.filter(
    v => v.language === ttsLanguage && v.gender === ttsGender
  );

  // Select first voice when filters change
  useEffect(() => {
    if (filteredTtsVoices.length > 0 && !filteredTtsVoices.find(v => v.id === ttsVoice)) {
      setTtsVoice(filteredTtsVoices[0].id);
    }
  }, [filteredTtsVoices, ttsVoice]);

  // Fetch selected model
  const { data: modelData, isLoading: isLoadingModel, error: modelError } = useQuery({
    queryKey: ['voice-model', modelSlug],
    queryFn: () => voiceModelsApi.get(modelSlug!),
    enabled: !!modelSlug,
  });

  // API returns { model: {...} } not { data: {...} }
  const selectedModel: SystemVoiceModel | null = modelData?.model || null;

  // Debug log
  useEffect(() => {
    if (modelSlug) {
      console.log('Model slug from URL:', modelSlug);
      console.log('Model data response:', modelData);
      console.log('Selected model:', selectedModel);
    }
  }, [modelSlug, modelData, selectedModel]);

  // Connect to voice engine WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(VOICE_ENGINE_URL);

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
      console.log('Connected to voice engine');

      // Send model selection if available
      if (selectedModel) {
        ws.send(JSON.stringify({
          type: 'load_model',
          model_path: selectedModel.model_path,
          index_path: selectedModel.index_path,
        }));
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from voice engine');
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setError('Failed to connect to voice engine. Make sure the service is running.');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (e) {
        console.error('Failed to parse message:', e);
      }
    };

    wsRef.current = ws;
  }, [selectedModel]);

  // Handle incoming WebSocket messages
  const handleWebSocketMessage = (data: any) => {
    switch (data.type) {
      case 'model_loaded':
        setSuccess(`Model "${data.model_name}" loaded successfully`);
        setTimeout(() => setSuccess(null), 3000);
        break;

      case 'audio':
        // Received processed audio
        if (data.data) {
          const audioBytes = base64ToArrayBuffer(data.data);
          const blob = new Blob([audioBytes], { type: 'audio/wav' });
          const url = URL.createObjectURL(blob);
          setOutputUrl(url);
          setIsProcessing(false);
          setSuccess('Audio converted successfully!');
          setTimeout(() => setSuccess(null), 3000);
        }
        break;

      case 'tts_audio':
        // Received TTS audio
        if (data.data) {
          const audioBytes = base64ToArrayBuffer(data.data);
          const blob = new Blob([audioBytes], { type: 'audio/wav' });
          const url = URL.createObjectURL(blob);
          setOutputUrl(url);
          setIsProcessing(false);
          setSuccess('Speech generated successfully!');
          setTimeout(() => setSuccess(null), 3000);
        }
        break;

      case 'error':
        setError(data.message || 'An error occurred');
        setIsProcessing(false);
        break;

      case 'ack':
        // Acknowledgment - processing continues
        break;

      case 'pong':
        // Keep-alive response
        break;

      default:
        console.log('Unknown message type:', data.type);
    }
  };

  // Effect: Connect when model is selected
  useEffect(() => {
    if (selectedModel) {
      connectWebSocket();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [selectedModel, connectWebSocket]);

  // Effect: Update model on voice engine when it changes
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN && selectedModel) {
      wsRef.current.send(JSON.stringify({
        type: 'load_model',
        model_path: selectedModel.model_path,
        index_path: selectedModel.index_path,
      }));
    }
  }, [selectedModel]);

  // File upload handlers
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setAudioFile(file);
      setAudioUrl(URL.createObjectURL(file));
      setOutputUrl(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  // Convert uploaded file
  const handleFileConvert = async () => {
    if (!audioFile || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Please upload a file and ensure connection to voice engine');
      return;
    }

    setIsProcessing(true);
    setOutputUrl(null);
    setError(null);

    try {
      // Read file as array buffer
      const arrayBuffer = await audioFile.arrayBuffer();
      const base64 = arrayBufferToBase64(arrayBuffer);

      // Send to voice engine
      wsRef.current.send(JSON.stringify({
        type: 'audio',
        data: base64,
        final: true,
        settings: {
          f0_up_key: settings.pitch,
          index_rate: settings.indexRate,
          filter_radius: settings.filterRadius,
          rms_mix_rate: settings.rmsMixRate,
          protect: settings.protect,
        },
      }));
    } catch (err) {
      console.error('File conversion error:', err);
      setError('Failed to process audio file');
      setIsProcessing(false);
    }
  };

  // TTS conversion - use REST API like the working TTS page
  const handleTtsConvert = async () => {
    if (!ttsText.trim()) {
      setError('Please enter text to convert');
      return;
    }

    setIsProcessing(true);
    setOutputUrl(null);
    setError(null);

    try {
      // Use REST API for TTS generation (same as working /dashboard/tts page)
      const response = await ttsApi.generate({
        text: ttsText.trim(),
        voice: ttsVoice,
        style: 'default',
        rate: ttsRate > 0 ? `+${ttsRate}%` : `${ttsRate}%`,
        pitch: ttsPitch > 0 ? `+${ttsPitch}Hz` : `${ttsPitch}Hz`,
        // Pass the selected model for voice conversion
        voice_model_id: selectedModel?.id,
        f0_up_key: settings.pitch,
        index_rate: settings.indexRate,
      });

      // Convert base64 to blob URL (matching working implementation)
      const base64ToBlob = (base64: string, mimeType: string): Blob => {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
      };

      const audioBlob = base64ToBlob(response.audio, 'audio/wav');
      const url = URL.createObjectURL(audioBlob);
      setOutputUrl(url);
      setSuccess('Speech generated successfully!');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err: any) {
      console.error('TTS error:', err);
      setError(err.response?.data?.message || err.response?.data?.error || 'Failed to generate speech');
    } finally {
      setIsProcessing(false);
    }
  };

  // Recording handlers
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
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setRecordedAudio(audioBlob);
        setRecordedUrl(URL.createObjectURL(audioBlob));
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      console.error('Failed to start recording:', err);
      setError('Failed to access microphone. Please grant permission.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Convert recorded audio
  const handleSpeechConvert = async () => {
    if (!recordedAudio || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Please record audio and ensure connection to voice engine');
      return;
    }

    setIsProcessing(true);
    setOutputUrl(null);
    setError(null);

    try {
      const arrayBuffer = await recordedAudio.arrayBuffer();
      const base64 = arrayBufferToBase64(arrayBuffer);

      wsRef.current.send(JSON.stringify({
        type: 'audio',
        data: base64,
        final: true,
        format: 'webm',
        settings: {
          f0_up_key: settings.pitch,
          index_rate: settings.indexRate,
          filter_radius: settings.filterRadius,
          rms_mix_rate: settings.rmsMixRate,
          protect: settings.protect,
        },
      }));
    } catch (err) {
      console.error('Speech conversion error:', err);
      setError('Failed to process recorded audio');
      setIsProcessing(false);
    }
  };

  // Clear output and errors when switching tabs
  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab);
    setOutputUrl(null);
    setError(null);
    setSuccess(null);
  };

  // Tab configuration
  const tabs = [
    { id: 'file' as TabType, label: 'File Upload', icon: Upload },
    { id: 'tts' as TabType, label: 'Text to Speech', icon: MessageSquare },
    { id: 'speech' as TabType, label: 'Speech to Speech', icon: Radio },
  ];

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center h-16 gap-4">
            <Link
              href="/models"
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
            >
              <ChevronLeft className="h-5 w-5" />
              Back
            </Link>
            <div className="flex items-center gap-2">
              <Mic2 className="h-6 w-6 text-primary-500" />
              <span className="text-lg font-bold">Voice Converter</span>
            </div>

            {/* Connection Status */}
            <div className="ml-auto flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm text-gray-400">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="space-y-6">
          {/* Model Selection Display */}
          <section className="glass rounded-xl p-6">
            <h2 className="font-semibold mb-4">Selected Voice Model</h2>
            {isLoadingModel ? (
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin text-gray-400" />
                <span className="text-gray-400">Loading model...</span>
              </div>
            ) : selectedModel ? (
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-lg bg-gradient-to-br from-primary-600 to-accent-600 flex items-center justify-center">
                  <Mic2 className="h-7 w-7 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-lg">{selectedModel.name}</h3>
                  <p className="text-sm text-gray-400">
                    {selectedModel.model_file} • {selectedModel.size}
                    {selectedModel.has_index && ' • Has Index'}
                  </p>
                </div>
                <Link
                  href="/models"
                  className="text-primary-400 hover:text-primary-300 text-sm"
                >
                  Change Model
                </Link>
              </div>
            ) : (
              <div className="flex items-center justify-between">
                <p className="text-gray-400">No model selected</p>
                <Link
                  href="/models"
                  className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  <Volume2 className="h-4 w-4" />
                  Select a Model
                </Link>
              </div>
            )}
          </section>

          {/* Notifications */}
          {error && (
            <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-lg">
              <AlertCircle className="h-5 w-5 flex-shrink-0" />
              <p className="flex-1">{error}</p>
              <button onClick={() => setError(null)} className="hover:text-red-300">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {success && (
            <div className="flex items-center gap-3 bg-green-500/10 border border-green-500/20 text-green-400 px-4 py-3 rounded-lg">
              <CheckCircle2 className="h-5 w-5 flex-shrink-0" />
              <p className="flex-1">{success}</p>
              <button onClick={() => setSuccess(null)} className="hover:text-green-300">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {/* Tabs */}
          {selectedModel && (
            <>
              <div className="flex gap-2 border-b border-gray-800">
                {tabs.map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => handleTabChange(id)}
                    className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                      activeTab === id
                        ? 'border-primary-500 text-white'
                        : 'border-transparent text-gray-400 hover:text-white'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className="space-y-6">
                {/* File Upload Tab */}
                {activeTab === 'file' && (
                  <section className="glass rounded-xl p-6">
                    <h2 className="font-semibold mb-4 flex items-center gap-2">
                      <FileAudio className="h-5 w-5" />
                      Upload Audio File
                    </h2>
                    <div
                      {...getRootProps()}
                      className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
                        isDragActive
                          ? 'border-primary-500 bg-primary-500/10'
                          : 'border-gray-700 hover:border-gray-600'
                      }`}
                    >
                      <input {...getInputProps()} />
                      <Upload className="h-10 w-10 text-gray-500 mx-auto mb-4" />
                      {audioFile ? (
                        <p className="text-primary-400">{audioFile.name}</p>
                      ) : (
                        <>
                          <p className="text-gray-300 mb-1">
                            Drag & drop audio file here, or click to select
                          </p>
                          <p className="text-sm text-gray-500">
                            Supports WAV, MP3, FLAC, OGG, M4A (max 50MB)
                          </p>
                        </>
                      )}
                    </div>

                    {audioUrl && (
                      <div className="mt-4">
                        <p className="text-sm text-gray-400 mb-2">Preview:</p>
                        <audio controls src={audioUrl} className="w-full" />
                      </div>
                    )}
                  </section>
                )}

                {/* TTS Tab */}
                {activeTab === 'tts' && (
                  <section className="glass rounded-xl p-6 space-y-6">
                    <h2 className="font-semibold mb-4 flex items-center gap-2">
                      <MessageSquare className="h-5 w-5" />
                      Text to Speech
                    </h2>

                    {/* Voice Selection */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">Language</label>
                        <select
                          value={ttsLanguage}
                          onChange={(e) => setTtsLanguage(e.target.value)}
                          className="w-full bg-gray-800/50 border border-gray-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                        >
                          {ttsLanguages.map(lang => (
                            <option key={lang} value={lang}>{lang}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">Gender</label>
                        <div className="flex gap-2">
                          <button
                            onClick={() => setTtsGender('male')}
                            className={`flex-1 py-2 rounded-lg border transition-colors ${
                              ttsGender === 'male'
                                ? 'bg-primary-600 border-primary-500 text-white'
                                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
                            }`}
                          >
                            Male
                          </button>
                          <button
                            onClick={() => setTtsGender('female')}
                            className={`flex-1 py-2 rounded-lg border transition-colors ${
                              ttsGender === 'female'
                                ? 'bg-primary-600 border-primary-500 text-white'
                                : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:border-gray-600'
                            }`}
                          >
                            Female
                          </button>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">Voice</label>
                        <select
                          value={ttsVoice}
                          onChange={(e) => setTtsVoice(e.target.value)}
                          className="w-full bg-gray-800/50 border border-gray-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                        >
                          {filteredTtsVoices.map(v => (
                            <option key={v.id} value={v.id}>{v.name}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* Rate and Pitch */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">
                          Speech Rate: {ttsRate > 0 ? '+' : ''}{ttsRate}%
                        </label>
                        <input
                          type="range"
                          min="-50"
                          max="50"
                          value={ttsRate}
                          onChange={(e) => setTtsRate(parseInt(e.target.value))}
                          className="w-full accent-primary-500"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>Slower</span>
                          <span>Normal</span>
                          <span>Faster</span>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">
                          Voice Pitch: {ttsPitch > 0 ? '+' : ''}{ttsPitch}Hz
                        </label>
                        <input
                          type="range"
                          min="-50"
                          max="50"
                          value={ttsPitch}
                          onChange={(e) => setTtsPitch(parseInt(e.target.value))}
                          className="w-full accent-primary-500"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>Lower</span>
                          <span>Normal</span>
                          <span>Higher</span>
                        </div>
                      </div>
                    </div>

                    {/* Text Input */}
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Text</label>
                      <textarea
                        value={ttsText}
                        onChange={(e) => setTtsText(e.target.value)}
                        placeholder="Enter text to convert to speech with the selected voice..."
                        className="w-full h-32 bg-gray-800/50 border border-gray-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                      />
                      <p className="text-sm text-gray-500 mt-2">
                        {ttsText.length} characters
                      </p>
                    </div>

                    <p className="text-xs text-gray-500">
                      💡 The TTS audio will be converted using the selected voice model above for a unique character voice.
                    </p>
                  </section>
                )}

                {/* Speech-to-Speech Tab */}
                {activeTab === 'speech' && (
                  <section className="glass rounded-xl p-6">
                    <h2 className="font-semibold mb-4 flex items-center gap-2">
                      <Radio className="h-5 w-5" />
                      Speech to Speech
                    </h2>
                    <div className="text-center py-8">
                      <button
                        onClick={isRecording ? stopRecording : startRecording}
                        className={`w-24 h-24 rounded-full flex items-center justify-center transition-all ${
                          isRecording
                            ? 'bg-red-600 hover:bg-red-700 animate-pulse'
                            : 'bg-primary-600 hover:bg-primary-700'
                        }`}
                      >
                        {isRecording ? (
                          <Square className="h-10 w-10 text-white" />
                        ) : (
                          <Mic2 className="h-10 w-10 text-white" />
                        )}
                      </button>
                      <p className="mt-4 text-gray-400">
                        {isRecording ? 'Recording... Click to stop' : 'Click to start recording'}
                      </p>
                    </div>

                    {recordedUrl && !isRecording && (
                      <div className="mt-4">
                        <p className="text-sm text-gray-400 mb-2">Recorded Audio:</p>
                        <audio controls src={recordedUrl} className="w-full" />
                      </div>
                    )}
                  </section>
                )}

                {/* Settings */}
                <section className="glass rounded-xl p-6">
                  <h2 className="font-semibold mb-4 flex items-center gap-2">
                    <Sliders className="h-5 w-5" />
                    Conversion Settings
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Pitch */}
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">
                        Pitch Shift: {settings.pitch > 0 ? `+${settings.pitch}` : settings.pitch} semitones
                      </label>
                      <input
                        type="range"
                        min="-12"
                        max="12"
                        value={settings.pitch}
                        onChange={(e) => setSettings({ ...settings, pitch: parseInt(e.target.value) })}
                        className="w-full accent-primary-500"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>-12 (Lower)</span>
                        <span>0</span>
                        <span>+12 (Higher)</span>
                      </div>
                    </div>

                    {/* Index Rate */}
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">
                        Index Rate: {settings.indexRate.toFixed(2)}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={settings.indexRate}
                        onChange={(e) => setSettings({ ...settings, indexRate: parseFloat(e.target.value) })}
                        className="w-full accent-primary-500"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>0 (No Index)</span>
                        <span>1 (Full Index)</span>
                      </div>
                    </div>

                    {/* RMS Mix Rate */}
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">
                        RMS Mix Rate: {settings.rmsMixRate.toFixed(2)}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={settings.rmsMixRate}
                        onChange={(e) => setSettings({ ...settings, rmsMixRate: parseFloat(e.target.value) })}
                        className="w-full accent-primary-500"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>0 (Input)</span>
                        <span>1 (Target)</span>
                      </div>
                    </div>

                    {/* Protect */}
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">
                        Protect: {settings.protect.toFixed(2)}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="0.5"
                        step="0.01"
                        value={settings.protect}
                        onChange={(e) => setSettings({ ...settings, protect: parseFloat(e.target.value) })}
                        className="w-full accent-primary-500"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>0 (No Protect)</span>
                        <span>0.5 (Max)</span>
                      </div>
                    </div>
                  </div>
                </section>

                {/* Convert Button */}
                <button
                  onClick={
                    activeTab === 'file'
                      ? handleFileConvert
                      : activeTab === 'tts'
                      ? handleTtsConvert
                      : handleSpeechConvert
                  }
                  disabled={
                    isProcessing ||
                    !isConnected ||
                    (activeTab === 'file' && !audioFile) ||
                    (activeTab === 'tts' && !ttsText.trim()) ||
                    (activeTab === 'speech' && !recordedAudio)
                  }
                  className="w-full bg-gradient-to-r from-primary-600 to-accent-600 hover:from-primary-700 hover:to-accent-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-4 rounded-xl transition-all flex items-center justify-center gap-2"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Mic2 className="h-5 w-5" />
                      {activeTab === 'tts' ? 'Generate Speech' : 'Convert Voice'}
                    </>
                  )}
                </button>

                {/* Output */}
                {outputUrl && (
                  <section className="glass rounded-xl p-6">
                    <h2 className="font-semibold mb-4 text-green-400 flex items-center gap-2">
                      <CheckCircle2 className="h-5 w-5" />
                      Conversion Complete!
                    </h2>
                    <audio controls src={outputUrl} className="w-full mb-4" />
                    <a
                      href={outputUrl}
                      download={`converted_${selectedModel?.name || 'audio'}.wav`}
                      className="flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-lg transition-colors"
                    >
                      <Download className="h-5 w-5" />
                      Download Result
                    </a>
                  </section>
                )}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

// Helper functions
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

// Export with Suspense wrapper for useSearchParams
export default function ConvertPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    }>
      <ConvertPageContent />
    </Suspense>
  );
}
