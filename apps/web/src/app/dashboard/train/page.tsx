'use client';

import { useState, useEffect, useCallback, useRef, useMemo, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Mic2, Mic, Upload, FileAudio, Check, X, ChevronLeft, ChevronRight,
  Loader2, AlertCircle, RotateCcw, Square, FolderUp, Wand2,
  Volume2, Waves, Target, Sparkles, Play, Plus, Music, Zap,
  Activity, ChevronDown, ChevronUp, GitBranch, History
} from 'lucide-react';
import { trainerApi, voiceModelsApi, VoiceModel } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import { DashboardLayout } from '@/components/dashboard-layout';
import { CreateEmptyModelForm } from '@/components/create-empty-model-form';
import { TrainingRunsHistory } from '@/components/training-runs-history';

type Step = 'select-model' | 'select-language' | 'training-areas' | 'recording' | 'start-training' | 'training-progress';

interface TrainingArea {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  score: number | null;  // null indicates "not analyzed"
  maxScore: number;
  details: string[];
}

interface PromptCategory {
  name: string;
  description: string;
  phonemes_covered?: string[];
  prompts?: string[];
  prompt_count?: number;
}

interface PromptsData {
  language: string;
  language_name: string;
  total_prompts: number;
  categories: Record<string, PromptCategory>;
}

interface WizardPrompt {
  index: number;
  text: string;
  phonemes: string[];
  ipa_text?: string;
}

interface WizardSession {
  session_id: string;
  language: string;
  exp_name: string;
  total_prompts: number;
  completed: number;
  skipped: number;
  status: 'created' | 'recording' | 'paused' | 'completed';
  current_index: number;
}

interface PhonemeStats {
  total: number;
  covered: number;
  missing: string[];
}

// Error Boundary component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <DashboardLayout>
          <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
            <AlertCircle className="w-12 h-12 text-red-500" />
            <h2 className="text-xl font-semibold">Something went wrong</h2>
            <p className="text-gray-400 text-center max-w-md">
              {this.state.error?.message || 'An error occurred while loading the training page.'}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-500 rounded-lg transition-colors"
            >
              Reload Page
            </button>
          </div>
        </DashboardLayout>
      );
    }

    return this.props.children;
  }
}

// Need to import React for class component
import * as React from 'react';

// Wrapper component for Suspense boundary (required for useSearchParams)
export default function TrainPage() {
  return (
    <ErrorBoundary>
      <Suspense fallback={
        <DashboardLayout>
          <div className="flex items-center justify-center min-h-[60vh]">
            <Loader2 className="h-8 w-8 animate-spin text-purple-500" />
          </div>
        </DashboardLayout>
      }>
        <TrainPageContent />
      </Suspense>
    </ErrorBoundary>
  );
}

function TrainPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const { user } = useAuthStore();
  
  // State
  const [step, setStep] = useState<Step>('select-model');
  const [selectedModel, setSelectedModel] = useState<VoiceModel | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('en');
  const [selectedArea, setSelectedArea] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [expandedArea, setExpandedArea] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  
  // Recording session state
  const [session, setSession] = useState<WizardSession | null>(null);
  const [currentPrompt, setCurrentPrompt] = useState<WizardPrompt | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isStartingSession, setIsStartingSession] = useState(false);
  
  // Import tab state
  const [importFiles, setImportFiles] = useState<File[]>([]);
  const [importUploading, setImportUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [filesUploadedCount, setFilesUploadedCount] = useState(0); // Track uploaded file count
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Training state
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState<string>('');
  const [trainingDetails, setTrainingDetails] = useState<any>(null);
  const [checkpointMode, setCheckpointMode] = useState<'continue' | 'add-audio' | 'new-audio' | null>(null);
  const [targetEpochs, setTargetEpochs] = useState<number>(200);
  const [batchSize, setBatchSize] = useState<number>(6);
  const [useAutoConfig, setUseAutoConfig] = useState<boolean>(false);
  const [forceReprocess, setForceReprocess] = useState<boolean>(false);
  const isPollingRef = useRef<boolean>(false);
  
  // Audio recording refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  
  // Get model ID from URL if present
  const modelIdFromUrl = searchParams.get('model');
  
  // Fetch user's models
  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels } = useQuery({
    queryKey: ['my-models-for-training'],
    queryFn: () => voiceModelsApi.myModels({ per_page: 100 }),
    // Also enable when we need to get training job ID for resume
    enabled: step === 'select-model' || step === 'training-areas' || (step === 'training-progress' && !trainingJobId),
  });
  
  // Fetch available languages
  const { data: languages } = useQuery({
    queryKey: ['trainer-languages'],
    queryFn: () => trainerApi.getLanguages(),
    enabled: step === 'select-language',
  });
  
  // Fetch prompts for selected language (to get categories)
  const { data: promptsData } = useQuery<PromptsData>({
    queryKey: ['trainer-prompts', selectedLanguage],
    queryFn: () => trainerApi.getPrompts(selectedLanguage),
    enabled: !!selectedLanguage && (step === 'training-areas' || step === 'recording'),
  });

  // Fetch category status (recordings per category) for the selected model
  const { data: categoryStatus, refetch: refetchCategoryStatus } = useQuery({
    queryKey: ['category-status', selectedModel?.slug, selectedLanguage],
    queryFn: () => selectedModel?.slug 
      ? trainerApi.getCategoryStatus(selectedModel.slug, selectedLanguage)
      : null,
    enabled: !!selectedModel?.slug && !!selectedLanguage && step === 'training-areas',
  });

  // Fetch model recordings summary
  const { data: modelRecordings, refetch: refetchModelRecordings } = useQuery({
    queryKey: ['model-recordings', selectedModel?.slug],
    queryFn: () => selectedModel?.slug 
      ? trainerApi.getModelRecordings(selectedModel.slug)
      : null,
    enabled: !!selectedModel?.slug && (step === 'select-language' || step === 'training-areas' || step === 'start-training'),
  });

  // Fetch model training info (includes checkpoint status)
  const { data: modelTrainingInfo, refetch: refetchTrainingInfo } = useQuery({
    queryKey: ['model-training-info', selectedModel?.slug],
    queryFn: () => selectedModel?.slug 
      ? trainerApi.getModelTrainingInfo(selectedModel.slug)
      : null,
    enabled: !!selectedModel?.slug && (step === 'training-areas' || step === 'start-training' || step === 'training-progress'),
  });
  
  // Memoize models
  const myModels = useMemo(() => modelsData?.data || [], [modelsData?.data]);
  
  // Calculate phoneme stats from selected model
  const phonemeStats = useMemo((): PhonemeStats | null => {
    if (!selectedModel || !selectedLanguage) return null;
    
    const isEnglish = selectedLanguage === 'en';
    const coverage = isEnglish 
      ? (selectedModel.en_phoneme_coverage || 0)
      : (selectedModel.is_phoneme_coverage || 0);
    const missingPhonemes = isEnglish
      ? (selectedModel.en_missing_phonemes || [])
      : (selectedModel.is_missing_phonemes || []);
    
    const totalPhonemes = isEnglish ? 44 : 42;
    const coveredCount = Math.round((coverage / 100) * totalPhonemes);
    
    return {
      total: totalPhonemes,
      covered: coveredCount,
      missing: missingPhonemes,
    };
  }, [selectedModel, selectedLanguage]);
  
  // Build training areas with scores
  const trainingAreas = useMemo((): TrainingArea[] => {
    const phonemeCoverage = phonemeStats 
      ? Math.round((phonemeStats.covered / phonemeStats.total) * 100)
      : 0;
    
    // Check if model has been scanned (language_scanned_at indicates phoneme analysis)
    const isScanned = selectedModel?.language_scanned_at !== null && selectedModel?.language_scanned_at !== undefined;
    
    // For new models without analysis, show null to indicate "Not analyzed"
    // These scores only make sense after actual training or analysis
    const clarityScore = (selectedModel as any)?.clarity_score ?? null;
    const pitchScore = (selectedModel as any)?.pitch_stability_score ?? null;
    const expressionScore = (selectedModel as any)?.expression_score ?? null;
    
    return [
      {
        id: 'phonemes',
        name: 'Phoneme Coverage',
        description: 'Train specific sounds and pronunciations',
        icon: <Mic2 className="w-6 h-6" />,
        color: 'purple',
        score: phonemeCoverage,
        maxScore: 100,
        details: (phonemeStats?.missing || []).slice(0, 5),
      },
      {
        id: 'clarity',
        name: 'Voice Clarity',
        description: 'Improve articulation and clearness',
        icon: <Volume2 className="w-6 h-6" />,
        color: 'blue',
        score: clarityScore,
        maxScore: 100,
        details: ['Consonant clarity', 'Vowel distinction', 'Word boundaries'],
      },
      {
        id: 'pitch',
        name: 'Pitch Stability',
        description: 'Consistent pitch and tone control',
        icon: <Waves className="w-6 h-6" />,
        color: 'green',
        score: pitchScore,
        maxScore: 100,
        details: ['Sustained notes', 'Pitch transitions', 'Vibrato control'],
      },
      {
        id: 'expression',
        name: 'Expression Range',
        description: 'Emotional variety and dynamics',
        icon: <Sparkles className="w-6 h-6" />,
        color: 'orange',
        score: expressionScore,
        maxScore: 100,
        details: ['Happy tone', 'Sad tone', 'Excited tone', 'Calm tone'],
      },
    ];
  }, [phonemeStats, selectedModel]);
  
  // Get categories for expanded area with recording counts
  const areaCategories = useMemo(() => {
    if (!expandedArea || !promptsData?.categories) return [];
    
    // Get recording counts from categoryStatus if available
    const recordingCounts = categoryStatus?.categories || {};
    
    // Helper to get existing coverage from model scan
    const getModelCoverage = (phonemes: string[]): boolean => {
      if (!selectedModel || !phonemes.length) return false;
      const isEnglish = selectedLanguage === 'en';
      const missingPhonemes = isEnglish 
        ? (selectedModel.en_missing_phonemes || [])
        : (selectedModel.is_missing_phonemes || []);
      // Check if none of the category's phonemes are in missing list
      return phonemes.every(p => !missingPhonemes.includes(p));
    };
    
    if (expandedArea === 'phonemes') {
      return Object.entries(promptsData.categories).map(([key, cat]) => {
        const status = recordingCounts[key];
        const phonemes = cat.phonemes_covered || [];
        return {
          id: key,
          name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          description: cat.description,
          phonemes: phonemes,
          promptCount: cat.prompt_count || cat.prompts?.length || 0,
          // New fields for status indicators
          recordingCount: status?.recordings || 0,
          hasRecordings: status?.has_recordings || false,
          modelHasCoverage: getModelCoverage(phonemes), // From existing model scan
          willHaveCoverage: (status?.recordings || 0) > 0 || getModelCoverage(phonemes), // After training
        };
      });
    }
    
    // For other areas, return static categories (could be expanded later)
    if (expandedArea === 'clarity') {
      return [
        { id: 'consonants', name: 'Consonant Exercises', description: 'Practice clear consonant sounds', phonemes: [], promptCount: 20, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
        { id: 'vowels', name: 'Vowel Exercises', description: 'Distinct vowel pronunciation', phonemes: [], promptCount: 15, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
        { id: 'diction', name: 'Diction Practice', description: 'Word clarity and enunciation', phonemes: [], promptCount: 25, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
      ];
    }
    
    if (expandedArea === 'pitch') {
      return [
        { id: 'sustained', name: 'Sustained Tones', description: 'Hold steady pitches', phonemes: [], promptCount: 10, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
        { id: 'scales', name: 'Scale Patterns', description: 'Ascending and descending patterns', phonemes: [], promptCount: 15, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
        { id: 'transitions', name: 'Pitch Transitions', description: 'Smooth pitch changes', phonemes: [], promptCount: 12, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
      ];
    }
    
    if (expandedArea === 'expression') {
      return [
        { id: 'happy', name: 'Happy & Excited', description: 'Positive emotional expressions', phonemes: [], promptCount: 20, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
        { id: 'sad', name: 'Sad & Melancholic', description: 'Subdued emotional tones', phonemes: [], promptCount: 15, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
        { id: 'angry', name: 'Angry & Intense', description: 'Strong emotional delivery', phonemes: [], promptCount: 15, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
        { id: 'calm', name: 'Calm & Neutral', description: 'Relaxed, even tones', phonemes: [], promptCount: 20, recordingCount: 0, hasRecordings: false, modelHasCoverage: false, willHaveCoverage: false },
      ];
    }
    
    return [];
  }, [expandedArea, promptsData, categoryStatus, selectedModel, selectedLanguage]);
  
  // Get resume flag from URL
  const resumeTraining = searchParams.get('resume') === 'true';
  
  // Auto-select model from URL and optionally resume training view
  useEffect(() => {
    if (modelIdFromUrl && myModels.length > 0) {
      const modelId = parseInt(modelIdFromUrl);
      // Try to find by ID first, then by slug
      let model = myModels.find((m: VoiceModel) => m.id === modelId);
      if (!model) {
        model = myModels.find((m: VoiceModel) => m.slug === modelIdFromUrl);
      }
      
      // Only auto-select if not already selected
      if (model && (!selectedModel || selectedModel.id !== model.id)) {
        setSelectedModel(model);
        
        // If resuming training and model is training, go directly to progress view
        if (resumeTraining && model.status === 'training') {
          setStep('training-progress');
          // Set job ID and initial progress - polling will be triggered by useEffect
          if (model.training_job_id) {
            setTrainingJobId(model.training_job_id);
            setTrainingProgress(model.training_progress || 0);
            setTrainingStatus(model.training_epoch && model.training_total_epochs 
              ? `Training... Epoch ${model.training_epoch}/${model.training_total_epochs}`
              : 'Training in progress...');
          }
        } else {
          setStep('select-language');
        }
      }
    }
  }, [modelIdFromUrl, myModels, resumeTraining]);
  
  // Handle model creation success
  const handleModelCreated = (model: VoiceModel) => {
    // First hide the form and set the model
    setShowCreateForm(false);
    // Use the model directly from the API response - it has all needed fields
    // including slug which is generated server-side
    setSelectedModel(model);
    // Navigate to language selection step
    setStep('select-language');
    // Refetch models in background to update the list
    refetchModels();
  };
  
  // Start recording session for a category
  const startCategorySession = async (categoryId: string) => {
    if (!selectedModel || isStartingSession) return;
    
    setError(null);
    setSelectedCategory(categoryId);
    setIsStartingSession(true);
    
    try {
      // Handle wizard mode (general recording without specific category)
      const isWizardMode = categoryId === '_wizard';
      
      // Get prompts for this category's phonemes if available
      const category = isWizardMode ? null : promptsData?.categories?.[categoryId];
      const targetPhonemes = category?.phonemes_covered;
      const promptCount = isWizardMode ? 20 : (category?.prompt_count || category?.prompts?.length || 20);
      
      const newSession = await trainerApi.createWizardSession(
        selectedLanguage,
        selectedModel.slug || selectedModel.name,
        promptCount,
        targetPhonemes
      );
      
      setSession({
        session_id: newSession.session_id,
        language: newSession.language,
        exp_name: newSession.exp_name,
        total_prompts: newSession.total_prompts,
        completed: newSession.completed_prompts || 0,
        skipped: 0,
        status: newSession.status as WizardSession['status'],
        current_index: newSession.current_prompt_index || 0,
      });
      
      await trainerApi.startWizardSession(newSession.session_id);
      const prompt = await trainerApi.getWizardPrompt(newSession.session_id);
      setCurrentPrompt(prompt);
      setStep('recording');
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || 'Failed to start recording session');
    } finally {
      setIsStartingSession(false);
    }
  };
  
  // Recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
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
        setAudioUrl(URL.createObjectURL(audioBlob));
      };
      
      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setError('Microphone access denied. Please allow microphone access to record.');
    }
  };
  
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      streamRef.current?.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };
  
  const clearRecording = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setRecordedAudio(null);
    setAudioUrl(null);
  };
  
  // Import file handling
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const audioFiles = files.filter(f => 
      f.type.startsWith('audio/') || 
      f.name.endsWith('.wav') || 
      f.name.endsWith('.mp3') || 
      f.name.endsWith('.ogg') ||
      f.name.endsWith('.flac')
    );
    setImportFiles(prev => [...prev, ...audioFiles]);
  };
  
  const removeImportFile = (index: number) => {
    setImportFiles(prev => prev.filter((_, i) => i !== index));
  };
  
  const uploadImportFiles = async () => {
    if (!selectedModel || importFiles.length === 0) return;
    
    setImportUploading(true);
    setUploadProgress(0);
    setError(null);
    
    try {
      const expName = selectedModel.slug || selectedModel.name;
      console.log('Uploading files to trainer:', { 
        exp_name: expName,
        fileCount: importFiles.length,
        language: selectedLanguage 
      });
      
      const result = await trainerApi.uploadTrainingAudio(
        expName,
        importFiles,
        selectedLanguage,
        (progress) => {
          console.log('Upload progress:', progress);
          setUploadProgress(progress);
        }
      );
      
      console.log('Upload result:', result);
      
      const uploadedCount = importFiles.length;
      setFilesUploadedCount(uploadedCount);
      setImportFiles([]);
      // New files uploaded - force reprocessing
      setForceReprocess(true);
      // Refresh recordings data then go to training step
      await refetchModelRecordings();
      setStep('start-training');
    } catch (err: any) {
      console.error('Upload failed:', err);
      setError(err.response?.data?.message || err.message || 'Failed to upload audio files');
    } finally {
      setImportUploading(false);
      setUploadProgress(0);
    }
  };
  
  // Submit recording
  const submitRecording = async () => {
    if (!session || !recordedAudio) return;
    
    setIsSubmitting(true);
    setError(null);
    
    try {
      const reader = new FileReader();
      const base64Promise = new Promise<string>((resolve, reject) => {
        reader.onload = () => {
          const result = reader.result as string;
          const base64 = result.split(',')[1];
          resolve(base64);
        };
        reader.onerror = reject;
      });
      reader.readAsDataURL(recordedAudio);
      const base64Audio = await base64Promise;
      
      const result = await trainerApi.submitRecording(
        session.session_id,
        base64Audio,
        48000,
        true,
        'webm'  // Browser records in webm/opus format
      );
      
      setSession(prev => prev ? {
        ...prev,
        completed: prev.completed + 1,
        current_index: prev.current_index + 1
      } : null);
      
      clearRecording();
      
      if (result.next_prompt) {
        // Normalize the prompt - backend returns prompt_text, frontend expects text
        const normalizedPrompt = {
          ...result.next_prompt,
          text: result.next_prompt.text || result.next_prompt.prompt_text || '',
        };
        setCurrentPrompt(normalizedPrompt);
      } else if (result.session?.status === 'completed') {
        // Refresh category status to show the new recordings
        refetchCategoryStatus();
        refetchModelRecordings();
        // New recordings - force reprocessing
        setForceReprocess(true);
        setStep('start-training');
      } else {
        const prompt = await trainerApi.getWizardPrompt(session.session_id);
        setCurrentPrompt(prompt);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to submit recording');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // Skip prompt
  const skipPrompt = async () => {
    if (!session) return;
    
    try {
      const result = await trainerApi.wizardSkip(session.session_id);
      setSession(prev => prev ? {
        ...prev,
        skipped: prev.skipped + 1,
        current_index: prev.current_index + 1
      } : null);
      
      clearRecording();
      
      if (result.next_prompt) {
        // Normalize the prompt - backend returns prompt_text, frontend expects text
        const normalizedPrompt = {
          ...result.next_prompt,
          text: result.next_prompt.text || result.next_prompt.prompt_text || '',
        };
        setCurrentPrompt(normalizedPrompt);
      } else if (result.session?.status === 'completed') {
        // Refresh category status to show the new recordings
        refetchCategoryStatus();
        refetchModelRecordings();
        // New recordings - force reprocessing
        setForceReprocess(true);
        setStep('start-training');
      } else {
        const prompt = await trainerApi.getWizardPrompt(session.session_id);
        setCurrentPrompt(prompt);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to skip prompt');
    }
  };
  
  // Finish recording early and go to training
  const finishRecording = async () => {
    if (session) {
      try {
        await trainerApi.completeWizardSession(session.session_id);
      } catch (err) {
        // Ignore errors
      }
    }
    // Refresh category status to show the new recordings
    refetchCategoryStatus();
    refetchModelRecordings();
    setStep('start-training');
  };
  
  // Start actual training using all collected recordings
  const startTraining = async (mode?: 'continue' | 'fresh') => {
    if (!selectedModel) return;
    
    setError(null);
    setTrainingStatus('Starting training...');
    
    // Determine training options based on mode and checkpointMode
    // checkpointMode:
    //   'continue' = just train more epochs (no reprocess, continue from checkpoint)
    //   'add-audio' = add new audio and continue (reprocess, continue from checkpoint)
    //   'new-audio' = fresh start with new audio (reprocess, no continue)
    
    // Mode param overrides checkpointMode for explicit button clicks
    let shouldContinue = checkpointMode === 'continue' || checkpointMode === 'add-audio';
    let needsReprocess = checkpointMode === 'add-audio' || checkpointMode === 'new-audio' || forceReprocess;
    
    // Override based on explicit mode parameter
    if (mode === 'fresh') {
      shouldContinue = false;
      needsReprocess = true; // Fresh always implies reprocess
    } else if (mode === 'continue') {
      shouldContinue = true;
      // Keep needsReprocess from checkpointMode
    }
    
    try {
      // Use the new API that collects all recordings from wizard sessions
      const trainingConfig: {
        force_reprocess: boolean;
        continue_from_checkpoint: boolean;
        epochs?: number;
        batch_size?: number;
      } = {
        force_reprocess: needsReprocess,
        continue_from_checkpoint: shouldContinue,
      };
      
      // Always respect user's explicit epoch/batch choices unless auto-config is enabled
      if (!useAutoConfig) {
        trainingConfig.epochs = targetEpochs;
        trainingConfig.batch_size = batchSize;
      }
      // When useAutoConfig is true, omit epochs/batch_size to let server auto-configure
      
      const response = await trainerApi.trainModelWithRecordings(
        selectedModel.slug || selectedModel.name,
        trainingConfig
      );
      
      setTrainingJobId(response.job_id);
      setStep('training-progress');
      
      // Polling will be triggered by useEffect
    } catch (err: any) {
      setError(err.response?.data?.message || err.response?.data?.detail || err.message || 'Failed to start training');
    }
  };
  
  // Poll training status
  const pollTrainingStatus = async (jobId: string) => {
    // Prevent multiple polling loops
    if (isPollingRef.current) return;
    isPollingRef.current = true;
    
    const poll = async () => {
      try {
        const status = await trainerApi.trainingStatus(jobId);
        setTrainingProgress(status.progress || 0);
        
        // Store detailed status for display
        setTrainingDetails(status);
        
        // Build a meaningful status message
        let statusMessage = status.message || status.status_message || status.status;
        if (status.status === 'training' && status.current_epoch && status.total_epochs) {
          statusMessage = `Training... Epoch ${status.current_epoch}/${status.total_epochs}`;
        } else if (status.status === 'preprocessing') {
          statusMessage = 'Preprocessing audio files...';
        } else if (status.status === 'extracting_f0') {
          statusMessage = 'Extracting pitch (F0)...';
        } else if (status.status === 'extracting_features') {
          statusMessage = 'Extracting audio features...';
        } else if (status.status === 'building_index') {
          statusMessage = 'Building model index...';
        }
        setTrainingStatus(statusMessage);
        
        if (status.status === 'completed') {
          isPollingRef.current = false;
          // Rescan model to update scores
          if (selectedModel) {
            await trainerApi.scanModel(selectedModel.id, [selectedLanguage]);
            queryClient.invalidateQueries({ queryKey: ['my-models-for-training'] });
          }
          // Refresh training info to update checkpoint status
          refetchTrainingInfo();
          setTrainingStatus('Training complete! Model updated.');
        } else if (status.status === 'failed') {
          isPollingRef.current = false;
          // Refresh training info - checkpoint may have been saved before failure
          refetchTrainingInfo();
          setError(status.error || 'Training failed');
        } else if (status.status === 'cancelled') {
          isPollingRef.current = false;
          // Refresh training info - checkpoint may have been saved
          refetchTrainingInfo();
          setTrainingStatus('Training cancelled. You can continue from the saved checkpoint.');
        } else {
          // Continue polling
          setTimeout(poll, 5000);
        }
      } catch (err: any) {
        isPollingRef.current = false;
        setError(err.message || 'Failed to check training status');
      }
    };
    
    poll();
  };

  // Start polling when trainingJobId is set and we're on progress step
  useEffect(() => {
    if (trainingJobId && step === 'training-progress') {
      // Reset polling flag and start polling
      isPollingRef.current = false;
      pollTrainingStatus(trainingJobId);
    }
    
    // Cleanup: stop polling when leaving
    return () => {
      isPollingRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingJobId, step]);

  // Fetch active training job when on progress step without job ID
  useEffect(() => {
    const fetchActiveJob = async () => {
      if (step === 'training-progress' && selectedModel?.slug && !trainingJobId) {
        try {
          // Fetch all jobs and find the one for this model
          const jobs = await trainerApi.listTrainingJobs();
          const activeJob = jobs.find((j: any) => 
            j.exp_name === selectedModel.slug && 
            ['training', 'queued', 'preprocessing'].includes(j.status)
          );
          
          if (activeJob) {
            setTrainingJobId(activeJob.job_id);
            setTrainingProgress(activeJob.progress || 0);
            setTrainingStatus(activeJob.current_epoch && activeJob.total_epochs 
              ? `Training... Epoch ${activeJob.current_epoch}/${activeJob.total_epochs}`
              : activeJob.message || 'Training in progress...');
          }
        } catch (err) {
          console.error('Failed to fetch active job:', err);
        }
      }
    };
    
    fetchActiveJob();
  }, [step, selectedModel?.slug, trainingJobId]);
  
  // Cancel and reset
  const cancelSession = async () => {
    if (session) {
      try {
        await trainerApi.cancelWizardSession(session.session_id);
      } catch (err) {
        // Ignore
      }
    }
    setSession(null);
    setCurrentPrompt(null);
    clearRecording();
    setSelectedArea(null);
    setSelectedCategory(null);
    setExpandedArea(null);
    setStep('select-model');
  };
  
  // Go back to training areas
  const backToAreas = () => {
    if (session) {
      trainerApi.completeWizardSession(session.session_id).catch(() => {});
    }
    setSession(null);
    setCurrentPrompt(null);
    clearRecording();
    setSelectedCategory(null);
    setStep('training-areas');
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      streamRef.current?.getTracks().forEach(track => track.stop());
    };
  }, [audioUrl]);

  // Step labels for progress indicator
  const steps: { key: Step; label: string }[] = [
    { key: 'select-model', label: 'Model' },
    { key: 'select-language', label: 'Language' },
    { key: 'training-areas', label: 'Training' },
    { key: 'recording', label: 'Record' },
    { key: 'start-training', label: 'Train' },
    { key: 'training-progress', label: 'Progress' },
  ];
  
  const currentStepIndex = steps.findIndex(s => s.key === step);

  // Color utility
  const getColorClasses = (color: string, variant: 'bg' | 'border' | 'text' = 'bg') => {
    const colors: Record<string, Record<string, string>> = {
      purple: { bg: 'bg-purple-500', border: 'border-purple-500', text: 'text-purple-400' },
      blue: { bg: 'bg-blue-500', border: 'border-blue-500', text: 'text-blue-400' },
      green: { bg: 'bg-green-500', border: 'border-green-500', text: 'text-green-400' },
      orange: { bg: 'bg-orange-500', border: 'border-orange-500', text: 'text-orange-400' },
    };
    return colors[color]?.[variant] || colors.purple[variant];
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Train Your Voice Model</h1>
            <p className="text-gray-400">Record audio samples and train your model</p>
          </div>
          {step !== 'select-model' && (
            <button
              onClick={cancelSession}
              className="flex items-center gap-2 px-4 py-2 text-gray-400 hover:text-white border border-gray-700 hover:border-gray-600 rounded-lg transition-colors"
            >
              <X className="w-4 h-4" />
              Cancel
            </button>
          )}
        </div>

        {/* Progress Steps */}
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          {steps.map((s, i) => (
            <div key={s.key} className="flex items-center flex-shrink-0">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                step === s.key
                  ? 'bg-primary-600 text-white'
                  : currentStepIndex > i
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-700 text-gray-400'
              }`}>
                {currentStepIndex > i ? (
                  <Check className="w-4 h-4" />
                ) : (
                  i + 1
                )}
              </div>
              <span className={`ml-2 text-sm hidden sm:inline ${
                step === s.key ? 'text-white' : 'text-gray-500'
              }`}>
                {s.label}
              </span>
              {i < steps.length - 1 && (
                <div className={`w-8 h-0.5 mx-2 ${
                  currentStepIndex > i ? 'bg-green-600' : 'bg-gray-700'
                }`} />
              )}
            </div>
          ))}
        </div>

        {/* Error Message */}
        {error && (
          <div className="flex items-center gap-2 p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
            <button onClick={() => setError(null)} className="ml-auto p-1 hover:bg-red-500/20 rounded">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Step Content */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
          
          {/* Step 1: Select Model */}
          {step === 'select-model' && (
            <div className="space-y-6">
              <h2 className="text-lg font-semibold">Select a Model to Train</h2>
              
              {showCreateForm ? (
                <div className="space-y-4">
                  <button
                    onClick={() => setShowCreateForm(false)}
                    className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                  >
                    <ChevronLeft className="w-4 h-4" />
                    Back to model selection
                  </button>
                  <CreateEmptyModelForm 
                    onSuccess={handleModelCreated}
                    onCancel={() => setShowCreateForm(false)}
                  />
                </div>
              ) : modelsLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {/* Create New Model Card */}
                  <button
                    onClick={() => setShowCreateForm(true)}
                    className="p-4 rounded-lg border border-dashed border-gray-600 hover:border-primary-500 bg-gray-800/30 hover:bg-primary-500/10 transition-all text-left group"
                  >
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 rounded-lg bg-primary-500/20 group-hover:bg-primary-500/30 flex items-center justify-center flex-shrink-0 transition-colors">
                        <Plus className="w-5 h-5 text-primary-400" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-primary-400">Create New Model</h3>
                        <p className="text-sm text-gray-500">
                          Start from scratch with a new voice model
                        </p>
                      </div>
                    </div>
                  </button>
                  
                  {/* Existing Models */}
                  {myModels.map((model: VoiceModel) => (
                    <button
                      key={model.id}
                      onClick={() => {
                        setSelectedModel(model);
                        setStep('select-language');
                      }}
                      className="p-4 rounded-lg border border-gray-700 hover:border-gray-600 bg-gray-800/50 transition-all text-left"
                    >
                      <div className="flex items-start gap-3">
                        <div className="w-10 h-10 rounded-lg bg-gray-700 flex items-center justify-center flex-shrink-0">
                          <Mic2 className="w-5 h-5 text-gray-400" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h3 className="font-medium truncate">{model.name}</h3>
                          <p className="text-sm text-gray-500 truncate">
                            {model.description || 'No description'}
                          </p>
                          {(model.en_readiness_score !== undefined && model.en_readiness_score !== null) && (
                            <div className="mt-2 flex items-center gap-2">
                              <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                                <div 
                                  className="h-full bg-primary-500 transition-all"
                                  style={{ width: `${model.en_readiness_score}%` }}
                                />
                              </div>
                              <span className="text-xs text-gray-400">
                                {Math.round(model.en_readiness_score)}% EN
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    </button>
                  ))}
                  
                  {myModels.length === 0 && (
                    <div className="col-span-full text-center py-8">
                      <FileAudio className="w-12 h-12 mx-auto text-gray-500 mb-4" />
                      <p className="text-gray-400">No models yet. Create your first one!</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Step 2: Select Language */}
          {step === 'select-language' && selectedModel && (
            <div className="space-y-6">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setStep('select-model')}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <div>
                  <h2 className="text-lg font-semibold">Select Training Language</h2>
                  <p className="text-sm text-gray-400">Training: {selectedModel.name}</p>
                </div>
              </div>
              
              {/* Recording count badge */}
              {modelRecordings && (
                <div className="flex items-center justify-center gap-2 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
                  <FileAudio className="w-5 h-5 text-primary-400" />
                  <span className="text-gray-300">
                    {modelRecordings.total_recordings > 0 
                      ? <><span className="font-semibold text-primary-400">{modelRecordings.total_recordings}</span> recordings ({Math.round(modelRecordings.total_duration_seconds / 60)} min)</>
                      : <span className="text-gray-500">No recordings yet</span>
                    }
                  </span>
                </div>
              )}

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {(languages || ['en', 'is']).map((lang) => (
                  <button
                    key={lang}
                    onClick={() => setSelectedLanguage(lang)}
                    className={`p-4 rounded-lg border transition-all ${
                      selectedLanguage === lang
                        ? 'border-primary-500 bg-primary-500/10'
                        : 'border-gray-700 hover:border-gray-600'
                    }`}
                  >
                    <div className="text-2xl mb-2">
                      {lang === 'en' ? '🇺🇸' : lang === 'is' ? '🇮🇸' : '🌍'}
                    </div>
                    <div className="font-medium">
                      {lang === 'en' ? 'English' : lang === 'is' ? 'Icelandic' : lang.toUpperCase()}
                    </div>
                  </button>
                ))}
              </div>
              
              <button
                onClick={() => setStep('training-areas')}
                className="w-full py-3 bg-primary-600 hover:bg-primary-500 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
              >
                Continue
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Step 3: Training Areas */}
          {step === 'training-areas' && selectedModel && (
            <div className="space-y-6">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setStep('select-language')}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <div className="flex-1">
                  <h2 className="text-lg font-semibold">Select Training Area</h2>
                  <p className="text-sm text-gray-400">
                    {selectedModel.name} • {selectedLanguage === 'en' ? 'English' : selectedLanguage === 'is' ? 'Icelandic' : selectedLanguage.toUpperCase()}
                  </p>
                </div>
                {/* Quick Start Wizard Button */}
                <button
                  onClick={() => startCategorySession('_wizard')}
                  disabled={isStartingSession}
                  className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-primary-600/50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
                >
                  {isStartingSession ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Wand2 className="w-4 h-4" />
                  )}
                  {isStartingSession ? 'Starting...' : 'Start Wizard'}
                </button>
              </div>

              {/* Recording count and quick actions */}
              <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                <div className="flex items-center gap-3">
                  <FileAudio className="w-5 h-5 text-primary-400" />
                  <div>
                    {modelRecordings && modelRecordings.total_recordings > 0 ? (
                      <>
                        <span className="font-semibold text-primary-400">{modelRecordings.total_recordings}</span>
                        <span className="text-gray-300"> recordings</span>
                        <span className="text-gray-500 ml-2">({Math.round(modelRecordings.total_duration_seconds / 60)} min)</span>
                      </>
                    ) : (
                      <span className="text-gray-400">No recordings yet</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => startCategorySession('_wizard')}
                    disabled={isStartingSession}
                    className="flex items-center gap-2 px-3 py-1.5 bg-green-600/20 hover:bg-green-600/30 border border-green-600/50 text-green-400 rounded-lg text-sm font-medium transition-colors"
                  >
                    <Mic className="w-4 h-4" />
                    Record
                  </button>
                  {modelRecordings && modelRecordings.total_recordings > 0 && (
                    <button
                      onClick={() => setStep('start-training')}
                      className="flex items-center gap-2 px-3 py-1.5 bg-primary-600/20 hover:bg-primary-600/30 border border-primary-600/50 text-primary-400 rounded-lg text-sm font-medium transition-colors"
                    >
                      <Play className="w-4 h-4" />
                      Train
                    </button>
                  )}
                </div>
              </div>

              {/* Training Area Cards */}
              <div className="space-y-4">
                {trainingAreas.map((area) => (
                  <div key={area.id} className="rounded-xl border border-gray-700 overflow-hidden">
                    {/* Area Header - Clickable */}
                    <button
                      onClick={() => setExpandedArea(expandedArea === area.id ? null : area.id)}
                      className={`w-full p-4 flex items-center gap-4 transition-colors ${
                        expandedArea === area.id 
                          ? 'bg-gray-800/70 border-b border-gray-700' 
                          : 'hover:bg-gray-800/50'
                      }`}
                    >
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        area.color === 'purple' ? 'bg-purple-500/20 text-purple-400' :
                        area.color === 'blue' ? 'bg-blue-500/20 text-blue-400' :
                        area.color === 'green' ? 'bg-green-500/20 text-green-400' :
                        'bg-orange-500/20 text-orange-400'
                      }`}>
                        {area.icon}
                      </div>
                      
                      <div className="flex-1 text-left">
                        <div className="flex items-center justify-between">
                          <h3 className="font-semibold">{area.name}</h3>
                          {area.score !== null ? (
                            <span className={`text-lg font-bold ${
                              area.score >= 80 ? 'text-green-400' :
                              area.score >= 50 ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                              {area.score}%
                            </span>
                          ) : (
                            <span className="text-sm text-gray-500 italic">Not analyzed</span>
                          )}
                        </div>
                        <p className="text-sm text-gray-400">{area.description}</p>
                        
                        {/* Progress bar */}
                        <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className={`h-full transition-all ${
                              area.color === 'purple' ? 'bg-purple-500' :
                              area.color === 'blue' ? 'bg-blue-500' :
                              area.color === 'green' ? 'bg-green-500' :
                              'bg-orange-500'
                            }`}
                            style={{ width: `${area.score ?? 0}%` }}
                          />
                        </div>
                      </div>
                      
                      {expandedArea === area.id ? (
                        <ChevronUp className="w-5 h-5 text-gray-400" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-gray-400" />
                      )}
                    </button>
                    
                    {/* Expanded Categories */}
                    {expandedArea === area.id && (
                      <div className="p-4 bg-gray-800/30">
                        <p className="text-sm text-gray-400 mb-4">
                          Select a category to start recording:
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {areaCategories.map((cat) => (
                            <button
                              key={cat.id}
                              onClick={() => startCategorySession(cat.id)}
                              disabled={isStartingSession}
                              className={`p-4 rounded-lg border transition-all text-left group ${
                                cat.modelHasCoverage 
                                  ? 'border-green-600/50 bg-green-900/10 hover:border-green-500' 
                                  : cat.hasRecordings
                                    ? 'border-yellow-600/50 bg-yellow-900/10 hover:border-yellow-500'
                                    : 'border-gray-600 hover:border-primary-500 bg-gray-800/50 hover:bg-primary-500/10'
                              } disabled:border-gray-700 disabled:bg-gray-800/30 disabled:cursor-not-allowed`}
                            >
                              <div className="flex items-center justify-between mb-2">
                                <h4 className="font-medium group-hover:text-primary-400 transition-colors flex items-center gap-2">
                                  {isStartingSession && selectedCategory === cat.id ? (
                                    <span className="flex items-center gap-2">
                                      <Loader2 className="w-4 h-4 animate-spin" />
                                      Starting...
                                    </span>
                                  ) : (
                                    <>
                                      {cat.name}
                                      {/* Status indicators */}
                                      {cat.modelHasCoverage && (
                                        <span title="Model already has this capability" className="text-green-400">
                                          <Check className="w-4 h-4" />
                                        </span>
                                      )}
                                      {!cat.modelHasCoverage && cat.hasRecordings && (
                                        <span title="Recordings added - will have capability after training" className="text-yellow-400">
                                          <Sparkles className="w-4 h-4" />
                                        </span>
                                      )}
                                    </>
                                  )}
                                </h4>
                                <div className="flex items-center gap-2">
                                  {/* Recording count badge */}
                                  {cat.recordingCount > 0 && (
                                    <span className="text-xs bg-yellow-600/30 text-yellow-300 px-2 py-1 rounded flex items-center gap-1">
                                      <FileAudio className="w-3 h-3" />
                                      {cat.recordingCount} recorded
                                    </span>
                                  )}
                                  <span className="text-xs bg-gray-700 px-2 py-1 rounded">
                                    {cat.promptCount} prompts
                                  </span>
                                </div>
                              </div>
                              <p className="text-sm text-gray-500">{cat.description}</p>
                              {(cat.phonemes?.length || 0) > 0 && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {(cat.phonemes || []).slice(0, 5).map((p, i) => (
                                    <span key={i} className="text-xs bg-gray-700/50 px-1.5 py-0.5 rounded text-gray-400">
                                      {p}
                                    </span>
                                  ))}
                                  {(cat.phonemes?.length || 0) > 5 && (
                                    <span className="text-xs text-gray-500">+{cat.phonemes.length - 5}</span>
                                  )}
                                </div>
                              )}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Import existing audio files section - OUTSIDE dropdowns */}
              <div className="rounded-xl border border-gray-700 p-6">
                <p className="text-sm text-gray-400 mb-3">Or import existing audio files:</p>
                <div 
                  className="border-2 border-dashed border-gray-600 hover:border-primary-500 rounded-xl p-6 text-center cursor-pointer transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept="audio/*,.wav,.mp3,.ogg,.flac"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <FolderUp className="w-8 h-8 mx-auto text-gray-500 mb-2" />
                  <p className="text-gray-300 text-sm">Click to select audio files</p>
                </div>
                
                {importFiles.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <h4 className="text-sm font-medium text-gray-300">
                      Selected files ({importFiles.length})
                    </h4>
                    <div className="max-h-32 overflow-y-auto space-y-1">
                      {importFiles.map((file, i) => (
                        <div key={i} className="flex items-center justify-between p-2 bg-gray-800/50 rounded-lg">
                          <div className="flex items-center gap-2 min-w-0">
                            <FileAudio className="w-4 h-4 text-gray-400 flex-shrink-0" />
                            <span className="text-sm truncate">{file.name}</span>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); removeImportFile(i); }}
                            className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ))}
                    </div>
                    
                    <button
                      onClick={uploadImportFiles}
                      disabled={importUploading}
                      className="w-full py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-gray-600 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                    >
                      {importUploading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Uploading... {uploadProgress > 0 && `${uploadProgress}%`}
                        </>
                      ) : (
                        <>
                          <Upload className="w-4 h-4" />
                          Upload &amp; Continue to Training
                        </>
                      )}
                    </button>
                    
                    {/* Upload progress bar */}
                    {importUploading && uploadProgress > 0 && (
                      <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                        <div 
                          className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${uploadProgress}%` }}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
              
              {/* Skip to training button if model already has data */}
              {(selectedModel.en_readiness_score || 0) > 0 && (
                <button
                  onClick={() => setStep('start-training')}
                  className="w-full py-3 border border-gray-600 hover:border-gray-500 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  <Zap className="w-4 h-4" />
                  Skip to Training (use existing data)
                </button>
              )}
            </div>
          )}

          {/* Step 4: Recording */}
          {step === 'recording' && session && currentPrompt && (
            <div className="space-y-6">
              {/* Header */}
              <div className="flex items-center gap-4">
                <button
                  onClick={backToAreas}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <div>
                  <h2 className="text-lg font-semibold">Recording Session</h2>
                  <p className="text-sm text-gray-400">
                    {selectedCategory?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </p>
                </div>
              </div>
              
              {/* Progress */}
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">
                  Prompt {session.current_index + 1} of {session.total_prompts}
                </span>
                <span className="text-gray-400">
                  {session.completed} recorded • {session.skipped} skipped
                </span>
              </div>
              
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary-500 transition-all"
                  style={{ width: `${((session.completed + session.skipped) / session.total_prompts) * 100}%` }}
                />
              </div>
              
              {/* Prompt Card */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-8 text-center">
                <p className="text-xl md:text-2xl font-medium leading-relaxed mb-4">
                  &ldquo;{currentPrompt.text}&rdquo;
                </p>
                {currentPrompt.ipa_text && (
                  <p className="text-sm text-gray-500 font-mono">
                    /{currentPrompt.ipa_text}/
                  </p>
                )}
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  {(currentPrompt.phonemes || []).slice(0, 10).map((phoneme, i) => (
                    <span 
                      key={i}
                      className="px-2 py-1 bg-gray-700/50 rounded text-xs text-gray-400"
                    >
                      {phoneme}
                    </span>
                  ))}
                </div>
              </div>
              
              {/* Recording Controls */}
              <div className="flex flex-col items-center gap-6">
                {isRecording && (
                  <div className="flex items-center gap-2 text-red-400">
                    <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                    <span className="font-medium">Recording...</span>
                  </div>
                )}
                
                {audioUrl && !isRecording && (
                  <div className="w-full max-w-md">
                    <audio src={audioUrl} controls className="w-full h-10" />
                  </div>
                )}
                
                <div className="flex items-center gap-4">
                  {!isRecording && !recordedAudio && (
                    <button
                      onClick={startRecording}
                      className="p-6 bg-red-600 hover:bg-red-500 rounded-full transition-colors shadow-lg shadow-red-600/30"
                    >
                      <Mic2 className="w-8 h-8" />
                    </button>
                  )}
                  
                  {isRecording && (
                    <button
                      onClick={stopRecording}
                      className="p-6 bg-gray-600 hover:bg-gray-500 rounded-full transition-colors"
                    >
                      <Square className="w-8 h-8" />
                    </button>
                  )}
                  
                  {recordedAudio && !isRecording && (
                    <>
                      <button
                        onClick={clearRecording}
                        className="p-4 bg-gray-700 hover:bg-gray-600 rounded-full transition-colors"
                        title="Re-record"
                      >
                        <RotateCcw className="w-6 h-6" />
                      </button>
                      
                      <button
                        onClick={submitRecording}
                        disabled={isSubmitting}
                        className="p-6 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 rounded-full transition-colors shadow-lg shadow-green-600/30"
                      >
                        {isSubmitting ? (
                          <Loader2 className="w-8 h-8 animate-spin" />
                        ) : (
                          <Check className="w-8 h-8" />
                        )}
                      </button>
                    </>
                  )}
                </div>
                
                <div className="flex items-center gap-4">
                  <button
                    onClick={skipPrompt}
                    disabled={isRecording || isSubmitting}
                    className="px-4 py-2 text-gray-400 hover:text-white disabled:opacity-50 transition-colors"
                  >
                    Skip this prompt
                  </button>
                  
                  {session.completed >= 3 && (
                    <button
                      onClick={finishRecording}
                      className="px-4 py-2 text-primary-400 hover:text-primary-300 transition-colors"
                    >
                      Finish &amp; Start Training
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Step 5: Start Training */}
          {step === 'start-training' && (
            <div className="space-y-6 text-center">
              <div className="w-16 h-16 mx-auto bg-primary-600/20 rounded-full flex items-center justify-center">
                <Zap className="w-8 h-8 text-primary-400" />
              </div>
              
              <div>
                <h2 className="text-xl font-semibold mb-2">Ready to Train</h2>
                <p className="text-gray-400">
                  {modelRecordings && modelRecordings.total_recordings > 0
                    ? `You have ${modelRecordings.total_recordings} recordings (${Math.round(modelRecordings.total_duration_seconds / 60)} min). Ready to train your model!`
                    : filesUploadedCount > 0
                      ? `You uploaded ${filesUploadedCount} audio file${filesUploadedCount > 1 ? 's' : ''}. Ready to train your model!`
                      : session 
                        ? `You recorded ${session.completed} samples. Ready to train your model!`
                        : 'Your audio files have been uploaded. Ready to train your model!'
                  }
                </p>
              </div>

              {/* Show checkpoint info - training can be resumed */}
              {modelTrainingInfo?.training?.latest_checkpoint && (
                <div className="max-w-xl mx-auto p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <RotateCcw className="w-5 h-5 text-blue-400" />
                    <span className="font-medium text-blue-300">Previous Training Found (Epoch {modelTrainingInfo.training.epochs_trained || 0})</span>
                  </div>
                  <p className="text-sm text-gray-400 mb-4 text-center">
                    You can continue from this checkpoint or start fresh with new data.
                  </p>
                  
                  {/* Checkpoint Training Options */}
                  <div className="space-y-2">
                    <button
                      onClick={() => setCheckpointMode('continue')}
                      className={`w-full p-3 rounded-lg border transition-all text-left ${
                        checkpointMode === 'continue'
                          ? 'bg-green-500/20 border-green-500/50 text-green-300'
                          : 'bg-gray-800/50 border-gray-700 hover:border-gray-600 text-gray-300'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`w-5 h-5 mt-0.5 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                          checkpointMode === 'continue' ? 'border-green-400' : 'border-gray-500'
                        }`}>
                          {checkpointMode === 'continue' && <div className="w-2.5 h-2.5 bg-green-400 rounded-full" />}
                        </div>
                        <div>
                          <div className="font-medium">Continue Training</div>
                          <div className="text-xs text-gray-500">Use existing audio files and continue from checkpoint</div>
                        </div>
                      </div>
                    </button>
                    
                    <button
                      onClick={() => setCheckpointMode('add-audio')}
                      className={`w-full p-3 rounded-lg border transition-all text-left ${
                        checkpointMode === 'add-audio'
                          ? 'bg-purple-500/20 border-purple-500/50 text-purple-300'
                          : 'bg-gray-800/50 border-gray-700 hover:border-gray-600 text-gray-300'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`w-5 h-5 mt-0.5 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                          checkpointMode === 'add-audio' ? 'border-purple-400' : 'border-gray-500'
                        }`}>
                          {checkpointMode === 'add-audio' && <div className="w-2.5 h-2.5 bg-purple-400 rounded-full" />}
                        </div>
                        <div>
                          <div className="font-medium">Add New Audio + Continue</div>
                          <div className="text-xs text-gray-500">Keep existing audio, add new recordings, continue from checkpoint</div>
                        </div>
                      </div>
                    </button>
                    
                    <button
                      onClick={() => setCheckpointMode('new-audio')}
                      className={`w-full p-3 rounded-lg border transition-all text-left ${
                        checkpointMode === 'new-audio'
                          ? 'bg-orange-500/20 border-orange-500/50 text-orange-300'
                          : 'bg-gray-800/50 border-gray-700 hover:border-gray-600 text-gray-300'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`w-5 h-5 mt-0.5 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                          checkpointMode === 'new-audio' ? 'border-orange-400' : 'border-gray-500'
                        }`}>
                          {checkpointMode === 'new-audio' && <div className="w-2.5 h-2.5 bg-orange-400 rounded-full" />}
                        </div>
                        <div>
                          <div className="font-medium">New Audio Only</div>
                          <div className="text-xs text-gray-500">Replace existing audio with new recordings, continue from checkpoint</div>
                        </div>
                      </div>
                    </button>
                  </div>
                  
                  {/* Show guidance based on selected mode */}
                  {checkpointMode === 'add-audio' && (
                    <div className="mt-3 p-2 bg-purple-500/10 rounded text-xs text-purple-300 text-center">
                      ⚠️ Adding new audio requires re-preprocessing all files. Model will continue from checkpoint with combined dataset.
                    </div>
                  )}
                  {checkpointMode === 'new-audio' && (
                    <div className="mt-3 p-2 bg-orange-500/10 rounded text-xs text-orange-300 text-center">
                      ⚠️ This will use the checkpoint weights but retrain on completely new audio data
                    </div>
                  )}
                </div>
              )}

              {/* Training Configuration */}
              <div className="max-w-md mx-auto p-4 bg-gray-800/50 border border-gray-700 rounded-lg space-y-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Training Configuration</h3>
                
                {/* Auto-config toggle */}
                <div className="flex items-center justify-between pb-3 border-b border-gray-700">
                  <div>
                    <label className="text-sm text-gray-400">Use Auto-Configuration</label>
                    <p className="text-xs text-gray-500">Let the system optimize based on your audio</p>
                  </div>
                  <button
                    onClick={() => setUseAutoConfig(!useAutoConfig)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${useAutoConfig ? 'bg-green-500' : 'bg-gray-600'}`}
                  >
                    <span className={`absolute w-5 h-5 bg-white rounded-full top-0.5 transition-transform ${useAutoConfig ? 'translate-x-6' : 'translate-x-0.5'}`} />
                  </button>
                </div>

                {/* Manual config (shown when auto-config is off) */}
                {!useAutoConfig && (
                  <>
                    {/* Epochs */}
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm text-gray-400">Target Epochs</label>
                        <p className="text-xs text-gray-500">More = better quality, longer training</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setTargetEpochs(Math.max(10, targetEpochs - 10))}
                          className="w-8 h-8 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 flex items-center justify-center"
                        >
                          -
                        </button>
                        <input
                          type="number"
                          value={targetEpochs}
                          onChange={(e) => setTargetEpochs(Math.max(10, Math.min(1000, parseInt(e.target.value) || 200)))}
                          className="w-20 px-2 py-1 bg-gray-900 border border-gray-600 rounded text-center text-white"
                        />
                        <button
                          onClick={() => setTargetEpochs(Math.min(1000, targetEpochs + 10))}
                          className="w-8 h-8 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 flex items-center justify-center"
                        >
                          +
                        </button>
                      </div>
                    </div>
                    
                    {/* Batch Size */}
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm text-gray-400">Batch Size</label>
                        <p className="text-xs text-gray-500">Lower = more steps, better for small datasets</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setBatchSize(Math.max(2, batchSize - 2))}
                          className="w-8 h-8 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 flex items-center justify-center"
                        >
                          -
                        </button>
                        <input
                          type="number"
                          value={batchSize}
                          onChange={(e) => setBatchSize(Math.max(2, Math.min(16, parseInt(e.target.value) || 6)))}
                          className="w-20 px-2 py-1 bg-gray-900 border border-gray-600 rounded text-center text-white"
                        />
                        <button
                          onClick={() => setBatchSize(Math.min(16, batchSize + 2))}
                          className="w-8 h-8 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 flex items-center justify-center"
                        >
                          +
                        </button>
                      </div>
                    </div>
                    
                    {/* Epoch recommendations */}
                    <div className="pt-2 border-t border-gray-700">
                      <p className="text-xs text-gray-500 mb-2">Quick presets:</p>
                      <div className="flex gap-2">
                        <button
                          onClick={() => { setTargetEpochs(100); setBatchSize(8); }}
                          className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
                        >
                          Quick (100)
                        </button>
                        <button
                          onClick={() => { setTargetEpochs(200); setBatchSize(6); }}
                          className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-500 rounded text-white"
                        >
                          Standard (200)
                        </button>
                        <button
                          onClick={() => { setTargetEpochs(300); setBatchSize(4); }}
                          className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
                        >
                          Quality (300)
                        </button>
                      </div>
                    </div>
                  </>
                )}
                
                {/* Auto-config info */}
                {useAutoConfig && (
                  <div className="p-3 bg-blue-500/10 rounded-lg">
                    <p className="text-xs text-blue-300">
                      🤖 The system will analyze your audio and automatically set optimal epochs and batch size based on duration, quality, and training type.
                    </p>
                  </div>
                )}
                
                {modelTrainingInfo?.training?.epochs_trained && !useAutoConfig && targetEpochs <= modelTrainingInfo.training.epochs_trained && (
                  <p className="mt-2 text-xs text-yellow-400">
                    ⚠️ Target ({targetEpochs}) is less than already trained ({modelTrainingInfo.training.epochs_trained}). Increase to train more.
                  </p>
                )}
              </div>
              
              {/* Show total recordings from all sessions */}
              {modelRecordings && modelRecordings.total_recordings > 0 && (
                <div className="space-y-4 max-w-2xl mx-auto">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 bg-gray-800/50 rounded-lg">
                      <div className="text-2xl font-bold text-green-400">{modelRecordings.total_recordings}</div>
                      <div className="text-sm text-gray-400">Total Recordings</div>
                    </div>
                    <div className="p-4 bg-gray-800/50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-400">{Math.round(modelRecordings.total_duration_seconds / 60)}</div>
                      <div className="text-sm text-gray-400">Minutes</div>
                    </div>
                    <div className="p-4 bg-gray-800/50 rounded-lg">
                      <div className="text-2xl font-bold text-purple-400">{Object.keys(modelRecordings.categories || {}).length}</div>
                      <div className="text-sm text-gray-400">Categories</div>
                    </div>
                  </div>
                  
                  {/* Expandable Recording Files List */}
                  <details className="bg-gray-800/30 border border-gray-700 rounded-lg">
                    <summary className="px-4 py-3 cursor-pointer flex items-center justify-between hover:bg-gray-800/50 rounded-lg transition-colors">
                      <span className="flex items-center gap-2 text-sm text-gray-300">
                        <FileAudio className="w-4 h-4" />
                        View Recording Files
                      </span>
                      <ChevronDown className="w-4 h-4 text-gray-500" />
                    </summary>
                    <div className="px-4 pb-4 max-h-64 overflow-y-auto">
                      <div className="space-y-1">
                        {modelRecordings.audio_paths.map((path, index) => {
                          const fileName = path.split('/').pop() || path;
                          return (
                            <div key={index} className="flex items-center gap-2 py-2 px-3 bg-gray-900/50 rounded text-sm">
                              <Music className="w-3 h-3 text-gray-500 flex-shrink-0" />
                              <span className="text-gray-400 truncate flex-1">{fileName}</span>
                              <span className="text-xs text-gray-600">#{index + 1}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </details>
                </div>
              )}

              {/* Show uploaded files count if files were uploaded but not tracked as recordings */}
              {(!modelRecordings || modelRecordings.total_recordings === 0) && filesUploadedCount > 0 && (
                <div className="grid grid-cols-1 gap-4 max-w-xs mx-auto">
                  <div className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="text-2xl font-bold text-green-400">{filesUploadedCount}</div>
                    <div className="text-sm text-gray-400">Files Uploaded</div>
                  </div>
                </div>
              )}

              {/* Show current session stats if no total recordings and no uploaded files */}
              {(!modelRecordings || modelRecordings.total_recordings === 0) && filesUploadedCount === 0 && session && (
                <div className="grid grid-cols-2 gap-4 max-w-sm mx-auto">
                  <div className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="text-2xl font-bold text-green-400">{session.completed}</div>
                    <div className="text-sm text-gray-400">Recorded</div>
                  </div>
                  <div className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="text-2xl font-bold text-yellow-400">{session.skipped}</div>
                    <div className="text-sm text-gray-400">Skipped</div>
                  </div>
                </div>
              )}

              {/* Error message */}
              {error && (
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm max-w-md mx-auto">
                  {error}
                </div>
              )}
              
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
                <button
                  onClick={() => {
                    setError(null);
                    setStep('training-areas');
                  }}
                  className="px-6 py-3 border border-gray-600 hover:border-gray-500 rounded-lg transition-colors"
                >
                  Add More Data
                </button>
                {/* Show Continue Training button if checkpoint exists and continue mode is selected */}
                {modelTrainingInfo?.training?.latest_checkpoint && (
                  <button
                    onClick={() => startTraining('continue')}
                    disabled={checkpointMode === 'add-audio' || checkpointMode === 'new-audio'}
                    className="px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-medium transition-colors flex items-center gap-2"
                    title={checkpointMode === 'add-audio' || checkpointMode === 'new-audio' ? 'Cannot continue - new audio requires fresh training' : ''}
                  >
                    <RotateCcw className="w-5 h-5" />
                    Continue Training
                  </button>
                )}
                <button
                  onClick={() => startTraining('fresh')}
                  disabled={
                    // Allow if: checkpoint exists OR files were uploaded OR have 10+ recordings OR have 2+ min of audio
                    !(modelTrainingInfo?.training?.latest_checkpoint ||
                      filesUploadedCount > 0 || 
                      (modelRecordings && (modelRecordings.total_recordings >= 10 || modelRecordings.total_duration_seconds >= 120)))
                  }
                  className="px-6 py-3 bg-primary-600 hover:bg-primary-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <Play className="w-5 h-5" />
                  {modelTrainingInfo?.training?.latest_checkpoint ? 'Start Fresh' : 'Start Training'}
                </button>
              </div>

              {/* Show requirement info - only for recordings mode, not file uploads */}
              {!modelTrainingInfo?.training?.latest_checkpoint && filesUploadedCount === 0 && modelRecordings && modelRecordings.total_recordings < 10 && modelRecordings.total_duration_seconds < 120 && (
                <p className="text-sm text-yellow-400">
                  Need at least 10 recordings or 2 minutes of audio to start training. Currently have {modelRecordings.total_recordings} recordings ({Math.round(modelRecordings.total_duration_seconds)} sec).
                </p>
              )}

              {/* Training History Section */}
              {selectedModel?.slug && (
                <div className="max-w-4xl mx-auto mt-8 pt-6 border-t border-gray-700">
                  <TrainingRunsHistory 
                    modelSlug={selectedModel.slug}
                    legacyTrainingInfo={modelTrainingInfo}
                    onStartTraining={(mode, options) => {
                      // When resuming/continuing from history, go to progress view
                      setStep('training-progress');
                      startTraining();
                    }}
                  />
                </div>
              )}
            </div>
          )}

          {/* Step 6: Training Progress */}
          {step === 'training-progress' && (
            <div className="space-y-6 text-center">
              <div className="w-16 h-16 mx-auto bg-primary-600/20 rounded-full flex items-center justify-center">
                {trainingProgress >= 100 ? (
                  <Check className="w-8 h-8 text-green-400" />
                ) : error ? (
                  <AlertCircle className="w-8 h-8 text-red-400" />
                ) : trainingStatus.includes('cancelled') ? (
                  <RotateCcw className="w-8 h-8 text-yellow-400" />
                ) : (
                  <Activity className="w-8 h-8 text-primary-400 animate-pulse" />
                )}
              </div>
              
              <div>
                <h2 className="text-xl font-semibold mb-2">
                  {trainingProgress >= 100 
                    ? 'Training Complete!' 
                    : error 
                      ? 'Training Failed'
                      : trainingStatus.includes('cancelled')
                        ? 'Training Cancelled'
                        : 'Training in Progress'}
                </h2>
                <p className="text-gray-400">{trainingStatus}</p>
              </div>
              
              <div className="max-w-md mx-auto">
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-gray-400">Progress</span>
                  <span className="font-medium">{Math.round(trainingProgress)}%</span>
                </div>
                <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all ${
                      trainingProgress >= 100 ? 'bg-green-500' 
                        : error ? 'bg-red-500'
                        : trainingStatus.includes('cancelled') ? 'bg-yellow-500'
                        : 'bg-primary-500'
                    }`}
                    style={{ width: `${trainingProgress}%` }}
                  />
                </div>
                
                {/* Detailed training stats */}
                {trainingDetails && trainingProgress < 100 && !error && !trainingStatus.includes('cancelled') && (
                  <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                    {trainingDetails.current_epoch > 0 && (
                      <>
                        <div className="p-2 bg-gray-800/50 rounded">
                          <div className="text-lg font-bold text-blue-400">{trainingDetails.current_epoch}</div>
                          <div className="text-xs text-gray-500">Current Epoch</div>
                        </div>
                        <div className="p-2 bg-gray-800/50 rounded">
                          <div className="text-lg font-bold text-gray-300">{trainingDetails.total_epochs}</div>
                          <div className="text-xs text-gray-500">Target Epochs</div>
                        </div>
                      </>
                    )}
                    <div className="p-2 bg-gray-800/50 rounded">
                      <div className="text-sm font-medium text-purple-400 capitalize">{trainingDetails.step || trainingDetails.status}</div>
                      <div className="text-xs text-gray-500">Current Step</div>
                    </div>
                    {trainingDetails.started_at && (
                      <div className="p-2 bg-gray-800/50 rounded">
                        <div className="text-sm font-medium text-orange-400">
                          {Math.round((Date.now() - new Date(trainingDetails.started_at).getTime()) / 60000)}m
                        </div>
                        <div className="text-xs text-gray-500">Elapsed</div>
                      </div>
                    )}
                  </div>
                )}
                
                {/* Progress breakdown */}
                {trainingProgress > 0 && trainingProgress < 100 && !error && (
                  <div className="mt-4 text-xs text-gray-500">
                    <div className="flex justify-between mb-1">
                      <span className={trainingProgress >= 5 ? 'text-green-400' : ''}>Preprocess</span>
                      <span className={trainingProgress >= 25 ? 'text-green-400' : ''}>F0 Extract</span>
                      <span className={trainingProgress >= 40 ? 'text-green-400' : ''}>Features</span>
                      <span className={trainingProgress >= 50 ? 'text-green-400' : ''}>Training</span>
                      <span className={trainingProgress >= 95 ? 'text-green-400' : ''}>Index</span>
                    </div>
                    <div className="flex gap-1">
                      <div className={`h-1 flex-1 rounded ${trainingProgress >= 5 ? 'bg-green-500' : 'bg-gray-600'}`} />
                      <div className={`h-1 flex-1 rounded ${trainingProgress >= 25 ? 'bg-green-500' : 'bg-gray-600'}`} />
                      <div className={`h-1 flex-1 rounded ${trainingProgress >= 40 ? 'bg-green-500' : 'bg-gray-600'}`} />
                      <div className={`h-1 flex-[3] rounded ${trainingProgress >= 95 ? 'bg-green-500' : trainingProgress >= 50 ? 'bg-blue-500' : 'bg-gray-600'}`} />
                      <div className={`h-1 flex-1 rounded ${trainingProgress >= 95 ? 'bg-green-500' : 'bg-gray-600'}`} />
                    </div>
                  </div>
                )}
              </div>

              {/* Show checkpoint info for failed/cancelled training */}
              {(error || trainingStatus.includes('cancelled')) && modelTrainingInfo?.training?.latest_checkpoint && (
                <div className="max-w-md mx-auto p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <RotateCcw className="w-5 h-5 text-blue-400" />
                    <span className="font-medium text-blue-300">Checkpoint Saved</span>
                  </div>
                  <p className="text-sm text-gray-400 mb-3">
                    Training progress was saved at epoch {modelTrainingInfo.training.epochs_trained || 0}. 
                    You can continue from this checkpoint to complete training.
                  </p>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="p-2 bg-gray-800/50 rounded">
                      <div className="text-lg font-bold text-blue-400">{modelTrainingInfo.training.epochs_trained || 0}</div>
                      <div className="text-xs text-gray-500">Epochs Completed</div>
                    </div>
                    <div className="p-2 bg-gray-800/50 rounded">
                      <div className="text-lg font-bold text-orange-400">{(modelTrainingInfo.training.target_epochs || 100) - (modelTrainingInfo.training.epochs_trained || 0)}</div>
                      <div className="text-xs text-gray-500">Epochs Remaining</div>
                    </div>
                  </div>
                </div>
              )}
              
              {trainingProgress >= 100 && (
                <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
                  <button
                    onClick={() => setStep('training-areas')}
                    className="px-6 py-3 border border-gray-600 hover:border-gray-500 rounded-lg transition-colors"
                  >
                    Train More
                  </button>
                  <button
                    onClick={() => router.push('/dashboard/models?tab=my-models')}
                    className="px-6 py-3 bg-primary-600 hover:bg-primary-500 rounded-lg font-medium transition-colors"
                  >
                    View My Models
                  </button>
                </div>
              )}

              {/* Show continue/retry buttons for failed/cancelled training */}
              {(error || trainingStatus.includes('cancelled')) && (
                <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
                  <button
                    onClick={() => {
                      setError(null);
                      setStep('start-training');
                      refetchTrainingInfo();
                    }}
                    className="px-6 py-3 border border-gray-600 hover:border-gray-500 rounded-lg transition-colors"
                  >
                    Back to Training Options
                  </button>
                  {modelTrainingInfo?.training?.latest_checkpoint && (
                    <button
                      onClick={() => {
                        setError(null);
                        setTrainingProgress(0);
                        setTrainingStatus('Resuming training...');
                        startTraining();
                      }}
                      className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-lg font-medium transition-colors flex items-center gap-2"
                    >
                      <RotateCcw className="w-5 h-5" />
                      Continue Training
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
