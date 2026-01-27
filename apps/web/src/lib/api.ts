import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
const BASE_URL = process.env.NEXT_PUBLIC_API_URL?.replace('/api', '') || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  withCredentials: true,
  withXSRFToken: true,
  timeout: 300000,  // 5 minutes for large file uploads
});

// Fetch CSRF cookie from Sanctum
export const getCsrfCookie = async () => {
  await axios.get(`${BASE_URL}/sanctum/csrf-cookie`, { withCredentials: true });
};

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = typeof window !== 'undefined' 
    ? localStorage.getItem('auth_token') 
    : null;
  
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('auth_token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// =============================================================================
// Types
// =============================================================================

export interface User {
  id: string;
  name: string;
  email: string;
  roles?: string[];
  permissions?: string[];
}

/**
 * Unified VoiceModel type - covers both system models and user-uploaded models.
 * System models have user_id = null, user-uploaded models have user_id set.
 */
export interface VoiceModel {
  id: number;
  uuid: string;
  slug: string;
  name: string;
  description: string | null;
  image_path: string | null;  // Cover image path
  image_url: string | null;   // Full URL to image
  gender: string | null;      // Optional: Male/Female for TTS auto-selection

  // Ownership
  user_id: number | null; // null = system model
  user?: { id: number; name: string } | null;

  // Storage (paths hidden from API)
  model_path: string | null;
  index_path: string | null;
  config_path: string | null;
  has_index: boolean;
  size_bytes: number;
  storage_type: "local" | "s3";

  // Computed from paths (appended by API)
  model_file: string | null;
  index_file: string | null;
  size: string; // Human readable size

  // Metadata
  engine: string;
  visibility: "public" | "private" | "unlisted";
  status: "pending" | "processing" | "ready" | "failed" | "training";
  tags: string[] | null;
  metadata: Record<string, any> | null;

  // Flags
  is_active: boolean;
  is_featured: boolean;
  has_consent: boolean;
  consent_notes: string | null;

  // Stats
  usage_count: number;
  download_count: number;

  // Language readiness scores (from trainer scan)
  en_readiness_score?: number | null;
  en_phoneme_coverage?: number | null;
  en_missing_phonemes?: string[] | null;
  is_readiness_score?: number | null;
  is_phoneme_coverage?: number | null;
  is_missing_phonemes?: string[] | null;
  language_scanned_at?: string | null;

  // Training progress (when status === 'training')
  training_progress?: number;
  training_job_id?: string | null;
  training_epoch?: number | null;
  training_total_epochs?: number | null;

  // URLs (optional, may be included by API)
  download_url?: string | null;
  index_download_url?: string | null;

  // Timestamps
  last_synced_at: string | null;
  created_at: string;
  updated_at: string;
}

// Helper to check if a model is a system model
export const isSystemModel = (model: VoiceModel): boolean => model.user_id === null;

// Legacy alias for backwards compatibility
export type SystemVoiceModel = VoiceModel;

export interface Job {
  id: string;
  type: 'inference' | 'training' | 'preprocessing';
  status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress?: number;
  error_message?: string;
  voice_model: VoiceModel;
  created_at: string;
  completed_at?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  per_page: number;
  current_page: number;
  last_page: number;
}

// =============================================================================
// Training Runs Types (Git-like training workflow)
// =============================================================================

export type TrainingRunMode = 'new' | 'resume' | 'continue' | 'branch';
export type TrainingRunStatus = 'pending' | 'preparing' | 'training' | 'paused' | 'completed' | 'failed' | 'cancelled';
export type DatasetVersionStatus = 'pending' | 'processing' | 'ready' | 'failed';

export interface TrainingRun {
  id: number;
  uuid: string;
  run_number: string;
  mode: TrainingRunMode;
  mode_display: string;
  status: TrainingRunStatus;
  target_epochs: number;
  completed_epochs: number;
  start_epoch: number;
  epochs_trained: number;
  best_loss: number | null;
  duration: string;
  duration_seconds?: number;
  can_resume: boolean;
  is_active: boolean;
  config_snapshot?: TrainingConfig;
  error_message?: string;
  metadata?: Record<string, any>;
  parent_run?: { id: number; run_number: string };
  parent_checkpoint?: TrainingCheckpoint;
  dataset_version?: DatasetVersion;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface TrainingCheckpoint {
  id: number;
  uuid: string;
  epoch: number;
  global_step: number;
  checkpoint_name: string;
  short_name: string;
  loss_g: number | null;
  loss_d?: number | null;
  loss_mel?: number | null;
  loss_kl?: number | null;
  loss_fm?: number | null;
  primary_loss: number | null;
  total_loss?: number | null;
  flags: string[];
  is_milestone: boolean;
  is_best: boolean;
  is_final: boolean;
  is_archived: boolean;
  file_size: string;
  notes: string | null;
  generator_path?: string;
  discriminator_path?: string;
  checkpoint_directory?: string;
  metadata?: Record<string, any>;
  created_at: string;
}

export interface DatasetVersion {
  id: number;
  uuid: string;
  version_number: number;
  audio_count: number;
  duration: string;
  total_duration_seconds?: number;
  segment_count: number;
  sample_rate: number;
  status: DatasetVersionStatus;
  is_latest: boolean;
  manifest_hash?: string;
  preprocessing_config?: Record<string, any>;
  metadata?: Record<string, any>;
  created_at: string;
}

export interface TrainingConfig {
  epochs?: number;
  sample_rate?: 32000 | 40000 | 48000;
  f0_method?: 'rmvpe' | 'pm' | 'harvest';
  batch_size?: number;
  save_every_epoch?: number;
  [key: string]: any;
}

export interface TrainingTreeNode {
  run: TrainingRun;
  checkpoints: TrainingCheckpoint[];
  children: TrainingTreeNode[];
}

// =============================================================================
// Auth API
// =============================================================================

export const authApi = {
  register: async (data: { name: string; email: string; password: string; password_confirmation: string }) => {
    await getCsrfCookie();
    const response = await api.post('/auth/register', data);
    return response.data;
  },

  login: async (data: { email: string; password: string }) => {
    await getCsrfCookie();
    const response = await api.post('/auth/login', data);
    return response.data;
  },

  logout: async () => {
    const response = await api.post('/auth/logout');
    return response.data;
  },

  me: async (): Promise<{ user: User }> => {
    const response = await api.get('/auth/me');
    return response.data;
  },

  // OAuth methods
  getOAuthUrl: async (provider: 'google' | 'github'): Promise<{ url: string }> => {
    const response = await api.get(`/oauth/${provider}/redirect`);
    return response.data;
  },

  handleOAuthCallback: async (provider: 'google' | 'github', code: string) => {
    const response = await api.post(`/oauth/${provider}/callback`, { code });
    return response.data;
  },
};

// =============================================================================
// Voice Models API (Unified - handles both system and user-uploaded models)
// =============================================================================

export interface VoiceModelListParams {
  type?: 'system' | 'user';
  search?: string;
  engine?: string;
  storage_type?: 'local' | 's3';
  has_index?: boolean;
  featured?: boolean;
  tags?: string[];
  sort?: 'name' | 'size_bytes' | 'usage_count' | 'created_at' | 'updated_at';
  direction?: 'asc' | 'desc';
  per_page?: number;
  page?: number;
  all?: boolean;
  include_pending?: boolean;
  include_inactive?: boolean;
  public_only?: boolean; // Only return public models (ignores auth)
}

export const voiceModelsApi = {
  /**
   * List voice models. Use type='system' for server-side models, type='user' for user-uploaded.
   * Use public_only=true to only show public models (for public browsing pages).
   */
  list: async (params?: VoiceModelListParams): Promise<{ data: VoiceModel[]; total: number }> => {
    const response = await api.get('/voice-models', { params });
    return response.data;
  },

  /**
   * List only public voice models (shorthand for list with public_only=true)
   */
  listPublic: async (params?: Omit<VoiceModelListParams, 'public_only'>): Promise<{ data: VoiceModel[]; total: number }> => {
    const response = await api.get('/voice-models', { params: { ...params, public_only: true } });
    return response.data;
  },

  /**
   * Get a single model by slug
   */
  get: async (slug: string): Promise<{ model: VoiceModel }> => {
    const response = await api.get(`/voice-models/${slug}`);
    return response.data;
  },

  /**
   * Get user's own uploaded models
   */
  myModels: async (params?: { page?: number; per_page?: number }) => {
    const response = await api.get('/voice-models/my', { params });
    return response.data;
  },

  /**
   * Create a new voice model (user-uploaded)
   */
  create: async (data: { 
    name: string; 
    description?: string; 
    engine?: string;
    visibility?: 'public' | 'private' | 'unlisted';
    tags?: string[];
    has_consent?: boolean;
    consent_notes?: string;
  }) => {
    const response = await api.post('/voice-models', data);
    return response.data;
  },

  /**
   * Update model metadata
   */
  update: async (slugOrId: string, data: Partial<Pick<VoiceModel, 'name' | 'description' | 'visibility' | 'tags' | 'is_active' | 'is_featured'>>) => {
    const response = await api.patch(`/voice-models/${slugOrId}`, data);
    return response.data;
  },

  /**
   * Delete a model (user-uploaded only)
   */
  delete: async (slugOrId: string) => {
    const response = await api.delete(`/voice-models/${slugOrId}`);
    return response.data;
  },

  /**
   * Get presigned upload URLs for model files
   */
  getUploadUrls: async (slugOrId: string) => {
    const response = await api.post(`/voice-models/${slugOrId}/upload-urls`);
    return response.data;
  },

  /**
   * Confirm upload completed
   */
  confirmUpload: async (slugOrId: string, data: { model_uploaded: boolean; index_uploaded?: boolean }) => {
    const response = await api.post(`/voice-models/${slugOrId}/confirm-upload`, data);
    return response.data;
  },

  /**
   * Get download URLs for model files
   */
  getDownloadUrls: async (slugOrId: string) => {
    const response = await api.get(`/voice-models/${slugOrId}/download-urls`);
    return response.data;
  },

  /**
   * Get stats
   */
  stats: async (params?: { system?: boolean }) => {
    const response = await api.get('/voice-models/stats', { params });
    return response.data;
  },

  /**
   * Sync system models from storage (admin only)
   */
  sync: async (params?: { prune?: boolean; storage?: 'local' | 's3' }) => {
    const response = await api.post('/admin/voice-models/sync', params);
    return response.data;
  },

  /**
   * Get storage config (admin only)
   */
  config: async () => {
    const response = await api.get('/admin/voice-models/config');
    return response.data;
  },

  /**
   * Upload model files directly (multipart form)
   */
  upload: async (data: FormData, onProgress?: (progress: number) => void) => {
    const response = await api.post('/voice-models/upload', data, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
    return response.data;
  },

  /**
   * Upload model image
   */
  uploadImage: async (slugOrId: string, imageFile: File) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await api.post(`/voice-models/${slugOrId}/image`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Upload additional files to existing model
   */
  uploadFiles: async (slugOrId: string, data: FormData) => {
    const response = await api.post(`/voice-models/${slugOrId}/files`, data, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Replace model file
   */
  replaceModel: async (slugOrId: string, data: FormData) => {
    const response = await api.post(`/voice-models/${slugOrId}/replace`, data, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

// Legacy alias - modelsApi points to the same functionality
export const modelsApi = voiceModelsApi;

// =============================================================================
// TTS API
// =============================================================================

export interface TTSVoice {
  id: string;
  name: string;
  language: string;
  gender: 'male' | 'female';
  supports_styles?: boolean;
}

export interface TTSStyle {
  id: string;
  name?: string;
  description: string;
}

export interface TTSRequest {
  text: string;
  voice?: string;
  style?: string;
  rate?: string;
  pitch?: string;
  voice_model_id?: number;
  f0_up_key?: number;
  index_rate?: number;
}

export interface TTSResponse {
  audio: string; // Base64 encoded audio
  sample_rate: number;
  format: string;
  text_length: number;
  voice: string;
  style: string;
  converted: boolean;
}

export const ttsApi = {
  /**
   * Get available TTS voices, styles, and languages
   */
  getVoices: async (): Promise<{ voices: TTSVoice[]; styles: TTSStyle[]; languages: string[] }> => {
    const response = await api.get('/tts/voices');
    return response.data;
  },

  /**
   * Generate TTS audio
   */
  generate: async (request: TTSRequest): Promise<TTSResponse> => {
    const response = await api.post('/tts/generate', request);
    return response.data;
  },

  /**
   * Stream TTS audio (for longer texts)
   */
  stream: async (request: TTSRequest) => {
    const response = await api.post('/tts/stream', request, {
      responseType: 'stream',
    });
    return response.data;
  },
};

// =============================================================================
// Jobs API
// =============================================================================

export const jobsApi = {
  list: async (params?: { page?: number; status?: string }) => {
    const response = await api.get('/jobs', { params });
    return response.data;
  },

  get: async (id: string) => {
    const response = await api.get(`/jobs/${id}`);
    return response.data;
  },

  createInference: async (data: { voice_model_id: string | number; parameters?: object }) => {
    const response = await api.post('/jobs/inference', data);
    return response.data;
  },

  getUploadUrl: async (id: string) => {
    const response = await api.post(`/jobs/${id}/upload-url`);
    return response.data;
  },

  start: async (id: string) => {
    const response = await api.post(`/jobs/${id}/start`);
    return response.data;
  },

  cancel: async (id: string) => {
    const response = await api.post(`/jobs/${id}/cancel`);
    return response.data;
  },

  getOutput: async (id: string) => {
    const response = await api.get(`/jobs/${id}/output`);
    return response.data;
  },
};

// =============================================================================
// Role Requests API
// =============================================================================

export interface RoleInfo {
  name: string;
  description: string;
  permissions: string[];
  has_role: boolean;
}

export interface RoleRequest {
  id: number;
  uuid: string;
  user_id: number;
  requested_role: string;
  message: string;
  status: 'pending' | 'approved' | 'rejected';
  admin_response: string | null;
  reviewed_by: number | null;
  reviewed_at: string | null;
  created_at: string;
  updated_at: string;
  user?: { id: number; name: string; email: string };
  reviewer?: { id: number; name: string };
}

export const roleRequestsApi = {
  getAvailableRoles: async (): Promise<{ roles: Record<string, RoleInfo> }> => {
    const response = await api.get('/role-requests/available-roles');
    return response.data;
  },

  myRequests: async (): Promise<{ requests: RoleRequest[] }> => {
    const response = await api.get('/role-requests/my');
    return response.data;
  },

  create: async (data: { role: string; message: string }): Promise<{ request: RoleRequest }> => {
    const response = await api.post('/role-requests', data);
    return response.data;
  },

  cancel: async (id: number): Promise<void> => {
    await api.delete(`/role-requests/${id}`);
  },

  // Admin endpoints
  adminList: async (params?: { status?: string; page?: number }) => {
    const response = await api.get('/admin/role-requests', { params });
    return response.data;
  },

  approve: async (id: number, response?: string) => {
    const res = await api.post(`/admin/role-requests/${id}/approve`, { response });
    return res.data;
  },

  reject: async (id: number, response?: string) => {
    const res = await api.post(`/admin/role-requests/${id}/reject`, { response });
    return res.data;
  },
};

// =============================================================================
// Audio Processing API
// =============================================================================

export interface VoiceModelConfig {
  model_id: number;
  f0_up_key?: number; // Pitch shift for this voice layer
  extraction_mode?: 'main' | 'all'; // 'main' = HP5 (main vocal only), 'all' = HP3 (all vocals including harmonies)
}

export interface AudioProcessRequest {
  audio: string; // Base64 encoded audio
  sample_rate?: number;
  mode: 'split' | 'convert' | 'swap';
  model_id?: number;
  f0_up_key?: number;
  index_rate?: number;
  pitch_shift_all?: number; // Pitch shift for both vocals and instrumental (split mode) or just instrumental (swap mode)
  instrumental_pitch?: number; // Separate instrumental pitch shift (optional, used when different from pitch_shift_all)
  extract_all_vocals?: boolean; // Extract all vocals including backups (true) or only main vocal (false)
  // Multi-voice swap configuration
  voice_count?: number; // Number of voice layers to extract (1-4), default 1
  voice_configs?: VoiceModelConfig[]; // Voice model configs for multi-voice swap
}

export interface AudioProcessResponse {
  mode: string;
  vocals?: string; // Base64 encoded
  instrumental?: string; // Base64 encoded
  converted?: string; // Base64 encoded
  sample_rate: number;
  format: string;
  job_id?: string; // Job tracking ID
}

export const audioProcessingApi = {
  /**
   * Process audio with various modes (calls Laravel backend for job tracking)
   */
  process: async (request: AudioProcessRequest): Promise<AudioProcessResponse> => {
    // Call Laravel API which handles job recording and forwards to voice engine
    const response = await api.post('/audio/process', {
      audio: request.audio,
      sample_rate: request.sample_rate || 44100,
      mode: request.mode,
      model_id: request.model_id,
      f0_up_key: request.f0_up_key || 0,
      index_rate: request.index_rate || 0.75,
      pitch_shift_all: request.pitch_shift_all || 0,
      instrumental_pitch: request.instrumental_pitch,
      extract_all_vocals: request.extract_all_vocals ?? false, // Default to main vocal only
      voice_count: request.voice_count || 1,
      voice_configs: request.voice_configs,
    });
    
    return response.data;
  },
};

// =============================================================================
// Voice Detection API
// =============================================================================

export interface VoiceDetectRequest {
  audio: string; // Base64 encoded audio
  sample_rate?: number;
  use_vocals_only?: boolean; // Separate vocals first (recommended for music)
  max_voices?: number; // Maximum number of voices to detect
}

export interface VoiceDetectResponse {
  voice_count: number; // Number of distinct voices detected
  confidence: number; // Confidence score (0-1)
  method: string; // Detection method used
  details: {
    segments_analyzed?: number;
    embeddings_computed?: number;
    max_voices_checked?: number;
    reason?: string;
  };
}

export const voiceDetectionApi = {
  /**
   * Detect number of distinct singers/speakers in audio
   * Useful for determining if backup vocals exist
   */
  detect: async (request: VoiceDetectRequest): Promise<VoiceDetectResponse> => {
    const response = await api.post('/voice-count/detect', {
      audio: request.audio,
      sample_rate: request.sample_rate || 44100,
      use_vocals_only: request.use_vocals_only ?? true,
      max_voices: request.max_voices || 5,
    });
    
    return response.data;
  },
};

// =============================================================================
// YouTube Song Search API
// =============================================================================

export interface YouTubeSearchResult {
  id: string;
  title: string;
  artist: string;
  duration: number; // seconds
  thumbnail: string;
  url: string;
  view_count: number;
  is_cached: boolean;
}

export interface YouTubeSearchResponse {
  results: YouTubeSearchResult[];
  query: string;
}

export interface YouTubeDownloadResponse {
  audio: string; // Base64 encoded WAV
  sample_rate: number;
  video_id: string;
  title: string;
  artist: string;
  duration: number;
}

export const youtubeApi = {
  /**
   * Search YouTube for songs
   */
  search: async (query: string, maxResults: number = 10): Promise<YouTubeSearchResponse> => {
    const response = await api.post('/youtube/search', { query, max_results: maxResults });
    return response.data;
  },

  /**
   * Download audio from YouTube video
   */
  download: async (videoId: string, useCache: boolean = true): Promise<YouTubeDownloadResponse> => {
    const response = await api.post('/youtube/download', { video_id: videoId, use_cache: useCache });
    return response.data;
  },

  /**
   * Get video info
   */
  info: async (videoId: string) => {
    const response = await api.get(`/youtube/info/${videoId}`);
    return response.data;
  },
};

// =============================================================================
// Trainer / Language Readiness API
// =============================================================================

export interface LanguageReadiness {
  readiness_score: number | null;
  phoneme_coverage: number | null;
  missing_phonemes: string[] | null;
}

export interface ModelLanguageScores {
  model_id: number;
  name: string;
  languages: {
    en: LanguageReadiness;
    is: LanguageReadiness;
  };
  full_results?: Record<string, unknown>;
  scanned_at: string | null;
}

export interface ScanModelResponse {
  success: boolean;
  model: {
    id: number;
    name: string;
    en_readiness_score: number | null;
    en_phoneme_coverage: number | null;
    en_missing_phonemes: string[] | null;
    is_readiness_score: number | null;
    is_phoneme_coverage: number | null;
    is_missing_phonemes: string[] | null;
    language_scanned_at: string | null;
  };
}

export interface ScanAllModelsResponse {
  success: boolean;
  results: {
    total: number;
    scanned: number;
    failed: number;
    skipped: number;
  };
}

export interface TrainerHealthResponse {
  available: boolean;
  languages: string[];
}

export interface GapAnalysisResponse {
  language: string;
  model_id: number;
  missing_phonemes: string[];
  coverage_percentage: number;
  suggested_prompts: Array<{
    text: string;
    phonemes: string[];
    category: string;
  }>;
}

export interface PhonemeInfo {
  language: string;
  phonemes: string[];
  count: number;
}

export interface WizardSession {
  session_id: string;
  language: string;
  exp_name: string;
  status: 'created' | 'in_progress' | 'paused' | 'completed' | 'cancelled';
  total_prompts: number;
  completed_prompts: number;
  current_prompt_index: number;
  progress_percentage: number;
}

export interface WizardPrompt {
  index: number;
  text: string;
  phonemes: string[];
  category?: string;
  ipa_text?: string;
  is_recorded?: boolean;
  is_skipped?: boolean;
  // Alternative field names from API
  prompt_text?: string;
  prompt_id?: string;
}

export interface InferenceTestResult {
  model_path: string;
  model_name: string;
  overall_score: number;
  language_scores: {
    [language: string]: {
      overall_score: number;
      pitch_stability: number;
      audio_clarity: number;
      artifact_score: number;
      tests_run: number;
      tests_passed: number;
    };
  };
  test_details: Array<{
    sentence: string;
    success: boolean;
    quality_score?: number;
    pitch_stability?: number;
    audio_clarity?: number;
    artifact_score?: number;
    duration_seconds?: number;
    error?: string;
  }>;
  recommendations: string[];
}

export const trainerApi = {
  /**
   * Check trainer API health
   */
  health: async (): Promise<TrainerHealthResponse> => {
    const response = await api.get('/trainer/health');
    return response.data;
  },

  /**
   * Get available languages
   */
  getLanguages: async (): Promise<string[]> => {
    const response = await api.get('/trainer/languages');
    return response.data.languages;
  },

  /**
   * Scan a model for language readiness
   */
  scanModel: async (modelId: number | string, languages: string[] = ['en', 'is']): Promise<ScanModelResponse> => {
    const response = await api.post(`/trainer/scan/${modelId}`, { languages });
    return response.data;
  },

  /**
   * Scan all models (admin only)
   */
  scanAllModels: async (languages: string[] = ['en', 'is']): Promise<ScanAllModelsResponse> => {
    const response = await api.post('/admin/voice-models/scan-all', { languages });
    return response.data;
  },

  /**
   * Get model readiness without rescanning
   */
  getReadiness: async (modelId: number | string): Promise<ModelLanguageScores> => {
    const response = await api.get(`/trainer/readiness/${modelId}`);
    return response.data;
  },

  /**
   * Analyze gaps for a model
   */
  analyzeGaps: async (modelId: number | string, language: string): Promise<GapAnalysisResponse> => {
    const response = await api.post(`/trainer/gaps/${modelId}`, { language });
    return response.data;
  },

  /**
   * Test a model using inference
   * 
   * Runs test sentences through the model to assess quality.
   * Useful for models without training data.
   */
  testModelInference: async (
    modelId: number | string, 
    languages: string[] = ['en'],
    testSentences?: string[],
    voice?: string
  ): Promise<{ success: boolean; model_id: number; model_name: string; results: InferenceTestResult }> => {
    const response = await api.post(`/trainer/test/${modelId}`, {
      languages,
      test_sentences: testSentences,
      voice: voice || 'en-US-GuyNeural',
    });
    return response.data;
  },

  /**
   * Get phonemes for a language
   */
  getPhonemes: async (language: string): Promise<PhonemeInfo> => {
    const response = await api.get(`/trainer/phonemes/${language}`);
    return response.data;
  },

  /**
   * Get prompts for a language
   */
  getPrompts: async (language: string) => {
    const response = await api.get(`/trainer/prompts/${language}`);
    return response.data;
  },

  /**
   * Get prompts for specific phonemes
   */
  getPromptsForPhonemes: async (language: string, phonemes: string[]) => {
    const response = await api.post(`/trainer/prompts/${language}/for-phonemes`, { phonemes });
    return response.data;
  },

  // Wizard session methods
  createWizardSession: async (language: string, expName: string, promptCount?: number, targetPhonemes?: string[]): Promise<WizardSession> => {
    const response = await api.post('/trainer/wizard/sessions', {
      language,
      exp_name: expName,
      prompt_count: promptCount,
      target_phonemes: targetPhonemes,
    });
    return response.data;
  },

  getWizardSession: async (sessionId: string): Promise<WizardSession> => {
    const response = await api.get(`/trainer/wizard/sessions/${sessionId}`);
    return response.data;
  },

  startWizardSession: async (sessionId: string): Promise<WizardSession> => {
    const response = await api.post(`/trainer/wizard/sessions/${sessionId}/start`);
    return response.data;
  },

  getWizardPrompt: async (sessionId: string): Promise<WizardPrompt> => {
    const response = await api.get(`/trainer/wizard/sessions/${sessionId}/prompt`);
    const data = response.data;
    
    // Normalize the response - API may return nested prompt object
    const prompt = data.prompt || data;
    
    return {
      index: prompt.index ?? data.current_index ?? 0,
      text: prompt.text || prompt.prompt_text || '',
      phonemes: prompt.phonemes || prompt.phonemes_covered || [],
      category: prompt.category,
      ipa_text: prompt.ipa_text,
      is_recorded: prompt.is_recorded,
      is_skipped: prompt.is_skipped,
    };
  },

  submitRecording: async (sessionId: string, audioBase64: string, sampleRate?: number, autoAdvance?: boolean, format?: string) => {
    const response = await api.post(`/trainer/wizard/sessions/${sessionId}/submit`, {
      audio: audioBase64,
      sample_rate: sampleRate,
      auto_advance: autoAdvance,
      format: format,
    });
    return response.data;
  },

  wizardNext: async (sessionId: string) => {
    const response = await api.post(`/trainer/wizard/sessions/${sessionId}/next`);
    return response.data;
  },

  wizardPrevious: async (sessionId: string) => {
    const response = await api.post(`/trainer/wizard/sessions/${sessionId}/previous`);
    return response.data;
  },

  wizardSkip: async (sessionId: string) => {
    const response = await api.post(`/trainer/wizard/sessions/${sessionId}/skip`);
    return response.data;
  },

  completeWizardSession: async (sessionId: string) => {
    const response = await api.post(`/trainer/wizard/sessions/${sessionId}/complete`);
    return response.data;
  },

  cancelWizardSession: async (sessionId: string) => {
    const response = await api.delete(`/trainer/wizard/sessions/${sessionId}`);
    return response.data;
  },

  /**
   * Upload audio file directly to training dataset
   */
  uploadAudio: async (expName: string, audioBase64: string, label: string, language?: string) => {
    // Convert base64 to blob for file upload
    const byteCharacters = atob(audioBase64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'audio/wav' });
    
    // Create form data
    const formData = new FormData();
    formData.append('exp_name', expName);
    formData.append('files[]', blob, `${label}.wav`);
    if (language) {
      formData.append('language', language);
    }
    
    const response = await api.post('/trainer/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Start a training job
   */
  startTraining: async (
    expName: string, 
    language: string,
    config?: {
      sample_rate?: number;
      f0_method?: string;
      epochs?: number;
      batch_size?: number;
    }
  ): Promise<{ job_id: string; status: string; message: string }> => {
    const response = await api.post('/trainer/start', {
      exp_name: expName,
      language,
      config,
    });
    return response.data;
  },

  /**
   * Get training job status
   */
  trainingStatus: async (jobId: string): Promise<{
    job_id: string;
    status: string;
    progress: number;
    message?: string;
    status_message?: string;
    current_epoch?: number;
    total_epochs?: number;
    error?: string;
  }> => {
    const response = await api.get(`/trainer/jobs/${jobId}`);
    return response.data;
  },

  /**
   * List all training jobs
   */
  listTrainingJobs: async (): Promise<Array<{
    job_id: string;
    exp_name: string;
    status: string;
    progress: number;
    created_at: string;
    current_epoch?: number;
    total_epochs?: number;
    message?: string;
  }>> => {
    const response = await api.get('/trainer/jobs');
    return response.data;
  },

  /**
   * Cancel a training job
   */
  cancelTraining: async (jobId: string): Promise<{ success: boolean; message: string }> => {
    const response = await api.post(`/trainer/jobs/${jobId}/cancel`);
    return response.data;
  },

  /**
   * Upload training audio files for a model
   * Uploads files in batches to avoid timeout issues with large uploads
   */
  uploadTrainingAudio: async (
    expName: string, 
    files: File[], 
    language: string = 'en',
    onProgress?: (progress: number) => void
  ): Promise<{ success: boolean; files_uploaded: number; files: string[] }> => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
    const BATCH_SIZE = 4;  // Upload 4 files at a time to avoid timeout issues
    
    const allUploadedFiles: string[] = [];
    let totalUploaded = 0;
    
    // Split files into batches
    const batches: File[][] = [];
    for (let i = 0; i < files.length; i += BATCH_SIZE) {
      batches.push(files.slice(i, i + BATCH_SIZE));
    }
    
    // Upload each batch sequentially
    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      const formData = new FormData();
      formData.append('exp_name', expName);
      formData.append('language', language);
      batch.forEach(file => {
        formData.append('files[]', file);
      });

      // Calculate progress including previous batches
      const previousBatchesProgress = (batchIndex / batches.length) * 100;
      const currentBatchWeight = 100 / batches.length;
      
      const response = await axios.post(`${API_URL}/trainer/upload`, formData, {
        headers: {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        withCredentials: true,
        timeout: 300000,  // 5 minutes per batch (smaller batches = shorter timeout needed)
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const batchProgress = (progressEvent.loaded / progressEvent.total) * currentBatchWeight;
            onProgress(Math.round(previousBatchesProgress + batchProgress));
          }
        },
      });
      
      if (response.data.files) {
        allUploadedFiles.push(...response.data.files);
      }
      totalUploaded += batch.length;
      
      // Report 100% for this batch before moving to next
      if (onProgress && batchIndex < batches.length - 1) {
        onProgress(Math.round(((batchIndex + 1) / batches.length) * 100));
      }
    }
    
    // Final progress
    if (onProgress) {
      onProgress(100);
    }
    
    return { 
      success: true, 
      files_uploaded: totalUploaded, 
      files: allUploadedFiles 
    };
  },

  // ============================================================================
  // Model Training Data (Cumulative recordings across sessions)
  // ============================================================================

  /**
   * Get all recordings for a model
   */
  getModelRecordings: async (modelSlug: string): Promise<{
    exp_name: string;
    total_recordings: number;
    total_duration_seconds: number;
    audio_paths: string[];
    categories: Record<string, { count: number; audio_paths: string[] }>;
    sessions: any[];
  }> => {
    const response = await api.get(`/trainer/model/${modelSlug}/recordings`);
    return response.data;
  },

  /**
   * Get category recording status for a model
   */
  getCategoryStatus: async (modelSlug: string, language: string): Promise<{
    exp_name: string;
    language: string;
    categories: Record<string, {
      name: string;
      total_prompts: number;
      recordings: number;
      has_recordings: boolean;
      phonemes_covered: string[];
    }>;
    model: {
      id: number;
      name: string;
      en_phoneme_coverage: number | null;
      en_missing_phonemes: string[];
      is_phoneme_coverage: number | null;
      is_missing_phonemes: string[];
      language_scanned_at: string | null;
    };
  }> => {
    const response = await api.get(`/trainer/model/${modelSlug}/category-status`, {
      params: { language }
    });
    return response.data;
  },

  /**
   * Start training using all collected recordings for a model
   */
  trainModelWithRecordings: async (modelSlug: string, config?: {
    epochs?: number;
    batch_size?: number;
    sample_rate?: number;
    f0_method?: string;
    force_reprocess?: boolean;
    continue_from_checkpoint?: boolean;
  }): Promise<{
    job_id: string;
    status: string;
    exp_name: string;
    audio_files: number;
    total_duration: number;
    config: { epochs: number; batch_size: number; sample_rate: number };
  }> => {
    const response = await api.post(`/trainer/model/${modelSlug}/train`, { config });
    return response.data;
  },

  /**
   * Get comprehensive model training info including checkpoint status
   * 
   * Returns information about recordings, preprocessed data,
   * training status, and checkpoint info for resuming training.
   */
  getModelTrainingInfo: async (modelSlug: string): Promise<{
    name: string;
    model_dir: string;
    recordings: {
      count: number;
      duration_seconds: number;
      duration_minutes: number;
    };
    preprocessed: {
      count: number;
      has_data: boolean;
    };
    training: {
      has_model: boolean;
      has_index: boolean;
      epochs_trained: number;
      target_epochs: number;
      last_trained: string | null;
      latest_checkpoint: string | null;
    };
    coverage: {
      phoneme_percent: number | null;
      phonemes_covered: number;
      phonemes_missing: number;
    };
    categories: Record<string, {
      name: string;
      recordings: number;
      satisfied: boolean;
    }>;
    model: {
      id: number;
      name: string;
      slug: string;
      status: string;
      en_readiness_score: number | null;
      language_scanned_at: string | null;
    };
  }> => {
    const response = await api.get(`/trainer/model/${modelSlug}/training-info`);
    return response.data;
  },

  // ============================================================================
  // Checkpoint Management
  // ============================================================================

  /**
   * Request a checkpoint save for a training job
   * 
   * @param jobId - Training job ID
   * @param stopAfter - If true, stop training after saving ("Save checkpoint & cancel")
   */
  requestCheckpoint: async (jobId: string, stopAfter: boolean = false): Promise<{
    job_id: string;
    checkpoint_requested: boolean;
    stop_after: boolean;
    status: string;
    message: string;
  }> => {
    const response = await api.post(`/trainer/checkpoint/${jobId}`, null, {
      params: { stop_after: stopAfter }
    });
    return response.data;
  },

  /**
   * Get checkpoint request status
   */
  getCheckpointStatus: async (jobId: string): Promise<{
    job_id: string;
    checkpoint_completed: boolean;
    success?: boolean;
    checkpoint_path?: string;
    error?: string;
    completed_at?: string;
    message?: string;
  }> => {
    const response = await api.get(`/trainer/checkpoint/${jobId}/status`);
    return response.data;
  },
};

// =============================================================================
// Training Runs API (Git-like training workflow)
// =============================================================================

export const trainingRunsApi = {
  /**
   * List all training runs for a voice model
   */
  listRuns: async (modelSlug: string): Promise<{
    runs: TrainingRun[];
    active_run: TrainingRun | null;
  }> => {
    const response = await api.get(`/training-runs/model/${modelSlug}`);
    return response.data;
  },

  /**
   * Get the training history tree for a model
   */
  getTree: async (modelSlug: string): Promise<{
    tree: TrainingTreeNode[];
    total_runs: number;
    total_epochs: number;
    total_time: string;
  }> => {
    const response = await api.get(`/training-runs/model/${modelSlug}/tree`);
    return response.data;
  },

  /**
   * Get runs that can be resumed
   */
  getResumable: async (modelSlug: string): Promise<{
    resumable_runs: Array<{
      run: TrainingRun;
      latest_checkpoint: TrainingCheckpoint | null;
    }>;
  }> => {
    const response = await api.get(`/training-runs/model/${modelSlug}/resumable`);
    return response.data;
  },

  /**
   * Get dataset versions for a model
   */
  getDatasetVersions: async (modelSlug: string): Promise<{
    versions: DatasetVersion[];
    current_version: DatasetVersion | null;
  }> => {
    const response = await api.get(`/training-runs/model/${modelSlug}/dataset-versions`);
    return response.data;
  },

  /**
   * Get details of a specific training run
   */
  getRun: async (runId: number): Promise<{
    run: TrainingRun;
    checkpoints: TrainingCheckpoint[];
    ancestry: TrainingRun[];
  }> => {
    const response = await api.get(`/training-runs/${runId}`);
    return response.data;
  },

  /**
   * Get current status of a training run
   */
  getRunStatus: async (runId: number): Promise<{
    run: TrainingRun;
    engine_status?: Record<string, any>;
  }> => {
    const response = await api.get(`/training-runs/${runId}/status`);
    return response.data;
  },

  /**
   * Start a new training run
   */
  startNewRun: async (modelSlug: string, config?: TrainingConfig): Promise<{
    success: boolean;
    run: TrainingRun;
  }> => {
    const response = await api.post(`/training-runs/model/${modelSlug}/start`, { config });
    return response.data;
  },

  /**
   * Resume training from the latest checkpoint
   */
  resumeRun: async (runId: number, additionalEpochs?: number): Promise<{
    success: boolean;
    run: TrainingRun;
    resumed_from: TrainingRun;
  }> => {
    const response = await api.post(`/training-runs/${runId}/resume`, {
      additional_epochs: additionalEpochs,
    });
    return response.data;
  },

  /**
   * Continue training from a specific checkpoint
   */
  continueFromCheckpoint: async (checkpointId: number, config: TrainingConfig): Promise<{
    success: boolean;
    run: TrainingRun;
    continued_from_checkpoint: TrainingCheckpoint;
  }> => {
    const response = await api.post(`/training-runs/checkpoints/${checkpointId}/continue`, { config });
    return response.data;
  },

  /**
   * Branch from a checkpoint with a new dataset
   */
  branchFromCheckpoint: async (
    checkpointId: number, 
    datasetVersionId: number, 
    config: TrainingConfig
  ): Promise<{
    success: boolean;
    run: TrainingRun;
    branched_from_checkpoint: TrainingCheckpoint;
    new_dataset: DatasetVersion;
  }> => {
    const response = await api.post(`/training-runs/checkpoints/${checkpointId}/branch`, {
      dataset_version_id: datasetVersionId,
      config,
    });
    return response.data;
  },

  /**
   * Cancel an active training run
   */
  cancelRun: async (runId: number): Promise<{
    success: boolean;
    run: TrainingRun;
  }> => {
    const response = await api.post(`/training-runs/${runId}/cancel`);
    return response.data;
  },

  /**
   * Get checkpoints for a training run
   */
  getCheckpoints: async (runId: number, options?: {
    activeOnly?: boolean;
    milestonesOnly?: boolean;
  }): Promise<{
    checkpoints: TrainingCheckpoint[];
    best_checkpoint: TrainingCheckpoint | null;
  }> => {
    const response = await api.get(`/training-runs/${runId}/checkpoints`, {
      params: {
        active_only: options?.activeOnly,
        milestones_only: options?.milestonesOnly,
      },
    });
    return response.data;
  },

  /**
   * Get a specific checkpoint
   */
  getCheckpoint: async (checkpointId: number): Promise<{
    checkpoint: TrainingCheckpoint;
    run: TrainingRun;
  }> => {
    const response = await api.get(`/training-runs/checkpoints/${checkpointId}`);
    return response.data;
  },

  /**
   * Archive a checkpoint
   */
  archiveCheckpoint: async (checkpointId: number): Promise<{
    success: boolean;
    checkpoint: TrainingCheckpoint;
  }> => {
    const response = await api.post(`/training-runs/checkpoints/${checkpointId}/archive`);
    return response.data;
  },

  /**
   * Add a note to a checkpoint
   */
  addCheckpointNote: async (checkpointId: number, notes: string): Promise<{
    success: boolean;
    checkpoint: TrainingCheckpoint;
  }> => {
    const response = await api.post(`/training-runs/checkpoints/${checkpointId}/note`, { notes });
    return response.data;
  },

  /**
   * Get a specific dataset version
   */
  getDatasetVersion: async (versionId: number): Promise<{
    version: DatasetVersion;
    runs_using_this: number;
  }> => {
    const response = await api.get(`/training-runs/dataset-versions/${versionId}`);
    return response.data;
  },
};

