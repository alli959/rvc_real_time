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
  status: "pending" | "processing" | "ready" | "failed";
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
    const response = await api.get(`/auth/${provider}/redirect`);
    return response.data;
  },

  handleOAuthCallback: async (provider: 'google' | 'github', code: string) => {
    const response = await api.post(`/auth/${provider}/callback`, { code });
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

export interface AudioProcessRequest {
  audio: string; // Base64 encoded audio
  sample_rate?: number;
  mode: 'split' | 'convert' | 'swap';
  model_id?: number;
  f0_up_key?: number;
  index_rate?: number;
  pitch_shift_all?: number; // Pitch shift for both vocals and instrumental (split mode) or just instrumental (swap mode)
  instrumental_pitch?: number; // Separate instrumental pitch shift (optional, used when different from pitch_shift_all)
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
