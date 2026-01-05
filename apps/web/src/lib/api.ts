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
  
  // Ownership
  user_id: number | null; // null = system model
  user?: { id: number; name: string } | null;
  
  // Storage (paths hidden from API)
  has_index: boolean;
  size_bytes: number;
  storage_type: 'local' | 's3';
  
  // Computed from paths (appended by API)
  model_file: string | null;
  index_file: string | null;
  size: string; // Human readable size
  
  // Metadata
  engine: string;
  visibility: 'public' | 'private' | 'unlisted';
  status: 'pending' | 'ready' | 'failed';
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
};

// Legacy alias - modelsApi points to the same functionality
export const modelsApi = voiceModelsApi;

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
