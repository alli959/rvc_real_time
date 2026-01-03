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

  me: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },
};

// =============================================================================
// Models API
// =============================================================================

export interface VoiceModel {
  id: string;
  name: string;
  slug: string;
  description?: string;
  visibility: 'public' | 'private' | 'unlisted';
  status: 'pending' | 'ready' | 'error';
  engine: string;
  tags?: string[];
  usage_count: number;
  user: {
    id: string;
    name: string;
  };
  created_at: string;
}

export interface SystemVoiceModel {
  id: number;
  slug: string;
  name: string;
  description: string | null;
  model_file: string;
  model_path: string;
  index_file: string | null;
  index_path: string | null;
  has_index: boolean;
  size: string;
  size_bytes: number;
  storage_type: 'local' | 's3';
  storage_path: string | null;
  index_storage_path: string | null;
  engine: string;
  metadata: Record<string, any> | null;
  is_active: boolean;
  is_featured: boolean;
  usage_count: number;
  download_url: string | null;
  index_download_url: string | null;
  last_synced_at: string;
  created_at: string;
  updated_at: string;
}

export const modelsApi = {
  list: async (params?: { page?: number; search?: string }) => {
    const response = await api.get('/models', { params });
    return response.data;
  },

  get: async (id: string) => {
    const response = await api.get(`/models/${id}`);
    return response.data;
  },

  myModels: async () => {
    const response = await api.get('/models/my');
    return response.data;
  },

  create: async (data: { name: string; description?: string; visibility?: string }) => {
    const response = await api.post('/models', data);
    return response.data;
  },

  update: async (id: string, data: Partial<VoiceModel>) => {
    const response = await api.put(`/models/${id}`, data);
    return response.data;
  },

  delete: async (id: string) => {
    const response = await api.delete(`/models/${id}`);
    return response.data;
  },

  getUploadUrls: async (id: string) => {
    const response = await api.post(`/models/${id}/upload-urls`);
    return response.data;
  },

  confirmUpload: async (id: string) => {
    const response = await api.post(`/models/${id}/confirm-upload`);
    return response.data;
  },

  getDownloadUrls: async (id: string) => {
    const response = await api.get(`/models/${id}/download-urls`);
    return response.data;
  },
};

// =============================================================================
// System Voice Models API (Server-side models from local dir or S3)
// =============================================================================

export const voiceModelsApi = {
  list: async (params?: { 
    search?: string; 
    engine?: string;
    storage_type?: 'local' | 's3';
    has_index?: boolean;
    featured?: boolean;
    sort?: string;
    direction?: 'asc' | 'desc';
    per_page?: number;
    all?: boolean;
  }): Promise<{ data: SystemVoiceModel[]; total: number }> => {
    const response = await api.get('/voice-models', { params });
    return response.data;
  },

  get: async (slug: string): Promise<{ model: SystemVoiceModel }> => {
    const response = await api.get(`/voice-models/${slug}`);
    return response.data;
  },

  stats: async () => {
    const response = await api.get('/voice-models/stats');
    return response.data;
  },

  config: async () => {
    const response = await api.get('/voice-models/config');
    return response.data;
  },

  sync: async (params?: { prune?: boolean; storage?: 'local' | 's3' }) => {
    const response = await api.post('/voice-models/sync', params);
    return response.data;
  },

  update: async (slug: string, data: { name?: string; description?: string; is_active?: boolean; is_featured?: boolean }) => {
    const response = await api.patch(`/voice-models/${slug}`, data);
    return response.data;
  },
};

// =============================================================================
// Jobs API
// =============================================================================

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

export const jobsApi = {
  list: async (params?: { page?: number; status?: string }) => {
    const response = await api.get('/jobs', { params });
    return response.data;
  },

  get: async (id: string) => {
    const response = await api.get(`/jobs/${id}`);
    return response.data;
  },

  createInference: async (data: { voice_model_id: string; parameters?: object }) => {
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
