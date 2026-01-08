import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface User {
  id: string;
  name: string;
  email: string;
  roles?: string[];
  permissions?: string[];
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isHydrated: boolean;
  setAuth: (user: User, token: string) => void;
  updateUser: (user: Partial<User>) => void;
  clearAuth: () => void;
  hasRole: (role: string) => boolean;
  hasPermission: (permission: string) => boolean;
  canUploadModels: () => boolean;
  canTrainModels: () => boolean;
  canPublishModels: () => boolean;
  isAdmin: () => boolean;
  setHydrated: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isHydrated: false,

      setAuth: (user, token) => {
        if (typeof window !== 'undefined') {
          localStorage.setItem('auth_token', token);
        }
        set({ user, token, isAuthenticated: true });
      },

      updateUser: (userData) => {
        const { user } = get();
        if (user) {
          set({ user: { ...user, ...userData } });
        }
      },

      clearAuth: () => {
        if (typeof window !== 'undefined') {
          localStorage.removeItem('auth_token');
        }
        set({ user: null, token: null, isAuthenticated: false });
      },

      hasRole: (role) => {
        const { user } = get();
        return user?.roles?.includes(role) ?? false;
      },

      hasPermission: (permission) => {
        const { user } = get();
        return user?.permissions?.includes(permission) ?? false;
      },

      canUploadModels: () => {
        const { user } = get();
        if (!user) return false;
        return !!(user.permissions?.includes('upload_models') || user.roles?.includes('admin') || user.roles?.includes('creator'));
      },

      canTrainModels: () => {
        const { user } = get();
        if (!user) return false;
        return !!(user.permissions?.includes('train_models') || user.roles?.includes('admin') || user.roles?.includes('trainer'));
      },

      canPublishModels: () => {
        const { user } = get();
        if (!user) return false;
        return !!(user.permissions?.includes('publish_models') || user.roles?.includes('admin') || user.roles?.includes('publisher'));
      },

      isAdmin: () => {
        const { user } = get();
        return user?.roles?.includes('admin') ?? false;
      },

      setHydrated: () => {
        set({ isHydrated: true });
      },
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ 
        user: state.user, 
        token: state.token, 
        isAuthenticated: state.isAuthenticated 
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHydrated();
      },
    }
  )
);
