'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState, useEffect, createContext, useContext } from 'react';
import { useAuthStore } from '@/lib/store';
import { authApi } from '@/lib/api';

// Auth context for managing authentication state
interface AuthContextType {
  isLoading: boolean;
  isReady: boolean;
}

const AuthContext = createContext<AuthContextType>({ isLoading: true, isReady: false });

export function useAuthContext() {
  return useContext(AuthContext);
}

function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(true);
  const [isReady, setIsReady] = useState(false);
  const { isAuthenticated, isHydrated, token, setAuth, clearAuth } = useAuthStore();

  useEffect(() => {
    // Wait for hydration before checking auth
    if (!isHydrated) return;

    const validateAuth = async () => {
      // If we have a token, validate it with the server
      if (token && isAuthenticated) {
        try {
          const response = await authApi.me();
          // Update user data from server
          setAuth(response.user, token);
        } catch (error) {
          // Token is invalid, clear auth
          console.log('Auth validation failed, clearing auth');
          clearAuth();
        }
      }
      setIsLoading(false);
      setIsReady(true);
    };

    validateAuth();
  }, [isHydrated, token, isAuthenticated, setAuth, clearAuth]);

  return (
    <AuthContext.Provider value={{ isLoading, isReady }}>
      {children}
    </AuthContext.Provider>
  );
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        {children}
      </AuthProvider>
    </QueryClientProvider>
  );
}
