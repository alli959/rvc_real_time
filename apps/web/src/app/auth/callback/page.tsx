'use client';

import { useEffect, useState, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

function AuthCallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const setAuth = useAuthStore((state) => state.setAuth);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [error, setError] = useState('');

  useEffect(() => {
    const token = searchParams.get('token');
    const provider = searchParams.get('provider');
    const errorMsg = searchParams.get('error');

    if (errorMsg) {
      setStatus('error');
      setError(decodeURIComponent(errorMsg));
      setTimeout(() => router.push('/login'), 3000);
      return;
    }

    if (token) {
      // Store the token and fetch user data
      handleToken(token);
    } else {
      setStatus('error');
      setError('No authentication token received');
      setTimeout(() => router.push('/login'), 3000);
    }
  }, [searchParams, router]);

  const handleToken = async (token: string) => {
    try {
      // Store token temporarily to make the /me request
      localStorage.setItem('auth_token', token);
      
      // Fetch user data
      const { user } = await authApi.me();
      
      // Set auth state
      setAuth(user, token);
      
      setStatus('success');
      
      // Redirect to dashboard
      setTimeout(() => router.push('/dashboard'), 1000);
    } catch (err: any) {
      setStatus('error');
      setError(err.response?.data?.message || 'Failed to authenticate');
      localStorage.removeItem('auth_token');
      setTimeout(() => router.push('/login'), 3000);
    }
  };

  return (
    <div className="text-center">
      {status === 'loading' && (
        <>
          <Loader2 className="h-12 w-12 animate-spin text-primary-500 mx-auto mb-4" />
          <h1 className="text-xl font-semibold text-white mb-2">Signing you in...</h1>
          <p className="text-gray-400">Please wait while we complete your authentication.</p>
        </>
      )}

      {status === 'success' && (
        <>
          <CheckCircle2 className="h-12 w-12 text-green-500 mx-auto mb-4" />
          <h1 className="text-xl font-semibold text-white mb-2">Welcome!</h1>
          <p className="text-gray-400">Redirecting to dashboard...</p>
        </>
      )}

      {status === 'error' && (
        <>
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h1 className="text-xl font-semibold text-white mb-2">Authentication Failed</h1>
          <p className="text-red-400 mb-2">{error}</p>
          <p className="text-gray-400">Redirecting to login...</p>
        </>
      )}
    </div>
  );
}

export default function AuthCallbackPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950">
      <Suspense fallback={
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-primary-500 mx-auto mb-4" />
          <h1 className="text-xl font-semibold text-white mb-2">Loading...</h1>
        </div>
      }>
        <AuthCallbackContent />
      </Suspense>
    </div>
  );
}
