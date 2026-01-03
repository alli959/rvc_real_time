'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Mic2, Mail, Lock, User, AlertCircle } from 'lucide-react';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

export default function RegisterPage() {
  const router = useRouter();
  const setAuth = useAuthStore((state) => state.setAuth);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== passwordConfirm) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      const response = await authApi.register({
        name,
        email,
        password,
        password_confirmation: passwordConfirm,
      });
      setAuth(response.user, response.token);
      router.push('/dashboard');
    } catch (err: any) {
      const errors = err.response?.data?.errors;
      if (errors) {
        const firstError = Object.values(errors)[0];
        setError(Array.isArray(firstError) ? firstError[0] : String(firstError));
      } else {
        setError(err.response?.data?.message || 'Registration failed');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center gap-2">
            <Mic2 className="h-10 w-10 text-primary-500" />
            <span className="text-2xl font-bold gradient-text">VoxMorph</span>
          </Link>
          <h1 className="mt-6 text-3xl font-bold">Create an account</h1>
          <p className="mt-2 text-gray-400">Start transforming your voice today</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="glass rounded-xl p-8 space-y-6">
          {error && (
            <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">
              Name
            </label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full bg-gray-800/50 border border-gray-700 rounded-lg pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="Your name"
                required
              />
            </div>
          </div>

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
              Email
            </label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-gray-800/50 border border-gray-700 rounded-lg pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="you@example.com"
                required
              />
            </div>
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-gray-800/50 border border-gray-700 rounded-lg pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="••••••••"
                minLength={8}
                required
              />
            </div>
          </div>

          <div>
            <label htmlFor="password-confirm" className="block text-sm font-medium text-gray-300 mb-2">
              Confirm Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
              <input
                id="password-confirm"
                type="password"
                value={passwordConfirm}
                onChange={(e) => setPasswordConfirm(e.target.value)}
                className="w-full bg-gray-800/50 border border-gray-700 rounded-lg pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="••••••••"
                minLength={8}
                required
              />
            </div>
          </div>

          <div className="flex items-start gap-2">
            <input type="checkbox" required className="mt-1 rounded border-gray-700 bg-gray-800/50" />
            <span className="text-sm text-gray-400">
              I agree to the{' '}
              <Link href="/terms" className="text-primary-400 hover:text-primary-300">
                Terms of Service
              </Link>{' '}
              and{' '}
              <Link href="/privacy" className="text-primary-400 hover:text-primary-300">
                Privacy Policy
              </Link>
            </span>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition-colors"
          >
            {loading ? 'Creating account...' : 'Create account'}
          </button>
        </form>

        <p className="mt-6 text-center text-gray-400">
          Already have an account?{' '}
          <Link href="/login" className="text-primary-400 hover:text-primary-300">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
