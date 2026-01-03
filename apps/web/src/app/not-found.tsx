'use client';

import Link from 'next/link';
import { Home, ArrowLeft, Search, Mic2 } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4">
      {/* Logo */}
      <div className="flex items-center gap-2 mb-8">
        <Mic2 className="h-10 w-10 text-primary-500" />
        <span className="text-2xl font-bold gradient-text">VoxMorph</span>
      </div>

      {/* Error Display */}
      <div className="text-center">
        <h1 className="text-9xl font-bold text-gray-800">404</h1>
        <h2 className="text-2xl font-semibold mt-4 mb-2">Page Not Found</h2>
        <p className="text-gray-400 max-w-md mx-auto mb-8">
          The page you're looking for doesn't exist or has been moved.
          Let's get you back on track.
        </p>
      </div>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-4">
        <Link
          href="/"
          className="flex items-center justify-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-6 py-3 rounded-lg transition-colors"
        >
          <Home className="h-5 w-5" />
          Go Home
        </Link>
        <Link
          href="/models"
          className="flex items-center justify-center gap-2 bg-gray-800 hover:bg-gray-700 text-white px-6 py-3 rounded-lg transition-colors"
        >
          <Search className="h-5 w-5" />
          Browse Models
        </Link>
      </div>

      {/* Back Link */}
      <button
        onClick={() => window.history.back()}
        className="flex items-center gap-2 text-gray-500 hover:text-gray-300 mt-8 transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Go back to previous page
      </button>
    </div>
  );
}
