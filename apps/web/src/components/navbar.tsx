'use client';

import Link from 'next/link';
import { Mic2 } from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { useAuthContext } from '@/components/providers';

export function Navbar() {
  const { isAuthenticated, user } = useAuthStore();
  const { isReady } = useAuthContext();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-2">
            <Mic2 className="h-8 w-8 text-primary-500" />
            <span className="text-xl font-bold gradient-text">MorphVox</span>
          </Link>
          <div className="flex items-center gap-4">
            <Link 
              href="/models" 
              className="text-gray-300 hover:text-white transition-colors"
            >
              Models
            </Link>
            
            {/* Show loading state or auth-dependent links */}
            {!isReady ? (
              // Placeholder while hydrating to prevent flash
              <div className="w-[180px] h-[38px]" />
            ) : isAuthenticated ? (
              // Logged in - show Dashboard link
              <Link 
                href="/dashboard" 
                className="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg transition-colors"
              >
                Dashboard
              </Link>
            ) : (
              // Not logged in - show Sign In and Get Started
              <>
                <Link 
                  href="/login" 
                  className="text-gray-300 hover:text-white transition-colors"
                >
                  Sign In
                </Link>
                <Link 
                  href="/register" 
                  className="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  Get Started
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
