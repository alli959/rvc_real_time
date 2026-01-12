'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Mic2, LogOut, ChevronDown, User } from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { useAuthContext } from '@/components/providers';

export function Navbar() {
  const router = useRouter();
  const { isAuthenticated, user, clearAuth } = useAuthStore();
  const { isReady } = useAuthContext();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = () => {
    clearAuth();
    setShowUserMenu(false);
    router.push('/');
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-2 flex-shrink-0">
            <Mic2 className="h-6 w-6 sm:h-8 sm:w-8 text-primary-500" />
            <span className="text-lg sm:text-xl font-bold gradient-text">MorphVox</span>
          </Link>
          <div className="flex items-center gap-2 sm:gap-4">
            <Link 
              href={isAuthenticated ? "/dashboard/models" : "/models"} 
              className="hidden sm:block text-gray-300 hover:text-white transition-colors"
            >
              Models
            </Link>
            
            {/* Show loading state or auth-dependent links */}
            {!isReady ? (
              // Placeholder while hydrating to prevent flash
              <div className="w-20 sm:w-[180px] h-[38px]" />
            ) : isAuthenticated ? (
              // Logged in - show Dashboard link and user menu
              <div className="flex items-center gap-2 sm:gap-3">
                <Link 
                  href="/dashboard" 
                  className="bg-primary-600 hover:bg-primary-700 text-white px-3 sm:px-4 py-2 rounded-lg transition-colors text-sm sm:text-base"
                >
                  Dashboard
                </Link>
                
                {/* User dropdown menu */}
                <div className="relative" ref={menuRef}>
                  <button
                    onClick={() => setShowUserMenu(!showUserMenu)}
                    className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-2 rounded-lg text-gray-300 hover:text-white hover:bg-gray-800 transition-colors"
                  >
                    <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-primary-600 flex items-center justify-center text-white text-xs sm:text-sm font-medium">
                      {user?.name?.charAt(0)?.toUpperCase() || <User className="h-4 w-4" />}
                    </div>
                    <ChevronDown className={`h-4 w-4 transition-transform ${showUserMenu ? 'rotate-180' : ''}`} />
                  </button>
                  
                  {showUserMenu && (
                    <div className="absolute right-0 mt-2 w-48 bg-gray-900 border border-gray-700 rounded-lg shadow-xl py-1 z-50">
                      <div className="px-4 py-2 border-b border-gray-700">
                        <p className="text-sm font-medium text-white truncate">{user?.name}</p>
                        <p className="text-xs text-gray-400 truncate">{user?.email}</p>
                      </div>
                      <button
                        onClick={handleLogout}
                        className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors"
                      >
                        <LogOut className="h-4 w-4" />
                        Sign Out
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              // Not logged in - show Sign In and Get Started
              <>
                <Link 
                  href="/login" 
                  className="text-sm sm:text-base text-gray-300 hover:text-white transition-colors"
                >
                  Sign In
                </Link>
                <Link 
                  href="/register" 
                  className="bg-primary-600 hover:bg-primary-700 text-white px-3 sm:px-4 py-2 rounded-lg transition-colors text-sm sm:text-base"
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
