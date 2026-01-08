'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';
import {
  Mic2,
  LayoutDashboard,
  Box,
  ListMusic,
  Settings,
  LogOut,
  Volume2,
  Upload,
  FileAudio,
  Menu,
  X,
  Loader2,
} from 'lucide-react';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import { useAuthContext } from './providers';
import { Footer } from './footer';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, user, clearAuth, isHydrated, canUploadModels } = useAuthStore();
  const { isLoading, isReady } = useAuthContext();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Redirect if not authenticated (only after hydration is complete)
  useEffect(() => {
    if (isHydrated && isReady && !isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, isHydrated, isReady, router]);

  const handleLogout = async () => {
    try {
      await authApi.logout();
    } catch (e) {
      // Ignore error
    }
    clearAuth();
    router.push('/');
  };

  // Show loading state during hydration
  if (!isHydrated || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  const navItems = [
    { href: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { href: '/dashboard/models', icon: Box, label: 'My Models' },
    { href: '/dashboard/tts', icon: Volume2, label: 'Text to Speech' },
    { href: '/dashboard/audio', icon: FileAudio, label: 'Audio Processing' },
    { href: '/dashboard/jobs', icon: ListMusic, label: 'My Jobs' },
    { href: '/dashboard/settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Mobile Header */}
      <header className="lg:hidden bg-gray-900/50 border-b border-gray-800 p-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <Mic2 className="h-8 w-8 text-primary-500" />
          <span className="text-xl font-bold gradient-text">MorphVox</span>
        </Link>
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-2 text-gray-400 hover:text-white"
        >
          {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </header>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div 
          className="lg:hidden fixed inset-0 z-40 bg-black/50"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed lg:static inset-y-0 left-0 z-50 w-64 
        bg-gray-900/95 lg:bg-gray-900/50 
        border-r border-gray-800 flex flex-col
        transform transition-transform duration-300 ease-in-out
        ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="p-4 border-b border-gray-800 hidden lg:block">
          <Link href="/" className="flex items-center gap-2">
            <Mic2 className="h-8 w-8 text-primary-500" />
            <span className="text-xl font-bold gradient-text">MorphVox</span>
          </Link>
        </div>

        <div className="p-4 lg:hidden border-b border-gray-800 flex justify-end">
          <button
            onClick={() => setMobileMenuOpen(false)}
            className="p-2 text-gray-400 hover:text-white"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {navItems.map(({ href, icon, label }) => (
            <NavItem 
              key={href}
              href={href} 
              icon={icon} 
              active={pathname === href}
              onClick={() => setMobileMenuOpen(false)}
            >
              {label}
            </NavItem>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-full bg-primary-600 flex items-center justify-center">
              <span className="font-semibold">{user?.name?.charAt(0).toUpperCase()}</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{user?.name}</p>
              <p className="text-sm text-gray-500 truncate">{user?.email}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors w-full"
          >
            <LogOut className="h-4 w-4" />
            Sign out
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-4 lg:p-8 overflow-x-hidden flex flex-col">
        <div className="max-w-6xl mx-auto flex-1">
          {children}
        </div>
        <div className="hidden lg:block mt-8">
          <Footer minimal />
        </div>
      </main>

      {/* Mobile Bottom Nav */}
      <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-gray-900/95 border-t border-gray-800 px-2 py-2 z-30">
        <div className="flex justify-around items-center">
          {navItems.slice(0, 5).map(({ href, icon: Icon, label }) => (
            <Link
              key={href}
              href={href}
              className={`flex flex-col items-center p-2 rounded-lg transition-colors ${
                pathname === href
                  ? 'text-primary-400'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Icon className="h-5 w-5" />
              <span className="text-xs mt-1 hidden sm:block">{label.split(' ')[0]}</span>
            </Link>
          ))}
        </div>
      </nav>

      {/* Bottom padding for mobile nav */}
      <div className="lg:hidden h-16" />
    </div>
  );
}

function NavItem({
  href,
  icon: Icon,
  children,
  active = false,
  onClick,
}: {
  href: string;
  icon: any;
  children: React.ReactNode;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <Link
      href={href}
      onClick={onClick}
      className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
        active
          ? 'bg-primary-600/20 text-primary-400'
          : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
      }`}
    >
      <Icon className="h-5 w-5" />
      {children}
    </Link>
  );
}

export default DashboardLayout;
