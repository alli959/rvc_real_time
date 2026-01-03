'use client';

import { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';
import {
  Mic2,
  LayoutDashboard,
  Box,
  ListMusic,
  Settings,
  LogOut,
} from 'lucide-react';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, user, clearAuth } = useAuthStore();

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, router]);

  const handleLogout = async () => {
    try {
      await authApi.logout();
    } catch (e) {
      // Ignore error
    }
    clearAuth();
    router.push('/');
  };

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-900/50 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <Link href="/" className="flex items-center gap-2">
            <Mic2 className="h-8 w-8 text-primary-500" />
            <span className="text-xl font-bold gradient-text">VoxMorph</span>
          </Link>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          <NavItem href="/dashboard" icon={LayoutDashboard} active={pathname === '/dashboard'}>
            Dashboard
          </NavItem>
          <NavItem href="/dashboard/models" icon={Box} active={pathname === '/dashboard/models'}>
            My Models
          </NavItem>
          <NavItem href="/dashboard/jobs" icon={ListMusic} active={pathname === '/dashboard/jobs'}>
            My Jobs
          </NavItem>
          <NavItem href="/dashboard/settings" icon={Settings} active={pathname === '/dashboard/settings'}>
            Settings
          </NavItem>
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
      <main className="flex-1 p-8">
        <div className="max-w-6xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  );
}

function NavItem({
  href,
  icon: Icon,
  children,
  active = false,
}: {
  href: string;
  icon: any;
  children: React.ReactNode;
  active?: boolean;
}) {
  return (
    <Link
      href={href}
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
