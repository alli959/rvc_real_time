'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import {
  Mic2,
  LayoutDashboard,
  Box,
  ListMusic,
  Settings,
  LogOut,
  Plus,
  ChevronRight,
} from 'lucide-react';
import { authApi, modelsApi, jobsApi } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

export default function DashboardPage() {
  const router = useRouter();
  const { isAuthenticated, user, clearAuth } = useAuthStore();

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, router]);

  const { data: modelsData } = useQuery({
    queryKey: ['my-models'],
    queryFn: () => modelsApi.myModels(),
    enabled: isAuthenticated,
  });

  const { data: jobsData } = useQuery({
    queryKey: ['my-jobs'],
    queryFn: () => jobsApi.list({ page: 1 }),
    enabled: isAuthenticated,
  });

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

  const myModels = modelsData?.data || [];
  const recentJobs = jobsData?.data?.slice(0, 5) || [];

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-900/50 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <Link href="/" className="flex items-center gap-2">
            <Mic2 className="h-8 w-8 text-primary-500" />
            <span className="text-xl font-bold gradient-text">MorphVox</span>
          </Link>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          <NavItem href="/dashboard" icon={LayoutDashboard} active>
            Dashboard
          </NavItem>
          <NavItem href="/dashboard/models" icon={Box}>
            My Models
          </NavItem>
          <NavItem href="/dashboard/jobs" icon={ListMusic}>
            My Jobs
          </NavItem>
          <NavItem href="/dashboard/settings" icon={Settings}>
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
          <h1 className="text-2xl font-bold mb-8">Welcome back, {user?.name?.split(' ')[0]}!</h1>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <Link
              href="/dashboard/convert"
              className="glass rounded-xl p-6 hover:bg-white/10 transition-colors group"
            >
              <div className="w-12 h-12 rounded-lg bg-primary-600/20 flex items-center justify-center mb-4">
                <Mic2 className="h-6 w-6 text-primary-500" />
              </div>
              <h3 className="font-semibold mb-1 group-hover:text-primary-400 transition-colors">
                Convert Voice
              </h3>
              <p className="text-sm text-gray-400">Upload audio and transform it</p>
            </Link>

            <Link
              href="/models"
              className="glass rounded-xl p-6 hover:bg-white/10 transition-colors group"
            >
              <div className="w-12 h-12 rounded-lg bg-accent-600/20 flex items-center justify-center mb-4">
                <Box className="h-6 w-6 text-accent-500" />
              </div>
              <h3 className="font-semibold mb-1 group-hover:text-accent-400 transition-colors">
                Browse Models
              </h3>
              <p className="text-sm text-gray-400">Discover community voices</p>
            </Link>

            <Link
              href="/dashboard/models/new"
              className="glass rounded-xl p-6 hover:bg-white/10 transition-colors group"
            >
              <div className="w-12 h-12 rounded-lg bg-green-600/20 flex items-center justify-center mb-4">
                <Plus className="h-6 w-6 text-green-500" />
              </div>
              <h3 className="font-semibold mb-1 group-hover:text-green-400 transition-colors">
                Upload Model
              </h3>
              <p className="text-sm text-gray-400">Share your own voice model</p>
            </Link>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard label="My Models" value={myModels.length} />
            <StatCard label="Total Jobs" value={jobsData?.meta?.total || 0} />
            <StatCard label="Completed" value={recentJobs.filter((j: any) => j.status === 'completed').length} />
            <StatCard label="Processing" value={recentJobs.filter((j: any) => j.status === 'processing').length} />
          </div>

          {/* Recent Jobs */}
          <div className="glass rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold">Recent Jobs</h2>
              <Link
                href="/dashboard/jobs"
                className="text-sm text-primary-400 hover:text-primary-300 flex items-center gap-1"
              >
                View all
                <ChevronRight className="h-4 w-4" />
              </Link>
            </div>

            {recentJobs.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No jobs yet. Start by converting some audio!</p>
            ) : (
              <div className="space-y-3">
                {recentJobs.map((job: any) => (
                  <div
                    key={job.id}
                    className="flex items-center justify-between py-3 border-b border-gray-800 last:border-0"
                  >
                    <div>
                      <p className="font-medium">{job.voice_model?.name || 'Unknown Model'}</p>
                      <p className="text-sm text-gray-500">
                        {new Date(job.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <StatusBadge status={job.status} />
                  </div>
                ))}
              </div>
            )}
          </div>
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

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="glass rounded-xl p-4">
      <p className="text-sm text-gray-400 mb-1">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    pending: 'bg-yellow-500/20 text-yellow-400',
    queued: 'bg-blue-500/20 text-blue-400',
    processing: 'bg-primary-500/20 text-primary-400',
    completed: 'bg-green-500/20 text-green-400',
    failed: 'bg-red-500/20 text-red-400',
    cancelled: 'bg-gray-500/20 text-gray-400',
  };

  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${styles[status] || styles.pending}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}
