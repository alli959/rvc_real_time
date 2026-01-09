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
  Volume2,
  AudioWaveform,
  Music,
  Sparkles,
} from 'lucide-react';
import { authApi, modelsApi, jobsApi } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import { DashboardLayout } from '@/components/dashboard-layout';

export default function DashboardPage() {
  const router = useRouter();
  const { isAuthenticated, isHydrated, user, clearAuth } = useAuthStore();

  // Redirect if not authenticated (only after hydration is complete)
  useEffect(() => {
    if (isHydrated && !isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, isHydrated, router]);

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

  // Show loading while hydrating or checking auth
  if (!isHydrated || !isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
      </div>
    );
  }

  const myModels = modelsData?.data || [];
  const recentJobs = jobsData?.data?.slice(0, 5) || [];

  return (
    <DashboardLayout>
    <div className="min-h-screen flex">
      {/* Main Content */}
      <main className="flex-1 p-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-2xl font-bold mb-8">Welcome back, {user?.name?.split(' ')[0]}!</h1>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <Link
              href="/dashboard/tts"
              className="glass rounded-xl p-6 hover:bg-white/10 transition-colors group"
            >
              <div className="w-12 h-12 rounded-lg bg-purple-600/20 flex items-center justify-center mb-4">
                <Volume2 className="h-6 w-6 text-purple-500" />
              </div>
              <h3 className="font-semibold mb-1 group-hover:text-purple-400 transition-colors">
                Text to Speech
              </h3>
              <p className="text-sm text-gray-400">Generate speech from text</p>
            </Link>

            <Link
              href="/dashboard/audio"
              className="glass rounded-xl p-6 hover:bg-white/10 transition-colors group"
            >
              <div className="w-12 h-12 rounded-lg bg-cyan-600/20 flex items-center justify-center mb-4">
                <Sparkles className="h-6 w-6 text-cyan-500" />
              </div>
              <h3 className="font-semibold mb-1 group-hover:text-cyan-400 transition-colors">
                Voice Convert
              </h3>
              <p className="text-sm text-gray-400">Transform audio with AI voices</p>
            </Link>

            <Link
              href="/dashboard/song-remix"
              className="glass rounded-xl p-6 hover:bg-white/10 transition-colors group"
            >
              <div className="w-12 h-12 rounded-lg bg-accent-600/20 flex items-center justify-center mb-4">
                <Music className="h-6 w-6 text-accent-500" />
              </div>
              <h3 className="font-semibold mb-1 group-hover:text-accent-400 transition-colors">
                Song Remix
              </h3>
              <p className="text-sm text-gray-400">Split vocals & swap voices</p>
            </Link>

            <Link
              href="/models?tab=my-models"
              className="glass rounded-xl p-6 hover:bg-white/10 transition-colors group"
            >
              <div className="w-12 h-12 rounded-lg bg-green-600/20 flex items-center justify-center mb-4">
                <Box className="h-6 w-6 text-green-500" />
              </div>
              <h3 className="font-semibold mb-1 group-hover:text-green-400 transition-colors">
                My Models
              </h3>
              <p className="text-sm text-gray-400">Manage your voice models</p>
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
    </DashboardLayout>
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
