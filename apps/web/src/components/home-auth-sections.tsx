'use client';

import Link from 'next/link';
import { Sparkles, Mic2, Upload, Split, Music, MessageSquare, Globe } from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { useAuthContext } from '@/components/providers';

// Waveform animation component
function WaveformAnimation() {
  return (
    <div className="mt-16 flex items-center justify-center gap-1">
      {[...Array(20)].map((_, i) => (
        <div
          key={i}
          className="w-1 bg-gradient-to-t from-primary-600 to-accent-500 rounded-full animate-pulse"
          style={{
            height: `${20 + (i % 5) * 10}px`,
            animationDelay: `${i * 50}ms`,
            animationDuration: '1s',
          }}
        />
      ))}
    </div>
  );
}

// Quick action card for authenticated users
function QuickActionCard({
  href,
  icon,
  title,
  description,
}: {
  href: string;
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <Link
      href={href}
      className="glass rounded-xl p-6 hover:bg-white/10 transition-all hover:scale-105 group"
    >
      <div className="mb-4 text-primary-400 group-hover:text-primary-300 transition-colors">
        {icon}
      </div>
      <h3 className="text-lg font-semibold mb-2 text-white">{title}</h3>
      <p className="text-sm text-gray-400">{description}</p>
    </Link>
  );
}

export function AuthAwareHero() {
  const { isAuthenticated, user } = useAuthStore();
  const { isReady } = useAuthContext();

  // Show a minimal loading state while checking auth
  if (!isReady) {
    return (
      <section className="pt-32 pb-20 px-4">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-5xl md:text-7xl font-bold mb-6">
            Transform Your Voice with{' '}
            <span className="gradient-text">AI</span>
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8">
            Real-time voice conversion powered by state-of-the-art RVC technology. 
            Create, share, and use custom voice models in seconds.
          </p>
          <WaveformAnimation />
        </div>
      </section>
    );
  }

  // Authenticated user experience
  if (isAuthenticated && user) {
    return (
      <section className="pt-32 pb-20 px-4">
        <div className="max-w-7xl mx-auto">
          {/* Personalized greeting */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              Welcome back, <span className="gradient-text">{user.name?.split(' ')[0] || 'Creator'}</span>
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Ready to create something amazing? Pick up where you left off or try something new.
            </p>
          </div>

          {/* Quick Actions Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            <QuickActionCard
              href="/dashboard/tts"
              icon={<MessageSquare className="h-8 w-8" />}
              title="Text to Speech"
              description="Generate speech with AI voices and emotion tags"
            />
            <QuickActionCard
              href="/dashboard/audio"
              icon={<Sparkles className="h-8 w-8" />}
              title="Voice Convert"
              description="Transform audio with AI voice models"
            />
            <QuickActionCard
              href="/dashboard/song-remix"
              icon={<Music className="h-8 w-8" />}
              title="Song Remix"
              description="Split vocals and swap voices in songs"
            />
            <QuickActionCard
              href="/models?tab=community"
              icon={<Globe className="h-8 w-8" />}
              title="Community Models"
              description="Browse public voice models from creators"
            />
          </div>

          {/* Waveform */}
          <WaveformAnimation />
        </div>
      </section>
    );
  }

  // Guest (unauthenticated) experience - original hero
  return (
    <section className="pt-32 pb-20 px-4">
      <div className="max-w-7xl mx-auto text-center">
        <h1 className="text-5xl md:text-7xl font-bold mb-6">
          Transform Your Voice with{' '}
          <span className="gradient-text">AI</span>
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8">
          Real-time voice conversion powered by state-of-the-art RVC technology. 
          Create, share, and use custom voice models in seconds.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link 
            href="/register" 
            className="bg-primary-600 hover:bg-primary-700 text-white px-8 py-3 rounded-lg text-lg font-semibold transition-colors flex items-center gap-2"
          >
            <Sparkles className="h-5 w-5" />
            Start Converting
          </Link>
          <Link 
            href="/models" 
            className="glass hover:bg-white/10 text-white px-8 py-3 rounded-lg text-lg font-semibold transition-colors"
          >
            Browse Models
          </Link>
        </div>

        <WaveformAnimation />
      </div>
    </section>
  );
}

export function AuthAwareCTA() {
  const { isAuthenticated } = useAuthStore();
  const { isReady } = useAuthContext();

  // Show nothing or minimal while checking auth
  if (!isReady) {
    return null;
  }

  // Don't show CTA to authenticated users
  if (isAuthenticated) {
    return (
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto glass rounded-2xl p-12 text-center">
          <h2 className="text-3xl font-bold mb-4">Explore More Features</h2>
          <p className="text-gray-400 mb-8">
            Discover advanced tools for voice conversion, audio processing, and more in your dashboard.
          </p>
          <Link 
            href="/dashboard" 
            className="bg-gradient-to-r from-primary-600 to-accent-600 hover:from-primary-700 hover:to-accent-700 text-white px-8 py-3 rounded-lg text-lg font-semibold transition-all inline-flex items-center gap-2"
          >
            <Sparkles className="h-5 w-5" />
            Go to Dashboard
          </Link>
        </div>
      </section>
    );
  }

  // Guest CTA - original
  return (
    <section className="py-20 px-4">
      <div className="max-w-4xl mx-auto glass rounded-2xl p-12 text-center">
        <h2 className="text-3xl font-bold mb-4">Ready to Transform?</h2>
        <p className="text-gray-400 mb-8">
          Join thousands of creators using MorphVox for content creation, 
          entertainment, and accessibility.
        </p>
        <Link 
          href="/register" 
          className="bg-gradient-to-r from-primary-600 to-accent-600 hover:from-primary-700 hover:to-accent-700 text-white px-8 py-3 rounded-lg text-lg font-semibold transition-all"
        >
          Create Free Account
        </Link>
      </div>
    </section>
  );
}
