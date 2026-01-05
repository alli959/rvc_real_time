import Link from 'next/link';
import { Mic2, Sparkles, Zap, Shield } from 'lucide-react';

export default function Home() {
  return (
    <main className="min-h-screen">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <Mic2 className="h-8 w-8 text-primary-500" />
              <span className="text-xl font-bold gradient-text">MorphVox</span>
            </div>
            <div className="flex items-center gap-4">
              <Link 
                href="/models" 
                className="text-gray-300 hover:text-white transition-colors"
              >
                Models
              </Link>
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
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
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

          {/* Waveform Animation */}
          <div className="mt-16 flex items-center justify-center gap-1">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="w-1 bg-gradient-to-t from-primary-600 to-accent-500 rounded-full waveform-bar"
                style={{
                  height: `${Math.random() * 40 + 20}px`,
                  animationDelay: `${i * 50}ms`,
                }}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            Why <span className="gradient-text">MorphVox</span>?
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Zap className="h-8 w-8 text-primary-500" />}
              title="Real-Time Conversion"
              description="Experience instant voice transformation with our optimized inference engine. Perfect for live streaming and calls."
            />
            <FeatureCard
              icon={<Sparkles className="h-8 w-8 text-accent-500" />}
              title="Custom Models"
              description="Upload and train your own voice models. Create unique voices or clone existing ones with just a few samples."
            />
            <FeatureCard
              icon={<Shield className="h-8 w-8 text-green-500" />}
              title="Privacy First"
              description="Your voice data is processed securely. Control who can access your models with granular permissions."
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
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

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-gray-800">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Mic2 className="h-6 w-6 text-primary-500" />
            <span className="font-semibold">MorphVox</span>
          </div>
          <p className="text-gray-500 text-sm">
            © {new Date().getFullYear()} MorphVox. All rights reserved.
          </p>
        </div>
      </footer>
    </main>
  );
}

function FeatureCard({ 
  icon, 
  title, 
  description 
}: { 
  icon: React.ReactNode; 
  title: string; 
  description: string;
}) {
  return (
    <div className="glass rounded-xl p-6 hover:bg-white/10 transition-colors">
      <div className="mb-4">{icon}</div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </div>
  );
}
