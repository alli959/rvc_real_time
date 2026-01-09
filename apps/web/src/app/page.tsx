import Link from 'next/link';
import { Sparkles, Zap, Shield, Mic2, Upload, Split, Music } from 'lucide-react';
import { Navbar } from '@/components/navbar';
import { Footer } from '@/components/footer';
import { AuthAwareHero, AuthAwareCTA } from '@/components/home-auth-sections';

export default function Home() {
  return (
    <main className="min-h-screen">
      {/* Navigation */}
      <Navbar />

      {/* Hero Section - Auth Aware */}
      <AuthAwareHero />

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

      {/* CTA Section - Auth Aware */}
      <AuthAwareCTA />

      {/* Footer */}
      <Footer />
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
