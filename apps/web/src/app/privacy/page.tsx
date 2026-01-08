'use client';

import Link from 'next/link';
import { Mic2, ArrowLeft } from 'lucide-react';
import { Footer } from '@/components/footer';

export default function PrivacyPolicyPage() {
  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center h-16 gap-4">
            <Link href="/" className="flex items-center gap-2">
              <Mic2 className="h-8 w-8 text-primary-500" />
              <span className="text-xl font-bold gradient-text">MorphVox</span>
            </Link>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <Link 
          href="/" 
          className="inline-flex items-center gap-2 text-gray-400 hover:text-white mb-8 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Home
        </Link>

        <h1 className="text-4xl font-bold mb-8">Privacy Policy</h1>
        <p className="text-gray-400 mb-8">Last updated: January 7, 2026</p>

        <div className="prose prose-invert max-w-none space-y-8">
          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">1. Introduction</h2>
            <p className="text-gray-300 mb-4">
              Welcome to MorphVox (&quot;we,&quot; &quot;our,&quot; or &quot;us&quot;). We are committed to protecting your personal information 
              and your right to privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard 
              your information when you use our voice conversion platform.
            </p>
            <p className="text-gray-300">
              Please read this privacy policy carefully. If you do not agree with the terms of this privacy policy, 
              please do not access the platform.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">2. Information We Collect</h2>
            
            <h3 className="text-xl font-medium mb-3">2.1 Personal Information</h3>
            <p className="text-gray-300 mb-4">We may collect personal information that you voluntarily provide to us when you:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2 mb-4">
              <li>Register for an account</li>
              <li>Use our voice conversion services</li>
              <li>Upload voice models or audio files</li>
              <li>Contact us for support</li>
            </ul>
            <p className="text-gray-300 mb-4">This information may include:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Name and email address</li>
              <li>Account credentials</li>
              <li>Audio files and voice recordings you upload</li>
              <li>Voice models you create or upload</li>
              <li>Usage data and preferences</li>
            </ul>

            <h3 className="text-xl font-medium mb-3 mt-6">2.2 Automatically Collected Information</h3>
            <p className="text-gray-300 mb-4">When you access our platform, we may automatically collect:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Device information (browser type, operating system)</li>
              <li>IP address and location data</li>
              <li>Usage patterns and platform interactions</li>
              <li>Cookies and similar tracking technologies</li>
            </ul>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">3. How We Use Your Information</h2>
            <p className="text-gray-300 mb-4">We use the information we collect to:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Provide and maintain our voice conversion services</li>
              <li>Process your voice conversion requests</li>
              <li>Create and manage your account</li>
              <li>Respond to your inquiries and provide customer support</li>
              <li>Improve our platform and develop new features</li>
              <li>Send you updates and marketing communications (with your consent)</li>
              <li>Detect and prevent fraud or abuse</li>
              <li>Comply with legal obligations</li>
            </ul>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">4. Audio Data and Voice Models</h2>
            <p className="text-gray-300 mb-4">
              <strong>Audio Processing:</strong> When you use our voice conversion services, we process your audio 
              files to provide the requested conversions. Audio files are processed securely and are not permanently 
              stored unless you choose to save them.
            </p>
            <p className="text-gray-300 mb-4">
              <strong>Voice Models:</strong> Voice models you upload or create are stored securely and are subject 
              to your chosen privacy settings (public, private, or unlisted). We do not use your private voice 
              models for any purpose other than providing you with our services.
            </p>
            <p className="text-gray-300">
              <strong>Consent:</strong> You represent that you have obtained all necessary consents and permissions 
              for any voice data you upload, especially when creating voice models based on other individuals&apos; voices.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">5. Data Sharing and Disclosure</h2>
            <p className="text-gray-300 mb-4">We may share your information in the following circumstances:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li><strong>Service Providers:</strong> With third-party vendors who assist in operating our platform</li>
              <li><strong>Legal Requirements:</strong> When required by law or to respond to legal process</li>
              <li><strong>Business Transfers:</strong> In connection with a merger, acquisition, or sale of assets</li>
              <li><strong>With Your Consent:</strong> When you have given us permission to share your information</li>
            </ul>
            <p className="text-gray-300 mt-4">
              We do not sell your personal information to third parties.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">6. Data Security</h2>
            <p className="text-gray-300 mb-4">
              We implement appropriate technical and organizational security measures to protect your personal 
              information, including:
            </p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Encryption of data in transit and at rest</li>
              <li>Secure authentication mechanisms</li>
              <li>Regular security assessments and audits</li>
              <li>Access controls and monitoring</li>
            </ul>
            <p className="text-gray-300 mt-4">
              However, no method of transmission over the Internet is 100% secure, and we cannot guarantee 
              absolute security.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">7. Your Rights</h2>
            <p className="text-gray-300 mb-4">Depending on your location, you may have the following rights:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li><strong>Access:</strong> Request access to your personal information</li>
              <li><strong>Correction:</strong> Request correction of inaccurate information</li>
              <li><strong>Deletion:</strong> Request deletion of your personal information</li>
              <li><strong>Portability:</strong> Request a copy of your data in a portable format</li>
              <li><strong>Objection:</strong> Object to certain processing of your information</li>
              <li><strong>Withdrawal:</strong> Withdraw consent where processing is based on consent</li>
            </ul>
            <p className="text-gray-300 mt-4">
              To exercise these rights, please contact us at privacy@morphvox.ai.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">8. Cookies</h2>
            <p className="text-gray-300 mb-4">
              We use cookies and similar tracking technologies to collect and store information about your 
              preferences and activity on our platform. You can control cookies through your browser settings.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">9. Children&apos;s Privacy</h2>
            <p className="text-gray-300">
              Our platform is not intended for children under 13 years of age. We do not knowingly collect 
              personal information from children under 13. If we become aware that we have collected personal 
              information from a child under 13, we will take steps to delete such information.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">10. Changes to This Policy</h2>
            <p className="text-gray-300">
              We may update this Privacy Policy from time to time. We will notify you of any changes by posting 
              the new Privacy Policy on this page and updating the &quot;Last updated&quot; date. You are advised to review 
              this Privacy Policy periodically for any changes.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">11. Contact Us</h2>
            <p className="text-gray-300 mb-4">
              If you have any questions about this Privacy Policy, please contact us:
            </p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Email: privacy@morphvox.ai</li>
              <li>Website: <Link href="/" className="text-primary-400 hover:text-primary-300">morphvox.ai</Link></li>
            </ul>
          </section>
        </div>
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
}
