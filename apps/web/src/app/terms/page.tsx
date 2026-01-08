'use client';

import Link from 'next/link';
import { Mic2, ArrowLeft } from 'lucide-react';

export default function TermsOfServicePage() {
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

        <h1 className="text-4xl font-bold mb-8">Terms of Service</h1>
        <p className="text-gray-400 mb-8">Last updated: January 7, 2026</p>

        <div className="prose prose-invert max-w-none space-y-8">
          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">1. Acceptance of Terms</h2>
            <p className="text-gray-300 mb-4">
              By accessing or using MorphVox (&quot;Service&quot;), you agree to be bound by these Terms of Service 
              (&quot;Terms&quot;). If you do not agree to all the terms and conditions, you may not access or use 
              the Service.
            </p>
            <p className="text-gray-300">
              We reserve the right to modify these Terms at any time. Your continued use of the Service 
              following any changes constitutes your acceptance of the revised Terms.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">2. Description of Service</h2>
            <p className="text-gray-300 mb-4">
              MorphVox is an AI-powered voice conversion platform that allows users to:
            </p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Convert voices using AI voice models</li>
              <li>Generate text-to-speech audio</li>
              <li>Upload and manage voice models</li>
              <li>Process and transform audio files</li>
              <li>Access community-created voice models</li>
            </ul>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">3. User Accounts</h2>
            <p className="text-gray-300 mb-4">To use certain features of the Service, you must create an account. You agree to:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Provide accurate and complete information during registration</li>
              <li>Maintain the security of your account credentials</li>
              <li>Notify us immediately of any unauthorized access</li>
              <li>Be responsible for all activities that occur under your account</li>
            </ul>
            <p className="text-gray-300 mt-4">
              We reserve the right to suspend or terminate accounts that violate these Terms.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">4. User Content and Conduct</h2>
            
            <h3 className="text-xl font-medium mb-3">4.1 Your Content</h3>
            <p className="text-gray-300 mb-4">
              You retain ownership of content you upload to the Service, including audio files and voice models. 
              By uploading content, you grant us a license to use, process, and store it to provide the Service.
            </p>

            <h3 className="text-xl font-medium mb-3">4.2 Content Restrictions</h3>
            <p className="text-gray-300 mb-4">You agree NOT to upload, create, or share content that:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Infringes on intellectual property rights of others</li>
              <li>Impersonates real individuals without their consent</li>
              <li>Is used for fraud, deception, or misinformation</li>
              <li>Contains illegal, harmful, or offensive material</li>
              <li>Violates privacy rights of individuals</li>
              <li>Is used for harassment, threats, or intimidation</li>
              <li>Creates deepfakes or synthetic media for malicious purposes</li>
            </ul>

            <h3 className="text-xl font-medium mb-3 mt-6">4.3 Voice Model Consent</h3>
            <p className="text-gray-300">
              When creating voice models based on another person&apos;s voice, you must have explicit consent 
              from that individual. You represent and warrant that you have obtained all necessary permissions 
              and rights for any voice data you upload.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">5. Prohibited Uses</h2>
            <p className="text-gray-300 mb-4">You may not use the Service to:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Create non-consensual intimate imagery or audio</li>
              <li>Impersonate public figures or private individuals for fraudulent purposes</li>
              <li>Generate content for political misinformation or election interference</li>
              <li>Create content that could cause real-world harm</li>
              <li>Bypass security measures or access unauthorized features</li>
              <li>Reverse engineer or copy the Service&apos;s technology</li>
              <li>Use automated systems to access the Service without permission</li>
              <li>Interfere with the operation of the Service</li>
            </ul>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">6. Intellectual Property</h2>
            
            <h3 className="text-xl font-medium mb-3">6.1 Our Property</h3>
            <p className="text-gray-300 mb-4">
              The Service, including its technology, design, and content created by us, is protected by 
              intellectual property laws. You may not copy, modify, or distribute our proprietary materials 
              without permission.
            </p>

            <h3 className="text-xl font-medium mb-3">6.2 User Content License</h3>
            <p className="text-gray-300">
              When you share voice models publicly, you grant other users a license to use those models 
              within the Service according to your chosen visibility settings.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">7. Privacy</h2>
            <p className="text-gray-300">
              Your use of the Service is also governed by our <Link href="/privacy" className="text-primary-400 hover:text-primary-300">Privacy Policy</Link>, 
              which describes how we collect, use, and protect your information.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">8. Disclaimers</h2>
            <p className="text-gray-300 mb-4">
              THE SERVICE IS PROVIDED &quot;AS IS&quot; AND &quot;AS AVAILABLE&quot; WITHOUT WARRANTIES OF ANY KIND, 
              EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO:
            </p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Warranties of merchantability or fitness for a particular purpose</li>
              <li>Warranties that the Service will be uninterrupted or error-free</li>
              <li>Warranties regarding the accuracy or reliability of results</li>
            </ul>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">9. Limitation of Liability</h2>
            <p className="text-gray-300">
              TO THE MAXIMUM EXTENT PERMITTED BY LAW, WE SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, 
              SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING LOSS OF PROFITS, DATA, OR GOODWILL, 
              ARISING FROM YOUR USE OF THE SERVICE.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">10. Indemnification</h2>
            <p className="text-gray-300">
              You agree to indemnify and hold harmless MorphVox and its officers, directors, employees, 
              and agents from any claims, damages, losses, or expenses arising from:
            </p>
            <ul className="list-disc list-inside text-gray-300 space-y-2 mt-4">
              <li>Your use of the Service</li>
              <li>Your violation of these Terms</li>
              <li>Your content or voice models</li>
              <li>Your violation of any third-party rights</li>
            </ul>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">11. Termination</h2>
            <p className="text-gray-300 mb-4">
              We may terminate or suspend your account and access to the Service at our sole discretion, 
              without notice, for conduct that we believe violates these Terms or is harmful to other users, 
              us, or third parties.
            </p>
            <p className="text-gray-300">
              Upon termination, your right to use the Service will immediately cease. Provisions that by 
              their nature should survive termination will remain in effect.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">12. Governing Law</h2>
            <p className="text-gray-300">
              These Terms shall be governed by and construed in accordance with applicable laws, without 
              regard to conflict of law principles. Any disputes arising from these Terms shall be resolved 
              in the appropriate courts.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">13. Severability</h2>
            <p className="text-gray-300">
              If any provision of these Terms is found to be unenforceable or invalid, that provision will 
              be limited or eliminated to the minimum extent necessary, and the remaining provisions will 
              remain in full force and effect.
            </p>
          </section>

          <section className="glass rounded-xl p-6">
            <h2 className="text-2xl font-semibold mb-4">14. Contact Information</h2>
            <p className="text-gray-300 mb-4">
              For questions about these Terms of Service, please contact us:
            </p>
            <ul className="list-disc list-inside text-gray-300 space-y-2">
              <li>Email: legal@morphvox.ai</li>
              <li>Website: <Link href="/" className="text-primary-400 hover:text-primary-300">morphvox.ai</Link></li>
            </ul>
          </section>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-12 py-8">
        <div className="max-w-4xl mx-auto px-4 text-center text-gray-500 text-sm">
          <p>&copy; {new Date().getFullYear()} MorphVox. All rights reserved.</p>
          <div className="flex justify-center gap-4 mt-4">
            <Link href="/privacy" className="hover:text-white transition-colors">Privacy Policy</Link>
            <Link href="/terms" className="hover:text-white transition-colors">Terms of Service</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
