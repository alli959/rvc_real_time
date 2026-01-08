'use client';

import Link from 'next/link';
import { Mic2, Github, Linkedin, Mail } from 'lucide-react';

interface FooterProps {
  minimal?: boolean;
}

export function Footer({ minimal = false }: FooterProps) {
  const currentYear = new Date().getFullYear();

  if (minimal) {
    return (
      <footer className="py-4 px-4 border-t border-gray-800/50 bg-gray-900/30">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-gray-500">
          <span>© {currentYear} MorphVox</span>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/alli959/rvc_real_time"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary-400 transition-colors"
            >
              <Github className="h-4 w-4" />
            </a>
            <a
              href="https://www.linkedin.com/in/alexander-gu%C3%B0mundsson-053200189/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary-400 transition-colors"
            >
              <Linkedin className="h-4 w-4" />
            </a>
          </div>
        </div>
      </footer>
    );
  }

  return (
    <footer className="py-12 px-4 border-t border-gray-800 bg-gray-900/50">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <Mic2 className="h-8 w-8 text-primary-500" />
              <span className="text-xl font-bold">MorphVox</span>
            </div>
            <p className="text-gray-400 text-sm max-w-md">
              AI-powered voice conversion platform. Transform your voice with state-of-the-art 
              RVC technology, separate vocals, and generate natural text-to-speech.
            </p>
          </div>

          {/* Links */}
          <div>
            <h3 className="font-semibold text-white mb-4">Platform</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>
                <Link href="/dashboard" className="hover:text-primary-400 transition-colors">
                  Dashboard
                </Link>
              </li>
              <li>
                <Link href="/dashboard/models" className="hover:text-primary-400 transition-colors">
                  Voice Models
                </Link>
              </li>
              <li>
                <Link href="/dashboard/tts" className="hover:text-primary-400 transition-colors">
                  Text to Speech
                </Link>
              </li>
              <li>
                <Link href="/dashboard/audio" className="hover:text-primary-400 transition-colors">
                  Audio Processing
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal & Resources */}
          <div>
            <h3 className="font-semibold text-white mb-4">Resources</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>
                <a
                  href="https://github.com/alli959/rvc_real_time"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-primary-400 transition-colors flex items-center gap-2"
                >
                  <Github className="h-4 w-4" />
                  GitHub Repository
                </a>
              </li>
              <li>
                <Link href="/privacy" className="hover:text-primary-400 transition-colors">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/terms" className="hover:text-primary-400 transition-colors">
                  Terms of Service
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-gray-800 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-sm text-gray-500">
            © {currentYear} MorphVox. All rights reserved.
          </div>

          {/* Creator Info */}
          <div className="flex items-center gap-6">
            <span className="text-sm text-gray-400">
              Created by{' '}
              <a
                href="https://www.linkedin.com/in/alexander-gu%C3%B0mundsson-053200189/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-400 hover:text-primary-300 transition-colors font-medium"
              >
                Alexander Guðmundsson
              </a>
            </span>
            
            {/* Social Links */}
            <div className="flex items-center gap-3">
              <a
                href="https://github.com/alli959/rvc_real_time"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
                title="GitHub Repository"
              >
                <Github className="h-5 w-5" />
              </a>
              <a
                href="https://www.linkedin.com/in/alexander-gu%C3%B0mundsson-053200189/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
                title="LinkedIn"
              >
                <Linkedin className="h-5 w-5" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
