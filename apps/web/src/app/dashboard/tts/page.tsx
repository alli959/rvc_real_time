'use client';

import { DashboardLayout } from '@/components/dashboard-layout';
import { TTSGenerator } from '@/components/tts-generator';
import { Volume2 as SpeakerWaveIcon } from 'lucide-react';

export default function TTSPage() {
  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary-600/20 rounded-lg">
            <SpeakerWaveIcon className="h-6 w-6 text-primary-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Text to Speech</h1>
            <p className="text-gray-400">
              Generate expressive speech with emotions, sound effects, and voice conversion
            </p>
          </div>
        </div>

        {/* Info Card */}
        <div className="bg-gradient-to-r from-primary-900/30 to-purple-900/30 border border-primary-500/20 rounded-lg p-4">
          <h3 className="font-medium text-white mb-2">How it works</h3>
          <ol className="text-sm text-gray-300 space-y-1 list-decimal list-inside">
            <li>Choose your language and gender preference</li>
            <li>Enter the text you want to convert - add emotions using the picker!</li>
            <li>Select a Voice Model to transform the speech into any character</li>
            <li>Optionally add voice effects like Robot, Spooky, Phone, etc.</li>
            <li>Click Generate to create your audio</li>
          </ol>
        </div>

        {/* TTS Generator */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <TTSGenerator />
        </div>

        {/* Usage Tips */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
          <h3 className="font-medium text-white mb-3">Tips for best results</h3>
          <ul className="text-sm text-gray-400 space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-primary-400">•</span>
              <span>Use proper punctuation for natural pauses and intonation</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-400">•</span>
              <span>Add emotions like [happy], [sad], [angry] to make speech more expressive</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-400">•</span>
              <span>Insert sound effects like [laugh], [gasp], [sigh] for realistic reactions</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-400">•</span>
              <span>Adjust pitch shift if the converted voice sounds unnatural (negative = deeper)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-400">•</span>
              <span>Voice effects like Robot and Phone work best AFTER voice conversion</span>
            </li>
          </ul>
        </div>
      </div>
    </DashboardLayout>
  );
}
