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
              Generate natural-sounding speech from text with optional voice conversion
            </p>
          </div>
        </div>

        {/* Info Card */}
        <div className="bg-gradient-to-r from-primary-900/30 to-purple-900/30 border border-primary-500/20 rounded-lg p-4">
          <h3 className="font-medium text-white mb-2">How it works</h3>
          <ol className="text-sm text-gray-300 space-y-1 list-decimal list-inside">
            <li>Enter the text you want to convert to speech</li>
            <li>Select a voice and speaking style</li>
            <li>Optionally enable Voice Conversion to transform the speech using your custom voice models</li>
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
              <span>Try different speaking styles to match the emotion of your text</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-400">•</span>
              <span>Voice conversion works best with clean TTS output - adjust pitch if the result sounds unnatural</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-400">•</span>
              <span>For characters, use the Voice Conversion feature to apply their unique voice model</span>
            </li>
          </ul>
        </div>
      </div>
    </DashboardLayout>
  );
}
