'use client';

import { useState } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { TTSGenerator, getFullFeatureExample } from '@/components/tts-generator';
import { Volume2 as SpeakerWaveIcon, Gauge, Users, Sparkles, Copy, Check } from 'lucide-react';

export default function TTSPage() {
  const [selectedLanguage, setSelectedLanguage] = useState('');
  const [copiedFullExample, setCopiedFullExample] = useState(false);

  // Handle language change from child component
  const handleLanguageChange = (language: string) => {
    setSelectedLanguage(language);
  };

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
              Generate expressive speech with emotions, sound effects, speed control, and multi-voice support
            </p>
          </div>
        </div>

        {/* Info Card */}
        <div className="bg-gradient-to-r from-primary-900/30 to-purple-900/30 border border-primary-500/20 rounded-lg p-4">
          <h3 className="font-medium text-white mb-2">How it works</h3>
          <ol className="text-sm text-gray-300 space-y-1 list-decimal list-inside">
            <li>Choose your language and gender preference</li>
            <li>Enter the text you want to convert - add emotions, speed tags, or multi-voice!</li>
            <li>Adjust the base speech speed with the slider (or use inline speed tags)</li>
            <li>Select a Voice Model to transform the speech into any character</li>
            <li>Optionally add voice effects like Robot, Spooky, Phone, etc.</li>
            <li>Click Generate to create your audio</li>
          </ol>
        </div>

        {/* TTS Generator */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <TTSGenerator onLanguageChange={handleLanguageChange} />
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-4">
          {/* Speed Control */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Gauge className="h-5 w-5 text-cyan-400" />
              <h3 className="font-medium text-white">Speed Control</h3>
            </div>
            <p className="text-sm text-gray-400 mb-2">Control how fast or slow the speech is:</p>
            <ul className="text-xs text-gray-500 space-y-1">
              <li>• Use the base speed slider for overall speed</li>
              <li>• Use <code className="bg-gray-800 px-1 rounded">&lt;speed rate=&quot;-30%&quot;&gt;</code> for specific sections</li>
              <li>• Range: -50% (very slow) to +50% (very fast)</li>
            </ul>
          </div>

          {/* Multi-Voice */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Users className="h-5 w-5 text-pink-400" />
              <h3 className="font-medium text-white">Multi-Voice</h3>
            </div>
            <p className="text-sm text-gray-400 mb-2">Use multiple voice models in one generation:</p>
            <ul className="text-xs text-gray-500 space-y-1">
              <li>• Click &quot;Add Voice&quot; to insert another character</li>
              <li>• Each segment uses a different voice model</li>
              <li>• Great for dialogues and conversations!</li>
            </ul>
          </div>

          {/* Emotions & Effects */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="h-5 w-5 text-purple-400" />
              <h3 className="font-medium text-white">Emotions & Effects</h3>
            </div>
            <p className="text-sm text-gray-400 mb-2">Make speech more expressive:</p>
            <ul className="text-xs text-gray-500 space-y-1">
              <li>• Emotions: [happy], [sad], [angry], [whisper]</li>
              <li>• Sounds: [laugh], [gasp], [sigh], [scream]</li>
              <li>• Effects: [robot], [spooky], [phone], [underwater]</li>
            </ul>
          </div>
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
              <span className="text-cyan-400">•</span>
              <span><strong>Speed too fast?</strong> Use base speed slider set to -20% to -30% for most natural results</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-400">•</span>
              <span>Add emotions like [happy], [sad], [angry] to make speech more expressive</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-400">•</span>
              <span>Insert sound effects like [laugh], [gasp], [sigh] for realistic reactions</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-pink-400">•</span>
              <span><strong>Multi-voice dialogues:</strong> Use &lt;include voice_model_id=&quot;X&quot;&gt; for conversations between characters</span>
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

        {/* Example */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-white">Example with all features</h3>
            <button
              onClick={() => {
                navigator.clipboard.writeText(getFullFeatureExample(selectedLanguage));
                setCopiedFullExample(true);
                setTimeout(() => setCopiedFullExample(false), 2000);
              }}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-lg border border-gray-700 bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition-colors"
            >
              {copiedFullExample ? (
                <>
                  <Check className="h-3.5 w-3.5 text-green-400" />
                  <span className="text-green-400">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
          <code className="block text-xs text-gray-300 bg-gray-800 p-3 rounded-lg whitespace-pre-wrap break-all">
            {getFullFeatureExample(selectedLanguage)}
          </code>
        </div>
      </div>
    </DashboardLayout>
  );
}
