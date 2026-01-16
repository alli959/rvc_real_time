'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { voiceModelsApi } from '@/lib/api';
import { 
  X,
  Mic2,
  Loader2,
  Lock,
  Globe,
  Link as LinkIcon,
} from 'lucide-react';

interface CreateEmptyModelFormProps {
  onSuccess?: (model: any) => void;
  onCancel?: () => void;
}

export function CreateEmptyModelForm({ onSuccess, onCancel }: CreateEmptyModelFormProps) {
  const router = useRouter();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [visibility, setVisibility] = useState<'private' | 'unlisted' | 'public'>('private');
  const [tags, setTags] = useState('');
  const [hasConsent, setHasConsent] = useState(false);
  
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name.trim()) {
      setError('Model name is required');
      return;
    }

    if (!hasConsent) {
      setError('Please confirm you have consent to create this voice model');
      return;
    }

    setCreating(true);
    setError(null);

    try {
      const result = await voiceModelsApi.create({
        name: name.trim(),
        description: description.trim() || undefined,
        engine: 'rvc',
        visibility,
        tags: tags ? tags.split(',').map(t => t.trim()).filter(Boolean) : [],
        has_consent: hasConsent,
      });

      // API returns { model: VoiceModel, upload_urls: {...} }
      const createdModel = result.model;

      if (onSuccess) {
        // Parent component handles the state transition, don't navigate
        onSuccess(createdModel);
      } else {
        // Standalone usage: navigate to the training page with the new model selected
        router.push(`/dashboard/train?model=${createdModel.id}`);
      }
    } catch (err: any) {
      console.error('Failed to create model:', err);
      setError(err.response?.data?.message || 'Failed to create model');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary-500/20 rounded-lg">
            <Mic2 className="h-5 w-5 text-primary-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Create New Voice Model</h2>
            <p className="text-sm text-gray-400">Start from scratch with recordings</p>
          </div>
        </div>
        {onCancel && (
          <button
            onClick={onCancel}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Model Name */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Model Name <span className="text-red-400">*</span>
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g., My Voice Clone"
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
            autoFocus
          />
        </div>

        {/* Description */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe your voice model..."
            rows={3}
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 resize-none"
          />
        </div>

        {/* Visibility */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Visibility
          </label>
          <div className="grid grid-cols-3 gap-3">
            <button
              type="button"
              onClick={() => setVisibility('private')}
              className={`p-3 rounded-lg border transition-all ${
                visibility === 'private'
                  ? 'border-primary-500 bg-primary-500/10 text-white'
                  : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600'
              }`}
            >
              <Lock className="h-5 w-5 mx-auto mb-1" />
              <span className="text-xs">Private</span>
            </button>
            <button
              type="button"
              onClick={() => setVisibility('unlisted')}
              className={`p-3 rounded-lg border transition-all ${
                visibility === 'unlisted'
                  ? 'border-primary-500 bg-primary-500/10 text-white'
                  : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600'
              }`}
            >
              <LinkIcon className="h-5 w-5 mx-auto mb-1" />
              <span className="text-xs">Unlisted</span>
            </button>
            <button
              type="button"
              onClick={() => setVisibility('public')}
              className={`p-3 rounded-lg border transition-all ${
                visibility === 'public'
                  ? 'border-primary-500 bg-primary-500/10 text-white'
                  : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600'
              }`}
            >
              <Globe className="h-5 w-5 mx-auto mb-1" />
              <span className="text-xs">Public</span>
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {visibility === 'private' && 'Only you can see and use this model'}
            {visibility === 'unlisted' && 'Anyone with the link can use this model'}
            {visibility === 'public' && 'Visible to everyone in the community'}
          </p>
        </div>

        {/* Tags */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Tags <span className="text-gray-500 text-xs">(comma separated)</span>
          </label>
          <input
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="male, english, natural"
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
          />
        </div>

        {/* Consent Checkbox */}
        <div className="flex items-start gap-3">
          <input
            type="checkbox"
            id="consent"
            checked={hasConsent}
            onChange={(e) => setHasConsent(e.target.checked)}
            className="mt-1 w-4 h-4 rounded border-gray-600 bg-gray-800 text-primary-500 focus:ring-primary-500 focus:ring-offset-gray-900"
          />
          <label htmlFor="consent" className="text-sm text-gray-300">
            I confirm that I have proper consent from the voice owner to create and use this voice model
          </label>
        </div>

        {/* Error Message */}
        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3 pt-2">
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="flex-1 px-4 py-3 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg transition-colors"
            >
              Cancel
            </button>
          )}
          <button
            type="submit"
            disabled={creating || !name.trim() || !hasConsent}
            className="flex-1 px-4 py-3 bg-primary-600 hover:bg-primary-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {creating ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Mic2 className="h-5 w-5" />
                Create & Start Recording
              </>
            )}
          </button>
        </div>

        <p className="text-xs text-gray-500 text-center">
          After creating, you&apos;ll be taken to the recording wizard to capture voice samples for training.
        </p>
      </form>
    </div>
  );
}
