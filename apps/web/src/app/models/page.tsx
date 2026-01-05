'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Link from 'next/link';
import { Search, Mic2, HardDrive, FileAudio, Check, X, Volume2, Cloud, Server } from 'lucide-react';
import { voiceModelsApi, SystemVoiceModel } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

export default function ModelsPage() {
  const [search, setSearch] = useState('');
  const [selectedModel, setSelectedModel] = useState<SystemVoiceModel | null>(null);
  const user = useAuthStore((state) => state.user);

  const { data, isLoading, error } = useQuery({
    queryKey: ['voice-models', search],
    queryFn: () => voiceModelsApi.list({ search }),
  });

  const models: SystemVoiceModel[] = data?.data || [];

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center gap-2">
              <Mic2 className="h-8 w-8 text-primary-500" />
              <span className="text-xl font-bold gradient-text">MorphVox</span>
            </Link>
            <div className="flex items-center gap-4">
              {user ? (
                <Link href="/dashboard" className="text-gray-300 hover:text-white transition-colors">
                  Dashboard
                </Link>
              ) : (
                <>
                  <Link href="/login" className="text-gray-300 hover:text-white transition-colors">
                    Sign In
                  </Link>
                  <Link
                    href="/register"
                    className="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    Get Started
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Title */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Voice Models</h1>
          <p className="text-gray-400">
            Browse available voice models on this server. Select a model to use for voice conversion.
          </p>
        </div>

        {/* Search */}
        <div className="flex flex-col sm:flex-row gap-4 mb-8">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search models..."
              className="w-full bg-gray-800/50 border border-gray-700 rounded-lg pl-10 pr-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <HardDrive className="h-4 w-4" />
            <span>{models.length} models available</span>
          </div>
        </div>

        {/* Models Grid */}
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <ModelCardSkeleton key={i} />
            ))}
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <p className="text-red-400">Failed to load models. Make sure the API server is running.</p>
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-12">
            <Mic2 className="h-12 w-12 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">No models found</p>
            <p className="text-gray-500 text-sm mt-2">
              Add model folders to <code className="bg-gray-800 px-2 py-1 rounded">services/voice-engine/assets/models/</code>
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {models.map((model) => (
              <ModelCard 
                key={model.slug} 
                model={model} 
                isSelected={selectedModel?.slug === model.slug}
                onSelect={() => setSelectedModel(model)}
              />
            ))}
          </div>
        )}

        {/* Selected Model Action Bar */}
        {selectedModel && (
          <div className="fixed bottom-0 left-0 right-0 bg-gray-900/95 backdrop-blur-sm border-t border-gray-800 p-4 z-50">
            <div className="max-w-7xl mx-auto flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-primary-600 to-accent-600 rounded-lg flex items-center justify-center">
                  <Mic2 className="h-6 w-6 text-white" />
                </div>
                <div>
                  <p className="font-semibold">{selectedModel.name}</p>
                  <p className="text-sm text-gray-400">{selectedModel.model_file} • {selectedModel.size}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setSelectedModel(null)}
                  className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <Link
                  href={`/dashboard/convert?model=${encodeURIComponent(selectedModel.slug)}`}
                  className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-6 py-2 rounded-lg transition-colors"
                >
                  <Volume2 className="h-4 w-4" />
                  Use This Model
                </Link>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function ModelCard({ 
  model, 
  isSelected, 
  onSelect 
}: { 
  model: SystemVoiceModel; 
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={`glass rounded-xl overflow-hidden cursor-pointer transition-all group ${
        isSelected 
          ? 'ring-2 ring-primary-500 bg-primary-500/10' 
          : 'hover:bg-white/5'
      }`}
    >
      {/* Thumbnail */}
      <div className="aspect-video bg-gradient-to-br from-primary-900/50 to-accent-900/50 flex items-center justify-center relative">
        <Mic2 className="h-16 w-16 text-gray-600" />
        {isSelected && (
          <div className="absolute top-3 right-3 w-8 h-8 bg-primary-500 rounded-full flex items-center justify-center">
            <Check className="h-5 w-5 text-white" />
          </div>
        )}
        {/* Storage type indicator */}
        <div className="absolute top-3 left-3 bg-gray-900/80 text-gray-400 text-xs px-2 py-1 rounded flex items-center gap-1">
          {model.storage_type === 's3' ? (
            <>
              <Cloud className="h-3 w-3" />
              <span>Cloud</span>
            </>
          ) : (
            <>
              <Server className="h-3 w-3" />
              <span>Local</span>
            </>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        <h3 className="font-semibold text-lg mb-2 group-hover:text-primary-400 transition-colors">
          {model.name}
        </h3>

        <div className="space-y-2 text-sm">
          {/* Model file */}
          <div className="flex items-center gap-2 text-gray-400">
            <FileAudio className="h-4 w-4 flex-shrink-0" />
            <span className="truncate">{model.model_file}</span>
          </div>

          {/* Index file */}
          <div className="flex items-center gap-2">
            {model.has_index ? (
              <>
                <Check className="h-4 w-4 text-green-500 flex-shrink-0" />
                <span className="text-green-400 text-sm">Index: {model.index_file}</span>
              </>
            ) : (
              <>
                <X className="h-4 w-4 text-yellow-500 flex-shrink-0" />
                <span className="text-yellow-400 text-sm">No index file</span>
              </>
            )}
          </div>

          {/* Size */}
          <div className="flex items-center justify-between text-gray-500">
            <span>{model.size}</span>
            <span className="bg-gray-800 px-2 py-0.5 rounded text-xs uppercase">
              {model.engine}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ModelCardSkeleton() {
  return (
    <div className="glass rounded-xl overflow-hidden animate-pulse">
      <div className="aspect-video bg-gray-800" />
      <div className="p-4 space-y-3">
        <div className="h-5 bg-gray-800 rounded w-3/4" />
        <div className="h-4 bg-gray-800 rounded w-full" />
        <div className="h-4 bg-gray-800 rounded w-2/3" />
      </div>
    </div>
  );
}
