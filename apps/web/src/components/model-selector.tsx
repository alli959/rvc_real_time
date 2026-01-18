'use client';

import { useState, useEffect, useRef } from 'react';
import { ChevronDown, Globe, User, Users, Loader2, Search } from 'lucide-react';
import { voiceModelsApi, VoiceModel } from '@/lib/api';
import { useAuthStore } from '@/lib/store';

interface ModelSelectorProps {
  value: number | string | null;
  onChange: (id: number | null, model?: VoiceModel) => void;
  className?: string;
  disabled?: boolean;
  placeholder?: string;
  accentColor?: 'primary' | 'accent'; // For different page themes
}

interface GroupedModels {
  myModels: VoiceModel[];
  sharedModels: VoiceModel[];
  communityModels: VoiceModel[];
}

export function ModelSelector({
  value,
  onChange,
  className = '',
  disabled = false,
  placeholder = 'Select a voice model...',
  accentColor = 'primary',
}: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [groupedModels, setGroupedModels] = useState<GroupedModels>({
    myModels: [],
    sharedModels: [],
    communityModels: [],
  });
  const dropdownRef = useRef<HTMLDivElement>(null);
  const user = useAuthStore((state) => state.user);

  // Load models on mount
  useEffect(() => {
    loadModels();
  }, [user]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const loadModels = async () => {
    setLoading(true);
    setError(null);
    try {
      // Load all models (will include user's models if authenticated)
      const [allModelsRes, myModelsRes] = await Promise.all([
        voiceModelsApi.list({ per_page: 200 }),
        user ? voiceModelsApi.myModels({ per_page: 100 }) : Promise.resolve({ data: [] }),
      ]);

      const allModels = allModelsRes.data || [];
      const userModels = myModelsRes.data || [];

      // Separate owned models from shared models using is_owned flag
      const myModels = userModels.filter((m: VoiceModel & { is_owned?: boolean }) => m.is_owned !== false);
      const sharedModels = userModels.filter((m: VoiceModel & { is_owned?: boolean }) => m.is_owned === false);

      // Get IDs of user's accessible models (owned + shared)
      const userModelIds = new Set(userModels.map((m: VoiceModel) => m.id));

      // Community models are public models that are not in your accessible models
      const communityModels = allModels.filter(
        (m: VoiceModel) => m.visibility === 'public' && !userModelIds.has(m.id)
      );

      setGroupedModels({
        myModels,
        sharedModels,
        communityModels,
      });
    } catch (err: any) {
      console.error('Failed to load models:', err);
      setError('Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  // Get selected model
  const allModels = [
    ...groupedModels.myModels,
    ...groupedModels.sharedModels,
    ...groupedModels.communityModels,
  ];
  const selectedModel = allModels.find((m) => m.id === value);

  // Filter models by search
  const filterModels = (models: VoiceModel[]) => {
    if (!search) return models;
    const searchLower = search.toLowerCase();
    return models.filter((m) => m.name.toLowerCase().includes(searchLower));
  };

  const filteredMyModels = filterModels(groupedModels.myModels);
  const filteredSharedModels = filterModels(groupedModels.sharedModels);
  const filteredCommunityModels = filterModels(groupedModels.communityModels);

  const hasResults =
    filteredMyModels.length > 0 ||
    filteredSharedModels.length > 0 ||
    filteredCommunityModels.length > 0;

  const accentClasses = {
    primary: {
      ring: 'focus:ring-primary-500',
      border: 'focus:border-primary-500',
      hover: 'hover:bg-primary-600/20',
      selected: 'bg-primary-600/30 text-primary-300',
    },
    accent: {
      ring: 'focus:ring-accent-500',
      border: 'focus:border-accent-500',
      hover: 'hover:bg-accent-600/20',
      selected: 'bg-accent-600/30 text-accent-300',
    },
  };

  const colors = accentClasses[accentColor];

  const handleSelect = (model: VoiceModel) => {
    onChange(model.id, model);
    setIsOpen(false);
    setSearch('');
  };

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled || loading}
        className={`w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-left flex items-center justify-between gap-2 ${colors.ring} focus:ring-2 focus:outline-none ${
          disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:border-gray-600'
        }`}
      >
        {loading ? (
          <span className="flex items-center gap-2 text-gray-400">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading models...
          </span>
        ) : selectedModel ? (
          <span className="flex items-center gap-2 text-white truncate">
            {/* Selected Model Image */}
            <div className="w-6 h-6 rounded overflow-hidden bg-gray-700 flex-shrink-0">
              {selectedModel.image_url ? (
                <img
                  src={selectedModel.image_url}
                  alt={selectedModel.name}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-500">
                  <User className="w-3 h-3" />
                </div>
              )}
            </div>
            <span className="truncate">{selectedModel.name}</span>
          </span>
        ) : (
          <span className="text-gray-400">{placeholder}</span>
        )}
        <ChevronDown
          className={`h-4 w-4 text-gray-400 transition-transform flex-shrink-0 ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {/* Dropdown */}
      {isOpen && !loading && (
        <div className="absolute z-50 w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl max-h-80 overflow-hidden">
          {/* Search Input */}
          <div className="p-2 border-b border-gray-700">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search models..."
                className="w-full pl-9 pr-3 py-2 bg-gray-900 border border-gray-700 rounded-md text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                autoFocus
              />
            </div>
          </div>

          {/* Model Groups */}
          <div className="overflow-y-auto max-h-60">
            {error ? (
              <div className="p-4 text-center text-red-400 text-sm">{error}</div>
            ) : !hasResults ? (
              <div className="p-4 text-center text-gray-400 text-sm">No models found</div>
            ) : (
              <>
                {/* My Models */}
                {filteredMyModels.length > 0 && (
                  <ModelGroup
                    title="My Models"
                    icon={<User className="h-3.5 w-3.5" />}
                    models={filteredMyModels}
                    selectedId={value}
                    onSelect={handleSelect}
                    colors={colors}
                  />
                )}

                {/* Shared with Me */}
                {filteredSharedModels.length > 0 && (
                  <ModelGroup
                    title="Shared with Me"
                    icon={<Users className="h-3.5 w-3.5" />}
                    models={filteredSharedModels}
                    selectedId={value}
                    onSelect={handleSelect}
                    colors={colors}
                  />
                )}

                {/* Community Models */}
                {filteredCommunityModels.length > 0 && (
                  <ModelGroup
                    title="Community Models"
                    icon={<Globe className="h-3.5 w-3.5" />}
                    models={filteredCommunityModels}
                    selectedId={value}
                    onSelect={handleSelect}
                    colors={colors}
                  />
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

interface ModelGroupProps {
  title: string;
  icon: React.ReactNode;
  models: VoiceModel[];
  selectedId: number | string | null;
  onSelect: (model: VoiceModel) => void;
  colors: {
    hover: string;
    selected: string;
  };
}

function ModelGroup({ title, icon, models, selectedId, onSelect, colors }: ModelGroupProps) {
  return (
    <div>
      {/* Group Header */}
      <div className="px-3 py-2 bg-gray-900/50 border-y border-gray-700/50 sticky top-0">
        <span className="flex items-center gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          {icon}
          {title}
          <span className="text-gray-500 font-normal">({models.length})</span>
        </span>
      </div>

      {/* Group Items */}
      <div className="py-1">
        {models.map((model) => (
          <button
            key={model.id}
            onClick={() => onSelect(model)}
            className={`w-full px-3 py-2 text-left text-sm transition-colors flex items-center gap-3 ${
              selectedId === model.id
                ? colors.selected
                : `text-white ${colors.hover}`
            }`}
          >
            {/* Model Image */}
            <div className="w-8 h-8 rounded-md overflow-hidden bg-gray-700 flex-shrink-0">
              {model.image_url ? (
                <img
                  src={model.image_url}
                  alt={model.name}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-500">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
            {/* Model Name */}
            <span className="truncate">{model.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
