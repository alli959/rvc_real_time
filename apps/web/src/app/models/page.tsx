'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import Link from 'next/link';
import { 
  Search, Mic2, HardDrive, FileAudio, Check, X, Cloud, Server, Lock, Eye, Upload, 
  Plus, Trash2, Pencil, Globe, Link as LinkIcon, RefreshCw, Settings, Users,
  ChevronLeft, MessageSquare, AudioWaveform, Loader2, Music
} from 'lucide-react';
import { voiceModelsApi, VoiceModel, SystemVoiceModel } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import { VoiceConvertUpload } from '@/components/voice-convert-upload';
import { TTSGenerator } from '@/components/tts-generator';
import { SongRemixUpload } from '@/components/song-remix-upload';
import { ModelUploadForm } from '@/components/model-upload-form';
import { Navbar } from '@/components/navbar';
import { Footer } from '@/components/footer';

type MainTab = 'community' | 'my-models';
type WorkflowTab = 'upload' | 'tts' | 'remix';

function ModelsPageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [selectedModel, setSelectedModel] = useState<SystemVoiceModel | null>(null);
  const [activeWorkflowTab, setActiveWorkflowTab] = useState<WorkflowTab>('upload');
  const [showUploadForm, setShowUploadForm] = useState(false);
  
  // Get tab from URL or default based on auth
  const user = useAuthStore((state) => state.user);
  const { canUploadModels, isAdmin } = useAuthStore();
  const urlTab = searchParams.get('tab') as MainTab | null;
  const [activeTab, setActiveTab] = useState<MainTab>(urlTab || 'community');

  // Sync tab with URL
  useEffect(() => {
    if (urlTab && (urlTab === 'community' || urlTab === 'my-models')) {
      setActiveTab(urlTab);
    }
  }, [urlTab]);

  // Update URL when tab changes
  const handleTabChange = (tab: MainTab) => {
    // If switching to my-models but not logged in, don't allow
    if (tab === 'my-models' && !user) {
      router.push('/login?redirect=/models?tab=my-models');
      return;
    }
    setActiveTab(tab);
    router.push(`/models?tab=${tab}`, { scroll: false });
  };

  // Community models query
  const { data: communityData, isLoading: communityLoading, error: communityError } = useQuery({
    queryKey: ['community-voice-models', search],
    queryFn: () => voiceModelsApi.list({ search, public_only: true }),
    enabled: activeTab === 'community',
  });

  // My models query (only when logged in) - returns both owned and shared models
  const { data: myModelsData, isLoading: myModelsLoading, error: myModelsError, refetch: refetchMyModels } = useQuery({
    queryKey: ['my-voice-models', user?.id],
    queryFn: () => voiceModelsApi.myModels({ per_page: 100 }),
    enabled: activeTab === 'my-models' && !!user,
  });

  const communityModels: SystemVoiceModel[] = communityData?.data || [];
  // Filter to only show owned models in "My Models" tab (is_owned is true or undefined for backwards compat)
  const allUserModels: (VoiceModel & { is_owned?: boolean })[] = myModelsData?.data || [];
  const myModels: VoiceModel[] = allUserModels.filter(m => m.is_owned !== false);

  // Edit modal state
  const [editingModel, setEditingModel] = useState<VoiceModel | null>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editVisibility, setEditVisibility] = useState('private');
  const [editTags, setEditTags] = useState('');
  const [saving, setSaving] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const handleEdit = (model: VoiceModel) => {
    setEditingModel(model);
    setEditName(model.name);
    setEditDescription(model.description || '');
    setEditVisibility(model.visibility);
    setEditTags((model.tags || []).join(', '));
  };

  const handleSaveEdit = async () => {
    if (!editingModel) return;
    setSaving(true);
    try {
      await voiceModelsApi.update(editingModel.id.toString(), {
        name: editName,
        description: editDescription || undefined,
        visibility: editVisibility as 'private' | 'unlisted' | 'public',
        tags: editTags ? editTags.split(',').map(t => t.trim()).filter(Boolean) : [],
      });
      refetchMyModels();
      setEditingModel(null);
    } catch (err) {
      console.error('Failed to update model:', err);
      alert('Failed to update model');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this model? This action cannot be undone.')) return;
    setDeletingId(id);
    try {
      await voiceModelsApi.delete(id.toString());
      refetchMyModels();
    } catch (err) {
      console.error('Failed to delete model:', err);
      alert('Failed to delete model');
    } finally {
      setDeletingId(null);
    }
  };

  const handleUploadSuccess = () => {
    setShowUploadForm(false);
    refetchMyModels();
  };

  const userCanUpload = user && (canUploadModels() || isAdmin());

  const getVisibilityIcon = (visibility: string) => {
    switch (visibility) {
      case 'public': return <Globe className="h-4 w-4" />;
      case 'unlisted': return <LinkIcon className="h-4 w-4" />;
      default: return <Lock className="h-4 w-4" />;
    }
  };

  const getVisibilityStyle = (visibility: string) => {
    switch (visibility) {
      case 'public': return 'bg-green-500/20 text-green-400';
      case 'unlisted': return 'bg-blue-500/20 text-blue-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <Navbar />

      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {/* Page Title */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Voice Models</h1>
          <p className="text-gray-400">
            Browse community models or manage your own uploads
          </p>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-4 mb-6 border-b border-gray-800">
          <button
            onClick={() => handleTabChange('community')}
            className={`pb-3 px-1 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'community'
                ? 'border-primary-500 text-primary-400'
                : 'border-transparent text-gray-400 hover:text-white'
            }`}
          >
            <div className="flex items-center gap-2">
              <Globe className="h-4 w-4" />
              Community Models
            </div>
          </button>
          {user ? (
            <button
              onClick={() => handleTabChange('my-models')}
              className={`pb-3 px-1 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'my-models'
                  ? 'border-primary-500 text-primary-400'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              <div className="flex items-center gap-2">
                <Upload className="h-4 w-4" />
                My Models
              </div>
            </button>
          ) : null}

          {/* Upload button for my-models tab */}
          {activeTab === 'my-models' && userCanUpload && (
            <button
              onClick={() => setShowUploadForm(!showUploadForm)}
              className="ml-auto flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              <Plus className="h-5 w-5" />
              Upload Model
            </button>
          )}
        </div>

        {/* Upload Form */}
        {showUploadForm && activeTab === 'my-models' && userCanUpload && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-white">Upload New Model</h2>
              <button onClick={() => setShowUploadForm(false)} className="text-gray-400 hover:text-white text-2xl">×</button>
            </div>
            <ModelUploadForm onSuccess={handleUploadSuccess} />
          </div>
        )}

        {/* Community Models Tab */}
        {activeTab === 'community' && (
          <>
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
                <span>{communityModels.length} models available</span>
              </div>
            </div>

            {/* Models Grid */}
            {communityLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {[...Array(6)].map((_, i) => <ModelCardSkeleton key={i} />)}
              </div>
            ) : communityError ? (
              <div className="text-center py-12">
                <p className="text-red-400">Failed to load models. Make sure the API server is running.</p>
              </div>
            ) : communityModels.length === 0 ? (
              <div className="text-center py-12">
                <Mic2 className="h-12 w-12 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-400">No models found</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {communityModels.map((model) => (
                  <ModelCard 
                    key={model.slug} 
                    model={model} 
                    isSelected={selectedModel?.slug === model.slug}
                    onSelect={() => setSelectedModel(model)}
                  />
                ))}
              </div>
            )}
          </>
        )}

        {/* My Models Tab */}
        {activeTab === 'my-models' && user && (
          <>
            {myModelsLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
              </div>
            ) : myModels.length === 0 ? (
              <div className="text-center py-12 bg-gray-900/50 border border-gray-800 rounded-lg">
                <Upload className="h-12 w-12 text-gray-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-white mb-2">No models yet</h3>
                {userCanUpload ? (
                  <>
                    <p className="text-gray-400 mb-6">
                      You don&apos;t have any voice models yet. Get started by uploading or training one!
                    </p>
                    <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                      <button
                        onClick={() => setShowUploadForm(true)}
                        className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                      >
                        <Plus className="h-5 w-5" />
                        Upload a Model
                      </button>
                      <span className="text-gray-500">or</span>
                      <Link
                        href="/dashboard/train"
                        className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
                      >
                        <Mic2 className="h-5 w-5" />
                        Train Your Own
                      </Link>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="text-gray-400 mb-4">You don&apos;t have permission to upload voice models.</p>
                    <Link href="/dashboard/settings" className="inline-flex items-center gap-2 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors">
                      <Settings className="h-5 w-5" />
                      Request Access
                    </Link>
                  </>
                )}
              </div>
            ) : (
              <div className="grid gap-4">
                {myModels.map((model) => (
                  <div key={model.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors">
                    <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex flex-wrap items-center gap-2 mb-2">
                          <h3 className="text-lg font-medium text-white truncate">{model.name}</h3>
                          <span className={`flex items-center gap-1 text-xs px-2 py-1 rounded ${getVisibilityStyle(model.visibility)}`}>
                            {getVisibilityIcon(model.visibility)}
                            {model.visibility.charAt(0).toUpperCase() + model.visibility.slice(1)}
                          </span>
                          {model.status === 'ready' && (
                            <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded">Ready</span>
                          )}
                        </div>
                        {model.description && (
                          <p className="text-gray-400 text-sm mb-2 line-clamp-2">{model.description}</p>
                        )}
                        <div className="flex flex-wrap items-center gap-4 text-xs text-gray-500">
                          {model.tags && model.tags.length > 0 && (
                            <div className="flex gap-1">
                              {model.tags.slice(0, 3).map((tag, i) => (
                                <span key={i} className="px-2 py-0.5 bg-gray-800 rounded">{tag}</span>
                              ))}
                            </div>
                          )}
                          <span>Created {new Date(model.created_at).toLocaleDateString()}</span>
                          {model.usage_count > 0 && <span>{model.usage_count} uses</span>}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <button onClick={() => handleEdit(model)} className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg" title="Edit">
                          <Pencil className="h-5 w-5" />
                        </button>
                        <button
                          onClick={() => handleDelete(model.id)}
                          disabled={deletingId === model.id}
                          className="p-2 text-gray-400 hover:text-red-400 hover:bg-gray-800 rounded-lg disabled:opacity-50"
                          title="Delete"
                        >
                          {deletingId === model.id ? (
                            <Loader2 className="h-5 w-5 animate-spin" />
                          ) : (
                            <Trash2 className="h-5 w-5" />
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* Edit Modal */}
        {editingModel && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 max-w-md w-full">
              <h2 className="text-lg font-semibold text-white mb-4">Edit Model</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Name</label>
                  <input type="text" value={editName} onChange={(e) => setEditName(e.target.value)} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Description</label>
                  <textarea value={editDescription} onChange={(e) => setEditDescription(e.target.value)} rows={3} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white resize-none" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Visibility</label>
                  <select value={editVisibility} onChange={(e) => setEditVisibility(e.target.value)} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white">
                    <option value="private">Private</option>
                    <option value="unlisted">Unlisted</option>
                    <option value="public">Public</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Tags</label>
                  <input type="text" value={editTags} onChange={(e) => setEditTags(e.target.value)} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white" placeholder="male, deep, english" />
                </div>
              </div>
              <div className="flex gap-3 mt-6">
                <button onClick={() => setEditingModel(null)} className="flex-1 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600">Cancel</button>
                <button onClick={handleSaveEdit} disabled={saving || !editName.trim()} className="flex-1 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50">
                  {saving ? 'Saving...' : 'Save'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Selected Model Workflow Panel */}
        {selectedModel && (
          <div className="fixed inset-0 bg-gray-950/95 backdrop-blur-sm z-50 overflow-auto">
            <div className="max-w-4xl mx-auto px-4 py-8">
              <div className="flex items-center gap-4 mb-6">
                <button onClick={() => setSelectedModel(null)} className="p-2 rounded-lg hover:bg-gray-800">
                  <ChevronLeft className="h-6 w-6" />
                </button>
                <div className="flex items-center gap-3 flex-1">
                  <div className="w-12 h-12 bg-gradient-to-br from-primary-600 to-accent-600 rounded-lg flex items-center justify-center">
                    <Mic2 className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">{selectedModel.name}</h2>
                    <p className="text-sm text-gray-400">{selectedModel.model_file} • {selectedModel.size}</p>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                <div className="flex border-b border-gray-800">
                  <button
                    onClick={() => setActiveWorkflowTab('upload')}
                    className={`flex-1 px-6 py-4 text-sm font-medium transition-colors flex items-center justify-center gap-2 ${activeWorkflowTab === 'upload' ? 'bg-gray-800 text-white border-b-2 border-primary-500' : 'text-gray-400 hover:text-white'}`}
                  >
                    <Upload className="h-4 w-4" />
                    Voice Convert
                  </button>
                  <button
                    onClick={() => setActiveWorkflowTab('tts')}
                    className={`flex-1 px-6 py-4 text-sm font-medium transition-colors flex items-center justify-center gap-2 ${activeWorkflowTab === 'tts' ? 'bg-gray-800 text-white border-b-2 border-primary-500' : 'text-gray-400 hover:text-white'}`}
                  >
                    <MessageSquare className="h-4 w-4" />
                    Text to Speech
                  </button>
                  <button
                    onClick={() => setActiveWorkflowTab('remix')}
                    className={`flex-1 px-6 py-4 text-sm font-medium transition-colors flex items-center justify-center gap-2 ${activeWorkflowTab === 'remix' ? 'bg-gray-800 text-white border-b-2 border-primary-500' : 'text-gray-400 hover:text-white'}`}
                  >
                    <Music className="h-4 w-4" />
                    Song Remix
                  </button>
                </div>
                <div className="p-6">
                  {activeWorkflowTab === 'upload' && (
                    <VoiceConvertUpload selectedModelId={selectedModel.id} modelName={selectedModel.name} />
                  )}
                  {activeWorkflowTab === 'tts' && (
                    <TTSGenerator preSelectedModelId={selectedModel.id} hideModelSelector={true} />
                  )}
                  {activeWorkflowTab === 'remix' && (
                    <SongRemixUpload 
                      selectedModelId={selectedModel.id} 
                      modelName={selectedModel.name}
                      availableModels={[
                        ...communityModels.map(m => ({ id: m.id, name: m.name })),
                        ...myModels.map(m => ({ id: m.id, name: m.name })),
                      ].filter((m, i, arr) => arr.findIndex(x => x.id === m.id) === i)}
                    />
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}

function ModelCard({ model, isSelected, onSelect }: { model: SystemVoiceModel; isSelected: boolean; onSelect: () => void }) {
  return (
    <div
      onClick={onSelect}
      className={`glass rounded-xl overflow-hidden cursor-pointer transition-all group ${isSelected ? 'ring-2 ring-primary-500 bg-primary-500/10' : 'hover:bg-white/5'}`}
    >
      <div className="aspect-video bg-gradient-to-br from-primary-900/50 to-accent-900/50 flex items-center justify-center relative overflow-hidden">
        {model.image_url ? (
          <img 
            src={model.image_url} 
            alt={model.name} 
            className="w-full h-full object-cover"
          />
        ) : (
          <Mic2 className="h-16 w-16 text-gray-600" />
        )}
        {isSelected && (
          <div className="absolute top-3 right-3 w-8 h-8 bg-primary-500 rounded-full flex items-center justify-center">
            <Check className="h-5 w-5 text-white" />
          </div>
        )}
        <div className="absolute top-3 left-3 bg-gray-900/80 text-gray-400 text-xs px-2 py-1 rounded flex items-center gap-1">
          {model.storage_type === 's3' ? <><Cloud className="h-3 w-3" /><span>Cloud</span></> : <><Server className="h-3 w-3" /><span>Local</span></>}
        </div>
      </div>
      <div className="p-4">
        <h3 className="font-semibold text-lg mb-1 group-hover:text-primary-400">{model.name}</h3>
        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-2 text-gray-400">
            <FileAudio className="h-4 w-4 flex-shrink-0" />
            <span className="truncate">{model.model_file || 'Unknown'}</span>
          </div>
          <div className="flex items-center gap-2">
            {model.has_index ? (
              <><Check className="h-4 w-4 text-green-500" /><span className="text-green-400">Index: {model.index_file}</span></>
            ) : (
              <><X className="h-4 w-4 text-yellow-500" /><span className="text-yellow-400">No index</span></>
            )}
          </div>
          <div className="flex items-center justify-between text-gray-500 pt-2 border-t border-gray-800">
            <span>{model.size}</span>
            <div className="flex items-center gap-2">
              {model.usage_count > 0 && <span className="text-xs">{model.usage_count} uses</span>}
              <span className="bg-gray-800 px-2 py-0.5 rounded text-xs uppercase">{model.engine}</span>
            </div>
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

export default function ModelsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    }>
      <ModelsPageContent />
    </Suspense>
  );
}
