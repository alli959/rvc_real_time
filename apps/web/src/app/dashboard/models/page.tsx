'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { DashboardLayout } from '@/components/dashboard-layout';
import { ModelUploadForm } from '@/components/model-upload-form';
import { voiceModelsApi, VoiceModel } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import {
  Plus as PlusIcon,
  Trash2 as TrashIcon,
  Pencil as PencilIcon,
  FileUp as DocumentArrowUpIcon,
  Eye as EyeIcon,
  EyeOff as EyeSlashIcon,
  Link as LinkIcon,
  RefreshCw as ArrowPathIcon,
  Mic2,
  Settings,
  Lock,
  Users,
  Globe,
} from 'lucide-react';

export default function MyModelsPage() {
  const router = useRouter();
  const { canUploadModels, user, isAdmin } = useAuthStore();
  const [models, setModels] = useState<VoiceModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [editingModel, setEditingModel] = useState<VoiceModel | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  // Edit form state
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editVisibility, setEditVisibility] = useState('private');
  const [editTags, setEditTags] = useState('');
  const [saving, setSaving] = useState(false);

  const loadModels = async () => {
    try {
      const data = await voiceModelsApi.myModels();
      setModels(data.data || []);
    } catch (err) {
      console.error('Failed to load models:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
      return;
    }

    setDeletingId(id);
    try {
      await voiceModelsApi.delete(id.toString());
      setModels(models.filter(m => m.id !== id));
    } catch (err) {
      console.error('Failed to delete model:', err);
      alert('Failed to delete model');
    } finally {
      setDeletingId(null);
    }
  };

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
      const updated = await voiceModelsApi.update(editingModel.id.toString(), {
        name: editName,
        description: editDescription || undefined,
        visibility: editVisibility as 'private' | 'unlisted' | 'public',
        tags: editTags ? editTags.split(',').map(t => t.trim()).filter(Boolean) : [],
      });
      setModels(models.map(m => m.id === editingModel.id ? updated.model || updated : m));
      setEditingModel(null);
    } catch (err) {
      console.error('Failed to update model:', err);
      alert('Failed to update model');
    } finally {
      setSaving(false);
    }
  };

  const handleUploadSuccess = () => {
    setShowUploadForm(false);
    loadModels();
  };

  const handleModelClick = (model: VoiceModel) => {
    router.push(`/dashboard/convert?model=${model.slug}`);
  };

  const getVisibilityIcon = (visibility: string) => {
    switch (visibility) {
      case 'public':
        return <Globe className="h-4 w-4" />;
      case 'unlisted':
        return <LinkIcon className="h-4 w-4" />;
      default:
        return <Lock className="h-4 w-4" />;
    }
  };

  const getVisibilityLabel = (visibility: string) => {
    switch (visibility) {
      case 'public':
        return 'Public';
      case 'unlisted':
        return 'Unlisted';
      default:
        return 'Private';
    }
  };

  const getVisibilityStyle = (visibility: string) => {
    switch (visibility) {
      case 'public':
        return 'bg-green-500/20 text-green-400';
      case 'unlisted':
        return 'bg-blue-500/20 text-blue-400';
      default:
        return 'bg-gray-500/20 text-gray-400';
    }
  };

  // Check if user can upload
  const userCanUpload = canUploadModels() || isAdmin();

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">My Voice Models</h1>
            <p className="text-gray-400 mt-1">Manage your uploaded voice models</p>
          </div>
          {userCanUpload && (
            <button
              onClick={() => setShowUploadForm(!showUploadForm)}
              className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              <PlusIcon className="h-5 w-5" />
              Upload Model
            </button>
          )}
        </div>

        {/* Upload Form */}
        {showUploadForm && userCanUpload && (
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-white">Upload New Model</h2>
              <button
                onClick={() => setShowUploadForm(false)}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>
            <ModelUploadForm onSuccess={handleUploadSuccess} />
          </div>
        )}

        {/* Edit Modal */}
        {editingModel && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 max-w-md w-full">
              <h2 className="text-lg font-semibold text-white mb-4">Edit Model</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Name</label>
                  <input
                    type="text"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Description</label>
                  <textarea
                    value={editDescription}
                    onChange={(e) => setEditDescription(e.target.value)}
                    rows={3}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Visibility</label>
                  <select
                    value={editVisibility}
                    onChange={(e) => setEditVisibility(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  >
                    <option value="private">Private - Only you can see and use</option>
                    <option value="unlisted">Unlisted - Anyone with the link</option>
                    <option value="public">Public - Visible to everyone</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Tags (comma-separated)</label>
                  <input
                    type="text"
                    value={editTags}
                    onChange={(e) => setEditTags(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    placeholder="male, deep, english"
                  />
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setEditingModel(null)}
                  className="flex-1 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveEdit}
                  disabled={saving || !editName.trim()}
                  className="flex-1 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {saving ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Models List */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-12 bg-gray-900/50 border border-gray-800 rounded-lg">
            <DocumentArrowUpIcon className="h-12 w-12 text-gray-600 mx-auto mb-4" />
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
                    <PlusIcon className="h-5 w-5" />
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
                <p className="text-gray-400 mb-4">
                  You don&apos;t have permission to upload or train voice models.
                </p>
                <p className="text-gray-500 text-sm mb-6">
                  You can request additional roles through the settings page.
                </p>
                <Link
                  href="/dashboard/settings"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
                >
                  <Settings className="h-5 w-5" />
                  Go to Settings
                </Link>
              </>
            )}
          </div>
        ) : (
          <div className="grid gap-4">
            {models.map((model) => (
              <div
                key={model.id}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
              >
                <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                  <div 
                    className="flex-1 min-w-0 cursor-pointer"
                    onClick={() => handleModelClick(model)}
                  >
                    <div className="flex flex-wrap items-center gap-2 mb-2">
                      <h3 className="text-lg font-medium text-white hover:text-primary-400 transition-colors truncate">
                        {model.name}
                      </h3>
                      <span className={`flex items-center gap-1 text-xs px-2 py-1 rounded ${getVisibilityStyle(model.visibility)}`}>
                        {getVisibilityIcon(model.visibility)}
                        {getVisibilityLabel(model.visibility)}
                      </span>
                      {model.status === 'processing' && (
                        <span className="flex items-center gap-1 text-xs px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded">
                          <ArrowPathIcon className="h-3 w-3 animate-spin" />
                          Processing
                        </span>
                      )}
                      {model.status === 'ready' && (
                        <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded">
                          Ready
                        </span>
                      )}
                      {model.status === 'failed' && (
                        <span className="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded">
                          Failed
                        </span>
                      )}
                    </div>

                    {model.description && (
                      <p className="text-gray-400 text-sm mb-2 line-clamp-2">{model.description}</p>
                    )}

                    <div className="flex flex-wrap items-center gap-4 text-xs text-gray-500">
                      {model.tags && model.tags.length > 0 && (
                        <div className="flex gap-1">
                          {model.tags.slice(0, 3).map((tag, i) => (
                            <span key={i} className="px-2 py-0.5 bg-gray-800 rounded">
                              {tag}
                            </span>
                          ))}
                          {model.tags.length > 3 && (
                            <span className="px-2 py-0.5 bg-gray-800 rounded">
                              +{model.tags.length - 3}
                            </span>
                          )}
                        </div>
                      )}
                      <span>Created {new Date(model.created_at).toLocaleDateString()}</span>
                      {model.usage_count > 0 && (
                        <span>{model.usage_count.toLocaleString()} uses</span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 sm:ml-4">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleEdit(model);
                      }}
                      className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
                      title="Edit"
                    >
                      <PencilIcon className="h-5 w-5" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(model.id);
                      }}
                      disabled={deletingId === model.id}
                      className="p-2 text-gray-400 hover:text-red-400 hover:bg-gray-800 rounded-lg transition-colors disabled:opacity-50"
                      title="Delete"
                    >
                      {deletingId === model.id ? (
                        <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-500 border-t-transparent" />
                      ) : (
                        <TrashIcon className="h-5 w-5" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
