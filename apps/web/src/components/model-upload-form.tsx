'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { voiceModelsApi } from '@/lib/api';
import { 
  CloudUpload as CloudArrowUpIcon, 
  FileText as DocumentIcon, 
  X as XMarkIcon,
  CheckCircle as CheckCircleIcon,
  AlertCircle as ExclamationCircleIcon,
} from 'lucide-react';

interface UploadedFile {
  file: File;
  type: 'model' | 'index' | 'config';
}

interface ModelUploadFormProps {
  onSuccess?: (model: any) => void;
  onCancel?: () => void;
}

export function ModelUploadForm({ onSuccess, onCancel }: ModelUploadFormProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [visibility, setVisibility] = useState<'private' | 'unlisted' | 'public'>('private');
  const [tags, setTags] = useState('');
  const [hasConsent, setHasConsent] = useState(false);
  const [consentNotes, setConsentNotes] = useState('');
  
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = [];
    
    acceptedFiles.forEach(file => {
      const ext = file.name.split('.').pop()?.toLowerCase();
      
      if (ext === 'pth') {
        // Remove existing model file
        setFiles(prev => prev.filter(f => f.type !== 'model'));
        newFiles.push({ file, type: 'model' });
        // Auto-fill name from filename if empty
        if (!name) {
          setName(file.name.replace('.pth', ''));
        }
      } else if (ext === 'index') {
        setFiles(prev => prev.filter(f => f.type !== 'index'));
        newFiles.push({ file, type: 'index' });
      } else if (ext === 'json') {
        setFiles(prev => prev.filter(f => f.type !== 'config'));
        newFiles.push({ file, type: 'config' });
      }
    });
    
    setFiles(prev => [...prev, ...newFiles]);
    setError(null);
  }, [name]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.pth', '.index'],
      'application/json': ['.json'],
    },
    maxSize: 1024 * 1024 * 1024, // 1GB
  });

  const removeFile = (type: string) => {
    setFiles(prev => prev.filter(f => f.type !== type));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const modelFile = files.find(f => f.type === 'model');
    if (!modelFile) {
      setError('Model file (.pth) is required');
      return;
    }
    
    if (!name.trim()) {
      setError('Model name is required');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('name', name.trim());
      formData.append('description', description.trim());
      formData.append('visibility', visibility);
      formData.append('has_consent', hasConsent ? '1' : '0');
      
      if (consentNotes.trim()) {
        formData.append('consent_notes', consentNotes.trim());
      }
      
      if (tags.trim()) {
        const tagArray = tags.split(',').map(t => t.trim()).filter(Boolean);
        tagArray.forEach(tag => formData.append('tags[]', tag));
      }
      
      formData.append('model_file', modelFile.file);
      
      const indexFile = files.find(f => f.type === 'index');
      if (indexFile) {
        formData.append('index_file', indexFile.file);
      }
      
      const configFile = files.find(f => f.type === 'config');
      if (configFile) {
        formData.append('config_file', configFile.file);
      }

      const result = await voiceModelsApi.upload(formData, setUploadProgress);
      
      setSuccess(true);
      if (onSuccess) {
        onSuccess(result.model);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || err.response?.data?.error || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  if (success) {
    return (
      <div className="text-center py-12">
        <CheckCircleIcon className="mx-auto h-16 w-16 text-green-500" />
        <h3 className="mt-4 text-lg font-semibold text-white">Model Uploaded Successfully!</h3>
        <p className="mt-2 text-gray-400">Your voice model is now ready to use.</p>
        <button
          onClick={() => {
            setSuccess(false);
            setFiles([]);
            setName('');
            setDescription('');
            setTags('');
            setHasConsent(false);
            setConsentNotes('');
          }}
          className="mt-6 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          Upload Another Model
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-primary-500 bg-primary-500/10'
            : 'border-gray-600 hover:border-gray-500'
        }`}
      >
        <input {...getInputProps()} />
        <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-4 text-gray-300">
          {isDragActive
            ? 'Drop files here...'
            : 'Drag & drop model files, or click to browse'}
        </p>
        <p className="mt-2 text-sm text-gray-500">
          Supported: .pth (required), .index (optional), config.json (optional)
        </p>
        <p className="mt-1 text-xs text-gray-600">Max file size: 500MB</p>
      </div>

      {/* Selected Files */}
      {files.length > 0 && (
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">Selected Files</label>
          <div className="space-y-2">
            {files.map((f) => (
              <div
                key={f.type}
                className="flex items-center justify-between bg-gray-800 rounded-lg px-4 py-3"
              >
                <div className="flex items-center gap-3">
                  <DocumentIcon className="h-5 w-5 text-gray-400" />
                  <div>
                    <p className="text-sm text-white">{f.file.name}</p>
                    <p className="text-xs text-gray-500">
                      {formatFileSize(f.file.size)} • {f.type.toUpperCase()}
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile(f.type)}
                  className="text-gray-400 hover:text-red-400"
                >
                  <XMarkIcon className="h-5 w-5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Details */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Model Name <span className="text-red-400">*</span>
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="My Voice Model"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={3}
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="Describe your voice model..."
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Visibility</label>
          <select
            value={visibility}
            onChange={(e) => setVisibility(e.target.value as any)}
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="private">Private - Only you can see and use</option>
            <option value="unlisted">Unlisted - Anyone with link can use</option>
            <option value="public">Public - Visible to everyone (requires approval)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Tags <span className="text-gray-500">(comma separated)</span>
          </label>
          <input
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="male, english, natural"
          />
        </div>

        {/* Consent */}
        <div className="bg-gray-800/50 rounded-lg p-4 space-y-3">
          <div className="flex items-start gap-3">
            <input
              type="checkbox"
              id="consent"
              checked={hasConsent}
              onChange={(e) => setHasConsent(e.target.checked)}
              className="mt-1 h-4 w-4 rounded border-gray-600 bg-gray-700 text-primary-600 focus:ring-primary-500"
            />
            <label htmlFor="consent" className="text-sm text-gray-300">
              I have obtained proper consent from the voice owner to create and use this voice model
            </label>
          </div>
          
          {hasConsent && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Consent Notes
              </label>
              <textarea
                value={consentNotes}
                onChange={(e) => setConsentNotes(e.target.value)}
                rows={2}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
                placeholder="Describe how consent was obtained..."
              />
            </div>
          )}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-center gap-2 text-red-400 bg-red-400/10 px-4 py-3 rounded-lg">
          <ExclamationCircleIcon className="h-5 w-5 flex-shrink-0" />
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Progress Bar */}
      {uploading && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm text-gray-400">
            <span>Uploading...</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-500 transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-end gap-3">
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            disabled={uploading}
            className="px-4 py-2 text-gray-300 hover:text-white disabled:opacity-50"
          >
            Cancel
          </button>
        )}
        <button
          type="submit"
          disabled={uploading || files.length === 0}
          className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {uploading ? (
            <>
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Uploading...
            </>
          ) : (
            <>
              <CloudArrowUpIcon className="h-5 w-5" />
              Upload Model
            </>
          )}
        </button>
      </div>
    </form>
  );
}
