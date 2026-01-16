'use client';

import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { DashboardLayout } from '@/components/dashboard-layout';
import { jobsApi, trainerApi } from '@/lib/api';
import {
  ListMusic,
  Play,
  Pause,
  Square,
  Download,
  Trash2,
  RefreshCw,
  Clock,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  FileAudio,
  ChevronRight,
  Save,
  StopCircle,
} from 'lucide-react';

type JobFilter = 'all' | 'pending' | 'queued' | 'processing' | 'completed' | 'failed';

// Training step definitions (matching backend)
const TRAINING_STEPS = [
  { key: 'preprocessing', label: 'Preprocessing Audio', weight: 0.10 },
  { key: 'f0_extraction', label: 'Extracting Pitch', weight: 0.15 },
  { key: 'feature_extraction', label: 'Extracting Features', weight: 0.15 },
  { key: 'training', label: 'Training Model', weight: 0.50 },
  { key: 'index_building', label: 'Building Index', weight: 0.05 },
  { key: 'packaging', label: 'Packaging Model', weight: 0.05 },
];

const getJobTypeLabel = (type: string | undefined): string => {
  if (!type) return 'Job';
  const labels: Record<string, string> = {
    'inference': 'Voice Conversion',
    'tts': 'Text to Speech',
    'audio_convert': 'Audio Conversion',
    'audio_split': 'Vocal Separation',
    'audio_swap': 'Voice Swap',
    'training': 'Model Training',
    'training_rvc': 'Model Training',
    'preprocessing': 'Preprocessing',
  };
  return labels[type] || type.charAt(0).toUpperCase() + type.slice(1).replace(/_/g, ' ');
};

const isTrainingJob = (type: string | undefined): boolean => {
  return type === 'training' || type === 'training_rvc' || type === 'training_fine_tune';
};

export default function JobsPage() {
  const [filter, setFilter] = useState<JobFilter>('all');
  const [page, setPage] = useState(1);
  const [checkpointPending, setCheckpointPending] = useState<Record<string, boolean>>({});
  const queryClient = useQueryClient();

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['jobs', filter, page],
    queryFn: () => jobsApi.list({ 
      page, 
      status: filter === 'all' ? undefined : filter 
    }),
    refetchInterval: 5000, // Poll every 5 seconds for updates
  });

  const jobs = data?.data || [];
  const meta = data?.meta;

  const filters: { id: JobFilter; label: string; icon: any }[] = [
    { id: 'all', label: 'All Jobs', icon: ListMusic },
    { id: 'processing', label: 'Processing', icon: Loader2 },
    { id: 'queued', label: 'Queued', icon: Clock },
    { id: 'completed', label: 'Completed', icon: CheckCircle2 },
    { id: 'failed', label: 'Failed', icon: XCircle },
  ];

  const handleCancelJob = async (jobId: string) => {
    try {
      await jobsApi.cancel(jobId);
      refetch();
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  const handleCheckpointAndStop = async (jobId: string) => {
    try {
      setCheckpointPending(prev => ({ ...prev, [jobId]: true }));
      await trainerApi.requestCheckpoint(jobId, true);
      // The job will save a checkpoint and stop
    } catch (err) {
      console.error('Failed to request checkpoint:', err);
      setCheckpointPending(prev => ({ ...prev, [jobId]: false }));
    }
  };

  const handleDownload = async (jobId: string) => {
    try {
      const { download_url } = await jobsApi.getOutput(jobId);
      window.open(download_url, '_blank');
    } catch (err) {
      console.error('Failed to get download URL:', err);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-5 w-5 text-gray-400" />;
      case 'queued':
        return <Clock className="h-5 w-5 text-blue-400" />;
      case 'processing':
        return <Loader2 className="h-5 w-5 text-primary-400 animate-spin" />;
      case 'completed':
        return <CheckCircle2 className="h-5 w-5 text-green-400" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-400" />;
      case 'cancelled':
        return <Square className="h-5 w-5 text-gray-400" />;
      default:
        return <AlertCircle className="h-5 w-5 text-yellow-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      pending: 'bg-gray-500/20 text-gray-400',
      queued: 'bg-blue-500/20 text-blue-400',
      processing: 'bg-primary-500/20 text-primary-400',
      completed: 'bg-green-500/20 text-green-400',
      failed: 'bg-red-500/20 text-red-400',
      cancelled: 'bg-gray-500/20 text-gray-400',
    };

    return (
      <span className={`px-2 py-1 rounded text-xs font-medium ${styles[status] || styles.pending}`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    );
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <ListMusic className="h-7 w-7" />
              My Jobs
            </h1>
            <p className="text-gray-400 mt-1">View and manage your voice conversion jobs</p>
          </div>
          <button
            onClick={() => refetch()}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>

        {/* Filters */}
        <div className="flex gap-2 overflow-x-auto pb-2">
          {filters.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => {
                setFilter(id);
                setPage(1);
              }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${
                filter === id
                  ? 'bg-primary-600/20 text-primary-400 border border-primary-500/50'
                  : 'bg-gray-800 text-gray-400 hover:text-white border border-transparent'
              }`}
            >
              <Icon className={`h-4 w-4 ${id === 'processing' && filter === id ? 'animate-spin' : ''}`} />
              {label}
            </button>
          ))}
        </div>

        {/* Jobs List */}
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="flex flex-col items-center gap-4">
              <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
              <p className="text-gray-400">Loading jobs...</p>
            </div>
          </div>
        ) : error ? (
          <div className="text-center py-12 bg-gray-900/50 border border-gray-800 rounded-lg">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">Failed to load jobs</h3>
            <p className="text-gray-400 mb-4">There was an error loading your jobs.</p>
            <button
              onClick={() => refetch()}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-12 bg-gray-900/50 border border-gray-800 rounded-lg">
            <FileAudio className="h-12 w-12 text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">No jobs found</h3>
            <p className="text-gray-400">
              {filter === 'all'
                ? "You haven't created any voice conversion jobs yet."
                : `No ${filter} jobs found.`}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {jobs.map((job: any) => (
              <div
                key={job.id}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-4 flex-1 min-w-0">
                    <div className="mt-1">{getStatusIcon(job.status)}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className="font-medium text-white truncate">
                          {job.voice_model?.name || 'Unknown Model'}
                        </h3>
                        {getStatusBadge(job.status)}
                      </div>
                      
                      <p className="text-sm text-gray-500 mb-2">
                        {getJobTypeLabel(job.type)} • 
                        Created {new Date(job.created_at).toLocaleString()}
                      </p>

                      {/* Training-specific progress with steps */}
                      {job.status === 'processing' && isTrainingJob(job.type) && (
                        <div className="mt-3 space-y-2">
                          {/* Step indicators */}
                          <div className="flex items-center gap-1 text-xs">
                            {TRAINING_STEPS.map((step, idx) => {
                              const currentStep = job.step || job.current_step || '';
                              const stepIdx = TRAINING_STEPS.findIndex(s => s.key === currentStep);
                              const isComplete = idx < stepIdx;
                              const isCurrent = step.key === currentStep;
                              
                              return (
                                <div key={step.key} className="flex items-center gap-1">
                                  <div
                                    className={`h-2 flex-1 rounded ${
                                      isComplete
                                        ? 'bg-green-500'
                                        : isCurrent
                                        ? 'bg-primary-500 animate-pulse'
                                        : 'bg-gray-700'
                                    }`}
                                    style={{ minWidth: `${step.weight * 100}px` }}
                                    title={step.label}
                                  />
                                </div>
                              );
                            })}
                          </div>
                          
                          {/* Current step label */}
                          <div className="flex justify-between text-xs text-gray-400">
                            <span>
                              {TRAINING_STEPS.find(s => s.key === (job.step || job.current_step))?.label || 'Processing...'}
                              {job.current_epoch && job.total_epochs && (
                                <span className="ml-2 text-primary-400">
                                  Epoch {job.current_epoch}/{job.total_epochs}
                                </span>
                              )}
                            </span>
                            <span>{job.progress}%</span>
                          </div>
                          
                          {/* Overall progress bar */}
                          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary-500 rounded-full transition-all duration-500"
                              style={{ width: `${job.progress}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Regular progress bar for non-training jobs */}
                      {job.status === 'processing' && !isTrainingJob(job.type) && job.progress !== undefined && (
                        <div className="mt-2">
                          <div className="flex justify-between text-xs text-gray-400 mb-1">
                            <span>Progress</span>
                            <span>{job.progress}%</span>
                          </div>
                          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary-500 rounded-full transition-all duration-500"
                              style={{ width: `${job.progress}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Error message */}
                      {job.status === 'failed' && job.error_message && (
                        <div className="mt-2 text-sm text-red-400 bg-red-500/10 px-3 py-2 rounded">
                          {job.error_message}
                        </div>
                      )}

                      {/* Completed info */}
                      {job.status === 'completed' && job.completed_at && (
                        <p className="text-xs text-gray-500 mt-1">
                          Completed {new Date(job.completed_at).toLocaleString()}
                        </p>
                      )}

                      {/* Queue position */}
                      {job.status === 'queued' && job.queue_position && (
                        <p className="text-sm text-blue-400 mt-1">
                          Position in queue: #{job.queue_position}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    {job.status === 'completed' && (
                      <button
                        onClick={() => handleDownload(job.id)}
                        className="p-2 text-gray-400 hover:text-green-400 hover:bg-gray-800 rounded-lg transition-colors"
                        title="Download"
                      >
                        <Download className="h-5 w-5" />
                      </button>
                    )}
                    {/* Checkpoint & Stop button for training jobs */}
                    {job.status === 'processing' && isTrainingJob(job.type) && (
                      <button
                        onClick={() => handleCheckpointAndStop(job.id)}
                        disabled={checkpointPending[job.id]}
                        className="flex items-center gap-1 px-3 py-1.5 text-xs text-amber-400 hover:text-amber-300 bg-amber-500/10 hover:bg-amber-500/20 rounded-lg transition-colors disabled:opacity-50"
                        title="Save checkpoint and stop training"
                      >
                        {checkpointPending[job.id] ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            <span>Saving...</span>
                          </>
                        ) : (
                          <>
                            <Save className="h-4 w-4" />
                            <span>Save & Stop</span>
                          </>
                        )}
                      </button>
                    )}
                    {(job.status === 'pending' || job.status === 'queued' || job.status === 'processing') && (
                      <button
                        onClick={() => handleCancelJob(job.id)}
                        className="p-2 text-gray-400 hover:text-red-400 hover:bg-gray-800 rounded-lg transition-colors"
                        title="Cancel"
                      >
                        <Square className="h-5 w-5" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {meta && meta.last_page > 1 && (
          <div className="flex items-center justify-center gap-2 pt-4">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 bg-gray-800 text-gray-400 rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <span className="text-gray-400 px-4">
              Page {meta.current_page} of {meta.last_page}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(meta.last_page, p + 1))}
              disabled={page === meta.last_page}
              className="px-3 py-1 bg-gray-800 text-gray-400 rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
