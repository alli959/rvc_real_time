'use client';

import { useState } from 'react';
import { useAudioJobs, ActiveJob } from '@/contexts/audio-job-context';
import { jobsApi } from '@/lib/api';
import {
  X, Loader2, CheckCircle2, AlertCircle, Download, Play, Pause,
  Trash2, Bookmark, BookmarkCheck, XCircle,
} from 'lucide-react';

function JobTypeIcon({ type }: { type: ActiveJob['type'] }) {
  const labels: Record<string, string> = {
    audio_swap: '🎤',
    audio_split: '✂️',
    audio_convert: '🔄',
    tts: '💬',
  };
  return <span className="text-sm">{labels[type] || '🎵'}</span>;
}

function JobCard({ job }: { job: ActiveJob }) {
  const { cancelJob, saveJob, unsaveJob, dismissJob } = useAudioJobs();
  const [audioPlaying, setAudioPlaying] = useState(false);

  const isActive = job.status === 'queued' || job.status === 'processing';
  const isComplete = job.status === 'completed';
  const isFailed = job.status === 'failed';

  return (
    <div className="p-3 bg-gray-800 rounded-lg border border-gray-700 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <JobTypeIcon type={job.type} />
          <span className="text-sm font-medium text-gray-200 truncate max-w-[120px]">
            {job.modelName}
          </span>
        </div>
        <button onClick={() => dismissJob(job.id)} className="text-gray-500 hover:text-gray-300">
          <X size={14} />
        </button>
      </div>

      {isActive && (
        <>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-purple-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${job.progress}%` }}
            />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">{job.progressMessage}</span>
            <button onClick={() => cancelJob(job.id)} className="text-xs text-red-400 hover:text-red-300">
              Cancel
            </button>
          </div>
        </>
      )}

      {isComplete && job.outputUrl && (
        <div className="flex items-center gap-2">
          <a
            href={`${job.outputUrl}${job.outputUrl.includes('?') ? '&' : '?'}download=1`}
            className="flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300"
          >
            <Download size={12} /> Download
          </a>
          <button
            onClick={() => job.saved ? unsaveJob(job.id) : saveJob(job.id)}
            className={`flex items-center gap-1 text-xs ${job.saved ? 'text-yellow-400' : 'text-gray-400 hover:text-yellow-400'}`}
          >
            {job.saved ? <BookmarkCheck size={12} /> : <Bookmark size={12} />}
            {job.saved ? 'Saved' : 'Save'}
          </button>
        </div>
      )}

      {isFailed && (
        <span className="text-xs text-red-400">{job.errorMessage || 'Processing failed'}</span>
      )}
    </div>
  );
}

export function FloatingJobsWidget() {
  const { activeJobs } = useAudioJobs();
  const [expanded, setExpanded] = useState(false);

  if (activeJobs.length === 0) return null;

  const inProgress = activeJobs.filter(j => j.status === 'queued' || j.status === 'processing');
  const completed = activeJobs.filter(j => j.status === 'completed');

  const badgeText = inProgress.length > 0
    ? `${inProgress.length} ⚡`
    : `${completed.length} ✓`;

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="fixed bottom-6 right-6 z-50 bg-purple-600 hover:bg-purple-500 text-white px-3 py-2 rounded-full shadow-lg flex items-center gap-2 transition-all"
      >
        {inProgress.length > 0 && <Loader2 size={14} className="animate-spin" />}
        <span className="text-sm font-medium">{badgeText}</span>
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-80 max-h-96 bg-gray-900 border border-gray-700 rounded-xl shadow-2xl flex flex-col">
      <div className="flex items-center justify-between p-3 border-b border-gray-700">
        <span className="text-sm font-semibold text-gray-200">Generations</span>
        <button onClick={() => setExpanded(false)} className="text-gray-400 hover:text-gray-200">
          <X size={16} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {activeJobs.map(job => (
          <JobCard key={job.id} job={job} />
        ))}
      </div>
    </div>
  );
}
