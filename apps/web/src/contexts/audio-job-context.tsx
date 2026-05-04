'use client';

import { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { jobsApi, JobResponse } from '@/lib/api';
import { toast } from 'sonner';

export interface ActiveJob {
  id: string;
  type: JobResponse['type'];
  status: JobResponse['status'];
  progress: number;
  progressMessage: string;
  stepNumber: number;
  totalSteps: number;
  outputUrl: string | null;
  outputUrls: Record<string, string> | null;
  modelName: string;
  saved: boolean;
  createdAt: string;
  completedAt: string | null;
  errorMessage: string | null;
}

interface AudioJobContextValue {
  activeJobs: ActiveJob[];
  submitJob: (submitFn: () => Promise<{ job_id: string }>) => Promise<string>;
  cancelJob: (jobId: string) => Promise<void>;
  saveJob: (jobId: string) => Promise<void>;
  unsaveJob: (jobId: string) => Promise<void>;
  dismissJob: (jobId: string) => void;
}

const AudioJobContext = createContext<AudioJobContextValue | null>(null);

export function useAudioJobs() {
  const ctx = useContext(AudioJobContext);
  if (!ctx) throw new Error('useAudioJobs must be used within AudioJobProvider');
  return ctx;
}

function mapJob(job: JobResponse): ActiveJob {
  return {
    id: job.uuid,
    type: job.type,
    status: job.status,
    progress: job.progress,
    progressMessage: job.progress_message || '',
    stepNumber: job.step_number,
    totalSteps: job.total_steps,
    outputUrl: job.output_url,
    outputUrls: job.output_urls,
    modelName: job.voice_model?.name || 'Unknown',
    saved: job.saved,
    createdAt: job.created_at,
    completedAt: job.completed_at,
    errorMessage: job.error_message,
  };
}

export function AudioJobProvider({ children }: { children: React.ReactNode }) {
  const [jobs, setJobs] = useState<ActiveJob[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const pollRef = useRef<NodeJS.Timeout | null>(null);
  const isFirstPoll = useRef<Record<string, boolean>>({});

  // Fetch initial jobs on mount
  useEffect(() => {
    const fetchInitial = async () => {
      try {
        const activeResp = await jobsApi.list({ status: 'processing' });
        const queuedResp = await jobsApi.list({ status: 'queued' });
        const active = activeResp?.data || activeResp || [];
        const queued = queuedResp?.data || queuedResp || [];
        const all = [...active, ...queued];
        setJobs(all.map(mapJob));
      } catch (e) {
        console.error('Failed to fetch initial jobs:', e);
      }
    };
    fetchInitial();
  }, []);

  // Poll active jobs — use serialized active IDs as dependency to avoid infinite re-render loop
  const activeIds = jobs.filter(j => j.status === 'queued' || j.status === 'processing').map(j => j.id).join(',');

  useEffect(() => {
    if (!activeIds) {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }

    const ids = activeIds.split(',');

    const poll = async () => {
      for (const id of ids) {
        try {
          const updated = await jobsApi.get(id);
          setJobs(prev => prev.map(j => j.id === id ? mapJob(updated) : j));
        } catch (e) {
          console.error(`Failed to poll job ${id}:`, e);
        }
      }
    };

    // First poll at 1s for quick jobs (TTS, voice convert)
    const hasNewJobs = ids.some(id => isFirstPoll.current[id]);
    if (hasNewJobs) {
      const timeout = setTimeout(() => {
        poll();
        ids.forEach(id => { isFirstPoll.current[id] = false; });
      }, 1000);
      return () => clearTimeout(timeout);
    }

    pollRef.current = setInterval(poll, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [activeIds]);

  const submitJob = useCallback(async (submitFn: () => Promise<{ job_id: string }>) => {
    const { job_id } = await submitFn();
    isFirstPoll.current[job_id] = true;
    
    // Add placeholder job
    setJobs(prev => [...prev, {
      id: job_id,
      type: 'audio_swap',
      status: 'queued',
      progress: 0,
      progressMessage: 'Queued...',
      stepNumber: 0,
      totalSteps: 1,
      outputUrl: null,
      outputUrls: null,
      modelName: '',
      saved: false,
      createdAt: new Date().toISOString(),
      completedAt: null,
      errorMessage: null,
    }]);

    return job_id;
  }, []);

  const cancelJob = useCallback(async (jobId: string) => {
    const response = await jobsApi.cancel(jobId);
    setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'cancelled' as const } : j));
    if (response.message?.includes('up to 60 seconds')) {
      toast.info('Cancellation requested — may take up to 60 seconds to fully stop.');
    }
  }, []);

  const saveJob = useCallback(async (jobId: string) => {
    await jobsApi.save(jobId);
    setJobs(prev => prev.map(j => j.id === jobId ? { ...j, saved: true } : j));
  }, []);

  const unsaveJob = useCallback(async (jobId: string) => {
    await jobsApi.unsave(jobId);
    setJobs(prev => prev.map(j => j.id === jobId ? { ...j, saved: false } : j));
  }, []);

  const dismissJob = useCallback((jobId: string) => {
    setDismissed(prev => new Set([...prev, jobId]));
    setJobs(prev => prev.filter(j => j.id !== jobId));
  }, []);

  const visibleJobs = jobs.filter(j => !dismissed.has(j.id));

  return (
    <AudioJobContext.Provider value={{ activeJobs: visibleJobs, submitJob, cancelJob, saveJob, unsaveJob, dismissJob }}>
      {children}
    </AudioJobContext.Provider>
  );
}
