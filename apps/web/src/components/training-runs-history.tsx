'use client';

import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  GitBranch, GitCommit, Play, Pause, StopCircle, RefreshCw,
  ChevronRight, ChevronDown, Loader2, AlertCircle, CheckCircle2,
  Clock, Zap, Target, Star, Archive, Trash2, MessageSquare,
  History, ArrowRightFromLine, Database
} from 'lucide-react';
import {
  trainingRunsApi,
  TrainingRun,
  TrainingCheckpoint,
  DatasetVersion,
  TrainingConfig,
  TrainingTreeNode,
} from '@/lib/api';

// Legacy training info structure from voice-engine
interface LegacyTrainingInfo {
  training?: {
    has_model?: boolean;
    has_index?: boolean;
    epochs_trained?: number;
    target_epochs?: number;
    latest_checkpoint?: string | null;
    status?: string;
    job_id?: string;
    current_epoch?: number;
    total_epochs?: number;
    progress?: number;
    last_trained?: string | null;
    checkpoint_count?: number;
  };
  recordings?: {
    count?: number;
    total_duration?: number;
    duration_seconds?: number;
    duration_minutes?: number;
  };
  preprocessed?: {
    count?: number;
    has_data?: boolean;
  };
  preprocessing?: {
    processed?: boolean;
  };
}

interface TrainingRunsHistoryProps {
  modelSlug: string;
  onStartTraining?: (mode: 'new' | 'resume' | 'continue' | 'branch', options?: any) => void;
  legacyTrainingInfo?: LegacyTrainingInfo | null;
}

const statusColors: Record<string, string> = {
  pending: 'text-yellow-400 bg-yellow-400/10',
  preparing: 'text-blue-400 bg-blue-400/10',
  training: 'text-green-400 bg-green-400/10 animate-pulse',
  paused: 'text-orange-400 bg-orange-400/10',
  completed: 'text-emerald-400 bg-emerald-400/10',
  failed: 'text-red-400 bg-red-400/10',
  cancelled: 'text-gray-400 bg-gray-400/10',
};

const statusIcons: Record<string, React.ReactNode> = {
  pending: <Clock className="w-4 h-4" />,
  preparing: <RefreshCw className="w-4 h-4 animate-spin" />,
  training: <Zap className="w-4 h-4" />,
  paused: <Pause className="w-4 h-4" />,
  completed: <CheckCircle2 className="w-4 h-4" />,
  failed: <AlertCircle className="w-4 h-4" />,
  cancelled: <StopCircle className="w-4 h-4" />,
};

const modeColors: Record<string, string> = {
  new: 'bg-blue-600',
  resume: 'bg-green-600',
  continue: 'bg-purple-600',
  branch: 'bg-orange-600',
};

export function TrainingRunsHistory({ modelSlug, onStartTraining, legacyTrainingInfo }: TrainingRunsHistoryProps) {
  const queryClient = useQueryClient();
  const [expandedRuns, setExpandedRuns] = useState<Set<number>>(new Set());
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<TrainingCheckpoint | null>(null);
  const [showBranchModal, setShowBranchModal] = useState(false);
  const [noteText, setNoteText] = useState('');
  const [editingNoteId, setEditingNoteId] = useState<number | null>(null);

  // Fetch training runs
  const { data: runsData, isLoading: loadingRuns } = useQuery({
    queryKey: ['training-runs', modelSlug],
    queryFn: () => trainingRunsApi.listRuns(modelSlug),
    refetchInterval: (query) => {
      // Refresh every 5 seconds if there's an active run
      return query.state.data?.active_run ? 5000 : false;
    },
  });

  // Fetch training tree
  const { data: treeData, isLoading: loadingTree } = useQuery({
    queryKey: ['training-tree', modelSlug],
    queryFn: () => trainingRunsApi.getTree(modelSlug),
  });

  // Fetch dataset versions
  const { data: datasetData } = useQuery({
    queryKey: ['dataset-versions', modelSlug],
    queryFn: () => trainingRunsApi.getDatasetVersions(modelSlug),
  });

  // Resume mutation
  const resumeMutation = useMutation({
    mutationFn: ({ runId, additionalEpochs }: { runId: number; additionalEpochs?: number }) =>
      trainingRunsApi.resumeRun(runId, additionalEpochs),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', modelSlug] });
    },
  });

  // Cancel mutation
  const cancelMutation = useMutation({
    mutationFn: (runId: number) => trainingRunsApi.cancelRun(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', modelSlug] });
    },
  });

  // Archive checkpoint mutation
  const archiveMutation = useMutation({
    mutationFn: (checkpointId: number) => trainingRunsApi.archiveCheckpoint(checkpointId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', modelSlug] });
    },
  });

  // Add note mutation
  const addNoteMutation = useMutation({
    mutationFn: ({ checkpointId, notes }: { checkpointId: number; notes: string }) =>
      trainingRunsApi.addCheckpointNote(checkpointId, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', modelSlug] });
      setEditingNoteId(null);
      setNoteText('');
    },
  });

  const toggleRunExpanded = (runId: number) => {
    setExpandedRuns(prev => {
      const next = new Set(prev);
      if (next.has(runId)) {
        next.delete(runId);
      } else {
        next.add(runId);
      }
      return next;
    });
  };

  const handleResume = (run: TrainingRun) => {
    if (onStartTraining) {
      onStartTraining('resume', { runId: run.id });
    } else {
      resumeMutation.mutate({ runId: run.id });
    }
  };

  const handleContinueFromCheckpoint = (checkpoint: TrainingCheckpoint) => {
    if (onStartTraining) {
      onStartTraining('continue', { checkpointId: checkpoint.id });
    }
    setSelectedCheckpoint(null);
  };

  const handleBranchFromCheckpoint = (checkpoint: TrainingCheckpoint) => {
    setSelectedCheckpoint(checkpoint);
    setShowBranchModal(true);
  };

  if (loadingRuns || loadingTree) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
      </div>
    );
  }

  const runs = runsData?.runs || [];
  const activeRun = runsData?.active_run;
  const tree = treeData?.tree || [];
  
  // Calculate stats from legacy info if no database runs
  const hasLegacyTraining = !runs.length && legacyTrainingInfo?.training?.latest_checkpoint;
  const legacyEpochs = legacyTrainingInfo?.training?.epochs_trained || 0;
  const totalRuns = hasLegacyTraining ? 1 : (treeData?.total_runs || runs.length);
  const totalEpochs = hasLegacyTraining ? legacyEpochs : (treeData?.total_epochs || 0);

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <div className="text-sm text-gray-400">Total Runs</div>
          <div className="text-2xl font-bold text-white">{totalRuns}</div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <div className="text-sm text-gray-400">Total Epochs</div>
          <div className="text-2xl font-bold text-white">{totalEpochs}</div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <div className="text-sm text-gray-400">Training Time</div>
          <div className="text-2xl font-bold text-white">{treeData?.total_time || '-'}</div>
        </div>
      </div>

      {/* Active Training Run */}
      {activeRun && (
        <div className="bg-gradient-to-r from-green-900/30 to-emerald-900/30 rounded-lg p-4 border border-green-700/50">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-green-400" />
              <span className="font-semibold text-green-400">Active Training</span>
            </div>
            <button
              onClick={() => cancelMutation.mutate(activeRun.id)}
              disabled={cancelMutation.isPending}
              className="px-3 py-1.5 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg text-sm flex items-center gap-2 transition-colors"
            >
              {cancelMutation.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <StopCircle className="w-4 h-4" />
              )}
              Cancel
            </button>
          </div>
          <RunCard run={activeRun} expanded={true} onToggle={() => {}} />
        </div>
      )}

      {/* Training History Tree */}
      <div className="bg-gray-800/50 rounded-lg border border-gray-700">
        <div className="p-4 border-b border-gray-700 flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-white">Training History</h3>
        </div>

        {runs.length === 0 && !hasLegacyTraining ? (
          <div className="p-8 text-center text-gray-400">
            <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No training runs yet</p>
            <p className="text-sm mt-1">Start your first training run to see the history here</p>
          </div>
        ) : runs.length === 0 && hasLegacyTraining ? (
          /* Show legacy training info for models trained before the tracking system */
          <div className="p-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="px-2 py-0.5 rounded text-xs font-medium bg-blue-600">
                Initial Training
              </div>
              <div className="font-mono text-sm text-gray-300">#1</div>
              <div className="flex items-center gap-1.5 px-2 py-1 rounded text-xs text-emerald-400 bg-emerald-400/10">
                <CheckCircle2 className="w-4 h-4" />
                <span>Completed</span>
              </div>
              <div className="flex-1" />
              <div className="text-sm text-gray-400">
                <span className="font-medium text-white">{legacyEpochs}</span>
                <span className="mx-1">/</span>
                <span>{legacyTrainingInfo?.training?.target_epochs || 78} epochs</span>
              </div>
            </div>
            
            {/* Show checkpoint info */}
            <div className="ml-8 pl-4 border-l-2 border-gray-700 space-y-2">
              <div className="flex items-center gap-3 p-3 bg-gray-700/30 rounded-lg">
                <GitCommit className="w-4 h-4 text-purple-400" />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Checkpoint</span>
                    <span className="text-xs text-gray-500">@ epoch {legacyEpochs}</span>
                    {legacyTrainingInfo?.training?.has_model && (
                      <span className="px-1.5 py-0.5 bg-green-600/20 text-green-400 rounded text-xs">Model Ready</span>
                    )}
                    {legacyTrainingInfo?.training?.has_index && (
                      <span className="px-1.5 py-0.5 bg-blue-600/20 text-blue-400 rounded text-xs">Indexed</span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {legacyTrainingInfo?.training?.latest_checkpoint}
                  </div>
                </div>
              </div>
            </div>
            
            <p className="text-xs text-gray-500 mt-4 ml-8 italic">
              This model was trained before the training history system was added.
              Future training runs will be fully tracked here.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-gray-700/50">
            {runs.map(run => (
              <div key={run.id} className="p-4">
                <RunCard
                  run={run}
                  expanded={expandedRuns.has(run.id)}
                  onToggle={() => toggleRunExpanded(run.id)}
                  onResume={() => handleResume(run)}
                  onCancel={() => cancelMutation.mutate(run.id)}
                  onContinueFromCheckpoint={handleContinueFromCheckpoint}
                  onBranchFromCheckpoint={handleBranchFromCheckpoint}
                  showActions={!run.is_active}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Branch Modal */}
      {showBranchModal && selectedCheckpoint && (
        <BranchModal
          checkpoint={selectedCheckpoint}
          datasetVersions={datasetData?.versions || []}
          onClose={() => {
            setShowBranchModal(false);
            setSelectedCheckpoint(null);
          }}
          onBranch={(datasetId, config) => {
            if (onStartTraining) {
              onStartTraining('branch', {
                checkpointId: selectedCheckpoint.id,
                datasetVersionId: datasetId,
                config,
              });
            }
            setShowBranchModal(false);
            setSelectedCheckpoint(null);
          }}
        />
      )}
    </div>
  );
}

interface RunCardProps {
  run: TrainingRun;
  expanded: boolean;
  onToggle: () => void;
  onResume?: () => void;
  onCancel?: () => void;
  onContinueFromCheckpoint?: (checkpoint: TrainingCheckpoint) => void;
  onBranchFromCheckpoint?: (checkpoint: TrainingCheckpoint) => void;
  showActions?: boolean;
}

function RunCard({
  run,
  expanded,
  onToggle,
  onResume,
  onCancel,
  onContinueFromCheckpoint,
  onBranchFromCheckpoint,
  showActions = true,
}: RunCardProps) {
  const queryClient = useQueryClient();
  
  // Fetch checkpoints when expanded
  const { data: checkpointsData } = useQuery({
    queryKey: ['run-checkpoints', run.id],
    queryFn: () => trainingRunsApi.getCheckpoints(run.id),
    enabled: expanded,
  });

  const checkpoints = checkpointsData?.checkpoints || [];
  const bestCheckpoint = checkpointsData?.best_checkpoint;

  return (
    <div className="space-y-3">
      {/* Run Header */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggle}
          className="p-1 hover:bg-gray-700/50 rounded transition-colors"
        >
          {expanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </button>

        <div className={`px-2 py-0.5 rounded text-xs font-medium ${modeColors[run.mode]}`}>
          {run.mode_display}
        </div>

        <div className="font-mono text-sm text-gray-300">{run.run_number}</div>

        <div className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs ${statusColors[run.status]}`}>
          {statusIcons[run.status]}
          <span className="capitalize">{run.status}</span>
        </div>

        <div className="flex-1" />

        {/* Progress */}
        <div className="text-sm text-gray-400">
          <span className="font-medium text-white">{run.completed_epochs}</span>
          <span className="mx-1">/</span>
          <span>{run.target_epochs} epochs</span>
        </div>

        {/* Duration */}
        {run.duration !== '-' && (
          <div className="flex items-center gap-1.5 text-sm text-gray-400">
            <Clock className="w-4 h-4" />
            {run.duration}
          </div>
        )}

        {/* Actions */}
        {showActions && (
          <div className="flex items-center gap-2">
            {run.can_resume && (
              <button
                onClick={onResume}
                className="p-2 hover:bg-green-600/20 text-green-400 rounded-lg transition-colors"
                title="Resume training"
              >
                <Play className="w-4 h-4" />
              </button>
            )}
            {run.is_active && onCancel && (
              <button
                onClick={onCancel}
                className="p-2 hover:bg-red-600/20 text-red-400 rounded-lg transition-colors"
                title="Cancel training"
              >
                <StopCircle className="w-4 h-4" />
              </button>
            )}
          </div>
        )}
      </div>

      {/* Progress Bar for active runs */}
      {run.is_active && (
        <div className="ml-8">
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-500"
              style={{
                width: `${(run.completed_epochs / run.target_epochs) * 100}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Run Details & Checkpoints */}
      {expanded && (
        <div className="ml-8 space-y-3">
          {/* Run Info */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {run.started_at && (
              <div>
                <div className="text-gray-500">Started</div>
                <div className="text-gray-300">
                  {new Date(run.started_at).toLocaleString()}
                </div>
              </div>
            )}
            {run.completed_at && (
              <div>
                <div className="text-gray-500">Completed</div>
                <div className="text-gray-300">
                  {new Date(run.completed_at).toLocaleString()}
                </div>
              </div>
            )}
            {run.best_loss && (
              <div>
                <div className="text-gray-500">Best Loss</div>
                <div className="text-gray-300">{run.best_loss.toFixed(4)}</div>
              </div>
            )}
            {run.parent_run && (
              <div>
                <div className="text-gray-500">Based On</div>
                <div className="text-gray-300 flex items-center gap-1">
                  <GitBranch className="w-3 h-3" />
                  {run.parent_run.run_number}
                </div>
              </div>
            )}
          </div>

          {/* Checkpoints */}
          {checkpoints.length > 0 && (
            <div className="border-t border-gray-700/50 pt-3">
              <div className="flex items-center gap-2 mb-2">
                <GitCommit className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-400">
                  Checkpoints ({checkpoints.length})
                </span>
              </div>
              <div className="space-y-2">
                {checkpoints.slice(0, 5).map(checkpoint => (
                  <CheckpointCard
                    key={checkpoint.id}
                    checkpoint={checkpoint}
                    isBest={bestCheckpoint?.id === checkpoint.id}
                    onContinue={() => onContinueFromCheckpoint?.(checkpoint)}
                    onBranch={() => onBranchFromCheckpoint?.(checkpoint)}
                  />
                ))}
                {checkpoints.length > 5 && (
                  <div className="text-sm text-gray-500 text-center">
                    + {checkpoints.length - 5} more checkpoints
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Message */}
          {run.error_message && (
            <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-red-400 text-sm">
                <AlertCircle className="w-4 h-4" />
                <span>{run.error_message}</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface CheckpointCardProps {
  checkpoint: TrainingCheckpoint;
  isBest?: boolean;
  onContinue?: () => void;
  onBranch?: () => void;
}

function CheckpointCard({ checkpoint, isBest, onContinue, onBranch }: CheckpointCardProps) {
  return (
    <div className={`flex items-center gap-3 p-2 rounded-lg ${
      isBest ? 'bg-yellow-900/20 border border-yellow-700/50' : 'bg-gray-800/50'
    }`}>
      <div className="w-2 h-2 rounded-full bg-gray-500" />
      
      <div className="font-mono text-sm text-gray-300">{checkpoint.short_name}</div>

      {/* Flags */}
      <div className="flex items-center gap-1">
        {checkpoint.is_best && (
          <span className="px-1.5 py-0.5 bg-yellow-600/20 text-yellow-400 text-xs rounded">
            Best
          </span>
        )}
        {checkpoint.is_milestone && (
          <span className="px-1.5 py-0.5 bg-blue-600/20 text-blue-400 text-xs rounded">
            Milestone
          </span>
        )}
        {checkpoint.is_final && (
          <span className="px-1.5 py-0.5 bg-green-600/20 text-green-400 text-xs rounded">
            Final
          </span>
        )}
      </div>

      {/* Loss */}
      {checkpoint.loss_g && (
        <div className="text-sm text-gray-400">
          Loss: <span className="text-gray-300">{checkpoint.loss_g.toFixed(4)}</span>
        </div>
      )}

      <div className="flex-1" />

      {/* Actions */}
      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={onContinue}
          className="p-1.5 hover:bg-purple-600/20 text-purple-400 rounded transition-colors"
          title="Continue from this checkpoint"
        >
          <ArrowRightFromLine className="w-4 h-4" />
        </button>
        <button
          onClick={onBranch}
          className="p-1.5 hover:bg-orange-600/20 text-orange-400 rounded transition-colors"
          title="Branch from this checkpoint"
        >
          <GitBranch className="w-4 h-4" />
        </button>
      </div>

      {/* Size & Time */}
      <div className="text-xs text-gray-500">
        {checkpoint.file_size}
      </div>
    </div>
  );
}

interface BranchModalProps {
  checkpoint: TrainingCheckpoint;
  datasetVersions: DatasetVersion[];
  onClose: () => void;
  onBranch: (datasetId: number, config: TrainingConfig) => void;
}

function BranchModal({ checkpoint, datasetVersions, onClose, onBranch }: BranchModalProps) {
  const [selectedDataset, setSelectedDataset] = useState<number | null>(
    datasetVersions.find(v => v.is_latest)?.id || null
  );
  const [epochs, setEpochs] = useState(100);

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-xl border border-gray-700 shadow-2xl max-w-md w-full mx-4 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Branch from Checkpoint</h3>

        <div className="space-y-4">
          <div>
            <div className="text-sm text-gray-400 mb-1">Branching from</div>
            <div className="flex items-center gap-2 p-3 bg-gray-800/50 rounded-lg">
              <GitCommit className="w-4 h-4 text-purple-400" />
              <span className="font-mono text-gray-300">{checkpoint.checkpoint_name}</span>
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Select Dataset Version</label>
            <select
              value={selectedDataset || ''}
              onChange={(e) => setSelectedDataset(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white"
            >
              {datasetVersions.map(v => (
                <option key={v.id} value={v.id}>
                  v{v.version_number} - {v.audio_count} files ({v.duration})
                  {v.is_latest ? ' (latest)' : ''}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Target Epochs</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              min={1}
              max={2000}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white"
            />
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => selectedDataset && onBranch(selectedDataset, { epochs })}
            disabled={!selectedDataset}
            className="px-4 py-2 bg-orange-600 hover:bg-orange-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <GitBranch className="w-4 h-4" />
            Create Branch
          </button>
        </div>
      </div>
    </div>
  );
}

export default TrainingRunsHistory;
