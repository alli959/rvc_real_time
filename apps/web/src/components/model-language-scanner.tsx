'use client';

import { useState } from 'react';
import { 
  Languages, 
  Scan, 
  Loader2, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Info
} from 'lucide-react';
import { trainerApi, VoiceModel, ScanModelResponse } from '@/lib/api';

interface ModelLanguageScannerProps {
  model: VoiceModel;
  onScanComplete?: (result: ScanModelResponse) => void;
  compact?: boolean;
}

// Get score color based on value
const getScoreColor = (score: number | null | undefined): string => {
  if (score === null || score === undefined) return 'text-gray-500';
  if (score >= 80) return 'text-green-400';
  if (score >= 60) return 'text-yellow-400';
  if (score >= 40) return 'text-orange-400';
  return 'text-red-400';
};

// Get score background color
const getScoreBgColor = (score: number | null | undefined): string => {
  if (score === null || score === undefined) return 'bg-gray-800';
  if (score >= 80) return 'bg-green-500/20';
  if (score >= 60) return 'bg-yellow-500/20';
  if (score >= 40) return 'bg-orange-500/20';
  return 'bg-red-500/20';
};

// Get score icon
const getScoreIcon = (score: number | null | undefined) => {
  if (score === null || score === undefined) return <Info className="h-4 w-4 text-gray-500" />;
  if (score >= 80) return <CheckCircle className="h-4 w-4 text-green-400" />;
  if (score >= 60) return <AlertTriangle className="h-4 w-4 text-yellow-400" />;
  return <XCircle className="h-4 w-4 text-red-400" />;
};

// Format score display
const formatScore = (score: number | null | undefined): string => {
  if (score === null || score === undefined) return 'Not scanned';
  return `${Math.round(score)}%`;
};

// Language display names
const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English',
  is: 'Icelandic',
};

export function ModelLanguageScanner({ model, onScanComplete, compact = false }: ModelLanguageScannerProps) {
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const handleScan = async () => {
    setScanning(true);
    setError(null);

    try {
      const result = await trainerApi.scanModel(model.id);
      onScanComplete?.(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Scan failed';
      setError(message);
    } finally {
      setScanning(false);
    }
  };

  const hasScores = model.en_readiness_score !== null || model.is_readiness_score !== null;
  const scannedAt = model.language_scanned_at 
    ? new Date(model.language_scanned_at).toLocaleDateString()
    : null;

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        {/* EN Score Badge */}
        <div 
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${getScoreBgColor(model.en_readiness_score)}`}
          title={`English readiness: ${formatScore(model.en_readiness_score)}`}
        >
          <span className="font-medium">EN</span>
          <span className={getScoreColor(model.en_readiness_score)}>
            {model.en_readiness_score != null ? `${Math.round(model.en_readiness_score)}` : '—'}
          </span>
        </div>

        {/* IS Score Badge */}
        <div 
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${getScoreBgColor(model.is_readiness_score)}`}
          title={`Icelandic readiness: ${formatScore(model.is_readiness_score)}`}
        >
          <span className="font-medium">IS</span>
          <span className={getScoreColor(model.is_readiness_score)}>
            {model.is_readiness_score != null ? `${Math.round(model.is_readiness_score)}` : '—'}
          </span>
        </div>

        {/* Scan Button */}
        <button
          onClick={handleScan}
          disabled={scanning}
          className="p-1 rounded hover:bg-gray-700 transition-colors disabled:opacity-50"
          title="Scan for language readiness"
        >
          {scanning ? (
            <Loader2 className="h-4 w-4 animate-spin text-primary-400" />
          ) : (
            <Scan className="h-4 w-4 text-gray-400 hover:text-primary-400" />
          )}
        </button>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div 
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-800/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <Languages className="h-5 w-5 text-primary-400" />
          <div>
            <h3 className="font-medium text-white">Language Readiness</h3>
            {scannedAt && (
              <p className="text-xs text-gray-500">Last scanned: {scannedAt}</p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Quick score badges */}
          <div className="flex items-center gap-2">
            <div className={`flex items-center gap-1 px-2 py-1 rounded ${getScoreBgColor(model.en_readiness_score)}`}>
              {getScoreIcon(model.en_readiness_score)}
              <span className="text-sm font-medium">EN</span>
              <span className={`text-sm ${getScoreColor(model.en_readiness_score)}`}>
                {formatScore(model.en_readiness_score)}
              </span>
            </div>
            <div className={`flex items-center gap-1 px-2 py-1 rounded ${getScoreBgColor(model.is_readiness_score)}`}>
              {getScoreIcon(model.is_readiness_score)}
              <span className="text-sm font-medium">IS</span>
              <span className={`text-sm ${getScoreColor(model.is_readiness_score)}`}>
                {formatScore(model.is_readiness_score)}
              </span>
            </div>
          </div>

          {expanded ? (
            <ChevronUp className="h-5 w-5 text-gray-400" />
          ) : (
            <ChevronDown className="h-5 w-5 text-gray-400" />
          )}
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="border-t border-gray-800 p-4 space-y-4">
          {/* Language Details */}
          {(['en', 'is'] as const).map((lang) => {
            const score = lang === 'en' ? model.en_readiness_score : model.is_readiness_score;
            const coverage = lang === 'en' ? model.en_phoneme_coverage : model.is_phoneme_coverage;
            const missing = lang === 'en' ? model.en_missing_phonemes : model.is_missing_phonemes;

            return (
              <div key={lang} className="space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-white">{LANGUAGE_NAMES[lang]}</h4>
                  <span className={`text-lg font-bold ${getScoreColor(score)}`}>
                    {formatScore(score)}
                  </span>
                </div>

                {score != null && score !== undefined && (
                  <>
                    {/* Progress bar */}
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-500 ${
                          score >= 80 ? 'bg-green-500' :
                          score >= 60 ? 'bg-yellow-500' :
                          score >= 40 ? 'bg-orange-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${score}%` }}
                      />
                    </div>

                    {/* Phoneme coverage */}
                    {coverage != null && coverage !== undefined && (
                      <p className="text-sm text-gray-400">
                        Phoneme coverage: {Math.round(coverage)}%
                      </p>
                    )}

                    {/* Missing phonemes */}
                    {missing && missing.length > 0 && (
                      <div className="mt-2">
                        <p className="text-sm text-gray-400 mb-1">Missing phonemes ({missing.length}):</p>
                        <div className="flex flex-wrap gap-1">
                          {missing.slice(0, 10).map((phoneme) => (
                            <span 
                              key={phoneme}
                              className="px-2 py-0.5 bg-red-500/20 text-red-400 rounded text-xs font-mono"
                            >
                              {phoneme}
                            </span>
                          ))}
                          {missing.length > 10 && (
                            <span className="px-2 py-0.5 bg-gray-700 text-gray-400 rounded text-xs">
                              +{missing.length - 10} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </>
                )}

                {score === null && (
                  <p className="text-sm text-gray-500 italic">
                    Not yet analyzed. Click &quot;Scan Model&quot; to analyze.
                  </p>
                )}
              </div>
            );
          })}

          {/* Actions */}
          <div className="flex items-center gap-3 pt-2 border-t border-gray-800">
            <button
              onClick={handleScan}
              disabled={scanning}
              className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-600/50 text-white rounded-lg transition-colors"
            >
              {scanning ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Scanning...
                </>
              ) : (
                <>
                  <Scan className="h-4 w-4" />
                  {hasScores ? 'Rescan Model' : 'Scan Model'}
                </>
              )}
            </button>

            {error && (
              <p className="text-sm text-red-400">{error}</p>
            )}
          </div>

          {/* Scoring explanation */}
          <div className="text-xs text-gray-500 bg-gray-800/50 p-3 rounded">
            <p className="font-medium mb-1">Scoring breakdown:</p>
            <ul className="space-y-0.5">
              <li>• <span className="text-green-400">80-100%</span>: Excellent - Ready for production</li>
              <li>• <span className="text-yellow-400">60-79%</span>: Good - Minor gaps</li>
              <li>• <span className="text-orange-400">40-59%</span>: Fair - Needs improvement</li>
              <li>• <span className="text-red-400">0-39%</span>: Poor - Significant training needed</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

// Bulk scanner for admin
interface BulkModelScannerProps {
  onComplete?: (results: { total: number; scanned: number; failed: number; skipped: number }) => void;
}

export function BulkModelScanner({ onComplete }: BulkModelScannerProps) {
  const [scanning, setScanning] = useState(false);
  const [results, setResults] = useState<{ total: number; scanned: number; failed: number; skipped: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleScanAll = async () => {
    setScanning(true);
    setError(null);
    setResults(null);

    try {
      const response = await trainerApi.scanAllModels(['en', 'is']);
      setResults(response.results);
      onComplete?.(response.results);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Bulk scan failed';
      setError(message);
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Languages className="h-5 w-5 text-primary-400" />
          <div>
            <h3 className="font-medium text-white">Scan All Models</h3>
            <p className="text-sm text-gray-400">Analyze language readiness for all active models</p>
          </div>
        </div>

        <button
          onClick={handleScanAll}
          disabled={scanning}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-600/50 text-white rounded-lg transition-colors"
        >
          {scanning ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Scanning All...
            </>
          ) : (
            <>
              <Scan className="h-4 w-4" />
              Scan All Models
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {results && (
        <div className="grid grid-cols-4 gap-4 mt-4">
          <div className="bg-gray-800 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-white">{results.total}</p>
            <p className="text-xs text-gray-400">Total Models</p>
          </div>
          <div className="bg-green-500/20 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-green-400">{results.scanned}</p>
            <p className="text-xs text-gray-400">Scanned</p>
          </div>
          <div className="bg-red-500/20 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-red-400">{results.failed}</p>
            <p className="text-xs text-gray-400">Failed</p>
          </div>
          <div className="bg-yellow-500/20 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-yellow-400">{results.skipped}</p>
            <p className="text-xs text-gray-400">Skipped</p>
          </div>
        </div>
      )}
    </div>
  );
}
