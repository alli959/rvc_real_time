'use client';

import { useState, useRef, useEffect } from 'react';
import { Play, Pause, Download } from 'lucide-react';

interface AudioPlayerProps {
  url: string;
  name: string;
  subtitle?: string;
  icon?: React.ReactNode;
  onDownload?: () => void;
  showDownload?: boolean;
  compact?: boolean;
}

export function AudioPlayer({
  url,
  name,
  subtitle,
  icon,
  onDownload,
  showDownload = true,
  compact = false,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    const audio = new Audio(url);
    audioRef.current = audio;

    audio.addEventListener('loadedmetadata', () => {
      setDuration(audio.duration);
    });

    audio.addEventListener('timeupdate', () => {
      setCurrentTime(audio.currentTime);
      setProgress((audio.currentTime / audio.duration) * 100 || 0);
    });

    audio.addEventListener('ended', () => {
      setIsPlaying(false);
      setProgress(0);
      setCurrentTime(0);
    });

    return () => {
      audio.pause();
      audio.src = '';
    };
  }, [url]);

  const togglePlay = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !duration) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const width = rect.width;
    const newTime = (clickX / width) * duration;
    
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    setProgress((newTime / duration) * 100);
  };

  const formatTime = (seconds: number) => {
    if (!seconds || !isFinite(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (compact) {
    return (
      <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-2">
        <button
          onClick={togglePlay}
          className="p-2 bg-primary-600 hover:bg-primary-700 rounded-full transition-colors flex-shrink-0"
        >
          {isPlaying ? (
            <Pause className="h-4 w-4 text-white" />
          ) : (
            <Play className="h-4 w-4 text-white" />
          )}
        </button>
        
        <div className="flex-1 min-w-0">
          <div
            className="h-2 bg-gray-700 rounded-full cursor-pointer overflow-hidden"
            onClick={handleProgressClick}
          >
            <div
              className="h-full bg-primary-500 transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
        
        <span className="text-xs text-gray-400 flex-shrink-0 w-10 text-right">
          {formatTime(currentTime)}
        </span>
        
        {showDownload && onDownload && (
          <button
            onClick={onDownload}
            className="p-2 text-gray-400 hover:text-white transition-colors flex-shrink-0"
          >
            <Download className="h-4 w-4" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-3 sm:p-4">
      {/* Header with icon, name, and download */}
      <div className="flex items-center gap-2 mb-2">
        <div className="flex-shrink-0 text-primary-400">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-white font-medium text-sm truncate" title={name}>{name}</p>
          {subtitle && <p className="text-xs text-gray-400">{subtitle}</p>}
        </div>
        {showDownload && onDownload && (
          <button
            onClick={onDownload}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors flex-shrink-0"
          >
            <Download className="h-4 w-4" />
          </button>
        )}
      </div>
      
      {/* Progress Bar with Play Button */}
      <div className="flex items-center gap-2">
        <button
          onClick={togglePlay}
          className="p-2 bg-primary-600 hover:bg-primary-700 rounded-full transition-colors flex-shrink-0"
        >
          {isPlaying ? (
            <Pause className="h-4 w-4 text-white" />
          ) : (
            <Play className="h-4 w-4 text-white" />
          )}
        </button>
        <span className="text-xs text-gray-400 w-8 flex-shrink-0 tabular-nums">{formatTime(currentTime)}</span>
        <div
          className="flex-1 min-w-[40px] h-2 bg-gray-700 rounded-full cursor-pointer overflow-hidden group"
          onClick={handleProgressClick}
        >
          <div
            className="h-full bg-primary-500 group-hover:bg-primary-400 transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
        </div>
        <span className="hidden sm:block text-xs text-gray-400 w-8 text-right flex-shrink-0 tabular-nums">{formatTime(duration)}</span>
      </div>
    </div>
  );
}
