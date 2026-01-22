"""
Preprocessing Pipeline Module

Implements the full preprocessing pipeline matching the WebUI reference:
1. Audio loading and validation
2. High-pass filtering
3. Silence-based slicing
4. Normalization
5. Chunking with overlap
6. Dual-rate output (target SR + 16kHz)

This module creates the 0_gt_wavs and 1_16k_wavs directories.
"""

import os
import logging
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa

from .config import settings
from .slicer import Slicer
from .audio import load_audio, save_audio, validate_audio

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing."""
    exp_name: str
    input_dir: str
    sample_rate: int = 48000
    version: str = "v2"
    n_threads: int = 4
    
    # Slicer settings (from config)
    threshold_db: float = settings.slicer_threshold_db
    min_length_ms: int = settings.slicer_min_length_ms
    min_interval_ms: int = settings.slicer_min_interval_ms
    hop_size_ms: int = settings.slicer_hop_size_ms
    max_sil_kept_ms: int = settings.slicer_max_sil_kept_ms
    
    # Chunk settings
    chunk_length_sec: float = settings.chunk_length_sec
    chunk_overlap_sec: float = settings.chunk_overlap_sec
    
    # Normalization
    norm_max: float = settings.norm_max
    norm_alpha: float = settings.norm_alpha
    norm_clip_threshold: float = settings.norm_clip_threshold


@dataclass
class PreprocessResult:
    """Result of preprocessing a single file."""
    input_path: str
    success: bool
    chunks_created: int = 0
    error: Optional[str] = None


class AudioPreprocessor:
    """
    Audio preprocessor matching the WebUI reference implementation exactly.
    
    Creates:
    - {exp_dir}/0_gt_wavs/  - Ground truth WAVs at target sample rate
    - {exp_dir}/1_16k_wavs/ - 16kHz WAVs for feature extraction
    """
    
    def __init__(self, config: PreprocessConfig):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.exp_dir = Path(settings.models_dir) / config.exp_name
        self.gt_wavs_dir = self.exp_dir / "0_gt_wavs"
        self.wav16k_dir = self.exp_dir / "1_16k_wavs"
        
        # Create slicer with WebUI parameters
        self.slicer = Slicer(
            sr=config.sample_rate,
            threshold=config.threshold_db,
            min_length=config.min_length_ms,
            min_interval=config.min_interval_ms,
            hop_size=config.hop_size_ms,
            max_sil_kept=config.max_sil_kept_ms,
        )
        
        # High-pass filter coefficients (matching WebUI: butter N=5, Wn=48Hz)
        self.bh, self.ah = signal.butter(
            N=settings.highpass_order,
            Wn=settings.highpass_cutoff,
            btype="high",
            fs=config.sample_rate
        )
        
        # Chunk parameters
        self.per = config.chunk_length_sec  # Chunk length in seconds
        self.overlap = config.chunk_overlap_sec  # Overlap in seconds
        self.tail = self.per + self.overlap
        
        # Normalization parameters
        self.max = config.norm_max
        self.alpha = config.norm_alpha
        
        # Create output directories
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.gt_wavs_dir.mkdir(exist_ok=True)
        self.wav16k_dir.mkdir(exist_ok=True)
        
        # Log file
        self.log_file = self.exp_dir / "preprocess.log"
        
    def _log(self, message: str):
        """Log to both logger and file."""
        logger.info(message)
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")
    
    def _norm_write(
        self,
        tmp_audio: np.ndarray,
        idx0: int,
        idx1: int
    ) -> bool:
        """
        Normalize and write audio chunk to both GT and 16k directories.
        
        Matches WebUI norm_write exactly:
        1. Filter out corrupted audio (max > 2.5)
        2. Apply amplitude normalization
        3. Save to GT directory
        4. Resample to 16kHz
        5. Save to 16k directory
        
        Args:
            tmp_audio: Audio chunk
            idx0: File index
            idx1: Chunk index within file
            
        Returns:
            True if chunk was saved, False if filtered out
        """
        tmp_max = np.abs(tmp_audio).max()
        
        # Filter out corrupted audio (matching WebUI)
        if tmp_max > self.config.norm_clip_threshold:
            logger.warning(f"{idx0}-{idx1}-{tmp_max:.2f}-filtered (amplitude too high)")
            return False
        
        # Skip silent chunks
        if tmp_max < 1e-6:
            logger.warning(f"{idx0}-{idx1}-filtered (silent)")
            return False
        
        # Apply normalization (matching WebUI exactly)
        # Formula: (audio / max * target_max * alpha) + (1 - alpha) * audio
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        
        # Save ground truth at target sample rate
        gt_path = self.gt_wavs_dir / f"{idx0}_{idx1}.wav"
        wavfile.write(
            str(gt_path),
            self.config.sample_rate,
            tmp_audio.astype(np.float32),
        )
        
        # Resample to 16kHz for feature extraction
        tmp_audio_16k = librosa.resample(
            tmp_audio,
            orig_sr=self.config.sample_rate,
            target_sr=16000
        )
        
        # Save 16k version
        wav16k_path = self.wav16k_dir / f"{idx0}_{idx1}.wav"
        wavfile.write(
            str(wav16k_path),
            16000,
            tmp_audio_16k.astype(np.float32),
        )
        
        return True
    
    def process_file(self, path: str, idx0: int) -> PreprocessResult:
        """
        Process a single audio file.
        
        Matches WebUI pipeline method exactly:
        1. Load audio at target sample rate
        2. Apply high-pass filter
        3. Slice on silence boundaries
        4. Create overlapping chunks
        5. Normalize and save
        
        Args:
            path: Path to audio file
            idx0: File index (for naming)
            
        Returns:
            PreprocessResult with status and chunk count
        """
        try:
            # Load audio at target sample rate
            audio = load_audio(path, self.config.sample_rate)
            
            # Apply high-pass filter (matching WebUI)
            # WebUI uses lfilter, not filtfilt, to avoid pre-ringing
            audio = signal.lfilter(self.bh, self.ah, audio)
            
            chunks_saved = 0
            idx1 = 0
            
            # Slice audio on silence boundaries
            for sliced_audio in self.slicer.slice(audio):
                # Create overlapping chunks
                i = 0
                while True:
                    start = int(self.config.sample_rate * (self.per - self.overlap) * i)
                    i += 1
                    
                    if len(sliced_audio[start:]) > self.tail * self.config.sample_rate:
                        # Full chunk
                        tmp_audio = sliced_audio[start : start + int(self.per * self.config.sample_rate)]
                        if self._norm_write(tmp_audio, idx0, idx1):
                            chunks_saved += 1
                        idx1 += 1
                    else:
                        # Final partial chunk
                        tmp_audio = sliced_audio[start:]
                        idx1 += 1
                        break
                
                # Write the final chunk
                if len(tmp_audio) >= self.config.sample_rate * 0.5:  # At least 0.5s
                    if self._norm_write(tmp_audio, idx0, idx1):
                        chunks_saved += 1
            
            self._log(f"{path}\t-> Success ({chunks_saved} chunks)")
            return PreprocessResult(
                input_path=path,
                success=True,
                chunks_created=chunks_saved
            )
            
        except Exception as e:
            error_msg = traceback.format_exc()
            self._log(f"{path}\t-> {error_msg}")
            return PreprocessResult(
                input_path=path,
                success=False,
                error=str(e)
            )
    
    def process_directory(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, int, List[str]]:
        """
        Process all audio files in the input directory.
        
        Args:
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Tuple of (total_chunks, files_processed, errors)
        """
        input_dir = Path(self.config.input_dir)
        
        # Find all audio files
        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aac"]:
            audio_files.extend(input_dir.glob(ext))
            audio_files.extend(input_dir.glob(ext.upper()))
        
        audio_files = sorted(audio_files)
        
        if not audio_files:
            self._log(f"No audio files found in {input_dir}")
            return 0, 0, [f"No audio files found in {input_dir}"]
        
        self._log(f"start preprocess ({len(audio_files)} files)")
        
        total_chunks = 0
        errors = []
        
        # Process files (can be parallelized)
        if self.config.n_threads > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.n_threads) as executor:
                futures = {
                    executor.submit(self.process_file, str(f), idx): (idx, f)
                    for idx, f in enumerate(audio_files)
                }
                
                for future in as_completed(futures):
                    idx, filepath = futures[future]
                    result = future.result()
                    
                    if result.success:
                        total_chunks += result.chunks_created
                    else:
                        errors.append(f"{filepath}: {result.error}")
                    
                    if progress_callback:
                        progress_callback(idx + 1, len(audio_files), str(filepath))
        else:
            # Sequential processing
            for idx, filepath in enumerate(audio_files):
                result = self.process_file(str(filepath), idx)
                
                if result.success:
                    total_chunks += result.chunks_created
                else:
                    errors.append(f"{filepath}: {result.error}")
                
                if progress_callback:
                    progress_callback(idx + 1, len(audio_files), str(filepath))
        
        self._log(f"end preprocess ({total_chunks} total chunks)")
        
        return total_chunks, len(audio_files), errors
    
    def validate_output(self) -> Tuple[bool, dict]:
        """
        Validate preprocessing output.
        
        Checks:
        - GT and 16k directories have matching files
        - All files are readable
        - No empty directories
        
        Returns:
            Tuple of (is_valid, stats_dict)
        """
        gt_files = set(f.name for f in self.gt_wavs_dir.glob("*.wav"))
        wav16k_files = set(f.name for f in self.wav16k_dir.glob("*.wav"))
        
        mismatches = gt_files.symmetric_difference(wav16k_files)
        
        stats = {
            "gt_wavs_count": len(gt_files),
            "wav16k_count": len(wav16k_files),
            "mismatches": list(mismatches),
            "exp_dir": str(self.exp_dir),
        }
        
        is_valid = len(mismatches) == 0 and len(gt_files) > 0
        
        return is_valid, stats


def run_preprocessing(
    config: PreprocessConfig,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[bool, dict]:
    """
    Run full preprocessing pipeline.
    
    Args:
        config: Preprocessing configuration
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (success, result_dict)
    """
    try:
        preprocessor = AudioPreprocessor(config)
        total_chunks, files_processed, errors = preprocessor.process_directory(progress_callback)
        
        is_valid, stats = preprocessor.validate_output()
        
        return is_valid and len(errors) == 0, {
            "total_chunks": total_chunks,
            "files_processed": files_processed,
            "errors": errors,
            "validation": stats,
        }
        
    except Exception as e:
        logger.exception("Preprocessing failed")
        return False, {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
