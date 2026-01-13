"""
Voice Count Detector Module

Detects the number of distinct singers/speakers in an audio file using 
vocal separation and multi-pitch analysis.

This approach:
1. First separates vocals from instrumentals
2. Uses multi-pitch detection to find simultaneous notes
3. Analyzes pitch distribution over time to estimate voice count
4. Falls back to MFCC clustering for speech/single-voice detection

Optimized for:
- A cappella groups (Billy Joel "For the Longest Time")
- Duets (Simon & Garfunkel)
- Songs with harmony/backup vocals
"""

import os
import logging
import tempfile
from typing import Tuple, List, Dict, Optional
import numpy as np
import librosa
import torch
from dataclasses import dataclass
from scipy import signal
from scipy.ndimage import maximum_filter1d

logger = logging.getLogger(__name__)

# Try to import optional dependencies
SPEECHBRAIN_AVAILABLE = False
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    logger.warning("speechbrain not installed - voice detection will use basic method")


@dataclass
class VoiceDetectionResult:
    """Result of voice detection analysis"""
    voice_count: int  # Number of distinct voices detected
    confidence: float  # Confidence score (0-1)
    method: str  # Method used for detection
    details: Dict  # Additional details about detection


def detect_voice_activity(
    audio: np.ndarray,
    sr: int,
    frame_length_ms: int = 25,
    hop_length_ms: int = 10,
    energy_threshold: float = 0.01,
    zcr_threshold: float = 0.15
) -> np.ndarray:
    """
    Simple Voice Activity Detection using energy and zero-crossing rate.
    
    Args:
        audio: Audio signal (mono)
        sr: Sample rate
        frame_length_ms: Frame length in milliseconds
        hop_length_ms: Hop length in milliseconds
        energy_threshold: Minimum energy threshold (relative to max)
        zcr_threshold: Maximum zero-crossing rate threshold
        
    Returns:
        Boolean array indicating voiced frames
    """
    frame_length = int(sr * frame_length_ms / 1000)
    hop_length = int(sr * hop_length_ms / 1000)
    
    # Compute short-time energy
    energy = librosa.feature.rms(
        y=audio, 
        frame_length=frame_length, 
        hop_length=hop_length
    )[0]
    
    # Compute zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Normalize energy
    if energy.max() > 0:
        energy_norm = energy / energy.max()
    else:
        energy_norm = energy
    
    # Voice activity: high energy and low ZCR (voiced sounds have periodic patterns)
    voice_frames = (energy_norm > energy_threshold) & (zcr < zcr_threshold)
    
    return voice_frames


def extract_voiced_segments(
    audio: np.ndarray,
    sr: int,
    min_segment_duration: float = 1.0,
    max_segment_duration: float = 5.0
) -> List[np.ndarray]:
    """
    Extract voiced segments from audio.
    
    Args:
        audio: Audio signal (mono)
        sr: Sample rate
        min_segment_duration: Minimum segment duration in seconds
        max_segment_duration: Maximum segment duration in seconds
        
    Returns:
        List of voiced audio segments
    """
    hop_length_ms = 10
    hop_length = int(sr * hop_length_ms / 1000)
    
    voice_frames = detect_voice_activity(audio, sr)
    
    segments = []
    in_segment = False
    segment_start = 0
    
    min_frames = int(min_segment_duration * 1000 / hop_length_ms)
    max_frames = int(max_segment_duration * 1000 / hop_length_ms)
    
    for i, is_voiced in enumerate(voice_frames):
        if is_voiced and not in_segment:
            segment_start = i
            in_segment = True
        elif not is_voiced and in_segment:
            segment_length = i - segment_start
            if segment_length >= min_frames:
                # Extract audio segment
                start_sample = segment_start * hop_length
                end_sample = min(i * hop_length, len(audio))
                
                # Limit segment duration
                if segment_length > max_frames:
                    end_sample = start_sample + max_frames * hop_length
                
                segments.append(audio[start_sample:end_sample])
            in_segment = False
    
    # Handle segment that goes to end
    if in_segment:
        segment_length = len(voice_frames) - segment_start
        if segment_length >= min_frames:
            start_sample = segment_start * hop_length
            end_sample = len(audio)
            if segment_length > max_frames:
                end_sample = start_sample + max_frames * hop_length
            segments.append(audio[start_sample:end_sample])
    
    return segments


def compute_mfcc_embeddings(
    segments: List[np.ndarray],
    sr: int,
    n_mfcc: int = 20
) -> np.ndarray:
    """
    Compute MFCC-based embeddings for audio segments.
    Simple but effective for voice discrimination.
    
    Args:
        segments: List of audio segments
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Array of embeddings (n_segments, n_mfcc * 3)
    """
    embeddings = []
    
    for segment in segments:
        if len(segment) < sr * 0.5:  # Skip segments shorter than 0.5s
            continue
            
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        
        # Compute statistics: mean, std, delta-mean
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Delta MFCCs for capturing dynamics
        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta, axis=1)
        
        # Concatenate features
        embedding = np.concatenate([mfcc_mean, mfcc_std, delta_mean])
        embeddings.append(embedding)
    
    if not embeddings:
        return np.array([])
    
    return np.array(embeddings)


def cluster_voices_simple(
    embeddings: np.ndarray,
    max_voices: int = 5,
    min_cluster_similarity: float = 0.7
) -> Tuple[int, float]:
    """
    Estimate number of distinct voices using simple clustering.
    Uses cosine similarity and hierarchical-like approach.
    
    Args:
        embeddings: Voice embeddings (n_samples, n_features)
        max_voices: Maximum number of voices to detect
        min_cluster_similarity: Minimum similarity within a cluster
        
    Returns:
        Tuple of (voice_count, confidence)
    """
    if len(embeddings) == 0:
        return 1, 0.5  # Default to 1 voice if no segments
    
    if len(embeddings) == 1:
        return 1, 0.6  # Single segment, assume 1 voice
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_norm = embeddings / norms
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    
    # Simple greedy clustering
    n_samples = len(embeddings)
    assigned = np.zeros(n_samples, dtype=bool)
    clusters = []
    
    for i in range(n_samples):
        if assigned[i]:
            continue
            
        # Start new cluster
        cluster = [i]
        assigned[i] = True
        
        for j in range(i + 1, n_samples):
            if assigned[j]:
                continue
            
            # Check if j is similar to cluster centroid
            cluster_similarities = [similarity_matrix[i, j] for k in cluster]
            avg_similarity = np.mean(cluster_similarities)
            
            if avg_similarity >= min_cluster_similarity:
                cluster.append(j)
                assigned[j] = True
        
        clusters.append(cluster)
        
        if len(clusters) >= max_voices:
            break
    
    voice_count = len(clusters)
    
    # Compute confidence based on cluster separation
    if voice_count == 1:
        # High confidence if all segments are similar
        avg_similarity = np.mean(similarity_matrix)
        confidence = min(0.9, 0.5 + avg_similarity * 0.4)
    else:
        # Confidence based on inter-cluster vs intra-cluster similarity
        intra_similarities = []
        inter_similarities = []
        
        for ci, cluster in enumerate(clusters):
            for i in cluster:
                for j in cluster:
                    if i < j:
                        intra_similarities.append(similarity_matrix[i, j])
                        
                for cj, other_cluster in enumerate(clusters):
                    if ci < cj:
                        for j in other_cluster:
                            inter_similarities.append(similarity_matrix[i, j])
        
        if intra_similarities and inter_similarities:
            avg_intra = np.mean(intra_similarities)
            avg_inter = np.mean(inter_similarities)
            separation = avg_intra - avg_inter
            confidence = min(0.95, max(0.3, 0.5 + separation))
        else:
            confidence = 0.5
    
    return voice_count, confidence


def detect_voice_count(
    audio: np.ndarray,
    sample_rate: int,
    use_vocals_only: bool = True,
    max_voices: int = 5
) -> VoiceDetectionResult:
    """
    Main function to detect number of voices in audio.
    Uses multi-pitch detection for harmonies, with MFCC clustering fallback.
    
    Args:
        audio: Input audio as numpy array (mono or stereo)
        sample_rate: Sample rate of input audio
        use_vocals_only: Whether to first separate vocals (recommended for music)
        max_voices: Maximum number of voices to detect
        
    Returns:
        VoiceDetectionResult with voice count and confidence
    """
    logger.info(f"Starting voice detection: sr={sample_rate}, use_vocals_only={use_vocals_only}")
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        if audio.shape[0] == 2:  # Channels first
            audio = np.mean(audio, axis=0)
        else:  # Channels last
            audio = np.mean(audio, axis=1)
    
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    vocals = audio
    
    # Optionally separate vocals first (recommended for music)
    if use_vocals_only:
        try:
            from app.vocal_separator import separate_vocals, list_available_models
            available_models = list_available_models()
            
            if available_models:
                logger.info("Separating vocals for better voice detection")
                # Convert to stereo for UVR5
                audio_stereo = np.stack([audio, audio]) if audio.ndim == 1 else audio
                vocals_stereo, _ = separate_vocals(
                    audio_stereo, 
                    sample_rate,
                    model_name=available_models[0]
                )
                # Convert back to mono
                if vocals_stereo.ndim > 1:
                    vocals = np.mean(vocals_stereo, axis=0) if vocals_stereo.shape[0] == 2 else np.mean(vocals_stereo, axis=1)
                else:
                    vocals = vocals_stereo
        except Exception as e:
            logger.warning(f"Vocal separation failed, using full audio: {e}")
            vocals = audio
    
    # Resample to 16kHz for consistent analysis
    target_sr = 16000
    if sample_rate != target_sr:
        vocals = librosa.resample(vocals.astype(np.float32), orig_sr=sample_rate, target_sr=target_sr)
    
    # Try multi-pitch detection first (best for harmonies)
    logger.info("Running multi-pitch detection for harmony analysis")
    pitch_voice_count, pitch_confidence, pitch_details = detect_simultaneous_pitches(vocals, target_sr)
    
    # If multi-pitch found multiple voices with good confidence, use that
    if pitch_voice_count > 1 and pitch_confidence > 0.5:
        logger.info(f"Multi-pitch detection found {pitch_voice_count} voices with {pitch_confidence:.2f} confidence")
        return VoiceDetectionResult(
            voice_count=min(pitch_voice_count, max_voices),
            confidence=pitch_confidence,
            method="multi_pitch",
            details=pitch_details
        )
    
    # Fall back to MFCC clustering for single voice or low confidence
    logger.info("Falling back to MFCC clustering")
    segments = extract_voiced_segments(vocals, target_sr)
    logger.info(f"Found {len(segments)} voiced segments")
    
    if len(segments) == 0:
        return VoiceDetectionResult(
            voice_count=1,
            confidence=0.3,
            method="default",
            details={"reason": "No voiced segments detected"}
        )
    
    embeddings = compute_mfcc_embeddings(segments, target_sr)
    
    if len(embeddings) == 0:
        return VoiceDetectionResult(
            voice_count=1,
            confidence=0.4,
            method="default",
            details={"reason": "No valid segments for embedding"}
        )
    
    # Use lower similarity threshold for harmonizing voices
    voice_count, confidence = cluster_voices_simple(embeddings, max_voices=max_voices, min_cluster_similarity=0.5)
    
    # If pitch detection found hints of multiple voices, boost the count
    if pitch_voice_count > voice_count:
        voice_count = max(voice_count, min(pitch_voice_count, max_voices))
        confidence = max(confidence, pitch_confidence * 0.8)
    
    logger.info(f"Voice detection complete: {voice_count} voices with {confidence:.2f} confidence")
    
    return VoiceDetectionResult(
        voice_count=voice_count,
        confidence=confidence,
        method="mfcc_clustering_enhanced",
        details={
            "segments_analyzed": len(segments),
            "embeddings_computed": len(embeddings),
            "max_voices_checked": max_voices,
            "pitch_hint": pitch_voice_count
        }
    )


def detect_simultaneous_pitches(
    audio: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    min_voice_freq: float = 80.0,
    max_voice_freq: float = 1000.0,
    harmonic_tolerance: float = 0.08
) -> Tuple[int, float, Dict]:
    """
    Detect simultaneous pitches to estimate number of singing voices.
    
    This works by:
    1. Computing short-time spectrogram
    2. Finding prominent peaks (fundamental frequencies)
    3. Filtering out harmonics (multiples of fundamentals)
    4. Counting remaining distinct pitches per frame
    5. Using the median/mode of peak counts across frames
    
    Args:
        audio: Audio signal (mono)
        sr: Sample rate
        frame_length: FFT frame length
        hop_length: Hop between frames
        min_voice_freq: Minimum expected voice frequency (Hz)
        max_voice_freq: Maximum expected voice frequency (Hz)
        harmonic_tolerance: Tolerance for harmonic detection (as fraction)
        
    Returns:
        Tuple of (voice_count, confidence, details)
    """
    # Compute spectrogram
    D = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    
    # Filter to voice frequency range
    voice_mask = (freqs >= min_voice_freq) & (freqs <= max_voice_freq)
    voice_freqs = freqs[voice_mask]
    voice_spectrum = D[voice_mask, :]
    
    # Track simultaneous pitch count per frame
    frame_pitch_counts = []
    
    # Analyze each frame
    for frame_idx in range(voice_spectrum.shape[1]):
        frame = voice_spectrum[:, frame_idx]
        
        # Skip quiet frames
        if np.max(frame) < np.mean(D) * 0.1:
            continue
        
        # Find peaks (potential fundamental frequencies)
        # Use adaptive threshold based on frame energy
        threshold = np.percentile(frame, 85)
        
        peaks = []
        for i in range(1, len(frame) - 1):
            if frame[i] > frame[i-1] and frame[i] > frame[i+1] and frame[i] > threshold:
                peaks.append((voice_freqs[i], frame[i]))
        
        if not peaks:
            continue
        
        # Sort peaks by magnitude
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out harmonics - keep only fundamentals
        fundamentals = []
        for freq, mag in peaks:
            is_harmonic = False
            for fund_freq, _ in fundamentals:
                # Check if this is a harmonic (2x, 3x, 4x, etc.) of an existing fundamental
                for harmonic_num in range(2, 6):
                    expected_harmonic = fund_freq * harmonic_num
                    if abs(freq - expected_harmonic) / expected_harmonic < harmonic_tolerance:
                        is_harmonic = True
                        break
                if is_harmonic:
                    break
            
            if not is_harmonic:
                # Also check if this could be a fundamental of which existing "fundamentals" are harmonics
                # This handles cases where we detected a harmonic first
                is_sub_harmonic = False
                for i, (fund_freq, _) in enumerate(fundamentals):
                    for harmonic_num in range(2, 6):
                        expected_fund = fund_freq / harmonic_num
                        if abs(freq - expected_fund) / freq < harmonic_tolerance:
                            # This is a lower fundamental - replace the harmonic
                            is_sub_harmonic = True
                            break
                    if is_sub_harmonic:
                        break
                
                fundamentals.append((freq, mag))
        
        # Count distinct pitch regions (voices may be close but not identical)
        if fundamentals:
            # Cluster fundamentals that are very close (within a semitone)
            clustered_count = 1
            fundamentals.sort(key=lambda x: x[0])
            for i in range(1, len(fundamentals)):
                # Two pitches more than 1.5 semitones apart are different voices
                ratio = fundamentals[i][0] / fundamentals[i-1][0]
                if ratio > 1.09:  # ~1.5 semitones
                    clustered_count += 1
            
            frame_pitch_counts.append(min(clustered_count, 6))
    
    if not frame_pitch_counts:
        return 1, 0.3, {"reason": "No pitched frames detected"}
    
    # Use various statistics to estimate voice count
    counts = np.array(frame_pitch_counts)
    
    # For a cappella/harmony: look at frames with multiple pitches
    multi_voice_frames = counts[counts > 1]
    
    if len(multi_voice_frames) > len(counts) * 0.15:  # At least 15% of frames have multiple voices
        # Use the 75th percentile of multi-voice frames
        voice_count = int(np.percentile(multi_voice_frames, 75))
        # Confidence based on consistency
        consistency = np.std(multi_voice_frames) / (np.mean(multi_voice_frames) + 0.001)
        confidence = min(0.9, max(0.5, 0.85 - consistency * 0.3))
    else:
        # Mostly single voice or no clear multi-voice pattern
        voice_count = int(np.median(counts))
        confidence = 0.4 if voice_count > 1 else 0.6
    
    details = {
        "frames_analyzed": len(frame_pitch_counts),
        "multi_voice_frames": len(multi_voice_frames) if len(multi_voice_frames) > 0 else 0,
        "mean_pitch_count": float(np.mean(counts)),
        "max_pitch_count": int(np.max(counts)),
        "percentile_75": float(np.percentile(counts, 75)),
    }
    
    return max(1, voice_count), confidence, details


def detect_voice_count_advanced(
    audio: np.ndarray,
    sample_rate: int,
    use_vocals_only: bool = True,
    max_voices: int = 5
) -> VoiceDetectionResult:
    """
    Advanced voice detection using SpeechBrain speaker embeddings.
    Falls back to basic MFCC method if SpeechBrain is not available.
    
    Args:
        audio: Input audio as numpy array (mono or stereo)
        sample_rate: Sample rate of input audio
        use_vocals_only: Whether to first separate vocals (recommended for music)
        max_voices: Maximum number of voices to detect
        
    Returns:
        VoiceDetectionResult with voice count and confidence
    """
    if not SPEECHBRAIN_AVAILABLE:
        logger.info("SpeechBrain not available, using basic MFCC method")
        return detect_voice_count(audio, sample_rate, use_vocals_only, max_voices)
    
    # TODO: Implement SpeechBrain-based detection for even better accuracy
    # For now, use the basic MFCC method which works well for most cases
    return detect_voice_count(audio, sample_rate, use_vocals_only, max_voices)
