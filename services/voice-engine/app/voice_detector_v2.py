"""
Voice Count Detector Module v2

Uses spectral and harmonic analysis to detect multiple simultaneous voices.

Key insight: Multiple singers create:
1. Wider spectral bandwidth (voices spread across more frequencies)
2. More simultaneous pitch tracks
3. Greater spectral flux/complexity
4. Richer harmonic content (intervals like thirds, fifths create distinctive patterns)

This approach works better for:
- A cappella groups (Billy Joel "For the Longest Time")
- Harmonies and backup vocals
- Duets (Simon & Garfunkel)
"""

import os
import logging
import tempfile
from typing import Tuple, List, Dict, Optional
import numpy as np
import librosa
from dataclasses import dataclass
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class VoiceDetectionResult:
    """Result of voice detection analysis"""
    voice_count: int  # Number of distinct voices detected
    confidence: float  # Confidence score (0-1)
    method: str  # Method used for detection
    details: Dict  # Additional details about detection


def detect_multiple_pitches_frame(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    min_freq: float = 80,
    max_freq: float = 1000,
    threshold_db: float = -40,
    min_distance_hz: float = 50
) -> List[float]:
    """
    Detect multiple pitch peaks in a single frame's spectrum.
    
    Uses peak detection with:
    - Minimum distance between peaks (to avoid harmonics of same voice)
    - Prominence threshold
    """
    # Limit to vocal range
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    limited_spectrum = spectrum[mask]
    limited_freqs = freqs[mask]
    
    if len(limited_spectrum) == 0:
        return []
    
    # Convert to dB
    spectrum_db = 20 * np.log10(limited_spectrum + 1e-10)
    max_db = spectrum_db.max()
    
    # Only consider peaks above threshold
    threshold = max_db + threshold_db
    
    # Find peaks with minimum distance (in Hz)
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1
    min_distance_bins = max(1, int(min_distance_hz / freq_resolution))
    
    peaks = []
    for i in range(1, len(spectrum_db) - 1):
        if spectrum_db[i] >= threshold:
            # Is it a local maximum?
            is_peak = True
            for j in range(max(0, i - min_distance_bins), min(len(spectrum_db), i + min_distance_bins + 1)):
                if j != i and spectrum_db[j] > spectrum_db[i]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(limited_freqs[i])
    
    return peaks


def estimate_spectral_complexity(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict:
    """
    Measure spectral complexity indicators that suggest multiple voices.
    
    Multiple voices singing together create:
    - Higher spectral flatness (more distributed energy)
    - Wider spectral bandwidth
    - Higher spectral centroid variance
    - More spectral flux
    """
    # Compute spectrogram
    D = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    
    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(S=D, sr=sr)[0]
    
    # Spectral bandwidth (spread)
    bandwidth = librosa.feature.spectral_bandwidth(S=D, sr=sr)[0]
    
    # Spectral flatness (noise-like vs tonal)
    flatness = librosa.feature.spectral_flatness(S=D)[0]
    
    # Spectral rolloff (where most energy is)
    rolloff = librosa.feature.spectral_rolloff(S=D, sr=sr, roll_percent=0.85)[0]
    
    # Spectral flux (rate of change)
    flux = np.sqrt(np.sum(np.diff(D, axis=1)**2, axis=0))
    
    return {
        'centroid_mean': float(np.mean(centroid)),
        'centroid_std': float(np.std(centroid)),
        'bandwidth_mean': float(np.mean(bandwidth)),
        'bandwidth_std': float(np.std(bandwidth)),
        'flatness_mean': float(np.mean(flatness)),
        'rolloff_mean': float(np.mean(rolloff)),
        'flux_mean': float(np.mean(flux)),
        'flux_std': float(np.std(flux)),
    }


def count_simultaneous_pitches_piptrack(
    audio: np.ndarray,
    sr: int,
    fmin: float = 80,
    fmax: float = 500,
    threshold: float = 0.1
) -> Dict:
    """
    Use librosa's piptrack to find multiple simultaneous pitches.
    
    piptrack returns ALL detected pitches per frame, not just the dominant one.
    This is key for detecting harmonies.
    """
    # Get pitch tracks - piptrack returns all detected pitches, not just one
    pitches, magnitudes = librosa.piptrack(
        y=audio, 
        sr=sr, 
        fmin=fmin, 
        fmax=fmax,
        threshold=threshold
    )
    
    # For each frame, count how many distinct pitches are detected
    pitch_counts = []
    
    for frame_idx in range(pitches.shape[1]):
        frame_pitches = pitches[:, frame_idx]
        frame_mags = magnitudes[:, frame_idx]
        
        # Get pitches with significant magnitude
        significant = frame_pitches[frame_mags > threshold * frame_mags.max()] if frame_mags.max() > 0 else []
        significant = significant[significant > 0]  # Remove zeros
        
        if len(significant) > 0:
            # Group nearby pitches (within 10% of each other - same voice/harmonics)
            significant = np.sort(significant)
            groups = []
            current_group = [significant[0]]
            
            for pitch in significant[1:]:
                # If pitch is close to the average of current group, add it
                if pitch / np.mean(current_group) < 1.15:  # Within 15%
                    current_group.append(pitch)
                else:
                    groups.append(np.mean(current_group))
                    current_group = [pitch]
            groups.append(np.mean(current_group))
            
            pitch_counts.append(len(groups))
        else:
            pitch_counts.append(0)
    
    pitch_counts = np.array(pitch_counts)
    voiced_frames = pitch_counts > 0
    
    if np.sum(voiced_frames) > 0:
        # Get statistics from voiced frames only
        voiced_counts = pitch_counts[voiced_frames]
        return {
            'max_simultaneous': int(np.max(voiced_counts)),
            'mean_simultaneous': float(np.mean(voiced_counts)),
            'median_simultaneous': float(np.median(voiced_counts)),
            'percentile_90': float(np.percentile(voiced_counts, 90)),
            'frames_with_multiple': float(np.mean(voiced_counts > 1)),
            'total_voiced_frames': int(np.sum(voiced_frames))
        }
    
    return {
        'max_simultaneous': 1,
        'mean_simultaneous': 1,
        'median_simultaneous': 1,
        'percentile_90': 1,
        'frames_with_multiple': 0,
        'total_voiced_frames': 0
    }


def analyze_harmonic_intervals(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    hop_length: int = 512
) -> Dict:
    """
    Analyze harmonic intervals in the audio.
    
    Multiple voices singing in harmony create distinctive intervals:
    - Minor/Major thirds (1.2 and 1.25 frequency ratio)
    - Perfect fifths (1.5 ratio)
    - Octaves (2.0 ratio)
    
    A solo voice with its own harmonics will have integer ratios (2, 3, 4...)
    Multiple voices create more complex ratio patterns.
    """
    # Compute chroma (pitch class distribution)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    
    # For each frame, count how many pitch classes are active
    pitch_class_counts = []
    for frame in range(chroma.shape[1]):
        active = np.sum(chroma[:, frame] > 0.5 * chroma[:, frame].max())
        pitch_class_counts.append(active)
    
    pitch_class_counts = np.array(pitch_class_counts)
    
    # Also compute harmonic-percussive separation to isolate harmonic content
    harmonic = librosa.effects.harmonic(audio)
    
    # Get harmonic pitch content
    harm_chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop_length)
    
    return {
        'mean_active_pitch_classes': float(np.mean(pitch_class_counts)),
        'max_active_pitch_classes': int(np.max(pitch_class_counts)),
        'std_active_pitch_classes': float(np.std(pitch_class_counts)),
        'frames_with_harmony': float(np.mean(pitch_class_counts >= 3)),
        'chroma_energy_std': float(np.std(harm_chroma.mean(axis=1))),
    }


def detect_voice_count_v2(
    audio: np.ndarray,
    sr: int,
    max_voices: int = 6
) -> VoiceDetectionResult:
    """
    Detect number of simultaneous voices using multiple methods.
    
    Strategy:
    1. Use piptrack to find simultaneous pitch tracks
    2. Analyze spectral complexity
    3. Analyze harmonic intervals/chroma
    4. Combine evidence to estimate voice count
    """
    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=-1) if audio.shape[-1] == 2 else audio[0]
    
    # Ensure float32
    audio = audio.astype(np.float32)
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    logger.info(f"Analyzing audio: {len(audio)/sr:.1f}s at {sr}Hz")
    
    # 1. Pitch track analysis
    pitch_info = count_simultaneous_pitches_piptrack(audio, sr)
    logger.info(f"Pitch analysis: {pitch_info}")
    
    # 2. Spectral complexity
    spectral = estimate_spectral_complexity(audio, sr)
    logger.info(f"Spectral analysis: {spectral}")
    
    # 3. Harmonic/interval analysis
    harmonic = analyze_harmonic_intervals(audio, sr)
    logger.info(f"Harmonic analysis: {harmonic}")
    
    # Combine evidence to estimate voice count
    # Pitch track evidence is most reliable when it finds multiple pitches
    
    estimates = []
    weights = []
    
    # Pitch track evidence (most direct) - STRONGEST indicator
    if pitch_info['frames_with_multiple'] > 0.3:
        # Strong evidence of multiple simultaneous pitches
        # Use max or 90th percentile depending on how consistent it is
        pitch_estimate = min(max_voices, max(2, int(round(pitch_info['percentile_90']))))
        estimates.append(pitch_estimate)
        weights.append(0.6)  # High weight when confident
        logger.info(f"Pitch estimate: {pitch_estimate} voices (90th percentile: {pitch_info['percentile_90']:.1f}, {pitch_info['frames_with_multiple']:.0%} frames)")
    elif pitch_info['frames_with_multiple'] > 0.1:
        pitch_estimate = min(max_voices, max(1, int(round(pitch_info['mean_simultaneous']))))
        estimates.append(pitch_estimate)
        weights.append(0.4)
        logger.info(f"Pitch estimate: {pitch_estimate} voices (mean: {pitch_info['mean_simultaneous']:.1f})")
    else:
        estimates.append(1)
        weights.append(0.2)
    
    # Chroma evidence
    # Multiple voices = more active pitch classes
    if harmonic['mean_active_pitch_classes'] > 4:
        chroma_estimate = min(max_voices, max(2, int(harmonic['mean_active_pitch_classes'] / 2)))
    elif harmonic['mean_active_pitch_classes'] > 2.5:
        chroma_estimate = 2
    else:
        chroma_estimate = 1
    estimates.append(chroma_estimate)
    weights.append(0.2)  # Lower weight - less reliable than pitch
    logger.info(f"Chroma estimate: {chroma_estimate} voices (mean active: {harmonic['mean_active_pitch_classes']:.1f})")
    
    # Spectral bandwidth evidence
    # Wider bandwidth = more voices
    # Typical vocal bandwidth is 500-2000 Hz for single voice
    if spectral['bandwidth_mean'] > 2500:
        bandwidth_estimate = min(max_voices, 3)
    elif spectral['bandwidth_mean'] > 1800:
        bandwidth_estimate = 2
    else:
        bandwidth_estimate = 1
    estimates.append(bandwidth_estimate)
    weights.append(0.2)  # Lower weight - less reliable than pitch
    logger.info(f"Bandwidth estimate: {bandwidth_estimate} voices (mean: {spectral['bandwidth_mean']:.0f}Hz)")
    
    # Weighted average
    weights = np.array(weights)
    estimates = np.array(estimates)
    weighted_estimate = np.sum(estimates * weights) / np.sum(weights)
    
    # Round to integer, but bias toward detecting multiple voices if evidence is mixed
    final_estimate = int(round(weighted_estimate + 0.25))  # Slight bias upward
    final_estimate = max(1, min(max_voices, final_estimate))
    
    # Confidence based on agreement between methods
    estimate_variance = np.var(estimates)
    base_confidence = 1.0 / (1.0 + estimate_variance)  # Higher agreement = higher confidence
    
    # Boost confidence if we have strong evidence
    if pitch_info['frames_with_multiple'] > 0.3 and harmonic['mean_active_pitch_classes'] > 3:
        confidence = min(1.0, base_confidence + 0.2)
    else:
        confidence = base_confidence
    
    logger.info(f"Final estimate: {final_estimate} voices with {confidence:.2f} confidence")
    
    return VoiceDetectionResult(
        voice_count=final_estimate,
        confidence=round(confidence, 2),
        method='spectral_harmonic_v2',
        details={
            'pitch_analysis': pitch_info,
            'spectral_analysis': spectral,
            'harmonic_analysis': harmonic,
            'individual_estimates': {
                'pitch': estimates[0],
                'chroma': estimates[1] if len(estimates) > 1 else None,
                'bandwidth': estimates[2] if len(estimates) > 2 else None,
            },
            'weighted_estimate': round(weighted_estimate, 2)
        }
    )


# Main function that wraps everything
def detect_voice_count(
    audio: np.ndarray,
    sr: int,
    use_vocals_only: bool = True,
    max_voices: int = 6
) -> VoiceDetectionResult:
    """
    Main entry point for voice detection.
    
    Args:
        audio: Audio samples (mono or stereo)
        sr: Sample rate
        use_vocals_only: Whether to separate vocals first (recommended for music)
        max_voices: Maximum number of voices to report
    
    Returns:
        VoiceDetectionResult with estimated voice count
    """
    # Ensure mono
    if len(audio.shape) > 1:
        if audio.shape[0] == 2:
            audio = audio.mean(axis=0)
        elif audio.shape[-1] == 2:
            audio = audio.mean(axis=-1)
    
    # If vocals_only requested, try to separate
    if use_vocals_only:
        try:
            # Import vocal separator
            from app.vocal_separator import VocalSeparator
            
            separator = VocalSeparator()
            if separator.is_available():
                logger.info("Separating vocals before voice detection...")
                
                # Ensure audio is in correct format
                audio_float = audio.astype(np.float32)
                if np.max(np.abs(audio_float)) > 1.0:
                    audio_float = audio_float / 32768.0  # Assume int16
                
                # Use HP2_all_vocals to get ALL vocals including backups
                vocals, _ = separator.separate(audio_float, sr, model_name="HP2_all_vocals")
                
                # Normalize
                if np.max(np.abs(vocals)) > 0:
                    vocals = vocals / np.max(np.abs(vocals))
                
                logger.info(f"Using separated vocals for analysis ({len(vocals)/sr:.1f}s)")
                audio = vocals
            else:
                logger.warning("Vocal separator not available, analyzing full mix")
        except Exception as e:
            logger.warning(f"Vocal separation failed: {e}, analyzing full mix")
    
    # Run detection
    return detect_voice_count_v2(audio, sr, max_voices)
