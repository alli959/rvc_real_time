"""
Voice Count Detector - Detects multiple simultaneous voices in audio.

Uses spectral and harmonic analysis for detection:
1. Wider spectral bandwidth (voices spread across more frequencies)
2. More simultaneous pitch tracks
3. Greater spectral flux/complexity
4. Richer harmonic content (intervals like thirds, fifths)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import librosa

logger = logging.getLogger(__name__)


@dataclass
class VoiceDetectionResult:
    """Result of voice detection analysis."""
    voice_count: int  # Number of distinct voices detected
    confidence: float  # Confidence score (0-1)
    method: str  # Method used for detection
    details: Optional[Dict] = None  # Additional detection details


def estimate_spectral_complexity(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict:
    """
    Measure spectral complexity indicators that suggest multiple voices.
    """
    D = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    
    centroid = librosa.feature.spectral_centroid(S=D, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=D, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=D)[0]
    rolloff = librosa.feature.spectral_rolloff(S=D, sr=sr, roll_percent=0.85)[0]
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


def count_simultaneous_pitches(
    audio: np.ndarray,
    sr: int,
    fmin: float = 80,
    fmax: float = 500,
    threshold: float = 0.1
) -> Dict:
    """
    Use librosa's piptrack to find multiple simultaneous pitches.
    """
    pitches, magnitudes = librosa.piptrack(
        y=audio, sr=sr, fmin=fmin, fmax=fmax, threshold=threshold
    )
    
    pitch_counts = []
    
    for frame_idx in range(pitches.shape[1]):
        frame_pitches = pitches[:, frame_idx]
        frame_mags = magnitudes[:, frame_idx]
        
        # Get pitches with significant magnitude
        significant = frame_pitches[frame_mags > threshold * frame_mags.max()] if frame_mags.max() > 0 else []
        significant = significant[significant > 0]
        
        if len(significant) > 0:
            significant = np.sort(significant)
            groups = []
            current_group = [significant[0]]
            
            for pitch in significant[1:]:
                if pitch / np.mean(current_group) < 1.15:
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
    hop_length: int = 512
) -> Dict:
    """
    Analyze harmonic intervals in the audio.
    
    Multiple voices singing in harmony create distinctive intervals:
    - Minor/Major thirds (1.2 and 1.25 frequency ratio)
    - Perfect fifths (1.5 ratio)
    """
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    
    pitch_class_counts = []
    for frame in range(chroma.shape[1]):
        active = np.sum(chroma[:, frame] > 0.5 * chroma[:, frame].max())
        pitch_class_counts.append(active)
    
    pitch_class_counts = np.array(pitch_class_counts)
    harmonic = librosa.effects.harmonic(audio)
    harm_chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop_length)
    
    return {
        'mean_active_pitch_classes': float(np.mean(pitch_class_counts)),
        'max_active_pitch_classes': int(np.max(pitch_class_counts)),
        'std_active_pitch_classes': float(np.std(pitch_class_counts)),
        'frames_with_harmony': float(np.mean(pitch_class_counts >= 3)),
        'chroma_energy_std': float(np.std(harm_chroma.mean(axis=1))),
    }


def detect_voice_count(
    audio: np.ndarray,
    sr: int,
    use_vocals_only: bool = True,
    max_voices: int = 6
) -> VoiceDetectionResult:
    """
    Detect number of simultaneous voices in audio.
    
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
    
    audio = audio.astype(np.float32)
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    logger.info(f"Analyzing audio: {len(audio)/sr:.1f}s at {sr}Hz")
    
    # Optional vocal separation
    if use_vocals_only:
        try:
            from app.services.audio_analysis.vocal_separator import separate_vocals
            logger.info("Separating vocals before voice detection...")
            vocals, _ = separate_vocals(audio, sr, model_name="HP3_all_vocals")
            if np.max(np.abs(vocals)) > 0:
                vocals = vocals / np.max(np.abs(vocals))
            audio = vocals
            logger.info(f"Using separated vocals for analysis")
        except Exception as e:
            logger.warning(f"Vocal separation failed: {e}, analyzing full mix")
    
    # Analysis
    pitch_info = count_simultaneous_pitches(audio, sr)
    spectral = estimate_spectral_complexity(audio, sr)
    harmonic = analyze_harmonic_intervals(audio, sr)
    
    # Combine evidence
    estimates = []
    weights = []
    
    # Pitch track evidence (most direct)
    if pitch_info['frames_with_multiple'] > 0.3:
        pitch_estimate = min(max_voices, max(2, int(round(pitch_info['percentile_90']))))
        estimates.append(pitch_estimate)
        weights.append(0.6)
    elif pitch_info['frames_with_multiple'] > 0.1:
        pitch_estimate = min(max_voices, max(1, int(round(pitch_info['mean_simultaneous']))))
        estimates.append(pitch_estimate)
        weights.append(0.4)
    else:
        estimates.append(1)
        weights.append(0.2)
    
    # Chroma evidence
    if harmonic['mean_active_pitch_classes'] > 4:
        chroma_estimate = min(max_voices, max(2, int(harmonic['mean_active_pitch_classes'] / 2)))
    elif harmonic['mean_active_pitch_classes'] > 2.5:
        chroma_estimate = 2
    else:
        chroma_estimate = 1
    estimates.append(chroma_estimate)
    weights.append(0.2)
    
    # Bandwidth evidence
    if spectral['bandwidth_mean'] > 2500:
        bandwidth_estimate = min(max_voices, 3)
    elif spectral['bandwidth_mean'] > 1800:
        bandwidth_estimate = 2
    else:
        bandwidth_estimate = 1
    estimates.append(bandwidth_estimate)
    weights.append(0.2)
    
    # Weighted average
    weights = np.array(weights)
    estimates = np.array(estimates)
    weighted_estimate = np.sum(estimates * weights) / np.sum(weights)
    
    final_estimate = int(round(weighted_estimate + 0.25))
    final_estimate = max(1, min(max_voices, final_estimate))
    
    # Confidence
    estimate_variance = np.var(estimates)
    base_confidence = 1.0 / (1.0 + estimate_variance)
    
    if pitch_info['frames_with_multiple'] > 0.3 and harmonic['mean_active_pitch_classes'] > 3:
        confidence = min(1.0, base_confidence + 0.2)
    else:
        confidence = base_confidence
    
    logger.info(f"Final estimate: {final_estimate} voices with {confidence:.2f} confidence")
    
    return VoiceDetectionResult(
        voice_count=final_estimate,
        confidence=round(confidence, 2),
        method='spectral_harmonic',
        details={
            'pitch_analysis': pitch_info,
            'spectral_analysis': spectral,
            'harmonic_analysis': harmonic,
            'individual_estimates': {
                'pitch': int(estimates[0]),
                'chroma': int(estimates[1]) if len(estimates) > 1 else None,
                'bandwidth': int(estimates[2]) if len(estimates) > 2 else None,
            },
            'weighted_estimate': round(weighted_estimate, 2)
        }
    )
