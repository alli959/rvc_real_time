"""
Phoneme Analyzer Module

Analyzes audio for phoneme coverage to determine language readiness.
Supports English (en) and Icelandic (is) languages.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Language Phoneme Definitions
# ============================================================================

# English phoneme set (IPA) - 44 phonemes
ENGLISH_PHONEMES = {
    # Consonants - Plosives
    "p", "b", "t", "d", "k", "g",
    # Consonants - Affricates
    "tʃ", "dʒ",
    # Consonants - Fricatives
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    # Consonants - Nasals
    "m", "n", "ŋ",
    # Consonants - Liquids
    "l", "r",
    # Consonants - Glides
    "w", "j",
    # Vowels - Monophthongs
    "iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ʌ",
    # Vowels - Mid
    "ə", "ɜː",
    # Vowels - Diphthongs
    "eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "ɪə", "eə", "ʊə"
}

# English consonants and vowels separately for analysis
ENGLISH_CONSONANTS = {
    "p", "b", "t", "d", "k", "g",
    "tʃ", "dʒ",
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "m", "n", "ŋ",
    "l", "r",
    "w", "j"
}

ENGLISH_VOWELS = {
    "iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ʌ",
    "ə", "ɜː",
    "eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "ɪə", "eə", "ʊə"
}

# Icelandic phoneme set (IPA) - 33 phonemes
ICELANDIC_PHONEMES = {
    # Consonants - Plosives (unaspirated)
    "p", "t", "c", "k",
    # Consonants - Plosives (aspirated) - represented with superscript h
    "pʰ", "tʰ", "cʰ", "kʰ",
    # Consonants - Fricatives
    "f", "v", "θ", "ð", "s", "ç", "x", "h",
    # Consonants - Nasals
    "m", "n", "ɲ", "ŋ",
    # Consonants - Liquids
    "l", "r",
    # Vowels
    "iː", "ɪ", "eː", "ɛ", "aː", "a", "ɔː", "ɔ", "uː", "ʏ", "œ",
    # Diphthongs
    "ei", "au", "ou", "ai"
}

ICELANDIC_CONSONANTS = {
    "p", "t", "c", "k",
    "pʰ", "tʰ", "cʰ", "kʰ",
    "f", "v", "θ", "ð", "s", "ç", "x", "h",
    "m", "n", "ɲ", "ŋ",
    "l", "r"
}

ICELANDIC_VOWELS = {
    "iː", "ɪ", "eː", "ɛ", "aː", "a", "ɔː", "ɔ", "uː", "ʏ", "œ",
    "ei", "au", "ou", "ai"
}

# Mapping of language codes to phoneme sets
LANGUAGE_PHONEMES = {
    "en": ENGLISH_PHONEMES,
    "is": ICELANDIC_PHONEMES,
}

LANGUAGE_CONSONANTS = {
    "en": ENGLISH_CONSONANTS,
    "is": ICELANDIC_CONSONANTS,
}

LANGUAGE_VOWELS = {
    "en": ENGLISH_VOWELS,
    "is": ICELANDIC_VOWELS,
}


# ============================================================================
# Phoneme Analyzer Data Classes
# ============================================================================

@dataclass
class PhonemeCount:
    """Count of a single phoneme's occurrences"""
    phoneme: str
    count: int
    frequency: float  # Percentage of total phonemes
    
    def to_dict(self) -> dict:
        return {
            "phoneme": self.phoneme,
            "count": int(self.count),
            "frequency": float(round(self.frequency, 2))
        }


@dataclass
class PhonemeCoverageReport:
    """Report of phoneme coverage for a language"""
    language: str
    total_phonemes_found: int
    unique_phonemes_found: int
    target_phonemes: int
    coverage_percentage: float
    
    # Detailed breakdowns
    found_phonemes: Set[str]
    missing_phonemes: Set[str]
    weak_phonemes: Set[str]  # Present but low frequency
    
    # Counts
    phoneme_counts: Dict[str, PhonemeCount]
    
    # Category breakdowns
    consonant_coverage: float
    vowel_coverage: float
    
    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "total_phonemes_found": int(self.total_phonemes_found),
            "unique_phonemes_found": int(self.unique_phonemes_found),
            "target_phonemes": int(self.target_phonemes),
            "coverage_percentage": float(round(self.coverage_percentage, 2)),
            "found_phonemes": sorted(list(self.found_phonemes)),
            "missing_phonemes": sorted(list(self.missing_phonemes)),
            "weak_phonemes": sorted(list(self.weak_phonemes)),
            "consonant_coverage": float(round(self.consonant_coverage, 2)),
            "vowel_coverage": float(round(self.vowel_coverage, 2)),
            "phoneme_distribution": {
                k: v.to_dict() for k, v in self.phoneme_counts.items()
            }
        }


@dataclass
class AudioQualityMetrics:
    """Audio quality measurements"""
    snr_db: float  # Signal-to-noise ratio
    rms_energy: float  # Average energy
    peak_amplitude: float
    silence_percentage: float
    clipping_percentage: float
    duration_seconds: float
    sample_rate: int
    
    def to_dict(self) -> dict:
        return {
            "snr_db": float(round(self.snr_db, 2)),
            "rms_energy": float(round(self.rms_energy, 4)),
            "peak_amplitude": float(round(self.peak_amplitude, 4)),
            "silence_percentage": float(round(self.silence_percentage, 2)),
            "clipping_percentage": float(round(self.clipping_percentage, 2)),
            "duration_seconds": float(round(self.duration_seconds, 2)),
            "sample_rate": int(self.sample_rate)
        }


@dataclass
class PitchAnalysis:
    """F0/pitch analysis results"""
    f0_mean: float
    f0_std: float
    f0_min: float
    f0_max: float
    f0_range_octaves: float
    voiced_percentage: float
    
    def to_dict(self) -> dict:
        return {
            "f0_mean": float(round(self.f0_mean, 2)),
            "f0_std": float(round(self.f0_std, 2)),
            "f0_min": float(round(self.f0_min, 2)),
            "f0_max": float(round(self.f0_max, 2)),
            "f0_range_octaves": float(round(self.f0_range_octaves, 2)),
            "voiced_percentage": float(round(self.voiced_percentage, 2))
        }


@dataclass
class LanguageReadinessScore:
    """Overall language readiness assessment"""
    language: str
    overall_score: float  # 0-100
    
    # Component scores (0-100)
    phoneme_coverage_score: float      # 35% weight
    vowel_variation_score: float       # 20% weight
    pitch_variation_score: float       # 15% weight
    speaking_rate_score: float         # 10% weight
    prosody_quality_score: float       # 10% weight
    audio_quality_score: float         # 10% weight
    
    # Details
    phoneme_report: Optional[PhonemeCoverageReport] = None
    pitch_analysis: Optional[PitchAnalysis] = None
    audio_quality: Optional[AudioQualityMetrics] = None
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    suggested_prompts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "overall_score": float(round(self.overall_score, 2)),
            "component_scores": {
                "phoneme_coverage": float(round(self.phoneme_coverage_score, 2)),
                "vowel_variation": float(round(self.vowel_variation_score, 2)),
                "pitch_variation": float(round(self.pitch_variation_score, 2)),
                "speaking_rate": float(round(self.speaking_rate_score, 2)),
                "prosody_quality": float(round(self.prosody_quality_score, 2)),
                "audio_quality": float(round(self.audio_quality_score, 2))
            },
            "weights": {
                "phoneme_coverage": 0.35,
                "vowel_variation": 0.20,
                "pitch_variation": 0.15,
                "speaking_rate": 0.10,
                "prosody_quality": 0.10,
                "audio_quality": 0.10
            },
            "phoneme_report": self.phoneme_report.to_dict() if self.phoneme_report else None,
            "pitch_analysis": self.pitch_analysis.to_dict() if self.pitch_analysis else None,
            "audio_quality": self.audio_quality.to_dict() if self.audio_quality else None,
            "recommendations": self.recommendations,
            "suggested_prompts": self.suggested_prompts[:10]  # Top 10
        }


# ============================================================================
# Phoneme Analyzer Class
# ============================================================================

class PhonemeAnalyzer:
    """
    Analyzes audio transcriptions for phoneme coverage.
    
    Uses phonemizer library with espeak-ng backend for IPA transcription.
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._phonemizer = None
        self._whisper_model = None
        
    def _get_phonemizer(self, language: str):
        """Lazy-load phonemizer for given language"""
        try:
            from phonemizer import phonemize
            from phonemizer.backend import EspeakBackend
            
            # Map language codes to espeak codes
            espeak_lang_map = {
                "en": "en-us",
                "is": "is"  # Icelandic
            }
            
            return lambda text: phonemize(
                text,
                language=espeak_lang_map.get(language, language),
                backend="espeak",
                strip=True,
                preserve_punctuation=False,
                with_stress=False
            )
        except ImportError:
            logger.warning("phonemizer not installed. Using fallback.")
            return self._fallback_phonemize
        except Exception as e:
            logger.warning(f"phonemizer error: {e}. Using fallback.")
            return self._fallback_phonemize
    
    def _fallback_phonemize(self, text: str) -> str:
        """
        Fallback phonemization using basic rules.
        Not as accurate as espeak but works without dependencies.
        """
        # Very basic English approximation
        text = text.lower()
        
        # Basic mappings (simplified)
        mappings = [
            ("th", "θ"),
            ("sh", "ʃ"),
            ("ch", "tʃ"),
            ("ng", "ŋ"),
            ("ph", "f"),
            ("wh", "w"),
            ("ee", "iː"),
            ("oo", "uː"),
            ("ou", "aʊ"),
            ("oi", "ɔɪ"),
            ("ai", "aɪ"),
            ("ea", "iː"),
        ]
        
        result = text
        for pattern, replacement in mappings:
            result = result.replace(pattern, replacement)
            
        return result
    
    def _get_whisper(self):
        """Lazy-load Whisper for transcription"""
        if self._whisper_model is None:
            try:
                import torch
                import whisper
                # Always use CUDA if available for better performance
                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info("Loading Whisper model on GPU")
                else:
                    device = "cpu"
                    logger.info("Loading Whisper model on CPU (no GPU available)")
                self._whisper_model = whisper.load_model("base", device=device)
            except ImportError:
                logger.error("whisper not installed")
                return None
        return self._whisper_model
    
    def transcribe_audio(
        self, 
        audio_path: str,
        language: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language hint (en, is)
            
        Returns:
            Tuple of (transcription, detected_language)
        """
        whisper = self._get_whisper()
        if whisper is None:
            return "", "unknown"
            
        try:
            result = whisper.transcribe(
                audio_path,
                language=language,
                task="transcribe"
            )
            return result["text"], result.get("language", language or "en")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", "unknown"
    
    def text_to_phonemes(
        self, 
        text: str, 
        language: str = "en"
    ) -> List[str]:
        """
        Convert text to list of phonemes.
        
        Args:
            text: Input text
            language: Language code (en, is)
            
        Returns:
            List of IPA phonemes
        """
        phonemizer = self._get_phonemizer(language)
        ipa_string = phonemizer(text)
        
        # Parse IPA string into individual phonemes
        phonemes = self._parse_ipa_string(ipa_string, language)
        return phonemes
    
    def _parse_ipa_string(
        self, 
        ipa_string: str, 
        language: str
    ) -> List[str]:
        """
        Parse IPA string into list of phonemes.
        Handles multi-character phonemes like 'tʃ', 'dʒ', etc.
        """
        target_phonemes = LANGUAGE_PHONEMES.get(language, ENGLISH_PHONEMES)
        
        phonemes = []
        i = 0
        ipa_string = ipa_string.replace(" ", "")
        
        while i < len(ipa_string):
            # Try matching longest phonemes first (2-char, then 1-char)
            matched = False
            
            for length in [3, 2, 1]:
                if i + length <= len(ipa_string):
                    candidate = ipa_string[i:i + length]
                    if candidate in target_phonemes:
                        phonemes.append(candidate)
                        i += length
                        matched = True
                        break
            
            if not matched:
                # Skip unknown characters
                i += 1
        
        return phonemes
    
    def analyze_phoneme_coverage(
        self,
        phonemes: List[str],
        language: str = "en",
        weak_threshold: float = 0.01  # Less than 1% is "weak"
    ) -> PhonemeCoverageReport:
        """
        Analyze phoneme coverage for a language.
        
        Args:
            phonemes: List of phonemes from transcribed audio
            language: Target language (en, is)
            weak_threshold: Frequency below this is considered "weak"
            
        Returns:
            PhonemeCoverageReport with coverage analysis
        """
        target_phonemes = LANGUAGE_PHONEMES.get(language, ENGLISH_PHONEMES)
        target_consonants = LANGUAGE_CONSONANTS.get(language, ENGLISH_CONSONANTS)
        target_vowels = LANGUAGE_VOWELS.get(language, ENGLISH_VOWELS)
        
        # Count phonemes
        counts = Counter(phonemes)
        total_count = sum(counts.values())
        
        # Build phoneme counts
        phoneme_counts = {}
        for phoneme, count in counts.items():
            if phoneme in target_phonemes:
                freq = count / total_count if total_count > 0 else 0
                phoneme_counts[phoneme] = PhonemeCount(
                    phoneme=phoneme,
                    count=count,
                    frequency=freq * 100
                )
        
        # Determine found, missing, weak
        found_phonemes = set(phoneme_counts.keys())
        missing_phonemes = target_phonemes - found_phonemes
        weak_phonemes = {
            p for p, pc in phoneme_counts.items() 
            if pc.frequency < weak_threshold * 100
        }
        
        # Calculate coverage percentages
        coverage_pct = len(found_phonemes) / len(target_phonemes) * 100
        
        # Consonant coverage
        found_consonants = found_phonemes & target_consonants
        consonant_coverage = len(found_consonants) / len(target_consonants) * 100
        
        # Vowel coverage
        found_vowels = found_phonemes & target_vowels
        vowel_coverage = len(found_vowels) / len(target_vowels) * 100
        
        return PhonemeCoverageReport(
            language=language,
            total_phonemes_found=total_count,
            unique_phonemes_found=len(found_phonemes),
            target_phonemes=len(target_phonemes),
            coverage_percentage=coverage_pct,
            found_phonemes=found_phonemes,
            missing_phonemes=missing_phonemes,
            weak_phonemes=weak_phonemes,
            phoneme_counts=phoneme_counts,
            consonant_coverage=consonant_coverage,
            vowel_coverage=vowel_coverage
        )
    
    def analyze_audio_file(
        self,
        audio_path: str,
        language: str = "en"
    ) -> PhonemeCoverageReport:
        """
        Full pipeline: transcribe audio and analyze phonemes.
        
        Args:
            audio_path: Path to audio file
            language: Target language
            
        Returns:
            PhonemeCoverageReport
        """
        # Transcribe
        text, detected_lang = self.transcribe_audio(audio_path, language)
        
        if not text:
            logger.warning(f"No transcription for {audio_path}")
            return self._empty_report(language)
        
        # Convert to phonemes
        phonemes = self.text_to_phonemes(text, language)
        
        # Analyze
        return self.analyze_phoneme_coverage(phonemes, language)
    
    def analyze_audio_directory(
        self,
        audio_dir: str,
        language: str = "en",
        extensions: List[str] = [".wav", ".mp3", ".flac", ".ogg"]
    ) -> PhonemeCoverageReport:
        """
        Analyze all audio files in a directory.
        
        Args:
            audio_dir: Directory containing audio files
            language: Target language
            extensions: Audio file extensions to process
            
        Returns:
            Combined PhonemeCoverageReport
        """
        audio_dir = Path(audio_dir)
        all_phonemes = []
        
        for ext in extensions:
            for audio_file in audio_dir.glob(f"**/*{ext}"):
                try:
                    text, _ = self.transcribe_audio(str(audio_file), language)
                    if text:
                        phonemes = self.text_to_phonemes(text, language)
                        all_phonemes.extend(phonemes)
                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {e}")
        
        if not all_phonemes:
            return self._empty_report(language)
        
        return self.analyze_phoneme_coverage(all_phonemes, language)
    
    def _empty_report(self, language: str) -> PhonemeCoverageReport:
        """Create an empty report for when no audio is found"""
        target = LANGUAGE_PHONEMES.get(language, ENGLISH_PHONEMES)
        return PhonemeCoverageReport(
            language=language,
            total_phonemes_found=0,
            unique_phonemes_found=0,
            target_phonemes=len(target),
            coverage_percentage=0.0,
            found_phonemes=set(),
            missing_phonemes=target,
            weak_phonemes=set(),
            phoneme_counts={},
            consonant_coverage=0.0,
            vowel_coverage=0.0
        )


# ============================================================================
# Audio Quality Analyzer
# ============================================================================

class AudioQualityAnalyzer:
    """Analyzes audio quality metrics"""
    
    @staticmethod
    def analyze(
        audio: np.ndarray,
        sample_rate: int,
        silence_threshold: float = 0.01,
        clipping_threshold: float = 0.99
    ) -> AudioQualityMetrics:
        """
        Analyze audio quality.
        
        Args:
            audio: Audio waveform (float32, -1 to 1)
            sample_rate: Sample rate
            silence_threshold: RMS below this is silence
            clipping_threshold: Amplitude above this is clipping
            
        Returns:
            AudioQualityMetrics
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()
        
        # Duration
        duration = len(audio) / sample_rate
        
        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Peak amplitude
        peak = np.max(np.abs(audio))
        
        # Calculate SNR (approximate using noise floor estimation)
        # Use bottom 10th percentile as noise estimate
        sorted_abs = np.sort(np.abs(audio))
        noise_floor = np.mean(sorted_abs[:len(sorted_abs) // 10])
        signal_level = rms
        snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))
        
        # Silence percentage (frame-based)
        frame_size = int(sample_rate * 0.025)  # 25ms frames
        hop_size = int(sample_rate * 0.010)    # 10ms hop
        
        silent_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            frame_rms = np.sqrt(np.mean(frame ** 2))
            if frame_rms < silence_threshold:
                silent_frames += 1
            total_frames += 1
        
        silence_pct = (silent_frames / total_frames * 100) if total_frames > 0 else 0
        
        # Clipping percentage
        clipping_samples = np.sum(np.abs(audio) > clipping_threshold)
        clipping_pct = clipping_samples / len(audio) * 100
        
        return AudioQualityMetrics(
            snr_db=snr_db,
            rms_energy=rms,
            peak_amplitude=peak,
            silence_percentage=silence_pct,
            clipping_percentage=clipping_pct,
            duration_seconds=duration,
            sample_rate=sample_rate
        )


# ============================================================================
# Pitch Analyzer
# ============================================================================

class PitchAnalyzer:
    """Analyzes pitch/F0 characteristics"""
    
    @staticmethod
    def analyze(
        audio: np.ndarray,
        sample_rate: int,
        f0_min: float = 50.0,
        f0_max: float = 800.0
    ) -> PitchAnalysis:
        """
        Analyze pitch characteristics.
        
        Uses parselmouth (Praat) if available, otherwise librosa.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            f0_min: Minimum F0 to consider
            f0_max: Maximum F0 to consider
            
        Returns:
            PitchAnalysis
        """
        try:
            import parselmouth
            return PitchAnalyzer._analyze_parselmouth(
                audio, sample_rate, f0_min, f0_max
            )
        except ImportError:
            return PitchAnalyzer._analyze_librosa(
                audio, sample_rate, f0_min, f0_max
            )
    
    @staticmethod
    def _analyze_parselmouth(
        audio: np.ndarray,
        sample_rate: int,
        f0_min: float,
        f0_max: float
    ) -> PitchAnalysis:
        """Analyze using Praat via parselmouth"""
        import parselmouth
        
        # Create Sound object
        sound = parselmouth.Sound(audio, sample_rate)
        
        # Extract pitch
        pitch = sound.to_pitch(
            pitch_floor=f0_min,
            pitch_ceiling=f0_max
        )
        
        # Get F0 values (exclude unvoiced)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]
        
        if len(f0_values) == 0:
            return PitchAnalysis(
                f0_mean=0, f0_std=0, f0_min=0, f0_max=0,
                f0_range_octaves=0, voiced_percentage=0
            )
        
        # Calculate statistics
        f0_mean = np.mean(f0_values)
        f0_std = np.std(f0_values)
        f0_min_val = np.min(f0_values)
        f0_max_val = np.max(f0_values)
        
        # Range in octaves
        f0_range_octaves = np.log2(f0_max_val / f0_min_val) if f0_min_val > 0 else 0
        
        # Voiced percentage
        total_frames = len(pitch.selected_array['frequency'])
        voiced_frames = len(f0_values)
        voiced_pct = voiced_frames / total_frames * 100 if total_frames > 0 else 0
        
        return PitchAnalysis(
            f0_mean=f0_mean,
            f0_std=f0_std,
            f0_min=f0_min_val,
            f0_max=f0_max_val,
            f0_range_octaves=f0_range_octaves,
            voiced_percentage=voiced_pct
        )
    
    @staticmethod
    def _analyze_librosa(
        audio: np.ndarray,
        sample_rate: int,
        f0_min: float,
        f0_max: float
    ) -> PitchAnalysis:
        """Fallback analysis using librosa"""
        import librosa
        
        # Use pyin for pitch tracking
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=f0_min,
            fmax=f0_max,
            sr=sample_rate
        )
        
        # Filter to voiced only
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) == 0:
            return PitchAnalysis(
                f0_mean=0, f0_std=0, f0_min=0, f0_max=0,
                f0_range_octaves=0, voiced_percentage=0
            )
        
        f0_mean = np.nanmean(f0_voiced)
        f0_std = np.nanstd(f0_voiced)
        f0_min_val = np.nanmin(f0_voiced)
        f0_max_val = np.nanmax(f0_voiced)
        
        f0_range_octaves = np.log2(f0_max_val / f0_min_val) if f0_min_val > 0 else 0
        
        voiced_pct = np.sum(voiced_flag) / len(voiced_flag) * 100
        
        return PitchAnalysis(
            f0_mean=f0_mean,
            f0_std=f0_std,
            f0_min=f0_min_val,
            f0_max=f0_max_val,
            f0_range_octaves=f0_range_octaves,
            voiced_percentage=voiced_pct
        )


# ============================================================================
# Language Readiness Scorer
# ============================================================================

class LanguageReadinessScorer:
    """
    Calculates overall language readiness score based on multiple factors.
    
    Scoring weights:
    - Phoneme coverage: 35%
    - Vowel variation: 20%
    - Pitch variation: 15%
    - Speaking rate: 10%
    - Prosody quality: 10%
    - Audio quality: 10%
    """
    
    WEIGHTS = {
        "phoneme_coverage": 0.35,
        "vowel_variation": 0.20,
        "pitch_variation": 0.15,
        "speaking_rate": 0.10,
        "prosody_quality": 0.10,
        "audio_quality": 0.10
    }
    
    # Ideal pitch range (in octaves) for good variation
    IDEAL_PITCH_RANGE = 1.5  # About 1.5 octaves
    
    # Ideal speaking rate variation (coefficient of variation)
    IDEAL_RATE_CV = 0.15
    
    def __init__(self, phoneme_analyzer: Optional[PhonemeAnalyzer] = None):
        self.phoneme_analyzer = phoneme_analyzer or PhonemeAnalyzer()
    
    def score(
        self,
        phoneme_report: PhonemeCoverageReport,
        pitch_analysis: PitchAnalysis,
        audio_quality: AudioQualityMetrics,
        speaking_rate_cv: float = 0.1  # Coefficient of variation
    ) -> LanguageReadinessScore:
        """
        Calculate overall language readiness score.
        
        Args:
            phoneme_report: Phoneme coverage analysis
            pitch_analysis: Pitch/F0 analysis
            audio_quality: Audio quality metrics
            speaking_rate_cv: Speaking rate coefficient of variation
            
        Returns:
            LanguageReadinessScore with all components
        """
        # 1. Phoneme coverage score (35%)
        phoneme_score = phoneme_report.coverage_percentage
        
        # 2. Vowel variation score (20%)
        vowel_score = phoneme_report.vowel_coverage
        
        # 3. Pitch variation score (15%)
        # Normalize pitch range to 0-100
        pitch_range_normalized = min(
            pitch_analysis.f0_range_octaves / self.IDEAL_PITCH_RANGE * 100, 
            100
        )
        pitch_score = pitch_range_normalized
        
        # 4. Speaking rate score (10%)
        # Higher variation is better (but cap at ideal)
        rate_score = min(speaking_rate_cv / self.IDEAL_RATE_CV * 100, 100)
        
        # 5. Prosody quality (10%)
        # Based on pitch std and voiced percentage
        prosody_score = min(
            (pitch_analysis.f0_std / 50) * 50 +  # Variation component
            (pitch_analysis.voiced_percentage / 100) * 50,  # Voiced component
            100
        )
        
        # 6. Audio quality score (10%)
        # Based on SNR and low clipping
        snr_component = min(audio_quality.snr_db / 30 * 50, 50)  # Cap at 30dB
        clipping_component = max(0, 50 - audio_quality.clipping_percentage * 10)
        audio_score = snr_component + clipping_component
        
        # Calculate weighted overall score
        overall = (
            phoneme_score * self.WEIGHTS["phoneme_coverage"] +
            vowel_score * self.WEIGHTS["vowel_variation"] +
            pitch_score * self.WEIGHTS["pitch_variation"] +
            rate_score * self.WEIGHTS["speaking_rate"] +
            prosody_score * self.WEIGHTS["prosody_quality"] +
            audio_score * self.WEIGHTS["audio_quality"]
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            phoneme_report, pitch_analysis, audio_quality, overall
        )
        
        # Get suggested prompts for missing phonemes
        suggested_prompts = self._suggest_prompts(phoneme_report)
        
        return LanguageReadinessScore(
            language=phoneme_report.language,
            overall_score=overall,
            phoneme_coverage_score=phoneme_score,
            vowel_variation_score=vowel_score,
            pitch_variation_score=pitch_score,
            speaking_rate_score=rate_score,
            prosody_quality_score=prosody_score,
            audio_quality_score=audio_score,
            phoneme_report=phoneme_report,
            pitch_analysis=pitch_analysis,
            audio_quality=audio_quality,
            recommendations=recommendations,
            suggested_prompts=suggested_prompts
        )
    
    def _generate_recommendations(
        self,
        phoneme_report: PhonemeCoverageReport,
        pitch_analysis: PitchAnalysis,
        audio_quality: AudioQualityMetrics,
        overall_score: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Phoneme recommendations
        if phoneme_report.coverage_percentage < 70:
            recommendations.append(
                f"Low phoneme coverage ({phoneme_report.coverage_percentage:.0f}%). "
                f"Record prompts containing: {', '.join(list(phoneme_report.missing_phonemes)[:5])}"
            )
        
        if phoneme_report.weak_phonemes:
            recommendations.append(
                f"Strengthen weak phonemes: {', '.join(list(phoneme_report.weak_phonemes)[:5])}"
            )
        
        # Vowel recommendations
        if phoneme_report.vowel_coverage < 70:
            recommendations.append(
                f"Improve vowel variety ({phoneme_report.vowel_coverage:.0f}% coverage). "
                "Record sentences with diverse vowel sounds."
            )
        
        # Pitch recommendations
        if pitch_analysis.f0_range_octaves < 0.8:
            recommendations.append(
                "Limited pitch range detected. Try recording with more vocal expression "
                "and varied intonation."
            )
        
        # Audio quality recommendations
        if audio_quality.snr_db < 20:
            recommendations.append(
                f"Low audio quality (SNR: {audio_quality.snr_db:.0f}dB). "
                "Use a better microphone or reduce background noise."
            )
        
        if audio_quality.clipping_percentage > 1:
            recommendations.append(
                "Audio clipping detected. Lower your recording volume."
            )
        
        if audio_quality.silence_percentage > 50:
            recommendations.append(
                "High silence percentage. Trim silent portions from recordings."
            )
        
        # Overall assessment
        if overall_score >= 80:
            recommendations.insert(0, "✓ Model is well-suited for this language!")
        elif overall_score >= 60:
            recommendations.insert(0, "Model has moderate language readiness. Consider improvements.")
        else:
            recommendations.insert(0, "Model needs significant improvement for this language.")
        
        return recommendations
    
    def _suggest_prompts(
        self,
        phoneme_report: PhonemeCoverageReport
    ) -> List[str]:
        """
        Suggest recording prompts to fill phoneme gaps.
        Returns prompts that contain missing/weak phonemes.
        """
        # This will be enhanced when prompt system is implemented
        # For now, return generic suggestions
        
        suggestions = []
        lang = phoneme_report.language
        
        missing = phoneme_report.missing_phonemes | phoneme_report.weak_phonemes
        
        # Basic prompt suggestions for common missing phonemes
        phoneme_prompts = {
            # English
            "θ": "The thick thatch thoughtfully thwarted the thieves.",
            "ð": "This, that, and the other thing.",
            "ʃ": "She sells seashells by the seashore.",
            "ʒ": "The unusual treasure measured pleasure.",
            "ŋ": "The king was singing along.",
            "tʃ": "The cheerful child chose chocolate chip.",
            "dʒ": "The judge enjoyed the gentle giant's joke.",
            # Icelandic
            "þ": "Þetta þarf þrjátíu þúsund.",
            "ð": "Það er gott veður í dag.",
            "ç": "Hver er þetta?",
        }
        
        for phoneme in missing:
            if phoneme in phoneme_prompts:
                suggestions.append(phoneme_prompts[phoneme])
        
        return suggestions[:10]  # Return top 10


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_model_training_data(
    audio_dir: str,
    languages: List[str] = ["en", "is"]
) -> Dict[str, LanguageReadinessScore]:
    """
    Analyze training data directory for multiple languages.
    
    Args:
        audio_dir: Directory containing training audio
        languages: Languages to analyze
        
    Returns:
        Dict mapping language code to readiness score
    """
    import soundfile as sf
    from pathlib import Path
    
    analyzer = PhonemeAnalyzer()
    scorer = LanguageReadinessScorer(analyzer)
    results = {}
    
    audio_dir = Path(audio_dir)
    
    # Collect all audio
    all_audio = []
    sample_rate = None
    
    for ext in [".wav", ".mp3", ".flac"]:
        for audio_file in audio_dir.glob(f"**/*{ext}"):
            try:
                audio, sr = sf.read(str(audio_file))
                if sample_rate is None:
                    sample_rate = sr
                # Resample if needed (simple decimation)
                if sr != sample_rate:
                    continue  # Skip mismatched sample rates
                all_audio.append(audio)
            except Exception as e:
                logger.warning(f"Error reading {audio_file}: {e}")
    
    if not all_audio:
        logger.warning(f"No audio files found in {audio_dir}")
        return {}
    
    # Concatenate audio for analysis
    combined_audio = np.concatenate(all_audio)
    
    # Analyze audio quality
    audio_quality = AudioQualityAnalyzer.analyze(combined_audio, sample_rate)
    
    # Analyze pitch
    pitch_analysis = PitchAnalyzer.analyze(combined_audio, sample_rate)
    
    # Analyze each language
    for lang in languages:
        phoneme_report = analyzer.analyze_audio_directory(audio_dir, lang)
        
        score = scorer.score(
            phoneme_report=phoneme_report,
            pitch_analysis=pitch_analysis,
            audio_quality=audio_quality
        )
        
        results[lang] = score
    
    return results


if __name__ == "__main__":
    # Test the analyzer
    import sys
    
    if len(sys.argv) > 1:
        audio_dir = sys.argv[1]
        results = analyze_model_training_data(audio_dir)
        
        for lang, score in results.items():
            print(f"\n{'='*60}")
            print(f"Language: {lang}")
            print(f"Overall Score: {score.overall_score:.1f}/100")
            print(f"{'='*60}")
            print(json.dumps(score.to_dict(), indent=2))
    else:
        print("Usage: python phoneme_analyzer.py <audio_directory>")
