"""
Model Scanner Module

Scans and analyzes existing voice models for:
- Language readiness scores
- Phoneme coverage gaps
- Audio quality assessment
- Improvement recommendations
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.analyzer import (
    PhonemeAnalyzer,
    AudioQualityAnalyzer,
    PitchAnalyzer,
    LanguageReadinessScorer,
    LanguageReadinessScore,
    LANGUAGE_PHONEMES,
)
from app.prompts import get_prompt_loader, PromptLoader

# ModelMetadata was moved to trainer service - define a minimal version here for compatibility
@dataclass
class ModelMetadata:
    """Minimal model metadata for scanning purposes"""
    name: str
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    language_scores: Dict[str, Any] = field(default_factory=dict)
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    prompts_used: List[str] = field(default_factory=list)
    total_audio_duration: float = 0.0

logger = logging.getLogger(__name__)


@dataclass
class ModelScanResult:
    """Result of scanning a voice model"""
    model_path: str
    model_name: str
    
    # Language readiness scores
    language_scores: Dict[str, LanguageReadinessScore]
    
    # Overall assessment
    primary_language: Optional[str] = None
    secondary_language: Optional[str] = None
    
    # Metadata
    has_metadata: bool = False
    metadata: Optional[ModelMetadata] = None
    
    # Training data analysis (if available)
    training_duration_seconds: float = 0.0
    training_clip_count: int = 0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "language_scores": {
                lang: score.to_dict() 
                for lang, score in self.language_scores.items()
            },
            "primary_language": self.primary_language,
            "secondary_language": self.secondary_language,
            "has_metadata": self.has_metadata,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "training_duration_seconds": self.training_duration_seconds,
            "training_clip_count": self.training_clip_count,
            "recommendations": self.recommendations
        }


@dataclass
class GapAnalysis:
    """Analysis of phoneme coverage gaps"""
    language: str
    missing_phonemes: Set[str]
    weak_phonemes: Set[str]
    coverage_percentage: float
    
    # Priority phonemes to record
    priority_phonemes: List[str]
    
    # Suggested prompts to fill gaps
    suggested_prompts: List[str]
    
    # Estimated recordings needed
    estimated_recordings: int
    estimated_duration_minutes: float
    
    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "missing_phonemes": sorted(list(self.missing_phonemes)),
            "weak_phonemes": sorted(list(self.weak_phonemes)),
            "coverage_percentage": round(self.coverage_percentage, 2),
            "priority_phonemes": self.priority_phonemes,
            "suggested_prompts": self.suggested_prompts,
            "estimated_recordings": self.estimated_recordings,
            "estimated_duration_minutes": round(self.estimated_duration_minutes, 1)
        }


class ModelScanner:
    """
    Scans voice models to analyze language readiness and identify gaps.
    
    Usage:
        scanner = ModelScanner()
        result = scanner.scan_model("/path/to/model.pth", languages=["en", "is"])
        gaps = scanner.analyze_gaps(result, "en")
    """
    
    def __init__(
        self,
        logs_dir: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the model scanner.
        
        Args:
            logs_dir: Directory containing training logs
            use_gpu: Whether to use GPU for analysis
        """
        self.logs_dir = Path(logs_dir) if logs_dir else None
        self.use_gpu = use_gpu
        
        self.phoneme_analyzer = PhonemeAnalyzer(use_gpu=use_gpu)
        self.scorer = LanguageReadinessScorer(self.phoneme_analyzer)
        self.prompt_loader = get_prompt_loader()
    
    def scan_model(
        self,
        model_path: str,
        languages: List[str] = ["en", "is"]
    ) -> ModelScanResult:
        """
        Scan a voice model for language readiness.
        
        This method:
        1. Locates training data if available
        2. Analyzes audio for phoneme coverage
        3. Calculates language readiness scores
        4. Generates recommendations
        
        Args:
            model_path: Path to .pth model file
            languages: Languages to analyze
            
        Returns:
            ModelScanResult with scores and recommendations
        """
        model_path = Path(model_path)
        model_name = model_path.stem
        
        logger.info(f"Scanning model: {model_name}")
        
        # Initialize result
        result = ModelScanResult(
            model_path=str(model_path),
            model_name=model_name,
            language_scores={}
        )
        
        # Try to find training data
        training_dir = self._find_training_dir(model_path)
        
        if training_dir:
            logger.info(f"Found training directory: {training_dir}")
            result = self._analyze_training_data(
                result, training_dir, languages
            )
        else:
            logger.warning("Training data not found - using metadata only")
            result = self._analyze_from_metadata(result, model_path, languages)
        
        # Load metadata if exists
        metadata_path = self._find_metadata(model_path)
        if metadata_path:
            try:
                result.metadata = ModelMetadata.load(metadata_path)
                result.has_metadata = True
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
        
        # Determine primary/secondary languages
        self._determine_languages(result)
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        return result
    
    def _find_training_dir(self, model_path: Path) -> Optional[Path]:
        """Find the training directory for a model"""
        # Check if model is in logs directory
        if "logs" in model_path.parts:
            # Model is in logs/<exp_name>/<model>.pth
            return model_path.parent
        
        # If model_path is a directory, check it directly first
        if model_path.is_dir():
            if (model_path / "0_gt_wavs").exists() or (model_path / "trainset").exists():
                logger.info(f"Found training data in model directory: {model_path}")
                return model_path
        
        # Check if the model's parent directory contains training data
        # RVC creates 0_gt_wavs for training audio, trainset for processed audio
        model_dir = model_path.parent
        if (model_dir / "0_gt_wavs").exists() or (model_dir / "trainset").exists():
            logger.info(f"Found training data in parent directory: {model_dir}")
            return model_dir
        
        # Check parallel logs directory
        if self.logs_dir:
            exp_name = model_path.stem.replace("_", "-")
            exp_dir = self.logs_dir / exp_name
            if exp_dir.exists():
                return exp_dir
        
        # Check for trainset directory nearby
        parents_to_check = [model_path, model_path.parent] if model_path.is_dir() else [model_path.parent, model_path.parent.parent]
        for parent in parents_to_check:
            trainset = parent / "trainset"
            if trainset.exists():
                return parent
            # Also check for 0_gt_wavs (RVC training format)
            gt_wavs = parent / "0_gt_wavs"
            if gt_wavs.exists():
                return parent
        
        return None
    
    def _find_metadata(self, model_path: Path) -> Optional[Path]:
        """Find metadata file for a model"""
        # Check same directory
        metadata = model_path.parent / "model_metadata.json"
        if metadata.exists():
            return metadata
        
        # Check parallel metadata
        metadata = model_path.with_suffix(".json")
        if metadata.exists():
            return metadata
        
        return None
    
    def _analyze_training_data(
        self,
        result: ModelScanResult,
        training_dir: Path,
        languages: List[str]
    ) -> ModelScanResult:
        """Analyze training data for language readiness"""
        import soundfile as sf
        
        # Find audio files in training directory
        audio_dirs = [
            training_dir / "trainset",
            training_dir / "0_gt_wavs",
            training_dir
        ]
        
        audio_files = []
        for audio_dir in audio_dirs:
            if audio_dir.exists():
                for ext in ["*.wav", "*.mp3", "*.flac"]:
                    audio_files.extend(audio_dir.glob(ext))
        
        if not audio_files:
            logger.warning("No audio files found in training directory")
            return result
        
        logger.info(f"Found {len(audio_files)} audio files")
        result.training_clip_count = len(audio_files)
        
        # Load and analyze audio
        all_audio = []
        sample_rate = None
        total_duration = 0.0
        
        for audio_file in audio_files[:100]:  # Limit for performance
            try:
                audio, sr = sf.read(str(audio_file))
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    continue
                
                all_audio.append(audio)
                total_duration += len(audio) / sr
                
            except Exception as e:
                logger.warning(f"Error reading {audio_file}: {e}")
        
        result.training_duration_seconds = total_duration
        
        if not all_audio or sample_rate is None:
            return result
        
        # Concatenate audio
        combined_audio = np.concatenate(all_audio)
        
        # Analyze audio quality
        audio_quality = AudioQualityAnalyzer.analyze(combined_audio, sample_rate)
        
        # Analyze pitch
        pitch_analysis = PitchAnalyzer.analyze(combined_audio, sample_rate)
        
        # Analyze each language
        for lang in languages:
            try:
                # Try phoneme analysis first (requires whisper)
                phoneme_report = None
                try:
                    phoneme_report = self.phoneme_analyzer.analyze_audio_directory(
                        str(training_dir), lang
                    )
                    # Check if we actually got phonemes (whisper may be missing)
                    if phoneme_report and phoneme_report.total_phonemes_found == 0:
                        logger.warning(f"No phonemes found for {lang} - using audio-based scoring")
                        phoneme_report = None
                except Exception as e:
                    logger.warning(f"Phoneme analysis failed for {lang}: {e}")
                    phoneme_report = None
                
                # Calculate readiness score
                if phoneme_report and phoneme_report.total_phonemes_found > 0:
                    # Full phoneme-based scoring
                    score = self.scorer.score(
                        phoneme_report=phoneme_report,
                        pitch_analysis=pitch_analysis,
                        audio_quality=audio_quality
                    )
                else:
                    # Audio-based fallback scoring (without transcription)
                    score = self._score_from_audio_stats(
                        lang=lang,
                        duration_seconds=total_duration,
                        clip_count=len(audio_files),
                        pitch_analysis=pitch_analysis,
                        audio_quality=audio_quality
                    )
                
                result.language_scores[lang] = score
                
            except Exception as e:
                logger.error(f"Error analyzing {lang}: {e}")
        
        return result
    
    def _score_from_audio_stats(
        self,
        lang: str,
        duration_seconds: float,
        clip_count: int,
        pitch_analysis: Optional[PitchAnalysis],
        audio_quality: Optional[AudioQualityMetrics]
    ) -> LanguageReadinessScore:
        """
        Calculate language readiness based on audio statistics when
        transcription/phoneme analysis is not available.
        
        Uses duration, clip count, pitch variation, and audio quality
        as proxies for model readiness.
        """
        recommendations = []
        
        # Duration score (0-100): 10min=50, 30min=80, 60min+=100
        duration_minutes = duration_seconds / 60
        if duration_minutes < 3:
            duration_score = duration_minutes / 3 * 30  # 0-30
            recommendations.append(f"Training data is very short ({duration_minutes:.1f} min). Need at least 10 minutes for good quality.")
        elif duration_minutes < 10:
            duration_score = 30 + (duration_minutes - 3) / 7 * 30  # 30-60
            recommendations.append(f"Training data is moderate ({duration_minutes:.1f} min). More recordings would improve quality.")
        elif duration_minutes < 30:
            duration_score = 60 + (duration_minutes - 10) / 20 * 25  # 60-85
        elif duration_minutes < 60:
            duration_score = 85 + (duration_minutes - 30) / 30 * 10  # 85-95
        else:
            duration_score = min(100, 95 + (duration_minutes - 60) / 60 * 5)
        
        # Clip count score: more clips = better variety
        if clip_count < 20:
            variety_score = clip_count / 20 * 40
            recommendations.append(f"Only {clip_count} clips. More variety would improve quality.")
        elif clip_count < 50:
            variety_score = 40 + (clip_count - 20) / 30 * 20
        elif clip_count < 100:
            variety_score = 60 + (clip_count - 50) / 50 * 20
        else:
            variety_score = min(100, 80 + (clip_count - 100) / 200 * 20)
        
        # Pitch variation score
        pitch_score = 50  # Default
        if pitch_analysis:
            # Good pitch range is 0.5-2 octaves
            octave_range = float(pitch_analysis.f0_range_octaves)
            if octave_range < 0.3:
                pitch_score = 30
                recommendations.append("Low pitch variation. Include more expressive speech.")
            elif octave_range < 0.7:
                pitch_score = 50 + (octave_range - 0.3) / 0.4 * 30
            elif octave_range < 1.5:
                pitch_score = 80 + (octave_range - 0.7) / 0.8 * 15
            else:
                pitch_score = 95 + min(5, (octave_range - 1.5) * 5)
        
        # Audio quality score
        quality_score = 50  # Default
        if audio_quality:
            snr = float(audio_quality.snr_db)
            silence_pct = float(audio_quality.silence_percentage)
            clipping_pct = float(audio_quality.clipping_percentage)
            
            # SNR: 20dB=70, 30dB=90, 40dB+=100
            if snr < 10:
                snr_score = 30
                recommendations.append("Audio quality is poor. Consider re-recording with better equipment.")
            elif snr < 20:
                snr_score = 30 + (snr - 10) / 10 * 40
            elif snr < 30:
                snr_score = 70 + (snr - 20) / 10 * 20
            else:
                snr_score = min(100, 90 + (snr - 30) / 10 * 10)
            
            # Penalize excessive silence or clipping
            silence_penalty = min(30, silence_pct * 0.5)
            clipping_penalty = min(30, clipping_pct * 3)
            
            quality_score = max(0, snr_score - silence_penalty - clipping_penalty)
            
            if clipping_pct > 5:
                recommendations.append("Audio has clipping. Reduce recording volume.")
        
        # Combined score with weights
        # Duration: 35%, Variety: 20%, Pitch: 20%, Quality: 25%
        overall_score = (
            duration_score * 0.35 +
            variety_score * 0.20 +
            pitch_score * 0.20 +
            quality_score * 0.25
        )
        
        # Add recommendation if no whisper
        recommendations.insert(0, "Phoneme analysis unavailable - scores based on audio statistics.")
        
        return LanguageReadinessScore(
            language=lang,
            overall_score=float(overall_score),
            phoneme_coverage_score=float(duration_score),  # Use duration as proxy
            vowel_variation_score=float(variety_score),
            pitch_variation_score=float(pitch_score),
            speaking_rate_score=50.0,  # Unknown without transcription
            prosody_quality_score=float(pitch_score),
            audio_quality_score=float(quality_score),
            phoneme_report=None,
            pitch_analysis=pitch_analysis,
            audio_quality=audio_quality,
            recommendations=recommendations,
            suggested_prompts=[]
        )
    
    def _analyze_from_metadata(
        self,
        result: ModelScanResult,
        model_path: Path,
        languages: List[str]
    ) -> ModelScanResult:
        """Analyze model using only metadata (no training data)"""
        # If we have metadata, use its language scores
        metadata_path = self._find_metadata(model_path)
        
        if metadata_path:
            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                
                lang_readiness = data.get("language_readiness", {})
                
                for lang in languages:
                    if lang in lang_readiness:
                        # Create score from metadata
                        score_data = lang_readiness[lang]
                        score = LanguageReadinessScore(
                            language=lang,
                            overall_score=score_data.get("overall_score", 0),
                            phoneme_coverage_score=score_data.get("phoneme_coverage", 0),
                            vowel_variation_score=score_data.get("vowel_variation", 0),
                            pitch_variation_score=score_data.get("pitch_variation", 0),
                            speaking_rate_score=score_data.get("speaking_rate", 0),
                            prosody_quality_score=score_data.get("prosody_quality", 0),
                            audio_quality_score=score_data.get("audio_quality", 0),
                            recommendations=score_data.get("recommendations", []),
                            suggested_prompts=score_data.get("suggested_prompts", [])
                        )
                        result.language_scores[lang] = score
                
            except Exception as e:
                logger.warning(f"Error reading metadata: {e}")
        
        # Create placeholder scores for missing languages
        for lang in languages:
            if lang not in result.language_scores:
                result.language_scores[lang] = LanguageReadinessScore(
                    language=lang,
                    overall_score=0,
                    phoneme_coverage_score=0,
                    vowel_variation_score=0,
                    pitch_variation_score=0,
                    speaking_rate_score=0,
                    prosody_quality_score=0,
                    audio_quality_score=0,
                    recommendations=["Unable to analyze - training data not found"],
                    suggested_prompts=[]
                )
        
        return result
    
    def _determine_languages(self, result: ModelScanResult):
        """Determine primary and secondary languages based on scores"""
        if not result.language_scores:
            return
        
        # Sort by overall score
        sorted_langs = sorted(
            result.language_scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        if sorted_langs:
            result.primary_language = sorted_langs[0][0]
            
            if len(sorted_langs) > 1:
                # Only set secondary if score is reasonable
                if sorted_langs[1][1].overall_score >= 40:
                    result.secondary_language = sorted_langs[1][0]
    
    def _generate_recommendations(self, result: ModelScanResult) -> List[str]:
        """Generate overall recommendations based on analysis"""
        recommendations = []
        
        # Duration-based recommendations
        if result.training_duration_seconds < 300:  # < 5 minutes
            recommendations.append(
                f"Training data is short ({result.training_duration_seconds/60:.1f} min). "
                "Consider adding more recordings for better quality."
            )
        
        # Language-specific recommendations
        for lang, score in result.language_scores.items():
            if score.overall_score < 50:
                recommendations.append(
                    f"Low {lang.upper()} readiness ({score.overall_score:.0f}%). "
                    f"Major gaps in phoneme coverage."
                )
            elif score.overall_score < 70:
                recommendations.append(
                    f"Moderate {lang.upper()} readiness ({score.overall_score:.0f}%). "
                    f"Some phoneme gaps to address."
                )
            else:
                recommendations.append(
                    f"Good {lang.upper()} readiness ({score.overall_score:.0f}%). "
                    f"Model is well-suited for this language."
                )
            
            # Add specific recommendations from score
            recommendations.extend(score.recommendations[:2])
        
        return recommendations
    
    def analyze_gaps(
        self,
        scan_result: ModelScanResult,
        language: str,
        max_prompts: int = 30
    ) -> GapAnalysis:
        """
        Analyze phoneme coverage gaps for a specific language.
        
        Args:
            scan_result: Result from scan_model()
            language: Language to analyze
            max_prompts: Maximum prompts to suggest
            
        Returns:
            GapAnalysis with suggested improvements
        """
        if language not in scan_result.language_scores:
            # No score available
            return GapAnalysis(
                language=language,
                missing_phonemes=LANGUAGE_PHONEMES.get(language, set()),
                weak_phonemes=set(),
                coverage_percentage=0,
                priority_phonemes=list(LANGUAGE_PHONEMES.get(language, set()))[:10],
                suggested_prompts=[],
                estimated_recordings=50,
                estimated_duration_minutes=10
            )
        
        score = scan_result.language_scores[language]
        phoneme_report = score.phoneme_report
        
        if not phoneme_report:
            return GapAnalysis(
                language=language,
                missing_phonemes=LANGUAGE_PHONEMES.get(language, set()),
                weak_phonemes=set(),
                coverage_percentage=0,
                priority_phonemes=[],
                suggested_prompts=[],
                estimated_recordings=50,
                estimated_duration_minutes=10
            )
        
        # Get missing and weak phonemes
        missing = phoneme_report.missing_phonemes
        weak = phoneme_report.weak_phonemes
        
        # Prioritize phonemes
        priority = self._prioritize_phonemes(missing, weak, language)
        
        # Get prompts for gaps
        suggested_prompts = self.prompt_loader.get_prompts_for_missing_phonemes(
            language, missing | weak, max_prompts
        )
        
        # Estimate recordings needed
        gap_count = len(missing) + len(weak)
        estimated_recordings = min(gap_count * 3, 50)  # ~3 recordings per phoneme
        
        # Estimate duration (avg 10 seconds per prompt)
        estimated_duration = estimated_recordings * 10 / 60
        
        return GapAnalysis(
            language=language,
            missing_phonemes=missing,
            weak_phonemes=weak,
            coverage_percentage=phoneme_report.coverage_percentage,
            priority_phonemes=priority,
            suggested_prompts=suggested_prompts,
            estimated_recordings=estimated_recordings,
            estimated_duration_minutes=estimated_duration
        )
    
    def _prioritize_phonemes(
        self,
        missing: Set[str],
        weak: Set[str],
        language: str
    ) -> List[str]:
        """Prioritize phonemes by importance"""
        # Define phoneme importance (higher = more important)
        # Common phonemes and those hard to synthesize without examples
        importance = {
            # High priority - very common or distinctive
            "θ": 10, "ð": 10,  # TH sounds (English)
            "ʃ": 9, "ʒ": 9,    # SH sounds
            "ŋ": 8,            # NG
            "r": 8, "l": 8,    # Liquids
            
            # Medium priority
            "tʃ": 7, "dʒ": 7,  # Affricates
            "w": 6, "j": 6,    # Glides
            
            # Vowels (all important)
            "iː": 7, "uː": 7, "aː": 7, "ɔː": 7,  # Long vowels
            "eɪ": 6, "aɪ": 6, "ɔɪ": 6, "aʊ": 6,  # Diphthongs
        }
        
        # Prioritize missing over weak
        priority_list = []
        
        for phoneme in sorted(missing, key=lambda p: -importance.get(p, 5)):
            priority_list.append(phoneme)
        
        for phoneme in sorted(weak, key=lambda p: -importance.get(p, 5)):
            if phoneme not in priority_list:
                priority_list.append(phoneme)
        
        return priority_list[:15]  # Top 15
    
    def compare_models(
        self,
        model_a_path: str,
        model_b_path: str,
        languages: List[str] = ["en", "is"]
    ) -> Dict[str, Any]:
        """
        Compare two models for language readiness.
        
        Args:
            model_a_path: Path to first model
            model_b_path: Path to second model
            languages: Languages to compare
            
        Returns:
            Comparison dict with differences
        """
        result_a = self.scan_model(model_a_path, languages)
        result_b = self.scan_model(model_b_path, languages)
        
        comparison = {
            "model_a": {
                "path": model_a_path,
                "name": result_a.model_name
            },
            "model_b": {
                "path": model_b_path,
                "name": result_b.model_name
            },
            "language_comparison": {}
        }
        
        for lang in languages:
            score_a = result_a.language_scores.get(lang)
            score_b = result_b.language_scores.get(lang)
            
            if score_a and score_b:
                comparison["language_comparison"][lang] = {
                    "model_a_score": score_a.overall_score,
                    "model_b_score": score_b.overall_score,
                    "difference": score_b.overall_score - score_a.overall_score,
                    "better_model": "b" if score_b.overall_score > score_a.overall_score else "a"
                }
        
        return comparison


# Convenience functions

def scan_model(model_path: str, languages: List[str] = ["en", "is"]) -> ModelScanResult:
    """Scan a model for language readiness"""
    scanner = ModelScanner()
    return scanner.scan_model(model_path, languages)


def analyze_model_gaps(
    model_path: str,
    language: str = "en"
) -> GapAnalysis:
    """Analyze gaps in a model for a specific language"""
    scanner = ModelScanner()
    result = scanner.scan_model(model_path, [language])
    return scanner.analyze_gaps(result, language)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        result = scan_model(model_path)
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("Usage: python model_scanner.py <model_path>")
