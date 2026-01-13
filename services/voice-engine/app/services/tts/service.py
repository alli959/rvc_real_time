"""
TTS Service - Main orchestration class

Handles TTS generation with automatic backend selection,
emotion processing, and voice conversion.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services.tts.bark_tts import (
    BARK_AVAILABLE, 
    BARK_SAMPLE_RATE,
    generate_with_bark,
)
from app.services.tts.edge_tts import (
    DEFAULT_VOICE,
    EDGE_VOICES,
    generate_with_edge_tts,
)
from app.services.tts.text_parser import (
    parse_text_for_tts,
    convert_tags_for_bark,
    EMOTION_PROSODY,
    SOUND_REPLACEMENTS,
)
from app.services.tts.audio_effects import (
    apply_emotion_effects,
    EMOTION_AUDIO_EFFECTS,
)

logger = logging.getLogger(__name__)


class TTSService:
    """
    Text-to-Speech service with multiple backends.
    
    Features:
    - Automatic backend selection (Bark preferred, Edge fallback)
    - Emotion/sound effect processing
    - Multi-voice segment support
    - Voice conversion integration
    """
    
    def __init__(
        self,
        prefer_bark: bool = True,
        default_voice: str = DEFAULT_VOICE,
    ):
        """
        Initialize TTS service.
        
        Args:
            prefer_bark: Use Bark when available (better emotions)
            default_voice: Default Edge TTS voice
        """
        self.prefer_bark = prefer_bark and BARK_AVAILABLE
        self.default_voice = default_voice
        
        logger.info(f"TTS Service initialized (bark={self.prefer_bark})")
    
    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        emotion: Optional[str] = None,
        use_bark: Optional[bool] = None,
        apply_effects: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.
        
        Args:
            text: Text to speak (can include emotion tags)
            voice: Voice name (Edge voice or Bark speaker)
            emotion: Override emotion for the entire text
            use_bark: Force Bark (True) or Edge (False), None for auto
            apply_effects: Apply audio effects for emotions
        
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        # Determine backend
        if use_bark is None:
            use_bark = self.prefer_bark
        
        if use_bark and not BARK_AVAILABLE:
            logger.warning("Bark requested but not available, using Edge TTS")
            use_bark = False
        
        # Parse text into segments
        segments = parse_text_for_tts(text)
        
        if len(segments) == 1 and not segments[0].get('emotion'):
            # Simple case - single segment, no emotions
            return self._generate_simple(
                segments[0]['text'],
                voice,
                emotion,
                use_bark,
                apply_effects
            )
        
        # Complex case - multiple segments with emotions
        return self._generate_with_segments(
            segments,
            voice,
            emotion,
            use_bark,
            apply_effects
        )
    
    def _generate_simple(
        self,
        text: str,
        voice: Optional[str],
        emotion: Optional[str],
        use_bark: bool,
        apply_effects: bool,
    ) -> Tuple[np.ndarray, int]:
        """Generate simple text without emotion parsing."""
        if use_bark:
            # Convert tags for Bark
            bark_text = convert_tags_for_bark(text)
            speaker = voice or "default"
            audio, sr = generate_with_bark(bark_text, speaker=speaker)
        else:
            # Get prosody from emotion
            rate = None
            pitch = None
            if emotion and emotion.lower() in EMOTION_PROSODY:
                prosody = EMOTION_PROSODY[emotion.lower()]
                rate = prosody.get('rate')
                pitch = prosody.get('pitch')
            
            audio, sr = generate_with_edge_tts(
                text,
                voice=voice or self.default_voice,
                rate=rate,
                pitch=pitch,
            )
        
        # Apply emotion effects
        if apply_effects and emotion:
            audio = apply_emotion_effects(audio, sr, emotion)
        
        return audio, sr
    
    def _generate_with_segments(
        self,
        segments: List[Dict],
        voice: Optional[str],
        global_emotion: Optional[str],
        use_bark: bool,
        apply_effects: bool,
    ) -> Tuple[np.ndarray, int]:
        """Generate text with multiple segments."""
        audio_parts = []
        target_sr = 24000 if use_bark else 44100
        
        for segment in segments:
            text = segment['text']
            emotion = segment.get('emotion') or global_emotion
            rate = segment.get('rate')
            
            if use_bark:
                bark_text = convert_tags_for_bark(text)
                speaker = voice or "default"
                audio, sr = generate_with_bark(bark_text, speaker=speaker)
            else:
                # Get prosody
                prosody_rate = rate
                prosody_pitch = None
                
                if emotion and emotion.lower() in EMOTION_PROSODY:
                    prosody = EMOTION_PROSODY[emotion.lower()]
                    if not prosody_rate:
                        prosody_rate = prosody.get('rate')
                    prosody_pitch = prosody.get('pitch')
                
                audio, sr = generate_with_edge_tts(
                    text,
                    voice=voice or self.default_voice,
                    rate=prosody_rate,
                    pitch=prosody_pitch,
                )
            
            # Apply emotion effects
            if apply_effects and emotion:
                audio = apply_emotion_effects(audio, sr, emotion)
            
            # Resample if needed
            if sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            audio_parts.append(audio)
        
        # Concatenate segments
        combined = np.concatenate(audio_parts)
        return combined, target_sr
    
    @property
    def available_voices(self) -> Dict[str, Dict]:
        """Get available voices for current backend."""
        if self.prefer_bark:
            from app.services.tts.bark_tts import BARK_SPEAKERS
            return {name: {'type': 'bark'} for name in BARK_SPEAKERS}
        else:
            return {name: {**info, 'type': 'edge'} for name, info in EDGE_VOICES.items()}
    
    @property
    def available_emotions(self) -> List[str]:
        """Get list of available emotions."""
        return list(EMOTION_PROSODY.keys())
    
    @property
    def available_sounds(self) -> List[str]:
        """Get list of available sound effects."""
        return list(SOUND_REPLACEMENTS.keys())
