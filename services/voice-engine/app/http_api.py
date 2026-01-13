"""
HTTP API Server for Voice Engine

Provides REST API endpoints for:
- Text-to-Speech (TTS) generation using Bark (with native emotions) or Edge TTS (with audio effects fallback)
- Voice conversion using RVC models
- Health checks and model listing

Emotion Support:
- Bark TTS: Native support for [laughter], [sighs], [gasps], etc.
- Edge TTS: Audio processing effects to simulate emotions (pitch, rate, tremolo, etc.)
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import enhanced TTS service
try:
    from app.tts_service import (
        generate_tts,
        is_bark_available,
        preload_bark_models,
        BARK_AVAILABLE
    )
    TTS_SERVICE_AVAILABLE = True
except ImportError:
    TTS_SERVICE_AVAILABLE = False
    BARK_AVAILABLE = False

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MorphVox Voice Engine API",
    description="Voice conversion and TTS API",
    version="1.0.0"
)

# Import YouTube service (optional)
try:
    from app.youtube_service import (
        search_youtube, 
        download_youtube_audio, 
        get_video_info,
        SearchResult,
        is_cached
    )
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    logger.warning("YouTube service not available - yt-dlp may not be installed")

# Import trainer API (optional)
try:
    from app.trainer_api import router as trainer_router, init_trainer_api
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    trainer_router = None
    logger.warning("Trainer API not available - missing dependencies")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include trainer router if available
if TRAINER_AVAILABLE and trainer_router:
    app.include_router(trainer_router)
    logger.info("Trainer API endpoints enabled")

# Global model manager reference (set from main.py)
model_manager = None
infer_params = None


def set_model_manager(mm, params):
    """Set the model manager instance from main.py"""
    global model_manager, infer_params
    model_manager = mm
    infer_params = params


# =============================================================================
# Request/Response Models
# =============================================================================

class TTSRequest(BaseModel):
    """TTS generation request"""
    text: str = Field(..., description="Text to convert to speech", max_length=10000)
    voice: str = Field(default="en-US-GuyNeural", description="Edge TTS voice ID")
    style: str = Field(default="default", description="Speaking style/emotion")
    rate: str = Field(default="+0%", description="Speech rate adjustment")
    pitch: str = Field(default="+0Hz", description="Pitch adjustment")
    # Bark TTS option (native emotion/sound effects support)
    use_bark: bool = Field(default=True, description="Use Bark TTS for native emotion support (falls back to Edge TTS if unavailable)")
    bark_speaker: str = Field(default="default", description="Bark speaker preset (default, male1, male2, female1, female2, dramatic, calm)")
    # Multi-voice generation support
    include_segments: Optional[List[dict]] = Field(default=None, description="List of segments with different voice models: [{text, voice_model_id, model_path, index_path, f0_up_key, index_rate}]")


class MultiVoiceSegment(BaseModel):
    """A segment for multi-voice generation"""
    text: str = Field(..., description="Text for this segment")
    voice: str = Field(default=None, description="Override TTS voice for this segment")
    voice_model_id: Optional[int] = Field(default=None, description="Voice model ID for RVC conversion")
    model_path: Optional[str] = Field(default=None, description="Path to .pth model file")
    index_path: Optional[str] = Field(default=None, description="Path to .index file")
    f0_up_key: int = Field(default=0, description="Pitch shift for this segment")
    index_rate: float = Field(default=0.75, description="Index rate for this segment")
    rate: Optional[str] = Field(default=None, description="Override speech rate for this segment")


class TTSResponse(BaseModel):
    """TTS generation response"""
    audio: str = Field(..., description="Base64 encoded WAV audio")
    sample_rate: int = Field(default=24000, description="Audio sample rate")
    format: str = Field(default="wav", description="Audio format")


class ConvertRequest(BaseModel):
    """Voice conversion request"""
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=16000, description="Input sample rate")
    model_path: str = Field(..., description="Path to .pth model file")
    index_path: Optional[str] = Field(default=None, description="Path to .index file")
    f0_up_key: int = Field(default=0, description="Pitch shift (-12 to 12)")
    f0_method: str = Field(default="rmvpe", description="F0 extraction method")
    # Adjusted defaults for more natural sound
    index_rate: float = Field(default=0.5, description="Index blend rate (0.0-1.0, lower=more natural, higher=more like target)")
    filter_radius: int = Field(default=3, description="Filter radius (0-7)")
    rms_mix_rate: float = Field(default=0.2, description="RMS mix rate (0.0-1.0, lower=keep original dynamics)")
    protect: float = Field(default=0.4, description="Protect consonants (0.0-0.5, higher=more natural speech)")
    # Quality preset option
    quality_preset: Optional[str] = Field(default=None, description="Quality preset: 'natural', 'balanced', 'accurate', or None for custom")
    # Post-conversion audio effects
    apply_effects: Optional[str] = Field(default=None, description="Emotion/effect name to apply after conversion (e.g. 'robot', 'whisper', 'terrified')")


class ApplyEffectsRequest(BaseModel):
    """Request to apply audio effects to existing audio"""
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=16000, description="Input sample rate")
    effect: str = Field(..., description="Effect name to apply (e.g. 'robot', 'whisper', 'terrified')")


class ConvertResponse(BaseModel):
    """Voice conversion response"""
    audio: str = Field(..., description="Base64 encoded converted audio")
    sample_rate: int = Field(default=16000, description="Output sample rate")
    format: str = Field(default="wav", description="Audio format")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    current_model: Optional[str]


# =============================================================================
# Edge TTS Integration with Emotion Support
# =============================================================================

# Emotion presets - adjust prosody to simulate emotions
# Format: (rate_adjustment, pitch_adjustment, description)
# Enhanced values for more noticeable effect
# Emotion intensity multiplier - lower = more subtle/natural prosody changes
# Set to 0.5-0.6 for more lifelike output, 1.0 for full effect
EMOTION_INTENSITY = 0.55

# Index rate reduction for emotional segments - lets more base TTS prosody survive RVC
# Higher values = more reduction (more natural but less voice similarity)
EMOTION_INDEX_RATE_REDUCTION = {
    'intense': 0.15,   # angry, furious, terrified, shocked, excited
    'moderate': 0.10,  # sad, happy, scared, surprised
    'mild': 0.05,      # calm, serious, neutral
}

EMOTION_PRESETS: Dict[str, Dict[str, str]] = {
    # Happy / Positive emotions - subtle pitch and speed (softened ~45%)
    'happy': {'rate': '+12%', 'pitch': '+8Hz', 'desc': 'Cheerful, upbeat tone', 'intensity': 'moderate'},
    'excited': {'rate': '+18%', 'pitch': '+12Hz', 'desc': 'Very enthusiastic', 'intensity': 'intense'},
    'cheerful': {'rate': '+14%', 'pitch': '+10Hz', 'desc': 'Light and positive', 'intensity': 'moderate'},
    'joyful': {'rate': '+12%', 'pitch': '+10Hz', 'desc': 'Full of joy', 'intensity': 'moderate'},
    
    # Sad / Negative emotions - gentler slowdown
    'sad': {'rate': '-15%', 'pitch': '-8Hz', 'desc': 'Melancholic, slow', 'intensity': 'moderate'},
    'melancholy': {'rate': '-20%', 'pitch': '-10Hz', 'desc': 'Deep sadness', 'intensity': 'moderate'},
    'depressed': {'rate': '-22%', 'pitch': '-12Hz', 'desc': 'Very low energy', 'intensity': 'intense'},
    'disappointed': {'rate': '-12%', 'pitch': '-6Hz', 'desc': 'Let down feeling', 'intensity': 'mild'},
    
    # Angry / Intense emotions - reduced aggression for natural delivery
    'angry': {'rate': '+8%', 'pitch': '+5Hz', 'desc': 'Frustrated, intense', 'intensity': 'intense'},
    'furious': {'rate': '+12%', 'pitch': '+8Hz', 'desc': 'Very angry', 'intensity': 'intense'},
    'annoyed': {'rate': '+5%', 'pitch': '+3Hz', 'desc': 'Mildly irritated', 'intensity': 'mild'},
    'frustrated': {'rate': '+7%', 'pitch': '+4Hz', 'desc': 'Exasperated', 'intensity': 'moderate'},
    'disgruntled': {'rate': '+3%', 'pitch': '+2Hz', 'desc': 'Unhappy, grumbling', 'intensity': 'mild'},
    
    # Calm / Neutral emotions - very subtle
    'calm': {'rate': '-10%', 'pitch': '-4Hz', 'desc': 'Relaxed, peaceful', 'intensity': 'mild'},
    'peaceful': {'rate': '-15%', 'pitch': '-6Hz', 'desc': 'Very serene', 'intensity': 'mild'},
    'relaxed': {'rate': '-12%', 'pitch': '-5Hz', 'desc': 'At ease', 'intensity': 'mild'},
    'neutral': {'rate': '+0%', 'pitch': '+0Hz', 'desc': 'Standard tone', 'intensity': 'mild'},
    
    # Surprised / Shocked emotions - toned down jumps
    'surprised': {'rate': '+12%', 'pitch': '+12Hz', 'desc': 'Caught off guard', 'intensity': 'moderate'},
    'shocked': {'rate': '+18%', 'pitch': '+18Hz', 'desc': 'Very surprised', 'intensity': 'intense'},
    'amazed': {'rate': '+10%', 'pitch': '+10Hz', 'desc': 'In awe', 'intensity': 'moderate'},
    
    # Fear / Anxiety emotions - reduced trembling
    'scared': {'rate': '+10%', 'pitch': '+10Hz', 'desc': 'Frightened', 'intensity': 'moderate'},
    'terrified': {'rate': '+15%', 'pitch': '+15Hz', 'desc': 'Extremely scared', 'intensity': 'intense'},
    'anxious': {'rate': '+8%', 'pitch': '+6Hz', 'desc': 'Nervous, worried', 'intensity': 'moderate'},
    'nervous': {'rate': '+6%', 'pitch': '+5Hz', 'desc': 'Slightly on edge', 'intensity': 'mild'},
    'worried': {'rate': '+5%', 'pitch': '+4Hz', 'desc': 'Concerned', 'intensity': 'mild'},
    
    # Special expressions - more natural
    'whisper': {'rate': '-15%', 'pitch': '-12Hz', 'desc': 'Quiet, secretive', 'intensity': 'moderate'},
    'whispering': {'rate': '-15%', 'pitch': '-12Hz', 'desc': 'Quiet, secretive', 'intensity': 'moderate'},
    'shouting': {'rate': '+12%', 'pitch': '+10Hz', 'desc': 'Loud, emphatic', 'intensity': 'intense'},
    'sarcastic': {'rate': '-5%', 'pitch': '+6Hz', 'desc': 'Ironic tone', 'intensity': 'mild'},
    'romantic': {'rate': '-12%', 'pitch': '-6Hz', 'desc': 'Soft, loving', 'intensity': 'moderate'},
    'affectionate': {'rate': '-10%', 'pitch': '-5Hz', 'desc': 'Warm, caring', 'intensity': 'moderate'},
    'serious': {'rate': '-8%', 'pitch': '-5Hz', 'desc': 'Grave, important', 'intensity': 'mild'},
    'playful': {'rate': '+12%', 'pitch': '+8Hz', 'desc': 'Fun, teasing', 'intensity': 'moderate'},
    'dramatic': {'rate': '-10%', 'pitch': '+8Hz', 'desc': 'Theatrical', 'intensity': 'moderate'},
    'mysterious': {'rate': '-12%', 'pitch': '-8Hz', 'desc': 'Enigmatic', 'intensity': 'moderate'},
    'gentle': {'rate': '-8%', 'pitch': '-4Hz', 'desc': 'Soft, kind', 'intensity': 'mild'},
    'embarrassed': {'rate': '-5%', 'pitch': '+3Hz', 'desc': 'Awkward, shy', 'intensity': 'mild'},
    
    # Actions / Sounds - gentler simulation
    'laugh': {'rate': '+8%', 'pitch': '+5Hz', 'desc': 'Laughing sound', 'intensity': 'mild'},
    'laughing': {'rate': '+8%', 'pitch': '+5Hz', 'desc': 'Laughing sound', 'intensity': 'mild'},
    'giggle': {'rate': '+10%', 'pitch': '+8Hz', 'desc': 'Light giggling', 'intensity': 'mild'},
    'chuckle': {'rate': '+5%', 'pitch': '+3Hz', 'desc': 'Soft laugh', 'intensity': 'mild'},
    'snicker': {'rate': '+6%', 'pitch': '+4Hz', 'desc': 'Suppressed laugh', 'intensity': 'mild'},
    'cackle': {'rate': '+10%', 'pitch': '+6Hz', 'desc': 'Witch-like laugh', 'intensity': 'moderate'},
    'sigh': {'rate': '-12%', 'pitch': '-5Hz', 'desc': 'Exhale sound', 'intensity': 'mild'},
    'gasp': {'rate': '+12%', 'pitch': '+8Hz', 'desc': 'Sharp intake', 'intensity': 'moderate'},
    'yawn': {'rate': '-15%', 'pitch': '-8Hz', 'desc': 'Tired sound', 'intensity': 'mild'},
    'cry': {'rate': '-8%', 'pitch': '+2Hz', 'desc': 'Sobbing', 'intensity': 'moderate'},
    'crying': {'rate': '-8%', 'pitch': '+2Hz', 'desc': 'Sobbing', 'intensity': 'moderate'},
    'sob': {'rate': '-10%', 'pitch': '+3Hz', 'desc': 'Deep crying', 'intensity': 'moderate'},
    'sniff': {'rate': '-5%', 'pitch': '+1Hz', 'desc': 'Sniffling', 'intensity': 'mild'},
    'groan': {'rate': '-10%', 'pitch': '-6Hz', 'desc': 'Pain/frustration', 'intensity': 'moderate'},
    'moan': {'rate': '-12%', 'pitch': '-4Hz', 'desc': 'Discomfort sound', 'intensity': 'moderate'},
    'scream': {'rate': '+15%', 'pitch': '+12Hz', 'desc': 'Screaming', 'intensity': 'intense'},
    'shriek': {'rate': '+18%', 'pitch': '+15Hz', 'desc': 'High pitched scream', 'intensity': 'intense'},
    'growl': {'rate': '-8%', 'pitch': '-10Hz', 'desc': 'Angry growl', 'intensity': 'moderate'},
    'hum': {'rate': '-20%', 'pitch': '+0Hz', 'desc': 'Humming'},
    'cough': {'rate': '+10%', 'pitch': '+5Hz', 'desc': 'Coughing'},
    'sneeze': {'rate': '+20%', 'pitch': '+10Hz', 'desc': 'Sneezing'},
    'hiccup': {'rate': '+15%', 'pitch': '+12Hz', 'desc': 'Hiccuping'},
    'burp': {'rate': '-10%', 'pitch': '-15Hz', 'desc': 'Burping'},
    'gulp': {'rate': '+5%', 'pitch': '+3Hz', 'desc': 'Swallowing'},
    'slurp': {'rate': '+8%', 'pitch': '+5Hz', 'desc': 'Slurping'},
    'whistle': {'rate': '+10%', 'pitch': '+20Hz', 'desc': 'Whistling'},
    'hiss': {'rate': '+5%', 'pitch': '+15Hz', 'desc': 'Hissing sound'},
    'shush': {'rate': '-15%', 'pitch': '+5Hz', 'desc': 'Shushing'},
    'clap': {'rate': '+10%', 'pitch': '+8Hz', 'desc': 'Clapping'},
    'snap': {'rate': '+15%', 'pitch': '+10Hz', 'desc': 'Finger snap'},
    'kiss': {'rate': '-10%', 'pitch': '+8Hz', 'desc': 'Kiss sound'},
    'blow': {'rate': '-15%', 'pitch': '-5Hz', 'desc': 'Blowing air'},
    'pant': {'rate': '+20%', 'pitch': '+5Hz', 'desc': 'Heavy breathing'},
    'breathe': {'rate': '-20%', 'pitch': '-5Hz', 'desc': 'Deep breath'},
    'inhale': {'rate': '-15%', 'pitch': '+3Hz', 'desc': 'Breathing in'},
    'exhale': {'rate': '-20%', 'pitch': '-5Hz', 'desc': 'Breathing out'},
    'stutter': {'rate': '-5%', 'pitch': '+3Hz', 'desc': 'Stuttering'},
    'mumble': {'rate': '-15%', 'pitch': '-5Hz', 'desc': 'Mumbling'},
    'stammer': {'rate': '-8%', 'pitch': '+2Hz', 'desc': 'Stammering'},
    
    # Thinking sounds
    'hmm': {'rate': '-15%', 'pitch': '+0Hz', 'desc': 'Thinking sound'},
    'thinking': {'rate': '-20%', 'pitch': '-3Hz', 'desc': 'Deep in thought'},
    'uhh': {'rate': '-10%', 'pitch': '+0Hz', 'desc': 'Hesitation'},
    'umm': {'rate': '-10%', 'pitch': '+0Hz', 'desc': 'Pause filler'},
    
    # Reactions
    'wow': {'rate': '+10%', 'pitch': '+12Hz', 'desc': 'Impressed'},
    'ooh': {'rate': '+5%', 'pitch': '+10Hz', 'desc': 'Intrigued'},
    'ahh': {'rate': '-5%', 'pitch': '+5Hz', 'desc': 'Realization'},
    'ugh': {'rate': '-10%', 'pitch': '-8Hz', 'desc': 'Disgusted'},
    'eww': {'rate': '+5%', 'pitch': '+8Hz', 'desc': 'Grossed out'},
    'yay': {'rate': '+25%', 'pitch': '+15Hz', 'desc': 'Celebration'},
    'boo': {'rate': '-5%', 'pitch': '-5Hz', 'desc': 'Disapproval'},
    'woohoo': {'rate': '+30%', 'pitch': '+20Hz', 'desc': 'Excitement'},
    'ow': {'rate': '+15%', 'pitch': '+10Hz', 'desc': 'Pain'},
    'ouch': {'rate': '+20%', 'pitch': '+12Hz', 'desc': 'Sharp pain'},
    'phew': {'rate': '-15%', 'pitch': '-5Hz', 'desc': 'Relief'},
    'tsk': {'rate': '+5%', 'pitch': '+5Hz', 'desc': 'Disapproval click'},
    'psst': {'rate': '-20%', 'pitch': '+10Hz', 'desc': 'Getting attention quietly'},
    
    # Special voice effects (with audio processing)
    'robot': {'rate': '+0%', 'pitch': '+0Hz', 'desc': 'Robotic voice effect'},
    'spooky': {'rate': '-15%', 'pitch': '-10Hz', 'desc': 'Spooky/haunted voice'},
    'ethereal': {'rate': '-10%', 'pitch': '+5Hz', 'desc': 'Ethereal/heavenly voice'},
    'phone': {'rate': '+0%', 'pitch': '+0Hz', 'desc': 'Phone call quality'},
    'radio': {'rate': '+0%', 'pitch': '+0Hz', 'desc': 'Radio broadcast quality'},
    'megaphone': {'rate': '+5%', 'pitch': '+3Hz', 'desc': 'Megaphone/PA system'},
    'echo': {'rate': '-5%', 'pitch': '+0Hz', 'desc': 'Echoey room'},
    'underwater': {'rate': '-10%', 'pitch': '-5Hz', 'desc': 'Underwater/muffled'},
}

# Sound effect text replacements (what the TTS will say for actions)
SOUND_REPLACEMENTS: Dict[str, str] = {
    # Laughs
    'laugh': 'haha haha',
    'laughing': 'hahaha haha',
    'giggle': 'hehe hehe',
    'chuckle': 'heh heh heh',
    'snicker': 'heh heh',
    'cackle': 'ah hahaha haha',
    
    # Crying/Sadness
    'cry': 'huuu huuu',
    'crying': 'huuu huuu huuu',
    'sob': 'huuuh huuuh',
    'sniff': 'sniff sniff',
    
    # Surprise/Fear
    'gasp': 'aah!',
    'scream': 'aaaah!',
    'shriek': 'eeeek!',
    
    # Pain/Discomfort  
    'groan': 'uuugh',
    'moan': 'mmmmh',
    'sigh': 'haaaah',
    'yawn': 'aaaahhh',
    
    # Body sounds
    'cough': 'ahem ahem',
    'sneeze': 'achoo!',
    'hiccup': 'hic!',
    'burp': 'burrrp',
    'gulp': 'gulp',
    'slurp': 'slurrrp',
    
    # Other vocalizations
    'growl': 'grrrrr',
    'hiss': 'ssssss',
    'hum': 'hmm hmm hmm',
    'whistle': 'wheeew',
    'shush': 'shhhh',
    'kiss': 'mwah',
    'blow': 'fwooo',
    
    # Breathing
    'pant': 'hah hah hah',
    'breathe': 'hhhhhh',
    'inhale': 'hhhhh',
    'exhale': 'haaaah',
    
    # Speech patterns
    'stutter': 'I I I',
    'mumble': 'mmm mmm',
    'stammer': 'uh uh um',
    
    # Thinking
    'hmm': 'hmmm',
    'thinking': 'hmmmm',
    'uhh': 'uhhh',
    'umm': 'ummm',
    
    # Reactions
    'clap': 'clap clap clap',
    'snap': 'snap',
    'wow': 'woooow',
    'ooh': 'oooooh',
    'ahh': 'aaaaaah',
    'ugh': 'uuugh',
    'eww': 'eeeww',
    'yay': 'yaaaaay',
    'boo': 'boooo',
    'woohoo': 'woo hoo!',
    'ow': 'ow ow ow',
    'ouch': 'ouch!',
    'phew': 'pheeew',
    'tsk': 'tsk tsk tsk',
    'psst': 'psssst',
    'thinking': 'hmmmm',
}

# =============================================================================
# Audio Effects for Enhanced Emotion Expression
# =============================================================================

# Audio effects to apply per emotion (processed after TTS generation)
# Uses scipy/numpy for DSP - no external dependencies needed
# Enhanced values for more noticeable emotional effects
EMOTION_EFFECTS: Dict[str, Dict] = {
    # Whisper: quieter with breathy quality
    'whisper': {'volume': 0.4, 'highpass': 250, 'noise': 0.02},
    'mysterious': {'volume': 0.6, 'reverb': 0.5, 'lowpass': 5000},
    
    # Shouting/Angry: louder with more aggressive saturation
    'shouting': {'volume': 1.6, 'saturation': 0.35, 'compression': True},
    'angry': {'volume': 1.4, 'saturation': 0.25, 'highpass': 150},
    'furious': {'volume': 1.5, 'saturation': 0.3, 'highpass': 180, 'compression': True},
    
    # Scared: stronger tremolo effect (voice shaking)
    'scared': {'tremolo': {'rate': 8, 'depth': 0.25}, 'pitch_wobble': 0.1},
    'terrified': {'tremolo': {'rate': 12, 'depth': 0.4}, 'pitch_wobble': 0.15, 'volume': 1.1},
    'nervous': {'tremolo': {'rate': 5, 'depth': 0.15}},
    'anxious': {'tremolo': {'rate': 6, 'depth': 0.2}},
    
    # Sad: quieter with more muffled quality (crying-like)
    'sad': {'volume': 0.7, 'lowpass': 5500, 'reverb': 0.15},
    'melancholy': {'volume': 0.65, 'lowpass': 5000, 'reverb': 0.2},
    'depressed': {'volume': 0.55, 'lowpass': 4500, 'reverb': 0.25},
    'crying': {'volume': 0.75, 'tremolo': {'rate': 4, 'depth': 0.2}, 'lowpass': 6000},
    'sob': {'volume': 0.7, 'tremolo': {'rate': 5, 'depth': 0.25}, 'lowpass': 5500},
    
    # Happy/Excited: brighter and slightly louder
    'happy': {'volume': 1.15, 'highpass': 100, 'brightness': 0.2},
    'excited': {'volume': 1.25, 'highpass': 120, 'brightness': 0.3},
    'cheerful': {'volume': 1.1, 'brightness': 0.15},
    'joyful': {'volume': 1.2, 'brightness': 0.25},
    
    # Dramatic: strong reverb for theater effect
    'dramatic': {'reverb': 0.45, 'volume': 1.1},
    
    # Robot/electronic effect
    'robot': {'bitcrush': 6, 'lowpass': 3500, 'formant_shift': 0.8},
    
    # Echo for spooky
    'spooky': {'reverb': 0.55, 'lowpass': 4500, 'volume': 0.8, 'echo': True},
    'ethereal': {'reverb': 0.65, 'highpass': 350, 'shimmer': 0.3},
    
    # Phone/radio effect
    'phone': {'lowpass': 3200, 'highpass': 350, 'volume': 0.85, 'saturation': 0.1},
    'radio': {'lowpass': 3800, 'highpass': 250, 'saturation': 0.15, 'noise': 0.015},
    
    # Megaphone
    'megaphone': {'lowpass': 4500, 'highpass': 500, 'saturation': 0.3, 'volume': 1.4, 'compression': True},
    
    # Echo/reverb
    'echo': {'reverb': 0.5, 'echo': True},
    
    # Underwater (heavy lowpass, slight pitch warble)
    'underwater': {'lowpass': 1200, 'volume': 0.75, 'reverb': 0.3, 'pitch_wobble': 0.08},
    
    # Laughing sounds (warmer, brighter)
    'laugh': {'brightness': 0.2, 'volume': 1.15},
    'laughing': {'brightness': 0.25, 'volume': 1.2},
    'giggle': {'brightness': 0.3, 'volume': 1.1, 'highpass': 150},
    
    # Scream/Shriek
    'scream': {'volume': 1.5, 'saturation': 0.2, 'highpass': 200},
    'shriek': {'volume': 1.4, 'saturation': 0.15, 'highpass': 300, 'brightness': 0.3},
}


def apply_audio_effects(audio_data: np.ndarray, sample_rate: int, effects: Dict) -> np.ndarray:
    """
    Apply audio effects to numpy audio data.
    
    Supported effects:
    - volume: Multiply amplitude (0.0-2.0)
    - lowpass: Low-pass filter cutoff frequency in Hz
    - highpass: High-pass filter cutoff frequency in Hz
    - saturation: Soft clipping (0.0-1.0)
    - tremolo: Volume wobble {'rate': Hz, 'depth': 0.0-1.0}
    - reverb: Simple reverb amount (0.0-1.0)
    - bitcrush: Bit depth reduction (1-16)
    - noise: Add subtle background noise (0.0-0.1)
    - brightness: Boost high frequencies (0.0-1.0)
    - compression: Apply dynamic range compression
    - pitch_wobble: Random pitch variation (0.0-0.3)
    - echo: Add distinct echo effect
    - shimmer: Add ethereal shimmer effect
    """
    from scipy import signal as scipy_signal
    
    if not effects:
        return audio_data
    
    audio = audio_data.astype(np.float32)
    
    # Normalize to -1 to 1 range if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    
    # Pitch wobble (random slight pitch variations for scared/underwater)
    if 'pitch_wobble' in effects:
        import librosa
        amount = effects['pitch_wobble']
        # Create random pitch curve
        wobble_rate = 3  # Hz
        t = np.arange(len(audio)) / sample_rate
        wobble = amount * np.sin(2 * np.pi * wobble_rate * t + np.random.random() * 2 * np.pi)
        # Apply time-varying pitch shift (approximated with amplitude modulation + slight resampling)
        # This is a simplified approximation
        mod_factor = 1 + wobble * 0.02
        indices = np.clip(np.arange(len(audio)) * np.interp(np.arange(len(audio)), 
                         np.arange(0, len(audio), 100), mod_factor[::100]), 0, len(audio)-1).astype(int)
        audio = audio[indices]
    
    # High-pass filter (remove low rumble)
    if 'highpass' in effects:
        cutoff = effects['highpass']
        nyquist = sample_rate / 2
        if cutoff < nyquist:
            b, a = scipy_signal.butter(3, cutoff / nyquist, btype='high')
            audio = scipy_signal.filtfilt(b, a, audio)
    
    # Low-pass filter (muffle sound)
    if 'lowpass' in effects:
        cutoff = effects['lowpass']
        nyquist = sample_rate / 2
        if cutoff < nyquist:
            b, a = scipy_signal.butter(3, cutoff / nyquist, btype='low')
            audio = scipy_signal.filtfilt(b, a, audio)
    
    # Brightness boost (enhance high frequencies)
    if 'brightness' in effects:
        amount = effects['brightness']
        # High shelf boost
        nyquist = sample_rate / 2
        cutoff = 3000 / nyquist
        if cutoff < 1:
            b, a = scipy_signal.butter(2, cutoff, btype='high')
            high_freq = scipy_signal.filtfilt(b, a, audio)
            audio = audio + high_freq * amount
    
    # Saturation (soft clipping for angry/distorted sound)
    if 'saturation' in effects:
        amount = effects['saturation']
        # More aggressive soft clipping using tanh
        drive = 1 + amount * 5
        audio = np.tanh(audio * drive) / np.tanh(drive)
    
    # Compression (dynamic range compression for shouting/megaphone)
    if effects.get('compression'):
        threshold = 0.5
        ratio = 4.0
        # Simple compression
        mask = np.abs(audio) > threshold
        audio[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
    
    # Tremolo (volume wobble for scared/nervous)
    if 'tremolo' in effects:
        rate = effects['tremolo'].get('rate', 5)  # Hz
        depth = effects['tremolo'].get('depth', 0.2)  # 0-1
        t = np.arange(len(audio)) / sample_rate
        # Use combination of sine waves for more natural tremolo
        modulation = 1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t) * 
                                  (1 + 0.3 * np.sin(2 * np.pi * rate * 1.5 * t)))
        audio = audio * modulation
    
    # Add noise (for radio/whisper breathiness)
    if 'noise' in effects:
        amount = effects['noise']
        noise = np.random.randn(len(audio)) * amount
        audio = audio + noise
    
    # Simple reverb (convolution with exponential decay)
    if 'reverb' in effects:
        amount = effects['reverb']
        reverb_time = 0.5  # seconds (increased from 0.3)
        reverb_samples = int(reverb_time * sample_rate)
        impulse = np.exp(-np.linspace(0, 6, reverb_samples))
        impulse = impulse / np.sum(impulse)  # Normalize
        reverb_signal = np.convolve(audio, impulse, mode='full')[:len(audio)]
        audio = audio * (1 - amount) + reverb_signal * amount
    
    # Echo effect (distinct delayed repeat)
    if effects.get('echo'):
        delay_time = 0.25  # seconds
        decay = 0.4
        delay_samples = int(delay_time * sample_rate)
        echo_signal = np.zeros_like(audio)
        if delay_samples < len(audio):
            echo_signal[delay_samples:] = audio[:-delay_samples] * decay
            # Add second echo
            if delay_samples * 2 < len(audio):
                echo_signal[delay_samples*2:] += audio[:-delay_samples*2] * decay * 0.5
            audio = audio + echo_signal
    
    # Shimmer effect (ethereal octave shimmer)
    if 'shimmer' in effects:
        import librosa
        amount = effects['shimmer']
        # Create octave-up shimmer
        shimmer = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=12)
        # Apply envelope following
        envelope = np.abs(audio)
        envelope = scipy_signal.filtfilt(*scipy_signal.butter(2, 10/(sample_rate/2)), envelope)
        shimmer = shimmer * envelope * amount
        audio = audio + shimmer
    
    # Bitcrush (lo-fi effect)
    if 'bitcrush' in effects:
        bits = effects['bitcrush']
        levels = 2 ** bits
        audio = np.round(audio * levels) / levels
    
    # Volume adjustment (apply last)
    if 'volume' in effects:
        audio = audio * effects['volume']
    
    # Final normalization to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0.95:
        audio = audio * (0.95 / max_val)
    
    # Clip to prevent any remaining distortion
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def get_emotion_effects(emotion: Optional[str]) -> Dict:
    """Get audio effects for an emotion."""
    if not emotion:
        return {}
    return EMOTION_EFFECTS.get(emotion.lower(), {})


def strip_emotion_tags(text: str) -> str:
    """
    Strip all emotion/effect tags from text, leaving just the plain spoken content.
    This is used to clean text before sending to TTS.
    """
    # Remove [emotion]...[/emotion] tags, keeping inner text
    text = re.sub(r'\[(\w+)\](.*?)\[/\1\]', r'\2', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove self-closing [emotion] or [emotion/] tags
    text = re.sub(r'\[(\w+)/?\]', '', text, flags=re.IGNORECASE)
    # Remove *action* tags
    text = re.sub(r'\*(\w+)\*', '', text, flags=re.IGNORECASE)
    # Remove (action) tags
    text = re.sub(r'\((\w+)\)', '', text, flags=re.IGNORECASE)
    # Remove <speed>...</speed> tags, keeping inner text
    text = re.sub(r'<speed\s+[^>]+>(.*?)</speed>', r'\1', text, flags=re.IGNORECASE | re.DOTALL)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_emotion_tags(text: str, parent_include: dict = None) -> List[Dict]:
    """
    Parse text with emotion tags and return segments with their emotions.
    
    Supported formats:
    - [happy] text [/happy] - Tagged sections
    - [laugh] - Sound effects (self-closing)
    - *laughing* - Action asterisks
    - (sigh) - Parenthetical actions
    - <speed rate="-20%">slow text</speed> - Speed control
    - <include voice_model_id="5">other voice text</include> - Multi-voice
    
    Returns list of segments: [{'text': str, 'emotion': str or None, 'rate': str or None, 'include': dict or None}, ...]
    """
    segments = []
    
    # First, handle <speed> tags - extract them before other processing
    # Pattern: <speed rate="value">text</speed> or <speed value="value">text</speed>
    speed_pattern = r'<speed\s+(?:rate|value)=["\']?([^"\'>\s]+)["\']?\s*>(.*?)</speed>'
    
    # Pattern: <include ...attributes...>text</include>
    include_pattern = r'<include\s+([^>]+)>(.*?)</include>'
    
    # Pattern for [emotion] text [/emotion] or [emotion/]
    tag_pattern = r'\[(\w+)\](.*?)\[/\1\]|\[(\w+)/?\]'
    # Pattern for *action* 
    action_pattern = r'\*(\w+)\*'
    # Pattern for (action)
    paren_pattern = r'\((\w+)\)'
    
    # Combined pattern to find all emotion markers plus speed and include tags
    combined_pattern = r'<speed\s+(?:rate|value)=["\']?([^"\'>\s]+)["\']?\s*>(.*?)</speed>|<include\s+([^>]+)>(.*?)</include>|\[(\w+)\](.*?)\[/\5\]|\[(\w+)/?\]|\*(\w+)\*|\((\w+)\)'
    
    last_end = 0
    
    for match in re.finditer(combined_pattern, text, re.IGNORECASE | re.DOTALL):
        # Add any text before this match as neutral
        if match.start() > last_end:
            plain_text = text[last_end:match.start()].strip()
            # Strip any remaining tags from plain text
            plain_text = strip_emotion_tags(plain_text)
            if plain_text:
                segments.append({'text': plain_text, 'emotion': None, 'rate': None, 'include': parent_include})
        
        # Determine which pattern matched
        if match.group(1) and match.group(2):  # <speed rate="value">text</speed>
            rate_value = match.group(1)
            inner_text = match.group(2).strip()
            # Strip any emotion tags from inner text
            inner_text = strip_emotion_tags(inner_text)
            # Ensure rate has +/- prefix and % suffix
            if not rate_value.startswith(('+', '-')):
                rate_value = f"+{rate_value}" if not rate_value.startswith('-') else rate_value
            if not rate_value.endswith('%'):
                rate_value = f"{rate_value}%"
            if inner_text:
                segments.append({'text': inner_text, 'emotion': None, 'rate': rate_value, 'include': parent_include})
        elif match.group(3) and match.group(4):  # <include attributes>text</include>
            attrs_str = match.group(3)
            inner_text = match.group(4).strip()
            # Parse attributes like voice_model_id="5" model_path="..." etc
            include_attrs = {}
            attr_pattern = r'(\w+)=["\']?([^"\'>\s]+)["\']?'
            for attr_match in re.finditer(attr_pattern, attrs_str):
                key = attr_match.group(1)
                value = attr_match.group(2)
                # Convert numeric values
                if key in ['voice_model_id', 'f0_up_key']:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                elif key == 'index_rate':
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                include_attrs[key] = value
            if inner_text:
                # Recursively parse inner text for emotions/speed tags, preserving include attrs
                inner_segments = parse_emotion_tags(inner_text, parent_include=include_attrs)
                segments.extend(inner_segments)
        elif match.group(5) and match.group(6):  # [emotion]text[/emotion]
            emotion = match.group(5).lower()
            inner_text = match.group(6).strip()
            # Strip any nested tags from the inner text
            inner_text = strip_emotion_tags(inner_text)
            if inner_text:
                segments.append({'text': inner_text, 'emotion': emotion, 'rate': None, 'include': parent_include})
        elif match.group(7):  # [emotion] or [emotion/] - self-closing/sound effect
            emotion = match.group(7).lower()
            if emotion in SOUND_REPLACEMENTS:
                segments.append({'text': SOUND_REPLACEMENTS[emotion], 'emotion': emotion, 'rate': None, 'include': parent_include})
            else:
                # Just an emotion marker, apply to next segment
                pass
        elif match.group(8):  # *action*
            action = match.group(8).lower()
            if action in SOUND_REPLACEMENTS:
                segments.append({'text': SOUND_REPLACEMENTS[action], 'emotion': action, 'rate': None, 'include': parent_include})
            elif action in EMOTION_PRESETS:
                # It's an emotion, mark but no text
                pass
        elif match.group(9):  # (action)
            action = match.group(9).lower()
            if action in SOUND_REPLACEMENTS:
                segments.append({'text': SOUND_REPLACEMENTS[action], 'emotion': action, 'rate': None, 'include': parent_include})
        
        last_end = match.end()
    
    # Add remaining text
    if last_end < len(text):
        remaining = text[last_end:].strip()
        # Strip any remaining tags
        remaining = strip_emotion_tags(remaining)
        if remaining:
            segments.append({'text': remaining, 'emotion': None, 'rate': None, 'include': parent_include})
    
    # If no segments were found, return the original text (stripped of tags)
    if not segments:
        clean_text = strip_emotion_tags(text)
        if clean_text:
            segments.append({'text': clean_text, 'emotion': None, 'rate': None, 'include': parent_include})
    
    return segments


def get_emotion_prosody(emotion: Optional[str], base_rate: str = "+0%", base_pitch: str = "+0Hz") -> Tuple[str, str]:
    """
    Get prosody adjustments for an emotion, combining with base adjustments.
    
    Returns (rate, pitch) strings for Edge TTS.
    """
    if not emotion or emotion not in EMOTION_PRESETS:
        return base_rate, base_pitch
    
    preset = EMOTION_PRESETS[emotion]
    
    # Parse base values
    def parse_adjustment(val: str) -> int:
        val = val.strip()
        if val.endswith('%'):
            return int(val[:-1].replace('+', ''))
        elif val.endswith('Hz'):
            return int(val[:-2].replace('+', ''))
        return 0
    
    def format_rate(val: int) -> str:
        return f"+{val}%" if val >= 0 else f"{val}%"
    
    def format_pitch(val: int) -> str:
        return f"+{val}Hz" if val >= 0 else f"{val}Hz"
    
    base_rate_val = parse_adjustment(base_rate)
    base_pitch_val = parse_adjustment(base_pitch)
    emotion_rate_val = parse_adjustment(preset['rate'])
    emotion_pitch_val = parse_adjustment(preset['pitch'])
    
    # Combine adjustments
    final_rate = base_rate_val + emotion_rate_val
    final_pitch = base_pitch_val + emotion_pitch_val
    
    return format_rate(final_rate), format_pitch(final_pitch)


async def generate_tts_audio(
    text: str,
    voice: str = "en-US-GuyNeural",
    style: str = "default",
    rate: str = "+0%",
    pitch: str = "+0Hz"
) -> Tuple[bytes, int]:
    """
    Generate TTS audio using Edge TTS with emotion/action tag support.
    
    Supports emotion tags in text:
    - [happy]Hello![/happy] - Tagged sections with emotions
    - [laugh] or *laugh* or (laugh) - Sound effects
    - [sad]I'm so sorry[/sad] - Sad tone
    - <speed rate="-30%">Slow speech</speed> - Speed control
    
    Returns:
        Tuple of (audio_bytes, sample_rate)
    """
    try:
        import edge_tts
        import librosa
        
        # Fix rate format - Edge TTS requires +/- prefix
        if rate and not rate.startswith(('+', '-')):
            rate = f"+{rate}"
        
        # Fix pitch format - Edge TTS requires +/- prefix  
        if pitch and not pitch.startswith(('+', '-')):
            pitch = f"+{pitch}"
        
        # Parse text for emotion tags
        segments = parse_emotion_tags(text)
        logger.info(f"Parsed {len(segments)} segment(s) from text")
        
        # Check if we have any segments with special handling needed
        has_special_segments = len(segments) > 1 or any(
            seg.get('emotion') or seg.get('rate') or seg.get('include') 
            for seg in segments
        )
        
        if has_special_segments:
            audio_chunks = []
            target_sr = None
            
            for i, segment in enumerate(segments):
                segment_text = segment['text']
                emotion = segment.get('emotion')
                segment_rate = segment.get('rate')  # Override rate from <speed> tag
                include_info = segment.get('include')
                
                # Skip include segments - they'll be handled separately with voice conversion
                if include_info:
                    # For include segments, we still generate TTS but mark them for later conversion
                    segment['needs_conversion'] = True
                
                # Determine the rate for this segment
                if segment_rate:
                    # Use the rate from <speed> tag
                    seg_rate = segment_rate
                    seg_pitch = pitch  # Keep original pitch
                    # Still apply emotion prosody on top of custom rate
                    if emotion:
                        _, emotion_pitch = get_emotion_prosody(emotion, "+0%", pitch)
                        seg_pitch = emotion_pitch
                else:
                    # Get prosody for this emotion
                    seg_rate, seg_pitch = get_emotion_prosody(emotion, rate, pitch)
                
                # Get audio effects for this emotion
                effects = get_emotion_effects(emotion)
                effects_str = ', '.join(effects.keys()) if effects else 'none'
                
                logger.info(f"Segment {i+1}: emotion={emotion}, rate={seg_rate}, pitch={seg_pitch}, effects=[{effects_str}], text='{segment_text[:50]}...'")
                
                # Generate this segment
                communicate = edge_tts.Communicate(segment_text, voice, rate=seg_rate, pitch=seg_pitch)
                
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    await communicate.save(tmp_path)
                    audio, sr = librosa.load(tmp_path, sr=None, mono=True)
                    
                    if target_sr is None:
                        target_sr = sr
                    elif sr != target_sr:
                        # Resample to target sample rate
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    
                    # Apply audio effects for this emotion
                    if effects:
                        audio = apply_audio_effects(audio, target_sr, effects)
                    
                    # Store segment info for potential include processing
                    audio_chunks.append({
                        'audio': audio,
                        'include': include_info,
                        'text': segment_text
                    })
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            # Concatenate all audio chunks
            if audio_chunks:
                combined_audio = np.concatenate([chunk['audio'] for chunk in audio_chunks])
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, combined_audio, target_sr, format='WAV')
                wav_buffer.seek(0)
                return wav_buffer.read(), target_sr
            else:
                raise HTTPException(status_code=500, detail="No audio generated")
        
        else:
            # Simple case - no emotion tags, generate directly
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                await communicate.save(tmp_path)
                audio, sr = librosa.load(tmp_path, sr=None, mono=True)
                
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, audio, sr, format='WAV')
                wav_buffer.seek(0)
                
                return wav_buffer.read(), sr
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="edge-tts not installed. Install with: pip install edge-tts"
        )
    except Exception as e:
        logger.exception(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


# =============================================================================
# YouTube / Song Search API Endpoints
# =============================================================================

class YouTubeSearchRequest(BaseModel):
    """YouTube search request"""
    query: str = Field(..., description="Search query (artist, song name, etc.)", max_length=200)
    max_results: int = Field(default=10, ge=1, le=25, description="Maximum results to return")


class YouTubeSearchResult(BaseModel):
    """Single search result"""
    id: str
    title: str
    artist: str
    duration: int
    thumbnail: str
    url: str
    view_count: int
    is_cached: bool = False


class YouTubeSearchResponse(BaseModel):
    """YouTube search response"""
    results: List[YouTubeSearchResult]
    query: str


class YouTubeDownloadRequest(BaseModel):
    """YouTube audio download request"""
    video_id: str = Field(..., description="YouTube video ID")
    use_cache: bool = Field(default=True, description="Use cached audio if available")


class YouTubeDownloadResponse(BaseModel):
    """YouTube download response"""
    audio: str = Field(..., description="Base64 encoded WAV audio")
    sample_rate: int
    video_id: str
    title: str = ""
    artist: str = ""
    duration: int = 0


@app.post("/youtube/search", response_model=YouTubeSearchResponse)
async def youtube_search(request: YouTubeSearchRequest):
    """
    Search YouTube for songs.
    
    Returns a list of search results with video IDs that can be used
    with the /youtube/download endpoint.
    """
    if not YOUTUBE_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="YouTube service not available. Install yt-dlp: pip install yt-dlp"
        )
    
    try:
        results = await search_youtube(request.query, request.max_results)
        
        return YouTubeSearchResponse(
            results=[
                YouTubeSearchResult(
                    id=r.id,
                    title=r.title,
                    artist=r.artist,
                    duration=r.duration,
                    thumbnail=r.thumbnail,
                    url=r.url,
                    view_count=r.view_count,
                    is_cached=is_cached(r.id)
                )
                for r in results
            ],
            query=request.query
        )
    except Exception as e:
        logger.exception(f"YouTube search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/youtube/download", response_model=YouTubeDownloadResponse)
async def youtube_download(request: YouTubeDownloadRequest):
    """
    Download audio from a YouTube video.
    
    Returns base64-encoded WAV audio that can be used directly with
    the /audio/process endpoint for vocal splitting or swapping.
    """
    if not YOUTUBE_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="YouTube service not available. Install yt-dlp: pip install yt-dlp"
        )
    
    try:
        # Get video info first
        info = await get_video_info(request.video_id)
        
        # Download audio
        audio_bytes, sample_rate = await download_youtube_audio(
            request.video_id, 
            use_cache=request.use_cache
        )
        
        return YouTubeDownloadResponse(
            audio=base64.b64encode(audio_bytes).decode('utf-8'),
            sample_rate=sample_rate,
            video_id=request.video_id,
            title=info.get('title', ''),
            artist=info.get('artist', ''),
            duration=info.get('duration', 0)
        )
        
    except Exception as e:
        logger.exception(f"YouTube download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/youtube/info/{video_id}")
async def youtube_info(video_id: str):
    """Get information about a YouTube video"""
    if not YOUTUBE_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="YouTube service not available. Install yt-dlp: pip install yt-dlp"
        )
    
    try:
        info = await get_video_info(video_id)
        info['is_cached'] = is_cached(video_id)
        return info
    except Exception as e:
        logger.exception(f"Failed to get video info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get info: {str(e)}")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager is not None and model_manager.model_name is not None,
        current_model=model_manager.model_name if model_manager else None
    )


@app.get("/emotions")
async def list_emotions():
    """
    List available emotion tags for TTS.
    
    Use these in your text like:
    - [happy]Hello there![/happy]
    - [laugh] or *laugh* 
    - [sad]I'm sorry to hear that[/sad]
    - [robot]Beep boop[/robot] - Voice effect
    """
    emotions_by_category = {
        'positive': {name: preset['desc'] for name, preset in EMOTION_PRESETS.items() 
                     if name in ['happy', 'excited', 'cheerful', 'joyful']},
        'negative': {name: preset['desc'] for name, preset in EMOTION_PRESETS.items() 
                     if name in ['sad', 'melancholy', 'depressed', 'disappointed']},
        'angry': {name: preset['desc'] for name, preset in EMOTION_PRESETS.items() 
                  if name in ['angry', 'furious', 'annoyed', 'frustrated']},
        'calm': {name: preset['desc'] for name, preset in EMOTION_PRESETS.items() 
                 if name in ['calm', 'peaceful', 'relaxed', 'neutral']},
        'surprised': {name: preset['desc'] for name, preset in EMOTION_PRESETS.items() 
                      if name in ['surprised', 'shocked', 'amazed']},
        'fear': {name: preset['desc'] for name, preset in EMOTION_PRESETS.items() 
                 if name in ['scared', 'terrified', 'anxious', 'nervous']},
        'special': {name: preset['desc'] for name, preset in EMOTION_PRESETS.items() 
                    if name in ['whisper', 'shouting', 'sarcastic', 'romantic', 'serious', 'playful', 'dramatic', 'mysterious']},
        'effects': {name: preset['desc'] for name, preset in EMOTION_PRESETS.items() 
                    if name in ['robot', 'spooky', 'ethereal', 'phone', 'radio', 'megaphone', 'echo', 'underwater']},
        'sounds': {name: SOUND_REPLACEMENTS.get(name, 'sound effect') for name in SOUND_REPLACEMENTS.keys()},
    }
    
    # Show what audio processing each effect applies
    effects_details = {}
    for name, effect_config in EMOTION_EFFECTS.items():
        desc = EMOTION_PRESETS.get(name, {}).get('desc', '')
        effects_details[name] = {
            'description': desc,
            'processing': list(effect_config.keys())
        }
    
    return {
        'emotions': emotions_by_category,
        'audio_effects': effects_details,
        'usage': {
            'tagged': '[emotion]Your text here[/emotion]',
            'sound_bracket': '[laugh]',
            'sound_asterisk': '*laugh*',
            'sound_paren': '(sigh)',
        },
        'examples': [
            '[happy]I am so excited to see you![/happy]',
            '[sad]I miss you so much[/sad]',
            'Hello! *laugh* That was funny!',
            '[whisper]This is a secret[/whisper]',
            'Oh no! (gasp) What happened?',
            '[excited]We won the game![/excited] [laugh]',
            '[robot]I am a robot. Beep boop.[/robot]',
            '[spooky]The ghost haunts this place...[/spooky]',
            '[megaphone]Attention please![/megaphone]',
        ]
    }


@app.get("/voices")
async def list_voices():
    """List available TTS voices with expanded language support"""
    voices = {
        # English - US
        'en-US-GuyNeural': {'name': 'Guy', 'language': 'English (US)', 'gender': 'male', 'supports_styles': True},
        'en-US-JennyNeural': {'name': 'Jenny', 'language': 'English (US)', 'gender': 'female', 'supports_styles': True},
        'en-US-AriaNeural': {'name': 'Aria', 'language': 'English (US)', 'gender': 'female', 'supports_styles': True},
        'en-US-DavisNeural': {'name': 'Davis', 'language': 'English (US)', 'gender': 'male', 'supports_styles': True},
        'en-US-TonyNeural': {'name': 'Tony', 'language': 'English (US)', 'gender': 'male', 'supports_styles': True},
        'en-US-SaraNeural': {'name': 'Sara', 'language': 'English (US)', 'gender': 'female', 'supports_styles': True},
        # English - UK
        'en-GB-RyanNeural': {'name': 'Ryan', 'language': 'English (UK)', 'gender': 'male', 'supports_styles': False},
        'en-GB-SoniaNeural': {'name': 'Sonia', 'language': 'English (UK)', 'gender': 'female', 'supports_styles': False},
        'en-GB-LibbyNeural': {'name': 'Libby', 'language': 'English (UK)', 'gender': 'female', 'supports_styles': False},
        # English - Australia
        'en-AU-NatashaNeural': {'name': 'Natasha', 'language': 'English (AU)', 'gender': 'female', 'supports_styles': False},
        'en-AU-WilliamNeural': {'name': 'William', 'language': 'English (AU)', 'gender': 'male', 'supports_styles': False},
        # Spanish
        'es-ES-AlvaroNeural': {'name': 'Alvaro', 'language': 'Spanish (Spain)', 'gender': 'male', 'supports_styles': False},
        'es-ES-ElviraNeural': {'name': 'Elvira', 'language': 'Spanish (Spain)', 'gender': 'female', 'supports_styles': False},
        'es-MX-DaliaNeural': {'name': 'Dalia', 'language': 'Spanish (Mexico)', 'gender': 'female', 'supports_styles': False},
        'es-MX-JorgeNeural': {'name': 'Jorge', 'language': 'Spanish (Mexico)', 'gender': 'male', 'supports_styles': False},
        # French
        'fr-FR-HenriNeural': {'name': 'Henri', 'language': 'French (France)', 'gender': 'male', 'supports_styles': False},
        'fr-FR-DeniseNeural': {'name': 'Denise', 'language': 'French (France)', 'gender': 'female', 'supports_styles': False},
        'fr-CA-SylvieNeural': {'name': 'Sylvie', 'language': 'French (Canada)', 'gender': 'female', 'supports_styles': False},
        'fr-CA-JeanNeural': {'name': 'Jean', 'language': 'French (Canada)', 'gender': 'male', 'supports_styles': False},
        # German
        'de-DE-ConradNeural': {'name': 'Conrad', 'language': 'German', 'gender': 'male', 'supports_styles': False},
        'de-DE-KatjaNeural': {'name': 'Katja', 'language': 'German', 'gender': 'female', 'supports_styles': False},
        # Japanese
        'ja-JP-KeitaNeural': {'name': 'Keita', 'language': 'Japanese', 'gender': 'male', 'supports_styles': False},
        'ja-JP-NanamiNeural': {'name': 'Nanami', 'language': 'Japanese', 'gender': 'female', 'supports_styles': False},
        # Chinese
        'zh-CN-YunxiNeural': {'name': 'Yunxi', 'language': 'Chinese (Mandarin)', 'gender': 'male', 'supports_styles': True},
        'zh-CN-XiaoxiaoNeural': {'name': 'Xiaoxiao', 'language': 'Chinese (Mandarin)', 'gender': 'female', 'supports_styles': True},
        # Korean
        'ko-KR-InJoonNeural': {'name': 'InJoon', 'language': 'Korean', 'gender': 'male', 'supports_styles': False},
        'ko-KR-SunHiNeural': {'name': 'SunHi', 'language': 'Korean', 'gender': 'female', 'supports_styles': False},
        # Russian
        'ru-RU-DmitryNeural': {'name': 'Dmitry', 'language': 'Russian', 'gender': 'male', 'supports_styles': False},
        'ru-RU-SvetlanaNeural': {'name': 'Svetlana', 'language': 'Russian', 'gender': 'female', 'supports_styles': False},
        # Portuguese - Brazil
        'pt-BR-AntonioNeural': {'name': 'Antonio', 'language': 'Portuguese (Brazil)', 'gender': 'male', 'supports_styles': False},
        'pt-BR-FranciscaNeural': {'name': 'Francisca', 'language': 'Portuguese (Brazil)', 'gender': 'female', 'supports_styles': False},
        # Portuguese - Portugal
        'pt-PT-DuarteNeural': {'name': 'Duarte', 'language': 'Portuguese (Portugal)', 'gender': 'male', 'supports_styles': False},
        'pt-PT-RaquelNeural': {'name': 'Raquel', 'language': 'Portuguese (Portugal)', 'gender': 'female', 'supports_styles': False},
        # Italian
        'it-IT-DiegoNeural': {'name': 'Diego', 'language': 'Italian', 'gender': 'male', 'supports_styles': False},
        'it-IT-ElsaNeural': {'name': 'Elsa', 'language': 'Italian', 'gender': 'female', 'supports_styles': False},
        # Dutch
        'nl-NL-MaartenNeural': {'name': 'Maarten', 'language': 'Dutch', 'gender': 'male', 'supports_styles': False},
        'nl-NL-ColetteNeural': {'name': 'Colette', 'language': 'Dutch', 'gender': 'female', 'supports_styles': False},
        # Polish
        'pl-PL-MarekNeural': {'name': 'Marek', 'language': 'Polish', 'gender': 'male', 'supports_styles': False},
        'pl-PL-ZofiaNeural': {'name': 'Zofia', 'language': 'Polish', 'gender': 'female', 'supports_styles': False},
        # Swedish
        'sv-SE-MattiasNeural': {'name': 'Mattias', 'language': 'Swedish', 'gender': 'male', 'supports_styles': False},
        'sv-SE-SofieNeural': {'name': 'Sofie', 'language': 'Swedish', 'gender': 'female', 'supports_styles': False},
        # Norwegian
        'nb-NO-FinnNeural': {'name': 'Finn', 'language': 'Norwegian', 'gender': 'male', 'supports_styles': False},
        'nb-NO-PernilleNeural': {'name': 'Pernille', 'language': 'Norwegian', 'gender': 'female', 'supports_styles': False},
        # Danish
        'da-DK-JeppeNeural': {'name': 'Jeppe', 'language': 'Danish', 'gender': 'male', 'supports_styles': False},
        'da-DK-ChristelNeural': {'name': 'Christel', 'language': 'Danish', 'gender': 'female', 'supports_styles': False},
        # Finnish
        'fi-FI-HarriNeural': {'name': 'Harri', 'language': 'Finnish', 'gender': 'male', 'supports_styles': False},
        'fi-FI-NooraNeural': {'name': 'Noora', 'language': 'Finnish', 'gender': 'female', 'supports_styles': False},
        # Icelandic
        'is-IS-GunnarNeural': {'name': 'Gunnar', 'language': 'Icelandic', 'gender': 'male', 'supports_styles': False},
        'is-IS-GudrunNeural': {'name': 'Guðrún', 'language': 'Icelandic', 'gender': 'female', 'supports_styles': False},
        # Arabic
        'ar-SA-HamedNeural': {'name': 'Hamed', 'language': 'Arabic (Saudi)', 'gender': 'male', 'supports_styles': False},
        'ar-SA-ZariyahNeural': {'name': 'Zariyah', 'language': 'Arabic (Saudi)', 'gender': 'female', 'supports_styles': False},
        # Hindi
        'hi-IN-MadhurNeural': {'name': 'Madhur', 'language': 'Hindi', 'gender': 'male', 'supports_styles': False},
        'hi-IN-SwaraNeural': {'name': 'Swara', 'language': 'Hindi', 'gender': 'female', 'supports_styles': False},
        # Thai
        'th-TH-NiwatNeural': {'name': 'Niwat', 'language': 'Thai', 'gender': 'male', 'supports_styles': False},
        'th-TH-PremwadeeNeural': {'name': 'Premwadee', 'language': 'Thai', 'gender': 'female', 'supports_styles': False},
        # Vietnamese
        'vi-VN-NamMinhNeural': {'name': 'Nam Minh', 'language': 'Vietnamese', 'gender': 'male', 'supports_styles': False},
        'vi-VN-HoaiMyNeural': {'name': 'Hoài My', 'language': 'Vietnamese', 'gender': 'female', 'supports_styles': False},
        # Indonesian
        'id-ID-ArdiNeural': {'name': 'Ardi', 'language': 'Indonesian', 'gender': 'male', 'supports_styles': False},
        'id-ID-GadisNeural': {'name': 'Gadis', 'language': 'Indonesian', 'gender': 'female', 'supports_styles': False},
        # Turkish
        'tr-TR-AhmetNeural': {'name': 'Ahmet', 'language': 'Turkish', 'gender': 'male', 'supports_styles': False},
        'tr-TR-EmelNeural': {'name': 'Emel', 'language': 'Turkish', 'gender': 'female', 'supports_styles': False},
        # Greek
        'el-GR-NestorasNeural': {'name': 'Nestoras', 'language': 'Greek', 'gender': 'male', 'supports_styles': False},
        'el-GR-AthinaNeural': {'name': 'Athina', 'language': 'Greek', 'gender': 'female', 'supports_styles': False},
        # Hebrew
        'he-IL-AvriNeural': {'name': 'Avri', 'language': 'Hebrew', 'gender': 'male', 'supports_styles': False},
        'he-IL-HilaNeural': {'name': 'Hila', 'language': 'Hebrew', 'gender': 'female', 'supports_styles': False},
        # Czech
        'cs-CZ-AntoninNeural': {'name': 'Antonín', 'language': 'Czech', 'gender': 'male', 'supports_styles': False},
        'cs-CZ-VlastaNeural': {'name': 'Vlasta', 'language': 'Czech', 'gender': 'female', 'supports_styles': False},
        # Hungarian
        'hu-HU-TamasNeural': {'name': 'Tamás', 'language': 'Hungarian', 'gender': 'male', 'supports_styles': False},
        'hu-HU-NoemiNeural': {'name': 'Noémi', 'language': 'Hungarian', 'gender': 'female', 'supports_styles': False},
        # Romanian
        'ro-RO-EmilNeural': {'name': 'Emil', 'language': 'Romanian', 'gender': 'male', 'supports_styles': False},
        'ro-RO-AlinaNeural': {'name': 'Alina', 'language': 'Romanian', 'gender': 'female', 'supports_styles': False},
        # Ukrainian
        'uk-UA-OstapNeural': {'name': 'Ostap', 'language': 'Ukrainian', 'gender': 'male', 'supports_styles': False},
        'uk-UA-PolinaNeural': {'name': 'Polina', 'language': 'Ukrainian', 'gender': 'female', 'supports_styles': False},
    }
    
    styles = {
        'default': {'name': 'Default', 'description': 'Normal speaking voice'},
        'cheerful': {'name': 'Cheerful', 'description': 'Expresses a positive and happy tone'},
        'sad': {'name': 'Sad', 'description': 'Expresses a sorrowful tone'},
        'angry': {'name': 'Angry', 'description': 'Expresses an angry and annoyed tone'},
        'fearful': {'name': 'Fearful', 'description': 'Expresses a scared and nervous tone'},
        'friendly': {'name': 'Friendly', 'description': 'Expresses a warm and pleasant tone'},
        'whispering': {'name': 'Whispering', 'description': 'Speaks softly with a whisper'},
        'shouting': {'name': 'Shouting', 'description': 'Speaks loudly with emphasis'},
        'excited': {'name': 'Excited', 'description': 'Expresses an upbeat and enthusiastic tone'},
        'hopeful': {'name': 'Hopeful', 'description': 'Expresses a warm and hoping tone'},
        'narration-professional': {'name': 'Narration', 'description': 'Neutral narration style'},
        'newscast-casual': {'name': 'Newscast', 'description': 'Casual news reading'},
        'customerservice': {'name': 'Customer Service', 'description': 'Friendly customer service voice'},
        'chat': {'name': 'Chat', 'description': 'Casual conversational style'},
        'assistant': {'name': 'Assistant', 'description': 'Warm and helpful assistant voice'},
    }
    
    # Get unique languages sorted
    languages = sorted(list(set(v['language'] for v in voices.values())))
    
    return {
        "voices": [{"id": k, **v} for k, v in voices.items()],
        "styles": [{"id": k, **v} for k, v in styles.items()],
        "languages": languages
    }


@app.post("/apply-effects")
async def apply_effects_endpoint(request: ApplyEffectsRequest):
    """
    Apply audio effects to existing audio.
    
    Use this to add effects like 'robot', 'whisper', 'terrified' etc. 
    after voice conversion or to any audio.
    """
    try:
        effects = get_emotion_effects(request.effect)
        if not effects:
            # Check if it's a valid emotion name
            if request.effect.lower() in EMOTION_PRESETS:
                # It's a valid emotion but has no special audio effects
                return {
                    'audio': request.audio,  # Return unchanged
                    'sample_rate': request.sample_rate,
                    'format': 'wav',
                    'effect_applied': request.effect,
                    'effects': [],
                    'note': 'This emotion only modifies TTS prosody, no post-processing effects'
                }
            raise HTTPException(status_code=400, detail=f"Unknown effect: {request.effect}")
        
        # Decode audio
        audio_bytes = base64.b64decode(request.audio)
        audio_buffer = io.BytesIO(audio_bytes)
        
        try:
            audio, sr = sf.read(audio_buffer, dtype='float32')
        except Exception:
            import librosa
            audio_buffer = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(audio_buffer, sr=None, mono=True)
        
        # Convert to mono if needed
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Apply effects
        logger.info(f"Applying effects for '{request.effect}': {list(effects.keys())}")
        processed_audio = apply_audio_effects(audio.astype(np.float32), sr, effects)
        
        # Convert back to WAV
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, processed_audio, sr, format='WAV')
        wav_buffer.seek(0)
        
        return {
            'audio': base64.b64encode(wav_buffer.read()).decode('utf-8'),
            'sample_rate': sr,
            'format': 'wav',
            'effect_applied': request.effect,
            'effects': list(effects.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to apply effects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply effects: {str(e)}")


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text with emotion and sound effect support.
    
    Supports two TTS engines:
    1. **Bark TTS** (if available): Native support for emotions and sounds
       - [laughter], [laughs], [sighs], [gasps], [clears throat]
       - ♪ for singing, CAPS for emphasis
       - ... or — for hesitations
    
    2. **Edge TTS** (fallback): Audio processing to simulate emotions
       - Prosody adjustments (rate, pitch) per emotion
       - Audio effects (tremolo, reverb, distortion, etc.)
    
    Tag formats:
    - [happy]text[/happy] - Emotion-wrapped text
    - [laugh] or *laugh* - Sound effects
    - <speed rate="-30%">slow text</speed> - Speed control
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        # Use the new TTS service if available
        if TTS_SERVICE_AVAILABLE:
            audio_bytes, sample_rate = await generate_tts(
                text=request.text,
                voice=request.voice,
                rate=request.rate,
                pitch=request.pitch,
                use_bark=request.use_bark,
                bark_speaker=request.bark_speaker
            )
        else:
            # Fallback to original generate_tts_audio function
            audio_bytes, sample_rate = await generate_tts_audio(
                text=request.text,
                voice=request.voice,
                style=request.style,
                rate=request.rate,
                pitch=request.pitch
            )
        
        return TTSResponse(
            audio=base64.b64encode(audio_bytes).decode('utf-8'),
            sample_rate=sample_rate,
            format="wav"
        )
    except Exception as e:
        logger.exception(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


class TTSCapabilitiesResponse(BaseModel):
    """TTS capabilities response"""
    bark_available: bool = Field(description="Whether Bark TTS is available (native emotions)")
    edge_tts_available: bool = Field(default=True, description="Whether Edge TTS is available (fallback)")
    supported_emotions: List[str] = Field(description="List of supported emotion tags")
    supported_sounds: List[str] = Field(description="List of supported sound effect tags")
    bark_speakers: List[str] = Field(description="Available Bark speaker presets")
    recommendation: str = Field(description="Recommendation for best results")


@app.get("/tts/capabilities", response_model=TTSCapabilitiesResponse)
async def get_tts_capabilities():
    """
    Get information about TTS capabilities and supported emotions/sounds.
    
    Returns whether Bark (native emotions) or Edge TTS (audio processing) is being used,
    along with lists of supported emotion and sound effect tags.
    """
    emotions = [
        'happy', 'excited', 'cheerful', 'joyful',
        'sad', 'melancholy', 'depressed', 'disappointed',
        'angry', 'furious', 'annoyed', 'frustrated',
        'calm', 'peaceful', 'relaxed', 'neutral',
        'surprised', 'shocked', 'amazed',
        'scared', 'terrified', 'anxious', 'nervous',
        'whisper', 'shouting', 'sarcastic', 'romantic',
        'serious', 'playful', 'dramatic', 'mysterious',
        'robot', 'spooky', 'ethereal', 'phone', 'radio',
        'megaphone', 'echo', 'underwater'
    ]
    
    sounds = [
        'laugh', 'laughing', 'giggle', 'chuckle', 'snicker', 'cackle',
        'cry', 'crying', 'sob', 'sniff',
        'gasp', 'scream', 'shriek',
        'groan', 'moan', 'sigh', 'yawn',
        'cough', 'sneeze', 'hiccup', 'burp', 'gulp', 'slurp',
        'growl', 'hiss', 'hum', 'whistle', 'shush', 'kiss', 'blow',
        'pant', 'breathe', 'inhale', 'exhale',
        'stutter', 'mumble', 'stammer',
        'hmm', 'thinking', 'uhh', 'umm',
        'clap', 'snap', 'wow', 'ooh', 'ahh', 'ugh', 'eww',
        'yay', 'boo', 'woohoo', 'ow', 'ouch', 'phew', 'tsk', 'psst'
    ]
    
    bark_speakers = ['default', 'male1', 'male2', 'female1', 'female2', 'dramatic', 'calm']
    
    if TTS_SERVICE_AVAILABLE and BARK_AVAILABLE:
        recommendation = "Bark TTS is active! Your emotions and sound effects like [laugh], [sigh], [gasp] will be natively rendered without reading the tags."
    else:
        recommendation = "Using Edge TTS with audio processing. Emotions are simulated via pitch/rate changes and audio effects. For native emotion support, install Bark: pip install git+https://github.com/suno-ai/bark.git"
    
    return TTSCapabilitiesResponse(
        bark_available=TTS_SERVICE_AVAILABLE and BARK_AVAILABLE,
        edge_tts_available=True,
        supported_emotions=emotions,
        supported_sounds=sounds,
        bark_speakers=bark_speakers,
        recommendation=recommendation
    )


class MultiVoiceTTSRequest(BaseModel):
    """Multi-voice TTS generation request with include segment support"""
    text: str = Field(..., description="Text with optional <include> tags for multi-voice generation", max_length=10000)
    voice: str = Field(default="en-US-GuyNeural", description="Default Edge TTS voice ID")
    style: str = Field(default="default", description="Speaking style/emotion")
    rate: str = Field(default="+0%", description="Default speech rate adjustment")
    pitch: str = Field(default="+0Hz", description="Pitch adjustment")
    # Default voice conversion settings
    default_model_path: Optional[str] = Field(default=None, description="Default model path for main voice")
    default_index_path: Optional[str] = Field(default=None, description="Default index path for main voice")
    default_f0_up_key: int = Field(default=0, description="Default pitch shift")
    default_index_rate: float = Field(default=0.75, description="Default index rate")
    # Include segment voice model mappings (voice_model_id -> paths)
    voice_model_mappings: Optional[Dict[str, dict]] = Field(default=None, description="Mapping of voice_model_id to {model_path, index_path}")


class MultiVoiceTTSResponse(BaseModel):
    """Multi-voice TTS response"""
    audio: str = Field(..., description="Base64 encoded WAV audio")
    sample_rate: int = Field(default=24000)
    format: str = Field(default="wav")
    segments_processed: int = Field(default=1, description="Number of segments processed")
    include_segments_used: int = Field(default=0, description="Number of include segments with different voices")


@app.post("/tts/multi-voice", response_model=MultiVoiceTTSResponse)
async def multi_voice_tts(request: MultiVoiceTTSRequest):
    """
    Generate multi-voice TTS with support for <include> tags.
    
    Each <include voice_model_id="X">text</include> segment will be:
    1. Generated with TTS
    2. Converted using the specified voice model
    3. Concatenated with other segments
    
    The main text (outside include tags) uses the default voice and model.
    """
    global model_manager
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        import edge_tts
        import librosa
        
        # Parse segments including <include> tags
        segments = parse_emotion_tags(request.text)
        logger.info(f"Multi-voice TTS: Parsed {len(segments)} segment(s)")
        
        audio_chunks = []
        target_sr = 24000
        include_count = 0
        
        for i, segment in enumerate(segments):
            segment_text = segment['text']
            emotion = segment.get('emotion')
            segment_rate = segment.get('rate') or request.rate
            include_info = segment.get('include')
            
            # Skip empty segments or segments with only punctuation/whitespace
            if not segment_text or not segment_text.strip():
                logger.info(f"Segment {i+1}: Skipping empty segment")
                continue
            
            # Check if segment has any actual words (not just punctuation)
            import string
            text_without_punct = segment_text.translate(str.maketrans('', '', string.punctuation + '…—–'))
            if not text_without_punct.strip():
                logger.info(f"Segment {i+1}: Skipping punctuation-only segment: '{segment_text}'")
                continue
            
            # Fix rate format
            if segment_rate and not segment_rate.startswith(('+', '-')):
                segment_rate = f"+{segment_rate}"
            
            # Get prosody adjustments
            seg_rate, seg_pitch = get_emotion_prosody(emotion, segment_rate, request.pitch)
            
            # Get audio effects
            effects = get_emotion_effects(emotion)
            
            logger.info(f"Segment {i+1}: emotion={emotion}, rate={seg_rate}, include={include_info is not None}, text='{segment_text[:50]}...'")
            
            # Generate TTS for this segment
            communicate = edge_tts.Communicate(segment_text, request.voice, rate=seg_rate, pitch=seg_pitch)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                await communicate.save(tmp_path)
                audio, sr = librosa.load(tmp_path, sr=None, mono=True)
                
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                
                # Apply emotion effects
                if effects:
                    audio = apply_audio_effects(audio, target_sr, effects)
                
                # Determine which voice model to use for conversion
                model_path = None
                index_path = None
                f0_up_key = request.default_f0_up_key
                index_rate = request.default_index_rate
                
                if include_info:
                    include_count += 1
                    # Use include-specific model if available
                    voice_model_id = str(include_info.get('voice_model_id', ''))
                    if voice_model_id and request.voice_model_mappings and voice_model_id in request.voice_model_mappings:
                        mapping = request.voice_model_mappings[voice_model_id]
                        model_path = mapping.get('model_path')
                        index_path = mapping.get('index_path')
                    elif include_info.get('model_path'):
                        model_path = include_info.get('model_path')
                        index_path = include_info.get('index_path')
                    
                    # Override pitch and index rate if specified
                    if 'f0_up_key' in include_info:
                        f0_up_key = include_info['f0_up_key']
                    if 'index_rate' in include_info:
                        index_rate = include_info['index_rate']
                else:
                    # Use default model for main voice
                    model_path = request.default_model_path
                    index_path = request.default_index_path
                
                # Auto-reduce index_rate for emotional segments to preserve more base TTS prosody
                # This makes RVC output sound more natural/expressive
                if emotion and emotion.lower() in EMOTION_PRESETS:
                    preset = EMOTION_PRESETS[emotion.lower()]
                    intensity = preset.get('intensity', 'mild')
                    reduction = EMOTION_INDEX_RATE_REDUCTION.get(intensity, 0.05)
                    original_index_rate = index_rate
                    index_rate = max(0.3, index_rate - reduction)  # Don't go below 0.3
                    if original_index_rate != index_rate:
                        logger.info(f"Segment {i+1}: Auto-reduced index_rate from {original_index_rate:.2f} to {index_rate:.2f} for '{emotion}' emotion ({intensity})")
                
                logger.info(f"Segment {i+1}: model_path={model_path}, index_path={index_path}, f0_up_key={f0_up_key}, index_rate={index_rate}")
                
                # Apply voice conversion if model is specified
                if model_path and model_manager:
                    # Resample to 16kHz for RVC
                    audio_16k = librosa.resample(audio, orig_sr=target_sr, target_sr=16000)
                    
                    # Load model
                    success = model_manager.load_model(model_path, index_path)
                    if success:
                        from app.model_manager import RVCInferParams
                        params = RVCInferParams(
                            f0_up_key=f0_up_key,
                            index_rate=index_rate,
                            protect=0.35,
                            rms_mix_rate=0.25,
                        )
                        
                        converted = model_manager.infer(audio_16k.astype(np.float32), params=params)
                        if converted is not None and len(converted) > 0:
                            # Resample back to target_sr
                            audio = librosa.resample(converted.astype(np.float32), orig_sr=16000, target_sr=target_sr)
                            logger.info(f"Segment {i+1}: Voice conversion applied")
                
                audio_chunks.append(audio)
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        # Concatenate all audio
        if audio_chunks:
            combined = np.concatenate(audio_chunks)
            
            # Normalize
            max_val = np.max(np.abs(combined))
            if max_val > 0.95:
                combined = combined * (0.95 / max_val)
            
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, combined, target_sr, format='WAV')
            wav_buffer.seek(0)
            
            return MultiVoiceTTSResponse(
                audio=base64.b64encode(wav_buffer.read()).decode('utf-8'),
                sample_rate=target_sr,
                format="wav",
                segments_processed=len(segments),
                include_segments_used=include_count
            )
        else:
            raise HTTPException(status_code=500, detail="No audio generated")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Multi-voice TTS failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-voice TTS failed: {str(e)}")


# Quality presets for voice conversion
QUALITY_PRESETS = {
    'natural': {
        'index_rate': 0.4,      # Lower = more natural, less like target
        'filter_radius': 3,     # Medium smoothing
        'rms_mix_rate': 0.15,   # Keep more original dynamics
        'protect': 0.45,        # High protection for consonants/breathing
        'description': 'Most natural sounding, preserves speech characteristics'
    },
    'balanced': {
        'index_rate': 0.55,     # Balanced blend
        'filter_radius': 3,     # Medium smoothing
        'rms_mix_rate': 0.25,   # Moderate dynamics blend
        'protect': 0.35,        # Moderate protection
        'description': 'Good balance between natural sound and voice accuracy'
    },
    'accurate': {
        'index_rate': 0.75,     # Higher = more like target voice
        'filter_radius': 3,     # Medium smoothing
        'rms_mix_rate': 0.3,    # More converted dynamics
        'protect': 0.25,        # Less protection, more transformation
        'description': 'Sounds more like target voice but may be less natural'
    },
    'maximum': {
        'index_rate': 0.9,      # Very high match to target
        'filter_radius': 2,     # Less smoothing
        'rms_mix_rate': 0.4,    # Use converted dynamics
        'protect': 0.15,        # Minimal protection
        'description': 'Maximum voice match, may sound artificial'
    }
}


@app.get("/convert/presets")
async def get_quality_presets():
    """Get available quality presets for voice conversion"""
    return {
        'presets': QUALITY_PRESETS,
        'parameters': {
            'index_rate': {
                'description': 'How much to match target voice timbre (0.0-1.0)',
                'low': 'More natural, less like target',
                'high': 'More like target, may sound robotic'
            },
            'protect': {
                'description': 'Protect consonants and breathing (0.0-0.5)',
                'low': 'More transformation, may lose speech clarity',
                'high': 'More natural speech, consonants preserved'
            },
            'rms_mix_rate': {
                'description': 'Volume dynamics blend (0.0-1.0)',
                'low': 'Keep original volume changes',
                'high': 'Use converted volume envelope'
            },
            'filter_radius': {
                'description': 'Pitch smoothing (0-7)',
                'low': 'More pitch detail but noisier',
                'high': 'Smoother pitch but may lose nuance'
            }
        },
        'recommendations': {
            'singing': {'preset': 'accurate', 'note': 'Higher index_rate works better for singing'},
            'speech': {'preset': 'natural', 'note': 'Lower index_rate sounds more natural for speech'},
            'characters': {'preset': 'balanced', 'note': 'Good for character voices'},
        }
    }


@app.post("/convert", response_model=ConvertResponse)
async def convert_voice(request: ConvertRequest):
    """Convert voice using RVC model"""
    global model_manager, infer_params
    
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        # Apply quality preset if specified
        index_rate = request.index_rate
        filter_radius = request.filter_radius
        rms_mix_rate = request.rms_mix_rate
        protect = request.protect
        
        if request.quality_preset and request.quality_preset in QUALITY_PRESETS:
            preset = QUALITY_PRESETS[request.quality_preset]
            index_rate = preset['index_rate']
            filter_radius = preset['filter_radius']
            rms_mix_rate = preset['rms_mix_rate']
            protect = preset['protect']
            logger.info(f"Using quality preset: {request.quality_preset}")
        
        # Decode audio
        audio_bytes = base64.b64decode(request.audio)
        
        # Try to load as WAV first
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_buffer, dtype='float32')
        except Exception:
            # Try loading with librosa for other formats
            import librosa
            audio_buffer = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(audio_buffer, sr=None, mono=True)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Load model if different from current
        current_model = model_manager.model_name
        if current_model is None or not request.model_path.endswith(current_model):
            success = model_manager.load_model(request.model_path, request.index_path)
            if not success:
                raise HTTPException(status_code=400, detail=f"Failed to load model: {request.model_path}")
        
        # Create inference params with adjusted values
        from app.model_manager import RVCInferParams
        params = RVCInferParams(
            f0_up_key=request.f0_up_key,
            f0_method=request.f0_method,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )
        
        logger.info(f"Conversion params: index_rate={index_rate}, protect={protect}, rms_mix={rms_mix_rate}")
        
        # Run inference
        output_audio = model_manager.infer(audio, params=params)
        
        if output_audio is None or len(output_audio) == 0:
            raise HTTPException(status_code=500, detail="Voice conversion failed")
        
        # Apply audio effects after conversion if specified
        if request.apply_effects:
            effects = get_emotion_effects(request.apply_effects)
            if effects:
                logger.info(f"Applying post-conversion effects for '{request.apply_effects}': {list(effects.keys())}")
                output_audio = apply_audio_effects(output_audio, 16000, effects)
        
        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, output_audio, 16000, format='WAV')
        wav_buffer.seek(0)
        
        return ConvertResponse(
            audio=base64.b64encode(wav_buffer.read()).decode('utf-8'),
            sample_rate=16000,
            format="wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Voice conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice conversion failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available voice models"""
    if model_manager is None:
        return {"models": [], "current_model": None}
    
    try:
        models = model_manager.list_available_models() if hasattr(model_manager, 'list_available_models') else []
        return {
            "models": models,
            "current_model": model_manager.model_name
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"models": [], "current_model": model_manager.model_name}


# =============================================================================
# Voice Detection Endpoints
# =============================================================================

class VoiceDetectRequest(BaseModel):
    """Voice detection request"""
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=44100, description="Input sample rate")
    use_vocals_only: bool = Field(default=True, description="Separate vocals first (recommended for music)")
    max_voices: int = Field(default=5, description="Maximum number of voices to detect")


class VoiceDetectResponse(BaseModel):
    """Voice detection response"""
    voice_count: int = Field(description="Number of distinct voices detected")
    confidence: float = Field(description="Confidence score (0-1)")
    method: str = Field(description="Detection method used")
    details: dict = Field(description="Additional detection details")


@app.post("/voice-count/detect", response_model=VoiceDetectResponse)
async def detect_voices(request: VoiceDetectRequest):
    """
    Detect the number of distinct singers/speakers in audio.
    
    Useful for:
    - Determining if a song has backup vocals
    - Knowing how many voice models to assign for multi-voice swap
    - Detecting duets, harmonies, or group vocals
    
    Examples:
    - Solo artist: 1 voice
    - Duet (Simon & Garfunkel): 2 voices
    - Group harmony (Billy Joel "For the Longest Time"): 4-6 voices
    """
    try:
        # Use the new v2 detector with spectral/harmonic analysis
        from app.voice_detector_v2 import detect_voice_count
        import librosa
        from pydub import AudioSegment
        import tempfile
        import subprocess
        
        logger.info(f"Voice detection request: use_vocals_only={request.use_vocals_only}, max_voices={request.max_voices}")
        
        # Decode audio - handle data URL prefix if present
        audio_data = request.audio
        if ',' in audio_data and audio_data.startswith('data:'):
            audio_data = audio_data.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_data)
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Try multiple methods to load audio
        audio = None
        sr = None
        
        # Method 1: Try soundfile directly
        try:
            audio, sr = sf.read(audio_buffer, dtype='float32')
            logger.info(f"Loaded audio with soundfile: sr={sr}, shape={audio.shape}")
        except Exception as e:
            logger.debug(f"soundfile failed: {e}")
            audio_buffer.seek(0)
            
            # Method 2: Try librosa
            try:
                audio, sr = librosa.load(audio_buffer, sr=None, mono=False)
                logger.info(f"Loaded audio with librosa: sr={sr}, shape={audio.shape}")
            except Exception as e:
                logger.debug(f"librosa failed: {e}")
                audio_buffer.seek(0)
                
                # Method 3: Use ffmpeg as fallback
                try:
                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_in:
                        tmp_in.write(audio_buffer.read())
                        tmp_in_path = tmp_in.name
                    
                    tmp_out_path = tmp_in_path.replace('.webm', '.wav')
                    
                    result = subprocess.run(
                        ['ffmpeg', '-y', '-i', tmp_in_path, '-acodec', 'pcm_s16le', '-ar', '44100', tmp_out_path],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode == 0:
                        audio, sr = sf.read(tmp_out_path, dtype='float32')
                        logger.info(f"Loaded audio with ffmpeg conversion: sr={sr}, shape={audio.shape}")
                    else:
                        raise Exception(f"ffmpeg conversion failed: {result.stderr}")
                    
                    import os
                    os.unlink(tmp_in_path)
                    if os.path.exists(tmp_out_path):
                        os.unlink(tmp_out_path)
                        
                except Exception as e:
                    logger.error(f"All audio loading methods failed: {e}")
                    raise HTTPException(status_code=400, detail=f"Could not decode audio file: {str(e)}")
        
        # Run voice detection
        result = detect_voice_count(
            audio=audio,
            sample_rate=sr,
            use_vocals_only=request.use_vocals_only,
            max_voices=request.max_voices
        )
        
        return VoiceDetectResponse(
            voice_count=result.voice_count,
            confidence=result.confidence,
            method=result.method,
            details=result.details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Voice detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice detection failed: {str(e)}")


# =============================================================================
# Audio Processing Endpoints
# =============================================================================

class VoiceModelConfig(BaseModel):
    """Configuration for a single voice in multi-voice swap"""
    model_path: str = Field(..., description="Path to the voice model")
    index_path: Optional[str] = Field(default=None, description="Path to index file")
    f0_up_key: int = Field(default=0, description="Pitch shift for this voice")
    extraction_mode: str = Field(default="main", description="Extraction mode: 'main' (HP5 - main vocal only) or 'all' (HP3 - all vocals including harmonies)")


class AudioProcessRequest(BaseModel):
    """Audio processing request"""
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=44100, description="Input sample rate")
    mode: str = Field(default="split", description="Processing mode: split, convert, or swap")
    model_path: Optional[str] = Field(default=None, description="Path to voice model for conversion")
    index_path: Optional[str] = Field(default=None, description="Path to index file")
    f0_up_key: int = Field(default=0, description="Pitch shift for vocals")
    index_rate: float = Field(default=0.5, description="Index blend rate (lower=more natural)")
    protect: float = Field(default=0.4, description="Protect consonants (higher=more natural)")
    rms_mix_rate: float = Field(default=0.2, description="RMS mix rate")
    pitch_shift_all: int = Field(default=0, description="Pitch shift for ALL audio in semitones")
    instrumental_pitch: Optional[int] = Field(default=None, description="Separate pitch shift for instrumental in semitones (overrides pitch_shift_all for instrumental)")
    quality_preset: Optional[str] = Field(default="natural", description="Quality preset: natural, balanced, accurate, maximum")
    extract_all_vocals: bool = Field(default=False, description="Extract all vocals including backups (True) or only main vocal (False). Default is False (main vocal only).")
    
    # Multi-voice swap configuration
    voice_count: int = Field(default=1, ge=1, le=4, description="Number of voice layers to extract and convert (1-4)")
    voice_configs: Optional[List[VoiceModelConfig]] = Field(default=None, description="List of voice model configurations for multi-voice swap. If voice_count > 1, provide config for each voice.")


class AudioProcessResponse(BaseModel):
    """Audio processing response"""
    mode: str
    vocals: Optional[str] = Field(default=None, description="Base64 encoded vocals audio")
    instrumental: Optional[str] = Field(default=None, description="Base64 encoded instrumental audio")
    converted: Optional[str] = Field(default=None, description="Base64 encoded converted audio")
    sample_rate: int = Field(default=44100)
    format: str = Field(default="wav")


@app.post("/audio/process", response_model=AudioProcessResponse)
async def process_audio(request: AudioProcessRequest):
    """
    Process audio with various modes:
    - split: Separate vocals from instrumentals using UVR5
    - convert: Apply voice conversion to audio
    - swap: Separate vocals, convert them, and merge back
    """
    # Log incoming request parameters
    logger.info(f"Audio processing request: mode={request.mode}, pitch_shift_all={request.pitch_shift_all}, f0_up_key={request.f0_up_key}")
    
    try:
        import librosa
        from pydub import AudioSegment
        import tempfile
        import subprocess
        
        # Decode audio - handle data URL prefix if present
        audio_data = request.audio
        if ',' in audio_data and audio_data.startswith('data:'):
            audio_data = audio_data.split(',')[1]
        
        audio_bytes = base64.b64decode(audio_data)
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Try multiple methods to load audio
        audio = None
        sr = None
        
        # Method 1: Try soundfile directly
        try:
            audio, sr = sf.read(audio_buffer, dtype='float32')
            logger.info(f"Loaded audio with soundfile: sr={sr}, shape={audio.shape}")
        except Exception as e:
            logger.debug(f"soundfile failed: {e}")
            audio_buffer.seek(0)
            
            # Method 2: Try librosa
            try:
                audio, sr = librosa.load(audio_buffer, sr=None, mono=False)
                logger.info(f"Loaded audio with librosa: sr={sr}, shape={audio.shape}")
            except Exception as e:
                logger.debug(f"librosa failed: {e}")
                audio_buffer.seek(0)
                
                # Method 3: Use pydub/ffmpeg as fallback for webm, opus, etc.
                try:
                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_in:
                        tmp_in.write(audio_buffer.read())
                        tmp_in_path = tmp_in.name
                    
                    tmp_out_path = tmp_in_path.replace('.webm', '.wav')
                    
                    # Convert to WAV using ffmpeg
                    result = subprocess.run(
                        ['ffmpeg', '-y', '-i', tmp_in_path, '-acodec', 'pcm_s16le', '-ar', '44100', tmp_out_path],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode == 0:
                        audio, sr = sf.read(tmp_out_path, dtype='float32')
                        logger.info(f"Loaded audio with ffmpeg conversion: sr={sr}, shape={audio.shape}")
                    else:
                        raise Exception(f"ffmpeg conversion failed: {result.stderr}")
                    
                    # Cleanup temp files
                    import os
                    os.unlink(tmp_in_path)
                    if os.path.exists(tmp_out_path):
                        os.unlink(tmp_out_path)
                        
                except Exception as e:
                    logger.error(f"All audio loading methods failed: {e}")
                    raise HTTPException(status_code=400, detail=f"Could not decode audio file. Supported formats: WAV, MP3, FLAC, OGG, WEBM. Error: {str(e)}")
        
        # Keep original sample rate for output
        output_sr = sr if sr > 0 else 44100
        
        def encode_audio(audio_data: np.ndarray, sample_rate: int) -> str:
            """Encode audio to base64 WAV, ensuring valid range"""
            # Clip to valid range to prevent distortion
            audio_clipped = np.clip(audio_data, -1.0, 1.0).astype(np.float32)
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_clipped, sample_rate, format='WAV')
            wav_buffer.seek(0)
            return base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        def pitch_shift_audio(audio_data: np.ndarray, sample_rate: int, semitones: int) -> np.ndarray:
            """Pitch shift audio by given number of semitones"""
            if semitones == 0:
                return audio_data
            # Use librosa's pitch_shift for high-quality time-preserving pitch shift
            return librosa.effects.pitch_shift(
                audio_data.astype(np.float32), 
                sr=sample_rate, 
                n_steps=semitones
            )
        
        if request.mode == "split":
            # Vocal/Instrumental separation using UVR5
            try:
                from app.vocal_separator import separate_vocals, list_available_models
                
                # Check if UVR5 models are available
                available_models = list_available_models()
                if not available_models:
                    raise HTTPException(
                        status_code=500, 
                        detail="No UVR5 models available. Run: bash scripts/download_uvr5_assets.sh"
                    )
                
                # Choose model based on whether to extract all vocals or just main
                if request.extract_all_vocals:
                    # HP3_all_vocals extracts ALL vocals including backups/harmonies (better quality than HP2)
                    uvr_model = "HP3_all_vocals"
                    if uvr_model not in available_models:
                        uvr_model = "HP2_all_vocals"  # Fallback
                else:
                    # HP5_only_main_vocal extracts only the main/lead vocal
                    uvr_model = "HP5_only_main_vocal"
                
                if uvr_model not in available_models and available_models:
                    uvr_model = available_models[0]
                
                logger.info(f"Starting vocal separation with model: {uvr_model}")
                
                # Run UVR5 separation
                vocals, instrumental = separate_vocals(
                    audio=audio,
                    sample_rate=sr,
                    model_name=uvr_model,
                    agg=10
                )
                
                logger.info(f"Separation complete: vocals shape={vocals.shape}, instrumental shape={instrumental.shape}")
                
                # Apply pitch shift to vocals
                vocals_pitch = request.pitch_shift_all
                # Use separate instrumental_pitch if provided, otherwise use pitch_shift_all
                instrumental_pitch = request.instrumental_pitch if request.instrumental_pitch is not None else request.pitch_shift_all
                
                if vocals_pitch != 0:
                    logger.info(f"Applying pitch shift of {vocals_pitch} semitones to vocals")
                    vocals = pitch_shift_audio(vocals, 44100, vocals_pitch)
                    logger.info(f"Vocals pitch shift applied: shape={vocals.shape}")
                
                if instrumental_pitch != 0:
                    logger.info(f"Applying pitch shift of {instrumental_pitch} semitones to instrumental")
                    instrumental = pitch_shift_audio(instrumental, 44100, instrumental_pitch)
                    logger.info(f"Instrumental pitch shift applied: shape={instrumental.shape}")
                
                return AudioProcessResponse(
                    mode="split",
                    vocals=encode_audio(vocals, 44100),
                    instrumental=encode_audio(instrumental, 44100),
                    sample_rate=44100,
                    format="wav"
                )
                
            except ImportError as e:
                logger.error(f"UVR5 import error: {e}")
                raise HTTPException(status_code=500, detail=f"UVR5 not available: {str(e)}")
            except FileNotFoundError as e:
                logger.error(f"UVR5 model not found: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.exception(f"Vocal separation failed: {e}")
                raise HTTPException(status_code=500, detail=f"Vocal separation failed: {str(e)}")
            
        elif request.mode == "convert":
            # Voice conversion
            if not request.model_path:
                raise HTTPException(status_code=400, detail="Model path required for conversion")
            
            if model_manager is None:
                raise HTTPException(status_code=500, detail="Model manager not initialized")
            
            # Convert to mono for RVC
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0) if audio.shape[0] == 2 else np.mean(audio, axis=1)
            
            # Resample to 16kHz for RVC
            if sr != 16000:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            
            # Load model
            success = model_manager.load_model(request.model_path, request.index_path)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to load model")
            
            # Apply quality preset if specified
            index_rate = request.index_rate
            protect = request.protect
            rms_mix_rate = request.rms_mix_rate
            
            if request.quality_preset and request.quality_preset in QUALITY_PRESETS:
                preset = QUALITY_PRESETS[request.quality_preset]
                index_rate = preset['index_rate']
                protect = preset['protect']
                rms_mix_rate = preset['rms_mix_rate']
                logger.info(f"Using quality preset: {request.quality_preset}")
            
            # Run conversion
            from app.model_manager import RVCInferParams
            params = RVCInferParams(
                f0_up_key=request.f0_up_key,
                index_rate=index_rate,
                protect=protect,
                rms_mix_rate=rms_mix_rate,
            )
            logger.info(f"Conversion params: index_rate={index_rate}, protect={protect}, rms_mix={rms_mix_rate}")
            
            output_audio = model_manager.infer(audio.astype(np.float32), params=params)
            
            if output_audio is None or len(output_audio) == 0:
                raise HTTPException(status_code=500, detail="Voice conversion failed")
            
            return AudioProcessResponse(
                mode="convert",
                converted=encode_audio(output_audio.astype(np.float32), 16000),
                sample_rate=16000,
                format="wav"
            )
            
        elif request.mode == "swap":
            # Vocal swap: split -> convert vocals -> merge back with instrumental
            # Supports multi-voice cascading extraction for songs with multiple vocal layers
            
            if model_manager is None:
                raise HTTPException(status_code=500, detail="Model manager not initialized")
            
            # Validate voice configuration
            voice_count = request.voice_count or 1
            
            # Debug logging for multi-voice
            logger.info(f"Multi-voice config: voice_count={voice_count}, model_path={request.model_path}")
            logger.info(f"voice_configs from request: {request.voice_configs}")
            
            # Build voice configs list
            voice_configs = []
            if voice_count == 1:
                # Single voice - use main model_path
                if not request.model_path:
                    raise HTTPException(status_code=400, detail="Model path required for vocal swap")
                voice_configs.append({
                    "model_path": request.model_path,
                    "index_path": request.index_path,
                    "f0_up_key": request.f0_up_key,
                    "extraction_mode": "main"  # Default to HP5 for single voice
                })
            else:
                # Multi-voice mode
                # Check if voice_configs already has ALL voices (frontend sends complete list)
                if request.voice_configs and len(request.voice_configs) >= voice_count:
                    # Use voice_configs directly - it contains all voice configurations
                    for vc in request.voice_configs[:voice_count]:
                        voice_configs.append({
                            "model_path": vc.model_path,
                            "index_path": vc.index_path,
                            "f0_up_key": vc.f0_up_key,
                            "extraction_mode": getattr(vc, 'extraction_mode', 'main')  # 'main' (HP5) or 'all' (HP3)
                        })
                else:
                    # Fallback: Voice 1 from model_path, additional from voice_configs
                    if not request.model_path:
                        raise HTTPException(status_code=400, detail="Model path required for vocal swap (Voice 1)")
                    
                    # Add Voice 1 from main model_path
                    voice_configs.append({
                        "model_path": request.model_path,
                        "index_path": request.index_path,
                        "f0_up_key": request.f0_up_key,
                        "extraction_mode": "main"  # Default to HP5 for voice 1
                    })
                    
                    # Add additional voices from voice_configs (Voice 2, 3, 4...)
                    additional_needed = voice_count - 1
                    if request.voice_configs:
                        for vc in request.voice_configs[:additional_needed]:
                            voice_configs.append({
                                "model_path": vc.model_path,
                                "index_path": vc.index_path,
                                "f0_up_key": vc.f0_up_key,
                                "extraction_mode": getattr(vc, 'extraction_mode', 'main')
                            })
                
                # Check we have enough voices
                if len(voice_configs) < voice_count:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Need {voice_count} voice configurations but only got {len(voice_configs)}"
                    )
            
            # Debug: Log final voice configs
            logger.info(f"Final voice_configs for processing: {voice_configs}")
            
            try:
                from app.vocal_separator import separate_vocals, list_available_models
                
                # Check if UVR5 models are available
                available_models = list_available_models()
                if not available_models:
                    raise HTTPException(
                        status_code=500, 
                        detail="No UVR5 models available. Run: bash scripts/download_uvr5_assets.sh"
                    )
                
                # Determine which models we need
                main_vocal_model = "HP5_only_main_vocal"
                all_vocals_model = "HP3_all_vocals"  # HP3 is better quality than HP2
                if all_vocals_model not in available_models:
                    all_vocals_model = "HP2_all_vocals"
                if main_vocal_model not in available_models and available_models:
                    main_vocal_model = available_models[0]
                
                # Apply quality preset if specified
                index_rate = request.index_rate
                protect = request.protect
                rms_mix_rate = request.rms_mix_rate
                
                if request.quality_preset and request.quality_preset in QUALITY_PRESETS:
                    preset = QUALITY_PRESETS[request.quality_preset]
                    index_rate = preset['index_rate']
                    protect = preset['protect']
                    rms_mix_rate = preset['rms_mix_rate']
                    logger.info(f"Using quality preset for swap: {request.quality_preset}")
                
                logger.info(f"Multi-voice swap: voice_count={voice_count}")
                logger.info(f"Input audio: sr={sr}, length={len(audio)} samples, duration={len(audio)/sr:.2f}s")
                
                # CRITICAL: UVR5 outputs at 44100Hz regardless of input sample rate
                uvr_output_sr = 44100
                original_duration = len(audio) / sr  # Duration in seconds
                target_length_44k = int(original_duration * uvr_output_sr)
                
                converted_vocals_list = []
                
                # Helper functions for all voice modes
                from app.model_manager import RVCInferParams
                
                def ensure_length(arr, target_len):
                    """Pad or trim array to target length"""
                    if len(arr) < target_len:
                        return np.pad(arr, (0, target_len - len(arr)), mode='constant')
                    return arr[:target_len]
                
                def convert_vocal(vocal_audio, voice_config, voice_num):
                    """Convert a vocal track with RVC model, preserving original RMS level"""
                    vc = voice_config
                    logger.info(f"Converting vocals_{voice_num} with model: {vc['model_path'].split('/')[-1]}")
                    
                    # Measure original vocal RMS before conversion
                    original_rms = np.sqrt(np.mean(vocal_audio**2))
                    logger.info(f"vocals_{voice_num} original RMS before RVC: {original_rms:.4f}")
                    
                    # Resample to 16kHz for RVC
                    vocal_16k = librosa.resample(vocal_audio, orig_sr=uvr_output_sr, target_sr=16000)
                    
                    success = model_manager.load_model(vc["model_path"], vc["index_path"])
                    if not success:
                        raise HTTPException(status_code=400, detail=f"Failed to load model for voice {voice_num}")
                    
                    params = RVCInferParams(
                        f0_up_key=vc["f0_up_key"],
                        index_rate=index_rate,
                        protect=protect,
                        rms_mix_rate=rms_mix_rate,
                    )
                    
                    converted = model_manager.infer(vocal_16k, params=params)
                    if converted is None or len(converted) == 0:
                        raise HTTPException(status_code=500, detail=f"Voice {voice_num} conversion failed")
                    
                    # Resample back to 44.1kHz
                    converted_44k = librosa.resample(converted.astype(np.float32), orig_sr=16000, target_sr=uvr_output_sr)
                    converted_44k = ensure_length(converted_44k, target_length_44k)
                    
                    # Measure converted RMS
                    converted_rms = np.sqrt(np.mean(converted_44k**2))
                    logger.info(f"vocals_{voice_num} RMS after RVC: {converted_rms:.4f}")
                    
                    # Gentle RMS normalization - only boost if conversion dropped volume significantly
                    # Use conservative boost to avoid clipping
                    if converted_rms > 0.0001 and original_rms > 0.0001:
                        rms_ratio = original_rms / converted_rms
                        # Only boost, never reduce, and limit to 2x max
                        if rms_ratio > 1.2 and rms_ratio <= 2.0:
                            converted_44k = converted_44k * rms_ratio
                            final_rms = np.sqrt(np.mean(converted_44k**2))
                            logger.info(f"vocals_{voice_num} RMS boosted: {converted_rms:.4f} → {final_rms:.4f} (x{rms_ratio:.2f})")
                        elif rms_ratio > 2.0:
                            # Large drop - use conservative 2x boost
                            converted_44k = converted_44k * 2.0
                            final_rms = np.sqrt(np.mean(converted_44k**2))
                            logger.info(f"vocals_{voice_num} RMS boosted (capped): {converted_rms:.4f} → {final_rms:.4f} (x2.00)")
                    
                    # Ensure no clipping - peak normalize if needed
                    peak = np.max(np.abs(converted_44k))
                    if peak > 0.95:
                        converted_44k = converted_44k * (0.95 / peak)
                        logger.info(f"vocals_{voice_num} peak limited: {peak:.4f} → 0.95")
                    
                    final_rms = np.sqrt(np.mean(converted_44k**2))
                    logger.info(f"Voice {voice_num} converted: length={len(converted_44k)}, RMS={final_rms:.4f}")
                    return converted_44k
                
                if voice_count == 1:
                    # ============================================================
                    # SINGLE VOICE MODE: HP5 for both instrumental AND main vocal
                    # ============================================================
                    # STEP 1: HP5(original) → instrumental_clean (KEEP) + vocals_1 → convert with Model_1
                    # FINAL: Mix = instrumental_clean + converted_vocals_1
                    # ============================================================
                    
                    logger.info("Single voice extraction pipeline: 1 voice requested")
                    
                    # Helper function to clear memory
                    def clear_memory():
                        """Clear GPU and CPU memory"""
                        import gc
                        import torch as torch_mem
                        gc.collect()
                        if torch_mem.cuda.is_available():
                            torch_mem.cuda.empty_cache()
                            torch_mem.cuda.synchronize()
                    
                    # STEP 1: HP5(original) → instrumental_clean + vocals_1
                    logger.info("STEP 1: HP5(original) → instrumental_clean + vocals_1")
                    vocals_1, instrumental_clean = separate_vocals(
                        audio=audio,
                        sample_rate=sr,
                        model_name="HP5_only_main_vocal",
                        agg=10
                    )
                    del audio  # Free original
                    clear_memory()
                    
                    instrumental_clean = ensure_length(instrumental_clean, target_length_44k)
                    vocals_1 = ensure_length(vocals_1, target_length_44k)
                    
                    inst_rms = np.sqrt(np.mean(instrumental_clean**2))
                    v1_rms = np.sqrt(np.mean(vocals_1**2))
                    logger.info(f"instrumental_clean extracted: length={len(instrumental_clean)}, RMS={inst_rms:.4f} (KEEP)")
                    logger.info(f"vocals_1 extracted: RMS={v1_rms:.4f} (KEEP)")
                    
                    # Convert vocals_1 with Model_1
                    converted_vocals_1 = convert_vocal(vocals_1, voice_configs[0], 1)
                    converted_vocals_list.append(converted_vocals_1)
                    del vocals_1  # No longer needed
                    clear_memory()
                    
                else:
                    # Multi-voice cascading extraction:
                    # ============================================================
                    # MAX-4-VOICE EXTRACTION PIPELINE (HP3 + iterative HP5)
                    # ============================================================
                    # STEP 1: HP3(original) → instrumental_clean (KEEP) + all_vocals_tmp (DISCARD)
                    # STEP 2: HP5(original) → inst_minus_1 + vocals_1 → convert with Model_1
                    # STEP 3: HP5(inst_minus_1) → inst_minus_2 + vocals_2 → convert with Model_2
                    # STEP 4: HP5(inst_minus_2) → inst_minus_3 + vocals_3 → convert with Model_3 [if 3+ voices]
                    # STEP 5: HP5(inst_minus_3) → inst_minus_4 + vocals_4 → convert with Model_4 [if 4 voices]
                    # FINAL: Mix = instrumental_clean + all converted vocals
                    # ============================================================
                    
                    logger.info(f"Max-4-voice extraction pipeline: {voice_count} voices requested")
                    logger.info(f"Target output: sr={uvr_output_sr}, length={target_length_44k} samples, duration={original_duration:.2f}s")
                    
                    # Helper function to clear memory
                    def clear_memory():
                        """Clear GPU and CPU memory"""
                        import gc
                        import torch as torch_mem
                        gc.collect()
                        if torch_mem.cuda.is_available():
                            torch_mem.cuda.empty_cache()
                            torch_mem.cuda.synchronize()
                    
                    # Helper to get separator model based on extraction mode
                    def get_separator_model(voice_num, voice_config):
                        """Get the UVR model name based on extraction_mode"""
                        mode = voice_config.get("extraction_mode", "main")
                        if mode == "all":
                            model = "HP3_all_vocals"
                            logger.info(f"Voice {voice_num}: Using HP3 (all vocals including harmonies)")
                        else:
                            model = "HP5_only_main_vocal"
                            logger.info(f"Voice {voice_num}: Using HP5 (main vocal only)")
                        return model
                    
                    # ============================================================
                    # STEP 2: Extract vocals_1 from original → convert with Model_1
                    # Also capture the INSTRUMENTAL from HP5 (preserves full audio incl. intros!)
                    # ============================================================
                    sep_model_1 = get_separator_model(1, voice_configs[0])
                    logger.info(f"STEP 2: {sep_model_1}(original) → inst_minus_1 + vocals_1")
                    vocals_1, inst_minus_1 = separate_vocals(
                        audio=audio,  # IMPORTANT: Use ORIGINAL audio
                        sample_rate=sr,
                        model_name=sep_model_1,
                        agg=10
                    )
                    # Free original audio - we have what we need
                    del audio
                    clear_memory()
                    
                    vocals_1 = ensure_length(vocals_1, target_length_44k)
                    inst_minus_1 = ensure_length(inst_minus_1, target_length_44k)
                    v1_rms = np.sqrt(np.mean(vocals_1**2))
                    
                    # SAVE inst_minus_1 as our clean instrumental for final mix
                    # HP5's instrumental output preserves ALL audio including intros!
                    instrumental_clean = inst_minus_1.copy()
                    inst_rms = np.sqrt(np.mean(instrumental_clean**2))
                    logger.info(f"instrumental_clean FROM HP5: length={len(instrumental_clean)}, RMS={inst_rms:.4f} (KEEP FOR FINAL MIX)")
                    
                    # Log intro specifically to verify it's not silent
                    intro_samples = int(5 * uvr_output_sr)  # First 5 seconds
                    if len(instrumental_clean) > intro_samples:
                        intro_rms = np.sqrt(np.mean(instrumental_clean[:intro_samples]**2))
                        intro_max = np.max(np.abs(instrumental_clean[:intro_samples]))
                        logger.info(f"INSTRUMENTAL INTRO CHECK (first 5s): RMS={intro_rms:.4f}, max={intro_max:.4f}")
                    
                    logger.info(f"vocals_1 extracted: RMS={v1_rms:.4f} (KEEP)")
                    logger.info(f"inst_minus_1: length={len(inst_minus_1)} (input for next step)")
                    
                    # Convert vocals_1 with Model_1
                    converted_vocals_1 = convert_vocal(vocals_1, voice_configs[0], 1)
                    converted_vocals_list.append(converted_vocals_1)
                    del vocals_1  # No longer needed
                    clear_memory()
                    
                    # ============================================================
                    # STEP 3: Extract vocals_2 from inst_minus_1 → convert with Model_2
                    # ============================================================
                    if voice_count >= 2:
                        sep_model_2 = get_separator_model(2, voice_configs[1])
                        logger.info(f"STEP 3: {sep_model_2}(inst_minus_1) → inst_minus_2 + vocals_2")
                        vocals_2, inst_minus_2 = separate_vocals(
                            audio=inst_minus_1,  # Use inst_minus_1 as input
                            sample_rate=uvr_output_sr,  # Already at 44.1kHz from UVR
                            model_name=sep_model_2,
                            agg=10
                        )
                        del inst_minus_1  # DISCARD - no longer needed
                        clear_memory()
                        
                        vocals_2 = ensure_length(vocals_2, target_length_44k)
                        inst_minus_2 = ensure_length(inst_minus_2, target_length_44k)
                        v2_rms = np.sqrt(np.mean(vocals_2**2))
                        logger.info(f"vocals_2 extracted: RMS={v2_rms:.4f} (KEEP)")
                        
                        # Convert vocals_2 with Model_2
                        converted_vocals_2 = convert_vocal(vocals_2, voice_configs[1], 2)
                        converted_vocals_list.append(converted_vocals_2)
                        del vocals_2  # No longer needed
                        clear_memory()
                        
                        # ============================================================
                        # STEP 4: Extract vocals_3 from inst_minus_2 → convert with Model_3
                        # ============================================================
                        if voice_count >= 3:
                            sep_model_3 = get_separator_model(3, voice_configs[2])
                            logger.info(f"STEP 4: {sep_model_3}(inst_minus_2) → inst_minus_3 + vocals_3")
                            vocals_3, inst_minus_3 = separate_vocals(
                                audio=inst_minus_2,
                                sample_rate=uvr_output_sr,
                                model_name=sep_model_3,
                                agg=10
                            )
                            del inst_minus_2  # DISCARD
                            clear_memory()
                            
                            vocals_3 = ensure_length(vocals_3, target_length_44k)
                            inst_minus_3 = ensure_length(inst_minus_3, target_length_44k)
                            v3_rms = np.sqrt(np.mean(vocals_3**2))
                            logger.info(f"vocals_3 extracted: RMS={v3_rms:.4f} (KEEP)")
                            
                            # Convert vocals_3 with Model_3
                            converted_vocals_3 = convert_vocal(vocals_3, voice_configs[2], 3)
                            converted_vocals_list.append(converted_vocals_3)
                            del vocals_3
                            clear_memory()
                            
                            # ============================================================
                            # STEP 5: Extract vocals_4 from inst_minus_3 → convert with Model_4
                            # ============================================================
                            if voice_count >= 4:
                                sep_model_4 = get_separator_model(4, voice_configs[3])
                                logger.info(f"STEP 5: {sep_model_4}(inst_minus_3) → inst_minus_4 + vocals_4")
                                vocals_4, inst_minus_4 = separate_vocals(
                                    audio=inst_minus_3,
                                    sample_rate=uvr_output_sr,
                                    model_name=sep_model_4,
                                    agg=10
                                )
                                del inst_minus_3
                                del inst_minus_4  # DISCARD - end of chain
                                clear_memory()
                                
                                vocals_4 = ensure_length(vocals_4, target_length_44k)
                                v4_rms = np.sqrt(np.mean(vocals_4**2))
                                logger.info(f"vocals_4 extracted: RMS={v4_rms:.4f} (KEEP)")
                                
                                # Convert vocals_4 with Model_4
                                converted_vocals_4 = convert_vocal(vocals_4, voice_configs[3], 4)
                                converted_vocals_list.append(converted_vocals_4)
                                del vocals_4
                                clear_memory()
                            else:
                                del inst_minus_3  # DISCARD if not used for step 5
                                clear_memory()
                        else:
                            del inst_minus_2  # DISCARD - end of chain for 2 voices
                            clear_memory()
                    else:
                        del inst_minus_1  # DISCARD - only 1 voice
                        clear_memory()
                
                # ============================================================
                # FINAL MIX: instrumental_clean + all converted vocals
                # (This works for both single and multi-voice cases)
                # ============================================================
                logger.info(f"FINAL MIX: instrumental_clean + {len(converted_vocals_list)} converted vocals")
                
                # CRITICAL: Make a fresh copy of instrumental_clean for mixing
                # This ensures the original is never modified
                instrumental_for_mix = instrumental_clean.copy()
                inst_rms = np.sqrt(np.mean(instrumental_for_mix**2))
                logger.info(f"instrumental_clean RMS: {inst_rms:.4f}")
                
                # Log the first few seconds of instrumental to verify it's not silent
                intro_samples = int(5 * uvr_output_sr)  # First 5 seconds
                intro_rms = np.sqrt(np.mean(instrumental_for_mix[:intro_samples]**2))
                intro_max = np.max(np.abs(instrumental_for_mix[:intro_samples]))
                logger.info(f"Instrumental INTRO (first 5s): RMS={intro_rms:.4f}, max={intro_max:.4f}")
                
                # ============================================================
                # SIMPLE MIXING STRATEGY:
                # 1. Keep instrumental at original level - DO NOT MODIFY
                # 2. Keep main vocal (vocals_1) at its converted level
                # 3. Slightly boost secondary vocals to be audible as backing
                # 4. Simple peak normalization at the end if needed
                # ============================================================
                
                # Get RMS of main vocal for reference
                main_vocal_rms = np.sqrt(np.mean(converted_vocals_list[0]**2)) if converted_vocals_list else 0.05
                logger.info(f"Main vocal (vocals_1) RMS: {main_vocal_rms:.4f}")
                
                # Start with instrumental - use the fresh copy
                mixed = instrumental_for_mix.copy()
                logger.info(f"Starting mix with instrumental: RMS={inst_rms:.4f}")
                
                # Add each vocal layer
                for i, cv in enumerate(converted_vocals_list, start=1):
                    cv_rms = np.sqrt(np.mean(cv**2))
                    
                    if i == 1:
                        # Main vocal - add as-is
                        mixed = mixed + cv
                        logger.info(f"Added vocals_{i}: RMS={cv_rms:.4f} (main vocal, no scaling)")
                    else:
                        # Secondary vocals - boost if too quiet (target ~40% of main vocal RMS)
                        # Use very low threshold to not skip quiet backing vocals
                        if cv_rms > 0.0001:  # Very low threshold - almost never skip
                            target_rms = main_vocal_rms * 0.40  # Target 40% of main vocal
                            if cv_rms < target_rms:
                                boost = min(target_rms / cv_rms, 8.0)  # Max 8x boost for quiet backing
                                cv_boosted = cv * boost
                                new_rms = np.sqrt(np.mean(cv_boosted**2))
                                logger.info(f"Boosted vocals_{i}: {cv_rms:.4f} → {new_rms:.4f} (x{boost:.2f})")
                                mixed = mixed + cv_boosted
                            else:
                                mixed = mixed + cv
                                logger.info(f"Added vocals_{i}: RMS={cv_rms:.4f} (no boost needed)")
                        else:
                            logger.warning(f"vocals_{i} extremely quiet (RMS={cv_rms:.6f}), adding anyway with 10x boost")
                            cv_boosted = cv * 10.0
                            mixed = mixed + cv_boosted
                
                # Check peak and normalize only if clipping
                max_val = np.max(np.abs(mixed))
                mix_rms = np.sqrt(np.mean(mixed**2))
                
                # Verify instrumental is still in the mix by checking intro
                intro_samples = int(5 * uvr_output_sr)  # First 5 seconds
                mixed_intro_rms = np.sqrt(np.mean(mixed[:intro_samples]**2))
                mixed_intro_max = np.max(np.abs(mixed[:intro_samples]))
                logger.info(f"Mix INTRO (first 5s): RMS={mixed_intro_rms:.4f}, max={mixed_intro_max:.4f}")
                logger.info(f"Mix before normalization: RMS={mix_rms:.4f}, peak={max_val:.4f}")
                
                if max_val > 0.99:
                    # Simple peak normalization to prevent clipping
                    normalize_factor = 0.95 / max_val
                    mixed = mixed * normalize_factor
                    final_rms = np.sqrt(np.mean(mixed**2))
                    final_peak = np.max(np.abs(mixed))
                    logger.info(f"Normalized: factor={normalize_factor:.4f}, final RMS={final_rms:.4f}, peak={final_peak:.4f}")
                else:
                    logger.info("No normalization needed (peak < 0.99)")
                
                # Final check on intro after normalization
                final_intro_rms = np.sqrt(np.mean(mixed[:intro_samples]**2))
                logger.info(f"Final INTRO RMS: {final_intro_rms:.4f}")
                
                logger.info(f"Vocal swap complete: {len(converted_vocals_list)} voices + instrumental, output length={len(mixed)}, sr={uvr_output_sr}")
                
                return AudioProcessResponse(
                    mode="swap",
                    converted=encode_audio(mixed.astype(np.float32), uvr_output_sr),
                    sample_rate=uvr_output_sr,
                    format="wav"
                )
                
            except ImportError as e:
                logger.error(f"UVR5 import error: {e}")
                raise HTTPException(status_code=500, detail=f"UVR5 not available: {str(e)}")
            except FileNotFoundError as e:
                logger.error(f"UVR5 model not found: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.exception(f"Vocal swap failed: {e}")
                raise HTTPException(status_code=500, detail=f"Vocal swap failed: {str(e)}")
                
        else:
            raise HTTPException(status_code=400, detail=f"Unknown processing mode: {request.mode}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Audio processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")


# =============================================================================
# Model Cache Management Endpoints
# =============================================================================

class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    rvc_models_loaded: int
    uvr5_models_loaded: int
    bark_loaded: bool
    rvc_models: List[str]
    uvr5_models: List[str]
    cache_hits: int
    cache_misses: int
    rvc_loads: int
    rvc_evictions: int
    uvr5_loads: int
    uvr5_evictions: int
    estimated_memory_mb: float


class ClearCacheRequest(BaseModel):
    """Request model for clearing cache."""
    clear_rvc: bool = Field(default=True, description="Clear RVC models from cache")
    clear_uvr5: bool = Field(default=True, description="Clear UVR5 models from cache")
    clear_bark: bool = Field(default=False, description="Unload Bark TTS models")


class ClearCacheResponse(BaseModel):
    """Response model for cache clearing."""
    success: bool
    message: str
    memory_freed_estimate_mb: float


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get model cache statistics.
    
    Shows which models are currently loaded, memory estimates, and cache hit/miss rates.
    Useful for monitoring memory usage and optimizing model loading.
    """
    try:
        from app.services.model_cache import get_model_cache
        cache = get_model_cache()
        stats = cache.get_stats()
        
        return CacheStatsResponse(
            rvc_models_loaded=stats['rvc_models_loaded'],
            uvr5_models_loaded=stats['uvr5_models_loaded'],
            bark_loaded=stats['bark_loaded'],
            rvc_models=stats['rvc_models'],
            uvr5_models=stats['uvr5_models'],
            cache_hits=stats['cache_hits'],
            cache_misses=stats['cache_misses'],
            rvc_loads=stats['rvc_loads'],
            rvc_evictions=stats['rvc_evictions'],
            uvr5_loads=stats['uvr5_loads'],
            uvr5_evictions=stats['uvr5_evictions'],
            estimated_memory_mb=cache.get_memory_estimate_mb()
        )
    except ImportError:
        # ModelCache not available - return defaults
        return CacheStatsResponse(
            rvc_models_loaded=1 if model_manager and model_manager.model_name else 0,
            uvr5_models_loaded=0,
            bark_loaded=BARK_AVAILABLE,
            rvc_models=[model_manager.model_name] if model_manager and model_manager.model_name else [],
            uvr5_models=[],
            cache_hits=0,
            cache_misses=0,
            rvc_loads=0,
            rvc_evictions=0,
            uvr5_loads=0,
            uvr5_evictions=0,
            estimated_memory_mb=0.0
        )


@app.post("/cache/clear", response_model=ClearCacheResponse)
async def clear_cache(request: ClearCacheRequest):
    """
    Clear models from cache to free memory.
    
    Use this when memory is running low. Models will be reloaded on next use.
    
    Options:
    - clear_rvc: Clear all RVC voice models from cache
    - clear_uvr5: Clear UVR5 vocal separation models
    - clear_bark: Unload Bark TTS models (~1.5GB)
    """
    import gc
    
    try:
        from app.services.model_cache import get_model_cache
        cache = get_model_cache()
        
        memory_before = cache.get_memory_estimate_mb()
        
        if request.clear_rvc or request.clear_uvr5 or request.clear_bark:
            if request.clear_rvc and request.clear_uvr5 and request.clear_bark:
                cache.clear_all()
                message = "All models cleared from cache"
            else:
                messages = []
                if request.clear_rvc:
                    for name in list(cache._rvc_cache.keys()):
                        cache._evict_rvc_model(name, reason="manual clear")
                    messages.append("RVC models")
                if request.clear_uvr5:
                    for name in list(cache._uvr5_cache.keys()):
                        cache._evict_uvr5_model(name, reason="manual clear")
                    messages.append("UVR5 models")
                if request.clear_bark:
                    cache.unload_bark()
                    messages.append("Bark TTS")
                message = f"Cleared: {', '.join(messages)}"
        else:
            message = "No cache types selected for clearing"
        
        # Force garbage collection
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        memory_after = cache.get_memory_estimate_mb()
        
        return ClearCacheResponse(
            success=True,
            message=message,
            memory_freed_estimate_mb=max(0, memory_before - memory_after)
        )
        
    except ImportError:
        # ModelCache not available - try to clear Bark manually
        if request.clear_bark:
            try:
                from bark.generation import clean_models
                clean_models()
                gc.collect()
                return ClearCacheResponse(
                    success=True,
                    message="Bark TTS unloaded (ModelCache not available)",
                    memory_freed_estimate_mb=1500.0
                )
            except Exception as e:
                return ClearCacheResponse(
                    success=False,
                    message=f"Failed to unload Bark: {str(e)}",
                    memory_freed_estimate_mb=0.0
                )
        
        return ClearCacheResponse(
            success=False,
            message="ModelCache not available",
            memory_freed_estimate_mb=0.0
        )


@app.post("/cache/unload-bark")
async def unload_bark_models():
    """
    Unload Bark TTS models to free ~1.5GB of memory.
    
    Bark will be reloaded automatically on next TTS request that uses it.
    Use this when you need to free memory and don't need TTS immediately.
    """
    import gc
    
    try:
        from app.services.model_cache import get_model_cache
        cache = get_model_cache()
        cache.unload_bark()
    except ImportError:
        pass
    
    # Also try direct Bark cleanup
    try:
        from bark.generation import clean_models
        clean_models()
        logger.info("Bark models unloaded via clean_models()")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not call bark.clean_models: {e}")
    
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    return {"success": True, "message": "Bark TTS models unloaded"}


# =============================================================================
# Run Server
# =============================================================================

def run_http_server(host: str = "0.0.0.0", port: int = 8001, mm=None, params=None):
    """Run the HTTP API server"""
    import uvicorn
    
    if mm:
        set_model_manager(mm, params)
    
    # Initialize trainer API if available
    if TRAINER_AVAILABLE:
        try:
            init_trainer_api()
            logger.info("Trainer API initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize trainer API: {e}")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # For testing without full voice engine
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
