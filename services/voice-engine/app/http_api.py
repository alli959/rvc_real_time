"""
HTTP API Server for Voice Engine

Provides REST API endpoints for:
- Text-to-Speech (TTS) generation using Edge TTS with emotion support
- Voice conversion using RVC models
- Health checks and model listing
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

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MorphVox Voice Engine API",
    description="Voice conversion and TTS API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    text: str = Field(..., description="Text to convert to speech", max_length=5000)
    voice: str = Field(default="en-US-GuyNeural", description="Edge TTS voice ID")
    style: str = Field(default="default", description="Speaking style/emotion")
    rate: str = Field(default="+0%", description="Speech rate adjustment")
    pitch: str = Field(default="+0Hz", description="Pitch adjustment")


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
EMOTION_PRESETS: Dict[str, Dict[str, str]] = {
    # Happy / Positive emotions
    'happy': {'rate': '+10%', 'pitch': '+5Hz', 'desc': 'Cheerful, upbeat tone'},
    'excited': {'rate': '+20%', 'pitch': '+10Hz', 'desc': 'Very enthusiastic'},
    'cheerful': {'rate': '+15%', 'pitch': '+8Hz', 'desc': 'Light and positive'},
    'joyful': {'rate': '+10%', 'pitch': '+7Hz', 'desc': 'Full of joy'},
    
    # Sad / Negative emotions
    'sad': {'rate': '-15%', 'pitch': '-5Hz', 'desc': 'Melancholic, slow'},
    'melancholy': {'rate': '-20%', 'pitch': '-8Hz', 'desc': 'Deep sadness'},
    'depressed': {'rate': '-25%', 'pitch': '-10Hz', 'desc': 'Very low energy'},
    'disappointed': {'rate': '-10%', 'pitch': '-3Hz', 'desc': 'Let down feeling'},
    
    # Angry / Intense emotions
    'angry': {'rate': '+5%', 'pitch': '+3Hz', 'desc': 'Frustrated, intense'},
    'furious': {'rate': '+10%', 'pitch': '+5Hz', 'desc': 'Very angry'},
    'annoyed': {'rate': '+3%', 'pitch': '+2Hz', 'desc': 'Mildly irritated'},
    'frustrated': {'rate': '+5%', 'pitch': '+2Hz', 'desc': 'Exasperated'},
    
    # Calm / Neutral emotions
    'calm': {'rate': '-10%', 'pitch': '-2Hz', 'desc': 'Relaxed, peaceful'},
    'peaceful': {'rate': '-15%', 'pitch': '-3Hz', 'desc': 'Very serene'},
    'relaxed': {'rate': '-12%', 'pitch': '-2Hz', 'desc': 'At ease'},
    'neutral': {'rate': '+0%', 'pitch': '+0Hz', 'desc': 'Standard tone'},
    
    # Surprised / Shocked emotions
    'surprised': {'rate': '+15%', 'pitch': '+12Hz', 'desc': 'Caught off guard'},
    'shocked': {'rate': '+20%', 'pitch': '+15Hz', 'desc': 'Very surprised'},
    'amazed': {'rate': '+10%', 'pitch': '+10Hz', 'desc': 'In awe'},
    
    # Fear / Anxiety emotions
    'scared': {'rate': '+10%', 'pitch': '+8Hz', 'desc': 'Frightened'},
    'terrified': {'rate': '+15%', 'pitch': '+12Hz', 'desc': 'Extremely scared'},
    'anxious': {'rate': '+8%', 'pitch': '+5Hz', 'desc': 'Nervous, worried'},
    'nervous': {'rate': '+5%', 'pitch': '+3Hz', 'desc': 'Slightly on edge'},
    
    # Special expressions
    'whisper': {'rate': '-20%', 'pitch': '-15Hz', 'desc': 'Quiet, secretive'},
    'shouting': {'rate': '+15%', 'pitch': '+10Hz', 'desc': 'Loud, emphatic'},
    'sarcastic': {'rate': '-5%', 'pitch': '+5Hz', 'desc': 'Ironic tone'},
    'romantic': {'rate': '-15%', 'pitch': '-5Hz', 'desc': 'Soft, loving'},
    'serious': {'rate': '-8%', 'pitch': '-3Hz', 'desc': 'Grave, important'},
    'playful': {'rate': '+12%', 'pitch': '+8Hz', 'desc': 'Fun, teasing'},
    'dramatic': {'rate': '-10%', 'pitch': '+5Hz', 'desc': 'Theatrical'},
    'mysterious': {'rate': '-15%', 'pitch': '-8Hz', 'desc': 'Enigmatic'},
    
    # Actions / Sounds (simulated via text and prosody)
    'laugh': {'rate': '+15%', 'pitch': '+10Hz', 'desc': 'Laughing sound'},
    'laughing': {'rate': '+15%', 'pitch': '+10Hz', 'desc': 'Laughing sound'},
    'giggle': {'rate': '+20%', 'pitch': '+15Hz', 'desc': 'Light giggling'},
    'chuckle': {'rate': '+10%', 'pitch': '+5Hz', 'desc': 'Soft laugh'},
    'snicker': {'rate': '+12%', 'pitch': '+8Hz', 'desc': 'Suppressed laugh'},
    'cackle': {'rate': '+18%', 'pitch': '+12Hz', 'desc': 'Witch-like laugh'},
    'sigh': {'rate': '-25%', 'pitch': '-10Hz', 'desc': 'Exhale sound'},
    'gasp': {'rate': '+25%', 'pitch': '+15Hz', 'desc': 'Sharp intake'},
    'yawn': {'rate': '-30%', 'pitch': '-15Hz', 'desc': 'Tired sound'},
    'cry': {'rate': '-15%', 'pitch': '+3Hz', 'desc': 'Sobbing'},
    'crying': {'rate': '-15%', 'pitch': '+3Hz', 'desc': 'Sobbing'},
    'sob': {'rate': '-20%', 'pitch': '+5Hz', 'desc': 'Deep crying'},
    'sniff': {'rate': '-10%', 'pitch': '+2Hz', 'desc': 'Sniffling'},
    'groan': {'rate': '-20%', 'pitch': '-12Hz', 'desc': 'Pain/frustration'},
    'moan': {'rate': '-25%', 'pitch': '-8Hz', 'desc': 'Discomfort sound'},
    'scream': {'rate': '+30%', 'pitch': '+25Hz', 'desc': 'Screaming'},
    'shriek': {'rate': '+35%', 'pitch': '+30Hz', 'desc': 'High pitched scream'},
    'growl': {'rate': '-15%', 'pitch': '-20Hz', 'desc': 'Angry growl'},
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
EMOTION_EFFECTS: Dict[str, Dict] = {
    # Whisper: quieter, slight high-pass to remove rumble
    'whisper': {'volume': 0.6, 'highpass': 200},
    'mysterious': {'volume': 0.7, 'reverb': 0.3, 'lowpass': 6000},
    
    # Shouting/Angry: louder, slight distortion/saturation
    'shouting': {'volume': 1.4, 'saturation': 0.2},
    'angry': {'volume': 1.2, 'saturation': 0.1},
    'furious': {'volume': 1.3, 'saturation': 0.15},
    
    # Scared: tremolo effect (volume wobble)
    'scared': {'tremolo': {'rate': 6, 'depth': 0.15}},
    'terrified': {'tremolo': {'rate': 8, 'depth': 0.25}},
    'nervous': {'tremolo': {'rate': 4, 'depth': 0.1}},
    'anxious': {'tremolo': {'rate': 5, 'depth': 0.12}},
    
    # Sad: slightly quieter, subtle lowpass
    'sad': {'volume': 0.85, 'lowpass': 7000},
    'melancholy': {'volume': 0.8, 'lowpass': 6500},
    'depressed': {'volume': 0.75, 'lowpass': 6000},
    
    # Dramatic: reverb for theater effect
    'dramatic': {'reverb': 0.25},
    
    # Robot/electronic effect (for fun)
    'robot': {'bitcrush': 8, 'lowpass': 4000},
    
    # Echo for spooky
    'spooky': {'reverb': 0.4, 'lowpass': 5000, 'volume': 0.9},
    'ethereal': {'reverb': 0.5, 'highpass': 300},
    
    # Phone/radio effect
    'phone': {'lowpass': 3400, 'highpass': 300, 'volume': 0.9},
    'radio': {'lowpass': 4000, 'highpass': 200, 'saturation': 0.1},
    
    # Megaphone
    'megaphone': {'lowpass': 5000, 'highpass': 400, 'saturation': 0.2, 'volume': 1.2},
    
    # Echo/reverb
    'echo': {'reverb': 0.35},
    
    # Underwater (heavy lowpass, slight pitch warble)
    'underwater': {'lowpass': 1500, 'volume': 0.85, 'reverb': 0.2},
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
    """
    from scipy import signal as scipy_signal
    
    if not effects:
        return audio_data
    
    audio = audio_data.astype(np.float32)
    
    # Normalize to -1 to 1 range if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    
    # High-pass filter (remove low rumble)
    if 'highpass' in effects:
        cutoff = effects['highpass']
        nyquist = sample_rate / 2
        if cutoff < nyquist:
            b, a = scipy_signal.butter(2, cutoff / nyquist, btype='high')
            audio = scipy_signal.filtfilt(b, a, audio)
    
    # Low-pass filter (muffle sound)
    if 'lowpass' in effects:
        cutoff = effects['lowpass']
        nyquist = sample_rate / 2
        if cutoff < nyquist:
            b, a = scipy_signal.butter(2, cutoff / nyquist, btype='low')
            audio = scipy_signal.filtfilt(b, a, audio)
    
    # Saturation (soft clipping for angry/distorted sound)
    if 'saturation' in effects:
        amount = effects['saturation']
        # Soft clipping using tanh
        audio = np.tanh(audio * (1 + amount * 3)) / np.tanh(1 + amount * 3)
    
    # Tremolo (volume wobble for scared/nervous)
    if 'tremolo' in effects:
        rate = effects['tremolo'].get('rate', 5)  # Hz
        depth = effects['tremolo'].get('depth', 0.2)  # 0-1
        t = np.arange(len(audio)) / sample_rate
        modulation = 1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
        audio = audio * modulation
    
    # Simple reverb (convolution with exponential decay)
    if 'reverb' in effects:
        amount = effects['reverb']
        reverb_time = 0.3  # seconds
        reverb_samples = int(reverb_time * sample_rate)
        impulse = np.exp(-np.linspace(0, 5, reverb_samples))
        impulse = impulse / np.sum(impulse)  # Normalize
        reverb_signal = np.convolve(audio, impulse, mode='full')[:len(audio)]
        audio = audio * (1 - amount) + reverb_signal * amount
    
    # Bitcrush (lo-fi effect)
    if 'bitcrush' in effects:
        bits = effects['bitcrush']
        levels = 2 ** bits
        audio = np.round(audio * levels) / levels
    
    # Volume adjustment (apply last)
    if 'volume' in effects:
        audio = audio * effects['volume']
    
    # Clip to prevent distortion
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def get_emotion_effects(emotion: Optional[str]) -> Dict:
    """Get audio effects for an emotion."""
    if not emotion:
        return {}
    return EMOTION_EFFECTS.get(emotion.lower(), {})


def parse_emotion_tags(text: str) -> List[Dict]:
    """
    Parse text with emotion tags and return segments with their emotions.
    
    Supported formats:
    - [happy] text [/happy] - Tagged sections
    - [laugh] - Sound effects (self-closing)
    - *laughing* - Action asterisks
    - (sigh) - Parenthetical actions
    
    Returns list of segments: [{'text': str, 'emotion': str or None}, ...]
    """
    segments = []
    
    # Pattern for [emotion] text [/emotion] or [emotion/]
    tag_pattern = r'\[(\w+)\](.*?)\[/\1\]|\[(\w+)/?\]'
    # Pattern for *action* 
    action_pattern = r'\*(\w+)\*'
    # Pattern for (action)
    paren_pattern = r'\((\w+)\)'
    
    # Combined pattern to find all emotion markers
    combined_pattern = r'\[(\w+)\](.*?)\[/\1\]|\[(\w+)/?\]|\*(\w+)\*|\((\w+)\)'
    
    last_end = 0
    
    for match in re.finditer(combined_pattern, text, re.IGNORECASE | re.DOTALL):
        # Add any text before this match as neutral
        if match.start() > last_end:
            plain_text = text[last_end:match.start()].strip()
            if plain_text:
                segments.append({'text': plain_text, 'emotion': None})
        
        # Determine which pattern matched
        if match.group(1) and match.group(2):  # [emotion]text[/emotion]
            emotion = match.group(1).lower()
            inner_text = match.group(2).strip()
            if inner_text:
                segments.append({'text': inner_text, 'emotion': emotion})
        elif match.group(3):  # [emotion] or [emotion/] - self-closing/sound effect
            emotion = match.group(3).lower()
            if emotion in SOUND_REPLACEMENTS:
                segments.append({'text': SOUND_REPLACEMENTS[emotion], 'emotion': emotion})
            else:
                # Just an emotion marker, apply to next segment
                pass
        elif match.group(4):  # *action*
            action = match.group(4).lower()
            if action in SOUND_REPLACEMENTS:
                segments.append({'text': SOUND_REPLACEMENTS[action], 'emotion': action})
            elif action in EMOTION_PRESETS:
                # It's an emotion, mark but no text
                pass
        elif match.group(5):  # (action)
            action = match.group(5).lower()
            if action in SOUND_REPLACEMENTS:
                segments.append({'text': SOUND_REPLACEMENTS[action], 'emotion': action})
        
        last_end = match.end()
    
    # Add remaining text
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            segments.append({'text': remaining, 'emotion': None})
    
    # If no segments were found, return the original text
    if not segments:
        segments.append({'text': text, 'emotion': None})
    
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
        
        # If we have multiple segments or emotions, process each separately
        if len(segments) > 1 or (segments and segments[0].get('emotion')):
            audio_chunks = []
            target_sr = None
            
            for i, segment in enumerate(segments):
                segment_text = segment['text']
                emotion = segment.get('emotion')
                
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
                    
                    audio_chunks.append(audio)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            # Concatenate all audio chunks
            if audio_chunks:
                combined_audio = np.concatenate(audio_chunks)
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
    """Generate speech from text using Edge TTS"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
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
# Audio Processing Endpoints
# =============================================================================

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
    quality_preset: Optional[str] = Field(default="natural", description="Quality preset: natural, balanced, accurate, maximum")


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
        
        # Decode audio
        audio_bytes = base64.b64decode(request.audio)
        audio_buffer = io.BytesIO(audio_bytes)
        
        try:
            audio, sr = sf.read(audio_buffer, dtype='float32')
        except Exception:
            audio_buffer.seek(0)
            audio, sr = librosa.load(audio_buffer, sr=None, mono=False)
        
        # Keep original sample rate for output
        output_sr = sr if sr > 0 else 44100
        
        def encode_audio(audio_data: np.ndarray, sample_rate: int) -> str:
            """Encode audio to base64 WAV"""
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
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
                
                # Use HP5_only_main_vocal by default (best for general use)
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
                
                # Apply pitch shift to both vocals and instrumental if requested
                pitch_shift = request.pitch_shift_all
                if pitch_shift != 0:
                    logger.info(f"Applying pitch shift of {pitch_shift} semitones to both tracks")
                    vocals = pitch_shift_audio(vocals, 44100, pitch_shift)
                    instrumental = pitch_shift_audio(instrumental, 44100, pitch_shift)
                    logger.info(f"Pitch shift applied: vocals shape={vocals.shape}, instrumental shape={instrumental.shape}")
                
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
            if not request.model_path:
                raise HTTPException(status_code=400, detail="Model path required for vocal swap")
            
            if model_manager is None:
                raise HTTPException(status_code=500, detail="Model manager not initialized")
            
            try:
                from app.vocal_separator import separate_vocals, list_available_models
                
                # Check if UVR5 models are available
                available_models = list_available_models()
                if not available_models:
                    raise HTTPException(
                        status_code=500, 
                        detail="No UVR5 models available. Run: bash scripts/download_uvr5_assets.sh"
                    )
                
                uvr_model = "HP5_only_main_vocal"
                if uvr_model not in available_models and available_models:
                    uvr_model = available_models[0]
                
                logger.info(f"Starting vocal swap with model: {uvr_model}")
                
                # Step 1: Separate vocals and instrumental
                vocals, instrumental = separate_vocals(
                    audio=audio,
                    sample_rate=sr,
                    model_name=uvr_model,
                    agg=10
                )
                
                # Step 2: Convert vocals using RVC
                # Resample vocals to 16kHz for RVC
                vocals_16k = librosa.resample(vocals, orig_sr=44100, target_sr=16000)
                
                # Load model and convert
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
                    logger.info(f"Using quality preset for swap: {request.quality_preset}")
                
                from app.model_manager import RVCInferParams
                params = RVCInferParams(
                    f0_up_key=request.f0_up_key,
                    index_rate=index_rate,
                    protect=protect,
                    rms_mix_rate=rms_mix_rate,
                )
                logger.info(f"Swap conversion params: index_rate={index_rate}, protect={protect}, rms_mix={rms_mix_rate}")
                
                converted_vocals = model_manager.infer(vocals_16k, params=params)
                
                if converted_vocals is None or len(converted_vocals) == 0:
                    raise HTTPException(status_code=500, detail="Voice conversion failed")
                
                # Resample converted vocals back to 44100Hz
                converted_vocals_44k = librosa.resample(
                    converted_vocals.astype(np.float32), 
                    orig_sr=16000, 
                    target_sr=44100
                )
                
                # Apply pitch shift to instrumental if requested (vocals already shifted via f0_up_key)
                instrumental_final = instrumental
                instrumental_pitch = request.pitch_shift_all
                if instrumental_pitch != 0:
                    logger.info(f"Applying pitch shift of {instrumental_pitch} semitones to instrumental")
                    instrumental_final = pitch_shift_audio(instrumental, 44100, instrumental_pitch)
                    logger.info(f"Instrumental pitch shifted: shape={instrumental_final.shape}")
                
                # Step 3: Mix converted vocals with instrumental
                # Ensure same length
                min_len = min(len(converted_vocals_44k), len(instrumental_final))
                mixed = converted_vocals_44k[:min_len] + instrumental_final[:min_len]
                
                # Normalize to prevent clipping
                max_val = np.max(np.abs(mixed))
                if max_val > 1.0:
                    mixed = mixed / max_val * 0.95
                
                logger.info(f"Vocal swap complete: output length={len(mixed)}")
                
                return AudioProcessResponse(
                    mode="swap",
                    converted=encode_audio(mixed.astype(np.float32), 44100),
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
# Run Server
# =============================================================================

def run_http_server(host: str = "0.0.0.0", port: int = 8001, mm=None, params=None):
    """Run the HTTP API server"""
    import uvicorn
    
    if mm:
        set_model_manager(mm, params)
    
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # For testing without full voice engine
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
