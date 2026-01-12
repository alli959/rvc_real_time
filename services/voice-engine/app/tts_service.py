"""
Enhanced TTS Service with Emotion and Sound Effect Support

Provides multiple TTS backends:
1. Bark (Suno) - Best for emotions and sound effects (GPU recommended)
2. Edge TTS - Fallback with enhanced audio processing for emotion simulation

Bark natively supports:
- [laughter], [laughs], [sighs], [gasps], [clears throat]
- [music], ♪ for singing
- CAPITALIZATION for emphasis
- ... or — for hesitations

For edge-tts fallback, we apply audio DSP to simulate emotions.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# =============================================================================
# Set up Bark cache directory to use local models if available
# =============================================================================

def setup_bark_cache():
    """
    Configure Bark to use local models from assets/bark if they exist.
    This avoids downloading ~13GB of models on first run.
    
    Sets up:
    - XDG_CACHE_HOME for Bark models (text, coarse, fine)
    - TORCH_HOME for encodec model
    """
    # Check for local Bark models in assets directory
    local_bark_dir = Path(__file__).parent.parent / "assets" / "bark"
    required_bark_models = ["text_2.pt", "coarse_2.pt", "fine_2.pt"]
    encodec_model = "encodec_24khz-d7cc33bc.th"
    
    if local_bark_dir.exists():
        existing_models = [m for m in required_bark_models if (local_bark_dir / m).exists()]
        has_encodec = (local_bark_dir / encodec_model).exists()
        
        if len(existing_models) == len(required_bark_models):
            # All Bark models exist locally - set up cache to use them
            # Bark uses XDG_CACHE_HOME/suno/bark_v0 for models
            cache_dir = local_bark_dir.parent.parent  # Use assets parent as cache root
            bark_cache = cache_dir / "suno" / "bark_v0"
            
            # Create symlink structure if needed
            bark_cache.parent.mkdir(parents=True, exist_ok=True)
            
            if not bark_cache.exists():
                try:
                    # Create symlink: <cache>/suno/bark_v0 -> assets/bark
                    bark_cache.symlink_to(local_bark_dir)
                    logger.info(f"Linked Bark cache to local models: {local_bark_dir}")
                except OSError as e:
                    logger.warning(f"Could not create Bark symlink: {e}")
            
            # Set environment variable for Bark
            os.environ["XDG_CACHE_HOME"] = str(cache_dir)
            
            # Also set up torch hub cache for encodec if it exists locally
            if has_encodec:
                torch_cache = cache_dir / "torch" / "hub" / "checkpoints"
                torch_cache.mkdir(parents=True, exist_ok=True)
                encodec_target = torch_cache / encodec_model
                
                if not encodec_target.exists():
                    try:
                        encodec_target.symlink_to(local_bark_dir / encodec_model)
                        logger.info(f"Linked encodec model: {encodec_target}")
                    except OSError as e:
                        logger.warning(f"Could not create encodec symlink: {e}")
                
                os.environ["TORCH_HOME"] = str(cache_dir / "torch")
            
            logger.info(f"Using local Bark models from {local_bark_dir}")
            return True
        else:
            missing = set(required_bark_models) - set(existing_models)
            logger.info(f"Local Bark models incomplete, missing: {missing}")
    
    logger.info("No local Bark models found - will download on first use (~13GB)")
    return False

# Set up Bark cache before importing
_bark_local = setup_bark_cache()

# =============================================================================
# Check for Bark availability
# =============================================================================

BARK_AVAILABLE = False
try:
    from bark import generate_audio as bark_generate, SAMPLE_RATE as BARK_SAMPLE_RATE
    from bark.generation import preload_models
    BARK_AVAILABLE = True
    logger.info("Bark TTS is available - emotions and sound effects enabled")
except ImportError:
    logger.info("Bark TTS not installed - using edge-tts with audio processing fallback")
    logger.info("To enable native emotions, install Bark: pip install git+https://github.com/suno-ai/bark.git")

# =============================================================================
# Bark Sound Effect Mappings
# =============================================================================

# Map our emotion/sound tags to Bark's native tags
BARK_TAG_MAPPING = {
    # Laughter
    'laugh': '[laughter]',
    'laughing': '[laughter]',
    'giggle': '[laughter]',
    'chuckle': '[laughter]',
    'snicker': '[laughter]',
    'cackle': '[laughter]',
    
    # Sighs/Breathing
    'sigh': '[sighs]',
    'exhale': '[sighs]',
    'breathe': '[sighs]',
    'phew': '[sighs]',
    
    # Gasps/Surprise
    'gasp': '[gasps]',
    'shocked': '[gasps]',
    'surprised': '[gasps]',
    
    # Throat clearing
    'ahem': '[clears throat]',
    'cough': '[clears throat]',
    'clear_throat': '[clears throat]',
    
    # Music/Singing
    'singing': '♪',
    'hum': '♪ hmm hmm ♪',
    'whistle': '♪',
    
    # Hesitations (Bark handles these naturally)
    'hmm': 'hmm...',
    'thinking': 'hmm...',
    'uhh': 'uhh...',
    'umm': 'umm...',
    
    # These don't have direct Bark equivalents - use text approximations
    'cry': '... huuu... huuu...',
    'crying': '... huuu huuu...',
    'sob': '... huuuh...',
    'sniff': 'sniff...',
    'scream': 'AAAAH!',
    'shriek': 'EEEEK!',
    'groan': 'uuugh...',
    'moan': 'mmmmh...',
    'yawn': 'aaaahhh...',
    'sneeze': 'ACHOO!',
    'hiccup': 'hic!',
    'growl': 'grrrrr...',
    'hiss': 'ssssss...',
    'kiss': 'mwah',
    'clap': '... clap clap clap...',
    'wow': 'WOOOOW!',
    'ooh': 'oooooh...',
    'ahh': 'aaaaaah...',
    'ugh': 'uuugh...',
    'eww': 'eeeww...',
    'yay': 'YAAAAAY!',
    'woohoo': 'WOO HOO!',
    'ow': 'OW!',
    'ouch': 'OUCH!',
    'psst': 'psssst...',
}

# Bark speaker presets - different voice styles
BARK_SPEAKERS = {
    'default': 'v2/en_speaker_6',
    'male1': 'v2/en_speaker_6',
    'male2': 'v2/en_speaker_3',
    'female1': 'v2/en_speaker_9',
    'female2': 'v2/en_speaker_0',
    'dramatic': 'v2/en_speaker_5',
    'calm': 'v2/en_speaker_1',
}

# =============================================================================
# Edge TTS Sound Replacements (Enhanced)
# =============================================================================

EDGE_SOUND_REPLACEMENTS: Dict[str, str] = {
    # Laughs - more expressive phonetic representations
    'laugh': 'hah hah hah hah',
    'laughing': 'hah hah hah hah hah',
    'giggle': 'hee hee hee hee',
    'chuckle': 'heh heh heh',
    'snicker': 'heh heh',
    'cackle': 'ah hahahaha haha',
    
    # Crying/Sadness
    'cry': 'huuu... huuu...',
    'crying': 'huuu huuu... huuu...',
    'sob': 'huuuuh... huuuuh...',
    'sniff': 'sniff... sniff...',
    
    # Surprise/Fear
    'gasp': 'aah!',
    'scream': 'aaaaaah!',
    'shriek': 'eeeeek!',
    
    # Pain/Discomfort  
    'groan': 'uuuugh...',
    'moan': 'mmmmmmh...',
    'sigh': 'haaaaaaah...',
    'yawn': 'aaaaaaaahhh...',
    
    # Body sounds
    'cough': 'ahem... ahem...',
    'sneeze': 'ah... ah... ACHOO!',
    'hiccup': 'hic!',
    'burp': 'burrrrrp',
    'gulp': 'gulp',
    'slurp': 'slurrrrrp',
    
    # Other vocalizations
    'growl': 'grrrrrrr...',
    'hiss': 'sssssssss...',
    'hum': 'hmm hmm hmm hmm...',
    'whistle': 'wheeeeeew...',
    'shush': 'shhhhhhhh...',
    'kiss': 'mwah!',
    'blow': 'fwooooo...',
    
    # Breathing
    'pant': 'hah... hah... hah...',
    'breathe': 'hhhhhhh...',
    'inhale': 'hhhhhh...',
    'exhale': 'haaaaaaah...',
    
    # Speech patterns
    'stutter': 'I... I... I...',
    'mumble': 'mmm mmm...',
    'stammer': 'uh... uh... um...',
    
    # Thinking
    'hmm': 'hmmmmmm...',
    'thinking': 'hmmmmmmm...',
    'uhh': 'uhhhhhh...',
    'umm': 'ummmmm...',
    
    # Reactions
    'clap': 'clap clap clap',
    'snap': 'snap!',
    'wow': 'wooooow!',
    'ooh': 'ooooooooh...',
    'ahh': 'aaaaaaah!',
    'ugh': 'uuuuugh...',
    'eww': 'eeeeeeww!',
    'yay': 'yaaaaaay!',
    'boo': 'booooooo!',
    'woohoo': 'woo hooooo!',
    'ow': 'ow ow ow!',
    'ouch': 'ouch!',
    'phew': 'pheeeeeew...',
    'tsk': 'tsk tsk tsk',
    'psst': 'psssssst...',
    'ahem': 'ahem... ahem...',
    'clear_throat': 'ahem...',
}

# =============================================================================
# Emotion Prosody Presets for Edge TTS
# =============================================================================

EMOTION_PROSODY: Dict[str, Dict[str, str]] = {
    # Happy emotions - faster and higher
    'happy': {'rate': '+25%', 'pitch': '+20Hz'},
    'excited': {'rate': '+40%', 'pitch': '+30Hz'},
    'cheerful': {'rate': '+30%', 'pitch': '+22Hz'},
    'joyful': {'rate': '+28%', 'pitch': '+25Hz'},
    
    # Sad emotions - slower and lower
    'sad': {'rate': '-30%', 'pitch': '-18Hz'},
    'melancholy': {'rate': '-40%', 'pitch': '-25Hz'},
    'depressed': {'rate': '-45%', 'pitch': '-30Hz'},
    'disappointed': {'rate': '-25%', 'pitch': '-12Hz'},
    
    # Angry emotions - faster with edge
    'angry': {'rate': '+20%', 'pitch': '+15Hz'},
    'furious': {'rate': '+30%', 'pitch': '+20Hz'},
    'annoyed': {'rate': '+10%', 'pitch': '+8Hz'},
    'frustrated': {'rate': '+15%', 'pitch': '+12Hz'},
    
    # Calm emotions
    'calm': {'rate': '-25%', 'pitch': '-10Hz'},
    'peaceful': {'rate': '-35%', 'pitch': '-15Hz'},
    'relaxed': {'rate': '-30%', 'pitch': '-12Hz'},
    'neutral': {'rate': '+0%', 'pitch': '+0Hz'},
    
    # Surprise
    'surprised': {'rate': '+30%', 'pitch': '+30Hz'},
    'shocked': {'rate': '+40%', 'pitch': '+40Hz'},
    'amazed': {'rate': '+25%', 'pitch': '+28Hz'},
    
    # Fear
    'scared': {'rate': '+25%', 'pitch': '+25Hz'},
    'terrified': {'rate': '+35%', 'pitch': '+35Hz'},
    'anxious': {'rate': '+18%', 'pitch': '+15Hz'},
    'nervous': {'rate': '+15%', 'pitch': '+12Hz'},
    
    # Special expressions
    'whisper': {'rate': '-35%', 'pitch': '-30Hz'},
    'shouting': {'rate': '+30%', 'pitch': '+25Hz'},
    'sarcastic': {'rate': '-12%', 'pitch': '+15Hz'},
    'romantic': {'rate': '-30%', 'pitch': '-15Hz'},
    'serious': {'rate': '-18%', 'pitch': '-12Hz'},
    'playful': {'rate': '+28%', 'pitch': '+22Hz'},
    'dramatic': {'rate': '-25%', 'pitch': '+18Hz'},
    'mysterious': {'rate': '-30%', 'pitch': '-22Hz'},
    
    # Sound effects - adjust prosody for more realistic sounds
    'laugh': {'rate': '+20%', 'pitch': '+15Hz'},
    'laughing': {'rate': '+20%', 'pitch': '+15Hz'},
    'giggle': {'rate': '+25%', 'pitch': '+20Hz'},
    'cry': {'rate': '-20%', 'pitch': '+5Hz'},
    'crying': {'rate': '-20%', 'pitch': '+5Hz'},
    'sigh': {'rate': '-30%', 'pitch': '-12Hz'},
    'gasp': {'rate': '+30%', 'pitch': '+20Hz'},
    'scream': {'rate': '+35%', 'pitch': '+30Hz'},
    'yawn': {'rate': '-35%', 'pitch': '-18Hz'},
}

# =============================================================================
# Audio Effects for Edge TTS Fallback
# =============================================================================

EMOTION_AUDIO_EFFECTS: Dict[str, Dict] = {
    # Whisper effects
    'whisper': {'volume': 0.4, 'highpass': 300, 'noise': 0.025},
    'mysterious': {'volume': 0.65, 'reverb': 0.5, 'lowpass': 4800},
    
    # Shouting/Angry - louder with saturation
    'shouting': {'volume': 1.7, 'saturation': 0.4, 'compression': True},
    'angry': {'volume': 1.5, 'saturation': 0.3, 'highpass': 180},
    'furious': {'volume': 1.6, 'saturation': 0.35, 'compression': True},
    
    # Fear - tremolo for shaking voice
    'scared': {'tremolo': {'rate': 9, 'depth': 0.3}, 'pitch_wobble': 0.12},
    'terrified': {'tremolo': {'rate': 14, 'depth': 0.45}, 'pitch_wobble': 0.18, 'volume': 1.15},
    'nervous': {'tremolo': {'rate': 6, 'depth': 0.18}},
    'anxious': {'tremolo': {'rate': 7, 'depth': 0.22}},
    
    # Sad - muffled and quieter
    'sad': {'volume': 0.7, 'lowpass': 5200, 'reverb': 0.18},
    'melancholy': {'volume': 0.62, 'lowpass': 4800, 'reverb': 0.25},
    'depressed': {'volume': 0.52, 'lowpass': 4300, 'reverb': 0.3},
    'crying': {'volume': 0.75, 'tremolo': {'rate': 4.5, 'depth': 0.25}, 'lowpass': 5800},
    'sob': {'volume': 0.7, 'tremolo': {'rate': 5.5, 'depth': 0.3}, 'lowpass': 5300},
    
    # Happy - brighter
    'happy': {'volume': 1.18, 'highpass': 120, 'brightness': 0.25},
    'excited': {'volume': 1.3, 'highpass': 140, 'brightness': 0.35},
    'cheerful': {'volume': 1.12, 'brightness': 0.2},
    'joyful': {'volume': 1.25, 'brightness': 0.3},
    
    # Dramatic effects
    'dramatic': {'reverb': 0.5, 'volume': 1.15},
    
    # Special effects
    'robot': {'bitcrush': 6, 'lowpass': 3300, 'formant_shift': 0.8},
    'spooky': {'reverb': 0.6, 'lowpass': 4200, 'volume': 0.78, 'echo': True},
    'ethereal': {'reverb': 0.7, 'highpass': 400, 'shimmer': 0.35},
    'phone': {'lowpass': 3100, 'highpass': 400, 'volume': 0.85, 'saturation': 0.12},
    'radio': {'lowpass': 3600, 'highpass': 280, 'saturation': 0.18, 'noise': 0.018},
    'megaphone': {'lowpass': 4200, 'highpass': 550, 'saturation': 0.35, 'volume': 1.45, 'compression': True},
    'echo': {'reverb': 0.55, 'echo': True},
    'underwater': {'lowpass': 1100, 'volume': 0.72, 'reverb': 0.35, 'pitch_wobble': 0.1},
    
    # Sound effect enhancements
    'laugh': {'brightness': 0.25, 'volume': 1.2},
    'laughing': {'brightness': 0.3, 'volume': 1.25},
    'giggle': {'brightness': 0.35, 'volume': 1.15, 'highpass': 180},
    'scream': {'volume': 1.6, 'saturation': 0.25, 'highpass': 220},
    'shriek': {'volume': 1.5, 'saturation': 0.18, 'highpass': 320, 'brightness': 0.35},
    'sigh': {'volume': 0.8, 'lowpass': 5500, 'reverb': 0.15},
    'gasp': {'volume': 1.2, 'brightness': 0.2},
    'yawn': {'volume': 0.85, 'lowpass': 4500},
    'groan': {'volume': 0.9, 'lowpass': 4000, 'saturation': 0.1},
}


# =============================================================================
# Text Parsing Utilities
# =============================================================================

def convert_tags_for_bark(text: str) -> str:
    """
    Convert our emotion/sound tags to Bark's native format.
    
    Input formats:
    - [laugh] or [laughing] - sound effects
    - [happy]text[/happy] - emotion wrapped text
    - *action* - asterisk actions
    - (action) - parenthetical actions
    
    Bark supports:
    - [laughter], [laughs], [sighs], [gasps], [clears throat]
    - ♪ for music/singing
    - CAPS for emphasis
    - ... or — for hesitations
    """
    result = text
    
    # Replace sound effect tags [sound] with Bark equivalents
    for our_tag, bark_tag in BARK_TAG_MAPPING.items():
        # Match [tag] or [tag/]
        result = re.sub(
            rf'\[{our_tag}\]|\[{our_tag}/\]',
            bark_tag,
            result,
            flags=re.IGNORECASE
        )
        # Match *tag*
        result = re.sub(
            rf'\*{our_tag}\*',
            bark_tag,
            result,
            flags=re.IGNORECASE
        )
        # Match (tag)
        result = re.sub(
            rf'\({our_tag}\)',
            bark_tag,
            result,
            flags=re.IGNORECASE
        )
    
    # Handle emotion-wrapped text [emotion]text[/emotion]
    # For Bark, we remove the tags as it handles prosody naturally
    # But we can add emphasis with CAPS for certain emotions
    
    # Excited/shouting - CAPITALIZE
    result = re.sub(
        r'\[(excited|shouting|angry|furious)\](.*?)\[/\1\]',
        lambda m: m.group(2).upper(),
        result,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Whisper/calm - add ellipsis for slower delivery
    result = re.sub(
        r'\[(whisper|calm|mysterious|sad)\](.*?)\[/\1\]',
        lambda m: '... ' + m.group(2) + '...',
        result,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Other emotions - just remove tags, Bark handles prosody
    result = re.sub(
        r'\[(\w+)\](.*?)\[/\1\]',
        r'\2',
        result,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Clean up any remaining empty brackets
    result = re.sub(r'\[\s*\]', '', result)
    
    # Clean up excessive whitespace/punctuation
    result = re.sub(r'\.{4,}', '...', result)
    result = re.sub(r'\s{3,}', ' ', result)
    
    return result.strip()


def parse_text_for_edge_tts(text: str) -> List[Dict]:
    """
    Parse text into segments for edge-tts processing.
    
    Returns list of segments with:
    - text: The text to speak
    - emotion: Optional emotion for prosody/effects
    - rate: Optional rate override from <speed> tags
    - include: Optional include attributes for multi-voice
    """
    segments = []
    
    # Combined pattern for all tag types
    combined_pattern = (
        r'<speed\s+(?:rate|value)=["\']?([^"\'>\s]+)["\']?\s*>(.*?)</speed>'  # <speed>
        r'|<include\s+([^>]+)>(.*?)</include>'  # <include>
        r'|\[(\w+)\](.*?)\[/\5\]'  # [emotion]text[/emotion]
        r'|\[(\w+)/?\]'  # [sound] or [sound/]
        r'|\*(\w+)\*'  # *action*
        r'|\((\w+)\)'  # (action)
    )
    
    last_end = 0
    
    for match in re.finditer(combined_pattern, text, re.IGNORECASE | re.DOTALL):
        # Add plain text before this match
        if match.start() > last_end:
            plain_text = text[last_end:match.start()].strip()
            if plain_text:
                segments.append({
                    'text': plain_text,
                    'emotion': None,
                    'rate': None,
                    'include': None
                })
        
        # Process the match
        if match.group(1) and match.group(2) is not None:  # <speed rate="x">text</speed>
            rate = match.group(1)
            inner_text = match.group(2).strip()
            if not rate.startswith(('+', '-')):
                rate = f"+{rate}"
            if not rate.endswith('%'):
                rate = f"{rate}%"
            if inner_text:
                segments.append({
                    'text': inner_text,
                    'emotion': None,
                    'rate': rate,
                    'include': None
                })
                
        elif match.group(3) and match.group(4) is not None:  # <include>text</include>
            attrs_str = match.group(3)
            inner_text = match.group(4).strip()
            attrs = {}
            for attr_match in re.finditer(r'(\w+)=["\']?([^"\'>\s]+)["\']?', attrs_str):
                key, value = attr_match.groups()
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
                attrs[key] = value
            if inner_text:
                segments.append({
                    'text': inner_text,
                    'emotion': None,
                    'rate': None,
                    'include': attrs
                })
                
        elif match.group(5) and match.group(6) is not None:  # [emotion]text[/emotion]
            emotion = match.group(5).lower()
            inner_text = match.group(6).strip()
            if inner_text:
                segments.append({
                    'text': inner_text,
                    'emotion': emotion,
                    'rate': None,
                    'include': None
                })
                
        elif match.group(7):  # [sound] or [sound/]
            sound = match.group(7).lower()
            replacement = EDGE_SOUND_REPLACEMENTS.get(sound, sound)
            segments.append({
                'text': replacement,
                'emotion': sound,
                'rate': None,
                'include': None
            })
            
        elif match.group(8):  # *action*
            action = match.group(8).lower()
            replacement = EDGE_SOUND_REPLACEMENTS.get(action, action)
            segments.append({
                'text': replacement,
                'emotion': action,
                'rate': None,
                'include': None
            })
            
        elif match.group(9):  # (action)
            action = match.group(9).lower()
            replacement = EDGE_SOUND_REPLACEMENTS.get(action, action)
            segments.append({
                'text': replacement,
                'emotion': action,
                'rate': None,
                'include': None
            })
        
        last_end = match.end()
    
    # Add remaining text
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            segments.append({
                'text': remaining,
                'emotion': None,
                'rate': None,
                'include': None
            })
    
    # If nothing parsed, return original
    if not segments:
        segments.append({
            'text': text,
            'emotion': None,
            'rate': None,
            'include': None
        })
    
    return segments


# =============================================================================
# Audio Effects Processing
# =============================================================================

def apply_audio_effects(audio: np.ndarray, sample_rate: int, effects: Dict) -> np.ndarray:
    """Apply audio effects to simulate emotions."""
    from scipy import signal as scipy_signal
    
    if not effects:
        return audio
    
    audio = audio.astype(np.float32)
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    
    # Pitch wobble for scared/underwater
    if 'pitch_wobble' in effects:
        amount = effects['pitch_wobble']
        wobble_rate = 3.5
        t = np.arange(len(audio)) / sample_rate
        wobble = amount * np.sin(2 * np.pi * wobble_rate * t + np.random.random() * 2 * np.pi)
        mod_factor = 1 + wobble * 0.025
        indices = np.clip(
            np.arange(len(audio)) * np.interp(
                np.arange(len(audio)),
                np.arange(0, len(audio), 100),
                mod_factor[::100]
            ),
            0, len(audio) - 1
        ).astype(int)
        audio = audio[indices]
    
    # High-pass filter
    if 'highpass' in effects:
        cutoff = effects['highpass']
        nyquist = sample_rate / 2
        if cutoff < nyquist:
            b, a = scipy_signal.butter(3, cutoff / nyquist, btype='high')
            audio = scipy_signal.filtfilt(b, a, audio)
    
    # Low-pass filter
    if 'lowpass' in effects:
        cutoff = effects['lowpass']
        nyquist = sample_rate / 2
        if cutoff < nyquist:
            b, a = scipy_signal.butter(3, cutoff / nyquist, btype='low')
            audio = scipy_signal.filtfilt(b, a, audio)
    
    # Brightness boost
    if 'brightness' in effects:
        amount = effects['brightness']
        nyquist = sample_rate / 2
        cutoff = 3000 / nyquist
        if cutoff < 1:
            b, a = scipy_signal.butter(2, cutoff, btype='high')
            high_freq = scipy_signal.filtfilt(b, a, audio)
            audio = audio + high_freq * amount
    
    # Saturation (soft clipping)
    if 'saturation' in effects:
        amount = effects['saturation']
        drive = 1 + amount * 6
        audio = np.tanh(audio * drive) / np.tanh(drive)
    
    # Compression
    if effects.get('compression'):
        threshold = 0.45
        ratio = 4.5
        mask = np.abs(audio) > threshold
        audio[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
    
    # Tremolo
    if 'tremolo' in effects:
        rate = effects['tremolo'].get('rate', 5)
        depth = effects['tremolo'].get('depth', 0.2)
        t = np.arange(len(audio)) / sample_rate
        modulation = 1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t) *
                                   (1 + 0.3 * np.sin(2 * np.pi * rate * 1.5 * t)))
        audio = audio * modulation
    
    # Add noise
    if 'noise' in effects:
        amount = effects['noise']
        noise = np.random.randn(len(audio)) * amount
        audio = audio + noise
    
    # Reverb
    if 'reverb' in effects:
        amount = effects['reverb']
        reverb_time = 0.55
        reverb_samples = int(reverb_time * sample_rate)
        impulse = np.exp(-np.linspace(0, 6.5, reverb_samples))
        impulse = impulse / np.sum(impulse)
        reverb_signal = np.convolve(audio, impulse, mode='full')[:len(audio)]
        audio = audio * (1 - amount) + reverb_signal * amount
    
    # Echo
    if effects.get('echo'):
        delay_time = 0.28
        decay = 0.42
        delay_samples = int(delay_time * sample_rate)
        echo_signal = np.zeros_like(audio)
        if delay_samples < len(audio):
            echo_signal[delay_samples:] = audio[:-delay_samples] * decay
            if delay_samples * 2 < len(audio):
                echo_signal[delay_samples * 2:] += audio[:-delay_samples * 2] * decay * 0.5
            audio = audio + echo_signal
    
    # Shimmer
    if 'shimmer' in effects:
        try:
            import librosa
            amount = effects['shimmer']
            shimmer = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=12)
            envelope = np.abs(audio)
            envelope = scipy_signal.filtfilt(*scipy_signal.butter(2, 10 / (sample_rate / 2)), envelope)
            shimmer = shimmer * envelope * amount
            audio = audio + shimmer
        except Exception as e:
            logger.warning(f"Shimmer effect failed: {e}")
    
    # Bitcrush
    if 'bitcrush' in effects:
        bits = effects['bitcrush']
        levels = 2 ** bits
        audio = np.round(audio * levels) / levels
    
    # Volume (apply last)
    if 'volume' in effects:
        audio = audio * effects['volume']
    
    # Final normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0.95:
        audio = audio * (0.95 / max_val)
    
    return np.clip(audio, -1.0, 1.0)


# =============================================================================
# TTS Generation Functions
# =============================================================================

async def generate_with_bark(
    text: str,
    speaker: str = 'default'
) -> Tuple[np.ndarray, int]:
    """
    Generate TTS audio using Bark with native emotion support.
    
    Bark natively handles:
    - [laughter], [laughs], [sighs], [gasps], [clears throat]
    - ♪ for music
    - CAPS for emphasis
    - ... for hesitations
    
    NOTE: Bark can hallucinate with very short texts. We use low temperature
    settings to improve accuracy, but for critical accuracy, use Edge TTS.
    """
    if not BARK_AVAILABLE:
        raise RuntimeError("Bark is not installed")
    
    # Convert our tags to Bark format
    bark_text = convert_tags_for_bark(text)
    
    # Bark works better with some minimum context - very short texts can hallucinate
    # If text is very short, add a subtle prefix that Bark handles well
    original_bark_text = bark_text
    if len(bark_text.strip()) < 20:
        # Short text - Bark tends to hallucinate. Log warning.
        logger.warning(f"Short text for Bark ({len(bark_text)} chars) - may have accuracy issues: '{bark_text}'")
    
    logger.info(f"Bark TTS input: '{bark_text}'")
    
    # Get speaker preset
    history_prompt = BARK_SPEAKERS.get(speaker, BARK_SPEAKERS['default'])
    
    # Generate audio (run in thread pool to not block)
    loop = asyncio.get_event_loop()
    
    def _generate_bark_audio():
        """Generate Bark audio with retry on failure."""
        import torch
        from bark.generation import generate_text_semantic, preload_models
        from bark.api import semantic_to_waveform
        
        # Clear CUDA cache before generation to help with stability
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use LOWER temperature for more accurate text reproduction
        # Default Bark uses text_temp=0.7, waveform_temp=0.7 which can hallucinate
        # Lower values = more deterministic but potentially less expressive
        TEXT_TEMP = 0.5  # Lower = more accurate to input text
        WAVEFORM_TEMP = 0.5  # Lower = more consistent audio
        
        try:
            # Use the semantic generation with controlled temperature
            # This gives us more control over accuracy vs expressiveness
            logger.info(f"Generating Bark audio with text_temp={TEXT_TEMP}, waveform_temp={WAVEFORM_TEMP}")
            audio = bark_generate(
                bark_text, 
                history_prompt=history_prompt,
                text_temp=TEXT_TEMP,
                waveform_temp=WAVEFORM_TEMP
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "probability tensor" in error_msg or "inf" in error_msg or "nan" in error_msg or "cuda error" in error_msg:
                # Clear cache and retry with even lower temp for stability
                logger.warning(f"Bark error: {e}, retrying with lower temperature...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                audio = bark_generate(
                    bark_text, 
                    history_prompt=history_prompt, 
                    text_temp=0.4,  # Even more conservative
                    waveform_temp=0.4
                )
            else:
                raise
        
        # Trim leading silence and filler sounds from Bark output
        try:
            audio = _trim_bark_audio(audio, BARK_SAMPLE_RATE)
        except Exception as e:
            logger.warning(f"Failed to trim Bark audio: {e}")
        
        return audio
    
    audio = await loop.run_in_executor(None, _generate_bark_audio)
    logger.info(f"Bark generation complete, audio length: {len(audio)} samples ({len(audio)/BARK_SAMPLE_RATE:.2f}s)")
    
    return audio, BARK_SAMPLE_RATE


def _trim_bark_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Trim leading silence and potential filler sounds (like 'umm') from Bark audio.
    Bark sometimes generates filler sounds at the beginning.
    """
    import librosa
    
    original_len = len(audio)
    
    # Find non-silent intervals
    # Use a threshold that catches quieter sounds
    intervals = librosa.effects.split(audio, top_db=25)
    
    if len(intervals) > 1:
        # Multiple intervals detected - check if first one is a short filler (< 0.4 seconds)
        first_interval_start = intervals[0][0]
        first_interval_end = intervals[0][1]
        first_interval_duration = (first_interval_end - first_interval_start) / sample_rate
        
        # Gap between first and second interval
        gap_to_next = (intervals[1][0] - first_interval_end) / sample_rate if len(intervals) > 1 else 0
        
        logger.info(f"Bark audio: {len(intervals)} intervals, first duration: {first_interval_duration:.3f}s, gap to next: {gap_to_next:.3f}s")
        
        # If first segment is short (<0.5s) and there's a gap to the next segment (>0.1s),
        # it's likely a filler sound like "um" or "ah"
        if first_interval_duration < 0.5 and gap_to_next > 0.08:
            # Skip the first interval entirely and start from the second
            trim_to = intervals[1][0]
            audio = audio[trim_to:]
            logger.info(f"Detected and removed filler sound: trimmed {trim_to} samples ({trim_to/sample_rate:.3f}s)")
        else:
            # Just trim leading silence
            start_sample = intervals[0][0]
            min_lead_samples = int(0.02 * sample_rate)  # 20ms buffer
            if start_sample > min_lead_samples:
                trim_to = max(0, start_sample - min_lead_samples)
                audio = audio[trim_to:]
                logger.info(f"Trimmed {trim_to} samples ({trim_to/sample_rate:.3f}s) of leading silence")
    elif len(intervals) == 1:
        # Only one interval - just trim leading silence
        start_sample = intervals[0][0]
        min_lead_samples = int(0.02 * sample_rate)  # 20ms buffer
        if start_sample > min_lead_samples:
            trim_to = max(0, start_sample - min_lead_samples)
            audio = audio[trim_to:]
            logger.info(f"Trimmed {trim_to} samples ({trim_to/sample_rate:.3f}s) of leading silence")
    else:
        logger.info("No intervals found for trimming")
    
    logger.info(f"Bark audio: {original_len} -> {len(audio)} samples")
    return audio


async def generate_with_edge_tts(
    text: str,
    voice: str = "en-US-GuyNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz"
) -> Tuple[np.ndarray, int]:
    """
    Generate TTS audio using Edge TTS with emotion simulation via audio processing.
    """
    import edge_tts
    import librosa
    
    # Fix rate/pitch format
    if rate and not rate.startswith(('+', '-')):
        rate = f"+{rate}"
    if pitch and not pitch.startswith(('+', '-')):
        pitch = f"+{pitch}"
    
    # Parse text for segments
    segments = parse_text_for_edge_tts(text)
    
    logger.info(f"Edge TTS: {len(segments)} segment(s)")
    
    audio_chunks = []
    target_sr = None
    
    for i, segment in enumerate(segments):
        seg_text = segment['text']
        emotion = segment.get('emotion')
        seg_rate_override = segment.get('rate')
        
        # Determine prosody
        if seg_rate_override:
            seg_rate = seg_rate_override
            seg_pitch = pitch
        elif emotion and emotion in EMOTION_PROSODY:
            prosody = EMOTION_PROSODY[emotion]
            # Combine with base rate/pitch
            base_rate_val = int(rate.replace('%', '').replace('+', ''))
            base_pitch_val = int(pitch.replace('Hz', '').replace('+', ''))
            emotion_rate_val = int(prosody['rate'].replace('%', '').replace('+', ''))
            emotion_pitch_val = int(prosody['pitch'].replace('Hz', '').replace('+', ''))
            
            final_rate = base_rate_val + emotion_rate_val
            final_pitch = base_pitch_val + emotion_pitch_val
            
            seg_rate = f"+{final_rate}%" if final_rate >= 0 else f"{final_rate}%"
            seg_pitch = f"+{final_pitch}Hz" if final_pitch >= 0 else f"{final_pitch}Hz"
        else:
            seg_rate = rate
            seg_pitch = pitch
        
        # Get audio effects
        effects = EMOTION_AUDIO_EFFECTS.get(emotion, {}) if emotion else {}
        
        logger.info(f"  Segment {i + 1}: emotion={emotion}, rate={seg_rate}, text='{seg_text[:40]}...'")
        
        # Generate with edge-tts
        communicate = edge_tts.Communicate(seg_text, voice, rate=seg_rate, pitch=seg_pitch)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            await communicate.save(tmp_path)
            audio, sr = librosa.load(tmp_path, sr=None, mono=True)
            
            if target_sr is None:
                target_sr = sr
            elif sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            # Apply audio effects
            if effects:
                audio = apply_audio_effects(audio, target_sr, effects)
            
            audio_chunks.append(audio)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    if not audio_chunks:
        raise RuntimeError("No audio generated")
    
    combined = np.concatenate(audio_chunks)
    return combined, target_sr


async def generate_tts(
    text: str,
    voice: str = "en-US-GuyNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    use_bark: bool = True,
    bark_speaker: str = 'default'
) -> Tuple[bytes, int]:
    """
    Generate TTS audio with emotion and sound effect support.
    
    Args:
        text: Text with optional emotion tags
        voice: Edge TTS voice (used if Bark unavailable)
        rate: Speech rate
        pitch: Pitch adjustment
        use_bark: Whether to use Bark if available
        bark_speaker: Bark speaker preset
    
    Returns:
        Tuple of (wav_bytes, sample_rate)
    """
    # Determine if we should use Bark
    # Bark only works well with English - detect non-English voices
    is_english_voice = voice.startswith('en-') or voice.startswith('en_')
    
    # Strip tags to get actual text length for Bark decision
    plain_text = re.sub(r'\[/?[^\]]+\]', '', text)  # Remove [tags]
    plain_text = re.sub(r'<[^>]+>', '', plain_text)  # Remove <tags>
    plain_text_length = len(plain_text.strip())
    
    # Bark hallucinates on very short texts (< 15 chars) - use Edge TTS instead
    MIN_BARK_TEXT_LENGTH = 15
    text_too_short_for_bark = plain_text_length < MIN_BARK_TEXT_LENGTH
    
    if text_too_short_for_bark and use_bark:
        logger.info(f"Text too short for Bark ({plain_text_length} chars < {MIN_BARK_TEXT_LENGTH}), using Edge TTS to avoid hallucination")
    
    # Try Bark first if available, requested, English, AND long enough
    if use_bark and BARK_AVAILABLE and is_english_voice and not text_too_short_for_bark:
        try:
            logger.info("Using Bark TTS (native emotion support)")
            audio, sr = await generate_with_bark(text, bark_speaker)
        except Exception as e:
            logger.warning(f"Bark failed, falling back to edge-tts: {e}")
            audio, sr = await generate_with_edge_tts(text, voice, rate, pitch)
    else:
        if use_bark and not is_english_voice:
            logger.info(f"Voice '{voice}' is non-English, using Edge TTS (Bark only supports English well)")
        audio, sr = await generate_with_edge_tts(text, voice, rate, pitch)
    
    # Convert to WAV bytes
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, sr, format='WAV')
    wav_buffer.seek(0)
    
    return wav_buffer.read(), sr


def is_bark_available() -> bool:
    """Check if Bark TTS is available."""
    return BARK_AVAILABLE


def preload_bark_models():
    """Preload Bark models for faster first inference."""
    if BARK_AVAILABLE:
        try:
            logger.info("Preloading Bark models...")
            preload_models()
            logger.info("Bark models preloaded")
        except Exception as e:
            logger.warning(f"Failed to preload Bark models: {e}")
