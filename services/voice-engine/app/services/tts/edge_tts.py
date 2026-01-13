"""
Edge TTS Backend

Microsoft Edge text-to-speech using the free API.
Fast and reliable fallback when Bark is not available.
"""

import asyncio
import io
import logging
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# =============================================================================
# Voice Presets
# =============================================================================

EDGE_VOICES = {
    # US English
    'en-US-JennyNeural': {'gender': 'female', 'style': 'friendly'},
    'en-US-GuyNeural': {'gender': 'male', 'style': 'friendly'},
    'en-US-AriaNeural': {'gender': 'female', 'style': 'news'},
    'en-US-DavisNeural': {'gender': 'male', 'style': 'calm'},
    'en-US-TonyNeural': {'gender': 'male', 'style': 'casual'},
    'en-US-SaraNeural': {'gender': 'female', 'style': 'cheerful'},
    
    # UK English
    'en-GB-SoniaNeural': {'gender': 'female', 'style': 'professional'},
    'en-GB-RyanNeural': {'gender': 'male', 'style': 'professional'},
    
    # Australian English
    'en-AU-NatashaNeural': {'gender': 'female', 'style': 'friendly'},
    'en-AU-WilliamNeural': {'gender': 'male', 'style': 'friendly'},
}

DEFAULT_VOICE = 'en-US-JennyNeural'


async def generate_with_edge_tts_async(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
    volume: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Generate speech using Edge TTS (async).
    
    Args:
        text: Text to speak
        voice: Voice name from EDGE_VOICES
        rate: Speaking rate (e.g., '+20%', '-10%')
        pitch: Pitch adjustment (e.g., '+10Hz', '-20Hz')
        volume: Volume adjustment (e.g., '+20%', '-10%')
    
    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts not installed. Install with: pip install edge-tts")
    
    # Build SSML if we have prosody modifications
    if rate or pitch or volume:
        ssml = _build_ssml(text, voice, rate, pitch, volume)
        communicate = edge_tts.Communicate(ssml, voice)
    else:
        communicate = edge_tts.Communicate(text, voice)
    
    # Generate to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        await communicate.save(tmp_path)
        
        # Load and convert to numpy
        audio, sr = sf.read(tmp_path, dtype='float32')
        
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        return audio, sr
        
    finally:
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def generate_with_edge_tts(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
    volume: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Generate speech using Edge TTS (sync wrapper).
    
    Args:
        text: Text to speak
        voice: Voice name from EDGE_VOICES
        rate: Speaking rate (e.g., '+20%', '-10%')
        pitch: Pitch adjustment (e.g., '+10Hz', '-20Hz')
        volume: Volume adjustment (e.g., '+20%', '-10%')
    
    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        generate_with_edge_tts_async(text, voice, rate, pitch, volume)
    )


def _build_ssml(
    text: str,
    voice: str,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
    volume: Optional[str] = None,
) -> str:
    """Build SSML with prosody tags."""
    prosody_attrs = []
    if rate:
        prosody_attrs.append(f'rate="{rate}"')
    if pitch:
        prosody_attrs.append(f'pitch="{pitch}"')
    if volume:
        prosody_attrs.append(f'volume="{volume}"')
    
    prosody_str = ' '.join(prosody_attrs)
    
    ssml = f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
    <voice name="{voice}">
        <prosody {prosody_str}>
            {text}
        </prosody>
    </voice>
</speak>'''
    
    return ssml


async def get_available_voices_async(language: str = "en") -> List[Dict]:
    """
    Get list of available Edge TTS voices for a language.
    
    Args:
        language: Language code prefix (e.g., 'en', 'es', 'fr')
    
    Returns:
        List of voice info dicts
    """
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts not installed")
    
    voices = await edge_tts.list_voices()
    
    filtered = [
        {
            'name': v['ShortName'],
            'gender': v['Gender'],
            'locale': v['Locale'],
        }
        for v in voices
        if v['Locale'].startswith(language)
    ]
    
    return filtered


def get_available_voices(language: str = "en") -> List[Dict]:
    """Sync wrapper for get_available_voices_async."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(get_available_voices_async(language))
