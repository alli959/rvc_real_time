"""
Text Parsing Utilities for TTS

Parses emotion tags, sound effects, and multi-voice segments.
"""

import re
from typing import Dict, List

# =============================================================================
# Emotion/Sound Mappings
# =============================================================================

SOUND_REPLACEMENTS: Dict[str, str] = {
    # Laughs
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
    
    # Thinking
    'hmm': 'hmmmmmm...',
    'thinking': 'hmmmmmmm...',
    'uhh': 'uhhhhhh...',
    'umm': 'ummmmm...',
    
    # Reactions
    'wow': 'wooooow!',
    'ooh': 'ooooooooh...',
    'ahh': 'aaaaaaah!',
    'ugh': 'uuuuugh...',
    'eww': 'eeeeeeww!',
    'yay': 'yaaaaaay!',
    'woohoo': 'woo hooooo!',
    'ow': 'ow ow ow!',
    'ouch': 'ouch!',
    'phew': 'pheeeeeew...',
    'psst': 'psssssst...',
    'ahem': 'ahem... ahem...',
    'clear_throat': 'ahem...',
}

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
    
    # Angry emotions
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
}

# Bark-specific tag mapping
BARK_TAG_MAPPING = {
    'laugh': '[laughter]',
    'laughing': '[laughter]',
    'giggle': '[laughter]',
    'chuckle': '[laughter]',
    'sigh': '[sighs]',
    'exhale': '[sighs]',
    'gasp': '[gasps]',
    'shocked': '[gasps]',
    'surprised': '[gasps]',
    'ahem': '[clears throat]',
    'cough': '[clears throat]',
    'singing': '♪',
    'hum': '♪ hmm hmm ♪',
}


def parse_text_for_tts(text: str) -> List[Dict]:
    """
    Parse text into segments for TTS processing.
    
    Handles:
    - [emotion]text[/emotion] - emotion wrapped text
    - [sound] or [sound/] - sound effects
    - *action* - asterisk actions
    - (action) - parenthetical actions
    - <speed rate="x">text</speed> - speed overrides
    - <include ...>text</include> - multi-voice segments
    
    Returns:
        List of segment dicts with keys: text, emotion, rate, include
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
        if match.group(1) and match.group(2) is not None:  # <speed>
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
                
        elif match.group(3) and match.group(4) is not None:  # <include>
            attrs_str = match.group(3)
            inner_text = match.group(4).strip()
            attrs = _parse_include_attrs(attrs_str)
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
            if sound in SOUND_REPLACEMENTS:
                segments.append({
                    'text': SOUND_REPLACEMENTS[sound],
                    'emotion': sound,
                    'rate': None,
                    'include': None
                })
                
        elif match.group(8):  # *action*
            action = match.group(8).lower()
            if action in SOUND_REPLACEMENTS:
                segments.append({
                    'text': SOUND_REPLACEMENTS[action],
                    'emotion': action,
                    'rate': None,
                    'include': None
                })
                
        elif match.group(9):  # (action)
            action = match.group(9).lower()
            if action in SOUND_REPLACEMENTS:
                segments.append({
                    'text': SOUND_REPLACEMENTS[action],
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
    
    return segments


def _parse_include_attrs(attrs_str: str) -> Dict:
    """Parse attributes from <include ...> tag."""
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
    return attrs


def convert_tags_for_bark(text: str) -> str:
    """
    Convert emotion/sound tags to Bark's native format.
    
    Bark supports:
    - [laughter], [laughs], [sighs], [gasps], [clears throat]
    - ♪ for music/singing
    - CAPS for emphasis
    - ... or — for hesitations
    """
    result = text
    
    # Replace sound effect tags with Bark equivalents
    for our_tag, bark_tag in BARK_TAG_MAPPING.items():
        result = re.sub(
            rf'\[{our_tag}\]|\[{our_tag}/\]',
            bark_tag,
            result,
            flags=re.IGNORECASE
        )
        result = re.sub(
            rf'\*{our_tag}\*',
            bark_tag,
            result,
            flags=re.IGNORECASE
        )
        result = re.sub(
            rf'\({our_tag}\)',
            bark_tag,
            result,
            flags=re.IGNORECASE
        )
    
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
    
    # Other emotions - just remove tags
    result = re.sub(
        r'\[(\w+)\](.*?)\[/\1\]',
        r'\2',
        result,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Clean up
    result = re.sub(r'\[\s*\]', '', result)
    result = re.sub(r'\.{4,}', '...', result)
    result = re.sub(r'\s{3,}', ' ', result)
    
    return result.strip()
