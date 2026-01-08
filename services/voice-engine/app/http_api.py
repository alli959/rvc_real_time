"""
HTTP API Server for Voice Engine

Provides REST API endpoints for:
- Text-to-Speech (TTS) generation using Edge TTS
- Voice conversion using RVC models
- Health checks and model listing
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

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
    index_rate: float = Field(default=0.75, description="Index blend rate")
    filter_radius: int = Field(default=3, description="Filter radius")
    rms_mix_rate: float = Field(default=0.25, description="RMS mix rate")
    protect: float = Field(default=0.33, description="Protect rate")


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
# Edge TTS Integration
# =============================================================================

async def generate_tts_audio(
    text: str,
    voice: str = "en-US-GuyNeural",
    style: str = "default",
    rate: str = "+0%",
    pitch: str = "+0Hz"
) -> Tuple[bytes, int]:
    """
    Generate TTS audio using Edge TTS
    
    Note: edge_tts library doesn't support SSML or emotion/style parameters.
    Style parameter is ignored - for style support, use Azure Speech SDK directly.
    
    Returns:
        Tuple of (audio_bytes, sample_rate)
    """
    try:
        import edge_tts
        
        # Fix rate format - Edge TTS requires +/- prefix
        if rate and not rate.startswith(('+', '-')):
            rate = f"+{rate}"
        
        # Fix pitch format - Edge TTS requires +/- prefix  
        if pitch and not pitch.startswith(('+', '-')):
            pitch = f"+{pitch}"
        
        # Note: edge_tts doesn't support SSML or express-as styles
        # The style parameter is ignored - just use standard TTS
        # For emotion/style support, would need Azure Speech SDK with subscription
        if style and style != "default":
            logger.warning(f"Style '{style}' requested but edge_tts doesn't support styles. Using standard voice.")
        
        # Standard request (edge_tts doesn't support SSML)
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        
        # Generate audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            await communicate.save(tmp_path)
            
            # Read and convert to WAV format
            import librosa
            audio, sr = librosa.load(tmp_path, sr=None, mono=True)
            
            # Convert to WAV bytes
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, sr, format='WAV')
            wav_buffer.seek(0)
            
            return wav_buffer.read(), sr
            
        finally:
            # Clean up temp file
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


@app.post("/convert", response_model=ConvertResponse)
async def convert_voice(request: ConvertRequest):
    """Convert voice using RVC model"""
    global model_manager, infer_params
    
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
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
        
        # Create inference params
        from app.model_manager import RVCInferParams
        params = RVCInferParams(
            f0_up_key=request.f0_up_key,
            f0_method=request.f0_method,
            index_rate=request.index_rate,
            filter_radius=request.filter_radius,
            rms_mix_rate=request.rms_mix_rate,
            protect=request.protect,
        )
        
        # Run inference
        output_audio = model_manager.infer(audio, params=params)
        
        if output_audio is None or len(output_audio) == 0:
            raise HTTPException(status_code=500, detail="Voice conversion failed")
        
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
    index_rate: float = Field(default=0.75, description="Index blend rate")
    pitch_shift_all: int = Field(default=0, description="Pitch shift for ALL audio (vocals + instrumental) in semitones")


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
                
                logger.info(f"Separation complete: vocals={len(vocals)}, instrumental={len(instrumental)}")
                
                # Apply pitch shift to both vocals and instrumental if requested
                if request.pitch_shift_all and request.pitch_shift_all != 0:
                    logger.info(f"Applying pitch shift of {request.pitch_shift_all} semitones to both tracks")
                    vocals = pitch_shift_audio(vocals, 44100, request.pitch_shift_all)
                    instrumental = pitch_shift_audio(instrumental, 44100, request.pitch_shift_all)
                
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
            
            # Run conversion
            from app.model_manager import RVCInferParams
            params = RVCInferParams(
                f0_up_key=request.f0_up_key,
                index_rate=request.index_rate,
            )
            
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
                
                from app.model_manager import RVCInferParams
                params = RVCInferParams(
                    f0_up_key=request.f0_up_key,
                    index_rate=request.index_rate,
                )
                
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
                if request.pitch_shift_all and request.pitch_shift_all != 0:
                    logger.info(f"Applying pitch shift of {request.pitch_shift_all} semitones to instrumental")
                    instrumental_final = pitch_shift_audio(instrumental, 44100, request.pitch_shift_all)
                
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
