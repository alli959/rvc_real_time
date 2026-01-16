"""
Streaming API Module - WebSocket and Socket server for real-time audio streaming

Supports:
- Audio conversion (file upload and real-time)
- Model switching per client
- Text-to-Speech (TTS) with voice conversion
- Speech-to-Speech (real-time conversion)
"""

import asyncio
import websockets
import json
import numpy as np
import logging
from typing import Set, Optional, Dict, Any
import base64
import io
import soundfile as sf

logger = logging.getLogger(__name__)


class WebSocketServer:
    """WebSocket server for real-time audio streaming"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        stream_processor = None,
        model_manager = None,
        infer_params = None
    ):
        """
        Initialize WebSocket server
        
        Args:
            host: Server host address
            port: Server port
            stream_processor: StreamProcessor instance for audio processing (for real-time streaming)
            model_manager: ModelManager instance for batch processing (better quality)
            infer_params: RVCInferParams for voice conversion settings
        """
        self.host = host
        self.port = port
        self.stream_processor = stream_processor
        self.model_manager = model_manager
        self.infer_params = infer_params
        
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        
        # Audio buffer for accumulating chunks (per-client)
        self.client_buffers: Dict[int, list] = {}
        
        # Per-client model state (client_id -> model info)
        self.client_models: Dict[int, Dict[str, Any]] = {}
        
        # Per-client inference params overrides
        self.client_params: Dict[int, Any] = {}
    
    async def handle_client(self, websocket, path=None):
        """
        Handle individual WebSocket client connection
        
        Args:
            websocket: WebSocket connection
            path: Connection path (optional, not provided in websockets >= 10.0)
        """
        self.clients.add(websocket)
        client_id = id(websocket)
        import datetime
        logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} connected from {websocket.remote_address}")
        
        # Initialize buffer for this client
        self.client_buffers[client_id] = []
        
        try:
            async for message in websocket:
                logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Received message from client {client_id}: {message[:100]}")
                await self.process_message(websocket, message, client_id)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"[{datetime.datetime.utcnow().isoformat()}] Error handling client {client_id}: {e}")
        finally:
            self.clients.remove(websocket)
            # Clean up client state
            if client_id in self.client_buffers:
                del self.client_buffers[client_id]
            if client_id in self.client_models:
                del self.client_models[client_id]
            if client_id in self.client_params:
                del self.client_params[client_id]
    
    async def process_message(self, websocket, message, client_id):
        """
        Process incoming WebSocket message
        
        Args:
            websocket: WebSocket connection
            message: Incoming message
            client_id: Client identifier for buffer management
        """
        try:
            # Parse message
            data = json.loads(message)
            msg_type = data.get('type')
            import datetime
            logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Processing message type '{msg_type}' from client {client_id}")
            
            if msg_type == 'load_model':
                # Load a specific model for this client
                await self.handle_load_model(websocket, data, client_id)
            
            elif msg_type == 'audio':
                # Process audio data
                await self.handle_audio(websocket, data, client_id)
            
            elif msg_type == 'tts':
                # Text-to-Speech request
                await self.handle_tts(websocket, data, client_id)
            
            elif msg_type == 'config':
                # Handle configuration updates
                await self.handle_config(websocket, data, client_id)
            
            elif msg_type == 'ping':
                # Respond to ping
                await websocket.send(json.dumps({'type': 'pong'}))
            
            elif msg_type == 'list_models':
                # List available models
                await self.handle_list_models(websocket)
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
            await self.send_error(websocket, "Invalid JSON message")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_error(websocket, str(e))
    
    async def handle_load_model(self, websocket, data: dict, client_id: int):
        """Handle model loading request"""
        model_path = data.get('model_path')
        index_path = data.get('index_path')
        
        if not model_path:
            await self.send_error(websocket, "model_path is required")
            return
        
        try:
            logger.info(f"Client {client_id} loading model: {model_path}")
            
            # Load the model (this affects the global model_manager state)
            # In a production system, you might want per-client model instances
            success = self.model_manager.load_model(model_path, index_path)
            
            if success:
                # Store client's model preference
                self.client_models[client_id] = {
                    'model_path': model_path,
                    'index_path': index_path,
                    'model_name': self.model_manager.model_name
                }

                # Model warmup: run dummy inference to preload HuBERT/RMVPE for this model
                warmup_start = None
                warmup_success = False
                try:
                    import time
                    warmup_start = time.time()
                    dummy_audio = np.zeros(self.stream_processor.chunk_size if self.stream_processor else 1024, dtype=np.float32)
                    _ = self.model_manager.infer(dummy_audio, params=self.infer_params)
                    warmup_success = True
                    logger.info(f"Model warmup completed for {self.model_manager.model_name} (client {client_id}). Time: {time.time() - warmup_start:.2f}s")
                except Exception as e:
                    logger.error(f"Model warmup failed for {self.model_manager.model_name} (client {client_id}): {e}")

                response = {
                    'type': 'model_loaded',
                    'model_name': self.model_manager.model_name,
                    'model_path': model_path,
                    'has_index': self.model_manager.index_path is not None,
                    'warmup_success': warmup_success
                }
                await websocket.send(json.dumps(response))
            else:
                await self.send_error(websocket, f"Failed to load model: {model_path}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            await self.send_error(websocket, f"Error loading model: {str(e)}")
    
    async def handle_audio(self, websocket, data: dict, client_id: int):
        """Handle audio processing request"""
        audio_format = data.get('format')  # raw, webm, wav, mp3, flac, etc.
        audio_data = self.decode_audio(data.get('data'), audio_format)
        is_final = data.get('final', False)
        settings = data.get('settings', {})
        
        if audio_data is None:
            await self.send_error(websocket, "Invalid audio data")
            return
        
        # Apply client settings if provided
        infer_params = self.get_client_params(client_id, settings)
        
        # Use batch processing mode if model_manager available (better quality)
        if self.model_manager and is_final:
            # Accumulate this chunk
            self.client_buffers[client_id].append(audio_data)
            
            # Concatenate all buffered audio
            full_audio = np.concatenate(self.client_buffers[client_id])
            
            # Clear buffer for next utterance
            self.client_buffers[client_id] = []
            
            import datetime
            logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} queued final audio chunk: {len(full_audio)} samples ({len(full_audio)/16000:.2f}s)")
            # Normalize input
            max_val = np.max(np.abs(full_audio))
            if max_val > 1.0:
                full_audio = full_audio / max_val
            start_time = datetime.datetime.utcnow()
            logger.info(f"[{start_time.isoformat()}] Client {client_id} starting processing")
            processed = self.model_manager.infer(full_audio, params=infer_params)
            end_time = datetime.datetime.utcnow()
            logger.info(f"[{end_time.isoformat()}] Client {client_id} finished processing (duration: {(end_time-start_time).total_seconds():.3f}s)")
            # Apply gain and clip
            output_gain = 1.0
            if processed is not None and len(processed) > 0:
                processed = np.clip(processed * output_gain, -1.0, 1.0)
                logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} sending processed audio: {len(processed)} samples ({len(processed)/16000:.2f}s)")
                # Send processed audio back
                response = {
                    'type': 'audio',
                    'data': self.encode_audio(processed),
                    'final': True
                }
                try:
                    await websocket.send(json.dumps(response))
                    logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} processed audio sent successfully")
                except Exception as send_exc:
                    logger.error(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} failed to send processed audio: {send_exc}")
            else:
                logger.error(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} processing returned None or empty result")
                await self.send_error(websocket, "Processing failed")
        
        elif self.model_manager and not is_final:
            # Buffer chunk and send acknowledgment
            self.client_buffers[client_id].append(audio_data)
            import datetime
            logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} buffered audio chunk: {len(audio_data)} samples (buffered: {len(self.client_buffers[client_id])})")
            response = {
                'type': 'ack',
                'buffered': len(self.client_buffers[client_id])
            }
            await websocket.send(json.dumps(response))
        
        elif self.stream_processor:
            # Fallback to real-time streaming mode (lower quality but immediate)
            import datetime
            logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} processing audio chunk in real-time mode: {len(audio_data)} samples")
            processed = self.stream_processor.process_audio_chunk(audio_data)
            
            if processed is not None:
                logger.info(f"[{datetime.datetime.utcnow().isoformat()}] Client {client_id} sending real-time processed audio: {len(processed)} samples")
                response = {
                    'type': 'audio',
                    'data': self.encode_audio(processed)
                }
                await websocket.send(json.dumps(response))
    
    async def handle_tts(self, websocket, data: dict, client_id: int):
        """Handle Text-to-Speech request with Edge TTS and voice conversion"""
        text = data.get('text', '')
        settings = data.get('settings', {})
        voice = data.get('voice', 'en-US-GuyNeural')
        rate = data.get('rate', '+0%')
        pitch_shift = data.get('pitch', '+0Hz')  # TTS pitch
        
        if not text.strip():
            await self.send_error(websocket, "Text is required for TTS")
            return
        
        try:
            logger.info(f"TTS request from client {client_id}: {text[:50]}...")
            
            # Step 1: Generate TTS audio using Edge TTS
            try:
                import edge_tts
                import tempfile
                import os
                import soundfile as sf
                
                # Fix rate format - Edge TTS requires +/- prefix
                if rate and not rate.startswith(('+', '-')):
                    rate = f"+{rate}"
                
                # Fix pitch format - Edge TTS requires +/- prefix  
                if pitch_shift and not pitch_shift.startswith(('+', '-')):
                    pitch_shift = f"+{pitch_shift}"
                
                # Generate TTS
                communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch_shift)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                await communicate.save(tmp_path)
                
                # Load the audio
                import librosa
                audio, sr = librosa.load(tmp_path, sr=40000, mono=True)  # RVC expects 40kHz
                
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
            except ImportError:
                await self.send_error(websocket, "edge-tts not installed. Install with: pip install edge-tts")
                return
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
                await self.send_error(websocket, f"TTS generation failed: {str(e)}")
                return
            
            # Step 2: Convert the TTS audio using RVC model if loaded
            if self.model_manager and self.model_manager.model_name:
                try:
                    # Get inference params with settings overrides
                    params = self.get_client_params(client_id, settings)
                    
                    # Run voice conversion
                    converted_audio = self.model_manager.infer(audio, params=params)
                    
                    if converted_audio is not None:
                        audio = converted_audio
                        logger.info(f"TTS audio converted with model {self.model_manager.model_name}")
                    
                except Exception as e:
                    logger.error(f"Voice conversion failed: {e}")
                    # Continue with unconverted TTS audio
            
            # Step 3: Convert to WAV and encode as base64
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, 40000, format='WAV')
            wav_buffer.seek(0)
            audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
            
            # Send the audio response
            response = {
                'type': 'tts_audio',
                'data': audio_base64,
                'sample_rate': 40000,
                'model_applied': self.model_manager.model_name if self.model_manager and self.model_manager.model_name else None
            }
            await websocket.send(json.dumps(response))
            logger.info(f"TTS audio sent to client {client_id}")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            await self.send_error(websocket, f"TTS error: {str(e)}")
    
    async def handle_config(self, websocket, data: dict, client_id: int):
        """Handle configuration update"""
        settings = data.get('settings', {})
        
        # Store client-specific settings
        if settings:
            self.client_params[client_id] = settings
        
        response = {'type': 'config', 'status': 'ok'}
        await websocket.send(json.dumps(response))
    
    async def handle_list_models(self, websocket):
        """Handle request to list available models"""
        if self.model_manager:
            models = self.model_manager.list_available_models()
            response = {
                'type': 'models_list',
                'models': models,
                'current_model': self.model_manager.model_name
            }
        else:
            response = {
                'type': 'models_list',
                'models': [],
                'current_model': None
            }
        await websocket.send(json.dumps(response))
    
    async def send_error(self, websocket, message: str):
        """Send error message to client"""
        response = {'type': 'error', 'message': message}
        await websocket.send(json.dumps(response))
    
    def get_client_params(self, client_id: int, settings: dict):
        """Get inference params for client, applying any settings overrides"""
        from app.model_manager import RVCInferParams
        
        # Start with default params
        params = self.infer_params or RVCInferParams()
        
        # Apply stored client settings
        stored = self.client_params.get(client_id, {})
        
        # Apply request-specific settings (takes precedence)
        all_settings = {**stored, **settings}
        
        if all_settings:
            return RVCInferParams(
                sid=all_settings.get('sid', params.sid),
                f0_up_key=all_settings.get('f0_up_key', params.f0_up_key),
                f0_method=all_settings.get('f0_method', params.f0_method),
                index_rate=all_settings.get('index_rate', params.index_rate),
                filter_radius=all_settings.get('filter_radius', params.filter_radius),
                rms_mix_rate=all_settings.get('rms_mix_rate', params.rms_mix_rate),
                protect=all_settings.get('protect', params.protect),
                resample_sr=all_settings.get('resample_sr', params.resample_sr),
            )
        
        return params
    
    def decode_audio(self, encoded_data: str, audio_format: str = None) -> Optional[np.ndarray]:
        """
        Decode base64 encoded audio data. Supports various audio formats.
        
        Args:
            encoded_data: Base64 encoded audio
            audio_format: Optional format hint ('wav', 'mp3', 'flac', 'webm', 'raw')
            
        Returns:
            Numpy array of audio samples (float32, mono, resampled to 16kHz)
        """
        try:
            import soundfile as sf
            import librosa
            
            audio_bytes = base64.b64decode(encoded_data)
            logger.info(f"Decoding audio: {len(audio_bytes)} bytes, format hint: {audio_format}")
            
            # If it's raw float32 PCM and size is valid
            if audio_format == 'raw' or (audio_format is None and len(audio_bytes) % 4 == 0):
                try:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    if len(audio_array) > 0 and np.abs(audio_array).max() <= 2.0:
                        # Looks like valid float32 PCM
                        logger.info(f"Decoded as raw float32 PCM: {len(audio_array)} samples")
                        return audio_array
                except Exception:
                    pass
            
            # For webm format, use pydub which handles it well via ffmpeg
            if audio_format == 'webm':
                try:
                    from pydub import AudioSegment
                    audio_io = io.BytesIO(audio_bytes)
                    audio_segment = AudioSegment.from_file(audio_io, format='webm')
                    # Convert to mono and get raw samples
                    audio_segment = audio_segment.set_channels(1)
                    sr = audio_segment.frame_rate
                    # Get raw samples as numpy array
                    samples = np.array(audio_segment.get_array_of_samples())
                    # Normalize to float32 [-1, 1]
                    audio_array = samples.astype(np.float32) / 32768.0
                    logger.info(f"Decoded webm with pydub: {len(audio_array)} samples at {sr}Hz")
                    
                    # Resample to 16kHz if needed
                    if sr != 16000:
                        logger.info(f"Resampling from {sr}Hz to 16000Hz")
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    
                    return audio_array
                except Exception as pydub_error:
                    logger.warning(f"pydub failed for webm: {pydub_error}, trying librosa")
            
            # Try to decode as audio file using soundfile/librosa
            audio_io = io.BytesIO(audio_bytes)
            
            try:
                # Try soundfile first (handles WAV, FLAC, OGG)
                audio_array, sr = sf.read(audio_io, dtype='float32')
                logger.info(f"Decoded audio with soundfile: {len(audio_array)} samples at {sr}Hz")
            except Exception as sf_error:
                # Fall back to librosa (handles more formats including MP3)
                audio_io.seek(0)
                try:
                    audio_array, sr = librosa.load(audio_io, sr=None, mono=True)
                    logger.info(f"Decoded audio with librosa: {len(audio_array)} samples at {sr}Hz")
                except Exception as lr_error:
                    logger.error(f"Failed to decode audio - soundfile: {sf_error}, librosa: {lr_error}")
                    return None
            
            # Convert to mono if stereo
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Ensure float32
            audio_array = audio_array.astype(np.float32)
            
            # Resample to 16kHz if needed (RVC expects 16kHz input)
            if sr != 16000:
                logger.info(f"Resampling from {sr}Hz to 16000Hz")
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def encode_audio(self, audio_data: np.ndarray, as_wav: bool = True) -> str:
        """
        Encode audio data to base64
        
        Args:
            audio_data: Numpy array of audio samples (float32)
            as_wav: If True, encode as WAV file (playable in browser)
            
        Returns:
            Base64 encoded audio string
        """
        if as_wav:
            import soundfile as sf
            
            # Write to WAV format in memory
            wav_io = io.BytesIO()
            # RVC outputs at model's target sample rate (usually 40000 or 48000)
            # We'll use 16000 for consistency, but this should match the actual output
            sample_rate = 16000
            sf.write(wav_io, audio_data.astype(np.float32), sample_rate, format='WAV')
            wav_io.seek(0)
            return base64.b64encode(wav_io.read()).decode('utf-8')
        else:
            # Raw float32 bytes
            audio_bytes = audio_data.astype(np.float32).tobytes()
            return base64.b64encode(audio_bytes).decode('utf-8')
    
    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected clients
        
        Args:
            message: Message dictionary to broadcast
        """
        if self.clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_str) for client in self.clients],
                return_exceptions=True
            )
    
    async def start(self):
        """Start the WebSocket server"""
        # Increase max_size to 16 MB (default is 1 MB)
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=None,  # Disable ping to avoid timeout during long processing
            ping_timeout=None,
            close_timeout=60,
            max_size=16 * 1024 * 1024  # 16 MB
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port} (max_size=16MB)")
        # Keep server running
        await asyncio.Future()
    
    def run(self):
        """Run the WebSocket server (blocking)"""
        asyncio.run(self.start())


class SocketServer:
    """TCP Socket server for raw audio streaming"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9876,
        stream_processor = None
    ):
        """
        Initialize socket server
        
        Args:
            host: Server host address
            port: Server port
            stream_processor: StreamProcessor instance for audio processing
        """
        self.host = host
        self.port = port
        self.stream_processor = stream_processor
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Handle individual socket client
        
        Args:
            reader: Stream reader
            writer: Stream writer
        """
        addr = writer.get_extra_info('peername')
        logger.info(f"Socket client connected from {addr}")
        
        try:
            while True:
                # Read chunk size (4 bytes)
                size_data = await reader.read(4)
                if not size_data:
                    break
                
                chunk_size = int.from_bytes(size_data, byteorder='big')
                
                # Read audio data
                audio_data = await reader.read(chunk_size)
                if not audio_data:
                    break
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                # Process audio
                if self.stream_processor:
                    processed = self.stream_processor.process_audio_chunk(audio_array)
                    
                    if processed is not None:
                        # Send back processed audio
                        processed_bytes = processed.astype(np.float32).tobytes()
                        size_bytes = len(processed_bytes).to_bytes(4, byteorder='big')
                        
                        writer.write(size_bytes + processed_bytes)
                        await writer.drain()
        
        except Exception as e:
            logger.error(f"Error handling socket client: {e}")
        
        finally:
            logger.info(f"Socket client {addr} disconnected")
            writer.close()
            await writer.wait_closed()
    
    async def start(self):
        """Start the socket server"""
        server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port
        )
        
        logger.info(f"Socket server started on {self.host}:{self.port}")
        
        async with server:
            await server.serve_forever()
    
    def run(self):
        """Run the socket server (blocking)"""
        asyncio.run(self.start())
