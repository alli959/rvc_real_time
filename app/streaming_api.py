"""
Streaming API Module - WebSocket and Socket server for real-time audio streaming
"""

import asyncio
import websockets
import json
import numpy as np
import logging
from typing import Set, Optional
import base64

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
        self.client_buffers = {}  # client_id -> list of audio chunks
    
    async def handle_client(self, websocket, path):
        """
        Handle individual WebSocket client connection
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        self.clients.add(websocket)
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        
        # Initialize buffer for this client
        self.client_buffers[client_id] = []
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message, client_id)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        
        finally:
            self.clients.remove(websocket)
            # Clean up buffer
            if client_id in self.client_buffers:
                del self.client_buffers[client_id]
    
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
            
            if msg_type == 'audio':
                # Process audio data
                audio_data = self.decode_audio(data.get('data'))
                is_final = data.get('final', False)  # Check if this is the final chunk
                
                if audio_data is not None:
                    # Use batch processing mode if model_manager available (better quality)
                    # This processes entire utterances instead of tiny chunks
                    if self.model_manager and is_final:
                        # Accumulate this chunk
                        self.client_buffers[client_id].append(audio_data)
                        
                        # Concatenate all buffered audio
                        full_audio = np.concatenate(self.client_buffers[client_id])
                        
                        # Clear buffer for next utterance
                        self.client_buffers[client_id] = []
                        
                        # Process entire audio at once (better quality)
                        logger.info(f"Processing complete utterance: {len(full_audio)} samples ({len(full_audio)/16000:.2f}s)")
                        
                        # Normalize input
                        max_val = np.max(np.abs(full_audio))
                        if max_val > 1.0:
                            full_audio = full_audio / max_val
                        
                        # Process entire audio in one pass with proper params
                        processed = self.model_manager.infer(full_audio, params=self.infer_params)
                        
                        # Apply gain and clip
                        output_gain = 1.0
                        if processed is not None and len(processed) > 0:
                            processed = np.clip(processed * output_gain, -1.0, 1.0)
                            logger.info(f"Sending processed audio: {len(processed)} samples ({len(processed)/16000:.2f}s)")
                            
                            # Send processed audio back
                            response = {
                                'type': 'audio',
                                'data': self.encode_audio(processed),
                                'final': True
                            }
                            await websocket.send(json.dumps(response))
                        else:
                            logger.error(f"Processing returned None or empty result")
                    
                    elif self.model_manager and not is_final:
                        # Buffer chunk and send acknowledgment
                        self.client_buffers[client_id].append(audio_data)
                        response = {
                            'type': 'ack',
                            'buffered': len(self.client_buffers[client_id])
                        }
                        await websocket.send(json.dumps(response))
                    
                    elif self.stream_processor:
                        # Fallback to real-time streaming mode (lower quality but immediate)
                        processed = self.stream_processor.process_audio_chunk(audio_data)
                        
                        if processed is not None:
                            # Send processed audio back
                            response = {
                                'type': 'audio',
                                'data': self.encode_audio(processed)
                            }
                            await websocket.send(json.dumps(response))
            
            elif msg_type == 'config':
                # Handle configuration updates
                response = {'type': 'config', 'status': 'ok'}
                await websocket.send(json.dumps(response))
            
            elif msg_type == 'ping':
                # Respond to ping
                await websocket.send(json.dumps({'type': 'pong'}))
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def decode_audio(self, encoded_data: str) -> Optional[np.ndarray]:
        """
        Decode base64 encoded audio data
        
        Args:
            encoded_data: Base64 encoded audio
            
        Returns:
            Numpy array of audio samples
        """
        try:
            audio_bytes = base64.b64decode(encoded_data)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            return audio_array
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return None
    
    def encode_audio(self, audio_data: np.ndarray) -> str:
        """
        Encode audio data to base64
        
        Args:
            audio_data: Numpy array of audio samples
            
        Returns:
            Base64 encoded audio string
        """
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
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        
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
