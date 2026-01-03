#!/usr/bin/env python3
"""
Real-time Microphone Client for RVC Voice Conversion

Captures audio from your microphone in utterances (with pauses),
sends complete utterances to the RVC server for processing,
and plays back the converted audio.

This provides better quality than per-chunk streaming since RVC
needs larger audio segments for proper pitch/feature extraction.

Requirements:
    pip install sounddevice numpy websockets

Usage:
    python examples/realtime_mic_client.py
    python examples/realtime_mic_client.py --host localhost --port 8765
    
Controls:
    - Speak into your microphone
    - Pause for >0.5s to trigger processing
    - Press Ctrl+C to stop
"""

import asyncio
import websockets
import json
import numpy as np
import base64
import argparse
import sys
import queue
import threading

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class RealtimeMicClient:
    """Utterance-based microphone to RVC server client."""
    
    def __init__(self, host: str = "localhost", port: int = 8765, 
                 sample_rate: int = 16000, silence_threshold: float = 0.01,
                 silence_duration: float = 0.5):
        self.uri = f"ws://{host}:{port}"
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.websocket = None
        
        # For utterance detection
        self.buffer = []
        self.silent_chunks = 0
        self.max_silent_chunks = int((silence_duration * sample_rate) / 1024)  # blocksize assumption
        
        # Playback buffer for smooth output
        self.playback_buffer = np.array([], dtype=np.float32)
        self.playback_lock = threading.Lock()
        
    def audio_callback(self, indata, outdata, frames, time, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        
        # Put input audio in queue
        self.input_queue.put(indata[:, 0].copy())
        
        # Play from playback buffer
        with self.playback_lock:
            # Check if new audio arrived in output_queue and add to buffer
            try:
                while True:
                    new_audio = self.output_queue.get_nowait()
                    self.playback_buffer = np.concatenate([self.playback_buffer, new_audio])
            except queue.Empty:
                pass
            
            # Play from buffer
            if len(self.playback_buffer) >= frames:
                outdata[:, 0] = self.playback_buffer[:frames]
                self.playback_buffer = self.playback_buffer[frames:]
            elif len(self.playback_buffer) > 0:
                outdata[:len(self.playback_buffer), 0] = self.playback_buffer
                outdata[len(self.playback_buffer):, 0] = 0
                self.playback_buffer = np.array([], dtype=np.float32)
            else:
                outdata[:] = 0
    
    async def process_utterances(self):
        """Buffer audio and send complete utterances."""
        buffer = []
        silent_chunks = 0
        min_utterance_length = int(self.sample_rate * 0.5)  # At least 0.5s
        
        while self.running:
            try:
                # Get audio from input queue
                chunk = self.input_queue.get(timeout=0.1)
                
                # Check if chunk is silent
                rms = np.sqrt(np.mean(chunk ** 2))
                is_silent = rms < self.silence_threshold
                
                if not is_silent:
                    # Speaking - add to buffer
                    buffer.append(chunk)
                    silent_chunks = 0
                    print(".", end="", flush=True)  # Activity indicator
                elif len(buffer) > 0:
                    # Silence after speech - count silent chunks
                    buffer.append(chunk)  # Include some silence at end
                    silent_chunks += 1
                    
                    # If enough silence, process utterance
                    if silent_chunks >= self.max_silent_chunks:
                        utterance = np.concatenate(buffer)
                        
                        # Only process if long enough
                        if len(utterance) >= min_utterance_length:
                            print(f"\n[Processing {len(utterance)/self.sample_rate:.2f}s utterance...]", end="", flush=True)
                            
                            # Encode and send
                            audio_b64 = base64.b64encode(utterance.astype(np.float32).tobytes()).decode('utf-8')
                            await self.websocket.send(json.dumps({
                                "type": "audio",
                                "data": audio_b64,
                                "final": True  # Complete utterance
                            }))
                        
                        # Clear buffer
                        buffer = []
                        silent_chunks = 0
                    
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"\nProcess error: {e}")
                break
    
    async def receive_audio(self):
        """Receive processed audio from server."""
        while self.running:
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=0.5)
                data = json.loads(response)
                
                if data.get('type') == 'audio':
                    audio_bytes = base64.b64decode(data['data'])
                    audio = np.frombuffer(audio_bytes, dtype=np.float32)
                    print(f" Received {len(audio)} samples, adding to playback queue...")
                    self.output_queue.put(audio)
                    print(f" Done! ({len(audio)/self.sample_rate:.2f}s)")
                elif data.get('type') == 'ack':
                    # Server is buffering chunks
                    pass
                elif data.get('type') == 'error':
                    print(f"\nServer error: {data.get('message')}")
                else:
                    print(f"\nUnknown response type: {data.get('type')}")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"\nReceive error: {e}")
                break
    
    async def run(self):
        """Main run loop."""
        if not HAS_SOUNDDEVICE:
            print("Error: sounddevice not installed")
            print("Install with: pip install sounddevice")
            return
        
        print(f"Connecting to {self.uri}...")
        
        try:
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                print("Connected!")
                
                # Test connection
                await websocket.send(json.dumps({"type": "ping"}))
                response = await websocket.recv()
                print(f"Server ready: {response}")
                
                self.running = True
                
                print(f"\nListening for speech...")
                print(f"Sample rate: {self.sample_rate}Hz")
                print(f"Silence threshold: {self.silence_threshold}, Duration: {self.silence_duration}s")
                print(f"Speak into your microphone. Pause briefly to process.")
                print("Press Ctrl+C to stop\n")
                
                # Start audio stream
                with sd.Stream(
                    samplerate=self.sample_rate,
                    blocksize=1024,
                    channels=1,
                    dtype=np.float32,
                    callback=self.audio_callback
                ):
                    # Run processing tasks
                    process_task = asyncio.create_task(self.process_utterances())
                    recv_task = asyncio.create_task(self.receive_audio())
                    
                    try:
                        await asyncio.gather(process_task, recv_task)
                    except asyncio.CancelledError:
                        pass
                        
        except ConnectionRefusedError:
            print(f"\nError: Could not connect to server at {self.uri}")
            print("Make sure the server is running:")
            print("    python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth")
        finally:
            self.running = False


def main():
    parser = argparse.ArgumentParser(
        description="Utterance-based microphone voice conversion via RVC server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This client captures audio from your microphone in utterances (segments of speech
separated by pauses), sends complete utterances to the RVC server for better quality
voice conversion, and plays the converted audio through your speakers.

Requirements:
    pip install sounddevice

Example:
    python examples/realtime_mic_client.py
    
Make sure the server is running:
    python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth
"""
    )
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (default: 16000)")
    parser.add_argument("--silence-threshold", type=float, default=0.01, 
                        help="RMS threshold for silence detection (default: 0.01)")
    parser.add_argument("--silence-duration", type=float, default=0.5,
                        help="Duration of silence to trigger processing (default: 0.5s)")
    
    args = parser.parse_args()
    
    client = RealtimeMicClient(
        host=args.host,
        port=args.port,
        sample_rate=args.sample_rate,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration
    )
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
