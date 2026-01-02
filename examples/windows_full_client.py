#!/usr/bin/env python3
"""
Windows Full Client for RVC Voice Conversion

This runs entirely on Windows:
- Captures audio from your real microphone
- Sends it to the RVC server (WSL or local) for processing  
- Outputs processed audio to VB-Cable for Discord

This is a port of realtime_mic_client.py optimized for Windows + VB-Cable.

Run this in Windows PowerShell:
    python windows_full_client.py --list-devices
    python windows_full_client.py --input-device "Jabra" --output-device "CABLE Input"
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
    print("Error: sounddevice not installed")
    print("Install with: pip install sounddevice numpy websockets")
    sys.exit(1)


SAMPLE_RATE = 16000
CHUNK_SIZE = 1024


def list_devices():
    """List available audio devices"""
    print("\n=== Available Audio Devices ===\n")
    devices = sd.query_devices()
    
    print("INPUT DEVICES (Microphones):")
    print("-" * 60)
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            default = " [DEFAULT]" if i == sd.default.device[0] else ""
            print(f"  {i}: {dev['name']}{default}")
    
    print("\nOUTPUT DEVICES (for VB-Cable):")
    print("-" * 60)
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            default = " [DEFAULT]" if i == sd.default.device[1] else ""
            cable = " [VB-CABLE]" if 'cable' in dev['name'].lower() else ""
            print(f"  {i}: {dev['name']}{default}{cable}")
    
    print("\n=== Recommended Setup ===")
    print("1. Use your real mic (e.g., 'Jabra') as --input-device")
    print("2. Use 'CABLE Input' as --output-device")
    print("3. In Discord, select 'CABLE Output' as your microphone")
    print()


def find_device(name_pattern, is_input=False):
    """Find device index by name pattern"""
    devices = sd.query_devices()
    name_lower = name_pattern.lower()
    
    for i, dev in enumerate(devices):
        if is_input and dev['max_input_channels'] == 0:
            continue
        if not is_input and dev['max_output_channels'] == 0:
            continue
        if name_lower in dev['name'].lower():
            return i, dev['name']
    return None, None


class RVCClient:
    """Utterance-based RVC client - matches realtime_mic_client behavior."""
    
    def __init__(self, server_host, server_port, input_device, output_device,
                 silence_threshold=0.01, silence_duration=0.5):
        self.uri = f"ws://{server_host}:{server_port}"
        self.input_device = input_device
        self.output_device = output_device
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.websocket = None
        self.loop = None  # Will be set when running
        
        # Playback buffer (numpy array for efficiency)
        self.playback_buffer = np.array([], dtype=np.float32)
        self.playback_lock = threading.Lock()
        
        # Utterance detection
        self.max_silent_chunks = int((silence_duration * SAMPLE_RATE) / CHUNK_SIZE)
    
    def input_callback(self, indata, frames, time_info, status):
        """Capture audio from microphone"""
        if status:
            print(f"Input status: {status}", file=sys.stderr)
        self.input_queue.put(indata[:, 0].copy())
    
    def output_callback(self, outdata, frames, time_info, status):
        """Output to VB-Cable"""
        if status:
            print(f"Output status: {status}", file=sys.stderr)
        
        with self.playback_lock:
            # Get any new audio from output queue
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
        """Buffer audio and send complete utterances - non-blocking async version"""
        buffer = []
        silent_chunks = 0
        min_utterance_length = int(SAMPLE_RATE * 0.5)  # At least 0.5s
        
        while self.running:
            try:
                # Use run_in_executor to make queue.get non-blocking for asyncio
                try:
                    chunk = await asyncio.wait_for(
                        self.loop.run_in_executor(None, lambda: self.input_queue.get(timeout=0.05)),
                        timeout=0.1
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    await asyncio.sleep(0.01)
                    continue
                
                # Check if chunk is silent
                rms = np.sqrt(np.mean(chunk ** 2))
                is_silent = rms < self.silence_threshold
                
                if not is_silent:
                    buffer.append(chunk)
                    silent_chunks = 0
                    print(".", end="", flush=True)
                elif len(buffer) > 0:
                    buffer.append(chunk)
                    silent_chunks += 1
                    
                    if silent_chunks >= self.max_silent_chunks:
                        utterance = np.concatenate(buffer)
                        
                        if len(utterance) >= min_utterance_length:
                            print(f"\n[Sending {len(utterance)/SAMPLE_RATE:.2f}s]", end="", flush=True)
                            
                            audio_b64 = base64.b64encode(
                                utterance.astype(np.float32).tobytes()
                            ).decode('utf-8')
                            
                            await self.websocket.send(json.dumps({
                                "type": "audio",
                                "data": audio_b64,
                                "final": True
                            }))
                        
                        buffer = []
                        silent_chunks = 0
                        
            except Exception as e:
                print(f"\nProcess error: {e}")
                break
    
    async def receive_audio(self):
        """Receive processed audio from RVC server"""
        while self.running:
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=0.5)
                data = json.loads(response)
                
                if data.get('type') == 'audio':
                    audio_bytes = base64.b64decode(data['data'])
                    audio = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # Add to playback queue
                    self.output_queue.put(audio)
                    print(f" → Playing ({len(audio)/SAMPLE_RATE:.2f}s)")
                    
                elif data.get('type') == 'ack':
                    pass
                elif data.get('type') == 'error':
                    print(f"\nServer error: {data.get('message')}")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"\nReceive error: {e}")
                break
    
    async def run(self):
        """Main run loop"""
        # Resolve devices
        input_idx, input_name = find_device(self.input_device, is_input=True) if self.input_device else (None, None)
        output_idx, output_name = find_device(self.output_device, is_input=False) if self.output_device else (None, None)
        
        if self.input_device and input_idx is None:
            print(f"Error: Input device '{self.input_device}' not found")
            list_devices()
            return
        
        if self.output_device and output_idx is None:
            print(f"Error: Output device '{self.output_device}' not found")
            list_devices()
            return
        
        # Get device names
        devices = sd.query_devices()
        if input_name is None:
            input_name = devices[sd.default.device[0]]['name']
        if output_name is None:
            output_name = devices[sd.default.device[1]]['name']
        
        print(f"\n{'='*60}")
        print(f"RVC Voice Changer")
        print(f"{'='*60}")
        print(f"Input (Mic):         {input_name}")
        print(f"Output (VB-Cable):   {output_name}")
        print(f"Server:              {self.uri}")
        print(f"Sample Rate:         {SAMPLE_RATE}Hz")
        print(f"Silence Threshold:   {self.silence_threshold}")
        print(f"Silence Duration:    {self.silence_duration}s")
        print(f"{'='*60}")
        
        print(f"\nConnecting to RVC server...")
        
        try:
            # Store the event loop for run_in_executor
            self.loop = asyncio.get_event_loop()
            
            async with websockets.connect(
                self.uri,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=60
            ) as websocket:
                self.websocket = websocket
                
                await websocket.send(json.dumps({"type": "ping"}))
                response = await websocket.recv()
                print(f"Connected! Server: {response}")
                
                self.running = True
                
                print(f"\n✓ Voice changer active!")
                print(f"  In Discord, select 'CABLE Output' as your microphone")
                print(f"\nSpeak into your microphone. Pause briefly to process.")
                print("Press Ctrl+C to stop.\n")
                
                # Use separate input/output streams
                with sd.InputStream(
                    device=input_idx,
                    samplerate=SAMPLE_RATE,
                    blocksize=CHUNK_SIZE,
                    channels=1,
                    dtype=np.float32,
                    callback=self.input_callback
                ), sd.OutputStream(
                    device=output_idx,
                    samplerate=SAMPLE_RATE,
                    blocksize=CHUNK_SIZE,
                    channels=1,
                    dtype=np.float32,
                    callback=self.output_callback
                ):
                    process_task = asyncio.create_task(self.process_utterances())
                    recv_task = asyncio.create_task(self.receive_audio())
                    
                    try:
                        await asyncio.gather(process_task, recv_task)
                    except asyncio.CancelledError:
                        pass
                        
        except ConnectionRefusedError:
            print(f"\nError: Could not connect to RVC server at {self.uri}")
            print("\nMake sure the server is running:")
            print("  python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth --index ./assets/models/BillCipher/BillCipher.index")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.running = False


def main():
    parser = argparse.ArgumentParser(
        description="RVC Voice Changer - Windows client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List audio devices
    python windows_full_client.py --list-devices
    
    # Run with specific devices
    python windows_full_client.py --input-device "Jabra" --output-device "CABLE Input"
    
    # Adjust sensitivity for quiet microphone
    python windows_full_client.py --input-device "Jabra" --output-device "CABLE Input" --silence-threshold 0.005

Setup:
    1. Start the RVC server:
       python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth
    
    2. Run this client:
       python windows_full_client.py --input-device "Jabra" --output-device "CABLE Input"
    
    3. In Discord, select "CABLE Output" as your microphone
"""
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--host", default="localhost", help="RVC server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="RVC server port (default: 8765)")
    parser.add_argument("--input-device", "-i", help="Input device name (your microphone)")
    parser.add_argument("--output-device", "-o", help="Output device name (VB-Cable)")
    parser.add_argument("--silence-threshold", type=float, default=0.01, 
                        help="Silence threshold - lower = more sensitive (default: 0.01)")
    parser.add_argument("--silence-duration", type=float, default=0.5, 
                        help="Pause duration before processing (default: 0.5s)")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
        return
    
    client = RVCClient(
        server_host=args.host,
        server_port=args.port,
        input_device=args.input_device,
        output_device=args.output_device,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration
    )
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
