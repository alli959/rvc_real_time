#!/usr/bin/env python3
"""
Virtual Microphone Client for RVC Voice Conversion

Routes your real microphone through RVC and outputs to a virtual microphone
that other applications (Discord, Zoom, etc.) can use as input.

Your Voice → RVC Processing → Virtual Mic → Discord/Zoom/etc.

Requirements:
    pip install sounddevice numpy websockets

Linux Setup (PulseAudio):
    # Create virtual audio devices (run once)
    ./examples/setup_virtual_mic.sh
    
    # Or manually:
    pactl load-module module-null-sink sink_name=RVC_Sink sink_properties=device.description="RVC_Output"
    pactl load-module module-virtual-source source_name=RVC_Mic master=RVC_Sink.monitor source_properties=device.description="RVC_Microphone"

Windows Setup:
    1. Install VB-Cable: https://vb-audio.com/Cable/
    2. Use "CABLE Input" as output device in this script
    3. Select "CABLE Output" as microphone in Discord

Usage:
    # List available audio devices
    python3 examples/virtual_mic_client.py --list-devices
    
    # Run with auto-detected devices
    python3 examples/virtual_mic_client.py
    
    # Run with specific output device (virtual mic)
    python3 examples/virtual_mic_client.py --output-device "RVC_Sink"
    
    # On Windows with VB-Cable
    python3 examples/virtual_mic_client.py --output-device "CABLE Input"
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
import socket
import struct

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def list_audio_devices():
    """List all available audio devices."""
    print("\n=== Available Audio Devices ===\n")
    devices = sd.query_devices()
    
    print("INPUT DEVICES (Microphones):")
    print("-" * 50)
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            default = " [DEFAULT]" if i == sd.default.device[0] else ""
            print(f"  {i}: {dev['name']}{default}")
    
    print("\nOUTPUT DEVICES (Speakers/Virtual Mics):")
    print("-" * 50)
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            default = " [DEFAULT]" if i == sd.default.device[1] else ""
            virtual = " [VIRTUAL]" if any(x in dev['name'].lower() for x in ['virtual', 'cable', 'rvc', 'null', 'sink']) else ""
            print(f"  {i}: {dev['name']}{default}{virtual}")
    
    print("\n=== Recommended Setup ===")
    print("1. Use your real microphone as INPUT")
    print("2. Use a virtual device (RVC_Sink, CABLE Input) as OUTPUT")
    print("3. In Discord, select the virtual mic (RVC_Mic, CABLE Output) as your microphone")
    print()


def find_device(name_pattern: str, is_input: bool = False) -> int:
    """Find device index by name pattern."""
    devices = sd.query_devices()
    name_lower = name_pattern.lower()
    
    for i, dev in enumerate(devices):
        if is_input and dev['max_input_channels'] == 0:
            continue
        if not is_input and dev['max_output_channels'] == 0:
            continue
        if name_lower in dev['name'].lower():
            return i
    return None


class VirtualMicClient:
    """Routes microphone through RVC to virtual audio device."""
    
    def __init__(self, host: str = "localhost", port: int = 8765,
                 sample_rate: int = 16000, 
                 input_device = None, output_device = None,
                 silence_threshold: float = 0.01,
                 silence_duration: float = 0.5,
                 network_output: str = None):
        self.uri = f"ws://{host}:{port}"
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.output_device = output_device
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.network_output = network_output  # "host:port" for Windows receiver
        
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.websocket = None
        self.network_socket = None
        
        # Playback buffer
        self.playback_buffer = np.array([], dtype=np.float32)
        self.playback_lock = threading.Lock()
        
        # Utterance detection
        self.max_silent_chunks = int((silence_duration * sample_rate) / 1024)
    
    def input_callback(self, indata, frames, time, status):
        """Capture audio from real microphone."""
        if status:
            print(f"Input status: {status}", file=sys.stderr)
        self.input_queue.put(indata[:, 0].copy())
    
    def output_callback(self, outdata, frames, time, status):
        """Output processed audio to virtual microphone."""
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
            
            # Output from buffer
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
        """Buffer audio and send complete utterances to RVC."""
        buffer = []
        silent_chunks = 0
        min_utterance_length = int(self.sample_rate * 0.5)
        
        while self.running:
            try:
                chunk = self.input_queue.get(timeout=0.1)
                
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
                            print(f"\n[Processing {len(utterance)/self.sample_rate:.2f}s]", end="", flush=True)
                            
                            audio_b64 = base64.b64encode(utterance.astype(np.float32).tobytes()).decode('utf-8')
                            await self.websocket.send(json.dumps({
                                "type": "audio",
                                "data": audio_b64,
                                "final": True
                            }))
                        
                        buffer = []
                        silent_chunks = 0
                    
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"\nProcess error: {e}")
                break
    
    async def receive_audio(self):
        """Receive converted audio from RVC server."""
        while self.running:
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=0.5)
                data = json.loads(response)
                
                if data.get('type') == 'audio':
                    audio_bytes = base64.b64decode(data['data'])
                    audio = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    if self.network_socket:
                        # Send to Windows receiver over network
                        try:
                            audio_data = audio.astype(np.float32).tobytes()
                            header = struct.pack('>I', len(audio_data))
                            self.network_socket.sendall(header + audio_data)
                            print(f" → Network ({len(audio)/self.sample_rate:.2f}s)")
                        except Exception as e:
                            print(f"\nNetwork send error: {e}")
                    else:
                        # Output to local audio device
                        self.output_queue.put(audio)
                        print(f" → Virtual Mic ({len(audio)/self.sample_rate:.2f}s)")
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
        """Main run loop."""
        if not HAS_SOUNDDEVICE:
            print("Error: sounddevice not installed")
            print("Install with: pip install sounddevice")
            return
        
        # Handle network output mode
        use_network = self.network_output is not None
        
        if use_network:
            # Parse network target
            if ':' in self.network_output:
                net_host, net_port = self.network_output.rsplit(':', 1)
                net_port = int(net_port)
            else:
                net_host = self.network_output
                net_port = 9999
            
            print(f"\n{'='*60}")
            print(f"RVC Virtual Microphone (Network Mode)")
            print(f"{'='*60}")
            print(f"Audio Output:        {net_host}:{net_port} (Windows receiver)")
            print(f"Sample Rate:         {self.sample_rate}Hz")
            print(f"RVC Server:          {self.uri}")
            print(f"{'='*60}")
            
            print(f"\nConnecting to Windows audio receiver at {net_host}:{net_port}...")
            try:
                self.network_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.network_socket.connect((net_host, net_port))
                print(f"Connected to Windows receiver!")
            except Exception as e:
                print(f"\nError: Could not connect to Windows receiver: {e}")
                print("\nMake sure the Windows receiver is running:")
                print(f"    python windows_audio_receiver.py --output-device \"CABLE Input\"")
                return
        else:
            # Resolve device names to indices
            input_dev = self.input_device
            output_dev = self.output_device
            
            if isinstance(input_dev, str):
                input_dev = find_device(input_dev, is_input=True)
                if input_dev is None:
                    print(f"Error: Input device not found. Use --list-devices to see available devices.")
                    return
            
            if isinstance(output_dev, str):
                output_dev = find_device(output_dev, is_input=False)
                if output_dev is None:
                    print(f"Error: Output device not found. Use --list-devices to see available devices.")
                    return
            
            # Show device info
            devices = sd.query_devices()
            input_name = devices[input_dev]['name'] if input_dev is not None else devices[sd.default.device[0]]['name']
            output_name = devices[output_dev]['name'] if output_dev is not None else devices[sd.default.device[1]]['name']
            
            print(f"\n{'='*60}")
            print(f"RVC Virtual Microphone")
            print(f"{'='*60}")
            print(f"Input (Real Mic):    {input_name}")
            print(f"Output (Virtual):    {output_name}")
            print(f"Sample Rate:         {self.sample_rate}Hz")
            print(f"RVC Server:          {self.uri}")
            print(f"{'='*60}")
        
        print(f"\nConnecting to RVC server...")
        
        try:
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                
                await websocket.send(json.dumps({"type": "ping"}))
                response = await websocket.recv()
                print(f"Connected! Server: {response}")
                
                self.running = True
                
                if use_network:
                    print(f"\n✓ Network audio bridge active!")
                    print(f"  Audio will be sent to Windows receiver → VB-Cable → Discord")
                else:
                    print(f"\n✓ Virtual microphone active!")
                    print(f"  In Discord/Zoom/etc, select '{output_name}' or its monitor as your microphone")
                
                print(f"\nSpeak into your microphone. Press Ctrl+C to stop.\n")
                
                # Resolve input device for network mode
                if use_network:
                    input_dev = self.input_device
                    if isinstance(input_dev, str):
                        input_dev = find_device(input_dev, is_input=True)
                
                # Start input stream (always needed)
                # Output stream only needed for local playback
                if use_network:
                    with sd.InputStream(
                        device=input_dev,
                        samplerate=self.sample_rate,
                        blocksize=1024,
                        channels=1,
                        dtype=np.float32,
                        callback=self.input_callback
                    ):
                        process_task = asyncio.create_task(self.process_utterances())
                        recv_task = asyncio.create_task(self.receive_audio())
                        
                        try:
                            await asyncio.gather(process_task, recv_task)
                        except asyncio.CancelledError:
                            pass
                else:
                    with sd.InputStream(
                        device=input_dev,
                        samplerate=self.sample_rate,
                        blocksize=1024,
                        channels=1,
                        dtype=np.float32,
                        callback=self.input_callback
                    ), sd.OutputStream(
                        device=output_dev,
                        samplerate=self.sample_rate,
                        blocksize=1024,
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
            print("Make sure the server is running:")
            print("    python3 main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth --index ./assets/models/BillCipher/BillCipher.index")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.running = False
            if self.network_socket:
                self.network_socket.close()


def main():
    parser = argparse.ArgumentParser(
        description="Route microphone through RVC to virtual audio device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available audio devices
    python3 examples/virtual_mic_client.py --list-devices
    
    # Run with default devices
    python3 examples/virtual_mic_client.py
    
    # Run with specific virtual output (Linux)
    python3 examples/virtual_mic_client.py --output-device "RVC_Sink"
    
    # Run with VB-Cable (Windows)
    python3 examples/virtual_mic_client.py --output-device "CABLE Input"
    
    # WSL → Windows VB-Cable (Network Mode)
    # On Windows PowerShell: python windows_audio_receiver.py --output-device "CABLE Input"
    # In WSL: python3 examples/virtual_mic_client.py --network-output localhost

Setup:
    Linux (PulseAudio):
        pactl load-module module-null-sink sink_name=RVC_Sink
        pactl load-module module-virtual-source source_name=RVC_Mic master=RVC_Sink.monitor
        # Then select "RVC_Mic" in Discord
    
    Windows:
        1. Install VB-Cable from https://vb-audio.com/Cable/
        2. Use --output-device "CABLE Input"
        3. Select "CABLE Output" in Discord
    
    WSL to Windows (Network Bridge):
        1. Install VB-Cable on Windows
        2. Run on Windows: python windows_audio_receiver.py --output-device "CABLE Input"
        3. Run in WSL: python3 virtual_mic_client.py --network-output localhost
        4. Select "CABLE Output" in Discord
"""
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--host", default="localhost", help="RVC server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="RVC server port (default: 8765)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (default: 16000)")
    parser.add_argument("--input-device", default=None, help="Input device name or index (your microphone)")
    parser.add_argument("--output-device", default=None, help="Output device name or index (virtual mic sink)")
    parser.add_argument("--network-output", default=None, metavar="HOST[:PORT]",
                        help="Send audio to Windows receiver instead of local device (default port: 9999)")
    parser.add_argument("--silence-threshold", type=float, default=0.01, help="Silence detection threshold (default: 0.01)")
    parser.add_argument("--silence-duration", type=float, default=0.5, help="Silence duration to trigger processing (default: 0.5s)")
    
    args = parser.parse_args()
    
    if not HAS_SOUNDDEVICE:
        print("Error: sounddevice not installed")
        print("Install with: pip install sounddevice")
        sys.exit(1)
    
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)
    
    # Try to parse device as int, otherwise keep as string for name matching
    input_dev = args.input_device
    output_dev = args.output_device
    
    if input_dev is not None:
        try:
            input_dev = int(input_dev)
        except ValueError:
            pass
    
    if output_dev is not None:
        try:
            output_dev = int(output_dev)
        except ValueError:
            pass
    
    client = VirtualMicClient(
        host=args.host,
        port=args.port,
        sample_rate=args.sample_rate,
        input_device=input_dev,
        output_device=output_dev,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        network_output=args.network_output
    )
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
