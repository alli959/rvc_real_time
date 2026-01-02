#!/usr/bin/env python3
"""
Windows Audio Receiver for RVC Virtual Microphone

This script runs on Windows and receives processed audio from WSL,
then outputs it to VB-Cable (or any other audio device).

Run this in Windows PowerShell (not WSL):
    python windows_audio_receiver.py --list-devices
    python windows_audio_receiver.py --output-device "CABLE Input"

Then run virtual_mic_client.py in WSL with --network-output
"""

import argparse
import socket
import struct
import sys
import threading
import time

try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install sounddevice numpy")
    sys.exit(1)


SAMPLE_RATE = 16000
CHUNK_SIZE = 1024


def list_devices():
    """List available audio devices"""
    print("\nAvailable Audio Devices:")
    print("=" * 60)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        direction = []
        if dev['max_input_channels'] > 0:
            direction.append("IN")
        if dev['max_output_channels'] > 0:
            direction.append("OUT")
        print(f"  {i}: {dev['name']} [{'/'.join(direction)}]")
    print("=" * 60)
    print("\nFor VB-Cable, look for 'CABLE Input (VB-Audio Virtual Cable)'")
    print("Use that device name with --output-device")


def find_device(name):
    """Find device by name (partial match)"""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if name.lower() in dev['name'].lower():
            return i, dev['name']
    return None, None


class AudioReceiver:
    def __init__(self, output_device, port=9999):
        self.output_device = output_device
        self.port = port
        self.running = False
        self.output_stream = None
        self.playback_buffer = []
        self.buffer_lock = threading.Lock()
        
    def start(self):
        """Start the audio receiver server"""
        self.running = True
        
        # Set up output stream
        device_idx, device_name = find_device(self.output_device) if self.output_device else (None, "default")
        
        if self.output_device and device_idx is None:
            print(f"Error: Output device '{self.output_device}' not found")
            list_devices()
            return False
            
        print(f"\n{'=' * 60}")
        print("RVC Windows Audio Receiver")
        print(f"{'=' * 60}")
        print(f"Output Device:  {device_name or 'default'}")
        print(f"Listen Port:    {self.port}")
        print(f"Sample Rate:    {SAMPLE_RATE}Hz")
        print(f"{'=' * 60}")
        
        # Create output stream
        self.output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            device=device_idx,
            blocksize=CHUNK_SIZE,
            callback=self._playback_callback
        )
        self.output_stream.start()
        
        # Start server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(1)
        
        print(f"\n✓ Listening for connections on port {self.port}")
        print(f"  In WSL, run: python virtual_mic_client.py --network-output")
        print(f"\nIn Discord/Zoom, select '{device_name or 'CABLE Output'}' as microphone")
        print("\nPress Ctrl+C to stop.\n")
        
        try:
            while self.running:
                server_socket.settimeout(1.0)
                try:
                    client_socket, addr = server_socket.accept()
                    print(f"Connection from {addr}")
                    self._handle_client(client_socket)
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False
            server_socket.close()
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
                
        return True
        
    def _playback_callback(self, outdata, frames, time_info, status):
        """Callback for audio playback"""
        with self.buffer_lock:
            if len(self.playback_buffer) >= frames:
                outdata[:, 0] = np.array(self.playback_buffer[:frames], dtype=np.float32)
                self.playback_buffer = self.playback_buffer[frames:]
            else:
                # Pad with zeros if not enough data
                available = len(self.playback_buffer)
                if available > 0:
                    outdata[:available, 0] = np.array(self.playback_buffer, dtype=np.float32)
                    self.playback_buffer = []
                outdata[available:, 0] = 0
                
    def _handle_client(self, client_socket):
        """Handle incoming audio data from a client"""
        try:
            while self.running:
                # Read header (4 bytes for length)
                header = b''
                while len(header) < 4:
                    chunk = client_socket.recv(4 - len(header))
                    if not chunk:
                        return
                    header += chunk
                    
                length = struct.unpack('>I', header)[0]
                
                # Read audio data
                data = b''
                while len(data) < length:
                    chunk = client_socket.recv(min(4096, length - len(data)))
                    if not chunk:
                        return
                    data += chunk
                    
                # Convert to audio samples
                audio = np.frombuffer(data, dtype=np.float32)
                
                # Add to playback buffer
                with self.buffer_lock:
                    self.playback_buffer.extend(audio.tolist())
                    
                print(f".", end="", flush=True)
                
        except Exception as e:
            print(f"\nClient error: {e}")
        finally:
            client_socket.close()
            print("\nClient disconnected")


def main():
    parser = argparse.ArgumentParser(description="Windows Audio Receiver for RVC")
    parser.add_argument('--output-device', '-o', help='Output device name (e.g., "CABLE Input")')
    parser.add_argument('--port', '-p', type=int, default=9999, help='Port to listen on (default: 9999)')
    parser.add_argument('--list-devices', '-l', action='store_true', help='List audio devices')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
        return
        
    receiver = AudioReceiver(args.output_device, args.port)
    receiver.start()


if __name__ == "__main__":
    main()
