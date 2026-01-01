#!/usr/bin/env python3
"""
File-based WebSocket Client for RVC Voice Conversion

Sends an audio file to the RVC server and saves the converted result.

Usage:
    python examples/file_client.py input.wav output.wav
    python examples/file_client.py input.mp3 output.wav --host localhost --port 8765
"""

import asyncio
import websockets
import json
import numpy as np
import base64
import argparse
import sys
from pathlib import Path

# Try to import audio libraries
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    if HAS_LIBROSA:
        # librosa handles many formats and resampling
        audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return audio.astype(np.float32), sr
    elif HAS_SOUNDFILE:
        audio, sr = sf.read(str(path), dtype='float32')
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        # Basic resampling if needed
        if sr != target_sr:
            from scipy import signal
            from math import gcd
            g = gcd(sr, target_sr)
            audio = signal.resample_poly(audio, target_sr // g, sr // g)
        return audio.astype(np.float32), target_sr
    else:
        raise ImportError("Please install soundfile or librosa: pip install soundfile librosa")


def save_audio(path: str, audio: np.ndarray, sr: int = 16000):
    """Save audio to file."""
    if not HAS_SOUNDFILE:
        raise ImportError("Please install soundfile: pip install soundfile")
    
    # Ensure float32 in range [-1, 1]
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    sf.write(path, audio, sr)


async def convert_file(input_path: str, output_path: str, host: str = "localhost", port: int = 8765):
    """Send audio file to RVC server and save converted result."""
    
    print(f"Loading: {input_path}")
    audio, sr = load_audio(input_path, target_sr=16000)
    print(f"  Loaded {len(audio)} samples ({len(audio)/sr:.2f}s) at {sr}Hz")
    
    uri = f"ws://{host}:{port}"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected!")
        
        # Encode audio to base64
        audio_bytes = audio.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Send audio for processing
        message = {
            "type": "audio",
            "data": audio_b64,
            "final": True  # Indicates this is complete audio, not a chunk
        }
        
        print(f"Sending audio for conversion...")
        await websocket.send(json.dumps(message))
        
        # Receive processed audio
        print("Waiting for response...")
        response = await websocket.recv()
        response_data = json.loads(response)
        
        if response_data.get('type') == 'audio':
            # Decode processed audio
            processed_b64 = response_data.get('data')
            processed_bytes = base64.b64decode(processed_b64)
            processed_audio = np.frombuffer(processed_bytes, dtype=np.float32)
            
            print(f"Received {len(processed_audio)} samples ({len(processed_audio)/sr:.2f}s)")
            
            # Save output
            save_audio(output_path, processed_audio, sr)
            print(f"Saved: {output_path}")
            print("Done!")
            
        elif response_data.get('type') == 'error':
            print(f"Error from server: {response_data.get('message')}")
            sys.exit(1)
        else:
            print(f"Unexpected response: {response_data}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio file using RVC WebSocket server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/file_client.py input.wav output.wav
    python examples/file_client.py song.mp3 converted.wav --host 192.168.1.100
    
Make sure the server is running:
    python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth
"""
    )
    parser.add_argument("input", help="Input audio file (wav, mp3, flac, etc.)")
    parser.add_argument("output", help="Output audio file (wav)")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(convert_file(args.input, args.output, args.host, args.port))
    except ConnectionRefusedError:
        print(f"\nError: Could not connect to server at ws://{args.host}:{args.port}")
        print("Make sure the server is running:")
        print("    python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(0)


if __name__ == "__main__":
    main()
