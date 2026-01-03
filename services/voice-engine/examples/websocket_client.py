"""
WebSocket Client Example for RVC Real-time Voice Conversion

This example demonstrates how to connect to the RVC WebSocket server
and send/receive audio data for real-time voice conversion.
"""

import asyncio
import websockets
import json
import numpy as np
import base64


async def websocket_client_example():
    """Example WebSocket client"""
    uri = "ws://localhost:8765"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected!")
        
        # Send a ping
        await websocket.send(json.dumps({"type": "ping"}))
        response = await websocket.recv()
        print(f"Ping response: {response}")
        
        # Generate some test audio (1 second of 440 Hz sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Encode audio to base64
        audio_bytes = audio.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Send audio for processing
        message = {
            "type": "audio",
            "data": audio_b64
        }
        
        print(f"Sending audio data ({len(audio)} samples)...")
        await websocket.send(json.dumps(message))
        
        # Receive processed audio
        response = await websocket.recv()
        response_data = json.loads(response)
        
        if response_data.get('type') == 'audio':
            # Decode processed audio
            processed_b64 = response_data.get('data')
            processed_bytes = base64.b64decode(processed_b64)
            processed_audio = np.frombuffer(processed_bytes, dtype=np.float32)
            
            print(f"Received processed audio ({len(processed_audio)} samples)")
            print("Audio processing successful!")
        else:
            print(f"Unexpected response: {response_data}")


if __name__ == "__main__":
    print("RVC WebSocket Client Example")
    print("=============================")
    print("\nMake sure the RVC server is running:")
    print("  python main.py --mode api")
    print()
    
    try:
        asyncio.run(websocket_client_example())
    except ConnectionRefusedError:
        print("\nError: Could not connect to server.")
        print("Make sure the server is running on ws://localhost:8765")
    except KeyboardInterrupt:
        print("\nStopped by user")
