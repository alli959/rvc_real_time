"""
Socket Client Example for RVC Real-time Voice Conversion

This example demonstrates how to connect to the RVC TCP socket server
and send/receive raw audio data for real-time voice conversion.
"""

import socket
import numpy as np


def socket_client_example():
    """Example TCP socket client"""
    host = "localhost"
    port = 9876
    
    print(f"Connecting to {host}:{port}...")
    
    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    
    print("Connected!")
    
    try:
        # Generate test audio (1 second of 440 Hz sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Convert to bytes
        audio_bytes = audio.tobytes()
        
        # Send chunk size (4 bytes, big-endian)
        size_bytes = len(audio_bytes).to_bytes(4, byteorder='big')
        sock.send(size_bytes)
        
        # Send audio data
        print(f"Sending audio data ({len(audio)} samples)...")
        sock.send(audio_bytes)
        
        # Receive processed chunk size
        size_data = sock.recv(4)
        processed_size = int.from_bytes(size_data, byteorder='big')
        
        # Receive processed audio
        processed_bytes = sock.recv(processed_size)
        processed_audio = np.frombuffer(processed_bytes, dtype=np.float32)
        
        print(f"Received processed audio ({len(processed_audio)} samples)")
        print("Audio processing successful!")
    
    finally:
        sock.close()
        print("Connection closed")


if __name__ == "__main__":
    print("RVC Socket Client Example")
    print("==========================")
    print("\nMake sure the RVC server is running:")
    print("  python main.py --mode api")
    print()
    
    try:
        socket_client_example()
    except ConnectionRefusedError:
        print("\nError: Could not connect to server.")
        print("Make sure the server is running on localhost:9876")
    except KeyboardInterrupt:
        print("\nStopped by user")
