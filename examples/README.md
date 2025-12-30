# Example Scripts

This directory contains example scripts demonstrating how to use the RVC Real-time Voice Conversion application.

## Available Examples

### 1. WebSocket Client (`websocket_client.py`)

Demonstrates how to connect to the WebSocket server and process audio:

```bash
# Start the server first
python main.py --mode api

# In another terminal, run the example
python examples/websocket_client.py
```

### 2. Socket Client (`socket_client.py`)

Demonstrates how to connect to the TCP socket server and process audio:

```bash
# Start the server first
python main.py --mode api

# In another terminal, run the example
python examples/socket_client.py
```

## Requirements

The examples require the same dependencies as the main application:
- numpy
- websockets (for WebSocket client)

## Customization

You can modify these examples to:
- Process real audio from files or microphone
- Adjust audio parameters (sample rate, chunk size)
- Handle continuous streaming
- Add error handling and reconnection logic
