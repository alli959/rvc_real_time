# Example Scripts

This directory contains example scripts demonstrating how to use the RVC Real-time Voice Conversion application.

## Available Examples

### 1. Virtual Microphone Client (`virtual_mic_client.py`) ⭐ Recommended

Routes your real microphone through RVC and outputs to a **virtual microphone** that Discord, Zoom, or any other app can use as input.

```
Your Voice → RVC Processing → Virtual Mic → Discord/Zoom/etc.
```

**Linux Setup (PulseAudio/PipeWire):**
```bash
# Create virtual audio devices (run once after each reboot)
./examples/setup_virtual_mic.sh

# Start RVC server
python3 main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth --index ./assets/models/BillCipher/BillCipher.index

# Start virtual mic (in another terminal)
python3 examples/virtual_mic_client.py --output-device RVC_Sink

# In Discord: Settings → Voice → Input Device → Select "RVC_Mic"
```

**Windows Setup (VB-Cable):**
```bash
# 1. Install VB-Cable from https://vb-audio.com/Cable/

# 2. Start RVC server
python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth

# 3. Start virtual mic
python examples/virtual_mic_client.py --output-device "CABLE Input"

# 4. In Discord: Settings → Voice → Input Device → Select "CABLE Output"
```

**List available devices:**
```bash
python3 examples/virtual_mic_client.py --list-devices
```

### 2. File Client (`file_client.py`)

Convert audio files through the API:

```bash
# Start the server
python3 main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth

# Convert a file
python3 examples/file_client.py input.wav output.wav
```

### 3. Real-time Microphone Client (`realtime_mic_client.py`)

Captures mic audio, processes through RVC, and plays back through speakers (for testing):

```bash
python3 main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth
python3 examples/realtime_mic_client.py
```

### 4. WebSocket Client (`websocket_client.py`)

Low-level WebSocket client example:

```bash
python3 main.py --mode api
python3 examples/websocket_client.py
```

### 5. Socket Client (`socket_client.py`)

TCP socket client example:

```bash
python3 main.py --mode api
python3 examples/socket_client.py
```

### 6. Windows Full Client (`windows_full_client.py`)

Complete Windows client with GUI for voice conversion:

```bash
# On Windows:
python examples/windows_full_client.py
```

### 7. Windows Audio Receiver (`windows_audio_receiver.py`)

Receives audio from remote RVC server and plays through virtual audio device:

```bash
# On Windows (requires VB-Cable):
python examples/windows_audio_receiver.py --output-device "CABLE Input"
```

## Requirements

```bash
pip install sounddevice numpy websockets
```

## Virtual Microphone Setup

### Linux (PulseAudio/PipeWire)

Run the setup script to create virtual audio devices:
```bash
./examples/setup_virtual_mic.sh
```

Or manually:
```bash
pactl load-module module-null-sink sink_name=RVC_Sink sink_properties=device.description="RVC_Output"
pactl load-module module-virtual-source source_name=RVC_Mic master=RVC_Sink.monitor source_properties=device.description="RVC_Microphone"
```

### Windows

1. Download and install [VB-Cable](https://vb-audio.com/Cable/)
2. Use `--output-device "CABLE Input"` in the client
3. Select "CABLE Output" as your microphone in Discord/Zoom

### macOS

1. Install [BlackHole](https://existential.audio/blackhole/) or [Soundflower](https://github.com/mattingalls/Soundflower)
2. Use `--output-device "BlackHole"` in the client
3. Select "BlackHole" as your microphone in Discord/Zoom
