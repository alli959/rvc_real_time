# RVC Real-Time Voice Conversion

Real-time voice conversion application using RVC (Retrieval-based Voice Conversion) models. This modular Python codebase supports streaming audio input/output, efficient feature extraction, and dynamic model selection. Refactored for chunk-based processing with a streaming API (WebSocket/socket server). Containerized for easy deployment.

## Features

- 🎤 **Real-time Audio Processing**: Stream audio input/output with minimal latency
- 🔄 **Chunk-based Processing**: Efficient processing with configurable chunk size and overlap
- 🎯 **Dynamic Model Selection**: Load and switch between RVC models on the fly
- 🌐 **Streaming API**: WebSocket and TCP socket servers for remote clients
- 🐳 **Containerized**: Docker support for easy deployment
- ⚡ **Fast & Extensible**: Modular architecture for future development
- 🎛️ **Multiple Modes**: Streaming, API server, and local file processing

## Directory Structure

```
rvc_real_time/
├── app/                        # Core application modules
│   ├── __init__.py
│   ├── audio_stream.py        # Audio I/O streaming
│   ├── feature_extraction.py  # Feature extraction for RVC
│   ├── model_manager.py       # Model loading and inference
│   ├── chunk_processor.py     # Chunk-based processing
│   ├── streaming_api.py       # WebSocket/Socket servers
│   └── config.py              # Configuration management
├── assets/                     # Asset files
│   └── models/                # RVC model files (.pth, .pt, .ckpt)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── main.py                    # Application entry point
└── README.md                  # This file
```

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/alli959/rvc_real_time.git
cd rvc_real_time
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Place your RVC models in `assets/models/`

To run inference with WebUI-trained models you also need these assets:

- **HuBERT**: place `hubert_base.pt` at `assets/hubert/hubert_base.pt`
- **RMVPE** (for f0/pitch): place `rmvpe.pt` at `assets/rmvpe/rmvpe.pt`
- **Index** (optional but recommended): place the model's `.index` file in `assets/index/`

You can configure these paths in `.env` (see `.env.example`).

### Docker Installation

1. Build the Docker image:
```bash
docker build -t rvc-real-time .
```

2. Run the container:
```bash
docker run -p 8765:8765 -p 9876:9876 rvc-real-time
```

## Usage

### API Mode (Default)

Start WebSocket and Socket servers for remote clients:

```bash
python main.py --mode api
```

**WebSocket Server**: `ws://localhost:8765`
**Socket Server**: `tcp://localhost:9876`

### Streaming Mode

Real-time audio I/O with local processing:

```bash
python main.py --mode streaming --model your_model.pth --index assets/index/your_model.index
```

### Local File Processing

Process audio files:

```bash
python main.py --mode local --input input.wav --output output.wav --model your_model.pth
```

### Command Line Options

```
--mode {streaming,api,local}  Application mode (default: api)
--model MODEL                 Model file to load
--index INDEX                 Optional .index file
--input INPUT                 Input audio file (local mode)
--output OUTPUT               Output audio file (local mode)
--log-level LEVEL            Logging level (DEBUG, INFO, WARNING, ERROR)
--websocket-port PORT        WebSocket server port (default: 8765)
--socket-port PORT           Socket server port (default: 9876)
```

## Configuration

Configuration can be set via environment variables or command line arguments:

### Environment Variables

```bash
# Audio Configuration
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024
AUDIO_OVERLAP=0
AUDIO_CHANNELS=1

# Model/asset paths
MODEL_DIR=assets/models
INDEX_DIR=assets/index
HUBERT_PATH=assets/hubert/hubert_base.pt
RMVPE_DIR=assets/rmvpe

# Defaults
DEFAULT_MODEL=your_model.pth
# DEFAULT_INDEX=assets/index/your_model.index

# RVC inference defaults
F0_METHOD=rmvpe
F0_UP_KEY=0
INDEX_RATE=0.75
FILTER_RADIUS=3
RMS_MIX_RATE=0.25
PROTECT=0.33
RESAMPLE_SR=16000

# Device
DEVICE=auto  # auto, cpu, cuda

# Server Configuration
WEBSOCKET_HOST=0.0.0.0
WEBSOCKET_PORT=8765
SOCKET_HOST=0.0.0.0
SOCKET_PORT=9876

# Application
APP_MODE=api
LOG_LEVEL=INFO
```

## API Documentation

### WebSocket API

Connect to `ws://host:8765` and send/receive JSON messages:

**Audio Processing Request**:
```json
{
  "type": "audio",
  "data": "<base64-encoded-float32-audio>"
}
```

**Audio Processing Response**:
```json
{
  "type": "audio",
  "data": "<base64-encoded-float32-audio>"
}
```

**Configuration Update**:
```json
{
  "type": "config",
  "sample_rate": 16000,
  "chunk_size": 1024
}
```

**Ping/Pong**:
```json
{
  "type": "ping"
}
```

### Socket API

Connect to `tcp://host:9876` and send/receive raw audio data:

1. Send 4 bytes (big-endian) with chunk size
2. Send audio data as float32 bytes
3. Receive 4 bytes with processed chunk size
4. Receive processed audio as float32 bytes

## Architecture

### Core Modules

- **audio_stream.py**: Manages real-time audio I/O using PyAudio with queue-based buffering
- **feature_extraction.py**: Extracts mel spectrograms, pitch (F0), and MFCCs for RVC models
- **model_manager.py**: Handles model loading, inference, and GPU/CPU management
- **chunk_processor.py**: Processes audio in chunks with overlap and crossfading
- **streaming_api.py**: WebSocket and TCP socket servers for network streaming
- **config.py**: Configuration management with environment variable support

### Processing Pipeline

```
Input Audio → AudioStream → ChunkProcessor → FeatureExtractor → ModelManager → Output Audio
```

## Development

### Adding New Features

The modular architecture makes it easy to extend:

1. Add new feature extractors in `feature_extraction.py`
2. Add new model types in `model_manager.py`
3. Add new streaming protocols in `streaming_api.py`
4. Add new processing modes in `main.py`

### Testing

Run the application in different modes to test:

```bash
# Test API mode
python main.py --mode api --log-level DEBUG

# Test with a model
python main.py --mode streaming --model test_model.pth
```

## Docker Deployment

### Build and Run

```bash
# Build
docker build -t rvc-real-time:latest .

# Run with volume mount for models
docker run -d \
  -p 8765:8765 \
  -p 9876:9876 \
  -v $(pwd)/assets/models:/app/assets/models \
  --name rvc-server \
  rvc-real-time:latest

# View logs
docker logs -f rvc-server
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  rvc-server:
    build: .
    ports:
      - "8765:8765"
      - "9876:9876"
    volumes:
      - ./assets/models:/app/assets/models
    environment:
      - LOG_LEVEL=INFO
      - DEFAULT_MODEL=your_model.pth
```

Run with: `docker-compose up`

## Performance

- **Latency**: Configurable based on chunk size (default ~64ms)
- **Throughput**: Optimized for real-time processing
- **GPU Support**: Automatic CUDA detection and usage
- **Memory**: Efficient chunk-based processing minimizes memory usage

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Future Development

- [ ] Support for additional RVC model architectures
- [ ] Web UI for model management
- [ ] Multiple concurrent client support
- [ ] Audio effects and post-processing
- [ ] Performance metrics and monitoring
- [ ] Model fine-tuning interface

## Troubleshooting

### Docker Build Issues with pip/wheel

If you encounter errors like "Failed to build installable wheels for some pyproject.toml based projects":

**Solution 1: Use the updated Dockerfile**
The Dockerfile now includes build tools and upgraded pip/setuptools:
```bash
docker build -t rvc-real-time:latest .
```

**Solution 2: Build with more resources**
```bash
docker build --memory=4g --memory-swap=8g -t rvc-real-time:latest .
```

**Solution 3: Use pre-built images**
Consider using the CPU-only PyTorch version which has better pre-built wheels:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### PyAudio Installation Issues

On Linux:
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

On macOS:
```bash
brew install portaudio
pip install pyaudio
```

### Installation from requirements.txt fails

If `pip install -r requirements.txt` fails, try:

1. **Upgrade pip and setuptools first:**
```bash
pip install --upgrade pip setuptools wheel
```

2. **Install packages one by one:**
```bash
pip install numpy
pip install torch torchaudio
pip install librosa soundfile
pip install websockets aiohttp python-dotenv
pip install pyaudio  # May need system dependencies
```

3. **Use the minimal requirements:**
```bash
pip install -r requirements-minimal.txt
```

Then install heavy dependencies separately as needed.

### CUDA/GPU Issues

Ensure you have the correct PyTorch version for your CUDA version:
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Contact

For issues and questions, please use the GitHub issue tracker.