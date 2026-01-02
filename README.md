# RVC Real-Time Voice Conversion

Real-time voice conversion application using **RVC (Retrieval-based Voice Conversion)** models, compatible with **WebUI-trained `.pth` models** (v1/v2) + optional **retrieval `.index`**. Supports:

- **local**: process audio files
- **api**: WebSocket + TCP socket servers for remote clients
- **streaming**: real microphone/speaker loopback via PyAudio (requires a working system audio device)

> **Important:** `streaming` mode requires a real, working audio input/output device. In WSL/headless/Docker without audio passthrough, it will fail with ALSA/PyAudio errors.

---

## Features

- 🎤 **Real-time Audio Processing** (PyAudio I/O in `streaming`)
- 🔄 **Chunk-based Processing** with configurable chunk size
- 🎯 **WebUI model compatibility** (`.pth` + `config.json`, v1/v2)
- 🧠 **HuBERT feature extraction** (required)
- 🎼 **RMVPE pitch extraction** (recommended; required if `F0_METHOD=rmvpe`)
- 🧲 **Retrieval index support** (`.index`) with blend control (`INDEX_RATE`)
- 🌐 **Remote conversion** via WebSocket or raw TCP socket server
- ⚡ **GPU support** (CUDA auto-detect; inference runs on GPU when available)

---

## Directory Structure

```

rvc_real_time/
├── app/
│   ├── audio_stream.py
│   ├── model_manager.py
│   ├── chunk_processor.py
│   ├── streaming_api.py
│   └── config.py
├── rvc/                        # vendored RVC pipeline + models
├── assets/
│   ├── models/                 # .pth model folders (often include config.json, optional .index)
│   ├── hubert/                 # hubert_base.pt
│   └── rmvpe/                  # rmvpe.pt
├── requirements.txt
├── Dockerfile
├── main.py
└── README.md
```
---

## Installation

### Local Installation

1) Clone:

```bash
git clone https://github.com/alli959/rvc_real_time.git
cd rvc_real_time
````

2. Create venv + install deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Provide required assets:

#### Required: HuBERT

Place:

```
assets/hubert/hubert_base.pt
```

or set `HUBERT_PATH` to a file path.

#### Required for RMVPE pitch (`rmvpe`)

Place:

```
assets/rmvpe/rmvpe.pt
```

or set `RMVPE_DIR` to the directory containing `rmvpe.pt`.

#### Model files

Put your model folder under `assets/models/`, for example:

```
assets/models/BillCipher/BillCipher.pth
assets/models/BillCipher/config.json
assets/models/BillCipher/BillCipher.index
```

> Tip: If your model folder includes `config.json`, the repo can auto-build the synthesizer config from it.

---

## Quick Start

### Local mode (recommended for first run)

```bash
python main.py --mode local \
  --model ./assets/models/BillCipher/BillCipher.pth \
  --index ./assets/models/BillCipher/BillCipher.index \
  --input ./input/input.flac \
  --output ./outputs/output.wav \
  --f0-method rmvpe \
  --f0-up-key 0 \
  --index-rate 0.75 \
  --chunk-size 65536
```

### API mode (WebSocket + TCP socket)

```bash
python main.py --mode api \
  --model ./assets/models/BillCipher/BillCipher.pth \
  --index ./assets/models/BillCipher/BillCipher.index
```

* WebSocket: `ws://localhost:8765`
* TCP socket: `tcp://localhost:9876`

> If you see `426 Upgrade Required` / `invalid Connection header: keep-alive`, something is making a normal HTTP request to the WebSocket port. Use a real WebSocket client (JS WebSocket, websocat, etc.).

### 🎤 Virtual Microphone (Use with Discord/Zoom)

Change your voice in real-time for Discord, Zoom, or any other application:

```bash
# 1. Setup virtual audio (Linux - run once after reboot)
./examples/setup_virtual_mic.sh

# 2. Start RVC server
python3 main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth --index ./assets/models/BillCipher/BillCipher.index

# 3. Start virtual mic client (in another terminal)
python3 examples/virtual_mic_client.py --output-device RVC_Sink

# 4. In Discord: Settings → Voice → Input Device → Select "RVC_Mic"
```

See [examples/README.md](examples/README.md) for Windows/macOS setup instructions.

### Streaming mode (real mic/speaker loopback)

```bash
python main.py --mode streaming \
  --model ./assets/models/BillCipher/BillCipher.pth \
  --index ./assets/models/BillCipher/BillCipher.index \
  --f0-method rmvpe \
  --index-rate 0.75 \
  --chunk-size 65536
```

> **WSL/headless warning:** If you get ALSA errors like `cannot find card '0'` or PyAudio `Wait timed out`, your environment doesn’t expose a usable audio device. Run streaming mode on a machine with real audio I/O (native Linux/Windows/macOS), or configure audio passthrough.

---

## Command Line Options

```
--mode {streaming,api,local}     Application mode (default: api)

--model MODEL                    Model file to load (.pth)
--index INDEX                    Optional retrieval .index file

--input INPUT                    Input audio file (local mode only)
--output OUTPUT                  Output audio file (local mode only)

--f0-method METHOD               F0 method (e.g. rmvpe, dio, harvest)
--f0-up-key N                    Pitch shift in semitones
--index-rate R                   Index blend (0..1)
--protect P                      Protect (0..1)
--rms-mix-rate R                 RMS mix rate (0..1)
--filter-radius N                Filter radius
--resample-sr SR                 Output resample SR (0=auto/keep)

--chunk-size N                   Chunk size for processing
--output-gain G                  Output gain multiplier

--log-level {DEBUG,INFO,WARNING,ERROR}
--websocket-port PORT            WebSocket server port (default: 8765)
--socket-port PORT               Socket server port (default: 9876)
```

---

## Configuration (.env)

You can configure defaults via environment variables (see `.env.example`):

```bash
# Audio
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024
AUDIO_OVERLAP=0
AUDIO_CHANNELS=1

# Assets
MODEL_DIR=assets/models
INDEX_DIR=assets/index
HUBERT_PATH=assets/hubert/hubert_base.pt
RMVPE_DIR=assets/rmvpe

# Defaults
DEFAULT_MODEL=
DEFAULT_INDEX=

# Inference defaults
F0_METHOD=rmvpe
F0_UP_KEY=0
INDEX_RATE=0.75
FILTER_RADIUS=3
RMS_MIX_RATE=0.25
PROTECT=0.33
RESAMPLE_SR=0

# Device
DEVICE=auto   # auto, cpu, cuda

# Server
WEBSOCKET_HOST=0.0.0.0
WEBSOCKET_PORT=8765
SOCKET_HOST=0.0.0.0
SOCKET_PORT=9876

# App
APP_MODE=api
LOG_LEVEL=INFO
```

---

## API Notes

### WebSocket

The WebSocket server expects a **real WebSocket handshake**. If you hit it with curl/browser HTTP directly, you’ll see handshake errors.

### TCP Socket

TCP is a stream (not message framed). Clients must send audio in a consistent format and chunking strategy. If you see:

* `buffer size must be a multiple of element size`

…it means the server is trying to decode bytes into `float32` / `int16` but the received byte length is not aligned (TCP packet split). Clients should either:

* frame messages (length-prefix), or
* buffer until full frames are received before decoding.

---

## GPU vs CPU

If CUDA is available, the model should run on GPU for speed. Logs like:

```
Found GPU ... is_half:True, device:cuda:0
```

mean inference is configured for GPU. Some checkpoints are loaded with `map_location="cpu"` and then moved to GPU after weights are loaded; that’s normal.

---

## Troubleshooting

### Streaming mode fails with ALSA / PyAudio errors

Examples:

* `ALSA ... cannot find card '0'`
* `OSError: [Errno -9987] Wait timed out`

Cause: no accessible audio device (common in WSL, Docker, headless servers).

Fix:

* run streaming mode on a system with real audio I/O, or
* configure audio passthrough (PulseAudio/PipeWire/WSLg), or
* use `--mode local` / `--mode api` instead.

### WebSocket `426 Upgrade Required` / `invalid Connection header`

Cause: non-WebSocket HTTP client hitting the WS port.
Fix: use a WebSocket client (JS WebSocket, websocat).

### Output quality is bad

Common causes:

* wrong sample rate expectations / resampling issues
* chunk size too small/large for your use case
* retrieval index mismatch (wrong `.index`)
* wrong f0 method or protect/rms settings

Try:

* `--f0-method rmvpe`
* tune `--index-rate` (e.g. 0.5–0.8)
* tune `--protect` (0.2–0.5)
* ensure your `.index` actually matches the `.pth` model

---

## Docker

Docker can run `api` or `local` mode easily, but `streaming` mode typically needs audio passthrough.

Build:

```bash
docker build -t rvc-real-time:latest .
```

Run API:

```bash
docker run -p 8765:8765 -p 9876:9876 rvc-real-time:latest
```

---

## License

MIT

```
::contentReference[oaicite:0]{index=0}
```
