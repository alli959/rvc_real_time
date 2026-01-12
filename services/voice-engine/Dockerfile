
# Use NVIDIA CUDA base image with Python 3.10 and cuDNN for GPU support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-distutils python3-pip python3.10-venv \
    software-properties-common \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    git \
    gcc \
    g++ \
    espeak-ng \
    libespeak-ng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .



# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (matching CUDA version)
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122

# Install Cython before requirements to fix pyworld build
RUN pip install --no-cache-dir cython

# Install faiss-gpu separately with increased timeout and retries
RUN pip install --default-timeout=300 --no-cache-dir faiss-gpu>=1.7.2 || \
    pip install --default-timeout=300 --no-cache-dir faiss-gpu>=1.7.2 || \
    pip install --default-timeout=300 --no-cache-dir faiss-gpu>=1.7.2

# Install Python dependencies (excluding faiss since we installed it above)
RUN pip install --default-timeout=300 --no-cache-dir --prefer-binary -r requirements.txt

# Install fairseq from git without dependencies to avoid conflicts
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/fairseq.git@v0.12.2 --no-deps

# Install fairseq's required dependencies that don't conflict
RUN pip install --no-cache-dir hydra-core omegaconf cffi bitarray sacrebleu

# Install Bark TTS for native emotion/sound effect support (optional, requires GPU)
# This adds about 2GB to the image size but enables superior emotion handling
# Comment out if you want to use Edge TTS only (smaller image, simulated emotions)
ARG INSTALL_BARK=true
RUN if [ "$INSTALL_BARK" = "true" ]; then \
    pip install --no-cache-dir --default-timeout=300 \
        git+https://github.com/suno-ai/bark.git; \
    fi

# Copy and run the fairseq patching script
COPY patch_fairseq.py .
RUN python3 patch_fairseq.py && rm patch_fairseq.py

# Copy application code
COPY app/ ./app/
COPY main.py .

# Copy RVC module (required for voice conversion)
COPY rvc/ ./rvc/

# Copy infer module (required for audio inference)
COPY infer/ ./infer/

# Create directories for assets
RUN mkdir -p assets/models assets/hubert assets/rmvpe assets/bark

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_MODE=api
ENV WEBSOCKET_PORT=8765
ENV SOCKET_PORT=9876
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8765 9876 8000

# Health check - uses TCP socket server port instead of WebSocket
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 9876)); s.close()"

# Run the application
CMD ["python", "main.py", "--mode", "api"]
