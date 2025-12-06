# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY main.py .

# Create directories for assets
RUN mkdir -p assets/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_MODE=api
ENV WEBSOCKET_PORT=8765
ENV SOCKET_PORT=9876

# Expose ports
EXPOSE 8765 9876

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8765)); s.close()"

# Run the application
CMD ["python", "main.py", "--mode", "api"]
