#!/bin/bash

# =============================================================================
# MorphVox - Start Individual Services
# =============================================================================
# This script allows starting individual Python services for development
# without Docker, useful for debugging and faster iteration.
#
# Usage:
#   ./start-services.sh voice-engine    # Start voice engine
#   ./start-services.sh trainer         # Start trainer service
#   ./start-services.sh preprocess      # Start preprocessor service
#   ./start-services.sh all             # Start all Python services
#
# Prerequisites:
#   - Python 3.10+ with virtual environment
#   - CUDA-capable GPU (optional, for GPU acceleration)
#   - Redis and MinIO running (via Docker or locally)
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="$PROJECT_ROOT/services"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default environment variables
export S3_ENDPOINT=${S3_ENDPOINT:-http://localhost:9000}
export S3_ACCESS_KEY=${S3_ACCESS_KEY:-minioadmin}
export S3_SECRET_KEY=${S3_SECRET_KEY:-minioadmin}
export S3_BUCKET=${S3_BUCKET:-morphvox}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export DEVICE=${DEVICE:-cuda}

start_voice_engine() {
    log_info "Starting Voice Engine service..."
    cd "$SERVICES_DIR/voice-engine"
    
    # Check for virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        log_warn "No virtual environment found. Using system Python."
    fi
    
    export HTTP_PORT=8001
    export WS_PORT=8765
    export MODEL_PATH="$SERVICES_DIR/voice-engine/assets/models"
    export HUBERT_PATH="$SERVICES_DIR/voice-engine/assets/hubert/hubert_base.pt"
    export RMVPE_PATH="$SERVICES_DIR/voice-engine/assets/rmvpe"
    
    python main.py
}

start_trainer() {
    log_info "Starting Trainer service..."
    cd "$SERVICES_DIR/trainer"
    
    # Check for virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        log_warn "No virtual environment found. Using system Python."
    fi
    
    export TRAINER_PORT=8002
    export DATA_ROOT="$PROJECT_ROOT/data"
    export MODELS_ROOT="$SERVICES_DIR/voice-engine/assets/models"
    export RVC_ROOT="$SERVICES_DIR/voice-engine"
    export ASSETS_ROOT="$SERVICES_DIR/voice-engine/assets"
    export PREPROCESSOR_URL="http://localhost:8003"
    
    # Create data directory if it doesn't exist
    mkdir -p "$DATA_ROOT"
    
    python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
}

start_preprocess() {
    log_info "Starting Preprocessor service..."
    cd "$SERVICES_DIR/preprocessor"
    
    # Check for virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        log_warn "No virtual environment found. Using system Python."
    fi
    
    export PREPROCESS_PORT=8003
    export DATA_ROOT="$PROJECT_ROOT/data"
    export ASSETS_ROOT="$SERVICES_DIR/voice-engine/assets"
    
    # Create data directory if it doesn't exist
    mkdir -p "$DATA_ROOT"
    
    python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
}

show_usage() {
    echo ""
    echo "Usage: $0 <service>"
    echo ""
    echo "Services:"
    echo "  voice-engine   Start the voice engine (inference + WebSocket)"
    echo "  trainer        Start the trainer service"
    echo "  preprocess     Start the preprocessor service"
    echo "  all            Start all Python services (in separate terminals)"
    echo ""
    echo "Environment variables:"
    echo "  DEVICE         Compute device (default: cuda)"
    echo "  LOG_LEVEL      Logging level (default: INFO)"
    echo "  S3_ENDPOINT    MinIO/S3 endpoint (default: http://localhost:9000)"
    echo ""
}

case "${1:-}" in
    voice-engine)
        start_voice_engine
        ;;
    trainer)
        start_trainer
        ;;
    preprocess)
        start_preprocess
        ;;
    all)
        log_info "Starting all services in separate terminals..."
        log_warn "Note: Each service will run in its own process."
        
        # Start each service in background with output to separate log files
        mkdir -p "$PROJECT_ROOT/logs"
        
        log_info "Starting preprocessor..."
        (start_preprocess > "$PROJECT_ROOT/logs/preprocess.log" 2>&1) &
        sleep 2
        
        log_info "Starting trainer..."
        (start_trainer > "$PROJECT_ROOT/logs/trainer.log" 2>&1) &
        sleep 2
        
        log_info "Starting voice engine..."
        (start_voice_engine > "$PROJECT_ROOT/logs/voice-engine.log" 2>&1) &
        
        echo ""
        log_success "All services started in background!"
        echo ""
        echo "Logs are available at:"
        echo "  $PROJECT_ROOT/logs/preprocess.log"
        echo "  $PROJECT_ROOT/logs/trainer.log"
        echo "  $PROJECT_ROOT/logs/voice-engine.log"
        echo ""
        echo "Use 'tail -f \$log_file' to follow logs."
        echo "Use 'pkill -f uvicorn' to stop all services."
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
