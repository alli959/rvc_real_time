#!/bin/bash

# =============================================================================
# MorphVox - Start Individual Services (Local Development without Docker)
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
# NOTE: Path Configuration
# ========================
# - DATA_ROOT: Writable directory for preprocessing outputs + training data
#   Default: $PROJECT_ROOT/data
#   
# - MODELS_ROOT: Where final trained models are saved
#   Default: $PROJECT_ROOT/services/voice-engine/assets/models
#
# - ASSETS_ROOT: Shared assets (hubert, rmvpe, pretrained models)
#   Default: $PROJECT_ROOT/services/voice-engine/assets
#
# The preprocessor writes to DATA_ROOT/{exp_name}/, and the trainer reads
# from the same location after preprocessing is complete.
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="$PROJECT_ROOT/services"
ASSETS_DIR="$SERVICES_DIR/voice-engine/assets"
DATA_DIR="$PROJECT_ROOT/data"

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

# Create data directory if needed
mkdir -p "$DATA_DIR/uploads"

check_assets() {
    log_info "Checking required assets..."
    
    if [ ! -f "$ASSETS_DIR/hubert/hubert_base.pt" ]; then
        log_error "HuBERT model not found at $ASSETS_DIR/hubert/hubert_base.pt"
        log_error "Run: ./scripts/dev-up.sh --no-docker to download assets"
        return 1
    fi
    
    if [ ! -f "$ASSETS_DIR/rmvpe/rmvpe.pt" ]; then
        log_error "RMVPE model not found at $ASSETS_DIR/rmvpe/rmvpe.pt"
        log_error "Run: ./scripts/dev-up.sh --no-docker to download assets"
        return 1
    fi
    
    log_success "Required assets found"
}

start_voice_engine() {
    log_info "Starting Voice Engine service..."
    check_assets || exit 1
    
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
    export MODEL_PATH="$ASSETS_DIR/models"
    export HUBERT_PATH="$ASSETS_DIR/hubert/hubert_base.pt"
    export RMVPE_PATH="$ASSETS_DIR/rmvpe"
    
    python main.py
}

start_trainer() {
    log_info "Starting Trainer service..."
    check_assets || exit 1
    
    cd "$SERVICES_DIR/trainer"
    
    # Check for virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        log_warn "No virtual environment found. Using system Python."
    fi
    
    # UNIFIED PATHS - matching Docker compose configuration
    export TRAINER_PORT=8002
    export DATA_ROOT="$DATA_DIR"
    export MODELS_ROOT="$ASSETS_DIR/models"
    export RVC_ROOT="$SERVICES_DIR/voice-engine"
    export ASSETS_ROOT="$ASSETS_DIR"
    export PREPROCESSOR_URL="http://localhost:8003"
    
    # Ensure data directory exists
    mkdir -p "$DATA_ROOT"
    
    python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
}

start_preprocess() {
    log_info "Starting Preprocessor service..."
    check_assets || exit 1
    
    cd "$SERVICES_DIR/preprocessor"
    
    # Check for virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        log_warn "No virtual environment found. Using system Python."
    fi
    
    # UNIFIED PATHS - matching Docker compose configuration
    export HTTP_PORT=8003
    export DATA_ROOT="$DATA_DIR"
    export UPLOADS_DIR="$DATA_DIR/uploads"
    export ASSETS_ROOT="$ASSETS_DIR"
    export HUBERT_PATH="$ASSETS_DIR/hubert/hubert_base.pt"
    export RMVPE_PATH="$ASSETS_DIR/rmvpe"
    
    # Add RVC module to Python path (needed for RMVPE import)
    export PYTHONPATH="$SERVICES_DIR/voice-engine:$PYTHONPATH"
    
    # Ensure directories exist
    mkdir -p "$DATA_ROOT" "$UPLOADS_DIR"
    
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
