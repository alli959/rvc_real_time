#!/usr/bin/env bash
# =============================================================================
# MorphVox Development Environment - Complete Setup and Start
# =============================================================================
#
# This script:
#   1. Checks and downloads missing assets (HuBERT, RMVPE, pretrained models)
#   2. Ensures all required directories exist
#   3. Starts the development Docker Compose stack
#
# Usage:
#   ./scripts/dev-up.sh              # Start dev environment
#   ./scripts/dev-up.sh --prod       # Start production environment
#   ./scripts/dev-up.sh --no-docker  # Only setup assets, don't start Docker
#   ./scripts/dev-up.sh --rebuild    # Rebuild images before starting
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${CYAN}========================================${NC}"; echo -e "${CYAN}$1${NC}"; echo -e "${CYAN}========================================${NC}"; }

# =============================================================================
# Configuration
# =============================================================================

ASSETS_DIR="$PROJECT_ROOT/services/voice-engine/assets"
DATA_DIR="$PROJECT_ROOT/data"
COMPOSE_FILE="$PROJECT_ROOT/infra/compose/docker-compose.yml"
PROD_COMPOSE_FILE="$PROJECT_ROOT/infra/compose/docker-compose.prod.yml"

# Asset URLs
HUBERT_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
RMVPE_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

# Pretrained model URLs (RVC v2)
PRETRAINED_BASE_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2"
PRETRAINED_MODELS=(
    "f0G48k.pth"
    "f0D48k.pth"
    "f0G40k.pth"
    "f0D40k.pth"
    "f0G32k.pth"
    "f0D32k.pth"
)

# Parse arguments
PROD_MODE=false
NO_DOCKER=false
REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --prod) PROD_MODE=true; shift ;;
        --no-docker) NO_DOCKER=true; shift ;;
        --rebuild) REBUILD=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--prod] [--no-docker] [--rebuild]"
            echo ""
            echo "Options:"
            echo "  --prod       Use production compose file"
            echo "  --no-docker  Only setup assets, don't start Docker"
            echo "  --rebuild    Rebuild Docker images before starting"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Download Helper
# =============================================================================

download_file() {
    local url="$1"
    local output="$2"
    local name="$(basename "$output")"
    
    if [ -f "$output" ]; then
        log_info "✓ $name already exists"
        return 0
    fi
    
    log_info "Downloading $name..."
    
    # Create directory if needed
    mkdir -p "$(dirname "$output")"
    
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$url" -O "$output" || {
            log_error "Failed to download $name"
            rm -f "$output"
            return 1
        }
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output" "$url" || {
            log_error "Failed to download $name"
            rm -f "$output"
            return 1
        }
    else
        log_error "Neither wget nor curl is available"
        return 1
    fi
    
    log_success "Downloaded $name"
}

# =============================================================================
# Setup Functions
# =============================================================================

setup_directories() {
    log_header "Setting up directories"
    
    # Create asset directories
    mkdir -p "$ASSETS_DIR/hubert"
    mkdir -p "$ASSETS_DIR/rmvpe"
    mkdir -p "$ASSETS_DIR/pretrained_v2"
    mkdir -p "$ASSETS_DIR/models"
    mkdir -p "$ASSETS_DIR/index"
    
    # Create data directory (for training data shared volume)
    mkdir -p "$DATA_DIR"
    mkdir -p "$DATA_DIR/uploads"
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    log_success "Directories created"
}

download_hubert() {
    log_header "Checking HuBERT Model"
    download_file "$HUBERT_URL" "$ASSETS_DIR/hubert/hubert_base.pt"
}

download_rmvpe() {
    log_header "Checking RMVPE Model"
    download_file "$RMVPE_URL" "$ASSETS_DIR/rmvpe/rmvpe.pt"
}

download_pretrained() {
    log_header "Checking Pretrained Models (RVC v2)"
    
    for model in "${PRETRAINED_MODELS[@]}"; do
        download_file "$PRETRAINED_BASE_URL/$model" "$ASSETS_DIR/pretrained_v2/$model"
    done
}

verify_assets() {
    log_header "Verifying Assets"
    
    local missing=0
    
    # Check HuBERT
    if [ -f "$ASSETS_DIR/hubert/hubert_base.pt" ]; then
        log_info "✓ HuBERT model present"
    else
        log_error "✗ HuBERT model missing"
        missing=$((missing + 1))
    fi
    
    # Check RMVPE
    if [ -f "$ASSETS_DIR/rmvpe/rmvpe.pt" ]; then
        log_info "✓ RMVPE model present"
    else
        log_error "✗ RMVPE model missing"
        missing=$((missing + 1))
    fi
    
    # Check pretrained models
    for model in "${PRETRAINED_MODELS[@]}"; do
        if [ -f "$ASSETS_DIR/pretrained_v2/$model" ]; then
            log_info "✓ $model present"
        else
            log_error "✗ $model missing"
            missing=$((missing + 1))
        fi
    done
    
    if [ $missing -gt 0 ]; then
        log_error "$missing assets missing!"
        return 1
    fi
    
    log_success "All assets verified"
}

# =============================================================================
# Docker Functions
# =============================================================================

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        return 1
    fi
    
    log_success "Docker is ready"
}

start_docker() {
    log_header "Starting Docker Services"
    
    cd "$PROJECT_ROOT/infra/compose"
    
    local compose_file="$COMPOSE_FILE"
    if [ "$PROD_MODE" = true ]; then
        compose_file="$PROD_COMPOSE_FILE"
        log_info "Using production compose file"
    else
        log_info "Using development compose file"
    fi
    
    local build_flag=""
    if [ "$REBUILD" = true ]; then
        build_flag="--build"
        log_info "Rebuilding images..."
    fi
    
    # Use docker compose (v2) or docker-compose (v1)
    if docker compose version &> /dev/null; then
        docker compose -f "$compose_file" up $build_flag -d
    else
        docker-compose -f "$compose_file" up $build_flag -d
    fi
    
    log_success "Docker services started"
}

show_status() {
    log_header "Service Status"
    
    cd "$PROJECT_ROOT/infra/compose"
    
    local compose_file="$COMPOSE_FILE"
    if [ "$PROD_MODE" = true ]; then
        compose_file="$PROD_COMPOSE_FILE"
    fi
    
    if docker compose version &> /dev/null; then
        docker compose -f "$compose_file" ps
    else
        docker-compose -f "$compose_file" ps
    fi
    
    echo ""
    log_info "Service URLs:"
    echo "  - API:          http://localhost:8080"
    echo "  - Web UI:       http://localhost:3000"
    echo "  - Voice Engine: http://localhost:8001"
    echo "  - Trainer:      http://localhost:8002"
    echo "  - Preprocessor: http://localhost:8003"
    echo "  - MinIO:        http://localhost:9001"
    echo ""
    log_info "To view logs: docker compose -f $compose_file logs -f [service]"
}

# =============================================================================
# Main
# =============================================================================

main() {
    log_header "MorphVox Development Setup"
    
    cd "$PROJECT_ROOT"
    
    # Setup directories
    setup_directories
    
    # Download assets (conditional - skips if present)
    download_hubert
    download_rmvpe
    download_pretrained
    
    # Verify all assets
    verify_assets
    
    if [ "$NO_DOCKER" = true ]; then
        log_success "Asset setup complete (--no-docker specified)"
        exit 0
    fi
    
    # Check Docker
    check_docker
    
    # Start services
    start_docker
    
    # Show status
    show_status
    
    log_success "Development environment is ready!"
}

main "$@"
