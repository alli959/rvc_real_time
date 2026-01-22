#!/bin/bash

# =============================================================================
# MorphVox - Fresh Local Development Setup
# =============================================================================
# This script sets up and starts all services from scratch for local development.
#
# Usage:
#   ./run-local-fresh.sh              # Start all services
#   ./run-local-fresh.sh --rebuild    # Force rebuild all images
#   ./run-local-fresh.sh --clean      # Clean volumes and restart fresh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_DIR="$PROJECT_ROOT/infra/compose"

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

# Parse arguments
REBUILD=false
CLEAN=false

for arg in "$@"; do
    case $arg in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
    esac
done

cd "$COMPOSE_DIR"

echo ""
echo "=========================================="
echo "  MorphVox - Local Development Setup"
echo "=========================================="
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    log_warn "Cleaning up volumes and containers..."
    docker compose -f docker-compose.yml down -v --remove-orphans 2>/dev/null || true
fi

# Check for .env file
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        log_info "Creating .env file from .env.example..."
        cp .env.example .env
    else
        log_warn "No .env file found. Using defaults."
    fi
fi

# Start database and storage services first
log_info "Starting database and storage services..."
docker compose -f docker-compose.yml up -d db redis minio

# Wait for services to be healthy
log_info "Waiting for database to be ready..."
sleep 5

# Check if we need to rebuild
BUILD_ARG=""
if [ "$REBUILD" = true ]; then
    BUILD_ARG="--build"
    log_info "Rebuilding all images..."
fi

# Start API service
log_info "Starting API service..."
docker compose -f docker-compose.yml up -d $BUILD_ARG api

# Wait for API migrations
log_info "Waiting for API to initialize..."
sleep 10

# Run migrations if needed
log_info "Running database migrations..."
docker compose -f docker-compose.yml exec -T api php artisan migrate --force 2>/dev/null || true

# Start remaining services
log_info "Starting voice engine services..."
docker compose -f docker-compose.yml up -d $BUILD_ARG voice-engine preprocess trainer

# Start frontend
log_info "Starting web frontend..."
docker compose -f docker-compose.yml up -d $BUILD_ARG web

# Start queue worker
log_info "Starting queue worker..."
docker compose -f docker-compose.yml up -d $BUILD_ARG api-worker

echo ""
log_success "All services started!"
echo ""
echo "=========================================="
echo "  Service URLs"
echo "=========================================="
echo ""
echo "  Web UI:         http://localhost:3000"
echo "  API:            http://localhost:8080"
echo "  Voice Engine:   http://localhost:8001"
echo "  Trainer:        http://localhost:8002"
echo "  Preprocessor:   http://localhost:8003"
echo "  MinIO Console:  http://localhost:9001"
echo ""
echo "=========================================="
echo "  Useful Commands"
echo "=========================================="
echo ""
echo "  View logs:      docker compose -f docker-compose.yml logs -f [service]"
echo "  Stop all:       docker compose -f docker-compose.yml down"
echo "  Rebuild:        ./run-local-fresh.sh --rebuild"
echo "  Clean restart:  ./run-local-fresh.sh --clean"
echo ""
