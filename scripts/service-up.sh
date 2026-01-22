#!/usr/bin/env bash
# =============================================================================
# MorphVox - Start Individual Service
# =============================================================================
#
# Usage:
#   ./scripts/service-up.sh <service>     # Start specific service with dependencies
#   ./scripts/service-up.sh <service> -d  # Detached mode (background)
#
# Services:
#   api         - Laravel API backend
#   web         - Next.js frontend
#   voice-engine - Voice inference service
#   trainer     - RVC training service
#   preprocess  - Audio preprocessing service
#   infra       - Infrastructure only (db, redis, minio)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_DIR="$PROJECT_ROOT/infra/compose"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Service Dependencies
# =============================================================================

declare -A SERVICE_DEPS=(
    ["api"]="db redis minio"
    ["api-worker"]="db redis minio api"
    ["web"]="api"
    ["voice-engine"]="minio"
    ["trainer"]="preprocess minio"
    ["preprocess"]="minio"
    ["infra"]=""
)

declare -A SERVICE_PORTS=(
    ["api"]="8080"
    ["web"]="3000"
    ["voice-engine"]="8001 8765"
    ["trainer"]="8002"
    ["preprocess"]="8003"
    ["minio"]="9000 9001"
    ["redis"]="6379"
    ["db"]="3306"
)

# =============================================================================
# Functions
# =============================================================================

usage() {
    echo "Usage: $0 <service> [-d]"
    echo ""
    echo "Services:"
    echo "  api          Laravel API backend (ports: 8080)"
    echo "  web          Next.js frontend (ports: 3000)"
    echo "  voice-engine Voice inference (ports: 8001, 8765)"
    echo "  trainer      RVC training (ports: 8002)"
    echo "  preprocess   Audio preprocessing (ports: 8003)"
    echo "  infra        Infrastructure only (db, redis, minio)"
    echo ""
    echo "Options:"
    echo "  -d           Run in detached (background) mode"
    echo ""
    echo "Example:"
    echo "  $0 trainer        # Start trainer with dependencies"
    echo "  $0 infra -d       # Start infrastructure in background"
}

check_assets() {
    local service="$1"
    
    # Services that need assets
    case "$service" in
        voice-engine|trainer|preprocess)
            local assets_dir="$PROJECT_ROOT/services/voice-engine/assets"
            
            if [ ! -f "$assets_dir/hubert/hubert_base.pt" ]; then
                log_error "HuBERT model not found. Run: ./scripts/dev-up.sh --no-docker"
                return 1
            fi
            
            if [ ! -f "$assets_dir/rmvpe/rmvpe.pt" ]; then
                log_error "RMVPE model not found. Run: ./scripts/dev-up.sh --no-docker"
                return 1
            fi
            ;;
    esac
    
    return 0
}

get_services() {
    local service="$1"
    
    case "$service" in
        infra)
            echo "db redis minio minio-init"
            ;;
        *)
            local deps="${SERVICE_DEPS[$service]:-}"
            echo "$deps $service"
            ;;
    esac
}

start_service() {
    local service="$1"
    local detached="$2"
    
    check_assets "$service" || exit 1
    
    local services=$(get_services "$service")
    local flags=""
    
    if [ "$detached" = true ]; then
        flags="-d"
    fi
    
    log_info "Starting: $services"
    
    cd "$COMPOSE_DIR"
    
    # Use docker compose (v2) or docker-compose (v1)
    if docker compose version &> /dev/null; then
        docker compose -f docker-compose.yml up $flags $services
    else
        docker-compose -f docker-compose.yml up $flags $services
    fi
}

show_ports() {
    local service="$1"
    local ports="${SERVICE_PORTS[$service]:-}"
    
    if [ -n "$ports" ]; then
        log_info "Service $service available on port(s): $ports"
    fi
}

# =============================================================================
# Main
# =============================================================================

SERVICE="${1:-}"
DETACHED=false

if [ -z "$SERVICE" ]; then
    usage
    exit 1
fi

shift
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--detached) DETACHED=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate service
valid_services="api web voice-engine trainer preprocess infra"
if [[ ! " $valid_services " =~ " $SERVICE " ]]; then
    log_error "Unknown service: $SERVICE"
    usage
    exit 1
fi

start_service "$SERVICE" "$DETACHED"

if [ "$DETACHED" = true ]; then
    show_ports "$SERVICE"
    log_success "Service started in background"
fi
