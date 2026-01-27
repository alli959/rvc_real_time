#!/usr/bin/env bash
# =============================================================================
# MorphVox - Storage Migration Script
# =============================================================================
#
# This script migrates existing scattered data to the new unified storage layout.
#
# Usage:
#   ./scripts/migrate-storage.sh              # Dry run (show what would happen)
#   ./scripts/migrate-storage.sh --execute    # Actually perform migration
#   ./scripts/migrate-storage.sh --force      # Execute without confirmation
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
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_action() { echo -e "${CYAN}[ACTION]${NC} $1"; }

# Configuration
STORAGE_ROOT="$PROJECT_ROOT/storage"
OLD_VOICE_ENGINE_DIR="$PROJECT_ROOT/services/voice-engine"
OLD_DATA_DIR="$PROJECT_ROOT/data"
DRY_RUN=true
FORCE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --execute)
            DRY_RUN=false
            ;;
        --force)
            FORCE=true
            DRY_RUN=false
            ;;
        --help|-h)
            echo "Usage: $0 [--execute] [--force]"
            echo ""
            echo "Options:"
            echo "  --execute  Actually perform the migration (default: dry run)"
            echo "  --force    Execute without confirmation prompts"
            exit 0
            ;;
    esac
done

echo ""
echo "=============================================="
echo "  MorphVox Storage Migration Script"
echo "=============================================="
echo ""

if $DRY_RUN; then
    log_warn "DRY RUN MODE - No changes will be made"
    log_info "Run with --execute to perform actual migration"
    echo ""
fi

# =============================================================================
# Step 1: Create new storage directory structure
# =============================================================================

log_info "Step 1: Creating storage directory structure..."

create_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        if $DRY_RUN; then
            log_action "Would create: $dir"
        else
            mkdir -p "$dir"
            log_success "Created: $dir"
        fi
    else
        log_info "Exists: $dir"
    fi
}

# Create all directories
create_dir "$STORAGE_ROOT/logs/api"
create_dir "$STORAGE_ROOT/logs/api-worker"
create_dir "$STORAGE_ROOT/logs/web"
create_dir "$STORAGE_ROOT/logs/voice-engine"
create_dir "$STORAGE_ROOT/logs/trainer"
create_dir "$STORAGE_ROOT/logs/preprocess"
create_dir "$STORAGE_ROOT/logs/nginx"
create_dir "$STORAGE_ROOT/logs/mariadb"
create_dir "$STORAGE_ROOT/logs/redis"
create_dir "$STORAGE_ROOT/logs/minio"

create_dir "$STORAGE_ROOT/data/uploads"
create_dir "$STORAGE_ROOT/data/preprocess"
create_dir "$STORAGE_ROOT/data/training"
create_dir "$STORAGE_ROOT/data/outputs"
create_dir "$STORAGE_ROOT/data/infra/mariadb"
create_dir "$STORAGE_ROOT/data/infra/redis"
create_dir "$STORAGE_ROOT/data/infra/minio"
create_dir "$STORAGE_ROOT/data/infra/certbot/www"
create_dir "$STORAGE_ROOT/data/infra/certbot/conf"

create_dir "$STORAGE_ROOT/assets/hubert"
create_dir "$STORAGE_ROOT/assets/rmvpe"
create_dir "$STORAGE_ROOT/assets/pretrained_v2"
create_dir "$STORAGE_ROOT/assets/uvr5_weights"
create_dir "$STORAGE_ROOT/assets/bark"
create_dir "$STORAGE_ROOT/assets/whisper"
create_dir "$STORAGE_ROOT/assets/index"

create_dir "$STORAGE_ROOT/models"

echo ""

# =============================================================================
# Step 2: Migrate assets from voice-engine
# =============================================================================

log_info "Step 2: Migrating assets from voice-engine..."

migrate_dir() {
    local src="$1"
    local dst="$2"
    local desc="$3"
    
    if [ -d "$src" ] && [ "$(ls -A "$src" 2>/dev/null)" ]; then
        if $DRY_RUN; then
            log_action "Would move: $src/* -> $dst/ ($desc)"
        else
            # Copy contents (not the directory itself)
            cp -rn "$src"/* "$dst"/ 2>/dev/null || true
            log_success "Migrated: $desc"
        fi
    else
        log_info "Skip (empty/missing): $src"
    fi
}

migrate_file() {
    local src="$1"
    local dst="$2"
    local desc="$3"
    
    if [ -f "$src" ]; then
        if $DRY_RUN; then
            log_action "Would copy: $src -> $dst ($desc)"
        else
            cp -n "$src" "$dst" 2>/dev/null || true
            log_success "Migrated: $desc"
        fi
    else
        log_info "Skip (not found): $src"
    fi
}

# Migrate HuBERT
migrate_file "$OLD_VOICE_ENGINE_DIR/assets/hubert/hubert_base.pt" \
    "$STORAGE_ROOT/assets/hubert/hubert_base.pt" "HuBERT model"

# Migrate RMVPE
migrate_dir "$OLD_VOICE_ENGINE_DIR/assets/rmvpe" \
    "$STORAGE_ROOT/assets/rmvpe" "RMVPE models"

# Migrate pretrained_v2
migrate_dir "$OLD_VOICE_ENGINE_DIR/assets/pretrained_v2" \
    "$STORAGE_ROOT/assets/pretrained_v2" "Pretrained RVC models"

# Migrate UVR5 weights
migrate_dir "$OLD_VOICE_ENGINE_DIR/assets/uvr5_weights" \
    "$STORAGE_ROOT/assets/uvr5_weights" "UVR5 weights"

# Migrate Bark models
migrate_dir "$OLD_VOICE_ENGINE_DIR/assets/bark" \
    "$STORAGE_ROOT/assets/bark" "Bark TTS models"

# Migrate index files
migrate_dir "$OLD_VOICE_ENGINE_DIR/assets/index" \
    "$STORAGE_ROOT/assets/index" "FAISS index files"

echo ""

# =============================================================================
# Step 3: Migrate user models
# =============================================================================

log_info "Step 3: Migrating user voice models..."

# Migrate models directory
if [ -d "$OLD_VOICE_ENGINE_DIR/assets/models" ]; then
    for model_dir in "$OLD_VOICE_ENGINE_DIR/assets/models"/*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            
            # Skip if it's a training artifact directory
            if [[ "$model_name" == *"_gt_wavs"* ]] || [[ "$model_name" == *"_16k_wavs"* ]]; then
                continue
            fi
            
            # Look for .pth files
            for pth_file in "$model_dir"/*.pth; do
                if [ -f "$pth_file" ]; then
                    pth_name=$(basename "$pth_file" .pth)
                    dst_pth="$STORAGE_ROOT/models/$pth_name.pth"
                    
                    if $DRY_RUN; then
                        log_action "Would copy: $pth_file -> $dst_pth"
                    else
                        cp -n "$pth_file" "$dst_pth" 2>/dev/null || true
                        log_success "Migrated model: $pth_name.pth"
                    fi
                fi
            done
            
            # Look for .index files
            for index_file in "$model_dir"/*.index; do
                if [ -f "$index_file" ]; then
                    index_name=$(basename "$index_file" .index)
                    dst_index="$STORAGE_ROOT/models/$index_name.index"
                    
                    if $DRY_RUN; then
                        log_action "Would copy: $index_file -> $dst_index"
                    else
                        cp -n "$index_file" "$dst_index" 2>/dev/null || true
                        log_success "Migrated index: $index_name.index"
                    fi
                fi
            done
            
            # Create model metadata directory
            if $DRY_RUN; then
                log_action "Would create metadata dir: $STORAGE_ROOT/models/$model_name/"
            else
                mkdir -p "$STORAGE_ROOT/models/$model_name"
            fi
        fi
    done
fi

echo ""

# =============================================================================
# Step 4: Migrate uploads/data
# =============================================================================

log_info "Step 4: Migrating data directories..."

# Migrate old data/uploads to storage/data/uploads
if [ -d "$OLD_DATA_DIR/uploads" ]; then
    migrate_dir "$OLD_DATA_DIR/uploads" "$STORAGE_ROOT/data/uploads" "Upload data"
fi

# Migrate voice-engine uploads
if [ -d "$OLD_VOICE_ENGINE_DIR/uploads" ]; then
    migrate_dir "$OLD_VOICE_ENGINE_DIR/uploads" "$STORAGE_ROOT/data/uploads" "Voice engine uploads"
fi

# Migrate voice-engine data
if [ -d "$OLD_VOICE_ENGINE_DIR/data" ]; then
    migrate_dir "$OLD_VOICE_ENGINE_DIR/data" "$STORAGE_ROOT/data" "Voice engine data"
fi

# Migrate voice-engine outputs
if [ -d "$OLD_VOICE_ENGINE_DIR/outputs" ]; then
    migrate_dir "$OLD_VOICE_ENGINE_DIR/outputs" "$STORAGE_ROOT/data/outputs" "Voice engine outputs"
fi

echo ""

# =============================================================================
# Step 5: Create compatibility symlinks
# =============================================================================

log_info "Step 5: Creating compatibility symlinks..."

create_symlink() {
    local src="$1"
    local dst="$2"
    local desc="$3"
    
    if [ -L "$dst" ]; then
        log_info "Symlink exists: $dst"
        return
    fi
    
    if [ -e "$dst" ]; then
        log_warn "Cannot create symlink, path exists: $dst"
        return
    fi
    
    if $DRY_RUN; then
        log_action "Would create symlink: $dst -> $src ($desc)"
    else
        # Ensure parent directory exists
        mkdir -p "$(dirname "$dst")"
        ln -sf "$src" "$dst"
        log_success "Created symlink: $dst -> $src"
    fi
}

# Create symlinks for backward compatibility
# These allow old paths to still work during transition

# Voice engine expects assets at /app/assets -> link to /storage/assets
# (This is handled by volume mounts in docker-compose)

# Create symlink from old location to new for local dev
if [ ! -e "$OLD_VOICE_ENGINE_DIR/assets/models" ] || [ -L "$OLD_VOICE_ENGINE_DIR/assets/models" ]; then
    create_symlink "$STORAGE_ROOT/models" "$OLD_VOICE_ENGINE_DIR/assets/models" "Models backward compat"
fi

echo ""

# =============================================================================
# Step 6: Update .env files
# =============================================================================

log_info "Step 6: Checking environment configuration..."

update_env_var() {
    local file="$1"
    local var="$2"
    local value="$3"
    
    if [ -f "$file" ]; then
        if grep -q "^${var}=" "$file"; then
            log_info "$file: $var already set"
        else
            if $DRY_RUN; then
                log_action "Would add to $file: ${var}=${value}"
            else
                echo "${var}=${value}" >> "$file"
                log_success "Added to $file: ${var}=${value}"
            fi
        fi
    fi
}

# Note: We don't automatically update .env files to avoid breaking existing setups
# Instead, we show what should be added
log_info "Recommended .env additions (add manually if not present):"
echo ""
echo "  STORAGE_ROOT=./storage"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "  Migration Summary"
echo "=============================================="
echo ""

if $DRY_RUN; then
    log_warn "DRY RUN COMPLETE - No changes were made"
    echo ""
    log_info "To perform the actual migration, run:"
    echo "  $0 --execute"
    echo ""
    log_info "To perform without confirmation prompts:"
    echo "  $0 --force"
else
    log_success "Migration complete!"
    echo ""
    log_info "Next steps:"
    echo "  1. Update your .env file with: STORAGE_ROOT=./storage"
    echo "  2. Stop running containers: docker compose down"
    echo "  3. Start with new compose file: docker compose up -d"
    echo "  4. Verify storage paths: docker exec morphvox-voice-engine ls -la /storage"
    echo ""
    log_warn "The old directories have been kept for safety."
    log_info "Once verified, you can remove:"
    echo "  - services/voice-engine/assets/* (except code files)"
    echo "  - services/voice-engine/uploads/"
    echo "  - services/voice-engine/data/"
    echo "  - services/voice-engine/outputs/"
    echo "  - data/ (old data directory)"
fi

echo ""
