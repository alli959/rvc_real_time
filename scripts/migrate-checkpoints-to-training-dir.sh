#!/bin/bash
#
# migrate-checkpoints-to-training-dir.sh
#
# This script migrates training checkpoints from /storage/models/<model>/
# to /storage/data/training/<model>/ as part of the storage refactoring.
#
# The new storage layout:
# - /storage/models/<model>/ - ONLY final inference files: <model>.pth, <model>.index, config.json, metadata.json
# - /storage/data/training/<model>/ - All training artifacts: G_*.pth, D_*.pth, *_e*_s*.pth, tensorboard logs
# - /storage/data/preprocess/<model>/ - Preprocessed audio/features
#
# Usage:
#   ./migrate-checkpoints-to-training-dir.sh              # Dry run (default)
#   ./migrate-checkpoints-to-training-dir.sh --execute    # Actually move files
#

set -e

STORAGE_ROOT="${STORAGE_ROOT:-/home/alexanderg/rvc_real_time/storage}"
MODELS_DIR="$STORAGE_ROOT/models"
TRAINING_DIR="$STORAGE_ROOT/data/training"

DRY_RUN=true
if [[ "$1" == "--execute" ]]; then
    DRY_RUN=false
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_dry() {
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY-RUN]${NC} Would: $1"
    fi
}

# Stats
MODELS_PROCESSED=0
FILES_MOVED=0
FILES_SKIPPED=0
BYTES_MOVED=0

echo ""
echo "=============================================="
echo "  Checkpoint Migration to Training Directory"
echo "=============================================="
echo ""
echo "Source: $MODELS_DIR"
echo "Destination: $TRAINING_DIR"
echo "Mode: $(if $DRY_RUN; then echo 'DRY RUN (use --execute to actually migrate)'; else echo 'EXECUTING MIGRATION'; fi)"
echo ""

if ! [ -d "$MODELS_DIR" ]; then
    log_error "Models directory not found: $MODELS_DIR"
    exit 1
fi

# Create training dir if needed
if ! $DRY_RUN && ! [ -d "$TRAINING_DIR" ]; then
    mkdir -p "$TRAINING_DIR"
    log_info "Created training directory: $TRAINING_DIR"
fi

# Find all model directories
for model_dir in "$MODELS_DIR"/*/; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model_name=$(basename "$model_dir")
    dest_dir="$TRAINING_DIR/$model_name"
    
    # Patterns to move:
    # - G_*.pth (generator checkpoints)
    # - D_*.pth (discriminator checkpoints)
    # - *_e*_s*.pth (intermediate extracted models)
    # - events.out.tfevents.* (tensorboard files)
    # - filelist.txt (training file list)
    # - config.json (training config - only if also exists as extracted model config)
    
    checkpoint_files=()
    tensorboard_files=()
    other_training_files=()
    
    # Find checkpoint files
    while IFS= read -r -d '' file; do
        checkpoint_files+=("$file")
    done < <(find "$model_dir" -maxdepth 1 \( -name "G_*.pth" -o -name "D_*.pth" -o -name "*_e*_s*.pth" \) -print0 2>/dev/null)
    
    # Find tensorboard files
    while IFS= read -r -d '' file; do
        tensorboard_files+=("$file")
    done < <(find "$model_dir" -maxdepth 1 -name "events.out.tfevents.*" -print0 2>/dev/null)
    
    # Find filelist.txt
    if [ -f "$model_dir/filelist.txt" ]; then
        other_training_files+=("$model_dir/filelist.txt")
    fi
    
    total_files=${#checkpoint_files[@]}+${#tensorboard_files[@]}+${#other_training_files[@]}
    
    if [ "${#checkpoint_files[@]}" -eq 0 ] && [ "${#tensorboard_files[@]}" -eq 0 ]; then
        continue
    fi
    
    log_info "Processing model: $model_name"
    log_info "  Checkpoints: ${#checkpoint_files[@]}, TensorBoard: ${#tensorboard_files[@]}, Other: ${#other_training_files[@]}"
    
    MODELS_PROCESSED=$((MODELS_PROCESSED + 1))
    
    # Create destination directory
    if ! $DRY_RUN && [ ! -d "$dest_dir" ]; then
        mkdir -p "$dest_dir"
        log_info "  Created: $dest_dir"
    else
        log_dry "mkdir -p $dest_dir"
    fi
    
    # Move checkpoint files
    for file in "${checkpoint_files[@]}"; do
        filename=$(basename "$file")
        dest_file="$dest_dir/$filename"
        
        if [ -f "$dest_file" ]; then
            log_warn "  Skip (exists): $filename"
            FILES_SKIPPED=$((FILES_SKIPPED + 1))
            continue
        fi
        
        file_size=$(stat -c%s "$file" 2>/dev/null || echo 0)
        BYTES_MOVED=$((BYTES_MOVED + file_size))
        
        if $DRY_RUN; then
            log_dry "mv $filename -> $dest_dir/"
        else
            mv "$file" "$dest_file"
            log_info "  Moved: $filename ($(numfmt --to=iec-i --suffix=B $file_size))"
        fi
        FILES_MOVED=$((FILES_MOVED + 1))
    done
    
    # Move tensorboard files
    for file in "${tensorboard_files[@]}"; do
        filename=$(basename "$file")
        dest_file="$dest_dir/$filename"
        
        if [ -f "$dest_file" ]; then
            log_warn "  Skip (exists): $filename"
            FILES_SKIPPED=$((FILES_SKIPPED + 1))
            continue
        fi
        
        file_size=$(stat -c%s "$file" 2>/dev/null || echo 0)
        BYTES_MOVED=$((BYTES_MOVED + file_size))
        
        if $DRY_RUN; then
            log_dry "mv $filename -> $dest_dir/"
        else
            mv "$file" "$dest_file"
            log_info "  Moved: $filename"
        fi
        FILES_MOVED=$((FILES_MOVED + 1))
    done
    
    # Copy (not move) filelist.txt if it doesn't exist in dest
    for file in "${other_training_files[@]}"; do
        filename=$(basename "$file")
        dest_file="$dest_dir/$filename"
        
        if [ -f "$dest_file" ]; then
            log_warn "  Skip (exists): $filename"
            FILES_SKIPPED=$((FILES_SKIPPED + 1))
            continue
        fi
        
        if $DRY_RUN; then
            log_dry "cp $filename -> $dest_dir/"
        else
            cp "$file" "$dest_file"
            log_info "  Copied: $filename"
        fi
    done
    
    echo ""
done

echo "=============================================="
echo "                  Summary"
echo "=============================================="
echo ""
echo "Models processed: $MODELS_PROCESSED"
echo "Files to migrate: $FILES_MOVED"
echo "Files skipped:    $FILES_SKIPPED"
echo "Data to migrate:  $(numfmt --to=iec-i --suffix=B $BYTES_MOVED 2>/dev/null || echo "${BYTES_MOVED} bytes")"
echo ""

if $DRY_RUN; then
    echo "This was a DRY RUN. No files were moved."
    echo "Run with --execute to perform the actual migration:"
    echo "  $0 --execute"
else
    echo "Migration complete!"
    echo ""
    echo "The following files remain in /storage/models/<model>/:"
    echo "  - <model>.pth (inference model)"
    echo "  - <model>.index (feature index)"
    echo "  - config.json (model config)"
    echo "  - metadata.json (model metadata)"
fi
