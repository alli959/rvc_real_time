#!/bin/bash
#
# start-api.sh - Quick launcher for RVC API mode with model selection
#
# Usage:
#   ./start-api.sh              # Interactive model selection
#   ./start-api.sh <model_name> # Direct model selection by name
#   ./start-api.sh 1            # Select first model
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/assets/models"

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Function to find the best .pth model file in a directory
# Priority: 1) *_infer.pth, 2) highest numbered G_*.pth, 3) any .pth file (not D_*)
find_model_file() {
    local model_dir="$1"
    
    # 1) Look for *_infer.pth (optimized inference checkpoint)
    local infer_file=$(ls -1 "$model_dir"/*_infer.pth 2>/dev/null | head -1)
    if [[ -n "$infer_file" && -f "$infer_file" ]]; then
        echo "$infer_file"
        return 0
    fi
    
    # 2) Look for G_*.pth files (generator checkpoints) and pick the highest numbered one
    local g_files=$(ls -1 "$model_dir"/G_*.pth 2>/dev/null | grep -v "_infer" || true)
    if [[ -n "$g_files" ]]; then
        local highest_num=0
        local highest_file=""
        while IFS= read -r f; do
            [[ -f "$f" ]] || continue
            # Extract number from G_1234.pth
            local basename=$(basename "$f")
            local num=$(echo "$basename" | sed -n 's/G_\([0-9]*\)\.pth/\1/p')
            if [[ -n "$num" && "$num" -gt "$highest_num" ]]; then
                highest_num=$num
                highest_file="$f"
            fi
        done <<< "$g_files"
        
        if [[ -n "$highest_file" ]]; then
            echo "$highest_file"
            return 0
        fi
    fi
    
    # 3) Look for any .pth file (pre-packaged models like BillCipher.pth), skip D_*.pth
    for f in "$model_dir"/*.pth; do
        [[ -f "$f" ]] || continue
        local basename=$(basename "$f")
        # Skip discriminator checkpoints
        [[ "$basename" == D_*.pth ]] && continue
        echo "$f"
        return 0
    done
    
    return 1
}

# Function to find the best .index file in a directory
find_index_file() {
    local model_dir="$1"
    
    # Look for .index files, prefer "added_*" or "trained_*" patterns
    local idx=$(ls -1 "$model_dir"/added_*.index 2>/dev/null | head -1)
    if [[ -n "$idx" && -f "$idx" ]]; then
        echo "$idx"
        return 0
    fi
    
    idx=$(ls -1 "$model_dir"/trained_*.index 2>/dev/null | head -1)
    if [[ -n "$idx" && -f "$idx" ]]; then
        echo "$idx"
        return 0
    fi
    
    # Fall back to any .index file
    idx=$(ls -1 "$model_dir"/*.index 2>/dev/null | head -1)
    if [[ -n "$idx" && -f "$idx" ]]; then
        echo "$idx"
        return 0
    fi
    
    return 1
}

# Function to resolve symlinks
resolve_path() {
    local path="$1"
    if [[ -L "$path" ]]; then
        readlink -f "$path"
    else
        echo "$path"
    fi
}

# Scan models directory and build model list
declare -a MODEL_NAMES
declare -A MODEL_PATHS
declare -A MODEL_FILES
declare -A INDEX_FILES

scan_models() {
    echo -e "${CYAN}Scanning models directory...${NC}"
    
    local count=0
    
    for dir in "$MODELS_DIR"/*; do
        # Skip non-directories and files
        [[ -d "$dir" ]] || continue
        
        # Resolve symlinks
        local resolved_dir=$(resolve_path "$dir")
        local model_name=$(basename "$dir")
        
        # Skip if resolved path doesn't exist
        [[ -d "$resolved_dir" ]] || continue
        
        # Find model file
        local model_file=$(find_model_file "$resolved_dir" || true)
        
        # If no model file found, check subdirectories (for nested structure or symlinks)
        if [[ -z "$model_file" ]]; then
            for subdir in "$resolved_dir"/*; do
                if [[ -L "$subdir" || -d "$subdir" ]]; then
                    local sub_resolved=$(resolve_path "$subdir")
                    if [[ -d "$sub_resolved" ]]; then
                        model_file=$(find_model_file "$sub_resolved" || true)
                        if [[ -n "$model_file" ]]; then
                            resolved_dir="$sub_resolved"
                            break
                        fi
                    fi
                fi
            done
        fi
        
        # Skip if no model file found
        [[ -n "$model_file" ]] || continue
        
        # Find index file
        local index_file=$(find_index_file "$resolved_dir" || true)
        
        # Add to arrays
        MODEL_NAMES+=("$model_name")
        MODEL_PATHS["$model_name"]="$resolved_dir"
        MODEL_FILES["$model_name"]="$model_file"
        INDEX_FILES["$model_name"]="$index_file"
        
        count=$((count + 1))
    done
    
    if [[ $count -eq 0 ]]; then
        echo -e "${RED}No models found in $MODELS_DIR${NC}"
        echo "Please add model directories with .pth files to assets/models/"
        exit 1
    fi
    
    echo -e "${GREEN}Found $count models${NC}"
}

# Display model selection menu
show_menu() {
    echo ""
    echo -e "${BOLD}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║                ${CYAN}RVC Real-Time Voice Conversion${NC}${BOLD}                         ║${NC}"
    echo -e "${BOLD}║                ${YELLOW}Select a Model to Start API${NC}${BOLD}                           ║${NC}"
    echo -e "${BOLD}╠════════════════════════════════════════════════════════════════════════╣${NC}"
    
    local i=1
    for name in "${MODEL_NAMES[@]}"; do
        local model_file="${MODEL_FILES[$name]}"
        local index_file="${INDEX_FILES[$name]}"
        local model_basename=$(basename "$model_file")
        
        # Format index info
        local index_info=""
        if [[ -n "$index_file" ]]; then
            index_info="${GREEN}✓${NC}"
        else
            index_info="${YELLOW}○${NC}"
        fi
        
        printf "${BOLD}║${NC} ${CYAN}%2d)${NC} %-22s ${BLUE}%-30s${NC} %b ${BOLD}║${NC}\n" \
            "$i" "$name" "$model_basename" "$index_info"
        i=$((i + 1))
    done
    
    echo -e "${BOLD}╠════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}✓${NC} = has .index file    ${YELLOW}○${NC} = no .index file                          ${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${RED}q)${NC} Quit                                                              ${BOLD}║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Get model by name or number
get_model_by_selection() {
    local selection="$1"
    
    # Check if it's a number
    if [[ "$selection" =~ ^[0-9]+$ ]]; then
        local idx=$((selection - 1))
        if [[ $idx -ge 0 && $idx -lt ${#MODEL_NAMES[@]} ]]; then
            echo "${MODEL_NAMES[$idx]}"
            return 0
        fi
    fi
    
    # Check if it's a name (case-insensitive partial match)
    for name in "${MODEL_NAMES[@]}"; do
        if [[ "${name,,}" == "${selection,,}" ]]; then
            echo "$name"
            return 0
        fi
    done
    
    # Partial match
    for name in "${MODEL_NAMES[@]}"; do
        if [[ "${name,,}" == *"${selection,,}"* ]]; then
            echo "$name"
            return 0
        fi
    done
    
    return 1
}

# Start the API server with selected model
start_api() {
    local model_name="$1"
    local model_file="${MODEL_FILES[$model_name]}"
    local index_file="${INDEX_FILES[$model_name]}"
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}Starting RVC API Server${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "  Model:  ${CYAN}$model_name${NC}"
    echo -e "  File:   ${BLUE}$(basename "$model_file")${NC}"
    if [[ -n "$index_file" ]]; then
        echo -e "  Index:  ${BLUE}$(basename "$index_file")${NC}"
    else
        echo -e "  Index:  ${YELLOW}(none)${NC}"
    fi
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Build command
    local cmd="python3 \"$SCRIPT_DIR/main.py\" --mode api --model \"$model_file\""
    
    if [[ -n "$index_file" ]]; then
        cmd="$cmd --index \"$index_file\""
    fi
    
    # Pass through any additional arguments
    shift
    if [[ $# -gt 0 ]]; then
        cmd="$cmd $@"
    fi
    
    echo -e "${YELLOW}Running:${NC} $cmd"
    echo ""
    
    # Execute
    eval $cmd
}

# Main
main() {
    cd "$SCRIPT_DIR"
    
    # Scan models
    scan_models
    
    # If model name provided as argument, use it directly
    if [[ -n "$1" && "$1" != "-"* ]]; then
        local model_name=$(get_model_by_selection "$1")
        if [[ -n "$model_name" ]]; then
            shift
            start_api "$model_name" "$@"
        else
            echo -e "${RED}Model '$1' not found.${NC}"
            echo "Available models:"
            for name in "${MODEL_NAMES[@]}"; do
                echo "  - $name"
            done
            exit 1
        fi
        return
    fi
    
    # Interactive mode
    while true; do
        show_menu
        
        echo -n -e "${BOLD}Select model (1-${#MODEL_NAMES[@]} or name):${NC} "
        read -r selection
        
        # Handle quit
        if [[ "${selection,,}" == "q" || "${selection,,}" == "quit" ]]; then
            echo -e "${YELLOW}Goodbye!${NC}"
            exit 0
        fi
        
        # Handle empty input
        if [[ -z "$selection" ]]; then
            continue
        fi
        
        # Try to get model
        local model_name=$(get_model_by_selection "$selection")
        if [[ -n "$model_name" ]]; then
            start_api "$model_name" "$@"
            break
        else
            echo -e "${RED}Invalid selection. Please try again.${NC}"
            sleep 1
        fi
    done
}

main "$@"
