#!/usr/bin/env bash
set -euo pipefail

# Downloads UVR5 (Ultimate Vocal Remover) models for vocal/instrumental separation
# These models are used for splitting vocals from music

BASE_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights"

# Define UVR5 models - HP5_only_main_vocal is best for general vocal extraction
declare -A UVR5_MODELS=(
    ["HP5_only_main_vocal.pth"]="$BASE_URL/HP5_only_main_vocal.pth"
    ["HP2_all_vocals.pth"]="$BASE_URL/HP2_all_vocals.pth"
    ["HP3_all_vocals.pth"]="$BASE_URL/HP3_all_vocals.pth"
)

# Optional de-echo/de-reverb models
declare -A DEREVERB_MODELS=(
    ["VR-DeEchoNormal.pth"]="$BASE_URL/VR-DeEchoNormal.pth"
    ["VR-DeEchoAggressive.pth"]="$BASE_URL/VR-DeEchoAggressive.pth"
)

# Create directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="$(dirname "$SCRIPT_DIR")/assets/uvr5_weights"
mkdir -p "$ASSETS_DIR"

# Function to download with curl or wget
download_file() {
    local url="$1"
    local output="$2"
    
    if command -v wget &> /dev/null; then
        echo "Downloading $(basename "$output") using wget..."
        wget -q --show-progress "$url" -O "$output"
    elif command -v curl &> /dev/null; then
        echo "Downloading $(basename "$output") using curl..."
        curl -L --progress-bar -o "$output" "$url"
    else
        echo "Error: Neither wget nor curl is available. Please install one of them."
        exit 1
    fi
}

echo "Downloading UVR5 vocal separation models..."
echo "============================================"
echo ""

# Download core UVR5 models
for model in "${!UVR5_MODELS[@]}"; do
    output_path="$ASSETS_DIR/$model"
    if [ -f "$output_path" ]; then
        echo "✓ $model already exists"
    else
        echo "Downloading $model..."
        download_file "${UVR5_MODELS[$model]}" "$output_path"
        echo "✓ Downloaded $model"
    fi
done

echo ""

# Parse command line for optional de-reverb models
if [[ "${1:-}" == "--with-dereverb" ]]; then
    echo "Downloading optional de-echo/de-reverb models..."
    for model in "${!DEREVERB_MODELS[@]}"; do
        output_path="$ASSETS_DIR/$model"
        if [ -f "$output_path" ]; then
            echo "✓ $model already exists"
        else
            echo "Downloading $model..."
            download_file "${DEREVERB_MODELS[$model]}" "$output_path"
            echo "✓ Downloaded $model"
        fi
    done
fi

echo ""
echo "✓ UVR5 models downloaded successfully!"
echo ""
echo "Available models in $ASSETS_DIR:"
ls -lh "$ASSETS_DIR"/*.pth 2>/dev/null || echo "  (no models found)"
