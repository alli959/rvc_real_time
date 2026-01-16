#!/usr/bin/env bash
set -euo pipefail

# Downloads the pretrained RVC models required for training
# These are the Generator and Discriminator models that provide a starting point for training

BASE_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

# Create directories
mkdir -p assets/pretrained_v2

# Function to download with curl or wget
download_file() {
    local url="$1"
    local output="$2"
    
    if [ -f "$output" ]; then
        echo "✓ $(basename "$output") already exists"
        return 0
    fi
    
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

echo "Downloading RVC pretrained models (v2)..."

# Download v2 pretrained models (with pitch guidance - f0)
# These are ~145MB each
download_file "$BASE_URL/pretrained_v2/f0G48k.pth" "assets/pretrained_v2/f0G48k.pth"
download_file "$BASE_URL/pretrained_v2/f0D48k.pth" "assets/pretrained_v2/f0D48k.pth"
download_file "$BASE_URL/pretrained_v2/f0G40k.pth" "assets/pretrained_v2/f0G40k.pth"
download_file "$BASE_URL/pretrained_v2/f0D40k.pth" "assets/pretrained_v2/f0D40k.pth"
download_file "$BASE_URL/pretrained_v2/f0G32k.pth" "assets/pretrained_v2/f0G32k.pth"
download_file "$BASE_URL/pretrained_v2/f0D32k.pth" "assets/pretrained_v2/f0D32k.pth"

echo ""
echo "✓ All pretrained models downloaded!"
echo "  Location: assets/pretrained_v2/"
