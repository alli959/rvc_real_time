#!/usr/bin/env bash
set -euo pipefail

# Downloads the two auxiliary checkpoints that RVC inference requires:
#  - HuBERT (content encoder)
#  - RMVPE (pitch extractor)

HUBERT_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
RMVPE_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

# Create directories
mkdir -p assets/hubert assets/rmvpe

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

# Download HuBERT
if [ -f "assets/hubert/hubert_base.pt" ]; then
    echo "✓ HuBERT model already exists (assets/hubert/hubert_base.pt)"
else
    download_file "$HUBERT_URL" "assets/hubert/hubert_base.pt"
    echo "✓ Downloaded HuBERT model"
fi

# Download RMVPE
if [ -f "assets/rmvpe/rmvpe.pt" ]; then
    echo "✓ RMVPE model already exists (assets/rmvpe/rmvpe.pt)"
else
    download_file "$RMVPE_URL" "assets/rmvpe/rmvpe.pt"
    echo "✓ Downloaded RMVPE model"
fi

echo ""
echo "✓ All required assets are available!"
