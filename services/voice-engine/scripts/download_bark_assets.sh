#!/usr/bin/env bash
set -euo pipefail

# Downloads Bark TTS models from HuggingFace
# These models enable native emotion and sound effect support in TTS
#
# Models downloaded:
#   - text_2.pt (~5GB) - Text encoder for semantic tokens
#   - coarse_2.pt (~3.7GB) - Coarse acoustic model
#   - fine_2.pt (~3.5GB) - Fine acoustic model
#   - encodec_24khz-d7cc33bc.th (~89MB) - Audio codec from Facebook
#
# Total size: ~13GB

BARK_REPO="suno/bark"
HF_BASE_URL="https://huggingface.co/${BARK_REPO}/resolve/main"
ENCODEC_URL="https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th"

# Define Bark models
declare -A BARK_MODELS=(
    ["text_2.pt"]="$HF_BASE_URL/text_2.pt"
    ["coarse_2.pt"]="$HF_BASE_URL/coarse_2.pt"
    ["fine_2.pt"]="$HF_BASE_URL/fine_2.pt"
)

# Create directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="$(dirname "$SCRIPT_DIR")/assets/bark"
mkdir -p "$ASSETS_DIR"

# Function to download with curl or wget
download_file() {
    local url="$1"
    local output="$2"
    local expected_size="${3:-}"
    
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
    
    # Verify download
    if [ -f "$output" ]; then
        local actual_size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null || echo "0")
        if [ "$actual_size" -lt 1000000 ]; then
            echo "⚠️  Warning: Downloaded file seems too small ($actual_size bytes)"
            rm -f "$output"
            return 1
        fi
    fi
}

echo "============================================"
echo "Bark TTS Model Downloader"
echo "============================================"
echo ""
echo "This will download ~13GB of model files."
echo "Target directory: $ASSETS_DIR"
echo ""

# Download Bark models from HuggingFace
for model in "${!BARK_MODELS[@]}"; do
    output_path="$ASSETS_DIR/$model"
    if [ -f "$output_path" ]; then
        echo "✓ $model already exists"
    else
        echo "Downloading $model (this may take a while)..."
        if download_file "${BARK_MODELS[$model]}" "$output_path"; then
            echo "✓ Downloaded $model"
        else
            echo "✗ Failed to download $model"
            exit 1
        fi
    fi
done

echo ""

# Download encodec model
encodec_path="$ASSETS_DIR/encodec_24khz-d7cc33bc.th"
if [ -f "$encodec_path" ]; then
    echo "✓ encodec_24khz-d7cc33bc.th already exists"
else
    echo "Downloading encodec_24khz-d7cc33bc.th..."
    if download_file "$ENCODEC_URL" "$encodec_path"; then
        echo "✓ Downloaded encodec_24khz-d7cc33bc.th"
    else
        echo "✗ Failed to download encodec"
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "✓ Bark TTS models downloaded successfully!"
echo "============================================"
echo ""
echo "Models in $ASSETS_DIR:"
ls -lh "$ASSETS_DIR"/*.pt "$ASSETS_DIR"/*.th 2>/dev/null || echo "  (no models found)"
echo ""
echo "Total size:"
du -sh "$ASSETS_DIR"
