#!/usr/bin/env bash
set -euo pipefail

# Downloads the BillCipher model files
# 
# IMPORTANT: You need to host these files somewhere and update the URLs below
# Recommended hosting options:
#   - Hugging Face: https://huggingface.co (recommended for ML models)
#   - Google Drive with direct download links
#   - Your own server/CDN

# TODO: Update these URLs with your actual hosting locations
BILLCIPHER_PTH_URL="YOUR_URL_HERE/BillCipher.pth"
BILLCIPHER_INDEX_URL="YOUR_URL_HERE/BillCipher.index"

MODEL_DIR="assets/models/BillCipher"

# Create directory
mkdir -p "$MODEL_DIR"

# Function to download with curl or wget
download_file() {
    local url="$1"
    local output="$2"
    
    if [ "$url" = "YOUR_URL_HERE"* ]; then
        echo "⚠️  Warning: URLs not configured!"
        echo "Please edit scripts/download_billcipher_model.sh and update the download URLs."
        echo ""
        echo "To host your models:"
        echo "1. Upload BillCipher.pth and BillCipher.index to Hugging Face or Google Drive"
        echo "2. Get direct download links"
        echo "3. Update BILLCIPHER_PTH_URL and BILLCIPHER_INDEX_URL in this script"
        echo ""
        return 1
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

# Download .pth file
if [ -f "$MODEL_DIR/BillCipher.pth" ]; then
    echo "✓ BillCipher.pth already exists"
else
    download_file "$BILLCIPHER_PTH_URL" "$MODEL_DIR/BillCipher.pth" || exit 1
    echo "✓ Downloaded BillCipher.pth"
fi

# Download .index file
if [ -f "$MODEL_DIR/BillCipher.index" ]; then
    echo "✓ BillCipher.index already exists"
else
    download_file "$BILLCIPHER_INDEX_URL" "$MODEL_DIR/BillCipher.index" || exit 1
    echo "✓ Downloaded BillCipher.index"
fi

echo ""
echo "✓ BillCipher model is available!"
