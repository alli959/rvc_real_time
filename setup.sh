#!/bin/bash
# Quick start script for RVC Real-time Voice Conversion

set -e

echo "RVC Real-time Voice Conversion - Quick Start"
echo "============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip

echo "Installing core dependencies..."
pip install -r requirements.txt

echo "Installing fairseq (this may take a few minutes)..."
# Install fairseq from git without dependencies to avoid conflicts
pip install git+https://github.com/facebookresearch/fairseq.git@v0.12.2 --no-deps

# Install fairseq's required dependencies that don't conflict
pip install hydra-core omegaconf cffi bitarray sacrebleu

# Patch fairseq's broken imports
echo "Patching fairseq..."
python3 << 'PATCH_FAIRSEQ'
import os
import shutil

fairseq_path = os.path.expanduser("~/.local/lib/python3.10/site-packages/fairseq")
if not os.path.exists(fairseq_path):
    # Try in venv
    fairseq_path = "venv/lib/python3.10/site-packages/fairseq"

if os.path.exists(fairseq_path):
    problematic_files = [
        "tasks/speech_dlm_task.py",
        "tasks/online_backtranslation.py",
        "models/speech_to_speech/__init__.py",
        "models/speech_to_speech/s2s_conformer.py",
        "models/speech_to_speech/s2s_transformer.py",
    ]
    
    for rel_path in problematic_files:
        full_path = os.path.join(fairseq_path, rel_path)
        backup_path = full_path + ".bak"
        
        if os.path.exists(full_path) and not os.path.exists(backup_path):
            shutil.move(full_path, backup_path)
            with open(full_path, 'w') as f:
                f.write("# Disabled due to missing dependencies\n")
            print(f"Patched: {rel_path}")
    
    print("Fairseq patched successfully!")
else:
    print("Warning: Could not find fairseq installation to patch")
PATCH_FAIRSEQ

# Create necessary directories
mkdir -p assets/models

# Download required model assets
echo ""
echo "Downloading required model assets (HuBERT and RMVPE)..."
if [ -f "scripts/download_rvc_assets.sh" ]; then
    bash scripts/download_rvc_assets.sh
else
    echo "Warning: scripts/download_rvc_assets.sh not found, skipping asset download"
    echo "You can manually download assets later by running: bash scripts/download_rvc_assets.sh"
fi

# Download example model (BillCipher)
echo ""
echo "Downloading example model (BillCipher)..."
if [ -f "scripts/download_billcipher_model.sh" ]; then
    bash scripts/download_billcipher_model.sh || echo "⚠️  Skipped BillCipher model download (configure URLs in scripts/download_billcipher_model.sh)"
else
    echo "Warning: scripts/download_billcipher_model.sh not found"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Usage examples:"
echo "  API mode:       python main.py --mode api"
echo "  Streaming mode: python main.py --mode streaming"
echo "  Local mode:     python main.py --mode local --input input.wav --output output.wav"
echo ""
echo "For more information, see README.md"
