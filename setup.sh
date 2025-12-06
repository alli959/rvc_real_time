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
pip install -r requirements.txt

# Create necessary directories
mkdir -p assets/models

echo ""
echo "Setup complete!"
echo ""
echo "Usage examples:"
echo "  API mode:       python main.py --mode api"
echo "  Streaming mode: python main.py --mode streaming"
echo "  Local mode:     python main.py --mode local --input input.wav --output output.wav"
echo ""
echo "For more information, see README.md"
