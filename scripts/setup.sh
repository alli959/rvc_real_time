#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-python3}
VENV=.venv

if [ ! -d "$VENV" ]; then
  $PY -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install -U pip wheel setuptools
pip install -r requirements.txt

mkdir -p assets/hubert assets/rmvpe

# Download hubert_base.pt if missing
if [ ! -f assets/hubert/hubert_base.pt ]; then
  echo "Downloading hubert_base.pt..."
  # put a real URL here (HF / official mirror you trust)
  curl -L -o assets/hubert/hubert_base.pt "<HUBERT_URL>"
fi

# Download rmvpe.pt if missing
if [ ! -f assets/rmvpe/rmvpe.pt ]; then
  echo "Downloading rmvpe.pt..."
  curl -L -o assets/rmvpe/rmvpe.pt "<RMVPE_URL>"
fi

echo ""
echo "Done."
echo "Run:"
echo "  source .venv/bin/activate"
echo "  python main.py --mode local --model <path_to_pth> --input <in> --output <out>"
