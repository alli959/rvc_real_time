# Sigurgeir Model

This directory contains the sigurgeir-0.5-model RVC v2 model files.

## Required Files

The model requires the trained generator checkpoint. Common files:
- `G_*_infer.pth` or `G_*.pth` - Generator model weights
- `*.index` - Retrieval index file (if available)
- `config.json` - Model configuration (included)

## Download

Place your model files in this directory or create a download script in `scripts/download_sigurgeir_model.sh` similar to the BillCipher example.

## Usage

```bash
# Find the inference model file (e.g., G_1360_infer.pth)
python main.py --mode local \
  --model ./assets/models/sigurgeir-0.5-model/G_1360_infer.pth \
  --index ./assets/models/sigurgeir-0.5-model/trained_*.index \
  --input ./input/input.flac \
  --output ./outputs/output.wav \
  --f0-method rmvpe
```
