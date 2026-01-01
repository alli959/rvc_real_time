# BillCipher Model

This directory contains the BillCipher RVC model files.

## Required Files

- `BillCipher.pth` - Model weights (~54 MB)
- `BillCipher.index` - Retrieval index (~158 MB)

## Download

### Option 1: Automatic Download (after configuration)

1. Edit `scripts/download_billcipher_model.sh` and update the download URLs
2. Run: `bash scripts/download_billcipher_model.sh`

### Option 2: Manual Download

1. Download or train the BillCipher model
2. Place the files in this directory:
   - `assets/models/BillCipher/BillCipher.pth`
   - `assets/models/BillCipher/BillCipher.index`

## Hosting Your Model

If you want to share this model, upload it to one of these platforms:

- **Hugging Face** (recommended): https://huggingface.co
- **Google Drive**: Get a direct download link
- **Dropbox**: Use direct download links
- Your own CDN/server

Then update the URLs in `scripts/download_billcipher_model.sh`.

## Usage

```bash
python main.py --mode local \
  --model ./assets/models/BillCipher/BillCipher.pth \
  --index ./assets/models/BillCipher/BillCipher.index \
  --input ./input/input.flac \
  --output ./outputs/output.wav \
  --f0-method rmvpe \
  --f0-up-key 0 \
  --index-rate 0.75 \
  --chunk-size 65536
```
