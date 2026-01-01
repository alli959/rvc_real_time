# Model Management Guide

This project excludes large model files from git to avoid GitHub's file size limits. Instead, models are downloaded via scripts or manually placed.

## Quick Start

Run the setup script to download required assets:

```bash
bash setup.sh
```

This will:
1. Set up Python virtual environment
2. Install dependencies  
3. Download HuBERT and RMVPE models (required)
4. Attempt to download example models (if configured)

## Required Assets (Auto-downloaded)

These are downloaded automatically by `setup.sh`:

- **HuBERT** (`assets/hubert/hubert_base.pt`) - Content encoder for voice conversion
- **RMVPE** (`assets/rmvpe/rmvpe.pt`) - Pitch extraction model

## Example Models

### BillCipher Model

To use the BillCipher example:

**Option 1: Host and auto-download**
1. Upload `BillCipher.pth` and `BillCipher.index` to Hugging Face, Google Drive, or similar
2. Edit `scripts/download_billcipher_model.sh` and update the URLs
3. Run: `bash scripts/download_billcipher_model.sh`

**Option 2: Manual placement**
1. Place your model files in `assets/models/BillCipher/`:
   - `BillCipher.pth` 
   - `BillCipher.index`

### Your Own Models

To add your own RVC models:

1. Create a directory: `assets/models/YourModelName/`
2. Place your files:
   - `YourModel.pth` - Model weights
   - `YourModel.index` - Retrieval index (optional but recommended)
   - `config.json` - Model config (optional, will use defaults if missing)
3. Use in commands:
   ```bash
   python main.py --mode local \
     --model ./assets/models/YourModelName/YourModel.pth \
     --index ./assets/models/YourModelName/YourModel.index \
     --input ./input/input.wav \
     --output ./outputs/output.wav
   ```

## Hosting Your Models

If you want to share models with others, recommended platforms:

- **Hugging Face** - Best for ML models, free, permanent hosting
  1. Create account at https://huggingface.co
  2. Create a model repository
  3. Upload your .pth and .index files
  4. Get direct download links (Format: `https://huggingface.co/username/repo/resolve/main/filename`)

- **Google Drive** - Easy but requires conversion to direct links
  1. Upload files
  2. Get shareable link
  3. Convert to direct download: Change `/file/d/FILE_ID/view` to `/uc?export=download&id=FILE_ID`

## File Size Reference

Typical RVC model sizes:
- `.pth` files: 50-200 MB (model weights)
- `.index` files: 50-500 MB (retrieval index)
- `hubert_base.pt`: ~180 MB
- `rmvpe.pt`: ~173 MB

GitHub limits:
- Warning at 50 MB
- Hard limit at 100 MB
- Use Git LFS or external hosting for larger files
