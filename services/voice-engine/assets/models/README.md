# RVC Real-Time Voice Conversion - Models Directory

Place your RVC model files here. Each model should be in its own subdirectory.

## Supported Formats

- `.pth` - PyTorch checkpoint files (WebUI RVC v1/v2 models)
- `.index` - FAISS retrieval index files (optional, improves quality)
- `config.json` - Model configuration (auto-generated if missing)

## Recommended Structure

```
assets/models/
├── ModelName/
│   ├── ModelName.pth          # Required: Model weights
│   ├── ModelName.index        # Optional: Retrieval index
│   └── config.json            # Optional: Model config
├── AnotherModel/
│   ├── AnotherModel.pth
│   └── AnotherModel.index
└── README.md
```

## Usage

Specify a model when running the application:

```bash
# By folder name (recommended)
python main.py --mode api --model ./assets/models/BillCipher/BillCipher.pth \
               --index ./assets/models/BillCipher/BillCipher.index

# Using interactive menu
./start-api.sh
```

Or set as default in `.env`:

```bash
DEFAULT_MODEL=./assets/models/BillCipher/BillCipher.pth
DEFAULT_INDEX=./assets/models/BillCipher/BillCipher.index
```

## Downloading Models

You can find RVC models on:
- [Hugging Face](https://huggingface.co) - Search for "RVC model"
- [Weights.gg](https://weights.gg) - Community voice models
- Train your own using [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
