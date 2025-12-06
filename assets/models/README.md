# RVC Real-Time Voice Conversion - Models Directory

Place your RVC model files here.

Supported formats:
- `.pth` - PyTorch checkpoint files
- `.pt` - PyTorch model files
- `.ckpt` - Checkpoint files

Example:
```
assets/models/
├── voice_model_1.pth
├── voice_model_2.pth
└── default_model.pt
```

To use a model, specify it when running the application:

```bash
python main.py --mode api --model voice_model_1.pth
```

Or set it as default in environment variables:

```bash
export DEFAULT_MODEL=voice_model_1.pth
python main.py --mode api
```
