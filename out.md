diff /home/alexanderg/rvc_real_time/.env.example ./.env.example
1c1,7
< # Audio Configuration
---
> # Real-time Voice Conversion Configuration
> 
> # Application mode: api, streaming, local
> APP_MODE=api
> LOG_LEVEL=INFO
> 
> # Audio configuration
4c10
< AUDIO_OVERLAP=256
---
> AUDIO_OVERLAP=0
7c13,18
< # Model Configuration
---
> # Model/asset paths
> # Required auxiliary assets:
> #  - HuBERT checkpoint: assets/hubert/hubert_base.pt
> #  - RMVPE checkpoint: assets/rmvpe/rmvpe.pt
> # Optional:
> #  - Retrieval index: assets/index/<model>.index
8a20,24
> INDEX_DIR=assets/index
> HUBERT_PATH=assets/hubert/hubert_base.pt
> RMVPE_DIR=assets/rmvpe
> 
> # Default model and (optional) matching .index
9a26,38
> # DEFAULT_INDEX=assets/index/your_model.index
> 
> # RVC inference defaults (these match typical RVC v2 f0=True models)
> F0_METHOD=rmvpe
> F0_UP_KEY=0
> INDEX_RATE=0.75
> FILTER_RADIUS=3
> RMS_MIX_RATE=0.25
> PROTECT=0.33
> # Output sample rate. Use 16000 to match the demo client in README.
> RESAMPLE_SR=16000
> 
> # Device: auto, cpu, cuda
12c41
< # Server Configuration
---
> # Server configuration
17,20d45
< 
< # Application Configuration
< APP_MODE=api
< LOG_LEVEL=INFO
Only in /home/alexanderg/rvc_real_time/: .git
diff /home/alexanderg/rvc_real_time/README.md ./README.md
51a52,59
> To run inference with WebUI-trained models you also need these assets:
> 
> - **HuBERT**: place `hubert_base.pt` at `assets/hubert/hubert_base.pt`
> - **RMVPE** (for f0/pitch): place `rmvpe.pt` at `assets/rmvpe/rmvpe.pt`
> - **Index** (optional but recommended): place the model's `.index` file in `assets/index/`
> 
> You can configure these paths in `.env` (see `.env.example`).
> 
82c90
< python main.py --mode streaming --model your_model.pth
---
> python main.py --mode streaming --model your_model.pth --index assets/index/your_model.index
97a106
> --index INDEX                 Optional .index file
115c124
< AUDIO_OVERLAP=256
---
> AUDIO_OVERLAP=0
118c127
< # Model Configuration
---
> # Model/asset paths
119a129,133
> INDEX_DIR=assets/index
> HUBERT_PATH=assets/hubert/hubert_base.pt
> RMVPE_DIR=assets/rmvpe
> 
> # Defaults
120a135,146
> # DEFAULT_INDEX=assets/index/your_model.index
> 
> # RVC inference defaults
> F0_METHOD=rmvpe
> F0_UP_KEY=0
> INDEX_RATE=0.75
> FILTER_RADIUS=3
> RMS_MIX_RATE=0.25
> PROTECT=0.33
> RESAMPLE_SR=16000
> 
> # Device
Common subdirectories: /home/alexanderg/rvc_real_time/app and ./app
Common subdirectories: /home/alexanderg/rvc_real_time/assets and ./assets
Common subdirectories: /home/alexanderg/rvc_real_time/examples and ./examples
Only in /home/alexanderg/rvc_real_time/: input
diff /home/alexanderg/rvc_real_time/main.py ./main.py
16a17,18
> from math import gcd
> from scipy import signal
21a24,30
> # Load .env if present
> try:
>     from dotenv import load_dotenv
>     load_dotenv()
> except Exception:
>     pass
> 
24,25c33
< from app.feature_extraction import FeatureExtractor
< from app.model_manager import ModelManager
---
> from app.model_manager import ModelManager, RVCInferParams
29,35c37,38
< # Import librosa and soundfile for local mode
< try:
<     import librosa
<     import soundfile as sf
<     LIBROSA_AVAILABLE = True
< except ImportError:
<     LIBROSA_AVAILABLE = False
---
> import soundfile as sf
> 
47a51,61
> def _resample_poly(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
>     """Resample 1D float waveform using polyphase filtering."""
>     y = np.asarray(y, dtype=np.float32).flatten()
>     if orig_sr == target_sr or y.size == 0:
>         return y
>     g = gcd(orig_sr, target_sr)
>     up = target_sr // g
>     down = orig_sr // g
>     return signal.resample_poly(y.astype(np.float64), up, down).astype(np.float32)
> 
> 
66,69d79
<     feature_extractor = FeatureExtractor(
<         sample_rate=config.audio.sample_rate
<     )
<     
71c81,97
<         model_dir=config.model.model_dir
---
>         model_dir=config.model.model_dir,
>         index_dir=config.model.index_dir,
>         hubert_path=config.model.hubert_path,
>         rmvpe_dir=config.model.rmvpe_dir,
>         input_sample_rate=config.audio.sample_rate,
>         device=config.model.device,
>     )
> 
>     infer_params = RVCInferParams(
>         sid=0,
>         f0_up_key=config.model.f0_up_key,
>         f0_method=config.model.f0_method,
>         index_rate=config.model.index_rate,
>         filter_radius=config.model.filter_radius,
>         rms_mix_rate=config.model.rms_mix_rate,
>         protect=config.model.protect,
>         resample_sr=config.model.resample_sr,
72a99
> 
76c103
<         model_manager.load_model(config.model.default_model)
---
>         model_manager.load_model(config.model.default_model, index_path=config.model.default_index)
81d107
<         feature_extractor=feature_extractor,
83c109,110
<         overlap=config.audio.overlap
---
>         output_gain=config.model.output_gain,
>         infer_params=infer_params,
118,120d144
<     feature_extractor = FeatureExtractor(
<         sample_rate=config.audio.sample_rate
<     )
123c147,163
<         model_dir=config.model.model_dir
---
>         model_dir=config.model.model_dir,
>         index_dir=config.model.index_dir,
>         hubert_path=config.model.hubert_path,
>         rmvpe_dir=config.model.rmvpe_dir,
>         input_sample_rate=config.audio.sample_rate,
>         device=config.model.device,
>     )
> 
>     infer_params = RVCInferParams(
>         sid=0,
>         f0_up_key=config.model.f0_up_key,
>         f0_method=config.model.f0_method,
>         index_rate=config.model.index_rate,
>         filter_radius=config.model.filter_radius,
>         rms_mix_rate=config.model.rms_mix_rate,
>         protect=config.model.protect,
>         resample_sr=config.model.resample_sr,
124a165
> 
128c169
<         model_manager.load_model(config.model.default_model)
---
>         model_manager.load_model(config.model.default_model, index_path=config.model.default_index)
133d173
<         feature_extractor=feature_extractor,
135c175,176
<         overlap=config.audio.overlap
---
>         output_gain=config.model.output_gain,
>         infer_params=infer_params,
178,179c219,222
<     if not LIBROSA_AVAILABLE:
<         logger.error("librosa and soundfile are required for local mode")
---
>     # Load audio (preserve original sample rate)
>     audio, sr = sf.read(input_file, dtype='float32', always_2d=False)
>     if audio is None:
>         logger.error('Failed to load input audio')
181,183c224,227
<     
<     # Load audio
<     audio, sr = librosa.load(input_file, sr=config.audio.sample_rate)
---
>     if isinstance(audio, np.ndarray) and audio.ndim == 2:
>         # stereo -> mono
>         audio = np.mean(audio, axis=1).astype(np.float32)
>     audio = np.asarray(audio, dtype=np.float32).flatten()
186,188d229
<     feature_extractor = FeatureExtractor(
<         sample_rate=config.audio.sample_rate
<     )
191c232,248
<         model_dir=config.model.model_dir
---
>         model_dir=config.model.model_dir,
>         index_dir=config.model.index_dir,
>         hubert_path=config.model.hubert_path,
>         rmvpe_dir=config.model.rmvpe_dir,
>         input_sample_rate=sr,
>         device=config.model.device,
>     )
> 
>     infer_params = RVCInferParams(
>         sid=0,
>         f0_up_key=config.model.f0_up_key,
>         f0_method=config.model.f0_method,
>         index_rate=config.model.index_rate,
>         filter_radius=config.model.filter_radius,
>         rms_mix_rate=config.model.rms_mix_rate,
>         protect=config.model.protect,
>         resample_sr=config.model.resample_sr,
192a250,251
>     infer_params.resample_sr = int(sr)  # keep file sample rate
> 
196c255
<         model_manager.load_model(config.model.default_model)
---
>         model_manager.load_model(config.model.default_model, index_path=config.model.default_index)
201d259
<         feature_extractor=feature_extractor,
203c261,262
<         overlap=config.audio.overlap
---
>         output_gain=config.model.output_gain,
>         infer_params=infer_params,
223c282
<     sf.write(output_file, output_audio, config.audio.sample_rate)
---
>     sf.write(output_file, output_audio, int(sr))
244a304,309
> 
>     parser.add_argument(
>         '--index',
>         type=str,
>         help='Optional .index file to use for retrieval enhancement'
>     )
290a356,357
>     if args.index:
>         config.model.default_index = args.index
315c382
<     main()
---
>     main()
\ No newline at end of file
Only in .: out.md
Only in /home/alexanderg/rvc_real_time/: outputs
diff /home/alexanderg/rvc_real_time/requirements.txt ./requirements.txt
1,2c1,3
< # Core dependencies
< numpy>=1.21.0,<2.0.0
---
> # Core
> numpy>=1.23.0
> scipy>=1.10.0
4,8c5,6
< # PyTorch - CPU version for smaller size and better compatibility
< torch>=2.0.0,<2.2.0
< torchaudio>=2.0.0,<2.2.0
< 
< # Audio processing
---
> # Audio I/O
> soundfile>=0.12.0
10,15d7
< librosa>=0.10.0,<0.11.0
< soundfile>=0.12.1
< 
< # Streaming API
< websockets>=11.0,<12.0
< aiohttp>=3.8.0,<4.0.0
17c9,10
< # Utilities
---
> # Web / streaming
> websockets>=10.0
18a12,23
> 
> # RVC dependencies
> fairseq>=0.12.2
> faiss-cpu>=1.7.4
> 
> # NOTE: PyTorch is required but is NOT pinned here (install the correct build for your system):
> #   https://pytorch.org/get-started/locally/
> #
> # Optional (file formats / extra f0 methods):
> #   av (PyAV)  -> enables reading more formats via ffmpeg in some RVC utilities
> #   pyworld    -> enables f0_method=harvest
> #   torchcrepe -> enables f0_method=crepe
Only in .: rvc
Only in .: scripts
Common subdirectories: /home/alexanderg/rvc_real_time/tests and ./tests
Only in /home/alexanderg/rvc_real_time/: venv
