"""
Real-time Voice Conversion Application - Main Entry Point

This is a real-time voice conversion application using RVC models.
It supports multiple modes:
  - streaming: Real-time audio I/O with local processing
  - api: WebSocket/Socket server for remote clients
  - local: Process audio files
"""

import argparse
import logging
import sys
import asyncio
import time
import numpy as np
from math import gcd
from scipy import signal
from pathlib import Path
import librosa

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from app.config import AppConfig
from app.audio_stream import AudioStream
from app.model_manager import ModelManager, RVCInferParams
from app.chunk_processor import StreamProcessor
from app.streaming_api import WebSocketServer, SocketServer

import soundfile as sf



def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def _resample_poly(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float waveform using polyphase filtering."""
    y = np.asarray(y, dtype=np.float32).flatten()
    if orig_sr == target_sr or y.size == 0:
        return y
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return signal.resample_poly(y.astype(np.float64), up, down).astype(np.float32)



def run_streaming_mode(config: AppConfig):
    """
    Run in streaming mode with real-time audio I/O
    
    Args:
        config: Application configuration
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting streaming mode...")
    
    # Initialize components
    audio_stream = AudioStream(
        sample_rate=config.audio.sample_rate,
        chunk_size=config.audio.chunk_size,
        channels=config.audio.channels
    )
    
    model_manager = ModelManager(
        model_dir=config.model.model_dir,
        index_dir=config.model.index_dir,
        hubert_path=config.model.hubert_path,
        rmvpe_dir=config.model.rmvpe_dir,
        input_sample_rate=config.audio.sample_rate,
        device=config.model.device,
    )

    infer_params = RVCInferParams(
        sid=0,
        f0_up_key=config.model.f0_up_key,
        f0_method=config.model.f0_method,
        index_rate=config.model.index_rate,
        filter_radius=config.model.filter_radius,
        rms_mix_rate=config.model.rms_mix_rate,
        protect=config.model.protect,
        resample_sr=config.model.resample_sr,
    )

    
    # Model warmup: preload HuBERT, RMVPE, and default RVC model (if configured)
    import time
    warmup_start = time.time()
    warmup_success = False
    if config.model.default_model:
        logger.info(f"Warming up model: {config.model.default_model}")
        warmup_success = model_manager.load_model(config.model.default_model, index_path=config.model.default_index)
        # Force HuBERT and RMVPE load by running a dummy inference
        dummy_audio = np.zeros(config.audio.chunk_size, dtype=np.float32)
        try:
            _ = model_manager.infer(dummy_audio, params=infer_params)
            logger.info("Model warmup completed successfully.")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
    else:
        logger.info("No default model configured for warmup.")
    logger.info(f"Model warmup time: {time.time() - warmup_start:.2f} seconds. Success: {warmup_success}")
    
    # NOTE: Bark TTS models are now lazy-loaded on first TTS request
    # Set PRELOAD_BARK=1 to preload at startup (uses ~1.5GB extra memory)
    import os
    if os.environ.get("PRELOAD_BARK", "0") == "1":
        try:
            from app.tts_service import setup_bark_cache, BARK_AVAILABLE
            if BARK_AVAILABLE:
                from bark.generation import preload_models
                logger.info("Preloading Bark TTS models (PRELOAD_BARK=1)...")
                bark_start = time.time()
                preload_models()
                logger.info(f"Bark TTS models loaded in {time.time() - bark_start:.2f} seconds")
        except ImportError as e:
            logger.info(f"Bark TTS not installed - skipping preload: {e}")
        except Exception as e:
            logger.warning(f"Failed to preload Bark models: {e}")
    else:
        logger.info("Bark TTS will lazy-load on first use (saves ~1.5GB startup memory)")
    
    # Create stream processor
    stream_processor = StreamProcessor(
        model_manager=model_manager,
        chunk_size=config.audio.chunk_size,
        output_gain=getattr(config.model, "output_gain", 1.0),
        infer_params=infer_params,
    )
    
    # Start audio streams
    logger.info("Starting audio streams...")
    audio_stream.start_input_stream()
    audio_stream.start_output_stream()
    
    # Start processing
    audio_stream.start_processing(stream_processor.process_audio_chunk)
    
    logger.info("Streaming mode active. Press Ctrl+C to stop.")
    
    try:
        # Keep running
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        audio_stream.stop()
        logger.info("Stopped.")


def run_api_mode(config: AppConfig):
    """
    Run in API mode with WebSocket, Socket, and HTTP servers
    
    Args:
        config: Application configuration
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting API mode...")
    
    # Initialize components
    
    model_manager = ModelManager(
        model_dir=config.model.model_dir,
        index_dir=config.model.index_dir,
        hubert_path=config.model.hubert_path,
        rmvpe_dir=config.model.rmvpe_dir,
        input_sample_rate=config.audio.sample_rate,
        device=config.model.device,
    )

    infer_params = RVCInferParams(
        sid=0,
        f0_up_key=config.model.f0_up_key,
        f0_method=config.model.f0_method,
        index_rate=config.model.index_rate,
        filter_radius=config.model.filter_radius,
        rms_mix_rate=config.model.rms_mix_rate,
        protect=config.model.protect,
        resample_sr=config.model.resample_sr,
    )

    
    # Model warmup: preload HuBERT, RMVPE, and default RVC model (if configured)
    import time
    warmup_start = time.time()
    warmup_success = False
    if config.model.default_model:
        logger.info(f"Warming up model: {config.model.default_model}")
        warmup_success = model_manager.load_model(config.model.default_model, index_path=config.model.default_index)
        # Force HuBERT and RMVPE load by running a dummy inference
        dummy_audio = np.zeros(config.audio.chunk_size, dtype=np.float32)
        try:
            _ = model_manager.infer(dummy_audio, params=infer_params)
            logger.info("Model warmup completed successfully.")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
    else:
        logger.info("No default model configured for warmup.")
    logger.info(f"Model warmup time: {time.time() - warmup_start:.2f} seconds. Success: {warmup_success}")
    
    # NOTE: Bark TTS models are now lazy-loaded on first TTS request
    # This saves ~1.5GB of memory at startup. The first TTS request will be slower (~18s)
    # but subsequent requests will be fast since models stay loaded until cache eviction.
    # To preload Bark (trade startup memory for faster first TTS), set PRELOAD_BARK=1
    import os
    if os.environ.get("PRELOAD_BARK", "0") == "1":
        try:
            from app.tts_service import setup_bark_cache, BARK_AVAILABLE
            if BARK_AVAILABLE:
                from bark.generation import preload_models
                logger.info("Preloading Bark TTS models (PRELOAD_BARK=1)...")
                bark_start = time.time()
                preload_models()
                logger.info(f"Bark TTS models loaded in {time.time() - bark_start:.2f} seconds")
        except ImportError as e:
            logger.info(f"Bark TTS not installed - skipping preload: {e}")
        except Exception as e:
            logger.warning(f"Failed to preload Bark models: {e}")
    else:
        logger.info("Bark TTS will lazy-load on first use (saves ~1.5GB startup memory)")
    
    # Create stream processor
    stream_processor = StreamProcessor(
        model_manager=model_manager,
        chunk_size=config.audio.chunk_size,
        output_gain=getattr(config.model, "output_gain", 1.0),
        infer_params=infer_params,
    )
    
    # Create servers
    websocket_server = WebSocketServer(
        host=config.server.websocket_host,
        port=config.server.websocket_port,
        stream_processor=stream_processor,
        model_manager=model_manager,
        infer_params=infer_params  # For batch processing mode (better quality)
    )
    
    socket_server = SocketServer(
        host=config.server.socket_host,
        port=config.server.socket_port,
        stream_processor=stream_processor
    )
    
    # Initialize HTTP API with model manager
    from app.http_api import set_model_manager
    set_model_manager(model_manager, infer_params)
    
    logger.info("Starting servers...")
    
    # Run all servers (WebSocket, Socket, and HTTP)
    async def run_servers():
        import uvicorn
        from app.http_api import app as http_app
        
        # Create uvicorn config for HTTP server
        http_config = uvicorn.Config(
            http_app,
            host=config.server.http_host,
            port=config.server.http_port,
            log_level="info"
        )
        http_server = uvicorn.Server(http_config)
        
        logger.info(f"Starting HTTP server on {config.server.http_host}:{config.server.http_port}")
        
        await asyncio.gather(
            websocket_server.start(),
            socket_server.start(),
            http_server.serve()
        )
    
    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        logger.info("Stopping servers...")


def run_local_mode(config: AppConfig, input_file: str, output_file: str):
    """
    Run in local mode to process audio files
    
    Args:
        config: Application configuration
        input_file: Input audio file path
        output_file: Output audio file path
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {input_file} -> {output_file}")
    
    # Load audio (preserve original sample rate)
    audio, sr = sf.read(input_file, dtype='float32', always_2d=False)
    if audio is None:
        logger.error('Failed to load input audio')
        sys.exit(1)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        # stereo -> mono
        audio = np.mean(audio, axis=1).astype(np.float32)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    
    # Initialize components
    
    model_manager = ModelManager(
        model_dir=config.model.model_dir,
        index_dir=config.model.index_dir,
        hubert_path=config.model.hubert_path,
        rmvpe_dir=config.model.rmvpe_dir,
        input_sample_rate=sr,
        device=config.model.device,
    )

    infer_params = RVCInferParams(
        sid=0,
        f0_up_key=config.model.f0_up_key,
        f0_method=config.model.f0_method,
        index_rate=config.model.index_rate,
        filter_radius=config.model.filter_radius,
        rms_mix_rate=config.model.rms_mix_rate,
        protect=config.model.protect,
        resample_sr=config.model.resample_sr,
    )
    infer_params.resample_sr = int(sr)  # keep file sample rate

    
    # Load default model if specified
    if config.model.default_model:
        model_manager.load_model(config.model.default_model, index_path=config.model.default_index)
    
    # For local file processing, process the ENTIRE audio at once (not chunked)
    # Chunked processing is only needed for real-time streaming
    # The RVC pipeline needs large audio segments for proper pitch/feature extraction
    output_gain = getattr(config.model, "output_gain", 1.0)
    
    # Normalize input
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    
    # Process entire audio in one pass
    output_audio = model_manager.infer(audio, params=infer_params)
    
    # Apply output gain and clip
    output_audio = np.clip(output_audio * output_gain, -1.0, 1.0)
    
    # Save output
    sf.write(output_file, output_audio, int(sr))
    logger.info(f"Saved processed audio to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Real-time Voice Conversion Application"
    )
    
    parser.add_argument(
        '--mode',
        choices=['streaming', 'api', 'local'],
        default='api',
        help='Application mode (default: api)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model file to load'
    )

    parser.add_argument(
        '--index',
        type=str,
        help='Optional .index file to use for retrieval enhancement'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input audio file (local mode only)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output audio file (local mode only)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--websocket-port',
        type=int,
        default=8765,
        help='WebSocket server port (default: 8765)'
    )
    
    parser.add_argument(
        '--socket-port',
        type=int,
        default=9876,
        help='Socket server port (default: 9876)'
    )
    
    parser.add_argument("--f0-method", type=str, default=None, help="F0 method (e.g. rmvpe, dio, harvest)")
    parser.add_argument("--f0-up-key", type=int, default=None, help="Pitch shift in semitones (e.g. -12..+12)")
    parser.add_argument("--index-rate", type=float, default=None, help="Index blend (0..1)")
    parser.add_argument("--protect", type=float, default=None, help="Protect (0..1)")
    parser.add_argument("--rms-mix-rate", type=float, default=None, help="RMS mix rate (0..1)")
    parser.add_argument("--filter-radius", type=int, default=None, help="Filter radius")
    parser.add_argument("--resample-sr", type=int, default=None, help="Output resample sr (0=auto)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for processing")
    parser.add_argument("--output-gain", type=float, default=None, help="Output gain multiplier")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = AppConfig.from_env()
    
    # Override with command line arguments
    if args.model:
        config.model.default_model = args.model
    if args.index:
        config.model.default_index = args.index
        
    if args.f0_method is not None:
        config.model.f0_method = args.f0_method
    if args.f0_up_key is not None:
        config.model.f0_up_key = args.f0_up_key
    if args.index_rate is not None:
        config.model.index_rate = args.index_rate
    if args.protect is not None:
        config.model.protect = args.protect
    if args.rms_mix_rate is not None:
        config.model.rms_mix_rate = args.rms_mix_rate
    if args.filter_radius is not None:
        config.model.filter_radius = args.filter_radius
    if args.resample_sr is not None:
        config.model.resample_sr = args.resample_sr

    if args.chunk_size is not None:
        config.audio.chunk_size = args.chunk_size
    if args.output_gain is not None:
        config.model.output_gain = args.output_gain
    
    config.mode = args.mode
    config.log_level = args.log_level
    config.server.websocket_port = args.websocket_port
    config.server.socket_port = args.socket_port
    
    logger.info(f"Starting RVC Real-time Voice Conversion in {config.mode} mode")
    
    # Run appropriate mode
    if config.mode == 'streaming':
        run_streaming_mode(config)
    elif config.mode == 'api':
        run_api_mode(config)
    elif config.mode == 'local':
        if not args.input or not args.output:
            logger.error("Local mode requires --input and --output arguments")
            sys.exit(1)
        run_local_mode(config, args.input, args.output)
    else:
        logger.error(f"Unknown mode: {config.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
