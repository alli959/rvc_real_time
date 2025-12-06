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
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import AppConfig
from app.audio_stream import AudioStream
from app.feature_extraction import FeatureExtractor
from app.model_manager import ModelManager
from app.chunk_processor import StreamProcessor
from app.streaming_api import WebSocketServer, SocketServer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


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
    
    feature_extractor = FeatureExtractor(
        sample_rate=config.audio.sample_rate
    )
    
    model_manager = ModelManager(
        model_dir=config.model.model_dir
    )
    
    # Load default model if specified
    if config.model.default_model:
        model_manager.load_model(config.model.default_model)
    
    # Create stream processor
    stream_processor = StreamProcessor(
        model_manager=model_manager,
        feature_extractor=feature_extractor,
        chunk_size=config.audio.chunk_size,
        overlap=config.audio.overlap
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
        import time
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        audio_stream.stop()
        logger.info("Stopped.")


def run_api_mode(config: AppConfig):
    """
    Run in API mode with WebSocket and Socket servers
    
    Args:
        config: Application configuration
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting API mode...")
    
    # Initialize components
    feature_extractor = FeatureExtractor(
        sample_rate=config.audio.sample_rate
    )
    
    model_manager = ModelManager(
        model_dir=config.model.model_dir
    )
    
    # Load default model if specified
    if config.model.default_model:
        model_manager.load_model(config.model.default_model)
    
    # Create stream processor
    stream_processor = StreamProcessor(
        model_manager=model_manager,
        feature_extractor=feature_extractor,
        chunk_size=config.audio.chunk_size,
        overlap=config.audio.overlap
    )
    
    # Create servers
    websocket_server = WebSocketServer(
        host=config.server.websocket_host,
        port=config.server.websocket_port,
        stream_processor=stream_processor
    )
    
    socket_server = SocketServer(
        host=config.server.socket_host,
        port=config.server.socket_port,
        stream_processor=stream_processor
    )
    
    logger.info("Starting servers...")
    
    # Run both servers
    async def run_servers():
        await asyncio.gather(
            websocket_server.start(),
            socket_server.start()
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
    
    import librosa
    import soundfile as sf
    
    # Load audio
    audio, sr = librosa.load(input_file, sr=config.audio.sample_rate)
    
    # Initialize components
    feature_extractor = FeatureExtractor(
        sample_rate=config.audio.sample_rate
    )
    
    model_manager = ModelManager(
        model_dir=config.model.model_dir
    )
    
    # Load default model if specified
    if config.model.default_model:
        model_manager.load_model(config.model.default_model)
    
    # Create stream processor
    stream_processor = StreamProcessor(
        model_manager=model_manager,
        feature_extractor=feature_extractor,
        chunk_size=config.audio.chunk_size,
        overlap=config.audio.overlap
    )
    
    # Process in chunks
    output_chunks = []
    for i in range(0, len(audio), config.audio.chunk_size):
        chunk = audio[i:i + config.audio.chunk_size]
        
        # Pad last chunk if needed
        if len(chunk) < config.audio.chunk_size:
            chunk = librosa.util.fix_length(chunk, size=config.audio.chunk_size)
        
        processed = stream_processor.process_audio_chunk(chunk)
        if processed is not None:
            output_chunks.append(processed)
    
    # Concatenate output
    import numpy as np
    output_audio = np.concatenate(output_chunks) if output_chunks else audio
    
    # Save output
    sf.write(output_file, output_audio, config.audio.sample_rate)
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = AppConfig.from_env()
    
    # Override with command line arguments
    if args.model:
        config.model.default_model = args.model
    
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
