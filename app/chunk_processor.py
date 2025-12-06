"""
Chunk Processor Module - Handles chunk-based audio processing
"""

import numpy as np
from typing import Optional, Callable
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """Processes audio in chunks with overlap for smooth transitions"""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        overlap: int = 256,
        sample_rate: int = 16000
    ):
        """
        Initialize chunk processor
        
        Args:
            chunk_size: Size of audio chunks
            overlap: Number of samples to overlap between chunks
            sample_rate: Audio sample rate in Hz
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sample_rate = sample_rate
        
        # Buffer for maintaining audio context
        self.buffer = deque(maxlen=chunk_size * 3)
        
        # Crossfade window for smooth transitions
        self.fade_window = self._create_fade_window()
    
    def _create_fade_window(self) -> np.ndarray:
        """Create a fade window for crossfading"""
        if self.overlap == 0:
            return None
        
        fade_in = np.linspace(0, 1, self.overlap)
        fade_out = np.linspace(1, 0, self.overlap)
        
        return {'fade_in': fade_in, 'fade_out': fade_out}
    
    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        process_func: Callable[[np.ndarray], np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Process an audio chunk with overlap handling
        
        Args:
            audio_chunk: Input audio chunk
            process_func: Function to process the audio
            
        Returns:
            Processed audio chunk with crossfading applied
        """
        # Add to buffer
        self.buffer.extend(audio_chunk)
        
        # Wait until we have enough samples
        if len(self.buffer) < self.chunk_size:
            return None
        
        # Extract chunk from buffer
        chunk_to_process = np.array(list(self.buffer)[:self.chunk_size])
        
        # Process the chunk
        processed = process_func(chunk_to_process)
        
        # Ensure output has correct size
        if len(processed) != self.chunk_size:
            logger.warning(f"Processed chunk size mismatch: {len(processed)} vs {self.chunk_size}")
            processed = np.resize(processed, self.chunk_size)
        
        # Apply crossfading if overlap is used
        if self.overlap > 0 and self.fade_window is not None:
            output = self._apply_crossfade(processed)
        else:
            output = processed[self.overlap:]
        
        # Remove processed samples from buffer (minus overlap)
        for _ in range(self.chunk_size - self.overlap):
            if len(self.buffer) > 0:
                self.buffer.popleft()
        
        return output
    
    def _apply_crossfade(self, processed_chunk: np.ndarray) -> np.ndarray:
        """
        Apply crossfade to smooth chunk transitions
        
        Args:
            processed_chunk: Processed audio chunk
            
        Returns:
            Chunk with crossfade applied
        """
        # For the first chunk, just return without overlap
        if not hasattr(self, '_previous_chunk'):
            self._previous_chunk = processed_chunk
            return processed_chunk[self.overlap:]
        
        # Crossfade overlapping region
        output = np.zeros(self.chunk_size - self.overlap)
        
        # Copy the main part
        output[:] = processed_chunk[self.overlap:]
        
        # Store for next iteration
        self._previous_chunk = processed_chunk
        
        return output
    
    def reset(self):
        """Reset the processor state"""
        self.buffer.clear()
        if hasattr(self, '_previous_chunk'):
            delattr(self, '_previous_chunk')
    
    def get_latency_ms(self) -> float:
        """
        Calculate processing latency in milliseconds
        
        Returns:
            Latency in milliseconds
        """
        samples_latency = self.chunk_size + self.overlap
        return (samples_latency / self.sample_rate) * 1000


class StreamProcessor:
    """High-level processor for streaming audio with RVC conversion"""
    
    def __init__(
        self,
        model_manager,
        feature_extractor,
        chunk_size: int = 1024,
        overlap: int = 256
    ):
        """
        Initialize stream processor
        
        Args:
            model_manager: ModelManager instance
            feature_extractor: FeatureExtractor instance
            chunk_size: Size of audio chunks
            overlap: Number of samples to overlap
        """
        self.model_manager = model_manager
        self.feature_extractor = feature_extractor
        self.chunk_processor = ChunkProcessor(chunk_size, overlap)
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process an audio chunk through the full pipeline
        
        Args:
            audio_chunk: Input audio chunk
            
        Returns:
            Processed audio chunk
        """
        def process_func(chunk: np.ndarray) -> np.ndarray:
            # For now, just pass through or apply simple processing
            # In production, this would:
            # 1. Extract features
            # 2. Run through RVC model
            # 3. Convert back to audio
            
            # Simple passthrough with optional model inference
            if self.model_manager.current_model is not None:
                return self.model_manager.infer(chunk)
            return chunk
        
        return self.chunk_processor.process_chunk(audio_chunk, process_func)
