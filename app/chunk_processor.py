"""
<<<<<<< HEAD
Chunk Processor Module - Handles chunk-based audio processing
"""

import numpy as np
from typing import Optional, Callable
from collections import deque
import logging
=======
Chunk Processor Module - Manages audio chunk processing for real-time conversion

This implementation feeds raw waveform chunks into the RVC inference pipeline
(via ModelManager).

Notes
- Input and output are mono float32 arrays in the range [-1, 1].
- The RVC pipeline may produce an output length slightly different from the input
  (due to resampling). We pad/trim to match the input chunk length so the
  streaming audio callback stays happy.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .model_manager import ModelManager, RVCInferParams
>>>>>>> 2c01fa3 (feat(rvc): run WebUI-trained RVC models in realtime pipeline)

logger = logging.getLogger(__name__)


<<<<<<< HEAD
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
            # No overlap, return the full processed chunk
            output = processed
        
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
=======
class StreamProcessor:
    """Processes audio chunks for real-time conversion."""

    def __init__(
        self,
        model_manager: ModelManager,
        feature_extractor=None,  # kept for backward compatibility
        chunk_size: int = 1024,
        overlap: int = 0,
        output_gain: float = 1.0,
        infer_params: Optional[RVCInferParams] = None,
    ):
        self.model_manager = model_manager
        self.feature_extractor = feature_extractor
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.output_gain = float(output_gain)
        self.infer_params = infer_params

        # For optional overlap crossfading
        self._prev_tail: Optional[np.ndarray] = None

        if self.overlap not in (0,):
            logger.warning(
                "overlap is currently not used for waveform processing; set AUDIO_OVERLAP=0 for lowest latency"
            )

    def _fit_length(self, y: np.ndarray, n: int) -> np.ndarray:
        """Pad/trim output to match expected chunk length."""
        y = np.asarray(y, dtype=np.float32).flatten()
        if y.size == n:
            return y
        if y.size > n:
            return y[:n]
        # pad
        return np.pad(y, (0, n - y.size), mode="constant")

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process a single audio chunk."""
        if audio_chunk is None:
            return None

        x = np.asarray(audio_chunk, dtype=np.float32).flatten()
        if x.size == 0:
            return x

        # Normalize to a safe range
        max_val = np.max(np.abs(x))
        if max_val > 1.0:
            x = x / max_val

        # Convert
        y = self.model_manager.infer(x, params=self.infer_params)

        # Match length for streaming
        y = self._fit_length(y, x.size)

        # Optional crossfade between chunks (helps reduce boundary artifacts).
        if self.overlap > 0 and y.size > 0:
            if self._prev_tail is None:
                # First chunk: just stash the tail.
                self._prev_tail = y[-self.overlap :].copy() if y.size >= self.overlap else y.copy()
            else:
                n = int(min(self.overlap, y.size, self._prev_tail.size))
                if n > 0:
                    fade_in = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
                    fade_out = 1.0 - fade_in
                    y[:n] = self._prev_tail[-n:] * fade_out + y[:n] * fade_in
                self._prev_tail = y[-self.overlap :].copy() if y.size >= self.overlap else y.copy()

        # Apply gain and clip
        y = np.clip(y * self.output_gain, -1.0, 1.0)
        return y

    def reset(self):
        """Reset internal state."""
        self._prev_tail = None
>>>>>>> 2c01fa3 (feat(rvc): run WebUI-trained RVC models in realtime pipeline)
