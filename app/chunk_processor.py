"""
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

logger = logging.getLogger(__name__)


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
