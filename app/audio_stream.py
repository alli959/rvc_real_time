"""
Audio Streaming Module - Handles real-time audio input/output
"""

import numpy as np
import pyaudio
import queue
import threading
from typing import Optional, Callable


class AudioStream:
    """Manages streaming audio input and output with chunk-based processing"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        format: int = pyaudio.paFloat32
    ):
        """
        Initialize audio stream
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Size of audio chunks for processing
            channels: Number of audio channels (1 for mono, 2 for stereo)
            format: PyAudio format
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        
        self.audio = pyaudio.PyAudio()
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
    
    def start_input_stream(self):
        """Start audio input stream"""
        self.input_stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._input_callback
        )
        self.input_stream.start_stream()
    
    def start_output_stream(self):
        """Start audio output stream"""
        self.output_stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._output_callback
        )
        self.output_stream.start_stream()
    
    def _input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.input_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def _output_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio output stream"""
        try:
            audio_data = self.output_queue.get_nowait()
<<<<<<< HEAD
            return (audio_data.tobytes(), pyaudio.paContinue)
        except queue.Empty:
            # Return silence if no data available
            return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)
=======
            audio_data = np.asarray(audio_data, dtype=np.float32).flatten()

            # PyAudio expects exactly frame_count frames. Pad/trim as needed.
            expected = frame_count * self.channels
            if audio_data.size < expected:
                audio_data = np.pad(audio_data, (0, expected - audio_data.size), mode="constant")
            elif audio_data.size > expected:
                audio_data = audio_data[:expected]

            return (audio_data.tobytes(), pyaudio.paContinue)
        except queue.Empty:
            # Return silence if no data available
            return (np.zeros(frame_count * self.channels, dtype=np.float32).tobytes(), pyaudio.paContinue)
>>>>>>> 2c01fa3 (feat(rvc): run WebUI-trained RVC models in realtime pipeline)
    
    def get_input_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get an audio chunk from the input queue"""
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def put_output_chunk(self, audio_data: np.ndarray):
        """Put an audio chunk into the output queue"""
        self.output_queue.put(audio_data)
    
    def start_processing(self, process_func: Callable[[np.ndarray], np.ndarray]):
        """
        Start processing audio chunks with the given function
        
        Args:
            process_func: Function that processes audio chunks
        """
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(process_func,)
        )
        self.processing_thread.start()
    
    def _processing_loop(self, process_func: Callable[[np.ndarray], np.ndarray]):
        """Main processing loop"""
        while self.is_running:
            chunk = self.get_input_chunk(timeout=0.1)
            if chunk is not None:
                processed = process_func(chunk)
                # Only put output if processing returned valid data
                if processed is not None:
                    self.put_output_chunk(processed)
    
    def stop(self):
        """Stop all streams and cleanup"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join()
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        self.audio.terminate()
