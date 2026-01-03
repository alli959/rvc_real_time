"""
Feature Extraction Module - Efficient feature extraction for RVC models
"""

import numpy as np
import librosa
from typing import Tuple, Optional


class FeatureExtractor:
    """Extracts audio features for RVC model inference"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """
        Initialize feature extractor
        
        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel bands
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch (F0) from audio
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Tuple of (pitch, voiced_flag)
        """
        # Use pyin for pitch extraction
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Replace NaN values with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        
        return f0, voiced_flag
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            audio: Audio waveform as numpy array
            n_mfcc: Number of MFCCs to extract
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def extract_all_features(self, audio: np.ndarray) -> dict:
        """
        Extract all features from audio chunk
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'mfcc': self.extract_mfcc(audio),
        }
        
        # Extract pitch if audio is long enough
        if len(audio) >= self.hop_length * 2:
            f0, voiced = self.extract_pitch(audio)
            features['f0'] = f0
            features['voiced'] = voiced
        
        return features
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to zero mean and unit variance
        
        Args:
            features: Feature array
            
        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=-1, keepdims=True)
        std = np.std(features, axis=-1, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        return (features - mean) / std
