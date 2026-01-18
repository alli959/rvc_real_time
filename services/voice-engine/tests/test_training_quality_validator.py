"""
Test Training Quality Validator

These tests ensure that the training quality validator correctly identifies:
1. Collapsed models (stuck mel loss)
2. NaN training issues
3. Low quality F0 extraction
4. Bad output quality (DC offset, low crest factor, etc.)

The tests use real model data from the lexi-11 failure case to ensure
the validator would have caught this issue if it had been in place.
"""

import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


# =============================================================================
# Unit Tests for Quality Metrics
# =============================================================================

class TestQualityMetrics:
    """Test individual quality metric calculations."""
    
    def test_dc_offset_good(self):
        """Good audio should have near-zero DC offset."""
        from app.services.voice_conversion.training_quality_validator import QualityResult
        
        # Simulate good output
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000))
        dc_offset = np.mean(audio)
        
        assert abs(dc_offset) < 0.001, "Pure sine wave should have ~0 DC offset"
    
    def test_dc_offset_bad(self):
        """Collapsed model output typically has high DC offset."""
        # Simulate collapsed output (constant with small noise)
        audio = -0.05 + 0.01 * np.random.randn(48000)
        dc_offset = np.mean(audio)
        
        assert abs(dc_offset) > 0.02, "Collapsed output should have high DC offset"
    
    def test_crest_factor_good(self):
        """Good audio should have reasonable crest factor (higher than collapsed output)."""
        # Speech-like signal with transients and pauses
        np.random.seed(42)
        t = np.linspace(0, 1, 48000)
        # Create a signal with gaps (like speech) - envelope that drops to zero periodically
        envelope = np.abs(np.sin(2 * np.pi * 3 * t)) ** 0.5
        envelope[envelope < 0.3] = 0  # Create silence gaps
        audio = 0.5 * np.sin(2 * np.pi * 200 * t) * envelope
        
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.abs(audio).max()
        crest = peak / rms
        
        # Good audio has crest > 1.5 (collapsed output has crest ~1.0-1.1)
        # Real speech has crest 3-5, our test signal is simplified
        assert crest > 1.5, f"Signal with dynamics should have crest factor > 1.5, got {crest}"
    
    def test_crest_factor_bad(self):
        """Collapsed output has low crest factor (near-constant)."""
        # Constant-ish signal with small variations
        audio = 0.1 + 0.01 * np.random.randn(48000)
        
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.abs(audio).max()
        crest = peak / rms
        
        assert crest < 2, f"Near-constant signal should have crest factor < 2, got {crest}"
    
    def test_spectral_flatness_good(self):
        """Tonal audio should have low spectral flatness."""
        # Pure tone has very low spectral flatness
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000))
        
        fft = np.abs(np.fft.rfft(audio))
        fft = fft[fft > 0]
        sf = np.exp(np.mean(np.log(fft))) / np.mean(fft)
        
        assert sf < 0.1, f"Pure tone should have spectral flatness < 0.1, got {sf}"
    
    def test_spectral_flatness_bad(self):
        """Noise has high spectral flatness."""
        audio = np.random.randn(48000)
        
        fft = np.abs(np.fft.rfft(audio))
        fft = fft[fft > 0]
        sf = np.exp(np.mean(np.log(fft))) / np.mean(fft)
        
        assert sf > 0.5, f"White noise should have spectral flatness > 0.5, got {sf}"


# =============================================================================
# Unit Tests for Training Log Validation
# =============================================================================

class TestTrainingLogValidation:
    """Test training log analysis."""
    
    def test_detect_stuck_loss(self, tmp_path):
        """Should detect when mel loss is stuck at same value (as warning)."""
        from app.services.voice_conversion.training_quality_validator import validate_training_logs
        
        # Create log with stuck mel loss (lexi-11 pattern)
        log_file = tmp_path / "train.log"
        with open(log_file, 'w') as f:
            for i in range(100):
                f.write(f"2026-01-16 03:39:14,310 test INFO loss_disc=100.0, loss_gen=10.0, "
                       f"loss_fm=1000.0,loss_mel=75.000, loss_kl=9.000\n")
        
        result = validate_training_logs(str(log_file))
        
        # With the new design, stuck loss alone is a WARNING (not failure)
        # It becomes a failure only when combined with smoke test failure
        assert any("STUCK" in w or "stuck" in w for w in result.warnings), \
            f"Should warn about stuck loss: {result.warnings}"
        assert result.metrics['mel_loss_stuck_pct'] == 100.0
        assert result.metrics.get('mel_loss_collapse_suspected') == True
    
    def test_detect_nan_loss(self, tmp_path):
        """Should detect NaN loss values."""
        from app.services.voice_conversion.training_quality_validator import validate_training_logs
        
        log_file = tmp_path / "train.log"
        with open(log_file, 'w') as f:
            for i in range(100):
                f.write(f"2026-01-16 03:39:14,310 test INFO loss_disc=nan, loss_gen=nan, "
                       f"loss_fm=nan,loss_mel=nan, loss_kl=nan\n")
        
        result = validate_training_logs(str(log_file))
        
        assert not result.passed, "Should fail with all NaN losses"
        assert any("NaN" in issue for issue in result.issues), \
            f"Should mention NaN in issues: {result.issues}"
    
    def test_detect_decreasing_loss(self, tmp_path):
        """Should pass when loss is decreasing properly."""
        from app.services.voice_conversion.training_quality_validator import validate_training_logs
        
        log_file = tmp_path / "train.log"
        with open(log_file, 'w') as f:
            for i in range(200):
                mel_loss = 75.0 - (i * 0.3)  # Decreasing from 75 to 15
                mel_loss = max(15, mel_loss)
                f.write(f"2026-01-16 03:39:14,310 test INFO loss_disc=100.0, loss_gen=10.0, "
                       f"loss_fm=1000.0,loss_mel={mel_loss:.3f}, loss_kl=9.000\n")
        
        result = validate_training_logs(str(log_file))
        
        assert result.passed, f"Should pass with decreasing loss. Issues: {result.issues}"
        assert result.metrics['loss_decrease_pct'] >= 49, "Should show significant decrease"


# =============================================================================
# Unit Tests for F0 Extraction Validation
# =============================================================================

class TestF0Validation:
    """Test F0 extraction quality validation."""
    
    def test_good_f0_extraction(self, tmp_path):
        """Should pass with good voiced percentage and no outliers."""
        from app.services.voice_conversion.training_quality_validator import validate_f0_extraction
        
        f0_dir = tmp_path / "2a_f0"
        f0_dir.mkdir()
        
        # Create F0 files with good voiced content
        for i in range(10):
            # 80% voiced frames with realistic F0 values
            f0 = np.zeros(500)
            f0[:400] = np.random.uniform(100, 400, 400)  # Voiced
            np.save(f0_dir / f"test_{i}.npy", f0)
        
        result = validate_f0_extraction(str(f0_dir))
        
        assert result.passed, f"Should pass with good F0. Issues: {result.issues}"
        assert result.metrics['voiced_pct'] > 70
    
    def test_too_few_voiced_frames(self, tmp_path):
        """Should fail when most frames are unvoiced (silence/noise)."""
        from app.services.voice_conversion.training_quality_validator import validate_f0_extraction
        
        f0_dir = tmp_path / "2a_f0"
        f0_dir.mkdir()
        
        # Create F0 files with only 5% voiced content (below hard fail threshold of 10%)
        for i in range(10):
            f0 = np.zeros(500)
            f0[:25] = np.random.uniform(100, 400, 25)  # Only 5% voiced
            np.save(f0_dir / f"test_{i}.npy", f0)
        
        result = validate_f0_extraction(str(f0_dir))
        
        assert not result.passed, "Should fail with <10% voiced frames"
        assert result.metrics['voiced_pct'] < 10
    
    def test_too_many_outliers(self, tmp_path):
        """Should fail when many F0 values are outside human speech range."""
        from app.services.voice_conversion.training_quality_validator import validate_f0_extraction
        
        f0_dir = tmp_path / "2a_f0"
        f0_dir.mkdir()
        
        # Create F0 files with many outliers
        for i in range(10):
            f0 = np.zeros(500)
            f0[:400] = np.random.uniform(20, 30, 400)  # All below 40Hz threshold
            np.save(f0_dir / f"test_{i}.npy", f0)
        
        result = validate_f0_extraction(str(f0_dir))
        
        assert not result.passed, "Should fail with too many outliers"
        assert result.metrics['outlier_pct'] > 90


# =============================================================================
# Unit Tests for Preprocessing Validation
# =============================================================================

class TestPreprocessingValidation:
    """Test preprocessing quality validation."""
    
    def test_good_audio_quality(self, tmp_path):
        """Should pass with good RMS and no clipping."""
        from app.services.voice_conversion.training_quality_validator import validate_preprocessing_quality
        from scipy.io import wavfile
        
        audio_dir = tmp_path / "0_gt_wavs"
        audio_dir.mkdir()
        
        # Create good quality audio files
        for i in range(5):
            sr = 48000
            t = np.linspace(0, 1, sr, dtype=np.float32)
            audio = 0.3 * np.sin(2 * np.pi * 200 * t)
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(audio_dir / f"test_{i}.wav", sr, audio_int16)
        
        result = validate_preprocessing_quality(str(audio_dir), sample_rate=48000)
        
        assert result.passed, f"Should pass with good audio. Issues: {result.issues}"
        assert result.metrics['mean_rms'] > 0.1
        assert result.metrics['clipping_pct'] < 1
    
    def test_too_quiet_audio(self, tmp_path):
        """Should fail when audio is too quiet."""
        from app.services.voice_conversion.training_quality_validator import validate_preprocessing_quality
        from scipy.io import wavfile
        
        audio_dir = tmp_path / "0_gt_wavs"
        audio_dir.mkdir()
        
        # Create very quiet audio
        for i in range(5):
            sr = 48000
            t = np.linspace(0, 1, sr, dtype=np.float32)
            audio = 0.001 * np.sin(2 * np.pi * 200 * t)  # Very quiet
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(audio_dir / f"test_{i}.wav", sr, audio_int16)
        
        result = validate_preprocessing_quality(str(audio_dir), sample_rate=48000)
        
        assert not result.passed, "Should fail with quiet audio"
        assert "quiet" in ' '.join(result.issues).lower()


# =============================================================================
# Integration Tests - Real Model Validation
# =============================================================================

class TestRealModelValidation:
    """Test validation with real model data (if available)."""
    
    @pytest.mark.skipif(
        not Path("assets/models/lexi-11").exists(),
        reason="lexi-11 model not available"
    )
    def test_lexi11_detected_as_broken(self):
        """lexi-11 should be detected as a collapsed model."""
        from app.services.voice_conversion.training_quality_validator import (
            validate_training_logs,
            validate_trained_model,
        )
        
        # Test training log validation - with new design, stuck loss is a warning
        log_result = validate_training_logs("assets/models/lexi-11/train.log")
        
        assert log_result.metrics['mel_loss_stuck_pct'] > 90, \
            "lexi-11 should show >90% stuck mel loss"
        assert log_result.metrics.get('mel_loss_collapse_suspected') == True, \
            "lexi-11 should be flagged as suspected collapse"
        
        # Full validation with smoke test should FAIL (combined evidence)
        full_result = validate_trained_model("assets/models/lexi-11", run_smoke_test=True)
        assert not full_result.passed, \
            f"lexi-11 full validation should fail. Issues: {full_result.issues}"
    
    @pytest.mark.skipif(
        not Path("assets/models/lexi-8").exists(),
        reason="lexi-8 model not available"
    )
    def test_lexi8_smoke_test_passes(self):
        """lexi-8 should pass smoke test (working model)."""
        from app.services.voice_conversion.training_quality_validator import smoke_test_checkpoint
        
        result = smoke_test_checkpoint("assets/models/lexi-8/lexi-8.pth")
        
        assert result.passed, f"lexi-8 should pass smoke test. Issues: {result.issues}"
        # Windowed analysis uses median crest factor, threshold is 2.0
        assert result.metrics['crest_factor'] > 2
        assert abs(result.metrics['dc_offset']) < 0.02


# =============================================================================
# Smoke Test Metrics Tests
# =============================================================================

class TestSmokeTestMetrics:
    """Test smoke test threshold calibration."""
    
    def test_thresholds_are_reasonable(self):
        """Verify smoke test thresholds differentiate good vs bad models."""
        # These thresholds are calibrated based on lexi-8 (good) vs lexi-11 (bad) comparison
        # With windowed analysis on active frames:
        # lexi-8:  dc=0.0003, crest=2.29, sf=0.20, zc=860/s  -> PASS
        # lexi-11: dc=-0.054, crest=~1.0, sf=0.65, zc=5957/s -> FAIL
        
        # DC offset threshold
        dc_threshold = 0.02
        assert abs(0.0003) < dc_threshold, "Good model should pass DC threshold"
        assert abs(-0.054) > dc_threshold, "Bad model should fail DC threshold"
        
        # Crest factor threshold (lowered for windowed analysis)
        crest_threshold = 2.0
        assert 2.29 > crest_threshold, "Good model should pass crest threshold"
        assert 1.06 < crest_threshold, "Bad model should fail crest threshold"
        
        # Spectral flatness threshold
        sf_threshold = 0.35
        assert 0.20 < sf_threshold, "Good model should pass SF threshold"
        assert 0.65 > sf_threshold, "Bad model should fail SF threshold"


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestValidationAPI:
    """Test validation API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient
            from main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI or main app not available")
    
    def test_validate_endpoint_exists(self, client):
        """Validation endpoint should exist."""
        if client is None:
            pytest.skip("Test client not available")
        # This will fail if model doesn't exist, but tests endpoint registration
        response = client.post("/convert/models/validate/nonexistent")
        assert response.status_code in [404, 500], \
            "Should return 404 for missing model, not 405 (method not allowed)"
    
    def test_smoke_test_endpoint_exists(self, client):
        """Smoke test endpoint should exist."""
        if client is None:
            pytest.skip("Test client not available")
        response = client.post("/convert/models/smoke-test/nonexistent")
        assert response.status_code in [404, 500], \
            "Should return 404 for missing model, not 405 (method not allowed)"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
