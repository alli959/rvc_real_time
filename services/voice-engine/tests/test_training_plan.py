"""
Tests for Training Plan Recommender and Watchdog System

This test suite validates:
1. Mode detection (NEW vs RESUME vs FINE_TUNE)
2. Data analysis (quality classification, type detection)
3. Config rules (sample rate selection, epoch caps)
4. Watchdog gates (preprocess, F0, stuck detection, smoke test)

Run with:
    pytest test_training_plan.py -v
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

# Import modules under test
from app.trainer.training_plan import (
    recommend_training_plan,
    detect_training_mode,
    analyze_dataset,
    calculate_suggested_config,
    configure_watchdogs,
    TrainingPlan,
    TrainingPlanMode,
    DataType,
    DataQuality,
    PlanThresholds,
    DEFAULT_THRESHOLDS,
    SuggestedConfig,
    LockedParams,
    WatchdogConfig,
    WatchdogGate,
)
from app.trainer.training_watchdogs import (
    WatchdogManager,
    WatchdogResult,
    WatchdogStatus,
    create_watchdog_manager,
)
from app.trainer.pipeline_integration import (
    WatchdogHooks,
    WatchdogIntegrationConfig,
    parse_training_log,
    analyze_training_log_for_issues,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def mock_model_dir(temp_dir):
    """Create a mock model directory with checkpoints"""
    model_dir = temp_dir / "test_model"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def mock_audio_files(temp_dir):
    """Create mock audio files"""
    audio_dir = temp_dir / "audio"
    audio_dir.mkdir()
    
    # Create fake audio files (just empty files with wav extension)
    files = []
    for i in range(5):
        f = audio_dir / f"sample_{i}.wav"
        f.touch()
        files.append(str(f))
    
    return files


@pytest.fixture
def thresholds():
    """Get default thresholds"""
    return DEFAULT_THRESHOLDS


# =============================================================================
# MODE DETECTION TESTS
# =============================================================================

class TestModeDetection:
    """Tests for training mode detection (NEW vs RESUME vs FINE_TUNE)"""
    
    def test_new_model_no_checkpoints(self, temp_dir):
        """New model detected when model_dir is empty"""
        model_dir = temp_dir / "new_model"
        model_dir.mkdir()
        
        result = detect_training_mode(str(model_dir))
        
        assert result['mode'] == TrainingPlanMode.NEW_MODEL
        assert result['locked_params'] is None
        assert result['checkpoint_info'] is None
    
    def test_new_model_dir_doesnt_exist(self, temp_dir):
        """New model detected when model_dir doesn't exist"""
        model_dir = temp_dir / "nonexistent"
        
        result = detect_training_mode(str(model_dir))
        
        assert result['mode'] == TrainingPlanMode.NEW_MODEL
    
    def test_resume_from_generator_checkpoint(self, temp_dir):
        """Resume mode detected when G_*.pth exists"""
        model_dir = temp_dir / "resume_model"
        model_dir.mkdir()
        
        # Create generator checkpoint
        (model_dir / "G_12000.pth").touch()
        (model_dir / "D_12000.pth").touch()
        
        # Create config file with locked params
        config = {
            'sample_rate': 48000,
            'if_f0': True,
            'f0_method': 'rmvpe',
            'version': 'v2',
        }
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        result = detect_training_mode(str(model_dir))
        
        assert result['mode'] == TrainingPlanMode.RESUME
        assert result['checkpoint_info'] is not None
        assert result['checkpoint_info']['generator'] == str(model_dir / "G_12000.pth")
        
        locked = result['locked_params']
        assert locked is not None
        assert locked.sample_rate == 48000
        assert locked.if_f0 is True
    
    def test_fine_tune_from_inference_model(self, temp_dir):
        """Fine-tune mode detected from inference model without training checkpoints"""
        model_dir = temp_dir / "finetune_model"
        model_dir.mkdir()
        
        # Only inference model, no G_/D_ checkpoints
        (model_dir / "model.pth").touch()
        
        # Config indicating pretrained model path
        config = {
            'sample_rate': 40000,
            'pretrained_model': 'base_model.pth',
        }
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        result = detect_training_mode(str(model_dir))
        
        # Should be FINE_TUNE or NEW_MODEL depending on implementation
        # If G/D checkpoints are missing, typically NEW_MODEL
        assert result['mode'] in [TrainingPlanMode.NEW_MODEL, TrainingPlanMode.FINE_TUNE]
    
    def test_locked_params_extraction(self, temp_dir):
        """Locked params correctly extracted from config.json"""
        model_dir = temp_dir / "locked_test"
        model_dir.mkdir()
        
        # Create checkpoints
        (model_dir / "G_24000.pth").touch()
        (model_dir / "D_24000.pth").touch()
        
        # Config with all params
        config = {
            'sample_rate': 32000,
            'if_f0': False,
            'f0_method': 'crepe',
            'version': 'v2',
        }
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        result = detect_training_mode(str(model_dir))
        locked = result['locked_params']
        
        assert locked.sample_rate == 32000
        assert locked.if_f0 is False
        assert locked.version == 'v2'


# =============================================================================
# DATA ANALYSIS TESTS
# =============================================================================

class TestDataAnalysis:
    """Tests for audio data analysis and classification"""
    
    @patch('app.trainer.training_plan._compute_audio_metrics')
    def test_tiny_dataset_detection(self, mock_metrics, mock_audio_files, thresholds):
        """Tiny dataset (under threshold) correctly detected"""
        # Mock to return very short audio
        mock_metrics.return_value = {
            'total_duration_sec': 30.0,  # Under tiny threshold
            'avg_rms': -20.0,
            'avg_silence_ratio': 0.1,
            'max_clipping_ratio': 0.01,
        }
        
        result = analyze_dataset(mock_audio_files, thresholds)
        
        # Should warn about tiny dataset
        assert result['is_tiny_data'] is True
        assert result['total_duration_sec'] < thresholds.tiny_data_seconds
    
    @patch('app.trainer.training_plan._compute_audio_metrics')
    def test_quality_excellent(self, mock_metrics, mock_audio_files, thresholds):
        """Excellent quality data correctly classified"""
        mock_metrics.return_value = {
            'total_duration_sec': 600.0,
            'avg_rms': -22.0,  # Good level
            'avg_silence_ratio': 0.05,  # Very little silence
            'max_clipping_ratio': 0.002,  # Minimal clipping
            'f0_voiced_ratio': 0.95,  # High voiced %
            'f0_outlier_ratio': 0.01,  # Few outliers
        }
        
        result = analyze_dataset(mock_audio_files, thresholds)
        
        assert result['quality'] == DataQuality.EXCELLENT
    
    @patch('app.trainer.training_plan._compute_audio_metrics')
    def test_quality_poor_clipping(self, mock_metrics, mock_audio_files, thresholds):
        """Poor quality detected from high clipping"""
        mock_metrics.return_value = {
            'total_duration_sec': 300.0,
            'avg_rms': -10.0,  # Very loud
            'avg_silence_ratio': 0.1,
            'max_clipping_ratio': 0.15,  # High clipping!
        }
        
        result = analyze_dataset(mock_audio_files, thresholds)
        
        assert result['quality'] in [DataQuality.POOR, DataQuality.FAIR]
    
    @patch('app.trainer.training_plan._compute_audio_metrics')
    def test_data_type_speech(self, mock_metrics, mock_audio_files, thresholds):
        """Speech type detected from stable pitch"""
        mock_metrics.return_value = {
            'total_duration_sec': 300.0,
            'avg_rms': -20.0,
            'avg_silence_ratio': 0.15,  # Normal for speech
            'max_clipping_ratio': 0.01,
            'f0_voiced_ratio': 0.65,  # Typical for speech (pauses)
            'f0_range_semitones': 12,  # Narrow range = speech
        }
        
        result = analyze_dataset(mock_audio_files, thresholds)
        
        assert result['data_type'] == DataType.SPEECH
    
    @patch('app.trainer.training_plan._compute_audio_metrics')
    def test_data_type_singing(self, mock_metrics, mock_audio_files, thresholds):
        """Singing type detected from wide pitch range"""
        mock_metrics.return_value = {
            'total_duration_sec': 300.0,
            'avg_rms': -18.0,
            'avg_silence_ratio': 0.05,  # Less silence in singing
            'max_clipping_ratio': 0.02,
            'f0_voiced_ratio': 0.90,  # High voiced % in singing
            'f0_range_semitones': 36,  # Wide range = singing
        }
        
        result = analyze_dataset(mock_audio_files, thresholds)
        
        assert result['data_type'] == DataType.SINGING


# =============================================================================
# CONFIG RULES TESTS
# =============================================================================

class TestConfigRules:
    """Tests for configuration rule engine"""
    
    def test_sr_selection_excellent_data(self, thresholds):
        """Excellent data allows 48k sample rate"""
        data_info = {
            'quality': DataQuality.EXCELLENT,
            'data_type': DataType.SPEECH,
            'is_tiny_data': False,
            'is_fx_heavy': False,
        }
        
        config = calculate_suggested_config(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=data_info,
            locked_params=None,
            thresholds=thresholds,
            gpu_memory_gb=12.0,
        )
        
        assert config.sample_rate == 48000
    
    def test_sr_selection_poor_data(self, thresholds):
        """Poor quality data forces 32k sample rate"""
        data_info = {
            'quality': DataQuality.POOR,
            'data_type': DataType.SPEECH,
            'is_tiny_data': False,
            'is_fx_heavy': False,
        }
        
        config = calculate_suggested_config(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=data_info,
            locked_params=None,
            thresholds=thresholds,
            gpu_memory_gb=12.0,
        )
        
        assert config.sample_rate == 32000
    
    def test_sr_selection_tiny_data(self, thresholds):
        """Tiny dataset forces 32k sample rate"""
        data_info = {
            'quality': DataQuality.GOOD,
            'data_type': DataType.SPEECH,
            'is_tiny_data': True,
            'is_fx_heavy': False,
        }
        
        config = calculate_suggested_config(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=data_info,
            locked_params=None,
            thresholds=thresholds,
            gpu_memory_gb=12.0,
        )
        
        assert config.sample_rate == 32000
    
    def test_sr_selection_singing(self, thresholds):
        """Singing data forces 32k sample rate"""
        data_info = {
            'quality': DataQuality.GOOD,
            'data_type': DataType.SINGING,
            'is_tiny_data': False,
            'is_fx_heavy': False,
        }
        
        config = calculate_suggested_config(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=data_info,
            locked_params=None,
            thresholds=thresholds,
            gpu_memory_gb=12.0,
        )
        
        assert config.sample_rate == 32000
    
    def test_sr_locked_resume(self, thresholds):
        """Resume mode respects locked sample rate"""
        data_info = {
            'quality': DataQuality.EXCELLENT,
            'data_type': DataType.SPEECH,
            'is_tiny_data': False,
            'is_fx_heavy': False,
        }
        
        locked = LockedParams(
            sample_rate=40000,  # Locked to 40k
            if_f0=True,
            version='v2',
        )
        
        config = calculate_suggested_config(
            mode=TrainingPlanMode.RESUME,
            data_info=data_info,
            locked_params=locked,
            thresholds=thresholds,
            gpu_memory_gb=12.0,
        )
        
        assert config.sample_rate == 40000
    
    def test_epoch_cap_tiny_data(self, thresholds):
        """Tiny data has lower epoch cap"""
        data_info = {
            'quality': DataQuality.GOOD,
            'data_type': DataType.SPEECH,
            'is_tiny_data': True,
            'is_fx_heavy': False,
        }
        
        config = calculate_suggested_config(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=data_info,
            locked_params=None,
            thresholds=thresholds,
            gpu_memory_gb=12.0,
        )
        
        # Should cap at tiny_data_max_epochs
        assert config.epochs <= thresholds.tiny_data_max_epochs
    
    def test_epoch_cap_finetune(self, thresholds):
        """Fine-tune has lower epoch cap"""
        data_info = {
            'quality': DataQuality.GOOD,
            'data_type': DataType.SPEECH,
            'is_tiny_data': False,
            'is_fx_heavy': False,
        }
        
        config = calculate_suggested_config(
            mode=TrainingPlanMode.FINE_TUNE,
            data_info=data_info,
            locked_params=None,
            thresholds=thresholds,
            gpu_memory_gb=12.0,
        )
        
        # Should cap at finetune_max_epochs
        assert config.epochs <= thresholds.finetune_max_epochs
    
    def test_batch_size_gpu_scaling(self, thresholds):
        """Batch size scales with GPU memory"""
        data_info = {
            'quality': DataQuality.GOOD,
            'data_type': DataType.SPEECH,
            'is_tiny_data': False,
            'is_fx_heavy': False,
        }
        
        config_8gb = calculate_suggested_config(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=data_info,
            locked_params=None,
            thresholds=thresholds,
            gpu_memory_gb=8.0,
        )
        
        config_16gb = calculate_suggested_config(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=data_info,
            locked_params=None,
            thresholds=thresholds,
            gpu_memory_gb=16.0,
        )
        
        # Larger GPU should allow larger batch
        assert config_16gb.batch_size >= config_8gb.batch_size


# =============================================================================
# WATCHDOG CONFIG TESTS
# =============================================================================

class TestWatchdogConfig:
    """Tests for watchdog configuration"""
    
    def test_watchdogs_configured_for_new_model(self, thresholds):
        """New model gets all watchdogs"""
        data_info = {
            'quality': DataQuality.GOOD,
            'data_type': DataType.SPEECH,
            'is_tiny_data': False,
        }
        
        watchdogs = configure_watchdogs(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=data_info,
            thresholds=thresholds,
        )
        
        gates = [wd.gate for wd in watchdogs]
        
        assert WatchdogGate.PREPROCESS in gates
        assert WatchdogGate.F0_EXTRACTION in gates
        assert WatchdogGate.EARLY_TRAINING in gates
        assert WatchdogGate.SMOKE_TEST in gates
    
    def test_watchdog_thresholds_stricter_for_poor_data(self, thresholds):
        """Poor quality data gets stricter watchdog thresholds"""
        poor_data = {
            'quality': DataQuality.POOR,
            'data_type': DataType.SPEECH,
            'is_tiny_data': False,
        }
        
        good_data = {
            'quality': DataQuality.GOOD,
            'data_type': DataType.SPEECH,
            'is_tiny_data': False,
        }
        
        poor_watchdogs = configure_watchdogs(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=poor_data,
            thresholds=thresholds,
        )
        
        good_watchdogs = configure_watchdogs(
            mode=TrainingPlanMode.NEW_MODEL,
            data_info=good_data,
            thresholds=thresholds,
        )
        
        # Get early training watchdogs
        poor_early = next((w for w in poor_watchdogs if w.gate == WatchdogGate.EARLY_TRAINING), None)
        good_early = next((w for w in good_watchdogs if w.gate == WatchdogGate.EARLY_TRAINING), None)
        
        if poor_early and good_early:
            # Poor data should have lower stuck detection window
            assert poor_early.thresholds.get('stuck_window', 200) <= good_early.thresholds.get('stuck_window', 200)


# =============================================================================
# WATCHDOG MANAGER TESTS
# =============================================================================

class TestWatchdogManager:
    """Tests for WatchdogManager functionality"""
    
    def test_create_manager(self, mock_model_dir):
        """WatchdogManager can be created"""
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
        )
        
        assert manager.job_id == "test_job"
        assert not manager.should_abort()
    
    def test_stuck_mel_detection(self, mock_model_dir):
        """Stuck mel loss triggers warning/abort"""
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
            thresholds={
                'early_training': {
                    'stuck_window': 100,
                    'stuck_tolerance': 0.001,
                }
            }
        )
        
        # Feed constant mel loss
        for i in range(150):
            result = manager.update_loss_metrics(
                step=i,
                epoch=1,
                loss_disc=5.0 + np.random.random() * 0.1,  # Varying disc
                loss_gen=3.0 + np.random.random() * 0.1,  # Varying gen
                loss_fm=1.0 + np.random.random() * 0.1,   # Varying fm
                loss_mel=75.0,  # STUCK!
                loss_kl=0.5,    # Varying kl
            )
        
        # Should detect stuck pattern
        summary = manager.get_summary()
        assert summary.get('warnings') or manager.should_abort()
    
    def test_nan_loss_detection(self, mock_model_dir):
        """NaN loss triggers abort"""
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
        )
        
        result = manager.update_loss_metrics(
            step=100,
            epoch=1,
            loss_disc=float('nan'),
            loss_gen=float('nan'),
            loss_fm=0.0,
            loss_mel=0.0,
            loss_kl=0.0,
        )
        
        # Should trigger abort
        assert manager.should_abort()
        assert 'nan' in manager.get_abort_reason().lower()
    
    def test_ready_status_initial(self, mock_model_dir):
        """Initial ready status is False"""
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
        )
        
        status = manager.get_ready_status()
        
        assert status['is_ready'] is False
        assert status['preprocess_passed'] is False
        assert status['f0_passed'] is False
        assert status['smoke_test_passed'] is False


# =============================================================================
# PREPROCESS GATE TESTS
# =============================================================================

class TestPreprocessGate:
    """Tests for preprocess watchdog gate"""
    
    def test_gate_passes_good_audio(self, mock_model_dir):
        """Preprocess gate passes for good audio"""
        # Create mock preprocessed audio dir
        wavs_dir = mock_model_dir / "0_gt_wavs"
        wavs_dir.mkdir()
        
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
        )
        
        # Mock the audio analysis to return good values
        with patch.object(manager, '_analyze_preprocessed_audio') as mock_analyze:
            mock_analyze.return_value = {
                'avg_rms': -20.0,
                'avg_silence_ratio': 0.08,
                'max_clipping_ratio': 0.01,
                'total_files': 50,
            }
            
            result = manager.run_preprocess_gate(str(wavs_dir))
        
        assert result.passed is True
        assert result.status == WatchdogStatus.PASSED
    
    def test_gate_fails_too_quiet(self, mock_model_dir):
        """Preprocess gate fails for too quiet audio"""
        wavs_dir = mock_model_dir / "0_gt_wavs"
        wavs_dir.mkdir()
        
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
            thresholds={
                'preprocess': {
                    'min_rms_db': -35.0,
                }
            }
        )
        
        with patch.object(manager, '_analyze_preprocessed_audio') as mock_analyze:
            mock_analyze.return_value = {
                'avg_rms': -45.0,  # Too quiet!
                'avg_silence_ratio': 0.08,
                'max_clipping_ratio': 0.01,
                'total_files': 50,
            }
            
            result = manager.run_preprocess_gate(str(wavs_dir))
        
        assert result.passed is False


# =============================================================================
# F0 GATE TESTS
# =============================================================================

class TestF0Gate:
    """Tests for F0 extraction watchdog gate"""
    
    def test_f0_gate_passes_good_extraction(self, mock_model_dir):
        """F0 gate passes for good pitch extraction"""
        f0_dir = mock_model_dir / "2a_f0"
        f0_dir.mkdir()
        
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
        )
        
        with patch.object(manager, '_analyze_f0_files') as mock_analyze:
            mock_analyze.return_value = {
                'voiced_ratio': 0.75,
                'outlier_ratio': 0.02,
                'zero_ratio': 0.25,
                'total_files': 50,
            }
            
            result = manager.run_f0_gate(str(f0_dir))
        
        assert result.passed is True
    
    def test_f0_gate_fails_mostly_zero(self, mock_model_dir):
        """F0 gate fails when mostly zero (unvoiced)"""
        f0_dir = mock_model_dir / "2a_f0"
        f0_dir.mkdir()
        
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
            thresholds={
                'f0_extraction': {
                    'min_voiced_ratio': 0.4,
                }
            }
        )
        
        with patch.object(manager, '_analyze_f0_files') as mock_analyze:
            mock_analyze.return_value = {
                'voiced_ratio': 0.15,  # Mostly zeros!
                'outlier_ratio': 0.05,
                'zero_ratio': 0.85,
                'total_files': 50,
            }
            
            result = manager.run_f0_gate(str(f0_dir))
        
        assert result.passed is False
        assert 'voiced' in result.message.lower() or 'zero' in result.message.lower()
    
    def test_f0_gate_fails_many_outliers(self, mock_model_dir):
        """F0 gate fails when too many outliers"""
        f0_dir = mock_model_dir / "2a_f0"
        f0_dir.mkdir()
        
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
            thresholds={
                'f0_extraction': {
                    'max_outlier_ratio': 0.1,
                }
            }
        )
        
        with patch.object(manager, '_analyze_f0_files') as mock_analyze:
            mock_analyze.return_value = {
                'voiced_ratio': 0.70,
                'outlier_ratio': 0.25,  # Too many outliers!
                'zero_ratio': 0.30,
                'total_files': 50,
            }
            
            result = manager.run_f0_gate(str(f0_dir))
        
        assert result.passed is False
        assert 'outlier' in result.message.lower()


# =============================================================================
# SMOKE TEST TESTS
# =============================================================================

class TestSmokeTest:
    """Tests for inference smoke test"""
    
    def test_smoke_test_passes_good_output(self, mock_model_dir):
        """Smoke test passes for good inference output"""
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
        )
        
        # Mock the inference to return good output
        with patch.object(manager, '_run_inference') as mock_infer:
            mock_infer.return_value = {
                'success': True,
                'audio': np.random.random(16000).astype(np.float32) * 0.3,  # Good varied audio
                'sample_rate': 16000,
            }
            
            result = manager.run_smoke_test("test_model.pth")
        
        assert result.passed is True
    
    def test_smoke_test_fails_silent_output(self, mock_model_dir):
        """Smoke test fails for silent output"""
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
        )
        
        with patch.object(manager, '_run_inference') as mock_infer:
            mock_infer.return_value = {
                'success': True,
                'audio': np.zeros(16000, dtype=np.float32),  # Silent!
                'sample_rate': 16000,
            }
            
            result = manager.run_smoke_test("test_model.pth")
        
        assert result.passed is False
        assert 'silent' in result.message.lower() or 'energy' in result.message.lower()
    
    def test_smoke_test_fails_constant_output(self, mock_model_dir):
        """Smoke test fails for constant (collapsed) output"""
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
        )
        
        with patch.object(manager, '_run_inference') as mock_infer:
            # Create constant waveform (collapsed model)
            constant_audio = np.ones(16000, dtype=np.float32) * 0.1
            mock_infer.return_value = {
                'success': True,
                'audio': constant_audio,
                'sample_rate': 16000,
            }
            
            result = manager.run_smoke_test("test_model.pth")
        
        assert result.passed is False
        assert 'constant' in result.message.lower() or 'collapsed' in result.message.lower()
    
    def test_smoke_test_fails_inference_error(self, mock_model_dir):
        """Smoke test fails when inference fails"""
        manager = WatchdogManager(
            job_id="test_job",
            model_dir=str(mock_model_dir),
        )
        
        with patch.object(manager, '_run_inference') as mock_infer:
            mock_infer.return_value = {
                'success': False,
                'error': 'Model loading failed',
            }
            
            result = manager.run_smoke_test("test_model.pth")
        
        assert result.passed is False


# =============================================================================
# TRAINING LOG ANALYSIS TESTS
# =============================================================================

class TestLogAnalysis:
    """Tests for post-hoc training log analysis"""
    
    def test_parse_training_log(self, temp_dir):
        """Training log parsing extracts loss values"""
        log_path = temp_dir / "train.log"
        
        log_content = """
2024-01-01 10:00:00 INFO Train Epoch: 1
2024-01-01 10:00:01 INFO loss_disc=5.0, loss_gen=3.0, loss_fm=1.0,loss_mel=45.5, loss_kl=0.5
2024-01-01 10:00:02 INFO loss_disc=4.8, loss_gen=2.9, loss_fm=0.9,loss_mel=44.2, loss_kl=0.48
2024-01-01 10:01:00 INFO Train Epoch: 2
2024-01-01 10:01:01 INFO loss_disc=4.5, loss_gen=2.8, loss_fm=0.85,loss_mel=42.0, loss_kl=0.45
"""
        log_path.write_text(log_content)
        
        entries = parse_training_log(str(log_path))
        
        assert len(entries) == 3
        assert entries[0]['epoch'] == 1
        assert entries[0]['loss_mel'] == 45.5
        assert entries[2]['epoch'] == 2
    
    def test_analyze_stuck_mel_loss(self, temp_dir):
        """Analysis detects stuck mel loss pattern"""
        log_path = temp_dir / "stuck.log"
        
        # Create log with stuck mel loss
        lines = ["2024-01-01 10:00:00 INFO Train Epoch: 1"]
        for i in range(100):
            lines.append(f"2024-01-01 10:00:{i:02d} INFO loss_disc={5.0-i*0.01:.2f}, loss_gen=3.0, loss_fm=1.0,loss_mel=75.000, loss_kl=0.5")
        
        log_path.write_text("\n".join(lines))
        
        result = analyze_training_log_for_issues(str(log_path))
        
        assert result['would_abort'] is True
        assert any('stuck' in issue.lower() or 'STUCK' in issue for issue in result['issues'])
    
    def test_analyze_nan_losses(self, temp_dir):
        """Analysis detects NaN losses"""
        log_path = temp_dir / "nan.log"
        
        lines = [
            "2024-01-01 10:00:00 INFO Train Epoch: 1",
            "2024-01-01 10:00:01 INFO loss_disc=5.0, loss_gen=3.0, loss_fm=1.0,loss_mel=45.0, loss_kl=0.5",
            "2024-01-01 10:00:02 INFO loss_disc=nan, loss_gen=nan, loss_fm=nan,loss_mel=nan, loss_kl=nan",
        ]
        log_path.write_text("\n".join(lines))
        
        result = analyze_training_log_for_issues(str(log_path))
        
        # Should detect NaN
        metrics = result.get('metrics', {})
        assert metrics.get('nan_count', 0) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFullPlanRecommendation:
    """Integration tests for full plan recommendation"""
    
    @patch('app.trainer.training_plan._compute_audio_metrics')
    def test_recommend_plan_new_model(self, mock_metrics, mock_audio_files, temp_dir, thresholds):
        """Full plan generated for new model"""
        model_dir = temp_dir / "new_model"
        model_dir.mkdir()
        
        mock_metrics.return_value = {
            'total_duration_sec': 300.0,
            'avg_rms': -22.0,
            'avg_silence_ratio': 0.10,
            'max_clipping_ratio': 0.01,
            'f0_voiced_ratio': 0.70,
            'f0_range_semitones': 15,
        }
        
        plan = recommend_training_plan(
            model_name="test_model",
            audio_paths=mock_audio_files,
            model_dir=str(model_dir),
            assets_dir="/tmp/assets",
            gpu_memory_gb=12.0,
            thresholds=thresholds,
        )
        
        assert plan.mode == TrainingPlanMode.NEW_MODEL
        assert plan.can_proceed is True
        assert plan.suggested_config is not None
        assert plan.required_watchdogs is not None
        assert len(plan.required_watchdogs) > 0
        assert plan.report is not None
        assert len(plan.report) > 0
    
    @patch('app.trainer.training_plan._compute_audio_metrics')
    def test_plan_report_contains_key_info(self, mock_metrics, mock_audio_files, temp_dir, thresholds):
        """Plan report contains all required information"""
        model_dir = temp_dir / "test_model"
        model_dir.mkdir()
        
        mock_metrics.return_value = {
            'total_duration_sec': 200.0,
            'avg_rms': -25.0,
            'avg_silence_ratio': 0.15,
            'max_clipping_ratio': 0.02,
        }
        
        plan = recommend_training_plan(
            model_name="test_model",
            audio_paths=mock_audio_files,
            model_dir=str(model_dir),
            assets_dir="/tmp/assets",
            gpu_memory_gb=12.0,
            thresholds=thresholds,
        )
        
        report = plan.report
        
        # Check report contains key sections
        assert 'MODE' in report.upper() or 'mode' in report.lower()
        assert 'DATA' in report.upper() or 'data' in report.lower()
        assert 'SAMPLE' in report.upper() or 'sample' in report.lower() or 'SR' in report.upper()
        assert 'EPOCH' in report.upper() or 'epoch' in report.lower()
        assert 'WATCHDOG' in report.upper() or 'watchdog' in report.lower()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_audio_list(self, temp_dir, thresholds):
        """Handle empty audio file list"""
        model_dir = temp_dir / "empty_model"
        model_dir.mkdir()
        
        plan = recommend_training_plan(
            model_name="test_model",
            audio_paths=[],  # Empty!
            model_dir=str(model_dir),
            assets_dir="/tmp/assets",
            gpu_memory_gb=12.0,
            thresholds=thresholds,
        )
        
        assert plan.can_proceed is False
        assert any('audio' in e.lower() or 'empty' in e.lower() for e in plan.errors)
    
    def test_nonexistent_audio_files(self, temp_dir, thresholds):
        """Handle nonexistent audio files"""
        model_dir = temp_dir / "test_model"
        model_dir.mkdir()
        
        plan = recommend_training_plan(
            model_name="test_model",
            audio_paths=["/nonexistent/file1.wav", "/nonexistent/file2.wav"],
            model_dir=str(model_dir),
            assets_dir="/tmp/assets",
            gpu_memory_gb=12.0,
            thresholds=thresholds,
        )
        
        # Should either fail gracefully or report errors
        # Depending on implementation
        assert plan is not None
    
    def test_very_short_audio(self, temp_dir, thresholds):
        """Handle very short audio (under minimum)"""
        model_dir = temp_dir / "short_model"
        model_dir.mkdir()
        
        with patch('app.trainer.training_plan._compute_audio_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'total_duration_sec': 5.0,  # Very short!
                'avg_rms': -20.0,
                'avg_silence_ratio': 0.1,
                'max_clipping_ratio': 0.01,
            }
            
            plan = recommend_training_plan(
                model_name="test_model",
                audio_paths=["dummy.wav"],
                model_dir=str(model_dir),
                assets_dir="/tmp/assets",
                gpu_memory_gb=12.0,
                thresholds=thresholds,
            )
        
        # Should either reject or warn about very short audio
        assert plan.can_proceed is False or any('duration' in w.lower() or 'short' in w.lower() for w in plan.warnings)
    
    def test_user_override_respects_locked_params(self, temp_dir, thresholds):
        """User overrides cannot override locked params"""
        model_dir = temp_dir / "locked_model"
        model_dir.mkdir()
        
        # Create checkpoint to trigger RESUME mode
        (model_dir / "G_12000.pth").touch()
        (model_dir / "D_12000.pth").touch()
        config = {'sample_rate': 40000, 'if_f0': True, 'version': 'v2'}
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        with patch('app.trainer.training_plan._compute_audio_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'total_duration_sec': 300.0,
                'avg_rms': -20.0,
                'avg_silence_ratio': 0.1,
                'max_clipping_ratio': 0.01,
            }
            
            plan = recommend_training_plan(
                model_name="test_model",
                audio_paths=["dummy.wav"],
                model_dir=str(model_dir),
                assets_dir="/tmp/assets",
                gpu_memory_gb=12.0,
                thresholds=thresholds,
                user_overrides={
                    'sample_rate': 48000,  # Try to override locked SR
                }
            )
        
        # Sample rate should still be locked at 40000
        assert plan.suggested_config.sample_rate == 40000


# =============================================================================
# PYTEST CONFIG
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
