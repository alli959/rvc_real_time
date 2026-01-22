"""
Trainer Service - Training Execution
Handles the actual training subprocess and progress monitoring
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import settings
from .jobs import TrainingJob, TrainingStatus, job_manager

logger = logging.getLogger(__name__)


class TrainingExecutor:
    """Executes RVC training jobs"""
    
    def __init__(self):
        self.rvc_root = Path(os.getenv("RVC_ROOT", "/app/rvc"))
        self.data_root = Path(settings.paths.data_root)
        self.models_root = Path(settings.paths.models_root)
        self.assets_root = Path(settings.paths.assets_root)
    
    async def validate_preprocessing(self, exp_name: str) -> Tuple[bool, str]:
        """
        Validate that preprocessing outputs exist and are complete.
        
        Required directories:
        - 0_gt_wavs/     - Ground truth wavs
        - 1_16k_wavs/    - 16kHz wavs for feature extraction
        - 2a_f0/         - F0 features
        - 2b_f0nsf/      - F0 nsf features
        - 3_feature768/  - HuBERT features (v2)
        """
        exp_dir = self.data_root / exp_name
        
        if not exp_dir.exists():
            return False, f"Experiment directory not found: {exp_dir}"
        
        required_dirs = {
            "0_gt_wavs": "Ground truth wavs",
            "1_16k_wavs": "16kHz wavs",
            "2a_f0": "F0 features",
            "2b_f0nsf": "F0 nsf features",
            "3_feature768": "HuBERT features"
        }
        
        for dir_name, description in required_dirs.items():
            dir_path = exp_dir / dir_name
            if not dir_path.exists():
                return False, f"Missing {description} directory: {dir_name}"
            
            # Check for files
            files = list(dir_path.glob("*"))
            if not files:
                return False, f"No files in {description} directory: {dir_name}"
        
        # Count segments and validate consistency
        gt_wavs = list((exp_dir / "0_gt_wavs").glob("*.wav"))
        wav_16k = list((exp_dir / "1_16k_wavs").glob("*.wav"))
        f0_files = list((exp_dir / "2a_f0").glob("*.npy"))
        feature_files = list((exp_dir / "3_feature768").glob("*.npy"))
        
        num_segments = len(gt_wavs)
        
        if len(wav_16k) != num_segments:
            return False, f"Mismatch: {len(wav_16k)} 16k wavs vs {num_segments} gt wavs"
        
        if len(f0_files) != num_segments:
            return False, f"Mismatch: {len(f0_files)} f0 files vs {num_segments} segments"
        
        if len(feature_files) != num_segments:
            return False, f"Mismatch: {len(feature_files)} feature files vs {num_segments} segments"
        
        logger.info(f"Validated preprocessing for {exp_name}: {num_segments} segments")
        return True, f"Valid: {num_segments} segments"
    
    async def generate_filelist(
        self,
        exp_name: str,
        sample_rate: int = 48000,
        version: str = "v2",
        use_pitch_guidance: bool = True
    ) -> str:
        """
        Generate filelist.txt for training.
        
        Format for each line (with F0):
        gt_wav_path|feature_path|f0nsf_path|f0_path|speaker_id
        
        Returns path to generated filelist.
        """
        exp_dir = self.data_root / exp_name
        
        gt_wavs_dir = exp_dir / "0_gt_wavs"
        feature_dir = exp_dir / "3_feature768"
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b_f0nsf"
        
        filelist_lines = []
        speaker_id = 0  # Single speaker for now
        
        for gt_wav in sorted(gt_wavs_dir.glob("*.wav")):
            basename = gt_wav.stem
            
            feature_path = feature_dir / f"{basename}.npy"
            f0_path = f0_dir / f"{basename}.npy"
            f0nsf_path = f0nsf_dir / f"{basename}.npy"
            
            # Verify all files exist
            if not feature_path.exists():
                logger.warning(f"Missing feature file: {feature_path}")
                continue
            if use_pitch_guidance:
                if not f0_path.exists():
                    logger.warning(f"Missing F0 file: {f0_path}")
                    continue
                if not f0nsf_path.exists():
                    logger.warning(f"Missing F0 nsf file: {f0nsf_path}")
                    continue
            
            # Create filelist line
            if use_pitch_guidance:
                line = f"{gt_wav}|{feature_path}|{f0nsf_path}|{f0_path}|{speaker_id}"
            else:
                line = f"{gt_wav}|{feature_path}|{speaker_id}"
            
            filelist_lines.append(line)
        
        # Write filelist
        filelist_path = exp_dir / "filelist.txt"
        with open(filelist_path, "w") as f:
            f.write("\n".join(filelist_lines))
        
        logger.info(f"Generated filelist with {len(filelist_lines)} entries: {filelist_path}")
        return str(filelist_path)
    
    async def create_training_config(
        self,
        exp_name: str,
        sample_rate: int = 48000,
        batch_size: int = 8,
        version: str = "v2"
    ) -> str:
        """
        Create config.json for training.
        
        This is the model hyperparameters configuration required by train.py.
        """
        exp_dir = self.data_root / exp_name
        
        # RVC v2 48k config
        config = {
            "train": {
                "log_interval": 200,
                "seed": 1234,
                "epochs": 20000,  # Will be overridden by CLI arg
                "learning_rate": 1e-4,
                "betas": [0.8, 0.99],
                "eps": 1e-9,
                "batch_size": batch_size,
                "fp16_run": True,
                "lr_decay": 0.999875,
                "segment_size": 17280,  # 48k: 17280 samples = 360ms
                "init_lr_ratio": 1,
                "warmup_epochs": 0,
                "c_mel": 45,
                "c_kl": 1.0
            },
            "data": {
                "max_wav_value": 32768.0,
                "sampling_rate": sample_rate,
                "filter_length": 2048,
                "hop_length": 480,  # 48k: 480 = 10ms
                "win_length": 2048,
                "n_mel_channels": 128,
                "mel_fmin": 0.0,
                "mel_fmax": None
            },
            "model": {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [12, 10, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [24, 20, 4, 4],
                "spk_embed_dim": 109,
                "gin_channels": 256,
                "sr": sample_rate
            },
            "version": version
        }
        
        config_path = exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created training config: {config_path}")
        return str(config_path)
    
    async def run_training(
        self,
        job: TrainingJob
    ) -> bool:
        """
        Run the training subprocess.
        
        Monitors progress and updates job state.
        Returns True if training completed successfully.
        """
        exp_name = job.exp_name
        config = job.config
        
        # Get training parameters
        epochs = config.get("epochs", 100)
        batch_size = config.get("batch_size", 8)
        save_every = config.get("save_every_epoch", 10)
        sample_rate = config.get("sample_rate", 48000)
        version = config.get("version", "v2")
        use_pitch_guidance = config.get("use_pitch_guidance", True)
        gpus = config.get("gpus", "0")
        
        # Resolve pretrained model paths
        pretrain_g = config.get("pretrain_g") or str(
            self.assets_root / "pretrained_v2" / "f0G48k.pth"
        )
        pretrain_d = config.get("pretrain_d") or str(
            self.assets_root / "pretrained_v2" / "f0D48k.pth"
        )
        
        # Build training command
        sr_str = f"{sample_rate // 1000}k"
        train_script = self.rvc_root / "infer" / "modules" / "train" / "train.py"
        
        train_cmd = [
            sys.executable,
            str(train_script),
            "-se", str(save_every),
            "-te", str(epochs),
            "-pg", pretrain_g,
            "-pd", pretrain_d,
            "-g", gpus,
            "-bs", str(batch_size),
            "-e", exp_name,
            "-sr", sr_str,
            "-sw", "1",  # save_every_weights
            "-v", version,
            "-f0", "1" if use_pitch_guidance else "0",
            "-l", "0",  # if_latest - keep all checkpoints
            "-c", "0",  # if_cache_data_in_gpu
        ]
        
        logger.info(f"Training command: {' '.join(train_cmd)}")
        
        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.rvc_root)
        env["PYTHONUNBUFFERED"] = "1"
        
        # Create logs directory symlink (train.py expects ./logs/{exp_name})
        logs_dir = self.rvc_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logs_exp_link = logs_dir / exp_name
        exp_dir = self.data_root / exp_name
        
        if logs_exp_link.is_symlink():
            logs_exp_link.unlink()
        if not logs_exp_link.exists():
            logs_exp_link.symlink_to(exp_dir.resolve())
        
        # Ensure output directory exists
        output_dir = self.models_root / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start training subprocess
        process = await asyncio.create_subprocess_exec(
            *train_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.rvc_root),
            env=env
        )
        
        job.process = process
        job.status = TrainingStatus.TRAINING
        job.total_epochs = epochs
        
        training_complete = False
        
        async def read_output(stream, stream_name: str):
            nonlocal training_complete
            
            while True:
                line = await stream.readline()
                if not line:
                    break
                
                line_str = line.decode('utf-8', errors='ignore').strip()
                if not line_str:
                    continue
                
                logger.info(f"[train.py {stream_name}] {line_str}")
                
                # Parse progress
                try:
                    # Epoch completion: "====> Epoch: X"
                    if "====> Epoch:" in line_str:
                        match = re.search(r"====> Epoch:\s*(\d+)", line_str)
                        if match:
                            current_epoch = int(match.group(1))
                            job.current_epoch = current_epoch
                            job.progress = current_epoch / epochs
                            job.log(f"Completed epoch {current_epoch}/{epochs}")
                    
                    # Batch progress: "Train Epoch: X [Y%]"
                    elif "Train Epoch:" in line_str:
                        match = re.search(r"Train Epoch:\s*(\d+)\s*\[(\d+(?:\.\d+)?)%\]", line_str)
                        if match:
                            current_epoch = int(match.group(1))
                            batch_pct = float(match.group(2))
                            job.current_epoch = current_epoch
                            epoch_progress = (current_epoch - 1 + batch_pct / 100) / epochs
                            job.progress = epoch_progress
                            job.log(f"Training epoch {current_epoch}/{epochs} ({batch_pct:.0f}%)")
                    
                    # Step info
                    elif "Global Step:" in line_str:
                        match = re.search(r"Global Step:\s*(\d+)", line_str)
                        if match:
                            job.current_step = int(match.group(1))
                    
                    # Training complete
                    if "Training is done" in line_str:
                        training_complete = True
                    
                except Exception as e:
                    logger.debug(f"Failed to parse training output: {e}")
                
                # Check for cancellation
                if job._cancel_requested:
                    process.terminate()
                    return
        
        # Read stdout and stderr concurrently
        await asyncio.gather(
            read_output(process.stdout, "stdout"),
            read_output(process.stderr, "stderr")
        )
        
        return_code = await process.wait()
        job.process = None
        
        # Check success (2333333 is RVC's success code)
        if return_code == 2333333 % 256 or return_code == 0 or training_complete:
            logger.info(f"Training completed successfully (return code: {return_code})")
            return True
        else:
            logger.error(f"Training failed with return code: {return_code}")
            return False
    
    async def extract_model(
        self,
        exp_name: str,
        sample_rate: int = 48000,
        version: str = "v2",
        epoch: Optional[int] = None
    ) -> Optional[str]:
        """
        Extract inference model from training checkpoint using RVC's savee().
        
        Uses the proper RVC process_ckpt.savee() function to ensure the output
        model has the correct format with properly extracted weights and config.
        
        If epoch is None, extracts from the latest checkpoint.
        Returns path to extracted model or None if failed.
        """
        exp_dir = self.data_root / exp_name
        
        # Find checkpoint
        if epoch:
            g_path = exp_dir / f"G_{epoch}.pth"
            d_path = exp_dir / f"D_{epoch}.pth"
            target_epoch = epoch
        else:
            # Find latest checkpoint
            checkpoints = sorted(exp_dir.glob("G_*.pth"), key=lambda p: p.stat().st_mtime)
            if not checkpoints:
                logger.error(f"No checkpoints found in {exp_dir}")
                return None
            g_path = checkpoints[-1]
            # Extract epoch number from filename (G_123.pth -> 123)
            target_epoch = int(g_path.stem.split("_")[1])
            d_path = exp_dir / f"D_{target_epoch}.pth"
        
        if not g_path.exists():
            logger.error(f"Checkpoint not found: {g_path}")
            return None
        
        logger.info(f"Extracting model from checkpoint: {g_path} (epoch {target_epoch})")
        
        # Output directory in /models (shared volume)
        output_dir = self.models_root / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import torch
            
            # Load checkpoint
            ckpt = torch.load(g_path, map_location="cpu")
            
            # Try to use RVC's savee() for proper model extraction
            try:
                # Add RVC to path if not already
                rvc_lib_path = str(self.rvc_root / "lib" / "train")
                if rvc_lib_path not in sys.path:
                    sys.path.insert(0, rvc_lib_path)
                
                from process_ckpt import savee
                
                # Load training config to get hps (hyperparameters)
                config_path = exp_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config_data = json.load(f)
                    
                    # Convert config to a simple namespace object for hps
                    class HParams:
                        def __init__(self, **kwargs):
                            for k, v in kwargs.items():
                                if isinstance(v, dict):
                                    setattr(self, k, HParams(**v))
                                else:
                                    setattr(self, k, v)
                    
                    hps = HParams(**config_data)
                    
                    # Use savee with output_dir parameter to write to /models
                    model_name = f"{exp_name}_e{target_epoch}"
                    if_f0 = 1  # F0 pitch guidance enabled
                    
                    # Get the model state dict from checkpoint
                    model_state = ckpt.get("model", ckpt)
                    
                    output_path = savee(
                        ckpt=model_state,
                        sr=sample_rate,
                        if_f0=if_f0,
                        name=model_name,
                        epoch=target_epoch,
                        version=version,
                        hps=hps,
                        output_dir=str(output_dir)
                    )
                    
                    if output_path and not output_path.startswith("Traceback"):
                        logger.info(f"Extracted model using savee() to: {output_path}")
                        return output_path
                    else:
                        logger.warning(f"savee() returned error, falling back: {output_path}")
                        raise ValueError("savee() failed")
                else:
                    logger.warning(f"Training config not found at {config_path}, using fallback extraction")
                    raise FileNotFoundError("config.json not found")
                    
            except Exception as savee_error:
                logger.warning(f"Could not use savee(), using fallback extraction: {savee_error}")
                
                # Fallback: manual extraction with proper weight filtering
                from collections import OrderedDict
                
                opt = OrderedDict()
                opt["weight"] = OrderedDict()
                
                # Get the model weights, filtering out encoder_q (not needed for inference)
                model_state = ckpt.get("model", ckpt)
                for key, value in model_state.items():
                    if "enc_q" in key:
                        continue  # Skip encoder_q weights (training only)
                    if hasattr(value, 'half'):
                        opt["weight"][key] = value.half()
                    else:
                        opt["weight"][key] = value
                
                # Standard config for 48k v2 model
                if version == "v2":
                    opt["config"] = [
                        1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                        [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                        [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256,
                        sample_rate
                    ]
                else:  # v1
                    opt["config"] = [
                        1025, 32, 192, 192, 256, 2, 6, 3, 0, "1",
                        [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                        [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256,
                        sample_rate
                    ]
                
                opt["info"] = f"{target_epoch}epoch"
                opt["sr"] = sample_rate
                opt["f0"] = 1  # F0 pitch guidance enabled
                opt["version"] = version
                
                output_path = output_dir / f"{exp_name}.pth"
                torch.save(opt, output_path)
                logger.info(f"Extracted model (fallback) to: {output_path}")
                
                return str(output_path)
            
        except Exception as e:
            logger.exception(f"Failed to extract model: {e}")
            return None
    
    async def build_index(self, exp_name: str, version: str = "v2") -> Optional[str]:
        """
        Build FAISS index from extracted features.
        
        Returns path to created index or None if failed.
        """
        import faiss
        
        exp_dir = self.data_root / exp_name
        feature_dim = 768 if version == "v2" else 256
        feature_dir = exp_dir / f"3_feature{feature_dim}"
        
        # Collect all features
        all_features = []
        for npy_file in sorted(feature_dir.glob("*.npy")):
            features = np.load(str(npy_file))
            all_features.append(features)
        
        if not all_features:
            logger.warning("No features found for index building")
            return None
        
        # Concatenate features
        big_npy = np.concatenate(all_features, axis=0).astype(np.float32)
        logger.info(f"Building index from {big_npy.shape[0]} feature vectors")
        
        # Save concatenated features
        total_fea_path = exp_dir / "total_fea.npy"
        np.save(str(total_fea_path), big_npy)
        
        # Build FAISS index
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        n_ivf = max(n_ivf, 1)
        
        index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
        index.train(big_npy)
        index.add(big_npy)
        
        # Save index
        output_dir = self.models_root / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / f"{exp_name}.index"
        faiss.write_index(index, str(index_path))
        
        logger.info(f"Created index at {index_path}")
        return str(index_path)


# Global executor instance
executor = TrainingExecutor()
