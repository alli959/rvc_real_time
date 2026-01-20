import os
import sys
import logging
import json
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))


# =============================================================================
# Training Collapse Watchdog (inline - fail-fast)
# =============================================================================

class TrainingCollapseWatchdog:
    """
    Lightweight watchdog to detect training collapse EARLY.
    
    Collapse signatures:
    1. loss_mel stuck at exactly the same value for N steps
    2. loss_kl stuck at exactly the same value for N steps
    3. Both mel and kl stuck = DEFINITE collapse
    
    This watchdog runs inline with training and can abort immediately.
    """
    
    def __init__(
        self,
        stuck_window: int = 50,      # Steps to track
        stuck_threshold: float = 0.001,  # Values within this delta = "same"
        abort_on_stuck: bool = True,
        logger: logging.Logger = None,
    ):
        self.stuck_window = stuck_window
        self.stuck_threshold = stuck_threshold
        self.abort_on_stuck = abort_on_stuck
        self.logger = logger or logging.getLogger(__name__)
        
        # Tracking buffers
        self.mel_history = deque(maxlen=stuck_window)
        self.kl_history = deque(maxlen=stuck_window)
        self.step_count = 0
        
        # State
        self.mel_stuck = False
        self.kl_stuck = False
        self.abort_requested = False
        self.abort_reason = ""
    
    def update(self, loss_mel: float, loss_kl: float, step: int, epoch: int) -> bool:
        """
        Update watchdog with new loss values.
        
        Returns True if training should ABORT.
        """
        self.step_count = step
        
        # Track raw values (NOT clamped!)
        self.mel_history.append(loss_mel)
        self.kl_history.append(loss_kl)
        
        # Need enough history to detect stuck
        if len(self.mel_history) < self.stuck_window:
            return False
        
        # Check for stuck mel
        mel_values = list(self.mel_history)
        mel_min, mel_max = min(mel_values), max(mel_values)
        self.mel_stuck = (mel_max - mel_min) < self.stuck_threshold
        
        # Check for stuck kl
        kl_values = list(self.kl_history)
        kl_min, kl_max = min(kl_values), max(kl_values)
        self.kl_stuck = (kl_max - kl_min) < self.stuck_threshold
        
        # CRITICAL: Both stuck = definite collapse
        if self.mel_stuck and self.kl_stuck:
            self.abort_reason = (
                f"TRAINING COLLAPSE DETECTED at step {step} (epoch {epoch}): "
                f"loss_mel stuck at {mel_values[-1]:.3f} and loss_kl stuck at {kl_values[-1]:.3f} "
                f"for {self.stuck_window} consecutive steps. "
                f"Model is NOT learning - aborting to save GPU time."
            )
            self.logger.error("=" * 60)
            self.logger.error(self.abort_reason)
            self.logger.error("=" * 60)
            self.abort_requested = True
            return self.abort_on_stuck
        
        # WARNING: One stuck (might recover)
        if self.mel_stuck:
            self.logger.warning(
                f"[WATCHDOG] loss_mel stuck at {mel_values[-1]:.3f} for {self.stuck_window} steps (step {step})"
            )
        if self.kl_stuck:
            self.logger.warning(
                f"[WATCHDOG] loss_kl stuck at {kl_values[-1]:.3f} for {self.stuck_window} steps (step {step})"
            )
        
        return False
    
    def should_abort(self) -> bool:
        return self.abort_requested
    
    def get_status(self) -> dict:
        return {
            'step_count': self.step_count,
            'mel_stuck': self.mel_stuck,
            'kl_stuck': self.kl_stuck,
            'abort_requested': self.abort_requested,
            'abort_reason': self.abort_reason,
        }

import datetime

from infer.lib.train import utils


# ============================================================================
# Checkpoint Control System
# ============================================================================

def check_checkpoint_request(model_dir: str) -> dict:
    """
    Check if there's a pending checkpoint request file.
    
    The control file format is: {model_dir}/.checkpoint_request.json
    Contents: {"action": "save_and_stop" | "save_and_continue", "requested_at": "..."}
    
    Returns empty dict if no request, otherwise returns the request.
    """
    request_file = Path(model_dir) / ".checkpoint_request.json"
    if request_file.exists():
        try:
            with open(request_file, 'r') as f:
                request = json.load(f)
            # Clear the request file
            request_file.unlink()
            return request
        except Exception as e:
            logger.warning(f"Failed to read checkpoint request: {e}")
    return {}


def write_checkpoint_response(model_dir: str, success: bool, checkpoint_path: str = "", error: str = ""):
    """
    Write checkpoint response file for the orchestrator to read.
    
    File: {model_dir}/.checkpoint_response.json
    """
    response_file = Path(model_dir) / ".checkpoint_response.json"
    response = {
        "success": success,
        "checkpoint_path": checkpoint_path,
        "error": error,
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    try:
        with open(response_file, 'w') as f:
            json.dump(response, f)
    except Exception as e:
        logger.warning(f"Failed to write checkpoint response: {e}")

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        from infer.modules.ipex.gradscaler import gradscaler_init
        from torch.xpu.amp import autocast

        GradScaler = gradscaler_init()
        ipex_init()
    else:
        from torch.cuda.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import savee

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    n_gpus = torch.cuda.device_count()

    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    logger = utils.get_logger(hps.model_dir)
    
    # For single GPU, run directly without multiprocessing overhead
    if n_gpus == 1:
        logger.info("Single GPU detected - running without multiprocessing")
        run(0, 1, hps, logger)
    else:
        # Multi-GPU: use process spawning
        children = []
        for i in range(n_gpus):
            subproc = mp.Process(
                target=run,
                args=(i, n_gpus, hps, logger),
            )
            children.append(subproc)
            subproc.start()

        for i in range(n_gpus):
            children[i].join()


def run(rank, n_gpus, hps, logger: logging.Logger):
    global global_step
    if rank == 0:
        # logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
        
        # Initialize training collapse watchdog
        watchdog = TrainingCollapseWatchdog(
            stuck_window=50,      # Detect stuck within 50 steps
            stuck_threshold=0.001,
            abort_on_stuck=True,
            logger=logger,
        )
        logger.info("[WATCHDOG] Training collapse detection ENABLED (window=50 steps)")
    else:
        watchdog = None

    # Only use distributed training for multi-GPU
    use_ddp = n_gpus > 1
    if use_ddp:
        dist.init_process_group(
            backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
        )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # Auto-adjust num_workers based on dataset size
    dataset_size = len(train_dataset)
    if dataset_size < 200:
        num_workers = 0  # Very small dataset: no multiprocessing overhead
    elif dataset_size < 1000:
        num_workers = 2  # Small/medium dataset: light prefetching
    elif dataset_size < 5000:
        num_workers = 4  # Medium/large dataset: moderate prefetching
    else:
        num_workers = 8  # Large dataset: full parallel prefetching
    
    # === STEP-BASED TRAINING METRICS (CRITICAL FOR CONVERGENCE) ===
    # This is logged at training start to help diagnose collapsed models.
    # Key insight: optimizer STEPS matter, not epochs!
    # Formula: total_steps = (num_segments / batch_size) * epochs
    batch_size = hps.train.batch_size
    total_epochs = hps.train.epochs
    steps_per_epoch = max(1, dataset_size // batch_size)
    estimated_total_steps = steps_per_epoch * total_epochs
    
    if rank == 0:
        logger.info("=" * 60)
        logger.info("[STEP-BASED TRAINING METRICS]")
        logger.info(f"  Dataset size:       {dataset_size} segments")
        logger.info(f"  Batch size:         {batch_size}")
        logger.info(f"  Steps per epoch:    {steps_per_epoch}")
        logger.info(f"  Total epochs:       {total_epochs}")
        logger.info(f"  ESTIMATED TOTAL STEPS: {estimated_total_steps}")
        logger.info("")
        
        # Warn if steps are too low (likely to collapse)
        MIN_RECOMMENDED_STEPS = 1500
        if estimated_total_steps < MIN_RECOMMENDED_STEPS:
            logger.warning("⚠️  WARNING: Only %d total steps planned!", estimated_total_steps)
            logger.warning("⚠️  Minimum recommended: %d steps", MIN_RECOMMENDED_STEPS)
            logger.warning("⚠️  Consider: reducing batch_size or increasing epochs")
            logger.warning("⚠️  Formula: batch_size=%d, dataset=%d → try batch_size=4", batch_size, dataset_size)
            suggested_epochs = max(1, (MIN_RECOMMENDED_STEPS + steps_per_epoch - 1) // steps_per_epoch)
            logger.warning("⚠️  Or increase epochs to at least %d", suggested_epochs)
        else:
            logger.info("✓ Step count looks healthy for convergence")
        
        logger.info("=" * 60)
        logger.info(f"Dataset size: {dataset_size} samples, using num_workers={num_workers}")
    
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    
    dataloader_kwargs = {
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": True,
        "collate_fn": collate_fn,
        "batch_sampler": train_sampler,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    # Only wrap in DDP for multi-GPU training
    if use_ddp:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            pass
        elif torch.cuda.is_available():
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )  ##测试不加载优化器
            else:
                logger.info(
                    net_g.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )  ##测试不加载优化器
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
                watchdog,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
                None,  # watchdog only on rank 0
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache, watchdog=None
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            # WATCHDOG: Check for training collapse with RAW loss values
            if watchdog is not None:
                # Get actual loss values (as float, not tensor)
                raw_loss_mel = float(loss_mel.item() if hasattr(loss_mel, 'item') else loss_mel)
                raw_loss_kl = float(loss_kl.item() if hasattr(loss_kl, 'item') else loss_kl)
                
                should_abort = watchdog.update(
                    loss_mel=raw_loss_mel,
                    loss_kl=raw_loss_kl,
                    step=global_step,
                    epoch=epoch,
                )
                
                if should_abort:
                    logger.error("=" * 60)
                    logger.error("WATCHDOG ABORT: Training collapse detected!")
                    logger.error(f"Reason: {watchdog.abort_reason}")
                    logger.error("Stopping training to prevent wasted GPU time.")
                    logger.error("=" * 60)
                    
                    # Write abort marker file
                    abort_file = Path(hps.model_dir) / ".training_aborted.json"
                    with open(abort_file, 'w') as f:
                        json.dump({
                            'reason': watchdog.abort_reason,
                            'step': global_step,
                            'epoch': epoch,
                            'status': watchdog.get_status(),
                        }, f, indent=2)
                    
                    sleep(1)
                    os._exit(99)  # Exit code 99 = watchdog abort
            
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                
                # Log RAW values first (for watchdog/debugging)
                raw_mel = float(loss_mel.item() if hasattr(loss_mel, 'item') else loss_mel)
                raw_kl = float(loss_kl.item() if hasattr(loss_kl, 'item') else loss_kl)
                
                # Clamp for Tensorboard display ONLY (not for logging!)
                display_mel = min(raw_mel, 75)
                display_kl = min(raw_kl, 9)

                logger.info([global_step, lr])
                # Log RAW values so we can detect stuck training
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={raw_mel:.3f}, loss_kl={raw_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                # Use clamped values for tensorboard display only
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": display_mel,
                        "loss/g/kl": display_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
        
        # === STEP-BASED EARLY SMOKE TEST ===
        # Save an early checkpoint and run smoke test at specific step milestones
        # This catches collapsed training BEFORE spending hours on a bad job
        SMOKE_TEST_STEP = getattr(hps, 'smoke_test_after_steps', 500)
        
        if rank == 0 and global_step == SMOKE_TEST_STEP:
            logger.info("=" * 60)
            logger.info(f"[SMOKE TEST] Reached step {global_step} - saving early checkpoint for quality check")
            
            # Save early checkpoint for smoke test
            early_ckpt_path = os.path.join(hps.model_dir, f"G_{global_step}.pth")
            utils.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch, early_ckpt_path
            )
            utils.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, f"D_{global_step}.pth")
            )
            
            logger.info(f"[SMOKE TEST] Early checkpoint saved: {early_ckpt_path}")
            logger.info(f"[SMOKE TEST] To run smoke test manually:")
            logger.info(f"[SMOKE TEST]   python -m app.trainer.training_watchdogs smoke_test {hps.model_dir}")
            logger.info("=" * 60)
            
            # Write marker file for external smoke test runner
            smoke_marker = Path(hps.model_dir) / ".smoke_test_ready.json"
            with open(smoke_marker, 'w') as f:
                json.dump({
                    'step': global_step,
                    'epoch': epoch,
                    'checkpoint': early_ckpt_path,
                    'timestamp': datetime.datetime.now().isoformat(),
                }, f, indent=2)
                
    # /Run steps

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
            )
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    
    # Check for checkpoint request (save & stop, save & continue)
    if rank == 0:
        request = check_checkpoint_request(hps.model_dir)
        if request:
            action = request.get("action", "")
            logger.info(f"Checkpoint request received: {action}")
            
            try:
                # Save checkpoint with proper naming
                g_path = os.path.join(hps.model_dir, "G_{}.pth".format(global_step))
                d_path = os.path.join(hps.model_dir, "D_{}.pth".format(global_step))
                
                utils.save_checkpoint(
                    net_g, optim_g, hps.train.learning_rate, epoch, g_path
                )
                utils.save_checkpoint(
                    net_d, optim_d, hps.train.learning_rate, epoch, d_path
                )
                
                # Also save extractable model
                if hasattr(net_g, "module"):
                    ckpt = net_g.module.state_dict()
                else:
                    ckpt = net_g.state_dict()
                
                saved_path = savee(
                    ckpt, hps.sample_rate, hps.if_f0,
                    hps.name + "_e%s_s%s" % (epoch, global_step),
                    epoch, hps.version, hps
                )
                
                logger.info(f"Checkpoint saved on request: {saved_path}")
                write_checkpoint_response(hps.model_dir, True, checkpoint_path=saved_path)
                
                if action == "save_and_stop":
                    logger.info("Stopping training as requested")
                    sleep(1)
                    os._exit(0)
                    
            except Exception as e:
                logger.error(f"Failed to save checkpoint on request: {e}")
                write_checkpoint_response(hps.model_dir, False, error=str(e))
    
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                savee(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        sleep(1)
        os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
