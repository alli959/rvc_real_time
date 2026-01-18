"""
Assets Management API Module

Provides endpoints for managing heavy assets/components:
- Bark TTS models
- RVC models
- UVR5 vocal separation
- Other GPU-heavy components
- Voice models discovery
- Pretrained models discovery
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin/assets", tags=["admin-assets"])


# =============================================================================
# Asset Types and Data Classes
# =============================================================================

class AssetType(str, Enum):
    """Type of asset"""
    MODEL = "model"
    SERVICE = "service"
    COMPONENT = "component"
    VOICE_MODEL = "voice_model"
    PRETRAINED = "pretrained"
    UVR_WEIGHT = "uvr_weight"


class AssetCategory(str, Enum):
    """Category for grouping assets in the UI"""
    CORE = "core"                # Core models (HuBERT, RMVPE, etc.)
    TTS = "tts"                  # Text-to-speech models
    VOICE_MODELS = "voice_models"  # RVC voice models
    PRETRAINED = "pretrained"    # Pretrained v2 models
    UVR_WEIGHTS = "uvr_weights"  # UVR5 weights
    PIPELINES = "pipelines"      # Inference/training pipelines


class ResourceType(str, Enum):
    """Type of resource the asset uses"""
    CPU = "cpu"
    GPU = "gpu"
    RAM = "ram"


class AssetStatus(str, Enum):
    """Asset status"""
    RUNNING = "running"          # Loaded and ready
    STOPPED = "stopped"          # Not loaded
    LOADING = "loading"          # Currently loading
    UNLOADING = "unloading"      # Currently unloading
    IN_USE = "in_use"            # Currently being used by a job
    ERROR = "error"
    UNKNOWN = "unknown"
    ON_DISK = "on_disk"          # File exists but not loaded


@dataclass
class AssetMetrics:
    """Resource usage metrics for an asset"""
    ram_usage_mb: Optional[float] = None
    vram_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    file_size_mb: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "ram_usage_mb": round(self.ram_usage_mb, 1) if self.ram_usage_mb else None,
            "vram_usage_mb": round(self.vram_usage_mb, 1) if self.vram_usage_mb else None,
            "cpu_percent": round(self.cpu_percent, 1) if self.cpu_percent else None,
            "file_size_mb": round(self.file_size_mb, 1) if self.file_size_mb else None,
        }


@dataclass
class Asset:
    """Represents a heavy asset that can be started/stopped"""
    id: str
    name: str
    description: str
    type: AssetType
    resource_type: ResourceType
    category: AssetCategory = AssetCategory.CORE
    status: AssetStatus = AssetStatus.STOPPED
    container: Optional[str] = None
    process_name: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metrics: AssetMetrics = field(default_factory=AssetMetrics)
    estimated_ram_mb: Optional[float] = None
    estimated_vram_mb: Optional[float] = None
    last_status_check: Optional[str] = None
    file_path: Optional[str] = None
    can_preload: bool = True
    # Tracking fields for load/usage timing
    loaded_at: Optional[str] = None  # ISO timestamp when asset was loaded
    last_used_at: Optional[str] = None  # ISO timestamp when asset was last used
    use_count: int = 0  # Number of times asset has been used since load
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "resource_type": self.resource_type.value,
            "category": self.category.value,
            "status": self.status.value,
            "container": self.container,
            "process_name": self.process_name,
            "dependencies": self.dependencies,
            "metrics": self.metrics.to_dict(),
            "estimated_ram_mb": self.estimated_ram_mb,
            "estimated_vram_mb": self.estimated_vram_mb,
            "last_status_check": self.last_status_check,
            "file_path": self.file_path,
            "can_preload": self.can_preload,
            "loaded_at": self.loaded_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
        }


# =============================================================================
# Asset Registry - Core Models
# =============================================================================

# Known heavy assets in the system
ASSET_REGISTRY: Dict[str, Asset] = {
    "bark_tts": Asset(
        id="bark_tts",
        name="Bark TTS",
        description="Suno Bark text-to-speech model for natural speech generation",
        type=AssetType.MODEL,
        resource_type=ResourceType.GPU,
        category=AssetCategory.TTS,
        process_name="bark",
        estimated_ram_mb=2000,
        estimated_vram_mb=3900,
        dependencies=[],
    ),
    "hubert": Asset(
        id="hubert",
        name="HuBERT Model",
        description="HuBERT base model for voice feature extraction",
        type=AssetType.MODEL,
        resource_type=ResourceType.GPU,
        category=AssetCategory.CORE,
        process_name="hubert",
        estimated_ram_mb=500,
        estimated_vram_mb=1000,
        dependencies=[],
    ),
    "rmvpe": Asset(
        id="rmvpe",
        name="RMVPE F0 Extractor",
        description="RMVPE model for pitch extraction",
        type=AssetType.MODEL,
        resource_type=ResourceType.GPU,
        category=AssetCategory.CORE,
        process_name="rmvpe",
        estimated_ram_mb=200,
        estimated_vram_mb=500,
        dependencies=[],
        file_path="/app/assets/rmvpe/rmvpe.pt",
    ),
    "uvr5": Asset(
        id="uvr5",
        name="UVR5 Vocal Separator",
        description="Ultimate Vocal Remover for vocal/instrumental separation",
        type=AssetType.MODEL,
        resource_type=ResourceType.GPU,
        category=AssetCategory.CORE,
        process_name="uvr5",
        estimated_ram_mb=2000,
        estimated_vram_mb=2000,
        dependencies=[],
    ),
    "rvc_inference": Asset(
        id="rvc_inference",
        name="RVC Inference Engine",
        description="RVC voice conversion inference pipeline",
        type=AssetType.COMPONENT,
        resource_type=ResourceType.GPU,
        category=AssetCategory.PIPELINES,
        process_name="rvc",
        estimated_ram_mb=1000,
        estimated_vram_mb=2000,
        dependencies=["hubert", "rmvpe"],
    ),
    "training_pipeline": Asset(
        id="training_pipeline",
        name="Training Pipeline",
        description="RVC model training pipeline",
        type=AssetType.COMPONENT,
        resource_type=ResourceType.GPU,
        category=AssetCategory.PIPELINES,
        process_name="train",
        estimated_ram_mb=4000,
        estimated_vram_mb=8000,
        dependencies=["hubert", "rmvpe"],
    ),
    "edge_tts": Asset(
        id="edge_tts",
        name="Edge TTS",
        description="Microsoft Edge text-to-speech (cloud-based, low resource)",
        type=AssetType.SERVICE,
        resource_type=ResourceType.CPU,
        category=AssetCategory.TTS,
        process_name="edge-tts",
        estimated_ram_mb=50,
        estimated_vram_mb=0,
        dependencies=[],
    ),
    "whisper": Asset(
        id="whisper",
        name="Whisper STT",
        description="OpenAI Whisper speech-to-text model",
        type=AssetType.MODEL,
        resource_type=ResourceType.GPU,
        category=AssetCategory.CORE,
        process_name="whisper",
        estimated_ram_mb=1500,
        estimated_vram_mb=2000,
        dependencies=[],
    ),
}


class AssetManager:
    """
    Manages heavy assets lifecycle.
    
    Tracks which assets are loaded and provides controls to
    start/stop them to manage resources.
    """
    
    def __init__(self):
        self._assets: Dict[str, Asset] = ASSET_REGISTRY.copy()
        self._loaded_models: Dict[str, bool] = {}
        self._loaded_voice_models: Set[str] = set()  # Currently loaded voice model IDs
        self._active_jobs: Dict[str, List[str]] = {}  # job_id -> [asset_ids]
        self._in_use_assets: Set[str] = set()  # Assets currently being used
        self._discovery_cache: Dict[str, List[Asset]] = {}
        self._last_discovery: Optional[datetime] = None
        
        # Register usage callback with the model manager
        self._register_usage_callback()
    
    def _register_usage_callback(self):
        """Register callback to track voice model usage from inference."""
        # Register voice conversion callback
        try:
            from app.services.voice_conversion import set_usage_callback
            
            def on_model_used(model_name: str):
                """Called when a voice model is used for inference."""
                # Find the asset by model name and record usage
                for asset_id, asset in self._assets.items():
                    if asset.category == AssetCategory.VOICE_MODELS:
                        # Match by model file name or asset name
                        if model_name and (
                            model_name in asset.name or 
                            model_name in asset.id or
                            (asset.file_path and model_name in asset.file_path)
                        ):
                            self.record_asset_usage(asset_id)
                            return
            
            set_usage_callback(on_model_used)
            logger.info("Voice model usage callback registered with AssetManager")
        except Exception as e:
            logger.warning(f"Could not register voice model usage callback: {e}")
        
        # Register UVR5 callback
        try:
            from app.vocal_separator import set_uvr_usage_callback
            
            def on_uvr_used(model_name: str):
                """Called when a UVR5 model is used for separation."""
                # Find the UVR asset by model name
                for asset_id, asset in self._assets.items():
                    if asset.category == AssetCategory.UVR_WEIGHTS:
                        if model_name and (
                            model_name in asset.name or
                            model_name in asset.id or
                            (asset.file_path and model_name in asset.file_path)
                        ):
                            self.record_asset_usage(asset_id)
                            return
                # Also track the main uvr5 asset
                if "uvr5" in self._assets:
                    self.record_asset_usage("uvr5")
            
            set_uvr_usage_callback(on_uvr_used)
            logger.info("UVR5 usage callback registered with AssetManager")
        except Exception as e:
            logger.warning(f"Could not register UVR5 usage callback: {e}")
    
    async def get_all_assets(self, category: Optional[str] = None) -> List[Asset]:
        """Get all registered assets with updated status"""
        # Re-discover dynamic assets periodically
        await self._discover_dynamic_assets()
        
        assets = []
        for asset_id, asset in self._assets.items():
            if category and asset.category.value != category:
                continue
            # Update status
            asset.status = await self._check_asset_status(asset)
            asset.last_status_check = datetime.utcnow().isoformat() + "Z"
            assets.append(asset)
        return assets
    
    async def get_assets_by_category(self) -> Dict[str, List[Asset]]:
        """Get all assets grouped by category"""
        await self._discover_dynamic_assets()
        
        by_category: Dict[str, List[Asset]] = {}
        for asset_id, asset in self._assets.items():
            category = asset.category.value
            if category not in by_category:
                by_category[category] = []
            asset.status = await self._check_asset_status(asset)
            asset.last_status_check = datetime.utcnow().isoformat() + "Z"
            by_category[category].append(asset)
        
        return by_category
    
    async def _discover_dynamic_assets(self):
        """Discover voice models, pretrained models, and UVR weights from disk"""
        # Only re-discover every 30 seconds
        now = datetime.utcnow()
        if self._last_discovery and (now - self._last_discovery).total_seconds() < 30:
            return
        
        self._last_discovery = now
        
        # Discover pretrained_v2 models
        await self._discover_pretrained_models()
        
        # Discover UVR5 weights
        await self._discover_uvr_weights()
        
        # Discover voice models
        await self._discover_voice_models()
    
    async def _discover_pretrained_models(self):
        """Discover pretrained_v2 models"""
        pretrained_dir = Path("/app/assets/pretrained_v2")
        if not pretrained_dir.exists():
            return
        
        for file_path in pretrained_dir.glob("*.pth"):
            if not file_path.exists():
                continue
            asset_id = f"pretrained_{file_path.stem}"
            if asset_id not in self._assets:
                try:
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                except (FileNotFoundError, OSError):
                    continue
                
                # Parse model type from filename
                name = file_path.stem
                if name.startswith("f0D"):
                    desc = f"Discriminator model ({name[-3:]}Hz)"
                    model_type = "discriminator"
                elif name.startswith("f0G"):
                    desc = f"Generator model ({name[-3:]}Hz)"
                    model_type = "generator"
                else:
                    desc = f"Pretrained model"
                    model_type = "other"
                
                self._assets[asset_id] = Asset(
                    id=asset_id,
                    name=name,
                    description=desc,
                    type=AssetType.PRETRAINED,
                    resource_type=ResourceType.GPU,
                    category=AssetCategory.PRETRAINED,
                    estimated_ram_mb=file_size,
                    estimated_vram_mb=file_size * 0.8,
                    file_path=str(file_path),
                    can_preload=False,  # Pretrained models are loaded during training only
                    metrics=AssetMetrics(file_size_mb=file_size),
                )
    
    async def _discover_uvr_weights(self):
        """Discover UVR5 weight files"""
        uvr_dir = Path("/app/assets/uvr5_weights")
        if not uvr_dir.exists():
            return
        
        for file_path in uvr_dir.glob("*.pth"):
            if not file_path.exists():
                continue
            asset_id = f"uvr_{file_path.stem}"
            if asset_id not in self._assets:
                try:
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                except (FileNotFoundError, OSError):
                    continue
                name = file_path.stem.replace("_", " ")
                
                self._assets[asset_id] = Asset(
                    id=asset_id,
                    name=name,
                    description=f"UVR5 weight for vocal separation",
                    type=AssetType.UVR_WEIGHT,
                    resource_type=ResourceType.GPU,
                    category=AssetCategory.UVR_WEIGHTS,
                    estimated_ram_mb=file_size,
                    estimated_vram_mb=file_size * 0.8,
                    file_path=str(file_path),
                    can_preload=True,
                    metrics=AssetMetrics(file_size_mb=file_size),
                )
    
    async def _discover_voice_models(self):
        """Discover RVC voice models"""
        models_dir = Path("/app/assets/models")
        if not models_dir.exists():
            return
        
        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            asset_id = f"voice_{model_dir.name}"
            if asset_id not in self._assets:
                try:
                    # Look for .pth and .index files
                    pth_files = [f for f in model_dir.glob("*.pth") if f.exists()]
                    index_files = [f for f in model_dir.glob("*.index") if f.exists()]
                    
                    if not pth_files:
                        continue
                    
                    # Calculate total size with error handling
                    total_size = 0
                    for f in pth_files + index_files:
                        try:
                            total_size += f.stat().st_size
                        except (FileNotFoundError, OSError):
                            continue
                    total_size = total_size / (1024 * 1024)
                    
                    self._assets[asset_id] = Asset(
                        id=asset_id,
                        name=model_dir.name,
                        description=f"RVC voice model",
                        type=AssetType.VOICE_MODEL,
                        resource_type=ResourceType.GPU,
                        category=AssetCategory.VOICE_MODELS,
                        estimated_ram_mb=total_size,
                        estimated_vram_mb=total_size * 0.8,
                        file_path=str(model_dir),
                        can_preload=True,
                        metrics=AssetMetrics(file_size_mb=total_size),
                    )
                except Exception as e:
                    # Skip problematic directories
                    logger.warning(f"Error discovering voice model {model_dir.name}: {e}")
                    continue
    
    def mark_asset_in_use(self, asset_id: str):
        """Mark an asset as currently being used"""
        self._in_use_assets.add(asset_id)
    
    def mark_asset_not_in_use(self, asset_id: str):
        """Mark an asset as no longer being used"""
        self._in_use_assets.discard(asset_id)
    
    def mark_voice_model_loaded(self, model_name: str):
        """Mark a voice model as loaded"""
        self._loaded_voice_models.add(model_name)
    
    def mark_voice_model_unloaded(self, model_name: str):
        """Mark a voice model as unloaded"""
        self._loaded_voice_models.discard(model_name)
    
    async def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get a specific asset"""
        await self._discover_dynamic_assets()
        asset = self._assets.get(asset_id)
        if asset:
            asset.status = await self._check_asset_status(asset)
            asset.last_status_check = datetime.utcnow().isoformat() + "Z"
        return asset
    
    async def start_asset(self, asset_id: str) -> Dict:
        """
        Start/load an asset.
        
        Returns:
            Status dict with success/error info
        """
        asset = self._assets.get(asset_id)
        if not asset:
            raise ValueError(f"Asset not found: {asset_id}")
        
        # Check dependencies
        for dep_id in asset.dependencies:
            dep_asset = self._assets.get(dep_id)
            if dep_asset:
                dep_status = await self._check_asset_status(dep_asset)
                if dep_status != AssetStatus.RUNNING:
                    # Start dependency first
                    await self.start_asset(dep_id)
        
        # Start the asset
        asset.status = AssetStatus.LOADING
        
        try:
            success = await self._load_asset(asset)
            if success:
                asset.status = AssetStatus.RUNNING
                asset.loaded_at = datetime.utcnow().isoformat() + "Z"
                asset.use_count = 0  # Reset use count on load
                self._loaded_models[asset_id] = True
                logger.info(f"Asset {asset_id} loaded at {asset.loaded_at}")
                return {
                    "success": True,
                    "asset_id": asset_id,
                    "status": asset.status.value,
                    "loaded_at": asset.loaded_at,
                }
            else:
                asset.status = AssetStatus.ERROR
                return {
                    "success": False,
                    "asset_id": asset_id,
                    "error": "Failed to load asset",
                }
        except Exception as e:
            asset.status = AssetStatus.ERROR
            logger.error(f"Error starting asset {asset_id}: {e}")
            return {
                "success": False,
                "asset_id": asset_id,
                "error": str(e),
            }
    
    async def stop_asset(self, asset_id: str, force: bool = False) -> Dict:
        """
        Stop/unload an asset.
        
        Args:
            asset_id: ID of asset to stop
            force: If True, stop even if jobs are using it
        
        Returns:
            Status dict with success/error info
        """
        asset = self._assets.get(asset_id)
        if not asset:
            raise ValueError(f"Asset not found: {asset_id}")
        
        # Check if any jobs are using this asset
        if not force:
            using_jobs = self._get_jobs_using_asset(asset_id)
            if using_jobs:
                return {
                    "success": False,
                    "asset_id": asset_id,
                    "error": "Asset is in use by active jobs",
                    "jobs": using_jobs,
                    "requires_force": True,
                }
        
        # Check if other assets depend on this one
        dependents = self._get_dependent_assets(asset_id)
        running_dependents = []
        for dep_id in dependents:
            dep_asset = self._assets.get(dep_id)
            if dep_asset and await self._check_asset_status(dep_asset) == AssetStatus.RUNNING:
                running_dependents.append(dep_id)
        
        if running_dependents and not force:
            return {
                "success": False,
                "asset_id": asset_id,
                "error": "Other assets depend on this one",
                "dependents": running_dependents,
                "requires_force": True,
            }
        
        # Stop dependent assets first if forcing
        if force and running_dependents:
            for dep_id in running_dependents:
                await self.stop_asset(dep_id, force=True)
        
        # Stop the asset
        asset.status = AssetStatus.UNLOADING
        
        try:
            success = await self._unload_asset(asset)
            if success:
                asset.status = AssetStatus.STOPPED
                # Clear tracking info on unload
                asset.loaded_at = None
                asset.last_used_at = None
                asset.use_count = 0
                self._loaded_models.pop(asset_id, None)
                logger.info(f"Asset {asset_id} unloaded")
                return {
                    "success": True,
                    "asset_id": asset_id,
                    "status": asset.status.value,
                }
            else:
                asset.status = AssetStatus.ERROR
                return {
                    "success": False,
                    "asset_id": asset_id,
                    "error": "Failed to unload asset",
                }
        except Exception as e:
            asset.status = AssetStatus.ERROR
            logger.error(f"Error stopping asset {asset_id}: {e}")
            return {
                "success": False,
                "asset_id": asset_id,
                "error": str(e),
            }
    
    def record_asset_usage(self, asset_id: str) -> None:
        """
        Record that an asset was just used.
        Updates last_used_at timestamp and increments use_count.
        
        Call this whenever an asset is used for inference/processing.
        """
        asset = self._assets.get(asset_id)
        if asset and asset.status in [AssetStatus.RUNNING, AssetStatus.IN_USE]:
            asset.last_used_at = datetime.utcnow().isoformat() + "Z"
            asset.use_count += 1
            logger.debug(f"Asset {asset_id} used (count: {asset.use_count})")
    
    def get_actual_memory(self) -> Dict:
        """
        Get actual system/container memory usage from /proc/meminfo.
        
        Returns:
            Dict with actual memory stats in MB and bytes
        """
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().split()[0]
                        meminfo[key] = int(value) * 1024  # Convert kB to bytes
            
            total = meminfo.get('MemTotal', 0)
            available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
            used = total - available
            
            # Get GPU memory if available
            gpu_used_mb = 0
            gpu_total_mb = 0
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) == 2:
                            gpu_used_mb += int(parts[0].strip())
                            gpu_total_mb += int(parts[1].strip())
            except Exception:
                pass  # nvidia-smi not available
            
            return {
                "ram_used_bytes": used,
                "ram_total_bytes": total,
                "ram_available_bytes": available,
                "ram_used_mb": round(used / (1024 * 1024), 1),
                "ram_total_mb": round(total / (1024 * 1024), 1),
                "ram_used_percent": round(used / total * 100, 1) if total > 0 else 0,
                "vram_used_mb": gpu_used_mb,
                "vram_total_mb": gpu_total_mb,
                "vram_used_percent": round(gpu_used_mb / gpu_total_mb * 100, 1) if gpu_total_mb > 0 else 0,
            }
        except Exception as e:
            logger.warning(f"Error getting actual memory: {e}")
            return {
                "ram_used_bytes": 0,
                "ram_total_bytes": 0,
                "ram_available_bytes": 0,
                "ram_used_mb": 0,
                "ram_total_mb": 0,
                "ram_used_percent": 0,
                "vram_used_mb": 0,
                "vram_total_mb": 0,
                "vram_used_percent": 0,
            }
    
    def register_job_assets(self, job_id: str, asset_ids: List[str]):
        """Register that a job is using certain assets"""
        self._active_jobs[job_id] = asset_ids
    
    def unregister_job(self, job_id: str):
        """Unregister a job's asset usage"""
        self._active_jobs.pop(job_id, None)
    
    def _get_jobs_using_asset(self, asset_id: str) -> List[str]:
        """Get list of jobs using an asset"""
        return [
            job_id for job_id, assets in self._active_jobs.items()
            if asset_id in assets
        ]
    
    def _get_dependent_assets(self, asset_id: str) -> List[str]:
        """Get assets that depend on this one"""
        return [
            aid for aid, asset in self._assets.items()
            if asset_id in asset.dependencies
        ]
    
    async def _check_asset_status(self, asset: Asset) -> AssetStatus:
        """Check the actual status of an asset by introspecting runtime state"""
        # Check if asset is marked as in use
        if asset.id in self._in_use_assets:
            return AssetStatus.IN_USE
        
        # Check based on asset type - introspect actual global state
        if asset.id == "bark_tts":
            return await self._check_bark_loaded()
        elif asset.id == "hubert":
            return await self._check_hubert_loaded()
        elif asset.id == "rmvpe":
            return await self._check_rmvpe_loaded()
        elif asset.id == "uvr5":
            return await self._check_uvr5_loaded()
        elif asset.id == "rvc_inference":
            # RVC is loaded if hubert is loaded
            return await self._check_hubert_loaded()
        elif asset.id == "training_pipeline":
            return await self._check_training_running()
        elif asset.id == "edge_tts":
            # Edge TTS is always available (cloud-based)
            return AssetStatus.RUNNING
        elif asset.id == "whisper":
            return await self._check_whisper_loaded()
        elif asset.type == AssetType.SERVICE:
            return await self._check_service_running(asset)
        elif asset.type == AssetType.VOICE_MODEL:
            return await self._check_voice_model_loaded(asset)
        elif asset.type == AssetType.PRETRAINED:
            # Pretrained models are on disk - they're loaded during training only
            return AssetStatus.ON_DISK if asset.file_path and Path(asset.file_path).exists() else AssetStatus.UNKNOWN
        elif asset.type == AssetType.UVR_WEIGHT:
            return await self._check_uvr_weight_loaded(asset)
        else:
            return AssetStatus.UNKNOWN
    
    async def _check_voice_model_loaded(self, asset: Asset) -> AssetStatus:
        """Check if a voice model is loaded"""
        model_name = asset.name
        if model_name in self._loaded_voice_models:
            return AssetStatus.RUNNING
        
        # Check if the model files exist
        if asset.file_path and Path(asset.file_path).exists():
            return AssetStatus.ON_DISK
        
        return AssetStatus.UNKNOWN
    
    async def _check_uvr_weight_loaded(self, asset: Asset) -> AssetStatus:
        """Check if a UVR weight is loaded"""
        # Check if UVR5 is currently using this specific weight
        try:
            from app import vocal_separator
            if hasattr(vocal_separator, '_current_weight') and vocal_separator._current_weight:
                if asset.name.lower() in vocal_separator._current_weight.lower():
                    return AssetStatus.RUNNING
        except:
            pass
        
        # Weight exists on disk
        if asset.file_path and Path(asset.file_path).exists():
            return AssetStatus.ON_DISK
        
        return AssetStatus.UNKNOWN
    
    async def _check_bark_loaded(self) -> AssetStatus:
        """Check if Bark TTS is actually loaded"""
        try:
            from app.tts_service import BARK_AVAILABLE
            if not BARK_AVAILABLE:
                return AssetStatus.STOPPED
            
            # Check if bark models are preloaded
            try:
                from bark.generation import SEMANTIC_MODEL, COARSE_MODEL, FINE_MODEL
                if SEMANTIC_MODEL is not None:
                    self._loaded_models["bark_tts"] = True
                    return AssetStatus.RUNNING
            except:
                pass
            
            return AssetStatus.STOPPED
        except ImportError:
            return AssetStatus.STOPPED
    
    async def _check_hubert_loaded(self) -> AssetStatus:
        """Check if HuBERT model is actually loaded"""
        try:
            # Check if there's a global model manager with hubert loaded
            from app import http_api
            if hasattr(http_api, '_model_manager') and http_api._model_manager:
                if hasattr(http_api._model_manager, 'vc'):
                    vc = http_api._model_manager.vc
                    if hasattr(vc, 'hubert_model') and vc.hubert_model is not None:
                        self._loaded_models["hubert"] = True
                        return AssetStatus.RUNNING
            return AssetStatus.STOPPED
        except Exception as e:
            logger.debug(f"Error checking hubert: {e}")
            return AssetStatus.UNKNOWN
    
    async def _check_rmvpe_loaded(self) -> AssetStatus:
        """Check if RMVPE model is actually loaded"""
        try:
            from app import http_api
            if hasattr(http_api, '_model_manager') and http_api._model_manager:
                if hasattr(http_api._model_manager, 'vc'):
                    vc = http_api._model_manager.vc
                    if hasattr(vc, 'rmvpe_model') and vc.rmvpe_model is not None:
                        self._loaded_models["rmvpe"] = True
                        return AssetStatus.RUNNING
            return AssetStatus.STOPPED
        except Exception as e:
            logger.debug(f"Error checking rmvpe: {e}")
            return AssetStatus.UNKNOWN
    
    async def _check_uvr5_loaded(self) -> AssetStatus:
        """Check if UVR5 model is actually loaded"""
        try:
            from app import vocal_separator
            if hasattr(vocal_separator, '_uvr_model') and vocal_separator._uvr_model is not None:
                self._loaded_models["uvr5"] = True
                return AssetStatus.RUNNING
            return AssetStatus.STOPPED
        except Exception as e:
            logger.debug(f"Error checking uvr5: {e}")
            return AssetStatus.UNKNOWN
    
    async def _check_whisper_loaded(self) -> AssetStatus:
        """Check if Whisper model is actually loaded"""
        try:
            # Check for whisper module and loaded model
            import sys
            if 'whisper' in sys.modules:
                whisper_mod = sys.modules['whisper']
                # If the module is loaded, it's likely in use
                self._loaded_models["whisper"] = True
                return AssetStatus.RUNNING
            return AssetStatus.STOPPED
        except Exception as e:
            logger.debug(f"Error checking whisper: {e}")
            return AssetStatus.UNKNOWN
    
    async def _check_training_running(self) -> AssetStatus:
        """Check if any training job is currently running"""
        try:
            from app.trainer_api import get_pipeline
            pipeline = get_pipeline()
            if pipeline:
                jobs = pipeline.get_active_jobs() if hasattr(pipeline, 'get_active_jobs') else []
                if jobs:
                    self._loaded_models["training_pipeline"] = True
                    return AssetStatus.RUNNING
            return AssetStatus.STOPPED
        except Exception as e:
            logger.debug(f"Error checking training: {e}")
            return AssetStatus.UNKNOWN
    
    async def _check_service_running(self, asset: Asset) -> AssetStatus:
        """Check if a service container is running"""
        if not asset.container:
            return AssetStatus.UNKNOWN
        
        try:
            result = await asyncio.create_subprocess_exec(
                "docker", "inspect", "-f", "{{.State.Running}}", asset.container,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(result.communicate(), timeout=5)
            
            if result.returncode == 0 and stdout.decode().strip() == "true":
                return AssetStatus.RUNNING
            return AssetStatus.STOPPED
            
        except Exception:
            return AssetStatus.UNKNOWN
    
    async def _load_asset(self, asset: Asset) -> bool:
        """Load an asset into memory"""
        logger.info(f"Loading asset: {asset.id}")
        
        # Import the actual loaders based on asset type
        if asset.id == "bark_tts":
            return await self._load_bark()
        elif asset.id == "hubert":
            return await self._load_hubert()
        elif asset.id == "rmvpe":
            return await self._load_rmvpe()
        elif asset.id == "uvr5":
            return await self._load_uvr5()
        else:
            # For components, just mark as loaded
            self._loaded_models[asset.id] = True
            return True
    
    async def _unload_asset(self, asset: Asset) -> bool:
        """Unload an asset from memory"""
        logger.info(f"Unloading asset: {asset.id}")
        
        if asset.id == "bark_tts":
            return await self._unload_bark()
        elif asset.id == "hubert":
            return await self._unload_hubert()
        elif asset.id == "rmvpe":
            return await self._unload_rmvpe()
        elif asset.id == "uvr5":
            return await self._unload_uvr5()
        else:
            self._loaded_models.pop(asset.id, None)
            return True
    
    # Asset-specific load/unload implementations
    
    async def _load_bark(self) -> bool:
        """Load Bark TTS models"""
        try:
            from app.tts_service import preload_bark_models
            preload_bark_models()
            return True
        except Exception as e:
            logger.error(f"Failed to load Bark: {e}")
            return False
    
    async def _unload_bark(self) -> bool:
        """Unload Bark TTS models"""
        try:
            import gc
            import torch
            
            # Clear bark models from global state
            from app import tts_service
            if hasattr(tts_service, '_bark_model'):
                del tts_service._bark_model
            if hasattr(tts_service, '_bark_processor'):
                del tts_service._bark_processor
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
        except Exception as e:
            logger.error(f"Failed to unload Bark: {e}")
            return False
    
    async def _load_hubert(self) -> bool:
        """Load HuBERT model"""
        try:
            # HuBERT is typically loaded on demand by the training pipeline
            # For preloading, we'd need to import and initialize it
            logger.info("HuBERT will be loaded on demand")
            self._loaded_models["hubert"] = True
            return True
        except Exception as e:
            logger.error(f"Failed to load HuBERT: {e}")
            return False
    
    async def _unload_hubert(self) -> bool:
        """Unload HuBERT model"""
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to unload HuBERT: {e}")
            return False
    
    async def _load_rmvpe(self) -> bool:
        """Load RMVPE model"""
        try:
            logger.info("RMVPE will be loaded on demand")
            self._loaded_models["rmvpe"] = True
            return True
        except Exception as e:
            logger.error(f"Failed to load RMVPE: {e}")
            return False
    
    async def _unload_rmvpe(self) -> bool:
        """Unload RMVPE model"""
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to unload RMVPE: {e}")
            return False
    
    async def _load_uvr5(self) -> bool:
        """Load UVR5 models"""
        try:
            logger.info("UVR5 will be loaded on demand")
            self._loaded_models["uvr5"] = True
            return True
        except Exception as e:
            logger.error(f"Failed to load UVR5: {e}")
            return False
    
    async def _unload_uvr5(self) -> bool:
        """Unload UVR5 models"""
        try:
            import gc
            import torch
            
            # Clear UVR5 model cache if exists
            from app import vocal_separator
            if hasattr(vocal_separator, '_model_cache'):
                vocal_separator._model_cache.clear()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to unload UVR5: {e}")
            return False


# Global asset manager
_asset_manager: Optional[AssetManager] = None


def get_asset_manager() -> AssetManager:
    """Get or create the global asset manager"""
    global _asset_manager
    if _asset_manager is None:
        _asset_manager = AssetManager()
    return _asset_manager


def _build_category_summary(by_category: Dict[str, List[Asset]]) -> Dict:
    """Build summary data for WebSocket updates"""
    result = {}
    total_running = 0
    total_vram = 0
    total_ram = 0
    all_assets = []
    
    for cat_name, cat_assets in by_category.items():
        running = sum(1 for a in cat_assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
        vram = sum(a.estimated_vram_mb or 0 for a in cat_assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
        ram = sum(a.estimated_ram_mb or 0 for a in cat_assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
        
        total_running += running
        total_vram += vram
        total_ram += ram
        
        result[cat_name] = {
            "name": cat_name.replace("_", " ").title(),
            "assets": [a.to_dict() for a in cat_assets],
            "count": len(cat_assets),
            "running": running,
            "vram_mb": vram,
            "ram_mb": ram,
        }
        all_assets.extend(cat_assets)
    
    # Get actual memory usage
    manager = get_asset_manager()
    actual_memory = manager.get_actual_memory()
    
    return {
        "categories": result,
        "assets": [a.to_dict() for a in all_assets],  # Flat list for backwards compatibility
        "total": len(all_assets),
        "summary": {
            "total_running": total_running,
            "total_vram_mb": total_vram,
            "total_ram_mb": total_ram,
        },
        "actual_memory": actual_memory,
    }


# =============================================================================
# REST Endpoints
# =============================================================================

@router.get("")
async def list_assets(category: Optional[str] = None):
    """List all registered assets with their status, optionally filtered by category"""
    manager = get_asset_manager()
    assets = await manager.get_all_assets(category=category)
    
    # Get summary stats
    running_count = sum(1 for a in assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
    total_vram = sum(a.estimated_vram_mb or 0 for a in assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
    total_ram = sum(a.estimated_ram_mb or 0 for a in assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
    
    return {
        "assets": [a.to_dict() for a in assets],
        "total": len(assets),
        "stats": {
            "running": running_count,
            "total_vram_mb": total_vram,
            "total_ram_mb": total_ram,
        }
    }


@router.get("/categories")
async def list_categories():
    """List available asset categories with counts"""
    manager = get_asset_manager()
    by_category = await manager.get_assets_by_category()
    
    categories = []
    for cat_name, cat_assets in by_category.items():
        running = sum(1 for a in cat_assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
        categories.append({
            "id": cat_name,
            "name": cat_name.replace("_", " ").title(),
            "count": len(cat_assets),
            "running": running,
        })
    
    return {"categories": categories}


@router.get("/by-category")
async def get_assets_by_category():
    """Get all assets grouped by category"""
    manager = get_asset_manager()
    by_category = await manager.get_assets_by_category()
    
    result = {}
    total_running = 0
    total_vram = 0
    total_ram = 0
    
    for cat_name, cat_assets in by_category.items():
        running = sum(1 for a in cat_assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
        vram = sum(a.estimated_vram_mb or 0 for a in cat_assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
        ram = sum(a.estimated_ram_mb or 0 for a in cat_assets if a.status in [AssetStatus.RUNNING, AssetStatus.IN_USE])
        
        total_running += running
        total_vram += vram
        total_ram += ram
        
        result[cat_name] = {
            "name": cat_name.replace("_", " ").title(),
            "assets": [a.to_dict() for a in cat_assets],
            "count": len(cat_assets),
            "running": running,
            "vram_mb": vram,
            "ram_mb": ram,
        }
    
    # Get actual memory usage
    actual_memory = manager.get_actual_memory()
    
    return {
        "categories": result,
        "summary": {
            "total_running": total_running,
            "total_vram_mb": total_vram,
            "total_ram_mb": total_ram,
        },
        "actual_memory": actual_memory,
    }


@router.get("/{asset_id}")
async def get_asset(asset_id: str):
    """Get info for a specific asset"""
    manager = get_asset_manager()
    asset = await manager.get_asset(asset_id)
    
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")
    
    return asset.to_dict()


@router.post("/{asset_id}/start")
async def start_asset(asset_id: str):
    """Start/load an asset"""
    manager = get_asset_manager()
    
    try:
        result = await manager.start_asset(asset_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{asset_id}/stop")
async def stop_asset(asset_id: str, force: bool = False):
    """
    Stop/unload an asset.
    
    Use force=true to stop even if jobs are using it.
    """
    manager = get_asset_manager()
    
    try:
        result = await manager.stop_asset(asset_id, force=force)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{asset_id}/restart")
async def restart_asset(asset_id: str):
    """Restart an asset (stop + start)"""
    manager = get_asset_manager()
    
    try:
        # Stop first
        stop_result = await manager.stop_asset(asset_id, force=True)
        if not stop_result.get("success"):
            return {
                "success": False,
                "phase": "stop",
                "error": stop_result.get("error"),
            }
        
        # Small delay
        await asyncio.sleep(1)
        
        # Start
        start_result = await manager.start_asset(asset_id)
        return {
            "success": start_result.get("success"),
            "phase": "start",
            "error": start_result.get("error"),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WebSocket Endpoint for Real-time Asset Status Updates
# =============================================================================

@router.websocket("/stream")
async def stream_assets(websocket: WebSocket):
    """
    WebSocket endpoint for real-time asset status updates.
    
    Updates at ~2 second intervals with current asset status.
    """
    await websocket.accept()
    
    manager = get_asset_manager()
    interval = 2.0  # Default 2 seconds
    
    async def handle_client_messages():
        nonlocal interval
        try:
            while True:
                data = await websocket.receive_json()
                if "interval" in data:
                    interval = max(1.0, min(10.0, data["interval"] / 1000))
                    await websocket.send_json({
                        "type": "config",
                        "interval": int(interval * 1000),
                    })
                elif data.get("action") == "refresh":
                    # Force immediate refresh with categorized data
                    by_category = await manager.get_assets_by_category()
                    summary = _build_category_summary(by_category)
                    await websocket.send_json({
                        "type": "assets",
                        "data": summary,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"Client message error: {e}")
    
    client_task = asyncio.create_task(handle_client_messages())
    
    try:
        while True:
            by_category = await manager.get_assets_by_category()
            summary = _build_category_summary(by_category)
            await websocket.send_json({
                "type": "assets",
                "data": summary,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
            await asyncio.sleep(interval)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected from assets stream")
    except Exception as e:
        logger.error(f"Error in assets stream: {e}")
    finally:
        client_task.cancel()
        try:
            await client_task
        except asyncio.CancelledError:
            pass
