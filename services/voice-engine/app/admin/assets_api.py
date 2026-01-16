"""
Assets Management API Module

Provides endpoints for managing heavy assets/components:
- Bark TTS models
- RVC models
- UVR5 vocal separation
- Other GPU-heavy components
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
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
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


class ResourceType(str, Enum):
    """Type of resource the asset uses"""
    CPU = "cpu"
    GPU = "gpu"
    RAM = "ram"


class AssetStatus(str, Enum):
    """Asset status"""
    RUNNING = "running"
    STOPPED = "stopped"
    LOADING = "loading"
    UNLOADING = "unloading"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class AssetMetrics:
    """Resource usage metrics for an asset"""
    ram_usage_mb: Optional[float] = None
    vram_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "ram_usage_mb": round(self.ram_usage_mb, 1) if self.ram_usage_mb else None,
            "vram_usage_mb": round(self.vram_usage_mb, 1) if self.vram_usage_mb else None,
            "cpu_percent": round(self.cpu_percent, 1) if self.cpu_percent else None,
        }


@dataclass
class Asset:
    """Represents a heavy asset that can be started/stopped"""
    id: str
    name: str
    description: str
    type: AssetType
    resource_type: ResourceType
    status: AssetStatus = AssetStatus.STOPPED
    container: Optional[str] = None
    process_name: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metrics: AssetMetrics = field(default_factory=AssetMetrics)
    estimated_ram_mb: Optional[float] = None
    estimated_vram_mb: Optional[float] = None
    last_status_check: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "resource_type": self.resource_type.value,
            "status": self.status.value,
            "container": self.container,
            "process_name": self.process_name,
            "dependencies": self.dependencies,
            "metrics": self.metrics.to_dict(),
            "estimated_ram_mb": self.estimated_ram_mb,
            "estimated_vram_mb": self.estimated_vram_mb,
            "last_status_check": self.last_status_check,
        }


# =============================================================================
# Asset Registry
# =============================================================================

# Known heavy assets in the system
ASSET_REGISTRY: Dict[str, Asset] = {
    "bark_tts": Asset(
        id="bark_tts",
        name="Bark TTS",
        description="Suno Bark text-to-speech model for natural speech generation",
        type=AssetType.MODEL,
        resource_type=ResourceType.GPU,
        process_name="bark",
        estimated_ram_mb=2000,
        estimated_vram_mb=4000,
        dependencies=[],
    ),
    "hubert": Asset(
        id="hubert",
        name="HuBERT Model",
        description="HuBERT base model for voice feature extraction",
        type=AssetType.MODEL,
        resource_type=ResourceType.GPU,
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
        process_name="rmvpe",
        estimated_ram_mb=200,
        estimated_vram_mb=500,
        dependencies=[],
    ),
    "uvr5": Asset(
        id="uvr5",
        name="UVR5 Vocal Separator",
        description="Ultimate Vocal Remover for vocal/instrumental separation",
        type=AssetType.MODEL,
        resource_type=ResourceType.GPU,
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
        process_name="train",
        estimated_ram_mb=4000,
        estimated_vram_mb=8000,
        dependencies=["hubert", "rmvpe"],
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
        self._active_jobs: Dict[str, List[str]] = {}  # job_id -> [asset_ids]
    
    async def get_all_assets(self) -> List[Asset]:
        """Get all registered assets with updated status"""
        assets = []
        for asset_id, asset in self._assets.items():
            # Update status
            asset.status = await self._check_asset_status(asset)
            asset.last_status_check = datetime.utcnow().isoformat() + "Z"
            assets.append(asset)
        return assets
    
    async def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get a specific asset"""
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
                self._loaded_models[asset_id] = True
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
                self._loaded_models.pop(asset_id, None)
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
        """Check the actual status of an asset"""
        # Check if we have it marked as loaded
        if asset.id in self._loaded_models:
            return AssetStatus.RUNNING
        
        # Check based on asset type
        if asset.type == AssetType.MODEL:
            return await self._check_model_loaded(asset)
        elif asset.type == AssetType.SERVICE:
            return await self._check_service_running(asset)
        else:
            return AssetStatus.UNKNOWN
    
    async def _check_model_loaded(self, asset: Asset) -> AssetStatus:
        """Check if a model is loaded in memory"""
        # This would require introspection of the actual model state
        # For now, use the tracked state
        return AssetStatus.RUNNING if asset.id in self._loaded_models else AssetStatus.STOPPED
    
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


# =============================================================================
# REST Endpoints
# =============================================================================

@router.get("")
async def list_assets():
    """List all registered assets with their status"""
    manager = get_asset_manager()
    assets = await manager.get_all_assets()
    
    return {
        "assets": [a.to_dict() for a in assets],
        "total": len(assets),
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
