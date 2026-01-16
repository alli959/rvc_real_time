"""
System Metrics API Module

Provides real-time system metrics including:
- CPU usage and load average
- Memory and swap usage
- Disk usage
- Network throughput
- GPU metrics (if NVIDIA available)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin/metrics", tags=["admin-metrics"])


# =============================================================================
# Metrics Data Classes
# =============================================================================

@dataclass
class CPUMetrics:
    """CPU usage metrics"""
    usage_percent: float
    load_avg: tuple[float, float, float]
    cores: List[Dict[str, float]]
    
    def to_dict(self) -> dict:
        return {
            "usage_percent": round(self.usage_percent, 1),
            "load_avg": [round(l, 2) for l in self.load_avg],
            "cores": self.cores,
        }


@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    used_bytes: int
    total_bytes: int
    available_bytes: int
    swap_used_bytes: int
    swap_total_bytes: int
    
    @property
    def used_percent(self) -> float:
        return (self.used_bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0
    
    @property
    def swap_used_percent(self) -> float:
        return (self.swap_used_bytes / self.swap_total_bytes * 100) if self.swap_total_bytes > 0 else 0
    
    def to_dict(self) -> dict:
        return {
            "used_bytes": self.used_bytes,
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "used_percent": round(self.used_percent, 1),
            "swap_used_bytes": self.swap_used_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_used_percent": round(self.swap_used_percent, 1),
        }


@dataclass
class DiskMetrics:
    """Disk usage metrics"""
    mount: str
    used_bytes: int
    total_bytes: int
    
    @property
    def used_percent(self) -> float:
        return (self.used_bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0
    
    def to_dict(self) -> dict:
        return {
            "mount": self.mount,
            "used_bytes": self.used_bytes,
            "total_bytes": self.total_bytes,
            "used_percent": round(self.used_percent, 1),
        }


@dataclass
class NetworkMetrics:
    """Network throughput metrics"""
    rx_bytes_per_sec: float
    tx_bytes_per_sec: float
    
    def to_dict(self) -> dict:
        return {
            "rx_bytes_per_sec": round(self.rx_bytes_per_sec, 0),
            "tx_bytes_per_sec": round(self.tx_bytes_per_sec, 0),
            "rx_mbps": round(self.rx_bytes_per_sec * 8 / 1_000_000, 2),
            "tx_mbps": round(self.tx_bytes_per_sec * 8 / 1_000_000, 2),
        }


@dataclass
class GPUMetrics:
    """NVIDIA GPU metrics"""
    name: str
    index: int
    utilization_percent: float
    vram_used_mb: float
    vram_total_mb: float
    temperature_c: float
    power_draw_w: Optional[float] = None
    
    @property
    def vram_used_percent(self) -> float:
        return (self.vram_used_mb / self.vram_total_mb * 100) if self.vram_total_mb > 0 else 0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "utilization_percent": round(self.utilization_percent, 1),
            "vram_used_mb": round(self.vram_used_mb, 0),
            "vram_total_mb": round(self.vram_total_mb, 0),
            "vram_used_percent": round(self.vram_used_percent, 1),
            "temperature_c": round(self.temperature_c, 0),
            "power_draw_w": round(self.power_draw_w, 0) if self.power_draw_w else None,
        }


@dataclass
class SystemMetrics:
    """Complete system metrics"""
    cpu: CPUMetrics
    memory: MemoryMetrics
    disks: List[DiskMetrics]
    network: NetworkMetrics
    gpus: List[GPUMetrics]
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "disks": [d.to_dict() for d in self.disks],
            "network": self.network.to_dict(),
            "gpus": [g.to_dict() for g in self.gpus],
            "gpu_available": len(self.gpus) > 0,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Metrics Collection
# =============================================================================

class MetricsCollector:
    """Collects system metrics from various sources"""
    
    def __init__(self):
        self._last_network_stats: Optional[Dict] = None
        self._last_network_time: float = 0
        self._last_cpu_stats: Optional[Dict] = None
        self._last_cpu_time: float = 0
    
    async def collect(self) -> SystemMetrics:
        """Collect all system metrics"""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Collect all metrics concurrently
        cpu_task = asyncio.create_task(self._collect_cpu())
        memory_task = asyncio.create_task(self._collect_memory())
        disk_task = asyncio.create_task(self._collect_disks())
        network_task = asyncio.create_task(self._collect_network())
        gpu_task = asyncio.create_task(self._collect_gpus())
        
        cpu = await cpu_task
        memory = await memory_task
        disks = await disk_task
        network = await network_task
        gpus = await gpu_task
        
        return SystemMetrics(
            cpu=cpu,
            memory=memory,
            disks=disks,
            network=network,
            gpus=gpus,
            timestamp=timestamp,
        )
    
    async def _collect_cpu(self) -> CPUMetrics:
        """Collect CPU metrics"""
        try:
            # Read /proc/stat for CPU usage
            with open('/proc/stat', 'r') as f:
                lines = f.readlines()
            
            # Parse first line for overall CPU
            cpu_line = lines[0].strip().split()
            current_stats = {
                'user': int(cpu_line[1]),
                'nice': int(cpu_line[2]),
                'system': int(cpu_line[3]),
                'idle': int(cpu_line[4]),
                'iowait': int(cpu_line[5]) if len(cpu_line) > 5 else 0,
            }
            
            current_time = time.time()
            
            # Calculate CPU usage
            usage_percent = 0.0
            if self._last_cpu_stats and self._last_cpu_time:
                total_diff = sum(current_stats.values()) - sum(self._last_cpu_stats.values())
                idle_diff = current_stats['idle'] - self._last_cpu_stats['idle']
                
                if total_diff > 0:
                    usage_percent = ((total_diff - idle_diff) / total_diff) * 100
            
            self._last_cpu_stats = current_stats
            self._last_cpu_time = current_time
            
            # Get load average
            with open('/proc/loadavg', 'r') as f:
                load_parts = f.read().strip().split()
                load_avg = (float(load_parts[0]), float(load_parts[1]), float(load_parts[2]))
            
            # Per-core metrics (simplified)
            cores = []
            for i, line in enumerate(lines[1:]):
                if not line.startswith('cpu'):
                    break
                cores.append({"core": i, "usage_percent": 0})  # Simplified
            
            return CPUMetrics(
                usage_percent=usage_percent,
                load_avg=load_avg,
                cores=cores,
            )
            
        except Exception as e:
            logger.warning(f"Error collecting CPU metrics: {e}")
            return CPUMetrics(
                usage_percent=0,
                load_avg=(0, 0, 0),
                cores=[],
            )
    
    async def _collect_memory(self) -> MemoryMetrics:
        """Collect memory metrics"""
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
            
            swap_total = meminfo.get('SwapTotal', 0)
            swap_free = meminfo.get('SwapFree', 0)
            swap_used = swap_total - swap_free
            
            return MemoryMetrics(
                used_bytes=used,
                total_bytes=total,
                available_bytes=available,
                swap_used_bytes=swap_used,
                swap_total_bytes=swap_total,
            )
            
        except Exception as e:
            logger.warning(f"Error collecting memory metrics: {e}")
            return MemoryMetrics(
                used_bytes=0,
                total_bytes=0,
                available_bytes=0,
                swap_used_bytes=0,
                swap_total_bytes=0,
            )
    
    async def _collect_disks(self) -> List[DiskMetrics]:
        """Collect disk metrics"""
        disks = []
        
        # Mounts to check
        mounts_to_check = ['/', '/home', '/var', '/app']
        
        try:
            for mount in mounts_to_check:
                if os.path.exists(mount):
                    stat = os.statvfs(mount)
                    total = stat.f_blocks * stat.f_frsize
                    free = stat.f_bfree * stat.f_frsize
                    used = total - free
                    
                    # Skip if total is 0 or mount is duplicate
                    if total > 0:
                        # Check for duplicate (same total/used)
                        is_dup = any(
                            d.total_bytes == total and d.used_bytes == used 
                            for d in disks
                        )
                        if not is_dup:
                            disks.append(DiskMetrics(
                                mount=mount,
                                used_bytes=used,
                                total_bytes=total,
                            ))
                            
        except Exception as e:
            logger.warning(f"Error collecting disk metrics: {e}")
        
        return disks
    
    async def _collect_network(self) -> NetworkMetrics:
        """Collect network metrics"""
        try:
            with open('/proc/net/dev', 'r') as f:
                lines = f.readlines()
            
            current_time = time.time()
            rx_total = 0
            tx_total = 0
            
            for line in lines[2:]:  # Skip header lines
                parts = line.split()
                if len(parts) >= 10:
                    interface = parts[0].rstrip(':')
                    # Skip loopback
                    if interface == 'lo':
                        continue
                    rx_total += int(parts[1])
                    tx_total += int(parts[9])
            
            current_stats = {'rx': rx_total, 'tx': tx_total}
            
            # Calculate rates
            rx_per_sec = 0.0
            tx_per_sec = 0.0
            
            if self._last_network_stats and self._last_network_time:
                time_diff = current_time - self._last_network_time
                if time_diff > 0:
                    rx_per_sec = (rx_total - self._last_network_stats['rx']) / time_diff
                    tx_per_sec = (tx_total - self._last_network_stats['tx']) / time_diff
            
            self._last_network_stats = current_stats
            self._last_network_time = current_time
            
            return NetworkMetrics(
                rx_bytes_per_sec=max(0, rx_per_sec),
                tx_bytes_per_sec=max(0, tx_per_sec),
            )
            
        except Exception as e:
            logger.warning(f"Error collecting network metrics: {e}")
            return NetworkMetrics(rx_bytes_per_sec=0, tx_bytes_per_sec=0)
    
    async def _collect_gpus(self) -> List[GPUMetrics]:
        """Collect GPU metrics via nvidia-smi"""
        gpus = []
        
        try:
            result = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=5)
            
            if result.returncode == 0:
                for line in stdout.decode('utf-8').strip().split('\n'):
                    if not line.strip():
                        continue
                    
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        try:
                            gpus.append(GPUMetrics(
                                name=parts[0],
                                index=int(parts[1]),
                                utilization_percent=float(parts[2]) if parts[2] != '[N/A]' else 0,
                                vram_used_mb=float(parts[3]),
                                vram_total_mb=float(parts[4]),
                                temperature_c=float(parts[5]),
                                power_draw_w=float(parts[6]) if len(parts) > 6 and parts[6] != '[N/A]' else None,
                            ))
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Error parsing GPU line: {e}")
                            
        except FileNotFoundError:
            logger.debug("nvidia-smi not found, GPU metrics unavailable")
        except asyncio.TimeoutError:
            logger.warning("nvidia-smi timeout")
        except Exception as e:
            logger.warning(f"Error collecting GPU metrics: {e}")
        
        return gpus


# Global collector instance
_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get or create the global metrics collector"""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


# =============================================================================
# REST Endpoints
# =============================================================================

@router.get("")
async def get_metrics():
    """Get current system metrics"""
    collector = get_collector()
    metrics = await collector.collect()
    return metrics.to_dict()


@router.get("/cpu")
async def get_cpu_metrics():
    """Get CPU metrics only"""
    collector = get_collector()
    metrics = await collector._collect_cpu()
    return metrics.to_dict()


@router.get("/memory")
async def get_memory_metrics():
    """Get memory metrics only"""
    collector = get_collector()
    metrics = await collector._collect_memory()
    return metrics.to_dict()


@router.get("/disks")
async def get_disk_metrics():
    """Get disk metrics only"""
    collector = get_collector()
    disks = await collector._collect_disks()
    return {"disks": [d.to_dict() for d in disks]}


@router.get("/network")
async def get_network_metrics():
    """Get network metrics only"""
    collector = get_collector()
    metrics = await collector._collect_network()
    return metrics.to_dict()


@router.get("/gpu")
async def get_gpu_metrics():
    """Get GPU metrics only"""
    collector = get_collector()
    gpus = await collector._collect_gpus()
    return {
        "gpus": [g.to_dict() for g in gpus],
        "available": len(gpus) > 0,
    }


# =============================================================================
# WebSocket Endpoint for Real-time Metrics
# =============================================================================

@router.websocket("/stream")
async def stream_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics streaming.
    
    Updates at ~1 second intervals.
    
    Clients can send:
    - {"interval": 1000} - Set update interval in ms (min 500, max 10000)
    """
    await websocket.accept()
    
    collector = get_collector()
    interval = 1.0  # Default 1 second
    
    async def handle_client_messages():
        nonlocal interval
        try:
            while True:
                data = await websocket.receive_json()
                if "interval" in data:
                    # Clamp interval between 0.5 and 10 seconds
                    interval = max(0.5, min(10.0, data["interval"] / 1000))
                    await websocket.send_json({
                        "type": "config",
                        "interval": int(interval * 1000),
                    })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"Client message error: {e}")
    
    client_task = asyncio.create_task(handle_client_messages())
    
    try:
        while True:
            metrics = await collector.collect()
            await websocket.send_json({
                "type": "metrics",
                "data": metrics.to_dict(),
            })
            await asyncio.sleep(interval)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected from metrics stream")
    except Exception as e:
        logger.error(f"Error in metrics stream: {e}")
    finally:
        client_task.cancel()
        try:
            await client_task
        except asyncio.CancelledError:
            pass
