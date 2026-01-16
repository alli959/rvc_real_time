"""
Logs API Module

Provides REST and WebSocket endpoints for log streaming and management.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .log_discovery import get_log_discovery, LogSource

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin/logs", tags=["admin-logs"])


class ServiceResponse(BaseModel):
    """Response containing service info"""
    name: str
    container_name: str
    status: str
    log_sources: list


class LogLinesResponse(BaseModel):
    """Response containing log lines"""
    service: str
    source_id: str
    lines: list[str]
    line_count: int


# =============================================================================
# REST Endpoints
# =============================================================================

@router.get("/services")
async def list_services():
    """
    List all services with their log sources.
    
    Returns services discovered from docker-compose.prod.yml with
    their running status and available log sources.
    """
    discovery = get_log_discovery()
    services = discovery.discover_services()
    
    return {
        "services": [s.to_dict() for s in services],
        "total": len(services),
    }


@router.get("/services/{service_name}")
async def get_service(service_name: str):
    """Get info for a specific service"""
    discovery = get_log_discovery()
    service = discovery.get_service(service_name)
    
    if not service:
        raise HTTPException(status_code=404, detail=f"Service not found: {service_name}")
    
    return service.to_dict()


@router.get("/services/{service_name}/sources")
async def get_log_sources(service_name: str):
    """Get available log sources for a service"""
    discovery = get_log_discovery()
    service = discovery.get_service(service_name)
    
    if not service:
        raise HTTPException(status_code=404, detail=f"Service not found: {service_name}")
    
    return {
        "service": service_name,
        "sources": [s.to_dict() for s in service.log_sources],
    }


@router.get("/tail/{service_name}/{source_id}")
async def tail_logs(
    service_name: str,
    source_id: str,
    lines: int = Query(default=200, ge=1, le=10000),
):
    """
    Get the last N lines from a log source.
    
    Args:
        service_name: Name of the service
        source_id: ID of the log source
        lines: Number of lines to return (default: 200, max: 10000)
    """
    discovery = get_log_discovery()
    source = discovery.get_log_source(service_name, source_id)
    
    if not source:
        raise HTTPException(status_code=404, detail=f"Log source not found: {source_id}")
    
    if source.type == "none":
        return LogLinesResponse(
            service=service_name,
            source_id=source_id,
            lines=["No log files discovered for this service"],
            line_count=1,
        )
    
    try:
        log_lines = await _read_log_lines(source, lines)
        return LogLinesResponse(
            service=service_name,
            source_id=source_id,
            lines=log_lines,
            line_count=len(log_lines),
        )
    except Exception as e:
        logger.error(f"Error reading logs from {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{service_name}/{source_id}")
async def download_logs(
    service_name: str,
    source_id: str,
    lines: int = Query(default=1000, ge=1, le=100000),
):
    """
    Download log lines (for larger exports).
    
    Returns plain text content suitable for download.
    """
    discovery = get_log_discovery()
    source = discovery.get_log_source(service_name, source_id)
    
    if not source:
        raise HTTPException(status_code=404, detail=f"Log source not found: {source_id}")
    
    try:
        log_lines = await _read_log_lines(source, lines)
        content = "\n".join(log_lines)
        
        return {
            "filename": f"{service_name}_{source_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            "content": content,
            "line_count": len(log_lines),
        }
    except Exception as e:
        logger.error(f"Error downloading logs from {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def refresh_discovery():
    """Refresh service and log source discovery"""
    discovery = get_log_discovery()
    discovery.refresh()
    
    services = discovery.discover_services()
    return {
        "status": "refreshed",
        "services_count": len(services),
    }


# =============================================================================
# WebSocket Endpoints for Live Streaming
# =============================================================================

@router.websocket("/stream/{service_name}/{source_id}")
async def stream_logs(
    websocket: WebSocket,
    service_name: str,
    source_id: str,
):
    """
    WebSocket endpoint for live log streaming (tail -f behavior).
    
    Clients can send:
    - {"action": "pause"} - Pause streaming
    - {"action": "resume"} - Resume streaming
    - {"action": "filter", "pattern": "..."} - Set client-side filter hint
    """
    await websocket.accept()
    
    discovery = get_log_discovery()
    source = discovery.get_log_source(service_name, source_id)
    
    if not source:
        await websocket.send_json({"error": f"Log source not found: {source_id}"})
        await websocket.close()
        return
    
    if source.type == "none":
        await websocket.send_json({
            "type": "info",
            "message": "No log files discovered for this service",
        })
        # Keep connection open but idle
        try:
            while True:
                await asyncio.sleep(60)
        except WebSocketDisconnect:
            return
    
    # Track streaming state
    paused = False
    
    async def handle_client_messages():
        nonlocal paused
        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")
                
                if action == "pause":
                    paused = True
                    await websocket.send_json({"type": "status", "paused": True})
                elif action == "resume":
                    paused = False
                    await websocket.send_json({"type": "status", "paused": False})
                    
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"Error handling client message: {e}")
    
    # Start client message handler
    client_task = asyncio.create_task(handle_client_messages())
    
    try:
        # Send initial buffer
        initial_lines = await _read_log_lines(source, 200)
        await websocket.send_json({
            "type": "initial",
            "lines": initial_lines,
        })
        
        # Stream new lines
        async for line in _stream_log_lines(source):
            if not paused:
                await websocket.send_json({
                    "type": "line",
                    "line": line,
                    "timestamp": datetime.now().isoformat(),
                })
            await asyncio.sleep(0.01)  # Prevent flooding
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from log stream: {service_name}/{source_id}")
    except Exception as e:
        logger.error(f"Error in log stream: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        client_task.cancel()
        try:
            await client_task
        except asyncio.CancelledError:
            pass


# =============================================================================
# Log Reading Helpers
# =============================================================================

async def _read_log_lines(source: LogSource, lines: int) -> list[str]:
    """Read the last N lines from a log source"""
    if source.type == "stdout":
        return await _read_container_logs(source.container, lines)
    elif source.type == "file":
        return await _read_file_logs(source.container, source.path, lines)
    else:
        return []


async def _read_container_logs(container: str, lines: int) -> list[str]:
    """Read container stdout/stderr logs via docker logs"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "--tail", str(lines), container,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        return stdout.decode('utf-8', errors='replace').strip().split('\n')
        
    except asyncio.TimeoutError:
        return ["Error: Timeout reading container logs"]
    except Exception as e:
        return [f"Error: {str(e)}"]


async def _read_file_logs(container: str, path: str, lines: int) -> list[str]:
    """Read log file from inside container via docker exec"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container, "tail", "-n", str(lines), path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        
        if proc.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='replace').strip()
            if "No such file" in error_msg:
                return [f"Log file not found: {path}"]
            return [f"Error reading log file: {error_msg}"]
        
        return stdout.decode('utf-8', errors='replace').strip().split('\n')
        
    except asyncio.TimeoutError:
        return ["Error: Timeout reading log file"]
    except Exception as e:
        return [f"Error: {str(e)}"]


async def _stream_log_lines(source: LogSource) -> AsyncGenerator[str, None]:
    """Stream log lines in real-time (tail -f behavior)"""
    if source.type == "stdout":
        async for line in _stream_container_logs(source.container):
            yield line
    elif source.type == "file":
        async for line in _stream_file_logs(source.container, source.path):
            yield line


async def _stream_container_logs(container: str) -> AsyncGenerator[str, None]:
    """Stream container logs via docker logs -f"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "-f", "--tail", "0", container,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line.decode('utf-8', errors='replace').rstrip()
            
    except Exception as e:
        logger.error(f"Error streaming container logs: {e}")
        yield f"Stream error: {str(e)}"


async def _stream_file_logs(container: str, path: str) -> AsyncGenerator[str, None]:
    """Stream log file via docker exec tail -F"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container, "tail", "-F", "-n", "0", path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        while True:
            line = await proc.stdout.readline()
            if not line:
                # Check if process is still running
                if proc.returncode is not None:
                    break
                await asyncio.sleep(0.1)
                continue
            yield line.decode('utf-8', errors='replace').rstrip()
            
    except Exception as e:
        logger.error(f"Error streaming file logs: {e}")
        yield f"Stream error: {str(e)}"
