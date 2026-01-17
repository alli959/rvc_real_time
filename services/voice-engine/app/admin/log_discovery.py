"""
Log Discovery Module

Discovers log sources from running containers and file systems.
Implements the explicit log source discovery rules.
"""

from __future__ import annotations

import os
import re
import subprocess
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LogSource:
    """Represents a discovered log source"""
    id: str
    name: str
    type: str  # 'stdout', 'file'
    path: Optional[str] = None
    container: Optional[str] = None
    priority: int = 50  # 0=highest, 100=lowest
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "container": self.container,
            "priority": self.priority,
            "description": self.description,
        }


@dataclass
class ServiceInfo:
    """Information about a service from docker-compose"""
    name: str
    container_name: str
    image: str
    status: str = "unknown"  # running, stopped, etc
    log_sources: List[LogSource] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "container_name": self.container_name,
            "image": self.image,
            "status": self.status,
            "log_sources": [s.to_dict() for s in self.log_sources],
        }


# Known log directories to check (in priority order)
LOG_DIRECTORIES = [
    "/app/storage/logs",
    "/app/logs", 
    "/var/log",
    "/logs",
    "/data/logs",
]

# File patterns to match (in priority order)
LOG_PATTERNS = [
    (r".*error.*\.log$", "error", 10),
    (r".*\.error$", "error", 10),
    (r"laravel.*\.log$", "application", 20),
    (r".*application.*\.log$", "application", 20),
    (r".*app.*\.log$", "application", 25),
    (r".*access.*\.log$", "access", 40),
    (r".*debug.*\.log$", "debug", 60),
    (r".*warning.*\.log$", "warning", 30),
    (r".*\.log$", "general", 50),
]

# Service-specific log configurations
SERVICE_LOG_CONFIGS = {
    "nginx": {
        "known_paths": [
            "/var/log/nginx/access.log",
            "/var/log/nginx/error.log",
        ],
        "patterns": [r"nginx"],
    },
    "api": {
        "known_paths": [
            "/app/storage/logs/laravel.log",
        ],
        "patterns": [r"laravel", r"php-fpm"],
    },
    "api-worker": {
        "known_paths": [
            "/app/storage/logs/laravel.log",
        ],
        "patterns": [r"laravel", r"queue"],
    },
    "voice-engine": {
        "known_paths": [
            "/app/logs/voice_engine.log",
            "/app/logs/training.log",
            "/app/logs/trainer.log",
            "/app/logs/inference.log",
        ],
        "patterns": [r"uvicorn", r"fastapi", r"training", r"trainer"],
    },
    "trainer": {
        "known_paths": [
            "/app/logs/training.log",
            "/app/logs/trainer.log",
        ],
        "patterns": [r"training", r"trainer", r"rvc"],
    },
    "db": {
        "known_paths": [
            "/var/log/mysql/error.log",
            "/var/log/mysql/slow.log",
        ],
        "patterns": [r"mysql", r"mariadb"],
    },
    "redis": {
        "known_paths": [],
        "patterns": [r"redis"],
    },
    "minio": {
        "known_paths": [],
        "patterns": [r"minio"],
    },
}


class LogDiscovery:
    """
    Discovers log sources from docker-compose and running containers.
    
    IMPORTANT: Since we're running inside the voice-engine container,
    we can only read LOCAL files. We cannot access other containers' files.
    """
    
    def __init__(self, compose_file: Optional[str] = None):
        """
        Initialize log discovery.
        
        Args:
            compose_file: Path to docker-compose file. If None, auto-detects.
        """
        self.compose_file = compose_file or self._find_compose_file()
        self._services: Dict[str, ServiceInfo] = {}
        self._discovered = False
    
    def _find_compose_file(self) -> str:
        """Find the production docker-compose file"""
        # Try common locations
        candidates = [
            "/home/alexanderg/rvc_real_time/infra/compose/docker-compose.prod.yml",
            "../../../infra/compose/docker-compose.prod.yml",
            "docker-compose.prod.yml",
            "docker-compose.yml",
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        return "docker-compose.prod.yml"
    
    def discover_services(self) -> List[ServiceInfo]:
        """
        Discover services with their log sources.
        
        Since we're inside voice-engine, we can only reliably read:
        - voice-engine's own logs in /app/logs/
        - Any other local files
        
        For other services, we indicate that their logs aren't accessible.
        """
        if self._discovered and self._services:
            return list(self._services.values())
        
        # Only voice-engine has accessible logs from within this container
        self._services = {
            "voice-engine": ServiceInfo(
                name="voice-engine",
                container_name="morphvox-voice-engine",
                image="services/voice-engine",
                status="running",
                log_sources=self._get_voice_engine_logs(),
            ),
        }
        
        self._discovered = True
        return list(self._services.values())
    
    def _get_voice_engine_logs(self) -> List[LogSource]:
        """Get available log sources for voice-engine (local files only)"""
        sources = []
        
        # Check known log paths
        log_paths = [
            ("/app/logs/voice_engine.log", "Voice Engine Log", "Main voice engine application log"),
            ("/app/logs/trainer.log", "Trainer Log", "Model training logs"),
            ("/app/logs/training.log", "Training Log", "Training session logs"),
            ("/app/logs/inference.log", "Inference Log", "Voice conversion inference logs"),
            ("/app/logs/uvicorn.log", "Uvicorn Log", "HTTP server logs"),
        ]
        
        for path, name, desc in log_paths:
            if os.path.exists(path):
                sources.append(LogSource(
                    id=f"voice-engine_{Path(path).stem}",
                    name=name,
                    type="file",
                    path=path,
                    container="morphvox-voice-engine",
                    priority=20,
                    description=desc,
                ))
        
        # Also scan /app/logs for any other .log files
        logs_dir = "/app/logs"
        if os.path.isdir(logs_dir):
            for filename in os.listdir(logs_dir):
                if filename.endswith('.log'):
                    full_path = os.path.join(logs_dir, filename)
                    # Skip if already added
                    if any(s.path == full_path for s in sources):
                        continue
                    sources.append(LogSource(
                        id=f"voice-engine_{Path(filename).stem}",
                        name=filename,
                        type="file",
                        path=full_path,
                        container="morphvox-voice-engine",
                        priority=50,
                        description=f"Discovered log: {full_path}",
                    ))
        
        if not sources:
            sources.append(LogSource(
                id="voice-engine_no_logs",
                name="No logs found",
                type="none",
                container="morphvox-voice-engine",
                priority=100,
                description="No log files found in /app/logs",
            ))
        
        return sources
    
    def _parse_compose_file(self) -> Dict[str, ServiceInfo]:
        """Parse docker-compose file to extract services"""
        services = {}
        
        try:
            import yaml
            
            with open(self.compose_file, 'r') as f:
                compose = yaml.safe_load(f)
            
            for name, config in compose.get('services', {}).items():
                container_name = config.get('container_name', f"compose_{name}_1")
                image = config.get('image', config.get('build', {}).get('context', 'custom'))
                
                services[name] = ServiceInfo(
                    name=name,
                    container_name=container_name,
                    image=str(image),
                )
                
        except Exception as e:
            logger.warning(f"Could not parse compose file: {e}")
            # Fallback to hardcoded list based on known structure
            services = self._get_fallback_services()
        
        return services
    
    def _get_fallback_services(self) -> Dict[str, ServiceInfo]:
        """Get fallback service list based on known structure"""
        return {
            "nginx": ServiceInfo("nginx", "morphvox-nginx", "nginx:alpine"),
            "api": ServiceInfo("api", "morphvox-api", "apps/api"),
            "api-worker": ServiceInfo("api-worker", "morphvox-api-worker", "apps/api"),
            "web": ServiceInfo("web", "morphvox-web", "apps/web"),
            "voice-engine": ServiceInfo("voice-engine", "morphvox-voice-engine", "services/voice-engine"),
            "db": ServiceInfo("db", "morphvox-db", "mariadb:10.11"),
            "redis": ServiceInfo("redis", "morphvox-redis", "redis:7-alpine"),
            "minio": ServiceInfo("minio", "morphvox-minio", "minio/minio:latest"),
        }
    
    def _get_running_containers(self) -> set:
        """Get set of running container names"""
        # Since we're running inside a container, we can't easily run docker commands
        # Instead, assume common services are running and let the actual log reading
        # determine if they're accessible
        return {
            "morphvox-nginx",
            "morphvox-api", 
            "morphvox-api-worker",
            "morphvox-web",
            "morphvox-voice-engine",
            "morphvox-db",
            "morphvox-redis",
            "morphvox-minio",
        }
    
    def _discover_log_sources(self, service: ServiceInfo) -> List[LogSource]:
        """Discover log sources for a service"""
        sources = []
        
        # Always include stdout/stderr
        sources.append(LogSource(
            id=f"{service.name}_stdout",
            name="Container Logs (stdout/stderr)",
            type="stdout",
            container=service.container_name,
            priority=5,
            description="Live container output",
        ))
        
        # Check known paths for this service
        service_config = SERVICE_LOG_CONFIGS.get(service.name, {})
        known_paths = service_config.get("known_paths", [])
        
        for path in known_paths:
            sources.append(LogSource(
                id=f"{service.name}_{Path(path).stem}",
                name=Path(path).name,
                type="file",
                path=path,
                container=service.container_name,
                priority=self._get_path_priority(path),
                description=f"Log file: {path}",
            ))
        
        # Discover additional log files by scanning directories
        discovered = self._scan_container_logs(service.container_name)
        
        for path, priority in discovered:
            # Skip if already in known paths
            if path in known_paths:
                continue
            
            log_id = f"{service.name}_{Path(path).stem}".replace("/", "_").replace(".", "_")
            sources.append(LogSource(
                id=log_id,
                name=Path(path).name,
                type="file",
                path=path,
                container=service.container_name,
                priority=priority,
                description=f"Discovered log: {path}",
            ))
        
        # Sort by priority and limit to 4 file-based sources (plus stdout)
        file_sources = [s for s in sources if s.type == "file"]
        file_sources.sort(key=lambda s: s.priority)
        file_sources = file_sources[:4]
        
        # Combine stdout + top file sources
        stdout_sources = [s for s in sources if s.type == "stdout"]
        sources = stdout_sources + file_sources
        
        # Add placeholder if no file logs found
        if len(file_sources) == 0:
            sources.append(LogSource(
                id=f"{service.name}_no_files",
                name="No file-based logs discovered",
                type="none",
                container=service.container_name,
                priority=100,
                description="No log files found in standard locations",
            ))
        
        return sources
    
    def _scan_container_logs(self, container_name: str) -> List[tuple]:
        """Scan for log files in local directories (since we're inside the container)"""
        discovered = []
        
        for log_dir in LOG_DIRECTORIES:
            try:
                # Check if directory exists locally
                if not os.path.isdir(log_dir):
                    continue
                    
                # Scan directory for log files
                for filename in os.listdir(log_dir):
                    full_path = os.path.join(log_dir, filename)
                    if os.path.isfile(full_path):
                        priority = self._get_path_priority(full_path)
                        if priority < 100:  # Valid log file
                            discovered.append((full_path, priority))
                                
            except Exception as e:
                # Directory doesn't exist or not accessible
                logger.debug(f"Could not scan {log_dir}: {e}")
                continue
        
        return discovered
    
    def _get_path_priority(self, path: str) -> int:
        """Get priority for a log file path based on patterns"""
        filename = Path(path).name.lower()
        
        for pattern, log_type, priority in LOG_PATTERNS:
            if re.match(pattern, filename, re.IGNORECASE):
                return priority
        
        return 100  # Unknown file type
    
    def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Get info for a specific service"""
        if not self._discovered:
            self.discover_services()
        return self._services.get(service_name)
    
    def get_log_source(self, service_name: str, source_id: str) -> Optional[LogSource]:
        """Get a specific log source"""
        service = self.get_service(service_name)
        if not service:
            return None
        
        for source in service.log_sources:
            if source.id == source_id:
                return source
        
        return None
    
    def refresh(self):
        """Refresh service and log source discovery"""
        self._discovered = False
        self._services = {}
        self.discover_services()


# Global instance
_log_discovery: Optional[LogDiscovery] = None


def get_log_discovery() -> LogDiscovery:
    """Get or create the global LogDiscovery instance"""
    global _log_discovery
    if _log_discovery is None:
        _log_discovery = LogDiscovery()
    return _log_discovery
