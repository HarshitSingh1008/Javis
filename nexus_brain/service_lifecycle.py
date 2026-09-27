"""
NEXUS AI v4.0 — Service lifecycle management with async context managers.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides base classes and utilities for managing service lifecycles:
- Async context managers for proper resource cleanup
- Startup/shutdown hooks
- Health checks
- Graceful degradation
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set
from functools import lru_cache

logger = logging.getLogger("nexus.lifecycle")


class ServiceState(Enum):
    """Service lifecycle states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass
class ServiceHealth:
    """Health status of a service."""
    name: str
    state: ServiceState = ServiceState.STOPPED
    healthy: bool = False
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: float = 0.0


class ServiceBase(ABC):
    """
    Base class for all NEXUS services with lifecycle management.
    
    Provides:
    - Async start/stop with proper error handling
    - Health checking
    - Dependency management
    - Context manager support
    """
    
    def __init__(self, name: str):
        self.name = name
        self._state = ServiceState.STOPPED
        self._start_time: float = 0.0
        self._health = ServiceHealth(name=name)
        self._shutdown_event = asyncio.Event()
        self._dependencies: List["ServiceBase"] = []
        self._dependents: List["ServiceBase"] = []
    
    @property
    def state(self) -> ServiceState:
        return self._state
    
    @property
    def is_running(self) -> bool:
        return self._state == ServiceState.RUNNING
    
    @property
    def health(self) -> ServiceHealth:
        return self._health
    
    def add_dependency(self, service: "ServiceBase") -> None:
        """Add a service dependency (will be started before this service)."""
        if service not in self._dependencies:
            self._dependencies.append(service)
            service._dependents.append(self)
    
    async def start(self) -> bool:
        """
        Start the service and all its dependencies.
        
        Returns:
            True if started successfully, False otherwise.
        """
        if self._state in (ServiceState.RUNNING, ServiceState.STARTING):
            return True
        
        self._state = ServiceState.STARTING
        self._health.state = ServiceState.STARTING
        logger.info("Starting service: %s", self.name)
        
        try:
            # Start dependencies first
            for dep in self._dependencies:
                if not await dep.start():
                    raise RuntimeError(f"Dependency {dep.name} failed to start")
            
            # Start this service
            await self._start()
            
            self._state = ServiceState.RUNNING
            self._start_time = asyncio.get_event_loop().time()
            self._health.state = ServiceState.RUNNING
            self._health.healthy = True
            self._health.message = "Service running"
            logger.info("Service started: %s", self.name)
            return True
            
        except Exception as e:
            self._state = ServiceState.ERROR
            self._health.state = ServiceState.ERROR
            self._health.healthy = False
            self._health.message = f"Start failed: {e}"
            logger.error("Service %s failed to start: %s", self.name, e)
            return False
    
    async def stop(self) -> None:
        """Stop the service and all its dependents."""
        if self._state in (ServiceState.STOPPED, ServiceState.STOPPING):
            return
        
        self._state = ServiceState.STOPPING
        self._health.state = ServiceState.STOPPING
        logger.info("Stopping service: %s", self.name)
        
        try:
            # Stop dependents first (reverse order)
            for dep in reversed(self._dependents):
                await dep.stop()
            
            # Stop this service
            await self._stop()
            
            self._state = ServiceState.STOPPED
            self._health.state = ServiceState.STOPPED
            self._health.healthy = False
            self._health.message = "Service stopped"
            logger.info("Service stopped: %s", self.name)
            
        except Exception as e:
            self._state = ServiceState.ERROR
            self._health.state = ServiceState.ERROR
            self._health.healthy = False
            self._health.message = f"Stop failed: {e}"
            logger.error("Service %s failed to stop: %s", self.name, e)
    
    @abstractmethod
    async def _start(self) -> None:
        """Service-specific startup logic."""
        pass
    
    @abstractmethod
    async def _stop(self) -> None:
        """Service-specific shutdown logic."""
        pass
    
    async def health_check(self) -> ServiceHealth:
        """Perform health check and update status."""
        try:
            healthy, message, details = await self._health_check()
            self._health.healthy = healthy
            self._health.message = message
            self._health.details = details
            self._health.last_check = asyncio.get_event_loop().time()
            
            if not healthy and self._state == ServiceState.RUNNING:
                self._state = ServiceState.DEGRADED
                self._health.state = ServiceState.DEGRADED
            elif healthy and self._state == ServiceState.DEGRADED:
                self._state = ServiceState.RUNNING
                self._health.state = ServiceState.RUNNING
                
        except Exception as e:
            self._health.healthy = False
            self._health.message = f"Health check error: {e}"
            self._health.state = ServiceState.ERROR
            self._state = ServiceState.ERROR
        
        return self._health
    
    @abstractmethod
    async def _health_check(self) -> tuple[bool, str, Dict[str, Any]]:
        """
        Service-specific health check.
        
        Returns:
            Tuple of (healthy: bool, message: str, details: dict)
        """
        return True, "OK", {}
    
    @asynccontextmanager
    async def lifespan(self) -> AsyncGenerator["ServiceBase", None]:
        """
        Async context manager for service lifecycle.
        
        Usage:
            async with service.lifespan():
                # service is running
                ...
            # service is stopped
        """
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    async def __aenter__(self) -> "ServiceBase":
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class ServiceManager:
    """
    Manages a collection of services with proper startup/shutdown ordering.
    
    Handles:
    - Dependency resolution
    - Parallel startup where possible
    - Graceful shutdown
    - Health monitoring
    """
    
    def __init__(self):
        self._services: Dict[str, ServiceBase] = {}
        self._startup_order: List[str] = []
        self._shutdown_event = asyncio.Event()
    
    def register(self, service: ServiceBase) -> None:
        """Register a service with the manager."""
        self._services[service.name] = service
        self._recalculate_order()
    
    def _recalculate_order(self) -> None:
        """Calculate topological startup order based on dependencies."""
        # Kahn's algorithm for topological sort
        in_degree = {name: 0 for name in self._services}
        adj = {name: [] for name in self._services}
        
        for name, service in self._services.items():
            for dep in service._dependencies:
                if dep.name in self._services:
                    adj[dep.name].append(name)
                    in_degree[name] += 1
        
        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            name = queue.pop(0)
            order.append(name)
            for neighbor in adj[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(order) != len(self._services):
            # Cycle detected, fall back to registration order
            logger.warning("Service dependency cycle detected, using registration order")
            order = list(self._services.keys())
        
        self._startup_order = order
    
    async def start_all(self) -> bool:
        """Start all services in dependency order."""
        logger.info("Starting all services...")
        
        for name in self._startup_order:
            service = self._services[name]
            if not await service.start():
                logger.error("Failed to start %s, stopping all services", name)
                await self.stop_all()
                return False
        
        logger.info("All services started successfully")
        return True
    
    async def stop_all(self) -> None:
        """Stop all services in reverse order."""
        logger.info("Stopping all services...")
        
        for name in reversed(self._startup_order):
            service = self._services[name]
            try:
                await service.stop()
            except Exception as e:
                logger.error("Error stopping %s: %s", name, e)
        
        logger.info("All services stopped")
    
    async def health_check_all(self) -> Dict[str, ServiceHealth]:
        """Check health of all services."""
        results = {}
        for name, service in self._services.items():
            results[name] = await service.health_check()
        return results
    
    def get_service(self, name: str) -> Optional[ServiceBase]:
        """Get a service by name."""
        return self._services.get(name)
    
    @asynccontextmanager
    async def lifespan(self) -> AsyncGenerator["ServiceManager", None]:
        """Context manager for all services."""
        try:
            await self.start_all()
            yield self
        finally:
            await self.stop_all()


# Global service manager
_service_manager: Optional[ServiceManager] = None


def get_service_manager() -> ServiceManager:
    """Get the global service manager."""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager


@asynccontextmanager
async def managed_services(*services: ServiceBase) -> AsyncGenerator[None, None]:
    """
    Context manager for ad-hoc service management.
    
    Usage:
        async with managed_services(service1, service2):
            # services running
            ...
    """
    manager = get_service_manager()
    for svc in services:
        manager.register(svc)
    async with manager.lifespan():
        yield