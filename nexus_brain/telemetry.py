"""
NEXUS AI v4.0 — Telemetry and metrics collection for observability.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides:
- Request/response metrics
- LLM call tracking
- Tool execution metrics
- System resource monitoring
- Performance histograms
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator
from functools import lru_cache
from contextlib import asynccontextmanager

logger = logging.getLogger("nexus.telemetry")


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Histogram:
    """Histogram for latency/duration tracking."""
    name: str
    buckets: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0])
    counts: List[int] = field(default_factory=list)
    total_count: int = 0
    total_sum: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * len(self.buckets)
    
    def observe(self, value: float) -> None:
        """Record a value."""
        self.total_count += 1
        self.total_sum += value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        for i, bucket in enumerate(self.buckets):
            if value <= bucket:
                self.counts[i] += 1
                break
        else:
            # Value exceeds all buckets
            if self.counts:
                self.counts[-1] += 1
    
    def percentiles(self) -> Dict[str, float]:
        """Calculate percentile values."""
        if self.total_count == 0:
            return {}
        
        result = {}
        cumulative = 0
        for i, count in enumerate(self.counts):
            cumulative += count
            pct = cumulative / self.total_count
            result[f"p{int(pct * 100)}"] = self.buckets[i]
        
        result["mean"] = self.total_sum / self.total_count
        result["min"] = self.min_val if self.min_val != float('inf') else 0
        result["max"] = self.max_val if self.max_val != float('-inf') else 0
        result["count"] = self.total_count
        result["sum"] = self.total_sum
        return result


class MetricsCollector:
    """
    Central metrics collector for NEXUS AI.
    
    Collects:
    - Counter metrics (requests, errors, etc.)
    - Gauge metrics (current state values)
    - Histogram metrics (latencies, durations)
    """
    
    def __init__(self, max_history: int = 10000):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._history: deque = deque(maxlen=max_history)
        self._lock = threading.RLock()
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        self._collection_interval = 60.0  # seconds
    
    # ── Counters ──────────────────────────────────────────────────────────────
    
    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value
            self._record_history(name, value, tags, "counter")
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, tags)
        with self._lock:
            return self._counters.get(key, 0.0)
    
    # ── Gauges ────────────────────────────────────────────────────────────────
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value
            self._record_history(name, value, tags, "gauge")
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            return self._gauges.get(key)
    
    # ── Histograms ────────────────────────────────────────────────────────────
    
    def get_histogram(self, name: str, tags: Optional[Dict[str, str]] = None, 
                      buckets: Optional[List[float]] = None) -> Histogram:
        """Get or create a histogram."""
        key = self._make_key(name, tags)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(name=name, tags=tags or {})
            return self._histograms[key]
    
    def observe(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a histogram."""
        hist = self.get_histogram(name, tags)
        hist.observe(value)
        self._record_history(name, value, tags, "histogram")
    
    def time_operation(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing an operation."""
        return _TimerContext(self, name, tags)
    
    # ── System Metrics ────────────────────────────────────────────────────────
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.set_gauge("system.cpu.percent", cpu_percent)
            self.set_gauge("system.memory.percent", memory.percent)
            self.set_gauge("system.memory.available_mb", memory.available / (1024 * 1024))
            self.set_gauge("system.disk.percent", disk.percent)
            self.set_gauge("system.disk.free_gb", disk.free / (1024 * 1024 * 1024))
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            }
        except ImportError:
            return {}
        except Exception as e:
            logger.debug("System metrics collection failed: %s", e)
            return {}
    
    # ── Export ────────────────────────────────────────────────────────────────
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: v.percentiles() for k, v in self._histograms.items()},
                "timestamp": time.time(),
            }
    
    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        with self._lock:
            # Counters
            for key, value in self._counters.items():
                name, tags = self._parse_key(key)
                tag_str = self._format_tags(tags)
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{tag_str} {value}")
            
            # Gauges
            for key, value in self._gauges.items():
                name, tags = self._parse_key(key)
                tag_str = self._format_tags(tags)
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{tag_str} {value}")
            
            # Histograms
            for key, hist in self._histograms.items():
                name, tags = self._parse_key(key)
                tag_str = self._format_tags(tags)
                lines.append(f"# TYPE {name} histogram")
                for bucket, count in zip(hist.buckets, hist.counts):
                    bucket_tags = {**tags, "le": str(bucket)}
                    bucket_str = self._format_tags(bucket_tags)
                    lines.append(f"{name}_bucket{bucket_str} {count}")
                lines.append(f"{name}_sum{tag_str} {hist.total_sum}")
                lines.append(f"{name}_count{tag_str} {hist.total_count}")
        
        return "\n".join(lines) + "\n"
    
    # ── Internal ──────────────────────────────────────────────────────────────
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key from name and tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"
    
    def _parse_key(self, key: str) -> tuple[str, Dict[str, str]]:
        """Parse a key back into name and tags."""
        if "{" not in key:
            return key, {}
        name, tag_part = key.split("{", 1)
        tag_part = tag_part.rstrip("}")
        tags = {}
        for pair in tag_part.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                tags[k] = v
        return name, tags
    
    def _format_tags(self, tags: Dict[str, str]) -> str:
        """Format tags for Prometheus."""
        if not tags:
            return ""
        return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(tags.items())) + "}"
    
    def _record_history(self, name: str, value: float, tags: Optional[Dict[str, str]], metric_type: str) -> None:
        """Record metric point in history."""
        self._history.append(MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=metric_type,
        ))
    
    # ── Lifecycle ─────────────────────────────────────────────────────────────
    
    async def start_collection(self, interval: float = 60.0) -> None:
        """Start periodic metric collection."""
        if self._running:
            return
        self._running = True
        self._collection_interval = interval
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started (interval=%.1fs)", interval)
    
    async def stop_collection(self) -> None:
        """Stop periodic metric collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self) -> None:
        """Background metric collection loop."""
        while self._running:
            try:
                await self.collect_system_metrics()
            except Exception as e:
                logger.debug("Metric collection error: %s", e)
            
            try:
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
    
    # ── Context Manager ───────────────────────────────────────────────────────
    
    @asynccontextmanager
    async def lifespan(self) -> AsyncGenerator["MetricsCollector", None]:
        """Context manager for metrics collection."""
        try:
            await self.start_collection()
            yield self
        finally:
            await self.stop_collection()


class _TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]]):
        self._collector = collector
        self._name = name
        self._tags = tags
        self._start = 0.0
    
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self._start
        self._collector.observe(self._name, duration, self._tags)
        if exc_type:
            self._collector.increment(f"{self._name}.errors", 1, self._tags)


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience functions
def increment_counter(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter."""
    get_metrics_collector().increment(name, value, tags)


def set_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Set a gauge."""
    get_metrics_collector().set_gauge(name, value, tags)


def observe_histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Observe a histogram value."""
    get_metrics_collector().observe(name, value, tags)


def time_operation(name: str, tags: Optional[Dict[str, str]] = None):
    """Time an operation."""
    return get_metrics_collector().time_operation(name, tags)


@asynccontextmanager
async def track_request(operation: str, tags: Optional[Dict[str, str]] = None):
    """Track a request with timing and error counting."""
    collector = get_metrics_collector()
    start = time.perf_counter()
    request_tags = {**(tags or {}), "operation": operation}
    
    collector.increment("requests.total", 1, request_tags)
    
    try:
        yield
        collector.increment("requests.success", 1, request_tags)
    except Exception as e:
        collector.increment("requests.error", 1, {**request_tags, "error": type(e).__name__})
        raise
    finally:
        duration = time.perf_counter() - start
        collector.observe("request.duration", duration, request_tags)