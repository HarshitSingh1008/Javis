"""
NEXUS AI v4.0 — Tool 13: System monitoring via psutil.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides real-time system metrics: CPU, RAM, disk, network, and process info.
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger("nexus.tool.system_monitor")


def system_monitor(
    metric: str,
) -> str:
    """
    Get real-time system performance metrics.
    
    Use this tool when: The user asks about CPU usage, RAM usage, disk space,
    network activity, or any system performance information.
    
    Args:
        metric: Which metric to return. Options:
                "all" - All metrics (default)
                "cpu" - CPU usage per core
                "ram" - RAM usage details
                "disk" - Disk usage for all partitions
                "network" - Network I/O counters
                "temperature" - Hardware temperatures (if available)
                "summary" - Quick overview of key metrics
    
    Returns:
        JSON string with keys:
          - success (bool): Whether metrics were collected.
          - result (dict): The requested metrics.
          - error (str or null): Error message if failed.
    
    Examples:
        >>> system_monitor("all")
        >>> system_monitor("cpu")
        >>> system_monitor("summary")
    """
    start = time.perf_counter()
    
    try:
        import psutil
        
        result = {}
        
        if metric in ("all", "cpu", "summary"):
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_freq = psutil.cpu_freq()
            result["cpu"] = {
                "percent_per_core": cpu_percent,
                "percent_total": sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0,
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "load_avg_1min": psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else None,
            }
        
        if metric in ("all", "ram", "summary"):
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            result["ram"] = {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "percent_used": mem.percent,
                "swap_total_gb": round(swap.total / (1024**3), 2),
                "swap_used_gb": round(swap.used / (1024**3), 2),
                "swap_percent": swap.percent,
            }
        
        if metric in ("all", "disk", "summary"):
            partitions = []
            for part in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    partitions.append({
                        "device": part.device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "total_gb": round(usage.total / (1024**3), 2),
                        "used_gb": round(usage.used / (1024**3), 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                        "percent_used": usage.percent,
                    })
                except (PermissionError, OSError):
                    continue
            result["disk"] = {
                "partitions": partitions,
                "io_counters": {
                    "read_mb": round(psutil.disk_io_counters().read_bytes / (1024**2), 1) if psutil.disk_io_counters() else 0,
                    "write_mb": round(psutil.disk_io_counters().write_bytes / (1024**2), 1) if psutil.disk_io_counters() else 0,
                } if psutil.disk_io_counters() else None,
            }
        
        if metric in ("all", "network"):
            net = psutil.net_io_counters()
            connections = []
            for conn in psutil.net_connections(kind="inet")[:20]:
                try:
                    connections.append({
                        "fd": conn.fd,
                        "family": str(conn.family),
                        "type": str(conn.type),
                        "laddr": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "",
                        "raddr": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                        "status": conn.status,
                        "pid": conn.pid,
                    })
                except Exception:
                    continue
            result["network"] = {
                "bytes_sent_mb": round(net.bytes_sent / (1024**2), 1),
                "bytes_recv_mb": round(net.bytes_recv / (1024**2), 1),
                "packets_sent": net.packets_sent,
                "packets_recv": net.packets_recv,
                "connections": connections[:10],
            }
        
        if metric == "temperature":
            temps = {}
            try:
                for name, entries in psutil.sensors_temperatures().items():
                    temps[name] = [{"label": e.label, "current": e.current, "high": e.high, "critical": e.critical} for e in entries]
            except (AttributeError, PermissionError):
                temps = {"error": "Temperature sensors not available on this system"}
            result["temperature"] = temps
        
        if metric == "summary":
            # Only return key metrics
            result = {
                "cpu_percent": result.get("cpu", {}).get("percent_total", 0),
                "ram_percent": result.get("ram", {}).get("percent_used", 0),
                "ram_available_gb": result.get("ram", {}).get("available_gb", 0),
                "disk_usage": [{"mount": p["mountpoint"], "percent": p["percent_used"]} for p in result.get("disk", {}).get("partitions", [])],
            }
        
        return json.dumps({
            "success": True,
            "result": result,
            "error": None,
        })
    
    except ImportError:
        return json.dumps({
            "success": False, "result": None,
            "error": "psutil not installed. Install with: pip install psutil"
        })
    except Exception as e:
        logger.error(f"system_monitor error: {e}", exc_info=True)
        return json.dumps({
            "success": False, "result": None,
            "error": f"{type(e).__name__}: {e}"
        })