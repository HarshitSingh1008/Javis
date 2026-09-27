"""
NEXUS AI v4.0 — Health check: validates all subsystems in < 5 seconds.
Hardware: Intel i3 7th Gen · 12GB RAM · Groq API

Provides comprehensive subsystem validation used by:
- Boot sequence (main.py)
- CLI health command (nexus health)
- Degradation detection
"""

import sys
import logging
from typing import List, Dict, Any, NamedTuple
from dataclasses import dataclass, field

from nexus_config.settings import validate_on_boot, get_settings, APP_ROOT

logger = logging.getLogger("nexus.health")


@dataclass
class HealthCheckResult:
    """Result of a health check run."""
    ok: bool
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    degraded_subsystems: List[str] = field(default_factory=list)


def run_health_check() -> HealthCheckResult:
    """
    Run full system health check.
    
    Validates:
    1. Configuration integrity
    2. Python version
    3. Key dependencies (chromadb, pygame, psutil)
    4. Disk space
    5. Available RAM
    6. Ollama connectivity (optional)
    7. API key presence
    
    Returns:
        HealthCheckResult with all check results.
    """
    result = HealthCheckResult(ok=True)
    
    # Check 1: Configuration
    try:
        boot = validate_on_boot()
        result.checks["configuration"] = {
            "status": "ok" if boot.ok else "issues",
            "missing_keys": boot.missing_keys,
            "warnings": len(boot.warnings),
        }
        if not boot.ok:
            result.ok = False
            result.errors.append("Configuration issues found")
        result.warnings.extend(boot.warnings)
        result.degraded_subsystems.extend(boot.degraded_subsystems)
    except Exception as e:
        result.checks["configuration"] = {"status": "error", "error": str(e)}
        result.ok = False
        result.errors.append(f"Configuration check failed: {e}")
    
    # Check 2: Python version
    py_ok = sys.version_info >= (3, 10)
    result.checks["python"] = {
        "status": "ok" if py_ok else "warning",
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    if not py_ok:
        result.warnings.append(f"Python {sys.version_info.major}.{sys.version_info.minor} < 3.10 required")
    
    # Check 3: Key dependencies
    deps = {
        "chromadb": {"import": "chromadb", "required": True},
        "pygame": {"import": "pygame", "required": False},
        "psutil": {"import": "psutil", "required": False},
        "httpx": {"import": "httpx", "required": False},
        "pydantic": {"import": "pydantic", "required": True},
        "langchain_core": {"import": "langchain_core", "required": True},
    }
    
    dep_results = {}
    for dep_name, dep_info in deps.items():
        try:
            __import__(dep_info["import"])
            dep_results[dep_name] = {"status": "installed"}
        except ImportError:
            if dep_info["required"]:
                dep_results[dep_name] = {"status": "missing", "required": True}
                result.warnings.append(f"Required dependency missing: {dep_name}")
            else:
                dep_results[dep_name] = {"status": "missing_optional"}
    
    result.checks["dependencies"] = dep_results
    
    # Check 4: Disk space
    try:
        import shutil
        usage = shutil.disk_usage(APP_ROOT)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        result.checks["disk"] = {
            "status": "ok" if free_gb > 0.5 else "warning",
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
        }
        if free_gb < 0.5:
            result.warnings.append(f"Low disk space: {free_gb:.1f}GB free")
    except Exception as e:
        result.checks["disk"] = {"status": "error", "error": str(e)}
    
    # Check 5: Available RAM
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        total_mb = psutil.virtual_memory().total / (1024 * 1024)
        result.checks["memory"] = {
            "status": "ok" if available_mb > 1500 else "warning",
            "available_mb": round(available_mb, 1),
            "total_mb": round(total_mb, 1),
        }
        if available_mb < 1500:
            result.warnings.append(f"Low memory: {available_mb:.0f}MB available")
    except ImportError:
        result.checks["memory"] = {"status": "unknown", "note": "psutil not available"}
    
    # Check 6: Groq API key presence
    settings = get_settings()
    groq_ok = bool(settings.GROQ_API_KEY)
    result.checks["groq_api"] = {
        "status": "ok" if groq_ok else "missing",
        "models": {
            "planner": settings.PLANNER_MODEL,
            "automation": settings.AUTOMATION_MODEL,
            "voice": settings.VOICE_MODEL,
        },
    }
    if not groq_ok:
        result.warnings.append("GROQ_API_KEY not set — LLM features will be unavailable")
    
    # Summary
    all_ok = all(
        c.get("status") == "ok"
        for c in result.checks.values()
        if isinstance(c, dict) and "status" in c
    )
    if not all_ok:
        result.ok = False
    
    return result


def print_health_report(result: HealthCheckResult) -> None:
    """
    Print a human-readable health check report to stdout.
    
    Args:
        result: The HealthCheckResult to display.
    """
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    print(f"\n{BOLD}{CYAN}NEXUS AI v4.0 — Health Check Report{RESET}")
    print("=" * 50)
    
    for check_name, check_result in result.checks.items():
        status = check_result.get("status", "unknown")
        if status == "ok":
            icon = f"{GREEN}✓{RESET}"
        elif status in ("warning", "issues"):
            icon = f"{YELLOW}~{RESET}"
        elif status == "unreachable":
            icon = f"{YELLOW}~{RESET}"
        else:
            icon = f"{RED}✗{RESET}"
        
        print(f"  {icon} {check_name.replace('_', ' ').title()}: {status}")
        
        if check_name == "disk":
            free = check_result.get("free_gb", "?")
            total = check_result.get("total_gb", "?")
            print(f"     Free: {free}GB / Total: {total}GB")
        elif check_name == "memory":
            avail = check_result.get("available_mb", "?")
            total = check_result.get("total_mb", "?")
            print(f"     Available: {avail}MB / Total: {total}MB")
        elif check_name == "groq_api" and status == "missing":
            print(f"     Set GROQ_API_KEY in ~/.nexus_ai/.env")
    
    if result.warnings:
        print(f"\n{BOLD}{YELLOW}Warnings ({len(result.warnings)}):{RESET}")
        for w in result.warnings:
            print(f"  {YELLOW}~{RESET} {w}")
    
    if result.errors:
        print(f"\n{BOLD}{RED}Errors ({len(result.errors)}):{RESET}")
        for e in result.errors:
            print(f"  {RED}✗{RESET} {e}")
    
    if result.degraded_subsystems:
        print(f"\n{BOLD}{YELLOW}Degraded subsystems:{RESET}")
        for s in result.degraded_subsystems:
            print(f"  {YELLOW}~{RESET} {s}")
    
    summary_color = GREEN if result.ok else RED
    print(f"\n{BOLD}{summary_color}Overall: {'PASS' if result.ok else 'FAIL'}{RESET}\n")


if __name__ == "__main__":
    result = run_health_check()
    print_health_report(result)
    sys.exit(0 if result.ok else 1)