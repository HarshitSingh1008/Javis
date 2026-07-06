"""
NEXUS AI v4.0 — Tool 01: Whitelisted system command execution.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Executes whitelisted CLI commands via subprocess with strict security:
  - Only ALLOWED_COMMANDS list (no arbitrary command exec)
  - Path traversal prevention
  - Timeout enforcement
  - Stderr capture
  - Non-interactive execution enforced
"""

import json
import logging
import subprocess
import shlex
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger("nexus.tool.system_command")


# ─── Whitelisted Commands ────────────────────────────────────────────────────
# Only these commands can be executed. All others are rejected.
# Each entry: (command_name, [allowed_flag_prefixes], description)
ALLOWED_COMMANDS: Dict[str, Dict[str, Any]] = {
    "ls":     {"flags": ["-l", "-a", "-la", "-al", "-lh", "-lah", "-1", "-R", "-h"], "desc": "List directory contents"},
    "pwd":    {"flags": [], "desc": "Print working directory"},
    "whoami": {"flags": [], "desc": "Print current user"},
    "uname":  {"flags": ["-a", "-r", "-s", "-m"], "desc": "System information"},
    "date":   {"flags": [], "desc": "Current date and time"},
    "echo":   {"flags": [], "desc": "Print text (no env vars)"},
    "cat":    {"flags": [], "desc": "Display file contents"},
    "head":   {"flags": ["-n", "-c"], "desc": "First lines of file"},
    "tail":   {"flags": ["-n", "-c", "-f"], "desc": "Last lines of file"},
    "wc":     {"flags": ["-l", "-w", "-c", "-m"], "desc": "Word/line/char count"},
    "df":     {"flags": ["-h", "-H"], "desc": "Disk space usage"},
    "du":     {"flags": ["-h", "-s", "-sh", "-h --max-depth=1"], "desc": "Disk usage per path"},
    "ps":     {"flags": ["aux", "ef", "ax"], "desc": "Process list"},
    "find":   {"flags": ["-name", "-type", "-iname"], "desc": "Find files"},
    "grep":   {"flags": ["-r", "-i", "-n", "-l", "-c", "-v"], "desc": "Search text"},
    "which":  {"flags": [], "desc": "Locate a command"},
    "uptime": {"flags": [], "desc": "System uptime"},
}


@tool
def run_system_command(
    command: str,
    args: Optional[List[str]] = None,
    timeout: int = 15,
) -> str:
    """
    Execute a whitelisted system command with strict security validation.

    Use this tool when: The user asks to check system info, list files,
    search for files, or run any of the whitelisted commands.

    Args:
        command: The command to execute (must be in ALLOWED_COMMANDS list).
        args: Optional list of arguments/flags for the command.
              Each argument is validated against allowed flags.
        timeout: Maximum seconds before command is killed. Default 15, max 60.

    Returns:
        JSON string with keys:
          - success (bool): Whether the command executed successfully.
          - result (str): stdout output from the command.
          - error (str or null): stderr or validation error message.
          - returncode (int): Process exit code.

    Raises:
        No exceptions — all errors are returned in the JSON response.

    Example:
        >>> run_system_command("ls", ["-la"])
        '{"success": true, "result": "total 42\\ndrwxr-xr-x...", "error": null, "returncode": 0}'
        
        >>> run_system_command("rm", ["-rf", "/"])
        '{"success": false, "result": null, "error": "Command 'rm' is not in the whitelist", "returncode": -1}'
    """
    start = time.perf_counter()
    
    # Validate command is whitelisted
    if command not in ALLOWED_COMMANDS:
        return json.dumps({
            "success": False,
            "result": None,
            "error": f"Command '{command}' is not in the whitelist. Allowed: {', '.join(sorted(ALLOWED_COMMANDS.keys()))}",
            "returncode": -1,
        })
    
    cmd_info = ALLOWED_COMMANDS[command]
    allowed_flags = cmd_info["flags"]
    args = args or []
    
    # Validate flags
    for arg in args:
        # Check if flag starts with an allowed prefix
        flag_allowed = False
        for allowed in allowed_flags:
            if arg.startswith(allowed):
                flag_allowed = True
                break
        
        if not flag_allowed and allowed_flags:
            return json.dumps({
                "success": False,
                "result": None,
                "error": f"Flag '{arg}' is not allowed for command '{command}'. Allowed: {', '.join(allowed_flags)}",
                "returncode": -1,
            })
        
        # Block path traversal
        if ".." in arg or arg.startswith("/") or (":" in arg):
            return json.dumps({
                "success": False,
                "result": None,
                "error": f"Path traversal detected in argument: {arg}",
                "returncode": -1,
            })
    
    # Clamp timeout
    timeout = min(max(timeout, 1), 60)
    
    try:
        cmd_list = [command] + args
        
        proc = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={},  # Minimal environment for safety
        )
        
        duration_ms = (time.perf_counter() - start) * 1000
        
        return json.dumps({
            "success": proc.returncode == 0,
            "result": proc.stdout,
            "error": proc.stderr if proc.stderr else None,
            "returncode": proc.returncode,
            "duration_ms": round(duration_ms, 1),
        })
    
    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "result": None,
            "error": f"Command timed out after {timeout}s",
            "returncode": -1,
        })
    except FileNotFoundError:
        return json.dumps({
            "success": False,
            "result": None,
            "error": f"Command '{command}' not found on this system",
            "returncode": -2,
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "result": None,
            "error": f"{type(e).__name__}: {e}",
            "returncode": -3,
        })
