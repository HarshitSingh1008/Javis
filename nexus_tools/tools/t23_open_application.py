"""
NEXUS AI v4.0 — Tool 23: Robust application launcher for Windows.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Launches GUI applications detached, verifies window/process appearance.
Never blocks on the launched process. Returns structured result with verification.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from langchain_core.tools import tool

logger = logging.getLogger("nexus.tool.open_application")

# ─── Windows-specific imports ──────────────────────────────────────────────────
if sys.platform.startswith("win"):
    try:
        import ctypes
        from ctypes import wintypes
        _CTYPES_AVAILABLE = True
    except ImportError:
        _CTYPES_AVAILABLE = False
else:
    _CTYPES_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ─── Application Alias Table (config-driven, extensible) ───────────────────────

APP_ALIASES: Dict[str, Dict[str, Any]] = {
    "notepad": {
        "executables": ["notepad.exe"],
        "uris": [],
        "start_menu_names": ["Notepad"],
        "verify_window_titles": ["Untitled - Notepad", "Notepad"],
        "verify_process_names": ["notepad.exe"],
    },
    "calculator": {
        "executables": ["calc.exe"],
        "uris": ["ms-calculator:"],
        "start_menu_names": ["Calculator"],
        "verify_window_titles": ["Calculator"],
        "verify_process_names": ["Calculator.exe", "calc.exe"],
    },
    "calc": {
        "executables": ["calc.exe"],
        "uris": ["ms-calculator:"],
        "start_menu_names": ["Calculator"],
        "verify_window_titles": ["Calculator"],
        "verify_process_names": ["Calculator.exe", "calc.exe"],
    },
    "paint": {
        "executables": ["mspaint.exe"],
        "uris": [],
        "start_menu_names": ["Paint"],
        "verify_window_titles": ["Untitled - Paint", "Paint"],
        "verify_process_names": ["mspaint.exe"],
    },
    "mspaint": {
        "executables": ["mspaint.exe"],
        "uris": [],
        "start_menu_names": ["Paint"],
        "verify_window_titles": ["Untitled - Paint", "Paint"],
        "verify_process_names": ["mspaint.exe"],
    },
    "explorer": {
        "executables": ["explorer.exe"],
        "uris": [],
        "start_menu_names": ["File Explorer", "This PC"],
        "verify_window_titles": ["File Explorer", "This PC", "Documents"],
        "verify_process_names": ["explorer.exe"],
    },
    "file explorer": {
        "executables": ["explorer.exe"],
        "uris": [],
        "start_menu_names": ["File Explorer"],
        "verify_window_titles": ["File Explorer", "This PC"],
        "verify_process_names": ["explorer.exe"],
    },
    "cmd": {
        "executables": ["cmd.exe"],
        "uris": [],
        "start_menu_names": ["Command Prompt"],
        "verify_window_titles": ["Command Prompt", "cmd.exe"],
        "verify_process_names": ["cmd.exe"],
    },
    "command prompt": {
        "executables": ["cmd.exe"],
        "uris": [],
        "start_menu_names": ["Command Prompt"],
        "verify_window_titles": ["Command Prompt"],
        "verify_process_names": ["cmd.exe"],
    },
    "powershell": {
        "executables": ["powershell.exe", "pwsh.exe"],
        "uris": [],
        "start_menu_names": ["Windows PowerShell", "PowerShell 7"],
        "verify_window_titles": ["Windows PowerShell", "PowerShell"],
        "verify_process_names": ["powershell.exe", "pwsh.exe"],
    },
    "settings": {
        "executables": [],
        "uris": ["ms-settings:"],
        "start_menu_names": ["Settings"],
        "verify_window_titles": ["Settings"],
        "verify_process_names": ["SystemSettings.exe"],
    },
    "control panel": {
        "executables": ["control.exe"],
        "uris": [],
        "start_menu_names": ["Control Panel"],
        "verify_window_titles": ["Control Panel"],
        "verify_process_names": ["control.exe"],
    },
    "chrome": {
        "executables": ["chrome.exe"],
        "uris": [],
        "start_menu_names": ["Google Chrome"],
        "verify_window_titles": ["Google Chrome", "New Tab - Google Chrome"],
        "verify_process_names": ["chrome.exe"],
    },
    "edge": {
        "executables": ["msedge.exe"],
        "uris": [],
        "start_menu_names": ["Microsoft Edge"],
        "verify_window_titles": ["Microsoft Edge", "New Tab - Microsoft Edge"],
        "verify_process_names": ["msedge.exe"],
    },
    "vscode": {
        "executables": ["code.exe"],
        "uris": [],
        "start_menu_names": ["Visual Studio Code"],
        "verify_window_titles": ["Visual Studio Code", "Code - "],
        "verify_process_names": ["code.exe"],
    },
    "vs code": {
        "executables": ["code.exe"],
        "uris": [],
        "start_menu_names": ["Visual Studio Code"],
        "verify_window_titles": ["Visual Studio Code"],
        "verify_process_names": ["code.exe"],
    },
    "task manager": {
        "executables": ["taskmgr.exe"],
        "uris": [],
        "start_menu_names": ["Task Manager"],
        "verify_window_titles": ["Task Manager"],
        "verify_process_names": ["Taskmgr.exe"],
    },
    "terminal": {
        "executables": ["wt.exe", "cmd.exe"],
        "uris": [],
        "start_menu_names": ["Windows Terminal"],
        "verify_window_titles": ["Windows Terminal", "Command Prompt"],
        "verify_process_names": ["WindowsTerminal.exe", "cmd.exe"],
    },
    "word": {
        "executables": ["winword.exe"],
        "uris": [],
        "start_menu_names": ["Word"],
        "verify_window_titles": ["Microsoft Word", "Document - Word"],
        "verify_process_names": ["WINWORD.EXE"],
    },
    "excel": {
        "executables": ["excel.exe"],
        "uris": [],
        "start_menu_names": ["Excel"],
        "verify_window_titles": ["Microsoft Excel", "Workbook - Excel"],
        "verify_process_names": ["EXCEL.EXE"],
    },
    "outlook": {
        "executables": ["outlook.exe"],
        "uris": [],
        "start_menu_names": ["Outlook"],
        "verify_window_titles": ["Microsoft Outlook", "Inbox - Outlook"],
        "verify_process_names": ["OUTLOOK.EXE"],
    },
}

# Known paths for common apps (Windows)
KNOWN_PATHS: Dict[str, List[str]] = {
    "notepad.exe": ["C:\\Windows\\System32\\notepad.exe", "C:\\Windows\\notepad.exe"],
    "calc.exe": ["C:\\Windows\\System32\\calc.exe"],
    "mspaint.exe": ["C:\\Windows\\System32\\mspaint.exe"],
    "explorer.exe": ["C:\\Windows\\explorer.exe"],
    "cmd.exe": ["C:\\Windows\\System32\\cmd.exe"],
    "powershell.exe": ["C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"],
    "control.exe": ["C:\\Windows\\System32\\control.exe"],
    "taskmgr.exe": ["C:\\Windows\\System32\\taskmgr.exe"],
    "code.exe": [
        os.path.expanduser("~\\AppData\\Local\\Programs\\Microsoft VS Code\\bin\\code.exe"),
        "C:\\Program Files\\Microsoft VS Code\\bin\\code.exe",
        "C:\\Program Files (x86)\\Microsoft VS Code\\bin\\code.exe",
    ],
    "chrome.exe": [
        os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe"),
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    ],
    "msedge.exe": [
        "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    ],
    "wt.exe": [
        os.path.expanduser("~\\AppData\\Local\\Microsoft\\WindowsApps\\wt.exe"),
        "C:\\Program Files\\WindowsApps\\Microsoft.WindowsTerminal_*\\wt.exe",
    ],
}


# ─── Window Enumeration (ctypes) ──────────────────────────────────────────────

def _enum_windows() -> List[Dict[str, Any]]:
    """Enumerate visible top-level windows using ctypes."""
    if not _CTYPES_AVAILABLE:
        return []
    
    windows = []
    
    EnumWindows = ctypes.windll.user32.EnumWindows
    GetWindowTextW = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    GetWindowThreadProcessId = ctypes.windll.user32.GetWindowThreadProcessId
    
    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def enum_callback(hwnd, lparam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLengthW(hwnd)
            if length > 0:
                buffer = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buffer, length + 1)
                title = buffer.value
                pid = wintypes.DWORD()
                GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                windows.append({"hwnd": hwnd, "title": title, "pid": pid.value})
        return True
    
    EnumWindows(enum_callback, 0)
    return windows


def _find_window_by_title(patterns: List[str]) -> Optional[Dict[str, Any]]:
    """Find a visible window whose title matches any pattern."""
    windows = _enum_windows()
    for w in windows:
        for pattern in patterns:
            if pattern.lower() in w["title"].lower():
                return w
    return None


def _find_process_by_name(names: List[str], after_time: float) -> Optional[Dict[str, Any]]:
    """Find a process by name that started after the given timestamp."""
    if not _PSUTIL_AVAILABLE:
        return None
    
    try:
        for proc in psutil.process_iter(["pid", "name", "create_time", "exe"]):
            try:
                pname = proc.info["name"]
                if pname and any(n.lower() in pname.lower() for n in names):
                    if proc.info["create_time"] >= after_time - 1.0:  # 1s tolerance
                        return {
                            "pid": proc.info["pid"],
                            "name": pname,
                            "exe": proc.info["exe"],
                            "create_time": proc.info["create_time"],
                        }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.debug(f"Process search failed: {e}")
    return None


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still alive."""
    if not _PSUTIL_AVAILABLE:
        return False
    try:
        return psutil.Process(pid).is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    except Exception:
        return False


# ─── Launch Strategies ────────────────────────────────────────────────────────

def _launch_via_os_startfile(path_or_uri: str) -> bool:
    """Launch using os.startfile (best for registered apps/URIs)."""
    try:
        os.startfile(path_or_uri)
        return True
    except Exception as e:
        logger.debug(f"os.startfile failed for {path_or_uri}: {e}")
        return False


def _launch_via_popen_detached(cmd: List[str]) -> Optional[int]:
    """Launch detached process via Popen (Windows)."""
    if not sys.platform.startswith("win"):
        try:
            proc = subprocess.Popen(cmd, start_new_session=True)
            return proc.pid
        except Exception:
            return None
    
    CREATE_NO_WINDOW = 0x08000000
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    
    flags = CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    
    try:
        proc = subprocess.Popen(
            cmd,
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc.pid
    except Exception as e:
        logger.debug(f"Popen detached failed for {cmd}: {e}")
        return None


def _launch_via_shell_start(app_name: str) -> bool:
    """Launch via 'start' command (Windows shell)."""
    if not sys.platform.startswith("win"):
        return False
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", app_name],
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        logger.debug(f"Shell start failed for {app_name}: {e}")
        return False


# ─── Main Tool ────────────────────────────────────────────────────────────────

@tool
def open_application(name: str) -> str:
    """
    Launch a desktop application by name and verify it opened.
    
    Use this tool when: The user asks to open/launch/start an application
    like Notepad, Calculator, Chrome, VS Code, Settings, etc.
    
    Args:
        name: Application name (e.g., "notepad", "calculator", "chrome", 
              "vscode", "settings", "explorer", "cmd", "powershell", "paint").
              Case-insensitive. Supports aliases from the built-in table.
    
    Returns:
        JSON string with keys:
          - success (bool): Whether launch was attempted
          - result (str): Human-readable status
          - error (str or null): Error details if failed
          - verified (bool): Whether window/process was confirmed
          - verification_detail (str): Details of verification
          - launch_method (str): How it was launched
          - pid (int or null): Process ID if captured
    
    Examples:
        >>> open_application("notepad")
        >>> open_application("calculator")
        >>> open_application("chrome")
        >>> open_application("settings")
    """
    start_time = time.perf_counter()
    name_lower = name.strip().lower()
    
    # Look up alias
    alias = APP_ALIASES.get(name_lower)
    if not alias:
        # Try fuzzy match
        for key in APP_ALIASES:
            if key in name_lower or name_lower in key:
                alias = APP_ALIASES[key]
                name_lower = key
                break
    
    tried_methods = []
    launched_pid = None
    launch_method = "none"
    launch_success = False
    
    if alias:
        # Strategy 1: URI scheme (fastest for Store apps like Settings)
        for uri in alias.get("uris", []):
            tried_methods.append(f"uri:{uri}")
            if _launch_via_os_startfile(uri):
                launch_success = True
                launch_method = f"uri:{uri}"
                break
        
        # Strategy 2: Known absolute paths
        if not launch_success:
            for exe in alias.get("executables", []):
                for path in KNOWN_PATHS.get(exe, []):
                    if Path(path).exists():
                        tried_methods.append(f"path:{path}")
                        pid = _launch_via_popen_detached([path])
                        if pid:
                            launched_pid = pid
                            launch_success = True
                            launch_method = f"popen:{path}"
                            break
                if launch_success:
                    break
        
        # Strategy 3: PATH lookup via shutil.which
        if not launch_success:
            for exe in alias.get("executables", []):
                tried_methods.append(f"which:{exe}")
                path = shutil.which(exe)
                if path:
                    pid = _launch_via_popen_detached([path])
                    if pid:
                        launched_pid = pid
                        launch_success = True
                        launch_method = f"popen:{path}"
                        break
        
        # Strategy 4: os.startfile on executable name (uses file associations)
        if not launch_success:
            for exe in alias.get("executables", []):
                tried_methods.append(f"startfile:{exe}")
                if _launch_via_os_startfile(exe):
                    launch_success = True
                    launch_method = f"startfile:{exe}"
                    break
        
        # Strategy 5: Shell 'start' command
        if not launch_success:
            for exe in alias.get("executables", []):
                tried_methods.append(f"shell:start {exe}")
                if _launch_via_shell_start(exe):
                    launch_success = True
                    launch_method = f"shell:start {exe}"
                    break
        
        # Strategy 6: Start Menu .lnk search (last resort)
        if not launch_success:
            for start_name in alias.get("start_menu_names", []):
                tried_methods.append(f"start_menu:{start_name}")
                # This is complex; skip for now, rely on above methods
    
    else:
        # No alias — try direct launch
        tried_methods.append(f"direct:{name_lower}")
        if _launch_via_os_startfile(name_lower):
            launch_success = True
            launch_method = f"startfile:{name_lower}"
        elif sys.platform.startswith("win") and _launch_via_shell_start(name_lower):
            launch_success = True
            launch_method = f"shell:start {name_lower}"
        else:
            path = shutil.which(name_lower)
            if path:
                pid = _launch_via_popen_detached([path])
                if pid:
                    launched_pid = pid
                    launch_success = True
                    launch_method = f"popen:{path}"
    
    # Verification phase (poll up to 5 seconds)
    verified = False
    verification_detail = ""
    verify_window_titles = alias.get("verify_window_titles", []) if alias else [name_lower]
    verify_process_names = alias.get("verify_process_names", [name_lower + ".exe"]) if alias else [name_lower + ".exe"]
    
    if launch_success:
        # Poll for window or process
        for i in range(50):  # 50 * 100ms = 5s
            time.sleep(0.1)
            
            # Check window (case-insensitive partial match)
            window = _find_window_by_title(verify_window_titles)
            if window:
                verified = True
                verification_detail = f"Window found: '{window['title']}' (PID {window['pid']})"
                if launched_pid is None:
                    launched_pid = window["pid"]
                break
            
            # Check process
            proc = _find_process_by_name(verify_process_names, start_time - 2.0)  # Allow 2s before start
            if proc:
                verified = True
                verification_detail = f"Process found: {proc['name']} (PID {proc['pid']})"
                if launched_pid is None:
                    launched_pid = proc["pid"]
                break
            
            # If we have a launched PID, check if it's still alive
            if launched_pid and _is_pid_alive(launched_pid):
                verified = True
                verification_detail = f"Launched process alive (PID {launched_pid})"
                break
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    if not launch_success:
        return json.dumps({
            "success": False,
            "result": None,
            "error": f"Failed to launch '{name}'. Tried: {', '.join(tried_methods)}",
            "verified": False,
            "verification_detail": "",
            "launch_method": "none",
            "pid": None,
            "duration_ms": round(duration_ms, 1),
        })
    
    if verified:
        result_msg = f"Launched {name_lower} (verified: {verification_detail})"
    else:
        result_msg = f"Launched {name_lower} via {launch_method} (window/process not yet confirmed)"
    
    return json.dumps({
        "success": True,
        "result": result_msg,
        "error": None,
        "verified": verified,
        "verification_detail": verification_detail,
        "launch_method": launch_method,
        "pid": launched_pid,
        "duration_ms": round(duration_ms, 1),
    })


# ─── Startup Assertion Helper ────────────────────────────────────────────────

def validate_tool_schema(dummy: str) -> Tuple[bool, str]:
    """Validate this tool has proper schema and description."""
    import inspect
    
    # Check docstring exists and is meaningful
    doc = open_application.__doc__ or ""
    if not doc.strip():
        return False, "open_application: missing docstring"
    
    if "Use this tool when" not in doc:
        return False, "open_application: missing 'Use this tool when' in docstring"
    
    # Check description length (first line)
    first_line = doc.strip().split("\n")[0]
    if len(first_line) > 200:
        return False, f"open_application: description too long ({len(first_line)} chars)"
    
    return True, "open_application: OK"