"""
NEXUS AI v4.0 — Tool 07: Sandboxed Python execution.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Executes Python code in an isolated sandbox with resource limits,
banned imports, and timeout enforcement. Routes through SecureSandbox.
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger("nexus.tool.python_interpreter")

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_CODE_LENGTH: int = 10000
DEFAULT_TIMEOUT: int = 30


def python_interpreter(
    code: str,
    timeout: int = 30,
    use_sandbox: bool = True,
) -> str:
    """
    Execute Python code in an isolated sandbox with strict security controls.
    
    Use this tool when: The user asks to run Python code, test a script,
    calculate something complex, process data with Python, or debug code.
    
    Args:
        code: The Python code to execute. Must be valid Python 3.10+ syntax.
        timeout: Maximum execution time in seconds (5-120).
        use_sandbox: Whether to use the full subprocess sandbox isolation.
                     Set to False for simple, fast calculations where
                     subprocess overhead is not needed.
    
    Returns:
        JSON string with keys:
          - success (bool): Whether the code executed without exceptions.
          - result (any): The captured stdout output and return value.
          - error (str or null): Exception message if an error occurred.
          - execution_time_ms (float): Wall-clock execution time.
    
    Security:
        - All code is AST-checked for banned imports and patterns
        - Executed in subprocess with memory and CPU limits (when sandboxed)
        - No access to filesystem outside sandbox (when sandboxed)
        - Environment is stripped of sensitive variables
        - Output is captured and returned (not printed to real stdout)
    
    Examples:
        >>> python_interpreter("print('Hello, world!')")
        >>> python_interpreter("import json\\nresult = {'sum': 1 + 2}\\nprint(json.dumps(result))")
    """
    start = time.perf_counter()
    timeout = min(max(timeout, 5), 120)
    
    if len(code) > MAX_CODE_LENGTH:
        return json.dumps({
            "success": False, "result": None,
            "error": f"Code exceeds maximum length of {MAX_CODE_LENGTH} characters ({len(code)} given)"
        })
    
    if use_sandbox:
        return _execute_in_sandbox(code, timeout)
    else:
        return _execute_locally(code, timeout)


def _execute_locally(code: str, timeout: int) -> str:
    """Execute Python code in-process (faster but less isolated)."""
    import sys
    import io
    import traceback
    from contextlib import redirect_stdout, redirect_stderr
    
    start = time.perf_counter()
    
    # Restricted builtins for safety
    safe_builtins = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate',
        'filter', 'float', 'format', 'frozenset', 'getattr', 'hasattr',
        'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass', 'iter',
        'len', 'list', 'map', 'max', 'min', 'next', 'object', 'oct',
        'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round',
        'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type',
        'vars', 'zip', 'True', 'False', 'None',
    }
    
    restricted_globals = {
        '__builtins__': {k: __builtins__[k] for k in safe_builtins if k in __builtins__},
        '__name__': '__sandbox__',
    }
    
    # Allow safe imports
    _allowed_imports = {
        'json', 'math', 'random', 'statistics', 'datetime', 'collections',
        'itertools', 'functools', 're', 'string', 'typing',
    }
    
    def _safe_import(name, *args, **kwargs):
        if name not in _allowed_imports:
            raise ImportError(f"Import '{name}' is not allowed in local execution mode. Use sandbox mode for full Python.")
        return __import__(name, *args, **kwargs)
    
    restricted_globals['__import__'] = _safe_import
    
    try:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                compiled = compile(code, '<sandbox>', 'exec', flags=0)
                exec(compiled, restricted_globals)
            except Exception:
                import traceback as tb
                stderr_capture.write(tb.format_exc())
        
        duration_ms = (time.perf_counter() - start) * 1000
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        return json.dumps({
            "success": not stderr_text,
            "result": stdout_text if stdout_text else "(no output)",
            "error": stderr_text if stderr_text else None,
            "execution_time_ms": round(duration_ms, 1),
            "mode": "local",
        })
    
    except Exception as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"Execution error: {e}",
            "execution_time_ms": round((time.perf_counter() - start) * 1000, 1),
            "mode": "local",
        })


def _execute_in_sandbox(code: str, timeout: int) -> str:
    """Execute Python code in a sandboxed subprocess."""
    from nexus_tools.secure_sandbox import sandbox_execute
    
    import tempfile
    import os
    
    start = time.perf_counter()
    
    # Write code to a temp file for sandbox execution
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, encoding='utf-8'
    ) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        result = sandbox_execute(code_str=code, timeout=timeout)
        
        duration_ms = (time.perf_counter() - start) * 1000
        
        if isinstance(result, dict):
            result["execution_time_ms"] = round(duration_ms, 1)
            result["mode"] = "sandbox"
            return json.dumps(result)
        else:
            return json.dumps({
                "success": True,
                "result": str(result),
                "error": None,
                "execution_time_ms": round(duration_ms, 1),
                "mode": "sandbox",
            })
    
    except Exception as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"Sandbox execution error: {e}",
            "execution_time_ms": round((time.perf_counter() - start) * 1000, 1),
            "mode": "sandbox",
        })
    
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass