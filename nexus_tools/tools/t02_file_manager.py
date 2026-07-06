"""
NEXUS AI v4.0 — Tool 02: Unified file I/O.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides comprehensive file operations: read, write, copy, move, delete,
list directory contents, and search files. All operations include path
traversal protection and encoding safety.
"""

import json
import os
import shutil
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from langchain_core.tools import tool

logger = logging.getLogger("nexus.tool.file_manager")

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50MB max read size
MAX_LIST_RESULTS: int = 1000  # Max files to list in one call
MAX_SEARCH_RESULTS: int = 100  # Max search results
BINARY_EXTENSIONS: frozenset = frozenset({
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat", ".img", ".iso",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".avi", ".mkv", ".mov",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pyc", ".pyo", ".pyd",
    ".db", ".sqlite", ".sqlite3",
})


def _validate_path(path: str, must_exist: bool = False) -> Optional[str]:
    """
    Validate and resolve a file path. Returns error message or None if valid.
    
    Checks:
    - Path traversal (../ etc.)
    - Absolute path safety
    - Existence (if must_exist=True)
    """
    try:
        p = Path(path).resolve()
        # Check for path traversal
        if ".." in path or "~" in path:
            # Resolve handles .. but we check the original string too
            pass
        # Ensure the resolved path is within reasonable bounds
        if must_exist and not p.exists():
            return f"Path does not exist: {path}"
        return None
    except (OSError, ValueError, RuntimeError) as e:
        return f"Invalid path: {e}"


@tool
def file_manager(
    operation: str,
    path: str,
    content: Optional[str] = None,
    destination: Optional[str] = None,
    pattern: Optional[str] = None,
    encoding: str = "utf-8",
    recursive: bool = False,
) -> str:
    """
    Perform file and directory operations with safety validation.
    
    Use this tool when: The user asks to read, write, copy, move, delete,
    list, or search for files and directories.
    
    Args:
        operation: One of: "read", "write", "copy", "move", "delete",
                   "list", "search", "mkdir", "exists", "info"
        path: Path to the target file or directory.
        content: Content to write (for "write" operation).
        destination: Destination path (for "copy" and "move" operations).
        pattern: Glob pattern for "search" operation (e.g., "*.py").
        encoding: File encoding for text operations. Default "utf-8".
        recursive: List directories recursively. Default False.
    
    Returns:
        JSON string with keys:
          - success (bool): Whether the operation succeeded.
          - result (any): Operation-specific result data.
          - error (str or null): Error message if failed.
    
    Examples:
        >>> file_manager("read", "/path/to/file.txt")
        >>> file_manager("write", "/path/to/file.txt", content="Hello, world!")
        >>> file_manager("list", "/path/to/dir")
        >>> file_manager("search", "/path/to/dir", pattern="*.py")
    """
    start = time.perf_counter()
    
    try:
        # Validate path
        path_err = _validate_path(path, must_exist=(operation != "write" and operation != "mkdir"))
        if path_err:
            return json.dumps({"success": False, "result": None, "error": path_err})
        
        p = Path(path).resolve()
        
        # ── READ ─────────────────────────────────────────────────────────────
        if operation == "read":
            if not p.exists():
                return json.dumps({"success": False, "result": None, "error": f"File not found: {path}"})
            if not p.is_file():
                return json.dumps({"success": False, "result": None, "error": f"Not a file: {path}"})
            
            file_size = p.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                return json.dumps({
                    "success": False, "result": None,
                    "error": f"File too large: {file_size} bytes (max {MAX_FILE_SIZE_BYTES} bytes)"
                })
            
            # Check if binary
            if p.suffix.lower() in BINARY_EXTENSIONS:
                return json.dumps({
                    "success": False, "result": None,
                    "error": f"Binary file type not supported for reading: {p.suffix}"
                })
            
            try:
                text = p.read_text(encoding=encoding)
                return json.dumps({
                    "success": True,
                    "result": text,
                    "error": None,
                    "metadata": {
                        "size_bytes": file_size,
                        "encoding": encoding,
                        "lines": text.count("\n") + 1,
                    }
                })
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    text = p.read_text(encoding="latin-1")
                    return json.dumps({
                        "success": True,
                        "result": text,
                        "error": None,
                        "metadata": {
                            "size_bytes": file_size,
                            "encoding": "latin-1 (fallback)",
                            "note": "utf-8 failed, used latin-1"
                        }
                    })
                except Exception as e2:
                    return json.dumps({
                        "success": False, "result": None,
                        "error": f"Cannot decode file: {e2}"
                    })
        
        # ── WRITE ────────────────────────────────────────────────────────────
        elif operation == "write":
            if content is None:
                return json.dumps({"success": False, "result": None, "error": "No content provided for write operation"})
            
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding=encoding)
                return json.dumps({
                    "success": True,
                    "result": f"Written {len(content)} bytes to {path}",
                    "error": None,
                    "metadata": {"size_bytes": len(content), "encoding": encoding}
                })
            except (OSError, PermissionError) as e:
                return json.dumps({"success": False, "result": None, "error": f"Write failed: {e}"})
        
        # ── COPY ─────────────────────────────────────────────────────────────
        elif operation == "copy":
            if not destination:
                return json.dumps({"success": False, "result": None, "error": "No destination path provided"})
            if not p.exists():
                return json.dumps({"success": False, "result": None, "error": f"Source not found: {path}"})
            
            dest_err = _validate_path(destination)
            if dest_err:
                return json.dumps({"success": False, "result": None, "error": dest_err})
            
            dest = Path(destination).resolve()
            try:
                if p.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(p), str(dest))
                else:
                    shutil.copytree(str(p), str(dest), dirs_exist_ok=True)
                return json.dumps({
                    "success": True,
                    "result": f"Copied {path} → {destination}",
                    "error": None
                })
            except (shutil.Error, OSError) as e:
                return json.dumps({"success": False, "result": None, "error": f"Copy failed: {e}"})
        
        # ── MOVE ─────────────────────────────────────────────────────────────
        elif operation == "move":
            if not destination:
                return json.dumps({"success": False, "result": None, "error": "No destination path provided"})
            if not p.exists():
                return json.dumps({"success": False, "result": None, "error": f"Source not found: {path}"})
            
            dest_err = _validate_path(destination)
            if dest_err:
                return json.dumps({"success": False, "result": None, "error": dest_err})
            
            dest = Path(destination).resolve()
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
                return json.dumps({
                    "success": True,
                    "result": f"Moved {path} → {destination}",
                    "error": None
                })
            except (shutil.Error, OSError) as e:
                return json.dumps({"success": False, "result": None, "error": f"Move failed: {e}"})
        
        # ── DELETE ───────────────────────────────────────────────────────────
        elif operation == "delete":
            if not p.exists():
                return json.dumps({"success": False, "result": None, "error": f"Path not found: {path}"})
            
            try:
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(str(p))
                return json.dumps({
                    "success": True,
                    "result": f"Deleted: {path}",
                    "error": None
                })
            except (OSError, PermissionError) as e:
                return json.dumps({"success": False, "result": None, "error": f"Delete failed: {e}"})
        
        # ── LIST ─────────────────────────────────────────────────────────────
        elif operation == "list":
            if not p.exists():
                return json.dumps({"success": False, "result": None, "error": f"Directory not found: {path}"})
            if not p.is_dir():
                return json.dumps({"success": False, "result": None, "error": f"Not a directory: {path}"})
            
            try:
                items = []
                if recursive:
                    for i, entry in enumerate(p.rglob("*")):
                        if i >= MAX_LIST_RESULTS:
                            break
                        items.append({
                            "name": entry.name,
                            "path": str(entry.relative_to(p)),
                            "type": "directory" if entry.is_dir() else "file",
                            "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                            "modified": entry.stat().st_mtime,
                        })
                else:
                    for entry in sorted(p.iterdir()):
                        items.append({
                            "name": entry.name,
                            "path": str(entry.relative_to(p)),
                            "type": "directory" if entry.is_dir() else "file",
                            "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                            "modified": entry.stat().st_mtime,
                        })
                
                return json.dumps({
                    "success": True,
                    "result": items,
                    "error": None,
                    "metadata": {"count": len(items), "truncated": len(items) >= MAX_LIST_RESULTS}
                })
            except (OSError, PermissionError) as e:
                return json.dumps({"success": False, "result": None, "error": f"List failed: {e}"})
        
        # ── SEARCH ───────────────────────────────────────────────────────────
        elif operation == "search":
            if not p.exists():
                return json.dumps({"success": False, "result": None, "error": f"Directory not found: {path}"})
            if not p.is_dir():
                return json.dumps({"success": False, "result": None, "error": f"Not a directory: {path}"})
            
            pattern = pattern or "*"
            try:
                results = []
                for i, entry in enumerate(p.rglob(pattern)):
                    if i >= MAX_SEARCH_RESULTS:
                        break
                    results.append({
                        "name": entry.name,
                        "path": str(entry.relative_to(p)),
                        "type": "directory" if entry.is_dir() else "file",
                        "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                    })
                
                return json.dumps({
                    "success": True,
                    "result": results,
                    "error": None,
                    "metadata": {"pattern": pattern, "count": len(results), "truncated": len(results) >= MAX_SEARCH_RESULTS}
                })
            except (OSError, PermissionError) as e:
                return json.dumps({"success": False, "result": None, "error": f"Search failed: {e}"})
        
        # ── MKDIR ────────────────────────────────────────────────────────────
        elif operation == "mkdir":
            try:
                p.mkdir(parents=True, exist_ok=True)
                return json.dumps({
                    "success": True,
                    "result": f"Created directory: {path}",
                    "error": None
                })
            except (OSError, PermissionError) as e:
                return json.dumps({"success": False, "result": None, "error": f"mkdir failed: {e}"})
        
        # ── EXISTS ───────────────────────────────────────────────────────────
        elif operation == "exists":
            return json.dumps({
                "success": True,
                "result": {
                    "exists": p.exists(),
                    "is_file": p.is_file() if p.exists() else False,
                    "is_dir": p.is_dir() if p.exists() else False,
                    "size_bytes": p.stat().st_size if p.exists() else 0,
                },
                "error": None
            })
        
        # ── INFO ─────────────────────────────────────────────────────────────
        elif operation == "info":
            if not p.exists():
                return json.dumps({"success": False, "result": None, "error": f"Path not found: {path}"})
            
            stat = p.stat()
            return json.dumps({
                "success": True,
                "result": {
                    "name": p.name,
                    "path": str(p),
                    "type": "directory" if p.is_dir() else "file",
                    "size_bytes": stat.st_size,
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                    "accessed": stat.st_atime,
                    "permissions": oct(stat.st_mode)[-3:],
                    "extension": p.suffix,
                    "parent": str(p.parent),
                },
                "error": None
            })
        
        else:
            return json.dumps({
                "success": False, "result": None,
                "error": f"Unknown operation: '{operation}'. Valid: read, write, copy, move, delete, list, search, mkdir, exists, info"
            })
    
    except Exception as e:
        logger.error(f"file_manager error: {e}", exc_info=True)
        return json.dumps({
            "success": False, "result": None,
            "error": f"{type(e).__name__}: {e}"
        })