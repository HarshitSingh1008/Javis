"""
NEXUS AI v4.0 — Tool 24: HTTP interactions (read text, download files).
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides simple HTTP interactions: read text from URL, download files.
Matches the original JARVIS web_manager tool.
"""

import json
import logging
import os
import time
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger("nexus.tool.web_manager")


@tool
def web_manager(
    action: str,
    url: str,
    save_path: str = "",
) -> str:
    """
    Simple HTTP interactions: 'read_text' or 'download_file'.
    
    Use this tool when: The user wants to read text from a web page or download a file from a URL.
    
    Args:
        action: One of: "read_text" or "download_file".
        url: The URL to fetch.
        save_path: Local path to save the file (required for download_file).
    
    Returns:
        JSON string with keys:
          - success (bool): Whether the operation succeeded.
          - result (str): Extracted text or save confirmation.
          - error (str or null): Error message if failed.
    
    Examples:
        >>> web_manager("read_text", "https://example.com")
        >>> web_manager("download_file", "https://example.com/file.pdf", "C:/Downloads/file.pdf")
    """
    start = time.perf_counter()
    
    import requests
    from bs4 import BeautifulSoup
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        if action == "read_text":
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator='\n', strip=True)
            return json.dumps({
                "success": True,
                "result": text[:2000],
                "error": None,
            })
        
        elif action == "download_file":
            if not save_path:
                return json.dumps({"success": False, "result": None, "error": "save_path required for download_file"})
            
            response = requests.get(url, headers=headers, stream=True, timeout=20)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return json.dumps({
                "success": True,
                "result": f"File saved to {save_path}",
                "error": None,
            })
        
        else:
            return json.dumps({"success": False, "result": None, "error": f"Invalid action: {action}. Use 'read_text' or 'download_file'."})
    
    except Exception as e:
        logger.error(f"web_manager error: {e}", exc_info=True)
        return json.dumps({
            "success": False, "result": None,
            "error": f"{type(e).__name__}: {e}"
        })