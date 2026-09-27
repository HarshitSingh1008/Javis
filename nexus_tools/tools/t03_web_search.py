"""
NEXUS AI v4.0 — Tool 03: Web search via DDGS (DuckDuckGo).
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides web search capability using the ddgs library (same as original JARVIS).
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger("nexus.tool.web_search")

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_RESULTS: int = 10
SEARCH_TIMEOUT: int = 15


@tool
def web_search(
    query: str,
    max_results: int = 5,
    search_type: str = "text",
) -> str:
    """
    Search the web for information using DDGS (DuckDuckGo).
    
    Use this tool when: The user asks to search the internet, find information,
    look up something online, or research a topic.
    
    Args:
        query: The search query string.
        max_results: Maximum number of search results to return (1-10).
        search_type: "text" for web search, "image" for image URLs.
    
    Returns:
        JSON string with keys:
          - success (bool): Whether the search succeeded.
          - result (list): List of search result dicts with title, url, snippet.
          - error (str or null): Error message if failed.
    
    Examples:
        >>> web_search("Python 3.12 new features")
        >>> web_search("latest AI news", max_results=3)
        >>> web_search("cat pictures", search_type="image")
    """
    start = time.perf_counter()
    max_results = min(max(max_results, 1), MAX_RESULTS)
    
    try:
        from ddgs import DDGS
        
        with DDGS() as ddgs:
            if search_type == "image":
                results = list(ddgs.images(query, max_results=max_results))
                if results:
                    return json.dumps({
                        "success": True,
                        "result": [{"image": r.get("image", ""), "title": r.get("title", ""), "url": r.get("url", "")} for r in results],
                        "error": None,
                        "source": "duckduckgo",
                    })
                return json.dumps({"success": False, "result": None, "error": "No images found.", "source": "duckduckgo"})
            else:
                results = list(ddgs.text(query, max_results=max_results))
                if results:
                    return json.dumps({
                        "success": True,
                        "result": [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results],
                        "error": None,
                        "source": "duckduckgo",
                    })
                return json.dumps({"success": False, "result": None, "error": "No search results found.", "source": "duckduckgo"})
    
    except ImportError:
        return json.dumps({
            "success": False, "result": None,
            "error": "ddgs not installed. Install with: pip install ddgs",
            "source": "duckduckgo"
        })
    except Exception as e:
        logger.error(f"web_search error: {e}", exc_info=True)
        return json.dumps({
            "success": False, "result": None,
            "error": f"{type(e).__name__}: {e}",
            "source": "duckduckgo"
        })