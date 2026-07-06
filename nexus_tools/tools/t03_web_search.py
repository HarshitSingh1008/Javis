"""
NEXUS AI v4.0 — Tool 03: Web search via DuckDuckGo + Brave Search.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides web search capability with DuckDuckGo as primary and Brave Search
as fallback. Results are returned as structured JSON with title, URL, snippet.
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
    source: str = "duckduckgo",
) -> str:
    """
    Search the web for information using DuckDuckGo or Brave Search.
    
    Use this tool when: The user asks to search the internet, find information,
    look up something online, or research a topic.
    
    Args:
        query: The search query string.
        max_results: Maximum number of search results to return (1-10).
        source: Search engine to use: "duckduckgo" (default) or "brave".
    
    Returns:
        JSON string with keys:
          - success (bool): Whether the search succeeded.
          - result (list): List of search result dicts with title, url, snippet.
          - error (str or null): Error message if failed.
          - source (str): Which search engine was used.
    
    Examples:
        >>> web_search("Python 3.12 new features")
        >>> web_search("latest AI news", max_results=3, source="brave")
    """
    start = time.perf_counter()
    max_results = min(max(max_results, 1), MAX_RESULTS)
    
    try:
        if source == "duckduckgo":
            return _search_duckduckgo(query, max_results)
        elif source == "brave":
            return _search_brave(query, max_results)
        else:
            return json.dumps({
                "success": False, "result": None,
                "error": f"Unknown search source: '{source}'. Use 'duckduckgo' or 'brave'."
            })
    except Exception as e:
        logger.error(f"web_search error: {e}", exc_info=True)
        return json.dumps({
            "success": False, "result": None,
            "error": f"{type(e).__name__}: {e}"
        })


def _search_duckduckgo(query: str, max_results: int) -> str:
    """Search using DuckDuckGo's HTML interface (no API key needed)."""
    import urllib.request
    import urllib.parse
    from html.parser import HTMLParser
    
    class DuckDuckGoParser(HTMLParser):
        """Minimal parser to extract search results from DuckDuckGo HTML."""
        def __init__(self):
            super().__init__()
            self.results: List[Dict[str, str]] = []
            self._in_result = False
            self._current = {"title": "", "url": "", "snippet": ""}
            self._tag_stack: List[str] = []
        
        def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
            self._tag_stack.append(tag)
            attrs_dict = dict(attrs)
            
            # Result articles have data-testid="result"
            if tag == "article" and attrs_dict.get("data-testid") == "result":
                self._in_result = True
                self._current = {"title": "", "url": "", "snippet": ""}
            
            # Extract URL from <a> tags with data-testid="result-title-a"
            if tag == "a" and attrs_dict.get("data-testid") == "result-title-a":
                href = attrs_dict.get("href", "")
                if href and not href.startswith("//"):
                    self._current["url"] = href
        
        def handle_data(self, data: str) -> None:
            if not self._in_result:
                return
            # Check parent tags to determine field
            parent = self._tag_stack[-2] if len(self._tag_stack) >= 2 else ""
            grandparent = self._tag_stack[-3] if len(self._tag_stack) >= 3 else ""
            
            if parent == "h2" and "result-title" in str(self._tag_stack):
                self._current["title"] += data.strip()
            elif parent == "span" and grandparent == "a":
                self._current["title"] += data.strip()
        
        def handle_endtag(self, tag: str) -> None:
            if self._tag_stack:
                self._tag_stack.pop()
            if tag == "article" and self._in_result:
                if self._current["title"] or self._current["url"]:
                    self.results.append(self._current)
                self._in_result = False
                self._current = {"title": "", "url": "", "snippet": ""}
    
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as response:
            html = response.read().decode("utf-8", errors="replace")
        
        # Simple regex-based extraction (more reliable than HTML parsing for DuckDuckGo)
        import re
        results = []
        
        # Find result blocks
        # DuckDuckGo HTML results are in <div class="result"> blocks
        result_blocks = re.findall(
            r'<div class="result[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
            html, re.DOTALL
        )
        
        if not result_blocks:
            # Try alternative pattern
            result_blocks = re.findall(
                r'<article[^>]*>(.*?)</article>',
                html, re.DOTALL
            )
        
        for block in result_blocks[:max_results]:
            title_match = re.search(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL)
            url_match = re.search(r'<a[^>]*href="(https?://[^"]+)"', block)
            snippet_match = re.search(r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL)
            
            if not title_match:
                title_match = re.search(r'<h2[^>]*>(.*?)</h2>', block, re.DOTALL)
            
            title = ""
            if title_match:
                title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
            
            url = url_match.group(1) if url_match else ""
            snippet = ""
            if snippet_match:
                snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
            
            if title or url:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })
        
        if not results:
            # Fallback: try to extract from any links
            all_links = re.findall(
                r'<a[^>]*href="(https?://[^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>',
                html, re.DOTALL
            )
            for url, title_html in all_links[:max_results]:
                title = re.sub(r'<[^>]+>', '', title_html).strip()
                results.append({"title": title, "url": url, "snippet": ""})
        
        return json.dumps({
            "success": True,
            "result": results[:max_results],
            "error": None,
            "source": "duckduckgo",
            "metadata": {"count": len(results[:max_results]), "query": query}
        })
        
    except urllib.error.HTTPError as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"DuckDuckGo HTTP error: {e.code}",
            "source": "duckduckgo"
        })
    except urllib.error.URLError as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"DuckDuckGo connection error: {e.reason}",
            "source": "duckduckgo"
        })
    except Exception as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"DuckDuckGo search failed: {e}",
            "source": "duckduckgo"
        })


def _search_brave(query: str, max_results: int) -> str:
    """Search using Brave Search API (requires API key in settings)."""
    from nexus_config.settings import get_settings
    
    settings = get_settings()
    brave_api_key = getattr(settings, "BRAVE_SEARCH_API_KEY", None) or ""
    
    if not brave_api_key:
        return json.dumps({
            "success": False, "result": None,
            "error": "Brave Search API key not configured. Set BRAVE_SEARCH_API_KEY in .env or use 'duckduckgo' source.",
            "source": "brave"
        })
    
    import urllib.request
    import urllib.parse
    
    try:
        url = f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote_plus(query)}&count={max_results}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": brave_api_key,
            }
        )
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        
        return json.dumps({
            "success": True,
            "result": results,
            "error": None,
            "source": "brave",
            "metadata": {"count": len(results), "query": query}
        })
        
    except urllib.error.HTTPError as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"Brave API HTTP error: {e.code}",
            "source": "brave"
        })
    except urllib.error.URLError as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"Brave API connection error: {e.reason}",
            "source": "brave"
        })
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"Brave API invalid response: {e}",
            "source": "brave"
        })
    except Exception as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"Brave search failed: {e}",
            "source": "brave"
        })