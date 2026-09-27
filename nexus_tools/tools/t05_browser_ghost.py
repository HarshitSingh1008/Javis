"""
NEXUS AI v4.0 — Tool 05: Playwright browser automation with Opera GX support.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides headless and headed browser automation via Playwright:
navigate, click, type, extract, screenshot, and execute JavaScript.
Supports Opera GX for user-visible automation.
"""

import json
import logging
import os
import time
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger("nexus.tool.browser_ghost")

# ─── Constants ────────────────────────────────────────────────────────────────
BROWSER_TIMEOUT: int = 30000  # milliseconds

# Opera GX path (Windows)
OPERA_GX_PATH = os.path.expanduser(r"~\AppData\Local\Programs\Opera GX\opera.exe")


@tool
def browser_ghost(
    action: str,
    url: Optional[str] = None,
    selector: Optional[str] = None,
    text: Optional[str] = None,
    javascript: Optional[str] = None,
    headless: bool = True,
    wait_seconds: float = 1.0,
    use_opera_gx: bool = False,
) -> str:
    """
    Control a browser to navigate, interact, and extract web content.
    
    Use this tool when: The user needs to interact with a web page that requires
    JavaScript execution, log in to a website, fill forms, or take screenshots.
    Set use_opera_gx=True to use Opera GX browser (headed mode only).
    
    Args:
        action: One of: "navigate", "click", "type", "extract", "screenshot",
                "execute_js", "get_html", "wait", "close"
        url: Target URL (required for "navigate" action).
        selector: CSS selector for element targeting.
        text: Text to type (for "type" action).
        javascript: JavaScript code to execute (for "execute_js" action).
        headless: Run browser in headless mode (no visible window).
        wait_seconds: Seconds to wait after action before returning.
        use_opera_gx: If True, launches Opera GX browser (requires headed mode).
    
    Returns:
        JSON string with keys:
          - success (bool): Whether the operation succeeded.
          - result (any): Action-specific result data.
          - error (str or null): Error message if failed.
    
    Examples:
        >>> browser_ghost("navigate", url="https://example.com")
        >>> browser_ghost("extract", selector="article.main-content")
        >>> browser_ghost("screenshot")
    """
    start = time.perf_counter()
    
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
        
        with sync_playwright() as p:
            browser_type = p.chromium
            
            # Determine browser executable
            executable_path = None
            if use_opera_gx and not headless:
                if os.path.exists(OPERA_GX_PATH):
                    executable_path = OPERA_GX_PATH
                    logger.info(f"Using Opera GX at {OPERA_GX_PATH}")
                else:
                    logger.warning(f"Opera GX not found at {OPERA_GX_PATH}, falling back to Chromium")
            
            browser = browser_type.launch(
                headless=headless,
                executable_path=executable_path,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ]
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()
            
            result = None
            
            try:
                if action == "navigate":
                    if not url:
                        return json.dumps({"success": False, "result": None, "error": "URL required for navigate action"})
                    page.goto(url, wait_until="networkidle", timeout=BROWSER_TIMEOUT)
                    if wait_seconds:
                        page.wait_for_timeout(int(wait_seconds * 1000))
                    result = {
                        "title": page.title(),
                        "url": page.url,
                    }
                
                elif action == "click":
                    if not selector:
                        return json.dumps({"success": False, "result": None, "error": "Selector required for click action"})
                    page.click(selector, timeout=BROWSER_TIMEOUT)
                    if wait_seconds:
                        page.wait_for_timeout(int(wait_seconds * 1000))
                    result = {"clicked": selector}
                
                elif action == "type":
                    if not selector or text is None:
                        return json.dumps({"success": False, "result": None, "error": "Both selector and text required for type action"})
                    page.fill(selector, text, timeout=BROWSER_TIMEOUT)
                    if wait_seconds:
                        page.wait_for_timeout(int(wait_seconds * 1000))
                    result = {"typed": text, "into": selector}
                
                elif action == "extract":
                    if selector:
                        elements = page.query_selector_all(selector)
                        result = [el.inner_text() for el in elements[:50]]
                    else:
                        result = page.inner_text("body")
                    if not result:
                        result = "No content found"
                
                elif action == "screenshot":
                    screenshot_bytes = page.screenshot(full_page=True)
                    import base64
                    result = base64.b64encode(screenshot_bytes).decode("utf-8")
                
                elif action == "execute_js":
                    if not javascript:
                        return json.dumps({"success": False, "result": None, "error": "JavaScript code required for execute_js action"})
                    result = page.evaluate(javascript)
                
                elif action == "get_html":
                    if selector:
                        el = page.query_selector(selector)
                        result = el.inner_html() if el else "Selector not found"
                    else:
                        result = page.content()
                
                elif action == "wait":
                    if wait_seconds:
                        page.wait_for_timeout(int(wait_seconds * 1000))
                    result = f"Waited {wait_seconds}s"
                
                elif action == "close":
                    result = "Browser closed"
                
                else:
                    return json.dumps({
                        "success": False, "result": None,
                        "error": f"Unknown action: '{action}'. Valid: navigate, click, type, extract, screenshot, execute_js, get_html, wait, close"
                    })
            
            finally:
                context.close()
                browser.close()
            
            return json.dumps({
                "success": True,
                "result": result,
                "error": None,
            })
    
    except ImportError:
        return json.dumps({
            "success": False, "result": None,
            "error": "Playwright not installed. Install with: pip install playwright && playwright install chromium"
        })
    except PwTimeout as e:
        return json.dumps({
            "success": False, "result": None,
            "error": f"Browser timeout: {e}"
        })
    except Exception as e:
        logger.error(f"browser_ghost error: {e}", exc_info=True)
        return json.dumps({
            "success": False, "result": None,
            "error": f"{type(e).__name__}: {e}"
        })