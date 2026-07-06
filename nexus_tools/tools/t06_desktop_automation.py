"""
NEXUS AI v4.0 — Tool 06: Desktop automation via PyAutoGUI.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Controls mouse and keyboard for desktop GUI automation: click, type,
screenshot, drag, scroll, and locate images on screen.
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from langchain_core.tools import tool

logger = logging.getLogger("nexus.tool.desktop_automation")

# ─── Constants ────────────────────────────────────────────────────────────────
UI_AUTOMATION_LOCK: Any = None  # Will be set to asyncio.Lock() at runtime


@tool
def desktop_automation(
    action: str,
    x: Optional[int] = None,
    y: Optional[int] = None,
    text: Optional[str] = None,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.1,
    duration: float = 0.2,
    image_path: Optional[str] = None,
    confidence: float = 0.8,
    scroll_amount: int = -1,
    region: Optional[List[int]] = None,
) -> str:
    """
    Control mouse and keyboard for desktop GUI automation.
    
    Use this tool when: The user needs to click on screen elements, type text,
    take screenshots, drag items, scroll, or automate any desktop application.
    
    Args:
        action: One of: "click", "double_click", "right_click", "type", "hotkey",
                "move", "drag", "scroll", "screenshot", "locate", "position",
                "keypress", "hold", "release"
        x: X coordinate for mouse actions.
        y: Y coordinate for mouse actions.
        text: Text to type (for "type" action) or key name (for "keypress").
        button: Mouse button: "left", "right", "middle".
        clicks: Number of clicks (for "click" action).
        interval: Seconds between actions.
        duration: Seconds for mouse movement animation.
        image_path: Path to image file for "locate" action.
        confidence: Confidence threshold for image recognition (0.0-1.0).
        scroll_amount: Number of scroll clicks (negative = down, positive = up).
        region: [x, y, width, height] for region-specific operations.
    
    Returns:
        JSON string with keys:
          - success (bool): Whether the operation succeeded.
          - result (any): Action-specific result data.
          - error (str or null): Error message if failed.
    
    Examples:
        >>> desktop_automation("click", x=500, y=300)
        >>> desktop_automation("type", text="Hello, world!")
        >>> desktop_automation("screenshot")
        >>> desktop_automation("hotkey", text="ctrl+c")
    """
    start = time.perf_counter()
    
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        result = None
        
        if action == "click":
            if x is not None and y is not None:
                pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval, duration=duration)
            else:
                pyautogui.click(button=button, clicks=clicks, interval=interval)
            result = f"Clicked {button} at ({x}, {y})" if x is not None else f"Clicked {button} at current position"
        
        elif action == "double_click":
            if x is not None and y is not None:
                pyautogui.doubleClick(x=x, y=y, interval=interval, duration=duration)
            else:
                pyautogui.doubleClick(interval=interval)
            result = f"Double-clicked at ({x}, {y})" if x is not None else "Double-clicked at current position"
        
        elif action == "right_click":
            if x is not None and y is not None:
                pyautogui.rightClick(x=x, y=y, duration=duration)
            else:
                pyautogui.rightClick()
            result = f"Right-clicked at ({x}, {y})" if x is not None else "Right-clicked at current position"
        
        elif action == "type":
            if not text:
                return json.dumps({"success": False, "result": None, "error": "Text required for type action"})
            pyautogui.write(text, interval=interval)
            result = f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}"
        
        elif action == "hotkey":
            if not text:
                return json.dumps({"success": False, "result": None, "error": "Key combination required for hotkey (e.g., 'ctrl+c')"})
            keys = text.lower().split("+")
            pyautogui.hotkey(*keys)
            result = f"Pressed: {text}"
        
        elif action == "move":
            if x is None or y is None:
                return json.dumps({"success": False, "result": None, "error": "x and y required for move action"})
            pyautogui.moveTo(x, y, duration=duration)
            result = f"Moved to ({x}, {y})"
        
        elif action == "drag":
            if x is None or y is None:
                return json.dumps({"success": False, "result": None, "error": "x and y required for drag action"})
            pyautogui.drag(x, y, duration=duration, button=button)
            result = f"Dragged by ({x}, {y})"
        
        elif action == "scroll":
            pyautogui.scroll(scroll_amount)
            result = f"Scrolled {scroll_amount} clicks"
        
        elif action == "screenshot":
            if region:
                screenshot = pyautogui.screenshot(region=tuple(region))
            else:
                screenshot = pyautogui.screenshot()
            import base64, io
            buf = io.BytesIO()
            screenshot.save(buf, format="PNG")
            result = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        elif action == "locate":
            if not image_path:
                return json.dumps({"success": False, "result": None, "error": "image_path required for locate action"})
            try:
                location = pyautogui.locateOnScreen(image_path, confidence=confidence)
                if location:
                    result = {
                        "x": location.left, "y": location.top,
                        "width": location.width, "height": location.height,
                        "center_x": location.left + location.width // 2,
                        "center_y": location.top + location.height // 2,
                    }
                else:
                    result = None
            except pyautogui.ImageNotFoundException:
                result = None
        
        elif action == "position":
            pos = pyautogui.position()
            result = {"x": pos.x, "y": pos.y}
        
        elif action == "keypress":
            if not text:
                return json.dumps({"success": False, "result": None, "error": "Key name required for keypress action"})
            pyautogui.press(text)
            result = f"Pressed key: {text}"
        
        elif action == "hold":
            if not text:
                return json.dumps({"success": False, "result": None, "error": "Key name required for hold action"})
            pyautogui.keyDown(text)
            result = f"Holding key: {text}"
        
        elif action == "release":
            if not text:
                return json.dumps({"success": False, "result": None, "error": "Key name required for release action"})
            pyautogui.keyUp(text)
            result = f"Released key: {text}"
        
        else:
            return json.dumps({
                "success": False, "result": None,
                "error": f"Unknown action: '{action}'. Valid: click, double_click, right_click, type, hotkey, move, drag, scroll, screenshot, locate, position, keypress, hold, release"
            })
        
        return json.dumps({"success": True, "result": result, "error": None})
    
    except ImportError:
        return json.dumps({
            "success": False, "result": None,
            "error": "PyAutoGUI not installed. Install with: pip install pyautogui"
        })
    except Exception as e:
        logger.error(f"desktop_automation error: {e}", exc_info=True)
        return json.dumps({
            "success": False, "result": None,
            "error": f"{type(e).__name__}: {e}"
        })