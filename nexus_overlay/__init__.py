"""
NEXUS AI v4.0 — Ring Overlay Subsystem
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

A separate, decoupled overlay process for audio-reactive circular HUD.
Communicates with NEXUS core via local WebSocket.
Render layer: pywebview (native OS webview - WebView2 on Windows).
"""

from nexus_overlay.config import OverlayConfig, get_overlay_config
from nexus_overlay.websocket_server import OverlayWebSocketServer, get_ws_server
from nexus_overlay.audio_tap import AudioTapManager, get_audio_tap_manager
from nexus_overlay.overlay_process import OverlayProcess, run_overlay
from nexus_overlay.core_integration import OverlayIntegration, get_overlay_integration, OverlayState

__all__ = [
    "OverlayConfig",
    "get_overlay_config",
    "OverlayWebSocketServer", 
    "get_ws_server",
    "AudioTapManager",
    "get_audio_tap_manager",
    "OverlayProcess",
    "run_overlay",
    "OverlayIntegration",
    "get_overlay_integration",
    "OverlayState",
]