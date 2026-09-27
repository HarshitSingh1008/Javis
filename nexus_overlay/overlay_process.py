"""
NEXUS AI v4.0 — Ring Overlay Process (pywebview)
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Standalone overlay process that runs the ring visualization.
Communicates with NEXUS core via WebSocket.
Render layer: pywebview (native OS webview - WebView2 on Windows).

Run standalone for visual testing:
    python -m nexus_overlay.overlay_process

Run with custom position:
    python -m nexus_overlay.overlay_process --x 100 --y 100
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus_overlay.config import get_overlay_config, OverlayConfig

logger = logging.getLogger("nexus.overlay.process")


class OverlayProcess:
    """
    Ring overlay process using pywebview.
    
    Creates a frameless, transparent, always-on-top window
    with the canvas-based ring visualization.
    """
    
    def __init__(self, config: Optional[OverlayConfig] = None):
        self._config = config or get_overlay_config()
        self._window = None
        self._running = False
        self._webview = None
        
        # Try to import pywebview
        try:
            import webview
            self._webview = webview
            self._webview_available = True
        except ImportError:
            self._webview_available = False
            logger.error("pywebview not installed. Install: pip install pywebview")
    
    def run(self, x: Optional[int] = None, y: Optional[int] = None) -> int:
        """
        Run the overlay process.
        
        Args:
            x: Window X position (None = auto bottom-right)
            y: Window Y position (None = auto bottom-right)
            
        Returns:
            Exit code (0 = success, 1 = error)
        """
        if not self._webview_available:
            logger.error("pywebview not available")
            return 1
        
        self._running = True
        
        # Determine window position
        window_x, window_y = self._calculate_position(x, y)
        
        # Get HTML file path
        html_path = Path(__file__).parent / "assets" / "overlay.html"
        if not html_path.exists():
            logger.error("Overlay HTML not found: %s", html_path)
            return 1
        
        # Create window
        try:
            self._window = self._webview.create_window(
                title="NEXUS Ring",
                url=str(html_path),
                width=self._config.window_width,
                height=self._config.window_height,
                x=window_x,
                y=window_y,
                frameless=self._config.frameless,
                transparent=self._config.transparent,
                on_top=self._config.always_on_top,
                resizable=self._config.resizable,
                min_size=(360, 520),
                background_color="#000000",
                text_select=False,
                zoomable=False,
                draggable=True,
                easy_drag=True,
            )
            
            # Set up window events
            self._window.events.closed += self._on_closed
            self._window.events.moved += self._on_moved
            
            logger.info("Starting overlay window at (%d, %d)", window_x, window_y)
            
            # Start the webview event loop (blocks)
            self._webview.start(
                func=self._on_started,
                debug=False,
                storage_path=str(Path.home() / ".nexus_ai" / "overlay_storage"),
            )
            
        except Exception as e:
            logger.error("Failed to start overlay: %s", e)
            return 1
        
        return 0
    
    def _calculate_position(self, x: Optional[int], y: Optional[int]) -> tuple:
        """Calculate window position (default: bottom-right corner)."""
        if x is not None and y is not None:
            return x, y
        
        # Try to get screen size and position at bottom-right
        try:
            import screeninfo
            monitors = screeninfo.get_monitors()
            if monitors:
                # Use primary monitor
                monitor = monitors[0]
                screen_w = monitor.width
                screen_h = monitor.height
                x = screen_w - self._config.window_width - 20
                y = screen_h - self._config.window_height - 60  # Account for taskbar
                return max(0, x), max(0, y)
        except Exception:
            pass
        
        # Fallback: use Windows API via ctypes
        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)  # SM_CXSCREEN
            screen_h = user32.GetSystemMetrics(1)  # SM_CYSCREEN
            px = screen_w - self._config.window_width - 20
            py = screen_h - self._config.window_height - 60
            return max(0, px), max(0, py)
        except Exception:
            pass

        # Final fallback
        return 100, 100
    
    def _on_started(self) -> None:
        """Called when webview starts."""
        logger.debug("Overlay window started")
        
        # Inject config into the page after load
        if self._window:
            self._inject_config()
    
    def _inject_config(self) -> None:
        """Inject config into the loaded page."""
        try:
            config_js = f"""
            (function() {{
                if (window.CONFIG) {{
                    window.CONFIG.ringCount = {self._config.ring_count};
                    window.CONFIG.barCount = {self._config.bar_count};
                    window.CONFIG.ringBaseRadiusRatio = {self._config.ring_base_radius_ratio};
                    window.CONFIG.ringMultipliers = {list(self._config.ring_multipliers)};
                    window.CONFIG.ringLineWidths = {list(self._config.ring_line_widths)};
                    window.CONFIG.ringDashPatterns = {list(self._config.ring_dash_patterns)};
                    window.CONFIG.ringRotationSpeeds = {list(self._config.ring_rotation_speeds)};
                    window.CONFIG.ringShadowBlur = {self._config.ring_shadow_blur};
                    window.CONFIG.ringGlobalAlphas = {list(self._config.ring_global_alphas)};
                    window.CONFIG.barBaseRadiusRatio = {self._config.bar_base_radius_ratio};
                    window.CONFIG.barMinLength = {self._config.bar_min_length};
                    window.CONFIG.barMaxLength = {self._config.bar_max_length};
                    window.CONFIG.barLineWidth = {self._config.bar_line_width};
                    window.CONFIG.barShadowBlur = {self._config.bar_shadow_blur};
                    window.CONFIG.coreRadiusRatio = {self._config.core_radius_ratio};
                    window.CONFIG.coreShadowBlur = {self._config.core_shadow_blur};
                    window.CONFIG.particleBurstCount = {self._config.particle_burst_count};
                    window.CONFIG.particleBaseSpeed = {self._config.particle_base_speed};
                    window.CONFIG.particleSpeedVariance = {self._config.particle_speed_variance};
                    window.CONFIG.particleBaseLife = {self._config.particle_base_life};
                    window.CONFIG.particleDecayMin = {self._config.particle_decay_min};
                    window.CONFIG.particleDecayMax = {self._config.particle_decay_max};
                    window.CONFIG.particleSize = {self._config.particle_size};
                    window.CONFIG.particleShadowBlur = {self._config.particle_shadow_blur};
                    window.CONFIG.idleBreatheSpeed = {self._config.idle_breathe_speed};
                    window.CONFIG.idleBreatheAmplitude = {self._config.idle_breathe_amplitude};
                    window.CONFIG.activeBreatheFactor = {self._config.active_breathe_factor};
                    window.CONFIG.sweepSpeedFactor = {self._config.sweep_speed_factor};
                    window.CONFIG.sweepArcAngle = {self._config.sweep_arc_angle};
                    window.CONFIG.errorJitterProbability = {self._config.error_jitter_probability};
                    window.CONFIG.errorJitterAlpha = {self._config.error_jitter_alpha};
                    window.STATES = {self._config.states};
                    console.log('[Overlay] Config injected from core');
                }}
            }})();
            """
            self._window.evaluate_js(config_js)
        except Exception as e:
            logger.debug("Config injection failed (page may not be ready): %s", e)
    
    def _on_closed(self) -> None:
        """Window closed event."""
        logger.info("Overlay window closed")
        self._running = False
    
    def _on_moved(self, x: int, y: int) -> None:
        """Window moved - persist position."""
        self._config.window_x = x
        self._config.window_y = y
        self._config.save()
        logger.debug("Overlay position saved: (%d, %d)", x, y)
    
    def stop(self) -> None:
        """Stop the overlay."""
        self._running = False
        if self._window:
            try:
                self._window.destroy()
            except Exception:
                pass


def run_overlay(x: Optional[int] = None, y: Optional[int] = None) -> int:
    """
    Run the overlay process.
    
    Args:
        x: Window X position
        y: Window Y position
        
    Returns:
        Exit code
    """
    overlay = OverlayProcess()
    return overlay.run(x, y)


def main():
    """Entry point for `python -m nexus_overlay.overlay_process`."""
    parser = argparse.ArgumentParser(description="NEXUS Ring Overlay")
    parser.add_argument("--x", type=int, help="Window X position")
    parser.add_argument("--y", type=int, help="Window Y position")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Run overlay
    exit_code = run_overlay(args.x, args.y)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()