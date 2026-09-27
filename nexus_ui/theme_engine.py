"""
NEXUS AI v4.0 — Color palette, font loading, DPI scaling, theme switching.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides theme management for the CustomTkinter HUD with JARVIS-inspired themes:
- jarvis (default): Iron Man JARVIS - deep blue/black with gold/cyan accents
- jarvis_classic: Original JARVIS - dark blue with warm gold accents
- jarvis_prime: Enhanced JARVIS - true black with electric blue/cyan
- midnight_blue: Deep blue with gold accents
- void_black: True black with purple accents

All themes are OLED-friendly with true black backgrounds.
"""

import logging
import platform
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
from functools import lru_cache
from pathlib import Path

from nexus_config.settings import get_settings, APP_ROOT

logger = logging.getLogger("nexus.ui.theme")


@dataclass(frozen=True)
class ThemeColors:
    """
    Immutable color palette for a NEXUS AI theme.
    
    All colors are hex strings (e.g., "#1a1a2e").
    """
    # Background colors
    bg_primary: str = "#000000"        # Main background - true black
    bg_secondary: str = "#0a0a12"      # Panel background - deep blue-black
    bg_tertiary: str = "#10101a"       # Input/button background
    bg_hover: str = "#1a1a2e"          # Hover state
    bg_glass: str = "#0d0d1a"          # Glass panel background
    bg_glass_border: str = "#1a1a3a"   # Glass panel border
    
    # Text colors
    text_primary: str = "#e8f0ff"       # Primary text - cool white
    text_secondary: str = "#8899bb"    # Secondary/muted text
    text_accent: str = "#00d4ff"       # Accent text - electric cyan
    text_gold: str = "#ffd700"         # Gold accent text
    text_dim: str = "#445577"          # Dim text
    
    # Accent colors - JARVIS palette
    accent_primary: str = "#00d4ff"    # Primary accent - electric cyan
    accent_secondary: str = "#0099cc"  # Secondary accent - deep cyan
    accent_gold: str = "#ffd700"       # Gold - JARVIS signature
    accent_gold_dim: str = "#ccaa00"   # Dim gold
    accent_success: str = "#00ff88"    # Success - bright green
    accent_warning: str = "#ffaa00"    # Warning - amber
    accent_error: str = "#ff3344"      # Error - red
    accent_info: str = "#4488ff"       # Info - blue
    
    # Border colors
    border_primary: str = "#1a2a4a"    # Default border - subtle blue
    border_focus: str = "#00d4ff"      # Focus border - cyan
    border_gold: str = "#ffd700"       # Gold border
    border_error: str = "#ff3344"      # Error border
    border_glass: str = "#2a3a5a"      # Glass border
    
    # Animation colors
    particle_color: str = "#00d4ff"    # Particle system - cyan
    particle_gold: str = "#ffd700"     # Gold particles
    arc_reactor_color: str = "#00d4ff" # Arc reactor - cyan
    arc_reactor_gold: str = "#ffd700"  # Arc reactor - gold
    waveform_color: str = "#00d4ff"    # Waveform - cyan
    waveform_gold: str = "#ffd700"     # Waveform - gold
    scanline_color: str = "#003344"    # Scanline overlay
    glow_cyan: str = "#00d4ff"         # Cyan glow
    glow_gold: str = "#ffd700"         # Gold glow
    
    # DAG visualization colors
    dag_pending: str = "#2a3a5a"       # Pending node
    dag_running: str = "#00d4ff"       # Running node - cyan
    dag_running_gold: str = "#ffd700"  # Running node - gold
    dag_success: str = "#00ff88"       # Completed node
    dag_failed: str = "#ff3344"        # Failed node
    dag_glow: str = "#00d4ff"          # Node glow
    
    # Font configuration
    font_family: str = "Segoe UI"      # Primary font
    font_mono: str = "Consolas"        # Monospace font
    font_size_small: int = 11
    font_size_normal: int = 13
    font_size_large: int = 16
    font_size_xl: int = 20
    font_size_title: int = 28
    font_size_hero: int = 48


# ─── Theme Definitions ──────────────────────────────────────────────────────

JARVIS = ThemeColors(
    bg_primary="#000000",
    bg_secondary="#050a14",
    bg_tertiary="#0a1428",
    bg_hover="#102040",
    bg_glass="#081020",
    bg_glass_border="#102040",
    text_primary="#e8f0ff",
    text_secondary="#88aacc",
    text_accent="#00d4ff",
    text_gold="#ffd700",
    text_dim="#446688",
    accent_primary="#00d4ff",
    accent_secondary="#0099cc",
    accent_gold="#ffd700",
    accent_gold_dim="#ccaa00",
    accent_success="#00ff88",
    accent_warning="#ffaa00",
    accent_error="#ff3344",
    accent_info="#4488ff",
    border_primary="#102040",
    border_focus="#00d4ff",
    border_gold="#ffd700",
    border_error="#ff3344",
    border_glass="#1a3050",
    particle_color="#00d4ff",
    particle_gold="#ffd700",
    arc_reactor_color="#00d4ff",
    arc_reactor_gold="#ffd700",
    waveform_color="#00d4ff",
    waveform_gold="#ffd700",
    scanline_color="#001a33",
    glow_cyan="#00d4ff",
    glow_gold="#ffd700",
    dag_pending="#1a2a4a",
    dag_running="#00d4ff",
    dag_running_gold="#ffd700",
    dag_success="#00ff88",
    dag_failed="#ff3344",
    dag_glow="#00d4ff",
    font_family="Segoe UI",
    font_mono="Consolas",
    font_size_small=11,
    font_size_normal=13,
    font_size_large=16,
    font_size_xl=20,
    font_size_title=28,
    font_size_hero=48,
)

JARVIS_CLASSIC = ThemeColors(
    bg_primary="#000814",
    bg_secondary="#001830",
    bg_tertiary="#002850",
    bg_hover="#003870",
    bg_glass="#001c38",
    bg_glass_border="#003060",
    text_primary="#ffeedd",
    text_secondary="#ccaa88",
    text_accent="#ffd700",
    text_gold="#ffd700",
    text_dim="#665533",
    accent_primary="#ffd700",
    accent_secondary="#ccaa00",
    accent_gold="#ffd700",
    accent_gold_dim="#ccaa00",
    accent_success="#88ff88",
    accent_warning="#ffaa00",
    accent_error="#ff4444",
    accent_info="#4488ff",
    border_primary="#003060",
    border_focus="#ffd700",
    border_gold="#ffd700",
    border_error="#ff4444",
    border_glass="#004080",
    particle_color="#ffd700",
    particle_gold="#ffd700",
    arc_reactor_color="#ffd700",
    arc_reactor_gold="#ffd700",
    waveform_color="#ffd700",
    waveform_gold="#ffd700",
    scanline_color="#001a33",
    glow_cyan="#4488ff",
    glow_gold="#ffd700",
    dag_pending="#002040",
    dag_running="#ffd700",
    dag_running_gold="#ffd700",
    dag_success="#88ff88",
    dag_failed="#ff4444",
    dag_glow="#ffd700",
    font_family="Segoe UI",
    font_mono="Consolas",
    font_size_small=11,
    font_size_normal=13,
    font_size_large=16,
    font_size_xl=20,
    font_size_title=28,
    font_size_hero=48,
)

JARVIS_PRIME = ThemeColors(
    bg_primary="#000000",
    bg_secondary="#000510",
    bg_tertiary="#001020",
    bg_hover="#002040",
    bg_glass="#000818",
    bg_glass_border="#001830",
    text_primary="#ffffff",
    text_secondary="#88ccff",
    text_accent="#00ffff",
    text_gold="#ffd700",
    text_dim="#4488aa",
    accent_primary="#00ffff",
    accent_secondary="#00aacc",
    accent_gold="#ffd700",
    accent_gold_dim="#ccaa00",
    accent_success="#00ff88",
    accent_warning="#ffaa00",
    accent_error="#ff3344",
    accent_info="#4488ff",
    border_primary="#001830",
    border_focus="#00ffff",
    border_gold="#ffd700",
    border_error="#ff3344",
    border_glass="#003060",
    particle_color="#00ffff",
    particle_gold="#ffd700",
    arc_reactor_color="#00ffff",
    arc_reactor_gold="#ffd700",
    waveform_color="#00ffff",
    waveform_gold="#ffd700",
    scanline_color="#001122",
    glow_cyan="#00ffff",
    glow_gold="#ffd700",
    dag_pending="#001020",
    dag_running="#00ffff",
    dag_running_gold="#ffd700",
    dag_success="#00ff88",
    dag_failed="#ff3344",
    dag_glow="#00ffff",
    font_family="Segoe UI",
    font_mono="Consolas",
    font_size_small=11,
    font_size_normal=13,
    font_size_large=16,
    font_size_xl=20,
    font_size_title=28,
    font_size_hero=48,
)

MIDNIGHT_BLUE = ThemeColors(
    bg_primary="#0a0a1a",
    bg_secondary="#12122a",
    bg_tertiary="#1a1a3a",
    bg_hover="#2a2a4a",
    bg_glass="#12122a",
    bg_glass_border="#2a2a4a",
    text_primary="#e0e0ff",
    text_secondary="#8888aa",
    text_accent="#ffd700",
    text_gold="#ffd700",
    text_dim="#666688",
    accent_primary="#ffd700",
    accent_secondary="#ccaa00",
    accent_gold="#ffd700",
    accent_gold_dim="#ccaa00",
    accent_success="#00ff88",
    accent_warning="#ff8800",
    accent_error="#ff4444",
    accent_info="#4488ff",
    border_primary="#2a2a4a",
    border_focus="#ffd700",
    border_gold="#ffd700",
    border_error="#ff4444",
    border_glass="#3a3a5a",
    particle_color="#ffd700",
    particle_gold="#ffd700",
    arc_reactor_color="#ffd700",
    arc_reactor_gold="#ffd700",
    waveform_color="#ffd700",
    waveform_gold="#ffd700",
    scanline_color="#001a33",
    glow_cyan="#4488ff",
    glow_gold="#ffd700",
    dag_pending="#444466",
    dag_running="#ffd700",
    dag_running_gold="#ffd700",
    dag_success="#00ff88",
    dag_failed="#ff4444",
    dag_glow="#ffd700",
    font_family="Segoe UI",
    font_mono="Consolas",
    font_size_small=11,
    font_size_normal=13,
    font_size_large=16,
    font_size_xl=20,
    font_size_title=28,
    font_size_hero=48,
)

VOID_BLACK = ThemeColors(
    bg_primary="#000000",
    bg_secondary="#0a0a0a",
    bg_tertiary="#1a1a1a",
    bg_hover="#2a2a2a",
    bg_glass="#0a0a0a",
    bg_glass_border="#1a1a1a",
    text_primary="#e0e0ff",
    text_secondary="#8888aa",
    text_accent="#bb88ff",
    text_gold="#ffd700",
    text_dim="#666666",
    accent_primary="#bb88ff",
    accent_secondary="#8844cc",
    accent_gold="#ffd700",
    accent_gold_dim="#ccaa00",
    accent_success="#00ff88",
    accent_warning="#ffaa00",
    accent_error="#ff4444",
    accent_info="#4488ff",
    border_primary="#1a1a1a",
    border_focus="#bb88ff",
    border_gold="#ffd700",
    border_error="#ff4444",
    border_glass="#2a2a2a",
    particle_color="#bb88ff",
    particle_gold="#ffd700",
    arc_reactor_color="#bb88ff",
    arc_reactor_gold="#ffd700",
    waveform_color="#bb88ff",
    waveform_gold="#ffd700",
    scanline_color="#0a0a0a",
    glow_cyan="#8844cc",
    glow_gold="#ffd700",
    dag_pending="#333333",
    dag_running="#bb88ff",
    dag_running_gold="#ffd700",
    dag_success="#00ff88",
    dag_failed="#ff4444",
    dag_glow="#bb88ff",
    font_family="Segoe UI",
    font_mono="Consolas",
    font_size_small=11,
    font_size_normal=13,
    font_size_large=16,
    font_size_xl=20,
    font_size_title=28,
    font_size_hero=48,
)

THEMES: Dict[str, ThemeColors] = {
    "jarvis": JARVIS,
    "jarvis_classic": JARVIS_CLASSIC,
    "jarvis_prime": JARVIS_PRIME,
    "midnight_blue": MIDNIGHT_BLUE,
    "void_black": VOID_BLACK,
}


class ThemeEngine:
    """
    Manages theme selection, font loading, and DPI scaling.
    
    Provides:
    - Theme switching at runtime
    - Font family resolution per platform
    - DPI-aware scaling
    - Color palette access
    """
    
    def __init__(self):
        self._settings = get_settings()
        self._current_theme_name: str = self._settings.UI_THEME
        self._current_colors: ThemeColors = THEMES.get(
            self._current_theme_name, JARVIS
        )
        self._dpi_scale: float = self._detect_dpi_scale()
        self._font_family: str = self._resolve_font_family()
        self._font_mono: str = self._resolve_mono_font()
    
    @property
    def colors(self) -> ThemeColors:
        """Get current theme colors."""
        return self._current_colors
    
    @property
    def theme_name(self) -> str:
        """Get current theme name."""
        return self._current_theme_name
    
    @property
    def dpi_scale(self) -> float:
        """Get DPI scaling factor (system DPI only, not user font scale)."""
        return self._dpi_scale
    
    def set_theme(self, theme_name: str) -> None:
        """
        Switch to a different theme at runtime.
        
        Args:
            theme_name: One of "dark_platinum", "midnight_blue", "void_black".
        
        Raises:
            ValueError: If theme_name is not recognized.
        """
        if theme_name not in THEMES:
            raise ValueError(f"Unknown theme: {theme_name}. Options: {list(THEMES.keys())}")
        self._current_theme_name = theme_name
        self._current_colors = THEMES[theme_name]
        logger.info(f"Theme switched to: {theme_name}")
    
    def get_font(self, size: str = "normal", bold: bool = False) -> Tuple[str, int, str]:
        """
        Get a font tuple for CustomTkinter.
        
        Args:
            size: One of "small", "normal", "large", "xl", "title", "hero".
            bold: Whether to use bold weight.
        
        Returns:
            Tuple of (font_family, font_size, weight).
        """
        sizes = {
            "small": self._current_colors.font_size_small,
            "normal": self._current_colors.font_size_normal,
            "large": self._current_colors.font_size_large,
            "xl": self._current_colors.font_size_xl,
            "title": self._current_colors.font_size_title,
            "hero": self._current_colors.font_size_hero,
        }
        # Use base size with user font scale only (CustomTkinter handles DPI scaling internally)
        base_size = sizes.get(size, self._current_colors.font_size_normal)
        font_size = int(base_size * self._settings.UI_FONT_SCALE)
        weight = "bold" if bold else "normal"
        return (self._font_family, font_size, weight)
    
    def get_mono_font(self, size: str = "normal") -> Tuple[str, int]:
        """
        Get a monospace font tuple.
        
        Args:
            size: One of "small", "normal", "large".
        
        Returns:
            Tuple of (font_family, font_size).
        """
        sizes = {
            "small": self._current_colors.font_size_small,
            "normal": self._current_colors.font_size_normal,
            "large": self._current_colors.font_size_large,
        }
        base_size = sizes.get(size, self._current_colors.font_size_normal)
        font_size = int(base_size * self._settings.UI_FONT_SCALE)
        return (self._font_mono, font_size)
    
    def _detect_dpi_scale(self) -> float:
        """Detect system DPI scaling factor."""
        try:
            import tkinter as tk
            root = tk.Tk()
            dpi = root.winfo_fpixels("1i")
            root.destroy()
            return dpi / 96.0  # 96 DPI is standard
        except Exception:
            return 1.0
    
    def _resolve_font_family(self) -> str:
        """Resolve the best available font family for the platform."""
        system = platform.system()
        if system == "Windows":
            return "Segoe UI"
        elif system == "Darwin":
            return "SF Pro Display"
        else:
            return "Ubuntu"
    
    def _resolve_mono_font(self) -> str:
        """Resolve the best available monospace font for the platform."""
        system = platform.system()
        if system == "Windows":
            return "Consolas"
        elif system == "Darwin":
            return "SF Mono"
        else:
            return "Ubuntu Mono"


@lru_cache(maxsize=1)
def get_theme_engine() -> ThemeEngine:
    """
    Return the singleton ThemeEngine instance.
    
    Returns:
        ThemeEngine: The singleton theme engine.
    """
    return ThemeEngine()