"""
NEXUS AI v4.0 — Ring Overlay Configuration
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Centralized configuration for the ring overlay process.
All visual parameters, WebSocket settings, and window behavior in one place.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from functools import lru_cache

from nexus_config.settings import get_settings, APP_ROOT

logger = logging.getLogger("nexus.overlay.config")


@dataclass
class OverlayConfig:
    """
    Configuration for the ring overlay.
    
    All visual parameters, WebSocket settings, and window behavior.
    """
    # WebSocket
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765
    ws_path: str = "/nexus/overlay"
    
    # Window
    window_width: int = 360
    window_height: int = 520
    window_x: Optional[int] = None  # None = auto (bottom-right)
    window_y: Optional[int] = None
    always_on_top: bool = True
    frameless: bool = True
    transparent: bool = True
    resizable: bool = False
    
    # Visual - Rings
    ring_count: int = 3
    ring_base_radius_ratio: float = 0.22
    ring_multipliers: tuple = (1.0, 0.78, 0.58)
    ring_line_widths: tuple = (2.5, 1.5, 1.5)
    ring_dash_patterns: tuple = ([14, 6], [2, 10], [14, 6])
    ring_rotation_speeds: tuple = (1.0, -1.3, 1.6)
    ring_shadow_blur: int = 14
    ring_global_alphas: tuple = (0.85, 0.7, 0.55)
    
    # Visual - Radial Bars
    bar_count: int = 40
    bar_base_radius_ratio: float = 1.15
    bar_min_length: int = 10
    bar_max_length: int = 44
    bar_line_width: int = 2
    bar_shadow_blur: int = 8
    
    # Visual - Core
    core_radius_ratio: float = 0.32
    core_shadow_blur: int = 20
    
    # Visual - Particles
    particle_burst_count: int = 18
    particle_base_speed: float = 0.6
    particle_speed_variance: float = 1.2
    particle_base_life: float = 1.0
    particle_decay_min: float = 0.01
    particle_decay_max: float = 0.025
    particle_size: float = 2.2
    particle_shadow_blur: int = 10
    
    # Visual - Breathing/Idle
    idle_breathe_speed: float = 1100  # ms per cycle
    idle_breathe_amplitude: float = 0.03
    active_breathe_factor: float = 0.08
    
    # Visual - Radar Sweep (thinking state)
    sweep_speed_factor: float = 0.03
    sweep_arc_angle: float = 0.5  # radians
    
    # Visual - Error Jitter
    error_jitter_probability: float = 0.15
    error_jitter_alpha: float = 0.25
    
    # Visual - State debounce
    state_debounce_ms: int = 150
    
    # State colors (matching provided implementation)
    states: dict = field(default_factory=lambda: {
        "idle": {
            "color": "#3fd3ff",
            "accent": "#8be9ff",
            "speed": 1.0,
        },
        "listening": {
            "color": "#3fd3ff", 
            "accent": "#c9f6ff",
            "speed": 1.6,
        },
        "thinking": {
            "color": "#ffab3f",
            "accent": "#ffd9a0",
            "speed": 2.4,
        },
        "speaking": {
            "color": "#3fffb0",
            "accent": "#b0ffdf",
            "speed": 2.0,
        },
        "error": {
            "color": "#ff5252",
            "accent": "#ffb0b0",
            "speed": 3.0,
        },
        "transcribing": {
            "color": "#ffab3f",
            "accent": "#ffd9a0",
            "speed": 2.0,
        },
    })
    
    # Audio
    fft_size: int = 1024
    fft_bins: int = 40  # Number of bins to send to overlay
    sample_rate: int = 16000
    audio_buffer_seconds: float = 0.5
    
    # Performance
    target_fps: int = 60
    max_cpu_percent: float = 15.0
    
    # Persistence
    config_file: Path = field(default_factory=lambda: APP_ROOT / "overlay_config.json")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert Path to string
        data["config_file"] = str(data["config_file"])
        # Convert tuples to lists
        for key in ["ring_multipliers", "ring_line_widths", "ring_dash_patterns", 
                    "ring_rotation_speeds", "ring_global_alphas"]:
            data[key] = list(data[key])
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "OverlayConfig":
        """Create config from dictionary."""
        # Convert lists back to tuples
        for key in ["ring_multipliers", "ring_line_widths", "ring_dash_patterns",
                    "ring_rotation_speeds", "ring_global_alphas"]:
            if key in data:
                data[key] = tuple(data[key])
        # Convert config_file back to Path
        if "config_file" in data and isinstance(data["config_file"], str):
            data["config_file"] = Path(data["config_file"])
        return cls(**data)
    
    def save(self) -> None:
        """Save config to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning("Failed to save overlay config: %s", e)
    
    @classmethod
    def load(cls) -> "OverlayConfig":
        """Load config from file, or create default."""
        config = cls()
        try:
            if config.config_file.exists():
                with open(config.config_file, "r") as f:
                    data = json.load(f)
                config = cls.from_dict(data)
                logger.info("Loaded overlay config from %s", config.config_file)
        except Exception as e:
            logger.warning("Failed to load overlay config, using defaults: %s", e)
        return config


# Merge with settings from .env if available
def _merge_with_settings(config: OverlayConfig) -> OverlayConfig:
    """Merge config with settings from nexus_config.settings."""
    settings = get_settings()
    
    # Override with settings if present
    if hasattr(settings, "OVERLAY_WS_PORT") and settings.OVERLAY_WS_PORT:
        config.ws_port = settings.OVERLAY_WS_PORT
    if hasattr(settings, "OVERLAY_WS_HOST") and settings.OVERLAY_WS_HOST:
        config.ws_host = settings.OVERLAY_WS_HOST
    if hasattr(settings, "OVERLAY_WINDOW_WIDTH") and settings.OVERLAY_WINDOW_WIDTH:
        config.window_width = settings.OVERLAY_WINDOW_WIDTH
    if hasattr(settings, "OVERLAY_WINDOW_HEIGHT") and settings.OVERLAY_WINDOW_HEIGHT:
        config.window_height = settings.OVERLAY_WINDOW_HEIGHT
    
    return config


@lru_cache(maxsize=1)
def get_overlay_config() -> OverlayConfig:
    """
    Get the singleton OverlayConfig instance.
    
    Loads from file, merges with settings, returns cached instance.
    """
    config = OverlayConfig.load()
    config = _merge_with_settings(config)
    return config