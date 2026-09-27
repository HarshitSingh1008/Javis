"""
NEXUS AI v4.0 — UI Package Init
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Exports the CustomTkinter HUD, theme engine, animation engine,
notification manager, onboarding wizard, and DAG visualizer.
"""

from nexus_ui.theme_engine import ThemeEngine, get_theme_engine, ThemeColors
from nexus_ui.animation_engine import AnimationEngine, Particle, ArcReactorRing, ArcReactorCore, WaveformBand
from nexus_ui.notification_manager import NotificationManager, get_notification_manager
from nexus_ui.custom_hud import NexusHUD, run_hud
from nexus_ui.onboarding import OnboardingWizard
from nexus_ui.dag_visualizer import DAGVisualizer

__all__ = [
    "ThemeEngine",
    "get_theme_engine",
    "ThemeColors",
    "AnimationEngine",
    "Particle",
    "ArcReactorRing",
    "ArcReactorCore",
    "WaveformBand",
    "NotificationManager",
    "get_notification_manager",
    "NexusHUD",
    "run_hud",
    "OnboardingWizard",
    "DAGVisualizer",
]