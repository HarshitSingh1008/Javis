# NEXUS AI v4.0 - Configuration Hub (Pydantic BaseSettings)
import os, sys, platform, logging
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
from functools import lru_cache
from collections import namedtuple
from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

APP_ROOT = Path.home() / ".nexus_ai"
_REQUIRED_DIRS = [
    APP_ROOT, APP_ROOT / "logs", APP_ROOT / "db", APP_ROOT / "db" / "model_cache",
    APP_ROOT / "plugins", APP_ROOT / "workflows", APP_ROOT / "synthesized_tools",
    APP_ROOT / "temp", APP_ROOT / "metrics", APP_ROOT / "crash_reports",
    APP_ROOT / "assets" / "sounds", APP_ROOT / "assets" / "fonts"
]
for d in _REQUIRED_DIRS:
    d.mkdir(parents=True, exist_ok=True)

BootStatus = namedtuple("BootStatus", ["ok", "missing_keys", "warnings", "platform_notes", "degraded_subsystems"])
def is_windows(): return platform.system() == "Windows"
def is_macos(): return platform.system() == "Darwin"
def is_linux(): return platform.system() == "Linux"


class Settings(BaseSettings):
    # ── LLM Primary (Ollama — Local, Free, Unlimited) ──────────────────────────
    OLLAMA_BASE_URL: str = Field(
        "http://localhost:11434",
        description="Ollama server URL. Default: localhost."
    )
    OLLAMA_MODEL: str = Field(
        "llama3.2:3b-instruct-q4_K_M",
        description="Ollama model name. Recommended: llama3.2:3b-instruct-q4_K_M (2.1GB RAM)"
    )
    OLLAMA_INTENT_MODEL: str = Field(
        "tinyllama:1.1b-chat-v1-q4_K_M",
        description="Ultra-fast local model for intent classification only."
    )

    # ── LLM Fallback (Groq — Cloud, Rate-Limited, Optional) ────────────────────
    GROQ_API_KEY: Optional[str] = Field(
        None,
        description="Optional Groq API key for cloud fallback. Get free at https://console.groq.com"
    )
    PLANNER_MODEL: str = Field("qwen/qwen3.8-27b")
    AUTOMATION_MODEL: str = Field("qwen/qwen3.8-27b")
    VOICE_MODEL: str = Field("qwen/qwen3.8-27b")

    # Backward compat aliases (used by orchestrator / context builder)
    PRIMARY_MODEL: str = Field("llama3.2:3b-instruct-q4_K_M")
    FALLBACK_MODEL: str = Field("llama3.2:3b-instruct-q4_K_M")
    INTENT_MODEL: str = Field("tinyllama:1.1b-chat-v1-q4_K_M")
    OPENAI_API_KEY: Optional[str] = Field(None)

    AGENT_MAX_RETRIES: int = Field(10, ge=1, le=10)
    AGENT_MAX_ITERATIONS: int = Field(50, ge=5, le=50)
    LLM_TEMPERATURE: float = Field(0.1, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(32768, ge=256, le=32768)
    LLM_REQUEST_TIMEOUT: int = Field(300, ge=10, le=300)
    CIRCUIT_BREAKER_FAILURES: int = Field(10, ge=1, le=10)
    CIRCUIT_BREAKER_RESET_SECONDS: int = Field(60, ge=10, le=600)
    GROQ_REQUESTS_PER_MINUTE: int = Field(1000, ge=1, le=1000)
    OPENAI_REQUESTS_PER_MINUTE: int = Field(1000, ge=1, le=1000)

    PORCUPINE_ACCESS_KEY: Optional[str] = Field(None)
    WAKE_WORDS: List[str] = Field(["nexus", "jarvis"])
    WAKE_WORD_SENSITIVITY: float = Field(0.5, ge=0.0, le=1.0)
    DEFAULT_TTS_VOICE: str = Field("NEXUS_PRIME")
    TTS_SPEAKING_RATE: str = Field("+10%")
    AUDIO_SAMPLE_RATE: int = Field(22050)
    AUDIO_BUFFER_SIZE: int = Field(512)
    WHISPER_MODEL_SIZE: str = Field("base")
    WHISPER_RECORD_SECONDS: float = Field(5.0, ge=1.0, le=30.0)
    ENABLE_WAKE_WORD: bool = Field(True)
    ENABLE_TTS: bool = Field(True)
    SANDBOX_TIMEOUT_SECONDS: int = Field(30, ge=5, le=300)
    SANDBOX_MAX_MEMORY_MB: int = Field(512, ge=64, le=4096)
    ENABLE_DOCKER_SANDBOX: bool = Field(False)
    PLUGIN_HOT_RELOAD: bool = Field(True)
    PLUGIN_RELOAD_DEBOUNCE_SECONDS: float = Field(2.0, ge=0.5, le=30.0)
    ENABLE_BUNDLED_PLUGINS: bool = Field(True)
    PARALLEL_TOOL_WORKERS: int = Field(3, ge=1, le=16)
    MEMORY_INJECTION_RESULTS: int = Field(5, ge=1, le=20)
    MEMORY_MAX_TOKENS: int = Field(500, ge=100, le=2000)
    SYNTHESIS_MAX_RETRIES: int = Field(3, ge=1, le=5)
    CONVERSATION_CONTEXT_WINDOW: int = Field(20, ge=5, le=100)
    WORKFLOW_HISTORY_LENGTH: int = Field(100, ge=10, le=500)
    UI_THEME: Literal["jarvis", "jarvis_classic", "jarvis_prime", "midnight_blue", "void_black"] = Field("jarvis")
    UI_PARTICLE_COUNT: int = Field(60, ge=0, le=200)
    UI_ANIMATION_FPS: int = Field(60, ge=30, le=120)
    UI_FONT_SCALE: float = Field(1.0, ge=0.5, le=2.0)
    WINDOW_WIDTH: int = Field(1400, ge=800, le=3840)
    WINDOW_HEIGHT: int = Field(900, ge=600, le=2160)
    
    # ── Ring Overlay Settings ─────────────────────────────────────────────────
    OVERLAY_ENABLED: bool = Field(True, description="Enable ring overlay (pywebview)")
    OVERLAY_WS_HOST: str = Field("127.0.0.1", description="WebSocket host for overlay")
    OVERLAY_WS_PORT: int = Field(8765, ge=1024, le=65535, description="WebSocket port for overlay")
    OVERLAY_WINDOW_WIDTH: int = Field(360, ge=160, le=480, description="Overlay window width")
    OVERLAY_WINDOW_HEIGHT: int = Field(520, ge=160, le=720, description="Overlay window height")
    OVERLAY_ALWAYS_ON_TOP: bool = Field(True, description="Keep overlay always on top")
    OVERLAY_FRAMeless: bool = Field(True, description="Frameless overlay window")
    OVERLAY_TRANSPARENT: bool = Field(True, description="Transparent overlay background")
    
    AUDIT_LOG_MAX_BYTES: int = Field(10485760)
    AUDIT_LOG_BACKUP_COUNT: int = Field(5, ge=1, le=20)
    METRICS_ENABLED: bool = Field(True)
    CRASH_REPORT_ENABLED: bool = Field(True)

    # ── LAW 1.1 Bounds ───────────────────────────────────────────────────
    MAX_RECURSION_DEPTH: int = Field(3, ge=1, le=10)
    MAX_DAG_NODES: int = Field(20, ge=1, le=100)
    MAX_DEBUG_RETRIES: int = Field(3, ge=1, le=10)
    MAX_SEND_DAILY: int = Field(20, ge=0, le=500)
    NEW_SENDER_DAILY_CAP: int = Field(20, ge=0, le=100)

    # ── Runtime State (not persisted, computed at boot) ───────────────────────
    api_key_missing: bool = Field(False, exclude=True)
    platform_name: str = Field(default_factory=lambda: {"Windows": "Windows", "Darwin": "macOS", "Linux": "Linux"}.get(platform.system(), "Unknown"), exclude=True)
    app_version: str = Field("4.0.0", exclude=True)

    model_config = {"env_file": str(APP_ROOT / ".env"), "env_file_encoding": "utf-8", "extra": "ignore", "case_sensitive": False}

    @field_validator("WAKE_WORDS")
    @classmethod
    def validate_wake_words(cls, v: List[str]) -> List[str]:
        """Ensure wake words are lowercase and non-empty."""
        validated = [w.lower().strip() for w in v if w.strip()]
        if not validated:
            raise ValueError("WAKE_WORDS must contain at least one non-empty keyword.")
        return validated

    @field_validator("OLLAMA_MODEL")
    @classmethod
    def validate_ollama_model(cls, v: str) -> str:
        """Warn if model name doesn't include quantization suffix."""
        if ":" not in v:
            logging.getLogger("nexus.settings").warning(
                f"OLLAMA_MODEL '{v}' has no quantization tag (e.g., :3b-instruct-q4_K_M). "
                f"This may pull a large model. Consider specifying a Q4 quantized variant."
            )
        return v

    @model_validator(mode="after")
    def warn_on_missing_groq_key(self) -> "Settings":
        """Emit a warning if GROQ_API_KEY is not set (optional cloud fallback)."""
        if not self.GROQ_API_KEY:
            logging.getLogger("nexus.settings").info(
                "GROQ_API_KEY not set. Using Ollama (local) only. "
                "Add GROQ_API_KEY to ~/.nexus_ai/.env for cloud fallback."
            )
        return self


def get_settings() -> Settings:
    """
    Get settings instance.
    
    Note: This is no longer cached to support hot-reload.
    Use nexus_brain.config_watcher.get_config_watcher() for hot-reload support.
    """
    load_dotenv(APP_ROOT / ".env", override=True)
    return Settings()


# Keep the old cached version for backward compatibility
import functools
@functools.lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    """Get cached settings (does not support hot-reload)."""
    load_dotenv(APP_ROOT / ".env", override=False)
    return Settings()


# Alias for backward compatibility
get_settings_cached = get_cached_settings


def validate_on_boot():
    settings = get_settings()
    missing_keys = []
    warnings = []
    platform_notes = []
    degraded = []

    # Check 1: Groq API key (optional — cloud fallback only)
    if not settings.GROQ_API_KEY:
        # Not a missing key — just info that cloud fallback is unavailable
        warnings.append(
            "No GROQ_API_KEY configured. NEXUS AI will use Ollama (local) only. "
            "Add GROQ_API_KEY to ~/.nexus_ai/.env for cloud fallback."
        )

    # Check 2: Porcupine key (required for wake word)
    if not settings.PORCUPINE_ACCESS_KEY:
        degraded.append("wake_word")
        warnings.append("No Porcupine key. Wake word disabled. Use Ctrl+Shift+N for voice input.")

    # Check 3: Python version
    if sys.version_info < (3, 10):
        warnings.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} detected. "
            f"NEXUS AI requires Python 3.10+. Some features may not work."
        )

    # Check 4: chromadb importability
    try:
        import chromadb
        chromadb.Client()
    except ImportError:
        degraded.append("vector_memory")
        warnings.append("chromadb not installed. Vector memory will use in-memory fallback (not persistent).")

    # Check 5: pygame audio
    try:
        import pygame
        pygame.mixer.pre_init(
            frequency=settings.AUDIO_SAMPLE_RATE,
            size=-16, channels=1,
            buffer=settings.AUDIO_BUFFER_SIZE
        )
        pygame.mixer.init()
        pygame.mixer.quit()
    except Exception as e:
        degraded.append("audio")
        warnings.append(f"Audio initialization failed: {e}. TTS disabled, using text fallback.")

    # Check 6: Disk space
    try:
        import shutil
        free_gb = shutil.disk_usage(APP_ROOT).free / (1024 ** 3)
        if free_gb < 0.5:
            warnings.append(f"Low disk space: {free_gb:.1f}GB free in {APP_ROOT}. Min 500MB recommended.")
        elif free_gb < 2.0:
            warnings.append(f"Disk space getting low: {free_gb:.1f}GB free. Consider cleanup.")
    except Exception:
        pass

    # Check 7: Available RAM
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        if available_mb < 1500:
            degraded.append("whisper")
            warnings.append(
                f"Low available RAM: {available_mb:.0f}MB free. "
                f"Whisper model will not pre-load. Voice transcription may be slow."
            )
    except ImportError:
        pass

    # Check 8: Ollama connectivity
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{settings.OLLAMA_BASE_URL}/api/tags",
            headers={"User-Agent": "NexusAI/4.0"},
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status != 200:
                degraded.append("ollama")
                warnings.append("Ollama not responding. Local model fallback unavailable.")
    except Exception:
        degraded.append("ollama")
        warnings.append(
            f"Ollama not reachable at {settings.OLLAMA_BASE_URL}. "
            f"Install Ollama: https://ollama.ai — then: ollama pull {settings.OLLAMA_MODEL}"
        )

    # Check 9: Platform-specific notes
    if is_windows():
        platform_notes.append("Windows: Using pyautogui for desktop automation. Run as administrator for full control.")
        platform_notes.append("Windows: Audio uses WASAPI backend via pygame.")
    elif is_macos():
        platform_notes.append("macOS: Screen recording permission required for desktop automation (System Settings → Privacy).")
        platform_notes.append("macOS: Microphone permission required for voice input.")
    elif is_linux():
        platform_notes.append("Linux: Ensure DISPLAY environment variable is set for pyautogui.")
        platform_notes.append("Linux: ALSA errors suppressed at startup (normal behavior).")

    ok = len(missing_keys) == 0 or (settings.OLLAMA_BASE_URL and "ollama" not in degraded)

    return BootStatus(
        ok=ok,
        missing_keys=missing_keys,
        warnings=warnings,
        platform_notes=platform_notes,
        degraded_subsystems=degraded,
    )