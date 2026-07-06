# NEXUS AI v4.0 - Configuration Hub (Pydantic BaseSettings)
import os,sys,platform,logging; from pathlib import Path; from typing import Optional,List,Literal, Dict, Any
from functools import lru_cache; from collections import namedtuple; from dotenv import load_dotenv; from pydantic import Field,field_validator,model_validator; from pydantic_settings import BaseSettings

APP_ROOT=Path.home()/".nexus_ai"
_REQUIRED_DIRS = [APP_ROOT,APP_ROOT/"logs",APP_ROOT/"db",APP_ROOT/"db"/"model_cache",APP_ROOT/"plugins",
    APP_ROOT/"workflows",APP_ROOT/"synthesized_tools",APP_ROOT/"temp",APP_ROOT/"metrics",APP_ROOT/"crash_reports",
    APP_ROOT/"assets"/"sounds",APP_ROOT/"assets"/"fonts"]
for d in _REQUIRED_DIRS: d.mkdir(parents=True,exist_ok=True)

BootStatus=namedtuple("BootStatus",["ok","missing_keys","warnings","platform_notes","degraded_subsystems"])
def is_windows():return platform.system()=="Windows"
def is_macos():return platform.system()=="Darwin"
def is_linux():return platform.system()=="Linux"

class Settings(BaseSettings):
    GROQ_API_KEY: Optional[str] = Field(None)
    PRIMARY_MODEL: str = Field("llama-3.3-70b-versatile")
    OPENAI_API_KEY: Optional[str] = Field(None)
    FALLBACK_MODEL: str = Field("gpt-4o-mini")
    OLLAMA_BASE_URL: str = Field("http://localhost:11434")
    OLLAMA_MODEL: str = Field("llama3.2:3b-instruct-q4_K_M")
    OLLAMA_INTENT_MODEL: str = Field("tinyllama:1.1b-chat-v1-q4_K_M")
    AGENT_MAX_RETRIES: int = Field(3, ge=1, le=10)
    AGENT_MAX_ITERATIONS: int = Field(25, ge=5, le=50)
    LLM_TEMPERATURE: float = Field(0.1, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(4096, ge=256, le=32768)
    LLM_REQUEST_TIMEOUT: int = Field(60, ge=10, le=300)
    CIRCUIT_BREAKER_FAILURES: int = Field(3, ge=1, le=10)
    CIRCUIT_BREAKER_RESET_SECONDS: int = Field(60, ge=10, le=600)
    GROQ_REQUESTS_PER_MINUTE: int = Field(30, ge=1, le=1000)
    OPENAI_REQUESTS_PER_MINUTE: int = Field(20, ge=1, le=1000)
    OLLAMA_REQUESTS_PER_MINUTE: int = Field(100, ge=1, le=10000)
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
    UI_THEME: Literal["dark_platinum", "midnight_blue", "void_black"] = Field("dark_platinum")
    UI_PARTICLE_COUNT: int = Field(60, ge=0, le=200)
    UI_ANIMATION_FPS: int = Field(60, ge=30, le=120)
    UI_FONT_SCALE: float = Field(1.0, ge=0.5, le=2.0)
    WINDOW_WIDTH: int = Field(1400, ge=800, le=3840)
    WINDOW_HEIGHT: int = Field(900, ge=600, le=2160)
    AUDIT_LOG_MAX_BYTES: int = Field(10485760)
    AUDIT_LOG_BACKUP_COUNT: int = Field(5, ge=1, le=20)
    METRICS_ENABLED: bool = Field(True)
    CRASH_REPORT_ENABLED: bool = Field(True)
    TIER: Literal["free", "personal_pro", "team", "enterprise"] = Field("free")
    FREE_TIER_MONTHLY_TASKS: int = Field(100)
    LICENSE_KEY: Optional[str] = Field(None)

    # ── Billing / Monetization (Section 1.3) ──────────────────────────────
    CLOUD_BACKEND_URL: str = Field("https://api.nexus-ai.dev")
    BILLING_PROVIDER: str = Field("stripe")  # "stripe" | "paddle" | "dodo" | "razorpay"
    LICENSE_REFRESH_INTERVAL_HOURS: int = Field(72, ge=1, le=720)
    ENABLE_USAGE_METERING: bool = Field(True)
    MARKETPLACE_REVENUE_SHARE_PLUGIN: float = Field(0.30, ge=0.0, le=1.0)
    MARKETPLACE_REVENUE_SHARE_WORKFLOW: float = Field(0.20, ge=0.0, le=1.0)

    # ── LAW 1.1 Bounds ───────────────────────────────────────────────────
    MAX_RECURSION_DEPTH: int = Field(3, ge=1, le=10)
    MAX_DAG_NODES: int = Field(20, ge=1, le=100)
    MAX_DEBUG_RETRIES: int = Field(3, ge=1, le=10)
    MAX_SEND_DAILY: int = Field(20, ge=0, le=500)
    NEW_SENDER_DAILY_CAP: int = Field(20, ge=0, le=100)

    api_key_missing:bool=Field(False,exclude=True)
    platform_name:str=Field(default_factory=lambda:{"Windows":"Windows","Darwin":"macOS","Linux":"Linux"}.get(platform.system(),"Unknown"),exclude=True)
    app_version:str=Field("4.0.0",exclude=True)

    model_config={"env_file":str(APP_ROOT/".env"),"env_file_encoding":"utf-8","extra":"ignore","case_sensitive":False}

    @field_validator("WAKE_WORDS")
    @classmethod
    def validate_wake_words(cls, v: List[str]) -> List[str]:
        """Ensure wake words are lowercase and non-empty."""
        validated = [w.lower().strip() for w in v if w.strip()]
        if not validated:
            raise ValueError("WAKE_WORDS must contain at least one non-empty keyword.")
        return validated

    @model_validator(mode="after")
    def warn_on_low_ram_with_large_model(self) -> "Settings":
        """Emit a warning if the chosen Ollama model may exhaust available RAM."""
        try:
            import psutil
            available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
            if "70b" in self.OLLAMA_MODEL.lower() and available_ram_mb < 48000:
                logging.getLogger("nexus.settings").warning(
                    f"OLLAMA_MODEL {self.OLLAMA_MODEL} requires ~40GB RAM. "
                    f"Available: {available_ram_mb:.0f}MB. This will likely OOM."
                )
            elif "13b" in self.OLLAMA_MODEL.lower() and available_ram_mb < 10000:
                logging.getLogger("nexus.settings").warning(
                    f"OLLAMA_MODEL {self.OLLAMA_MODEL} requires ~8GB RAM. "
                    f"Available: {available_ram_mb:.0f}MB. May cause system slowdown."
                )
        except ImportError:
            pass  # psutil might not be installed on first run
        return self

@lru_cache(maxsize=1)
def get_settings():
    load_dotenv(APP_ROOT/".env", override=False)
    return Settings()

def validate_on_boot():
    settings=get_settings();missing_keys=[];warnings=[];platform_notes=[];degraded=[]
    if not settings.GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
        if not settings.OPENAI_API_KEY:degraded.append("cloud_llm");warnings.append("No API keys. Using Ollama only.")
    if not settings.PORCUPINE_ACCESS_KEY:degraded.append("wake_word");warnings.append("No Porcupine key.")
    if sys.version_info<(3,10):warnings.append(f"Python {sys.version_info.major}.{sys.version_info.minor}. 3.10+ required.")
    try:import chromadb; chromadb.Client()
    except:degraded.append("vector_memory");warnings.append("chromadb not installed.")
    try:
        import pygame;pygame.mixer.pre_init(frequency=settings.AUDIO_SAMPLE_RATE,size=-16,channels=1,buffer=settings.AUDIO_BUFFER_SIZE);pygame.mixer.init();pygame.mixer.quit()
    except Exception as e:degraded.append("audio");warnings.append(f"Audio: {e}")
    try:
        import urllib.request;req=urllib.request.Request(f"{settings.OLLAMA_BASE_URL}/api/tags",headers={"User-Agent":"NexusAI/4.0"});urllib.request.urlopen(req,timeout=3)
    except:degraded.append("ollama");warnings.append(f"Ollama unreachable at {settings.OLLAMA_BASE_URL}.")
    if is_windows():platform_notes.append("Windows: Admin recommended.")
    elif is_macos():platform_notes.append("macOS: Screen recording permission required.")
    elif is_linux():platform_notes.append("Linux: DISPLAY must be set.")
    ok=len(missing_keys)==0 or ("ollama" not in degraded)
    return BootStatus(ok=ok,missing_keys=missing_keys,warnings=warnings,platform_notes=platform_notes,degraded_subsystems=degraded)
