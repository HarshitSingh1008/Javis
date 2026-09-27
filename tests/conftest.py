"""
NEXUS AI v4.0 — Shared pytest fixtures for all tests.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides mock settings, mock LLM, sandbox, registry, and vector_store fixtures.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import MagicMock, patch


# ─── Settings Fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_root() -> Generator[Path, None, None]:
    """Create a temporary test root directory."""
    tmp = Path(tempfile.mkdtemp(prefix="nexus_test_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    from nexus_config.settings import APP_ROOT
    
    test_env = {
        "GROQ_API_KEY": "test_groq_key",
        "OPENAI_API_KEY": "test_openai_key",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2:3b-instruct-q4_K_M",
        "ENABLE_WAKE_WORD": "false",
        "ENABLE_TTS": "false",
        "UI_THEME": "dark_platinum",
        "LOG_LEVEL": "DEBUG",
        "SANDBOX_TIMEOUT_SECONDS": "5",
        "AUDIT_LOG_MAX_BYTES": "1048576",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    # Clear cached settings
    from nexus_config.settings import get_settings
    get_settings.cache_clear()
    
    return get_settings()


# ─── Mock LLM Fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def mock_llm():
    """Mock LLM responses for testing."""
    mock = MagicMock()
    
    async def mock_agenerate(*args, **kwargs):
        from langchain_core.messages import AIMessage
        return {"generations": [[{"text": "Mock response", "message": AIMessage(content="Mock response")}]]}
    
    mock.agenerate = mock_agenerate
    return mock


# ─── Event Loop Fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="function")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ─── Vector Store Fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="function")
def vector_store(mock_settings):
    """Create a test vector store."""
    from nexus_memory.vector_store import get_vector_store
    get_vector_store.cache_clear()
    store = get_vector_store()
    yield store
    get_vector_store.cache_clear()


# ─── Tool Registry Fixture ────────────────────────────────────────────────

@pytest.fixture(scope="function")
def tool_registry(mock_settings):
    """Create a test tool registry."""
    from nexus_tools.registry import get_tool_registry
    get_tool_registry.cache_clear()
    registry = get_tool_registry()
    yield registry
    get_tool_registry.cache_clear()


# ─── Sandbox Fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def sandbox(mock_settings):
    """Create a test sandbox."""
    from nexus_tools.secure_sandbox import get_sandbox
    get_sandbox.cache_clear()
    sb = get_sandbox()
    yield sb
    get_sandbox.cache_clear()


# ─── Rate Limiter Fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="function")
def rate_limiter():
    """Create a test rate limiter with fast refill for testing."""
    from nexus_tools.rate_limiter import RateLimiter
    return RateLimiter(rate=100, per=1)  # 100 tokens per second for fast tests


# ─── Working Memory Fixture ───────────────────────────────────────────────

@pytest.fixture(scope="function")
def working_memory():
    """Create a test working memory."""
    return []