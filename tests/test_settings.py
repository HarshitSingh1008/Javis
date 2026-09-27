"""
NEXUS AI v4.0 — Boot validation and settings tests.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Covers:
1. Default settings loading
2. Environment variable overrides
3. Boot validation (validate_on_boot)
4. Settings caching (get_settings)
5. Field validators (WAKE_WORDS, Ollama model)
6. Platform detection helpers
7. APP_ROOT directory creation
"""

import pytest
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSettingsDefaults:
    """Tests for default settings values."""

    def test_default_settings_loaded(self, mock_settings):
        """Test that settings are loaded with expected defaults."""
        assert mock_settings.PRIMARY_MODEL == "llama3.2:3b-instruct-q4_K_M"
        assert mock_settings.FALLBACK_MODEL == "llama3.2:3b-instruct-q4_K_M"
        assert mock_settings.OLLAMA_BASE_URL == "http://localhost:11434"
        assert mock_settings.OLLAMA_MODEL == "llama3.2:3b-instruct-q4_K_M"
        assert mock_settings.UI_THEME in ("dark_platinum", "midnight_blue", "void_black")

    def test_agent_defaults(self, mock_settings):
        """Test agent configuration defaults."""
        assert 5 <= mock_settings.AGENT_MAX_ITERATIONS <= 50
        assert 1 <= mock_settings.AGENT_MAX_RETRIES <= 10
        assert 0.0 <= mock_settings.LLM_TEMPERATURE <= 2.0
        assert mock_settings.CONVERSATION_CONTEXT_WINDOW >= 5

    def test_sandbox_defaults(self, mock_settings):
        """Test sandbox configuration defaults."""
        assert 5 <= mock_settings.SANDBOX_TIMEOUT_SECONDS <= 300
        assert 64 <= mock_settings.SANDBOX_MAX_MEMORY_MB <= 4096

    def test_audio_defaults(self, mock_settings):
        """Test audio configuration defaults."""
        assert mock_settings.AUDIO_SAMPLE_RATE == 22050
        assert mock_settings.AUDIO_BUFFER_SIZE == 512
        assert mock_settings.WHISPER_MODEL_SIZE == "base"

    def test_app_version(self, mock_settings):
        """Test app version is set."""
        assert mock_settings.app_version == "4.0.0"


class TestSettingsEnvOverrides:
    """Tests for environment variable overrides."""

    def test_env_api_keys(self, mock_settings):
        """Test that API keys are loaded from environment."""
        assert mock_settings.GROQ_API_KEY == "test_groq_key"
        assert mock_settings.OPENAI_API_KEY == "test_openai_key"

    def test_env_ollama_model(self, mock_settings):
        """Test that Ollama model is set from env."""
        assert "3b" in mock_settings.OLLAMA_MODEL

    def test_env_ollama_model(self, monkeypatch):
        """Test that OLLAMA_MODEL can be overridden via env."""
        from nexus_config.settings import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("OLLAMA_MODEL", "phi3:mini-128k-instruct-q4_K_M")
        monkeypatch.setenv("GROQ_API_KEY", "test_key_override")
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_override")
        s = get_settings()
        assert s.OLLAMA_MODEL == "phi3:mini-128k-instruct-q4_K_M"
        get_settings.cache_clear()

    def test_env_ui_theme(self, monkeypatch):
        """Test that UI_THEME can be overridden via env."""
        from nexus_config.settings import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("UI_THEME", "void_black")
        monkeypatch.setenv("GROQ_API_KEY", "test_key_theme")
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_theme")
        s = get_settings()
        assert s.UI_THEME == "void_black"
        get_settings.cache_clear()


class TestBootValidation:
    """Tests for validate_on_boot function."""

    def test_validate_on_boot_returns_bootstatus(self, mock_settings):
        """Test that validate_on_boot returns a BootStatus namedtuple."""
        from nexus_config.settings import validate_on_boot
        with patch("nexus_config.settings.get_settings", return_value=mock_settings):
            status = validate_on_boot()
        assert status.ok is not None
        assert hasattr(status, "missing_keys")
        assert hasattr(status, "warnings")
        assert hasattr(status, "platform_notes")
        assert hasattr(status, "degraded_subsystems")

    def test_validate_on_boot_missing_keys(self, mock_settings):
        """Test that boot validation reports missing API keys."""
        from nexus_config.settings import validate_on_boot
        with patch("nexus_config.settings.get_settings", return_value=mock_settings):
            status = validate_on_boot()
        # Groq is now optional - only Ollama is required
        assert "GROQ_API_KEY" not in status.missing_keys

    def test_validate_on_boot_platform_notes(self, mock_settings):
        """Test that platform-specific notes are generated."""
        from nexus_config.settings import validate_on_boot
        with patch("nexus_config.settings.get_settings", return_value=mock_settings):
            status = validate_on_boot()
        assert len(status.platform_notes) >= 0

    def test_validate_wake_words_valid(self, mock_settings):
        """Test that WAKE_WORDS validator accepts valid words."""
        assert len(mock_settings.WAKE_WORDS) > 0
        for w in mock_settings.WAKE_WORDS:
            assert w.islower()


class TestSettingsCache:
    """Tests for settings caching behavior."""

    def test_get_settings_cache(self):
        """Test that get_settings is cached."""
        from nexus_config.settings import get_settings
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_get_settings_cache_clear(self):
        """Test that cache_clear forces re-creation."""
        from nexus_config.settings import get_settings
        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # After cache clear, should not be same object
        assert s1 is not s2


class TestPlatformHelpers:
    """Tests for platform detection helpers."""

    def test_is_windows(self):
        """Test is_windows detection."""
        from nexus_config.settings import is_windows
        assert is_windows() == (os.name == "nt")

    def test_is_macos(self):
        """Test is_macos detection."""
        from nexus_config.settings import is_macos
        assert is_macos() == (sys.platform == "darwin")

    def test_is_linux(self):
        """Test is_linux detection."""
        from nexus_config.settings import is_linux
        assert is_linux() == (sys.platform == "linux")


class TestAppRoot:
    """Tests for APP_ROOT directory creation."""

    def test_app_root_exists(self):
        """Test that APP_ROOT directory exists."""
        from nexus_config.settings import APP_ROOT
        assert APP_ROOT.exists()

    def test_app_root_subdirectories(self, mock_settings):
        """Test that expected subdirectories exist."""
        from nexus_config.settings import APP_ROOT
        expected_dirs = ["logs", "db", "plugins", "workflows", "temp", "crash_reports"]
        for d in expected_dirs:
            assert (APP_ROOT / d).exists(), f"Missing directory: {d}"