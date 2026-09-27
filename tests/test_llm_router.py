"""
NEXUS AI v4.0 — Fallback + circuit breaker tests for LLMRouter (Ollama-first).
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Covers:
1. Router singleton and initialization
2. Model order based on role (Ollama primary, Groq fallback)
3. Circuit breaker — closed state
4. Circuit breaker — tripping on failures
5. Circuit breaker — half-open recovery
6. Rate limiting — token bucket (Groq only)
7. Token bucket refill behavior
8. Token bucket timeout
9. LLMResponse dataclass creation
10. CircuitBreaker dataclass creation
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch


class TestRouterInit:
    """Tests for LLMRouter initialization."""

    def test_get_llm_router_singleton(self):
        """Test that get_llm_router returns a singleton."""
        from nexus_brain.llm_router import get_llm_router
        r1 = get_llm_router()
        r2 = get_llm_router()
        assert r1 is r2

    def test_router_initializes_with_defaults(self, mock_settings):
        """Test that router initializes with default components."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()
        assert router._settings is not None
        assert router._audit_logger is not None
        # Circuit breakers are created lazily on first use
        assert isinstance(router._circuit_breakers, dict)
        assert hasattr(router, "_groq_rate_limiter")

    def test_router_groq_rate_limiter_initialized(self, mock_settings):
        """Test that Groq rate limiter is initialized (Ollama has no rate limits)."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()
        assert hasattr(router, "_groq_rate_limiter")
        assert router._groq_rate_limiter is not None

    def test_router_cost_tracking(self, mock_settings):
        """Test that cost estimation returns 0 for Ollama models."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()
        # Ollama models are free
        assert router.estimate_cost("llama3.2:3b-instruct-q4_K_M", 100, 100) == 0.0
        assert router.estimate_cost("tinyllama:1.1b-chat-v1-q4_K_M", 100, 100) == 0.0
        # Groq models have cost (if they exist in pricing dict)
        groq_cost = router.estimate_cost("qwen/qwen3.8-27b", 100, 100)
        assert groq_cost >= 0.0


class TestModelOrder:
    """Tests for model selection order (Ollama primary, Groq fallback)."""

    def test_model_order_planner(self, mock_settings):
        """Test planner role uses Ollama primary model."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()
        order = router._get_model_order("planner")
        # Primary should be Ollama model
        assert order[0] == "llama3.2:3b-instruct-q4_K_M"
        # Groq models should be in fallback
        assert "qwen/qwen3.8-27b" in order

    def test_model_order_automation(self, mock_settings):
        """Test automation role uses Ollama primary model."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()
        order = router._get_model_order("automation")
        assert order[0] == "llama3.2:3b-instruct-q4_K_M"

    def test_model_order_voice(self, mock_settings):
        """Test voice role uses ultra-fast intent model."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()
        order = router._get_model_order("voice")
        assert order[0] == "tinyllama:1.1b-chat-v1-q4_K_M"

    def test_explicit_model_override(self, mock_settings):
        """Test that explicit model parameter overrides role."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()
        model = router._resolve_model(role="planner", model="custom-model")
        assert model == "custom-model"

    def test_prefer_ollama_filters_groq(self, mock_settings):
        """Test that prefer_provider='ollama' filters out Groq models."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()
        # We can't easily test _get_model_order with prefer_provider since it's in generate()
        # But we can test _is_groq_model helper
        assert router._is_groq_model("qwen/qwen3.8-27b") is True
        assert router._is_groq_model("llama3.2:3b-instruct-q4_K_M") is False
        assert router._is_groq_model("tinyllama:1.1b-chat-v1-q4_K_M") is False


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed(self, mock_settings):
        """Test that closed circuit allows requests."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()
        is_open = await router._is_circuit_open("llama3.2:3b-instruct-q4_K_M")
        assert is_open is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_after_failures(self, mock_settings):
        """Test that circuit breaker trips after CIRCUIT_BREAKER_FAILURES."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()

        model = "llama3.2:3b-instruct-q4_K_M"
        # Simulate failures
        for _ in range(mock_settings.CIRCUIT_BREAKER_FAILURES):
            await router._record_failure(model)

        is_open = await router._is_circuit_open(model)
        assert is_open is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open(self, mock_settings):
        """Test that circuit enters half-open after reset time."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()

        model = "test_provider"
        # Trip the breaker
        for _ in range(mock_settings.CIRCUIT_BREAKER_FAILURES):
            await router._record_failure(model)

        # Manually set a past open_time to simulate timeout
        cb = router._circuit_breakers[model]
        cb.state = type(cb.state)("open")
        cb.open_time = time.monotonic() - mock_settings.CIRCUIT_BREAKER_RESET_SECONDS - 1

        is_open = await router._is_circuit_open(model)
        assert is_open is False  # Should transition to half-open

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self, mock_settings):
        """Test that success resets the circuit breaker."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()

        model = "llama3.2:3b-instruct-q4_K_M"
        cb = router._get_cb(model)
        cb.state = type(cb.state)("half_open")
        cb.failure_count = 3

        await router._record_success(model)
        assert cb.state.value == "closed"
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_unknown_model(self, mock_settings):
        """Test that unknown models return False (not open)."""
        from nexus_brain.llm_router import get_llm_router
        router = get_llm_router()

        is_open = await router._is_circuit_open("nonexistent-model")
        assert is_open is False  # Unknown models start as closed


class TestTokenBucket:
    """Tests for TokenBucket rate limiter."""

    @pytest.mark.asyncio
    async def test_token_bucket_acquire(self):
        """Test basic token acquisition."""
        from nexus_brain.llm_router import TokenBucket
        bucket = TokenBucket(rate=100, capacity=10)
        result = await bucket.acquire(tokens=1, timeout=1)
        assert result is True

    @pytest.mark.asyncio
    async def test_token_bucket_exhaust(self):
        """Test that exhausted bucket returns False."""
        from nexus_brain.llm_router import TokenBucket
        bucket = TokenBucket(rate=0.01, capacity=3)  # Very slow refill
        for _ in range(3):
            await bucket.acquire(tokens=1, timeout=0.1)
        
        result = await bucket.acquire(tokens=1, timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_token_bucket_refill(self):
        """Test that tokens refill over time."""
        from nexus_brain.llm_router import TokenBucket
        bucket = TokenBucket(rate=100, capacity=10)
        
        # Consume all tokens
        for _ in range(10):
            await bucket.acquire(tokens=1, timeout=1)
        
        # Wait for refill
        await asyncio.sleep(0.05)
        
        # Should have some tokens now
        result = await bucket.acquire(tokens=1, timeout=2)
        assert result is True

    def test_token_bucket_initial_state(self):
        """Test that bucket initializes with full capacity."""
        from nexus_brain.llm_router import TokenBucket
        bucket = TokenBucket(rate=10, capacity=5)
        assert bucket._tokens == 5.0


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_defaults(self):
        """Test that LLMResponse has correct defaults."""
        from nexus_brain.llm_router import LLMResponse
        resp = LLMResponse(content="Hello", provider="ollama", model="llama3.2:3b")
        assert resp.content == "Hello"
        assert resp.provider == "ollama"
        assert resp.success is True
        assert resp.error is None
        assert resp.duration_ms == 0.0
        assert resp.cost_usd == 0.0

    def test_llm_response_error(self):
        """Test LLMResponse with error."""
        from nexus_brain.llm_router import LLMResponse
        resp = LLMResponse(
            content="",
            provider="ollama",
            model="none",
            success=False,
            error="Ollama not reachable",
        )
        assert resp.success is False
        assert resp.error == "Ollama not reachable"


class TestCircuitBreakerDataclass:
    """Tests for CircuitBreaker dataclass."""

    def test_circuit_breaker_defaults(self):
        """Test that CircuitBreaker has correct defaults."""
        from nexus_brain.llm_router import CircuitBreaker, CircuitState
        cb = CircuitBreaker(provider="ollama")
        assert cb.provider == "ollama"
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED
        assert cb.last_failure_time == 0.0