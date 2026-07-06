"""
NEXUS AI v4.0 — Triple LLM fallback with circuit breaker.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Routing hierarchy:
  1. Groq (primary) — llama-3.3-70b-versatile, 800+ tok/sec via LPU
  2. OpenAI (fallback) — gpt-4o-mini, cheap and capable
  3. Ollama (tertiary, offline) — llama3.2:3b-instruct-q4_K_M, ~18 tok/sec

The circuit breaker prevents repeatedly hammering a failing provider.
After CIRCUIT_BREAKER_FAILURES consecutive failures, the provider is
skipped for CIRCUIT_BREAKER_RESET_SECONDS seconds.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, AsyncIterator, List, Dict, Any, Literal, Tuple
from functools import lru_cache

from nexus_config.settings import get_settings
from nexus_config.audit_logger import get_audit_logger, audited


logger = logging.getLogger("nexus.llm_router")


# ─── Circuit Breaker States ───────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation — requests are sent
    OPEN = "open"          # Failing — requests are blocked
    HALF_OPEN = "half_open"  # Testing if provider recovered


@dataclass
class CircuitBreaker:
    """
    Per-provider circuit breaker state.
    
    Thread safety: Protected by asyncio.Lock within LLMRouter.
    Only accessed from the agent thread's event loop.
    """
    provider: str
    failure_count: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: float = 0.0
    open_time: float = 0.0


# ─── LLM Response Types ───────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str
    provider: str                             # "groq" | "openai" | "ollama"
    model: str                                # Model name used
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    success: bool = True
    error: Optional[str] = None
    streamed: bool = False


# ─── Rate Limiter Token Bucket ────────────────────────────────────────────────

class TokenBucket:
    """
    Simple token bucket rate limiter.
    
    Refills at rate tokens/second up to capacity.
    Thread-safe (asyncio.Lock).
    """
    
    def __init__(self, rate: float, capacity: int):
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1, timeout: float = 10.0) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum seconds to wait for tokens.
        
        Returns:
            True if tokens acquired, False if timeout.
        """
        start = time.monotonic()
        async with self._lock:
            while time.monotonic() - start < timeout:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                await asyncio.sleep(0.05)
            return False
    
    def _refill(self) -> None:
        elapsed = time.monotonic() - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = time.monotonic()


# ─── Main LLM Router ──────────────────────────────────────────────────────────

class LLMRouter:
    """
    Triple-provider LLM router with circuit breaker, rate limiting, and metrics.
    
    Usage:
        router = get_llm_router()
        response = await router.generate("Translate to French: Hello")
        
        # Streaming:
        async for chunk in router.stream("Write a poem"):
            print(chunk)
    """
    
    def __init__(self):
        self._settings = get_settings()
        self._audit_logger = get_audit_logger()
        
        # Circuit breakers per provider
        self._circuit_breakers: Dict[str, CircuitBreaker] = {
            "groq": CircuitBreaker("groq"),
            "openai": CircuitBreaker("openai"),
            "ollama": CircuitBreaker("ollama"),
        }
        self._cb_lock = asyncio.Lock()
        
        # Rate limiters
        self._rate_limiters: Dict[str, TokenBucket] = {
            "groq": TokenBucket(
                rate=self._settings.GROQ_REQUESTS_PER_MINUTE / 60.0,
                capacity=self._settings.GROQ_REQUESTS_PER_MINUTE // 2,
            ),
            "openai": TokenBucket(rate=60.0 / 60.0, capacity=30),  # 60 RPM
            "ollama": TokenBucket(rate=100.0 / 60.0, capacity=50),  # effectively unlimited locally
        }
        
        # HTTP clients (lazy-initialized)
        self._groq_client = None      # groq.AsyncGroq
        self._openai_client = None    # openai.AsyncOpenAI
        self._ollama_url = self._settings.OLLAMA_BASE_URL
        
        # Cost tracking (approximate)
        self._cost_per_1k_input: Dict[str, float] = {
            "groq": 0.00059,
            "openai": 0.00015,   # gpt-4o-mini
            "ollama": 0.0,
        }
        self._cost_per_1k_output: Dict[str, float] = {
            "groq": 0.00079,
            "openai": 0.00060,
            "ollama": 0.0,
        }
    
    # ── Provider Selection ───────────────────────────────────────────────────
    
    def _get_provider_order(self) -> List[str]:
        """
        Return providers in priority order, skipping circuit-broken providers.
        
        Respects API key availability:
        - If GROQ_API_KEY is set: groq → openai → ollama
        - If only OPENAI_API_KEY: openai → ollama
        - If neither: ollama only
        """
        order: List[str] = []
        
        if self._settings.GROQ_API_KEY:
            order.append("groq")
        if self._settings.OPENAI_API_KEY:
            order.append("openai")
        order.append("ollama")  # Always available (local)
        
        return order
    
    async def _is_circuit_open(self, provider: str) -> bool:
        """Check if a provider is circuit-broken."""
        async with self._cb_lock:
            cb = self._circuit_breakers.get(provider)
            if cb is None:
                return True
            
            if cb.state == CircuitState.OPEN:
                elapsed = time.monotonic() - cb.open_time
                if elapsed > self._settings.CIRCUIT_BREAKER_RESET_SECONDS:
                    cb.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {provider} → HALF_OPEN after {elapsed:.0f}s")
                    return False
                return True
            
            return False
    
    async def _record_failure(self, provider: str) -> None:
        """Record a failure and potentially trip the circuit breaker."""
        async with self._cb_lock:
            cb = self._circuit_breakers.get(provider)
            if cb is None:
                return
            
            cb.failure_count += 1
            cb.last_failure_time = time.monotonic()
            
            if cb.failure_count >= self._settings.CIRCUIT_BREAKER_FAILURES:
                cb.state = CircuitState.OPEN
                cb.open_time = time.monotonic()
                logger.warning(
                    f"Circuit breaker {provider} → OPEN after {cb.failure_count} failures"
                )
    
    async def _record_success(self, provider: str) -> None:
        """Reset circuit breaker on successful call."""
        async with self._cb_lock:
            cb = self._circuit_breakers.get(provider)
            if cb is None:
                return
            
            if cb.state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit breaker {provider} → CLOSED (recovered)")
            
            cb.state = CircuitState.CLOSED
            cb.failure_count = 0
    
    # ── LLM Call Methods ─────────────────────────────────────────────────────
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        prefer_provider: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send messages to the LLM, with automatic fallback.
        
        Args:
            messages: Chat messages [{"role": "user"/"system"/"assistant", "content": str}]
            temperature: Overrides the default LLM_TEMPERATURE setting.
            max_tokens: Overrides the default LLM_MAX_TOKENS setting.
            response_format: {"type": "json_object"} for structured output.
            prefer_provider: Force a specific provider (bypasses circuit breaker check).
        
        Returns:
            LLMResponse with the model's output.
        
        Raises:
            RuntimeError: If ALL providers fail simultaneously (should not happen
                         since Ollama is local and always available).
        """
        temp = temperature if temperature is not None else self._settings.LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else self._settings.LLM_MAX_TOKENS
        start = time.perf_counter()
        
        provider_order = self._get_provider_order()
        
        # If a specific provider is preferred, try it first
        if prefer_provider and prefer_provider in provider_order:
            provider_order.remove(prefer_provider)
            provider_order.insert(0, prefer_provider)
        
        last_error: Optional[str] = None
        last_provider: Optional[str] = None
        
        for provider in provider_order:
            # Skip circuit-broken providers (unless explicitly preferred)
            if provider != prefer_provider and await self._is_circuit_open(provider):
                logger.debug(f"Skipping {provider} — circuit is OPEN")
                continue
            
            # Rate limit
            limiter = self._rate_limiters.get(provider)
            if limiter:
                acquired = await limiter.acquire()
                if not acquired:
                    logger.warning(f"Rate limited on {provider}, trying next")
                    continue
            
            try:
                response = await self._call_provider(
                    provider=provider,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    response_format=response_format,
                )
                
                # Record success
                await self._record_success(provider)
                
                # Calculate cost
                duration = (time.perf_counter() - start) * 1000
                cost = (
                    response.input_tokens / 1000.0 * self._cost_per_1k_input.get(provider, 0)
                    + response.output_tokens / 1000.0 * self._cost_per_1k_output.get(provider, 0)
                )
                
                response.duration_ms = duration
                response.cost_usd = cost
                
                return response
                
            except Exception as e:
                last_error = str(e)
                last_provider = provider
                await self._record_failure(provider)
                logger.warning(f"{provider} failed: {e}. Trying next provider.")
        
        # All providers failed
        duration = (time.perf_counter() - start) * 1000
        return LLMResponse(
            content="",
            provider=last_provider or "none",
            model="none",
            success=False,
            error=last_error or "All LLM providers exhausted",
            duration_ms=duration,
        )
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the LLM with automatic fallback.
        
        Yields:
            Token strings as they arrive from the provider.
        
        Note:
            Unlike generate(), this does NOT fall back between providers mid-stream.
            If the primary provider fails during streaming, the error is yielded
            and iteration stops.
        """
        temp = temperature if temperature is not None else self._settings.LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else self._settings.LLM_MAX_TOKENS
        
        provider_order = self._get_provider_order()
        
        for provider in provider_order:
            if await self._is_circuit_open(provider):
                continue
            
            try:
                async for token in self._stream_provider(
                    provider=provider,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                ):
                    yield token
                await self._record_success(provider)
                return  # Successful stream — exit
            
            except Exception as e:
                await self._record_failure(provider)
                logger.warning(f"Streaming {provider} failed: {e}. Trying next.")
        
        # All failed
        yield f"[All LLM providers unavailable: {last_error}]"
    
    # ── Internal Provider Calls ─────────────────────────────────────────────
    
    async def _call_provider(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Route to the appropriate provider implementation."""
        if provider == "groq":
            return await self._call_groq(messages, temperature, max_tokens, response_format)
        elif provider == "openai":
            return await self._call_openai(messages, temperature, max_tokens, response_format)
        elif provider == "ollama":
            return await self._call_ollama(messages, temperature, max_tokens, response_format)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _stream_provider(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream from the appropriate provider."""
        if provider == "groq":
            async for token in self._stream_groq(messages, temperature, max_tokens):
                yield token
        elif provider == "openai":
            async for token in self._stream_openai(messages, temperature, max_tokens):
                yield token
        elif provider == "ollama":
            async for token in self._stream_ollama(messages, temperature, max_tokens):
                yield token
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    # ─── Groq ────────────────────────────────────────────────────────────────
    
    async def _get_groq_client(self):
        """Lazy import and initialize Groq client."""
        if self._groq_client is None:
            from groq import AsyncGroq
            self._groq_client = AsyncGroq(
                api_key=self._settings.GROQ_API_KEY,
                timeout=self._settings.LLM_REQUEST_TIMEOUT,
            )
        return self._groq_client
    
    async def _call_groq(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        client = await self._get_groq_client()
        kwargs = dict(
            model=self._settings.PRIMARY_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format
        
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        
        return LLMResponse(
            content=choice.message.content or "",
            provider="groq",
            model=self._settings.PRIMARY_MODEL,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
    
    async def _stream_groq(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        client = await self._get_groq_client()
        stream = await client.chat.completions.create(
            model=self._settings.PRIMARY_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
    
    # ─── OpenAI ──────────────────────────────────────────────────────────────
    
    async def _get_openai_client(self):
        """Lazy import and initialize OpenAI client."""
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(
                api_key=self._settings.OPENAI_API_KEY,
                timeout=self._settings.LLM_REQUEST_TIMEOUT,
            )
        return self._openai_client
    
    async def _call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        client = await self._get_openai_client()
        kwargs = dict(
            model=self._settings.FALLBACK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format
        
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        
        return LLMResponse(
            content=choice.message.content or "",
            provider="openai",
            model=self._settings.FALLBACK_MODEL,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
    
    async def _stream_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        client = await self._get_openai_client()
        stream = await client.chat.completions.create(
            model=self._settings.FALLBACK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
    
    # ─── Ollama (Local) ──────────────────────────────────────────────────────
    
    async def _call_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Call Ollama using the Python client for better compatibility."""
        try:
            from ollama import Client, ResponseError
            
            client = Client(host=self._ollama_url)
            
            # Convert messages to prompt format for compatibility
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"
            
            # Use the generate endpoint which is more stable
            response = client.generate(
                model=self._settings.OLLAMA_MODEL,
                prompt=prompt,
                temperature=temperature,
                num_predict=max_tokens,
                stream=False,
            )
            
            content = response.get("response", "")
            
            # Extract token counts
            input_tokens = response.get("prompt_eval_count", 0)
            output_tokens = response.get("eval_count", 0)
            
            return LLMResponse(
                content=content or "",
                provider="ollama",
                model=self._settings.OLLAMA_MODEL,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            raise RuntimeError(f"Ollama unavailable: {str(e)}")
    
    async def _stream_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream responses from Ollama."""
        try:
            from ollama import Client
            
            client = Client(host=self._ollama_url)
            
            # Convert messages to prompt format
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"
            
            # Stream from Ollama
            response = client.generate(
                model=self._settings.OLLAMA_MODEL,
                prompt=prompt,
                temperature=temperature,
                num_predict=max_tokens,
                stream=True,
            )
            
            for chunk in response:
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
        
        except Exception as e:
            logger.error("Ollama streaming failed: %s", e)
            yield f"[Ollama streaming error: {str(e)}]"


@lru_cache(maxsize=1)
def get_llm_router() -> LLMRouter:
    """
    Return the singleton LLMRouter instance.
    
    The router is initialized once and reused across all sessions.
    Client connections are lazy-initialized on first use.
    
    Returns:
        LLMRouter: The singleton router instance.
    """
    return LLMRouter()


# ─── Circuit Breaker States ───────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation — requests are sent
    OPEN = "open"            # Failing — requests are blocked
    HALF_OPEN = "half_open"  # Testing if provider recovered


@dataclass
class CircuitBreaker:
    """Per-provider circuit breaker state."""
    provider: str
    failure_count: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: float = 0.0
    open_time: float = 0.0


# ─── LLM Response Types ───────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    success: bool = True
    error: Optional[str] = None
    streamed: bool = False


# ─── Rate Limiter Token Bucket ────────────────────────────────────────────────

class TokenBucket:
    """
    Simple token bucket rate limiter.
    
    Refills at rate tokens/second up to capacity.
    Thread-safe (asyncio.Lock).
    """
    
    def __init__(self, rate: float, capacity: int):
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1, timeout: float = 10.0) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum seconds to wait for tokens.
        
        Returns:
            True if tokens acquired, False if timeout.
        """
        start = time.monotonic()
        async with self._lock:
            while time.monotonic() - start < timeout:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                await asyncio.sleep(0.05)
            return False
    
    def _refill(self) -> None:
        elapsed = time.monotonic() - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = time.monotonic()


# ─── Main LLM Router ──────────────────────────────────────────────────────────

class LLMRouter:
    """
    Triple-provider LLM router with circuit breaker, rate limiting, and metrics.
    
    Usage:
        router = get_llm_router()
        response = await router.generate([{"role": "user", "content": "Hello"}])
        
        # Streaming:
        async for chunk in router.stream([{"role": "user", "content": "Write a poem"}]):
            print(chunk)
    """
    
    def __init__(self):
        self._settings = get_settings()
        self._audit_logger = get_audit_logger()
        
        # Circuit breakers per provider
        self._circuit_breakers: Dict[str, CircuitBreaker] = {
            "groq": CircuitBreaker("groq"),
            "openai": CircuitBreaker("openai"),
            "ollama": CircuitBreaker("ollama"),
        }
        self._cb_lock = asyncio.Lock()
        
        # Rate limiters
        self._rate_limiters: Dict[str, TokenBucket] = {
            "groq": TokenBucket(
                rate=self._settings.GROQ_REQUESTS_PER_MINUTE / 60.0,
                capacity=max(1, self._settings.GROQ_REQUESTS_PER_MINUTE // 2),
            ),
            "openai": TokenBucket(rate=1.0, capacity=30),
            "ollama": TokenBucket(rate=2.0, capacity=50),
        }
        
        # HTTP clients (lazy-initialized)
        self._groq_client = None
        self._openai_client = None
        self._ollama_url = self._settings.OLLAMA_BASE_URL
        
        # Cost tracking (approximate USD per 1K tokens)
        self._cost_per_1k_input: Dict[str, float] = {
            "groq": 0.00059,
            "openai": 0.00015,
            "ollama": 0.0,
        }
        self._cost_per_1k_output: Dict[str, float] = {
            "groq": 0.00079,
            "openai": 0.00060,
            "ollama": 0.0,
        }
    
    # ── Provider Selection ───────────────────────────────────────────────────
    
    def _get_provider_order(self) -> List[str]:
        """
        Return providers in priority order, skipping unavailable ones.
        
        Rules:
        - If GROQ_API_KEY is set: groq → openai → ollama
        - If only OPENAI_API_KEY: openai → ollama
        - If neither: ollama only
        """
        order: List[str] = []
        if self._settings.GROQ_API_KEY:
            order.append("groq")
        if self._settings.OPENAI_API_KEY:
            order.append("openai")
        order.append("ollama")
        return order
    
    async def _is_circuit_open(self, provider: str) -> bool:
        """Check if a provider is circuit-broken and attempt half-open recovery."""
        async with self._cb_lock:
            cb = self._circuit_breakers.get(provider)
            if cb is None:
                return True
            if cb.state == CircuitState.OPEN:
                elapsed = time.monotonic() - cb.open_time
                if elapsed > self._settings.CIRCUIT_BREAKER_RESET_SECONDS:
                    cb.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker %s → HALF_OPEN after %.0fs", provider, elapsed)
                    return False
                return True
            return False
    
    async def _record_failure(self, provider: str) -> None:
        """Record a failure and potentially trip the circuit breaker."""
        async with self._cb_lock:
            cb = self._circuit_breakers.get(provider)
            if cb is None:
                return
            cb.failure_count += 1
            cb.last_failure_time = time.monotonic()
            if cb.failure_count >= self._settings.CIRCUIT_BREAKER_FAILURES:
                cb.state = CircuitState.OPEN
                cb.open_time = time.monotonic()
                logger.warning("Circuit breaker %s → OPEN (%d failures)", provider, cb.failure_count)
    
    async def _record_success(self, provider: str) -> None:
        """Reset circuit breaker on successful call."""
        async with self._cb_lock:
            cb = self._circuit_breakers.get(provider)
            if cb is None:
                return
            if cb.state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker %s → CLOSED (recovered)", provider)
            cb.state = CircuitState.CLOSED
            cb.failure_count = 0
    
    # ── Public API ───────────────────────────────────────────────────────────
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        prefer_provider: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send messages to the LLM with automatic fallback across providers.
        
        Args:
            messages: Chat messages.
            temperature: Overrides default LLM_TEMPERATURE.
            max_tokens: Overrides default LLM_MAX_TOKENS.
            response_format: {"type": "json_object"} for structured output.
            prefer_provider: Force a specific provider.
        
        Returns:
            LLMResponse with content and metadata.
        """
        temp = temperature if temperature is not None else self._settings.LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else self._settings.LLM_MAX_TOKENS
        start = time.perf_counter()
        
        provider_order = self._get_provider_order()
        if prefer_provider and prefer_provider in provider_order:
            provider_order.remove(prefer_provider)
            provider_order.insert(0, prefer_provider)
        
        last_error: Optional[str] = None
        last_provider: Optional[str] = None
        
        for provider in provider_order:
            if provider != prefer_provider and await self._is_circuit_open(provider):
                continue
            
            limiter = self._rate_limiters.get(provider)
            if limiter:
                acquired = await limiter.acquire()
                if not acquired:
                    logger.warning("Rate limited on %s, trying next", provider)
                    continue
            
            try:
                response = await self._call_provider(
                    provider=provider,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    response_format=response_format,
                )
                await self._record_success(provider)
                duration = (time.perf_counter() - start) * 1000
                cost = (
                    response.input_tokens / 1000.0 * self._cost_per_1k_input.get(provider, 0)
                    + response.output_tokens / 1000.0 * self._cost_per_1k_output.get(provider, 0)
                )
                response.duration_ms = duration
                response.cost_usd = cost
                return response
            except Exception as e:
                last_error = str(e)
                last_provider = provider
                await self._record_failure(provider)
                logger.warning("%s failed: %s", provider, e)
        
        duration = (time.perf_counter() - start) * 1000
        return LLMResponse(
            content="",
            provider=last_provider or "none",
            model="none",
            success=False,
            error=last_error or "All LLM providers exhausted",
            duration_ms=duration,
        )
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the LLM with automatic fallback.
        
        Unlike generate(), this does NOT fall back mid-stream.
        If the primary provider fails during streaming, the first available
        provider is used for the entire stream.
        
        Yields:
            Token strings as they arrive.
        """
        temp = temperature if temperature is not None else self._settings.LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else self._settings.LLM_MAX_TOKENS
        
        last_error = "No providers available"
        for provider in self._get_provider_order():
            if await self._is_circuit_open(provider):
                continue
            try:
                async for token in self._stream_provider(
                    provider=provider, messages=messages,
                    temperature=temp, max_tokens=max_tok,
                ):
                    yield token
                await self._record_success(provider)
                return
            except Exception as e:
                last_error = str(e)
                await self._record_failure(provider)
                logger.warning("Streaming %s failed: %s", provider, e)
        
        yield f"\n[NEXUS WARNING: All LLM providers unavailable — {last_error}]"
    
    # ── Internal Provider Dispatch ──────────────────────────────────────────
    
    async def _call_provider(
        self, provider: str, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        if provider == "groq":
            return await self._call_groq(messages, temperature, max_tokens, response_format)
        elif provider == "openai":
            return await self._call_openai(messages, temperature, max_tokens, response_format)
        elif provider == "ollama":
            return await self._call_ollama(messages, temperature, max_tokens, response_format)
        raise ValueError(f"Unknown provider: {provider}")
    
    async def _stream_provider(
        self, provider: str, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
    ) -> AsyncIterator[str]:
        if provider == "groq":
            async for t in self._stream_groq(messages, temperature, max_tokens):
                yield t
        elif provider == "openai":
            async for t in self._stream_openai(messages, temperature, max_tokens):
                yield t
        elif provider == "ollama":
            async for t in self._stream_ollama(messages, temperature, max_tokens):
                yield t
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    # ── Groq Implementation ─────────────────────────────────────────────────
    
    async def _get_groq_client(self):
        if self._groq_client is None:
            from groq import AsyncGroq
            self._groq_client = AsyncGroq(
                api_key=self._settings.GROQ_API_KEY,
                timeout=self._settings.LLM_REQUEST_TIMEOUT,
            )
        return self._groq_client
    
    async def _call_groq(
        self, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        client = await self._get_groq_client()
        kwargs: Dict[str, Any] = dict(
            model=self._settings.PRIMARY_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            provider="groq",
            model=self._settings.PRIMARY_MODEL,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
    
    async def _stream_groq(
        self, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
    ) -> AsyncIterator[str]:
        client = await self._get_groq_client()
        stream = await client.chat.completions.create(
            model=self._settings.PRIMARY_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
    
    # ── OpenAI Implementation ───────────────────────────────────────────────
    
    async def _get_openai_client(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(
                api_key=self._settings.OPENAI_API_KEY,
                timeout=self._settings.LLM_REQUEST_TIMEOUT,
            )
        return self._openai_client
    
    async def _call_openai(
        self, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        client = await self._get_openai_client()
        kwargs: Dict[str, Any] = dict(
            model=self._settings.FALLBACK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            provider="openai",
            model=self._settings.FALLBACK_MODEL,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
    
    async def _stream_openai(
        self, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
    ) -> AsyncIterator[str]:
        client = await self._get_openai_client()
        stream = await client.chat.completions.create(
            model=self._settings.FALLBACK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
    
    # ── Ollama Implementation ───────────────────────────────────────────────
    
    async def _call_ollama(
        self, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        import httpx
        payload: Dict[str, Any] = {
            "model": self._settings.OLLAMA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self._settings.LLM_REQUEST_TIMEOUT) as client:
            resp = await client.post(f"{self._ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        return LLMResponse(
            content=content or "",
            provider="ollama",
            model=self._settings.OLLAMA_MODEL,
            input_tokens=prompt_eval_count or 0,
            output_tokens=eval_count or 0,
        )
    
    async def _stream_ollama(
        self, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
    ) -> AsyncIterator[str]:
        import httpx
        import json
        payload: Dict[str, Any] = {
            "model": self._settings.OLLAMA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self._settings.LLM_REQUEST_TIMEOUT) as client:
            async with client.stream("POST", f"{self._ollama_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    def estimate_cost(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate the USD cost of an LLM call.
        
        Args:
            provider: "groq", "openai", or "ollama".
            input_tokens: Number of prompt tokens.
            output_tokens: Number of completion tokens.
        
        Returns:
            Estimated cost in USD.
        """
        return (
            input_tokens / 1000.0 * self._cost_per_1k_input.get(provider, 0)
            + output_tokens / 1000.0 * self._cost_per_1k_output.get(provider, 0)
        )


@lru_cache(maxsize=1)
def get_llm_router() -> LLMRouter:
    """
    Return the singleton LLMRouter instance.
    
    The router is initialized once and reused across all sessions.
    Client connections are lazy-initialized on first use.
    
    Returns:
        LLMRouter: The singleton router instance.
    """
    return LLMRouter()
