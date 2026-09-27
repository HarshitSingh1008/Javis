"""
NEXUS AI v4.0 — Local-first LLM router with Ollama (primary) and Groq (optional fallback).

Routing hierarchy:
  1. Ollama (local, unlimited, free) — primary for all roles
  2. Groq (cloud, rate-limited) — optional fallback if API key provided

Fallback chain: ollama → groq (circuit-breaker protected).
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, AsyncIterator, List, Dict, Any, Literal, Type, TypeVar
from functools import lru_cache

from nexus_config.settings import get_settings
from nexus_config.audit_logger import get_audit_logger, audited
from nexus_brain.telemetry import get_metrics_collector, track_request
from nexus_brain.testable_singleton import TestableSingleton

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    ValidationError = Exception


logger = logging.getLogger("nexus.llm_router")


# Type variable for structured output
T = TypeVar('T', bound=BaseModel)


# ─── Circuit Breaker States ───────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation — requests are sent
    OPEN = "open"          # Failing — requests are blocked
    HALF_OPEN = "half_open"  # Testing if provider recovered


@dataclass
class CircuitBreaker:
    """Per-provider circuit breaker state."""
    provider: str
    failure_count: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: float = 0.0
    open_time: float = 0.0


# ─── LLM Role Enum ───────────────────────────────────────────────────────────

LLMRole = Literal["planner", "automation", "voice"]


# ─── LLM Response Types ───────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str
    provider: str                             # "ollama" | "groq"
    model: str                                # Model name used
    role: str = "planner"                     # Role used: "planner" | "automation" | "voice"
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    cost_usd: float = 0.0                     # Always 0 for Ollama
    success: bool = True
    error: Optional[str] = None
    streamed: bool = False
    parsed: Optional[Any] = None              # Parsed structured output (if applicable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "role": self.role,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
            "success": self.success,
            "error": self.error,
            "streamed": self.streamed,
            "parsed": self.parsed,
        }


# ─── Rate Limiter Token Bucket ────────────────────────────────────────────────

class TokenBucket:
    """
    Simple token bucket rate limiter.

    Refills at rate tokens/second up to capacity.
    """

    def __init__(self, rate: float, capacity: int):
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1, timeout: float = 10.0) -> bool:
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

class LLMRouter(TestableSingleton):
    """
    Local-first LLM router with Ollama (primary) and Groq (optional fallback).

    Usage:
        router = get_llm_router()

        # Auto-select model by role (uses Ollama models)
        response = await router.generate(messages, role="planner")

        # Force a specific model
        response = await router.generate(messages, model="llama3.2:3b-instruct-q4_K_M")

        # Streaming
        async for chunk in router.stream(messages, role="voice"):
            print(chunk)
    """

    def __init__(self):
        self._settings = get_settings()
        self._audit_logger = get_audit_logger()

        # Circuit breakers per provider
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._cb_lock = asyncio.Lock()

        # Rate limiter for Groq only (Ollama has no rate limits)
        self._groq_rate_limiter = TokenBucket(
            rate=self._settings.GROQ_REQUESTS_PER_MINUTE / 60.0,
            capacity=max(1, self._settings.GROQ_REQUESTS_PER_MINUTE // 2),
        )

        # HTTP clients (lazy-initialized)
        self._groq_client = None
        self._ollama_client = None

        # Role → Ollama model mapping (local models)
        self._role_models: Dict[str, str] = {
            "planner": self._settings.OLLAMA_MODEL,
            "automation": self._settings.OLLAMA_MODEL,
            "voice": self._settings.OLLAMA_INTENT_MODEL,  # Ultra-fast model for voice
        }

        # Fallback order: Ollama first, then Groq if available
        self._fallback_order = [
            self._settings.OLLAMA_MODEL,
            self._settings.PLANNER_MODEL,  # Groq planner model
            self._settings.AUTOMATION_MODEL,  # Groq automation model
            self._settings.VOICE_MODEL,  # Groq voice model
        ]

    def _get_cb(self, model: str) -> CircuitBreaker:
        """Get or create circuit breaker for a model."""
        if model not in self._circuit_breakers:
            self._circuit_breakers[model] = CircuitBreaker(provider=model)
        return self._circuit_breakers[model]

    async def _is_circuit_open(self, model: str) -> bool:
        """Check if a model is circuit-broken and attempt half-open recovery."""
        async with self._cb_lock:
            cb = self._get_cb(model)
            if cb.state == CircuitState.OPEN:
                elapsed = time.monotonic() - cb.open_time
                if elapsed > self._settings.CIRCUIT_BREAKER_RESET_SECONDS:
                    cb.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker %s -> HALF_OPEN after %.0fs", model, elapsed)
                    return False
                return True
            return False

    async def _record_failure(self, model: str) -> None:
        async with self._cb_lock:
            cb = self._get_cb(model)
            cb.failure_count += 1
            cb.last_failure_time = time.monotonic()
            if cb.failure_count >= self._settings.CIRCUIT_BREAKER_FAILURES:
                cb.state = CircuitState.OPEN
                cb.open_time = time.monotonic()
                logger.warning("Circuit breaker %s -> OPEN (%d failures)", model, cb.failure_count)

    async def _record_success(self, model: str) -> None:
        async with self._cb_lock:
            cb = self._get_cb(model)
            if cb.state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker %s -> CLOSED (recovered)", model)
            cb.state = CircuitState.CLOSED
            cb.failure_count = 0

    # ── Model Selection ──────────────────────────────────────────────────────

    def _resolve_model(self, role: Optional[str] = None, model: Optional[str] = None) -> str:
        """
        Resolve which model to use.

        Priority:
          1. Explicit `model` parameter
          2. `role` parameter → role-to-model mapping
          3. Default: planner (Ollama)
        """
        if model:
            return model
        if role and role in self._role_models:
            return self._role_models[role]
        return self._settings.OLLAMA_MODEL

    def _get_model_order(self, role: Optional[str] = None) -> List[str]:
        """
        Return models in priority order for a given role, skipping circuit-broken ones.

        Fallback chain: role model → planner → automation → voice
        """
        primary = self._resolve_model(role)
        order = [primary]
        for m in self._fallback_order:
            if m not in order:
                order.append(m)
        return order

    def _is_groq_model(self, model: str) -> bool:
        """Check if a model name is a Groq model."""
        groq_models = {
            self._settings.PLANNER_MODEL,
            self._settings.AUTOMATION_MODEL,
            self._settings.VOICE_MODEL,
        }
        return model in groq_models

    # ── Public API ───────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        role: Optional[str] = None,
        model: Optional[str] = None,
        prefer_provider: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send messages to the LLM with automatic model fallback (Ollama → Groq).

        Args:
            messages: Chat messages.
            temperature: Overrides default LLM_TEMPERATURE.
            max_tokens: Overrides default LLM_MAX_TOKENS.
            response_format: {"type": "json_object"} for structured output.
            role: "planner" | "automation" | "voice" — selects model automatically.
            model: Force a specific model name (overrides role).
            prefer_provider: "ollama" | "groq" — prefers local or cloud.

        Returns:
            LLMResponse with content and metadata.
        """
        temp = temperature if temperature is not None else self._settings.LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else self._settings.LLM_MAX_TOKENS
        start = time.perf_counter()
        
        # Telemetry
        metrics = get_metrics_collector()
        request_tags = {"role": role or "planner", "prefer_provider": prefer_provider or "auto"}
        
        model_order = self._get_model_order(role)

        # Filter by preferred provider if specified
        if prefer_provider == "ollama":
            model_order = [m for m in model_order if not self._is_groq_model(m)]
        elif prefer_provider == "groq":
            model_order = [m for m in model_order if self._is_groq_model(m)]

        last_error: Optional[str] = None
        models_tried = 0

        for model_name in model_order:
            models_tried += 1
            if await self._is_circuit_open(model_name):
                logger.debug("Skipping %s — circuit is OPEN", model_name)
                continue

            # Apply rate limiting only for Groq models
            if self._is_groq_model(model_name):
                acquired = await self._groq_rate_limiter.acquire()
                if not acquired:
                    logger.warning("Groq rate limited, trying next model")
                    continue

            try:
                if self._is_groq_model(model_name):
                    response = await self._call_groq(
                        model=model_name,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tok,
                        response_format=response_format,
                    )
                else:
                    response = await self._call_ollama(
                        model=model_name,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tok,
                        response_format=response_format,
                    )

                await self._record_success(model_name)

                # If content is empty, try next model
                if not response.content.strip() and len(model_order) > 1:
                    logger.debug("Empty response from %s, trying next model", model_name)
                    continue

                duration = (time.perf_counter() - start) * 1000
                response.duration_ms = duration
                response.cost_usd = 0.0
                
                # Telemetry
                metrics.increment("llm.calls", 1, {**request_tags, "model": model_name, "provider": response.provider, "success": "true"})
                metrics.observe("llm.latency_ms", duration, {**request_tags, "model": model_name, "provider": response.provider})
                if response.input_tokens:
                    metrics.observe("llm.input_tokens", response.input_tokens, {**request_tags, "model": model_name})
                if response.output_tokens:
                    metrics.observe("llm.output_tokens", response.output_tokens, {**request_tags, "model": model_name})
                
                return response
            except Exception as e:
                last_error = str(e)
                await self._record_failure(model_name)
                logger.warning("%s failed: %s. Trying next model.", model_name, e)
                
                # Telemetry
                metrics.increment("llm.calls", 1, {**request_tags, "model": model_name, "success": "false", "error": type(e).__name__})

        duration = (time.perf_counter() - start) * 1000
        metrics.increment("llm.calls", 1, {**request_tags, "success": "false", "models_tried": str(models_tried)})
        metrics.observe("llm.latency_ms", duration, {**request_tags, "success": "false"})
        
        return LLMResponse(
            content="",
            provider="none",
            model="none",
            role=role or "planner",
            success=False,
            error=last_error or "All models exhausted",
            duration_ms=duration,
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        role: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the LLM with automatic model fallback.

        If the primary model fails during streaming, the next available
        model is used for the entire stream.

        Yields:
            Token strings as they arrive.
        """
        temp = temperature if temperature is not None else self._settings.LLM_TEMPERATURE
        max_tok = max_tokens if max_tokens is not None else self._settings.LLM_MAX_TOKENS

        model_order = self._get_model_order(role)
        last_error = "No models available"

        for model_name in model_order:
            if await self._is_circuit_open(model_name):
                continue

            if self._is_groq_model(model_name):
                acquired = await self._groq_rate_limiter.acquire()
                if not acquired:
                    continue

            try:
                if self._is_groq_model(model_name):
                    async for token in self._stream_groq(
                        model=model_name, messages=messages,
                        temperature=temp, max_tokens=max_tok,
                    ):
                        yield token
                else:
                    async for token in self._stream_ollama(
                        model=model_name, messages=messages,
                        temperature=temp, max_tokens=max_tok,
                    ):
                        yield token

                await self._record_success(model_name)
                return
            except Exception as e:
                last_error = str(e)
                await self._record_failure(model_name)
                logger.warning("Streaming %s failed: %s", model_name, e)

        yield f"\n[NEXUS WARNING: All models unavailable — {last_error}]"

    # ── Ollama Implementation (Primary) ──────────────────────────────────────

    async def _get_ollama_client(self):
        if self._ollama_client is None:
            from ollama import AsyncClient
            self._ollama_client = AsyncClient(
                host=self._settings.OLLAMA_BASE_URL,
                timeout=self._settings.LLM_REQUEST_TIMEOUT,
            )
        return self._ollama_client

    async def _call_ollama(
        self, model: str, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        logger.info("OLLAMA call starting (model=%s, %d msgs)", model, len(messages))
        call_start = time.perf_counter()
        client = await self._get_ollama_client()

        # Convert messages to Ollama format
        ollama_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        if response_format and response_format.get("type") == "json_object":
            options["format"] = "json"

        response = await client.chat(
            model=model,
            messages=ollama_messages,
            options=options,
            stream=False,
        )

        duration = (time.perf_counter() - call_start) * 1000
        content = response.get("message", {}).get("content", "")

        # Ollama doesn't provide token counts in non-streaming response
        # Estimate based on character count (rough approximation: 1 token ≈ 4 chars)
        input_chars = sum(len(m["content"]) for m in messages)
        output_chars = len(content)
        input_tokens = max(1, input_chars // 4)
        output_tokens = max(1, output_chars // 4)

        logger.info("OLLAMA call completed in %.0fms (est. %d+%d tokens) [%s]",
                    duration, input_tokens, output_tokens, model)

        return LLMResponse(
            content=content,
            provider="ollama",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _stream_ollama(
        self, model: str, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
    ) -> AsyncIterator[str]:
        logger.info("OLLAMA stream starting (model=%s)", model)
        client = await self._get_ollama_client()

        ollama_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        stream = await client.chat(
            model=model,
            messages=ollama_messages,
            options=options,
            stream=True,
        )

        token_count = 0
        async for chunk in stream:
            if chunk.get("message", {}).get("content"):
                token_count += 1
                yield chunk["message"]["content"]

        logger.info("OLLAMA stream completed (%d chunks) [%s]", token_count, model)

    # ── Groq Implementation (Optional Fallback) ──────────────────────────────

    async def _get_groq_client(self):
        if self._groq_client is None:
            from groq import AsyncGroq
            self._groq_client = AsyncGroq(
                api_key=self._settings.GROQ_API_KEY,
                timeout=self._settings.LLM_REQUEST_TIMEOUT,
            )
        return self._groq_client

    # Groq model max token limits (safety caps)
    _GROQ_MAX_TOKENS: Dict[str, int] = {
        "qwen/qwen3.8-27b": 8192,
        "llama3-70b-8192": 8192,
        "llama3-8b-8192": 8192,
        "mixtral-8x7b-32768": 32768,
        "gemma-7b-it": 8192,
    }

    def _get_groq_max_tokens(self, model: str, requested: int) -> int:
        """Get safe max tokens for Groq model."""
        cap = self._GROQ_MAX_TOKENS.get(model, 8192)
        return min(requested, cap)

    async def _call_groq(
        self, model: str, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        logger.info("GROQ call starting (model=%s, %d msgs)", model, len(messages))
        call_start = time.perf_counter()
        client = await self._get_groq_client()
        
        # Apply safety cap for Groq models
        safe_max_tokens = self._get_groq_max_tokens(model, max_tokens)
        if safe_max_tokens != max_tokens:
            logger.info("Capping max_tokens for %s: %d -> %d", model, max_tokens, safe_max_tokens)
        
        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=safe_max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        duration = (time.perf_counter() - call_start) * 1000

        # GPT-OSS reasoning models may return answer in reasoning field
        content = choice.message.content or ""
        if not content and hasattr(choice.message, "reasoning") and choice.message.reasoning:
            content = choice.message.reasoning

        logger.info("GROQ call completed in %.0fms (%d+%d tokens) [%s]",
                     duration, usage.prompt_tokens if usage else 0,
                     usage.completion_tokens if usage else 0, model)

        return LLMResponse(
            content=content,
            provider="groq",
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def _stream_groq(
        self, model: str, messages: List[Dict[str, str]],
        temperature: float, max_tokens: int,
    ) -> AsyncIterator[str]:
        logger.info("GROQ stream starting (model=%s)", model)
        client = await self._get_groq_client()
        
        # Apply safety cap for Groq models
        safe_max_tokens = self._get_groq_max_tokens(model, max_tokens)
        if safe_max_tokens != max_tokens:
            logger.info("Capping max_tokens for %s: %d -> %d", model, max_tokens, safe_max_tokens)
        
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=safe_max_tokens,
            stream=True,
        )
        token_count = 0
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    token_count += 1
                    yield delta.content
        logger.info("GROQ stream completed (%d chunks) [%s]", token_count, model)

    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        role: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 2,
    ) -> LLMResponse:
        """
        Generate a structured response parsed into a Pydantic model.
        
        Uses JSON mode with schema validation and automatic retry on parse failure.
        
        Args:
            messages: Chat messages.
            response_model: Pydantic model class to parse response into.
            temperature: Overrides default LLM_TEMPERATURE (set low for structured output).
            max_tokens: Overrides default LLM_MAX_TOKENS.
            role: "planner" | "automation" | "voice" — selects model automatically.
            model: Force a specific model name.
            max_retries: Maximum retries on parse failure.
        
        Returns:
            LLMResponse with parsed model in `parsed` field.
        """
        if not PYDANTIC_AVAILABLE:
            raise RuntimeError("Pydantic not available for structured output")
        
        # Get JSON schema from model
        schema = response_model.model_json_schema()
        
        # Add schema instruction to system message
        schema_instruction = (
            f"\n\nYou MUST respond with a valid JSON object matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            f"Do not include any explanation, markdown, or extra text."
        )
        
        # Inject schema instruction
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0].get("role") == "system":
            enhanced_messages[0]["content"] += schema_instruction
        else:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": f"You are a structured output generator.{schema_instruction}"
            })
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            response = await self.generate(
                messages=enhanced_messages,
                temperature=temperature if temperature is not None else 0.0,  # Low temp for structure
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                role=role,
                model=model,
            )
            
            if not response.success:
                last_error = response.error
                continue
            
            # Try to parse and validate
            try:
                parsed_data = json.loads(response.content)
                validated = response_model(**parsed_data)
                response.parsed = validated
                response.content = validated.model_dump_json()
                return response
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = f"Parse/validation failed: {e}"
                logger.warning("Structured output attempt %d failed: %s", attempt + 1, e)
                
                if attempt < max_retries:
                    # Add error feedback for retry
                    enhanced_messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    enhanced_messages.append({
                        "role": "user",
                        "content": f"The previous response was invalid JSON or didn't match the schema. Error: {e}. Please try again with valid JSON only."
                    })
                    continue
        
        # All retries exhausted
        return LLMResponse(
            content="",
            provider="none",
            model="none",
            role=role or "planner",
            success=False,
            error=last_error or "Structured output validation failed after retries",
        )

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate the USD cost of an LLM call. Returns 0 for Ollama (free)."""
        if not self._is_groq_model(model):
            return 0.0

        # Approximate Groq pricing (USD per 1K tokens)
        _GROQ_COST_1K_INPUT: Dict[str, float] = {
            "openai/gpt-oss-120b": 0.15,
            "openai/gpt-oss-20b": 0.075,
            "qwen/qwen3.8-27b": 0.80,
        }
        _GROQ_COST_1K_OUTPUT: Dict[str, float] = {
            "openai/gpt-oss-120b": 0.60,
            "openai/gpt-oss-20b": 0.30,
            "qwen/qwen3.8-27b": 4.00,
        }
        return (
            input_tokens / 1000.0 * _GROQ_COST_1K_INPUT.get(model, 0.0005)
            + output_tokens / 1000.0 * _GROQ_COST_1K_OUTPUT.get(model, 0.0007)
        )


def get_llm_router() -> LLMRouter:
    """
    Return the singleton LLMRouter instance.

    The router is initialized once and reused across all sessions.
    Client connections are lazy-initialized on first use.
    
    For testing, use SingletonMeta.reset(LLMRouter) or isolated_singletons(LLMRouter).

    Returns:
        LLMRouter: The singleton router instance.
    """
    return LLMRouter()


# For backward compatibility
def reset_llm_router():
    from nexus_brain.testable_singleton import SingletonMeta
    SingletonMeta.reset(LLMRouter)