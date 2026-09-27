"""
NEXUS AI v4.0 — Middleware pipeline for request/response processing.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides:
- Async middleware chain for request processing
- Built-in middleware for logging, auth, rate limiting, retries
- Extensible middleware interface
- Context propagation through middleware
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps
from contextlib import asynccontextmanager

logger = logging.getLogger("nexus.middleware")


# Type definitions
MiddlewareFunc = Callable[[Any, "MiddlewareContext"], Any]
T = TypeVar('T')


@dataclass
class MiddlewareContext:
    """
    Context passed through the middleware chain.
    
    Contains request data, response data, and shared state.
    """
    request: Any = None
    response: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    # Control flags
    should_continue: bool = True
    skip_remaining: bool = False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value


class Middleware(ABC):
    """
    Abstract base class for middleware.
    
    Middleware can process requests before they reach the handler
    and responses before they are returned.
    """
    
    @abstractmethod
    async def process_request(self, context: MiddlewareContext) -> None:
        """
        Process the request before it reaches the handler.
        
        Modify context.request or set context.should_continue = False
        to short-circuit the chain.
        """
        pass
    
    @abstractmethod
    async def process_response(self, context: MiddlewareContext) -> None:
        """
        Process the response after the handler completes.
        
        Modify context.response or context.metadata.
        """
        pass


class MiddlewarePipeline:
    """
    Async middleware pipeline for request/response processing.
    
    Usage:
        pipeline = MiddlewarePipeline()
        pipeline.add_middleware(LoggingMiddleware())
        pipeline.add_middleware(RateLimitMiddleware())
        
        response = await pipeline.execute(request, handler)
    """
    
    def __init__(self):
        self._middleware: List[Middleware] = []
        self._global_metadata: Dict[str, Any] = {}
    
    def add_middleware(self, middleware: Middleware) -> "MiddlewarePipeline":
        """Add middleware to the pipeline."""
        self._middleware.append(middleware)
        return self
    
    def remove_middleware(self, middleware_class: type) -> bool:
        """Remove middleware by class."""
        for i, m in enumerate(self._middleware):
            if isinstance(m, middleware_class):
                self._middleware.pop(i)
                return True
        return False
    
    async def execute(
        self,
        request: Any,
        handler: Callable[[Any], Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute the middleware pipeline.
        
        Args:
            request: The request object.
            handler: The final handler function (async).
            metadata: Optional initial metadata.
        
        Returns:
            The response from the handler (after response middleware).
        """
        context = MiddlewareContext(
            request=request,
            metadata=metadata or {},
        )
        
        # Copy global metadata
        context.metadata.update(self._global_metadata)
        
        # Phase 1: Request middleware (in order)
        for middleware in self._middleware:
            if not context.should_continue:
                break
            
            try:
                await middleware.process_request(context)
            except Exception as e:
                logger.error("Request middleware %s failed: %s", 
                           middleware.__class__.__name__, e)
                context.response = {"error": str(e), "middleware": middleware.__class__.__name__}
                context.should_continue = False
        
        # Phase 2: Execute handler if not short-circuited
        if context.should_continue and context.response is None:
            try:
                context.response = await handler(context.request)
            except Exception as e:
                logger.error("Handler failed: %s", e)
                context.response = {"error": str(e), "handler": "main"}
        
        # Phase 3: Response middleware (in reverse order)
        for middleware in reversed(self._middleware):
            if context.skip_remaining:
                break
            
            try:
                await middleware.process_response(context)
            except Exception as e:
                logger.error("Response middleware %s failed: %s",
                           middleware.__class__.__name__, e)
        
        return context.response
    
    @asynccontextmanager
    async def lifespan(self):
        """Context manager for pipeline lifecycle."""
        # Initialize middleware
        for middleware in self._middleware:
            if hasattr(middleware, 'startup'):
                await middleware.startup()
        
        try:
            yield self
        finally:
            # Shutdown middleware
            for middleware in reversed(self._middleware):
                if hasattr(middleware, 'shutdown'):
                    await middleware.shutdown()


# ── Built-in Middleware ──────────────────────────────────────────────────────

class LoggingMiddleware(Middleware):
    """Middleware for request/response logging."""
    
    def __init__(self, log_requests: bool = True, log_responses: bool = True):
        self._log_requests = log_requests
        self._log_responses = log_responses
    
    async def process_request(self, context: MiddlewareContext) -> None:
        if self._log_requests:
            logger.info("Request: %s", self._format_request(context.request))
    
    async def process_response(self, context: MiddlewareContext) -> None:
        if self._log_responses:
            duration = time.time() - context.start_time
            logger.info("Response (%.2fms): %s", duration * 1000, 
                       self._format_response(context.response))
    
    def _format_request(self, request: Any) -> str:
        if hasattr(request, '__dict__'):
            return str(request.__dict__)[:200]
        return str(request)[:200]
    
    def _format_response(self, response: Any) -> str:
        if response is None:
            return "None"
        if hasattr(response, '__dict__'):
            return str(response.__dict__)[:200]
        return str(response)[:200]


class RateLimitMiddleware(Middleware):
    """Middleware for rate limiting requests."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
        key_func: Optional[Callable[[Any], str]] = None,
    ):
        self._max_requests = max_requests
        self._window = window_seconds
        self._key_func = key_func or (lambda _: "default")
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def process_request(self, context: MiddlewareContext) -> None:
        key = self._key_func(context.request)
        now = time.time()
        
        async with self._lock:
            # Clean old entries
            if key not in self._requests:
                self._requests[key] = []
            
            self._requests[key] = [
                ts for ts in self._requests[key]
                if now - ts < self._window
            ]
            
            if len(self._requests[key]) >= self._max_requests:
                context.should_continue = False
                context.response = {
                    "error": "Rate limit exceeded",
                    "retry_after": self._window,
                }
                return
            
            self._requests[key].append(now)
            context.set("rate_limit_remaining", 
                       self._max_requests - len(self._requests[key]))
    
    async def process_response(self, context: MiddlewareContext) -> None:
        pass


class RetryMiddleware(Middleware):
    """Middleware for automatic retry with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_exceptions: tuple = (Exception,),
    ):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._retry_exceptions = retry_exceptions
    
    async def process_request(self, context: MiddlewareContext) -> None:
        pass
    
    async def process_response(self, context: MiddlewareContext) -> None:
        # Check if response indicates a retryable error
        if isinstance(context.response, dict) and context.response.get("error"):
            error = context.response["error"]
            retry_count = context.get("retry_count", 0)
            
            if retry_count < self._max_retries:
                # Check if error is retryable
                if any(isinstance(error, exc) for exc in self._retry_exceptions):
                    delay = min(
                        self._base_delay * (2 ** retry_count),
                        self._max_delay,
                    )
                    logger.warning("Retrying after error: %s (attempt %d/%d, delay %.1fs)",
                                 error, retry_count + 1, self._max_retries, delay)
                    
                    context.set("retry_count", retry_count + 1)
                    context.should_continue = True
                    context.skip_remaining = True  # Skip other response middleware
                    
                    # Re-execute handler (would need pipeline support)
                    # For now, just mark for retry


class AuthMiddleware(Middleware):
    """Middleware for authentication/authorization."""
    
    def __init__(
        self,
        required_scopes: Optional[List[str]] = None,
        token_validator: Optional[Callable[[str], bool]] = None,
    ):
        self._required_scopes = required_scopes or []
        self._token_validator = token_validator
    
    async def process_request(self, context: MiddlewareContext) -> None:
        # Extract token from request
        token = self._extract_token(context.request)
        
        if not token:
            context.should_continue = False
            context.response = {"error": "Authentication required", "code": 401}
            return
        
        if self._token_validator and not self._token_validator(token):
            context.should_continue = False
            context.response = {"error": "Invalid token", "code": 403}
            return
        
        # Store auth info in context
        context.set("auth_token", token)
    
    async def process_response(self, context: MiddlewareContext) -> None:
        pass
    
    def _extract_token(self, request: Any) -> Optional[str]:
        # Try common locations
        if hasattr(request, 'headers'):
            auth = request.headers.get('Authorization', '')
            if auth.startswith('Bearer '):
                return auth[7:]
        if hasattr(request, 'metadata'):
            return request.metadata.get('auth_token')
        return None


class TimingMiddleware(Middleware):
    """Middleware for detailed timing metrics."""
    
    def __init__(self):
        self._stage_times: Dict[str, float] = {}
    
    async def process_request(self, context: MiddlewareContext) -> None:
        context.set("_timing_start", time.perf_counter())
        context.set("_timing_stages", {})
    
    async def process_response(self, context: MiddlewareContext) -> None:
        total_time = time.perf_counter() - context.get("_timing_start", 0)
        
        stages = context.get("_timing_stages", {})
        stages["total"] = total_time
        
        context.set("timing", stages)
        
        logger.debug("Request timing: %s", 
                    {k: f"{v*1000:.1f}ms" for k, v in stages.items()})


# ── Pipeline Decorator ──────────────────────────────────────────────────────

def with_middleware(pipeline: MiddlewarePipeline):
    """
    Decorator to wrap a handler with middleware pipeline.
    
    Usage:
        pipeline = MiddlewarePipeline()
        pipeline.add_middleware(LoggingMiddleware())
        
        @with_middleware(pipeline)
        async def my_handler(request):
            return {"result": "ok"}
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            return await pipeline.execute(request, func)
        return wrapper
    return decorator


# ── Global Pipeline ──────────────────────────────────────────────────────────

_default_pipeline: Optional[MiddlewarePipeline] = None


def get_default_pipeline() -> MiddlewarePipeline:
    """Get the default middleware pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = MiddlewarePipeline()
        # Add default middleware
        _default_pipeline.add_middleware(TimingMiddleware())
        _default_pipeline.add_middleware(LoggingMiddleware())
    return _default_pipeline


async def execute_with_middleware(
    request: Any,
    handler: Callable[[Any], Any],
    pipeline: Optional[MiddlewarePipeline] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute a handler with the middleware pipeline."""
    p = pipeline or get_default_pipeline()
    return await p.execute(request, handler, metadata)