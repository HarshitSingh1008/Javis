"""
NEXUS AI v4.0 — High-precision timing tracer for request pipeline.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides perf_counter spans for: input_received, routed, memory_done,
llm_request_sent, first_token, ui_rendered, tool_started, tool_verified,
plus backend_used. Writes to audit log with request_id.
"""

import time
import uuid
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
from contextlib import contextmanager
from threading import local

from nexus_config.audit_logger import get_audit_logger

logger = logging.getLogger("nexus.timing")

_thread_local = local()

DEBUG = os.environ.get("NEXUS_DEBUG", "0") == "1"


@dataclass
class TimingSpan:
    """Single timing span with start/end perf_counter timestamps."""
    name: str
    start_ns: int
    end_ns: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_ns == 0:
            return (time.perf_counter_ns() - self.start_ns) / 1_000_000
        return (self.end_ns - self.start_ns) / 1_000_000

    def finish(self, **metadata) -> None:
        self.end_ns = time.perf_counter_ns()
        self.metadata.update(metadata)


@dataclass
class RequestTrace:
    """Complete trace for one user request."""
    request_id: str
    spans: List[TimingSpan] = field(default_factory=list)
    backend_used: str = "unknown"
    routed_intent: str = "unknown"
    input_type: str = "text"
    user_input_preview: str = ""

    def add_span(self, name: str, **metadata) -> TimingSpan:
        span = TimingSpan(name=name, start_ns=time.perf_counter_ns(), metadata=metadata)
        self.spans.append(span)
        return span

    def get_span(self, name: str) -> Optional[TimingSpan]:
        for span in self.spans:
            if span.name == name:
                return span
        return None

    def to_summary_line(self) -> str:
        parts = [f"req={self.request_id[:8]}"]
        for span in self.spans:
            parts.append(f"{span.name}={span.duration_ms:.1f}ms")
        parts.append(f"backend={self.backend_used}")
        parts.append(f"intent={self.routed_intent}")
        return " | ".join(parts)

    def to_audit_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "backend_used": self.backend_used,
            "routed_intent": self.routed_intent,
            "input_type": self.input_type,
            "user_input_preview": self.user_input_preview[:100],
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "metadata": s.metadata,
                }
                for s in self.spans
            ],
        }


class TimingTracer:
    """
    Thread-local timing tracer for the current request.
    
    Usage:
        tracer = TimingTracer.get()
        tracer.start_request(input_type="text")
        with tracer.span("memory_retrieval"):
            # do work
        tracer.set_backend("ollama")
        tracer.set_routed_intent("chat")
        tracer.end_request()
    """

    _current_trace: Optional[RequestTrace] = None

    @classmethod
    def get(cls) -> "TimingTracer":
        if not hasattr(_thread_local, "tracer"):
            _thread_local.tracer = TimingTracer()
        return _thread_local.tracer

    @classmethod
    def clear(cls) -> None:
        if hasattr(_thread_local, "tracer"):
            _thread_local.tracer = None

    def start_request(self, input_type: str = "text", user_input: str = "") -> RequestTrace:
        self._current_trace = RequestTrace(
            request_id=uuid.uuid4().hex[:12],
            input_type=input_type,
            user_input_preview=user_input[:100],
        )
        self._current_trace.add_span("input_received")
        return self._current_trace

    def end_request(self) -> Optional[RequestTrace]:
        if self._current_trace is None:
            return None
        
        trace = self._current_trace
        trace.add_span("request_complete")
        
        # Log to audit
        audit = get_audit_logger()
        audit.log(
            event_type="REQUEST_TIMING",
            data=trace.to_audit_dict(),
            module="nexus_brain.timing",
            function_name="end_request",
            duration_ms=sum(s.duration_ms for s in trace.spans),
            success=True,
        )
        
        if DEBUG:
            print(f"[TIMING] {trace.to_summary_line()}")
        
        self._current_trace = None
        return trace

    @contextmanager
    def span(self, name: str, **metadata):
        if self._current_trace is None:
            yield None
            return
        
        span = self._current_trace.add_span(name, **metadata)
        try:
            yield span
        finally:
            span.finish()

    def set_backend(self, backend: str) -> None:
        if self._current_trace:
            self._current_trace.backend_used = backend

    def set_routed_intent(self, intent: str) -> None:
        if self._current_trace:
            self._current_trace.routed_intent = intent

    def mark(self, name: str, **metadata) -> None:
        """Add an instantaneous marker span (duration ~0)."""
        if self._current_trace:
            span = self._current_trace.add_span(name, **metadata)
            span.finish()

    def get_current(self) -> Optional[RequestTrace]:
        return self._current_trace


def get_tracer() -> TimingTracer:
    """Get the thread-local timing tracer."""
    return TimingTracer.get()