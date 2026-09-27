"""
NEXUS AI v4.0 — Typed event bus for backend↔UI communication.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

All events are typed dataclasses. The UI subscribes ONLY to:
  - AssistantTextDelta / AssistantTextFinal (for chat rendering)
  - UIStateChange (idle/listening/thinking/speaking/error)
  - ToolProgress (optional, for DAG visualization)

ToolCall, ToolResult, Trace, Error go ONLY to audit/debug log — NEVER to chat.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, List, Callable, Awaitable
from uuid import uuid4
import asyncio
import logging

logger = logging.getLogger("nexus.events")


class EventType(str, Enum):
    # Chat rendering (UI sees these)
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_FINAL = "assistant_text_final"
    USER_MESSAGE = "user_message"
    
    # UI state (UI sees these)
    UI_STATE_CHANGE = "ui_state_change"
    TOOL_PROGRESS = "tool_progress"
    
    # Backend-only (audit/debug log only)
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TRACE = "trace"
    ERROR = "error"
    AGENT_THINKING = "agent_thinking"
    MEMORY_RETRIEVAL = "memory_retrieval"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"


class UIState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class BaseEvent:
    """Base class for all events."""
    event_type: EventType = field(default=EventType.TRACE)
    request_id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Chat Events (UI RENDERED) ────────────────────────────────────────────────

@dataclass
class AssistantTextDelta(BaseEvent):
    """Streaming text chunk from response AI."""
    content: str = ""
    is_first: bool = False
    
    def __post_init__(self):
        self.event_type = EventType.ASSISTANT_TEXT_DELTA


@dataclass
class AssistantTextFinal(BaseEvent):
    """Final complete response from response AI."""
    content: str = ""
    intent: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.ASSISTANT_TEXT_FINAL


@dataclass
class UserMessage(BaseEvent):
    """User message (typed or transcribed)."""
    content: str = ""
    source: str = "text"  # "text" | "voice"
    
    def __post_init__(self):
        self.event_type = EventType.USER_MESSAGE


# ── UI State Events (UI RENDERED) ────────────────────────────────────────────

@dataclass
class UIStateChange(BaseEvent):
    """UI state transition for HUD/overlay."""
    state: UIState = UIState.IDLE
    detail: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.UI_STATE_CHANGE


@dataclass
class ToolProgress(BaseEvent):
    """Tool execution progress for DAG visualization."""
    tool_name: str = ""
    status: str = "pending"  # "pending" | "running" | "success" | "failed" | "skipped"
    progress: float = 0.0
    message: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.TOOL_PROGRESS


# ── Backend-Only Events (AUDIT LOG ONLY - NEVER TO CHAT) ─────────────────────

@dataclass
class ToolCall(BaseEvent):
    """Tool invocation (backend only)."""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    is_ui: bool = False
    
    def __post_init__(self):
        self.event_type = EventType.TOOL_CALL


@dataclass
class ToolResult(BaseEvent):
    """Tool execution result (backend only)."""
    tool_name: str = ""
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    verified: bool = False
    verification_detail: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.TOOL_RESULT


@dataclass
class Trace(BaseEvent):
    """Internal trace/debug event (backend only)."""
    message: str = ""
    level: str = "debug"  # debug | info | warning
    component: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.TRACE


@dataclass
class ErrorEvent(BaseEvent):
    """Error event (backend only)."""
    error: str = ""
    component: str = ""
    recoverable: bool = True
    user_message: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.ERROR


@dataclass
class AgentThinking(BaseEvent):
    """Agent reasoning step (backend only)."""
    step: str = ""
    detail: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.AGENT_THINKING


@dataclass
class MemoryRetrieval(BaseEvent):
    """Memory retrieval event (backend only)."""
    query: str = ""
    collections: List[str] = field(default_factory=list)
    result_count: int = 0
    duration_ms: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.MEMORY_RETRIEVAL


@dataclass
class LLMRequest(BaseEvent):
    """LLM request sent (backend only)."""
    model: str = ""
    provider: str = ""
    role: str = ""
    messages_count: int = 0
    estimated_tokens: int = 0
    
    def __post_init__(self):
        self.event_type = EventType.LLM_REQUEST


@dataclass
class LLMResponse(BaseEvent):
    """LLM response received (backend only)."""
    model: str = ""
    provider: str = ""
    role: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0.0
    streamed: bool = False
    success: bool = False
    error: Optional[str] = None
    
    def __post_init__(self):
        self.event_type = EventType.LLM_RESPONSE


# ── Event Bus ────────────────────────────────────────────────────────────────

EventHandler = Callable[[BaseEvent], Awaitable[None]]


class EventBus:
    """
    Typed event bus with sync and async subscription.
    
    Separates UI events (delivered to UI queue) from backend events
    (delivered to audit log only).
    """
    
    UI_EVENT_TYPES = {
        EventType.ASSISTANT_TEXT_DELTA,
        EventType.ASSISTANT_TEXT_FINAL,
        EventType.USER_MESSAGE,
        EventType.UI_STATE_CHANGE,
        EventType.TOOL_PROGRESS,
    }
    
    BACKEND_EVENT_TYPES = {
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.TRACE,
        EventType.ERROR,
        EventType.AGENT_THINKING,
        EventType.MEMORY_RETRIEVAL,
        EventType.LLM_REQUEST,
        EventType.LLM_RESPONSE,
    }

    def __init__(self):
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._ui_queue: Optional[asyncio.Queue] = None
        self._audit_queue: Optional[asyncio.Queue] = None
        self._wildcard_handlers: List[EventHandler] = []

    def set_queues(self, ui_queue: asyncio.Queue, audit_queue: asyncio.Queue) -> None:
        self._ui_queue = ui_queue
        self._audit_queue = audit_queue

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        self._wildcard_handlers.append(handler)

    async def emit(self, event: BaseEvent) -> None:
        """Emit event to all subscribers and appropriate queues."""
        # Type-specific handlers
        for handler in self._subscribers.get(event.event_type, []):
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.event_type}: {e}")

        # Wildcard handlers
        for handler in self._wildcard_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Wildcard handler error: {e}")

        # Route to queues
        if event.event_type in self.UI_EVENT_TYPES and self._ui_queue:
            try:
                self._ui_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"UI queue full, dropping {event.event_type}")

        if event.event_type in self.BACKEND_EVENT_TYPES and self._audit_queue:
            try:
                self._audit_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Audit queue full, dropping {event.event_type}")

    def emit_sync(self, event: BaseEvent) -> None:
        """Emit event synchronously (for non-async contexts)."""
        # Type-specific handlers
        for handler in self._subscribers.get(event.event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Sync handler error for {event.event_type}: {e}")

        # Route to queues
        if event.event_type in self.UI_EVENT_TYPES and self._ui_queue:
            try:
                self._ui_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        if event.event_type in self.BACKEND_EVENT_TYPES and self._audit_queue:
            try:
                self._audit_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def set_event_bus(bus: EventBus) -> None:
    global _event_bus
    _event_bus = bus


# Convenience emitters
async def emit_ui_text_delta(request_id: str, content: str, is_first: bool = False) -> None:
    await get_event_bus().emit(AssistantTextDelta(request_id=request_id, content=content, is_first=is_first))


async def emit_ui_text_final(request_id: str, content: str, intent: str = "") -> None:
    await get_event_bus().emit(AssistantTextFinal(request_id=request_id, content=content, intent=intent))


async def emit_ui_state(state: UIState, detail: str = "", request_id: str = "") -> None:
    await get_event_bus().emit(UIStateChange(request_id=request_id or uuid4().hex[:12], state=state, detail=detail))


async def emit_user_message(request_id: str, content: str, source: str = "text") -> None:
    await get_event_bus().emit(UserMessage(request_id=request_id, content=content, source=source))


async def emit_tool_call(request_id: str, tool_name: str, arguments: Dict[str, Any], is_ui: bool = False) -> None:
    await get_event_bus().emit(ToolCall(request_id=request_id, tool_name=tool_name, arguments=arguments, is_ui=is_ui))


async def emit_tool_result(
    request_id: str,
    tool_name: str,
    success: bool,
    result: Any = None,
    error: Optional[str] = None,
    duration_ms: float = 0.0,
    verified: bool = False,
    verification_detail: str = "",
) -> None:
    await get_event_bus().emit(ToolResult(
        request_id=request_id,
        tool_name=tool_name,
        success=success,
        result=result,
        error=error,
        duration_ms=duration_ms,
        verified=verified,
        verification_detail=verification_detail,
    ))


async def emit_trace(request_id: str, message: str, level: str = "debug", component: str = "") -> None:
    await get_event_bus().emit(Trace(request_id=request_id, message=message, level=level, component=component))


async def emit_error(
    request_id: str,
    error: str,
    component: str = "",
    recoverable: bool = True,
    user_message: str = "",
) -> None:
    await get_event_bus().emit(ErrorEvent(
        request_id=request_id,
        error=error,
        component=component,
        recoverable=recoverable,
        user_message=user_message,
    ))