"""
NEXUS AI v4.0 — Three-Tier LangGraph ReAct Agent with Groq.
Hardware: Intel i3 7th Gen · 12GB RAM · Groq API

Three-tier routing:
  1. CHAT_ONLY: Greetings/small talk → Response AI only, no tools, no memory
  2. DETERMINISTIC: Known commands (open app, etc.) → Direct tool call, no LLM
  3. PLANNER: Everything else → Full ReAct loop with planner LLM
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from enum import Enum
from typing import Optional, Dict, Any, List, AsyncIterator, Tuple
from dataclasses import dataclass, field

from nexus_config.settings import get_settings
from nexus_config.audit_logger import get_audit_logger
from nexus_brain.agent_state import (
    NexusState,
    make_initial_state,
)
from nexus_brain.timing import get_tracer
from nexus_brain.events import (
    get_event_bus,
    EventType,
    UIState,
    AssistantTextDelta,
    AssistantTextFinal,
    UserMessage,
    UIStateChange,
    ToolProgress,
    ToolResult,
    emit_ui_text_delta,
    emit_ui_text_final,
    emit_ui_state,
    emit_user_message,
    emit_tool_call,
    emit_tool_result,
    emit_trace,
    emit_error,
)
from nexus_tools.registry import get_tool_registry
from nexus_audio.tts_engine import get_tts_engine, VoiceProfile

logger = logging.getLogger("nexus.orchestrator")


# ─── Routing Tier Enum ────────────────────────────────────────────────────────

class RouteTier(str, Enum):
    CHAT_ONLY = "chat_only"          # Response AI only, streamed
    DETERMINISTIC = "deterministic"  # Direct tool call, no LLM
    PLANNER = "planner"              # Full ReAct loop


# ─── Three-Tier Router ────────────────────────────────────────────────────────

# Patterns that map directly to tools without LLM
DETERMINISTIC_PATTERNS: List[Tuple[str, str, Dict[str, Any]]] = [
    (r"^open\s+(notepad|editor)$", "open_application", {"name": "notepad"}),
    (r"^open\s+(calculator|calc)$", "open_application", {"name": "calculator"}),
    (r"^open\s+(paint|mspaint)$", "open_application", {"name": "paint"}),
    (r"^open\s+(explorer|file explorer|files)$", "open_application", {"name": "explorer"}),
    (r"^open\s+(cmd|command prompt|terminal)$", "open_application", {"name": "cmd"}),
    (r"^open\s+(powershell)$", "open_application", {"name": "powershell"}),
    (r"^open\s+(settings)$", "open_application", {"name": "settings"}),
    (r"^launch\s+(\w+)$", "open_application", {"name": r"\1"}),
    (r"^start\s+(\w+)$", "open_application", {"name": r"\1"}),
]

_COMPILED_DETERMINISTIC = [(re.compile(p, re.IGNORECASE), tool, args) for p, tool, args in DETERMINISTIC_PATTERNS]

# Chat-only patterns (greetings, small talk)
CHAT_ONLY_PATTERNS = [
    r"^hello", r"^hi\b", r"^hey\b", r"^good morning", r"^good evening",
    r"^how are you", r"^what'?s up", r"^thanks", r"^thank you",
    r"^bye", r"^goodbye", r"^see you", r"^nice to meet",
    r"^who are you", r"^what can you do", r"^help",
]

_COMPILED_CHAT_ONLY = [re.compile(p, re.IGNORECASE) for p in CHAT_ONLY_PATTERNS]

# Response AI system prompt (Chat-Only path)
RESPONSE_AI_SYSTEM_PROMPT = """You are NEXUS AI's response voice. Reply in natural, concise English.
Never mention tools, function names, JSON, internal steps, or technical details.
Never use markdown formatting unless the user asks for it.
Be helpful, direct, and conversational."""


def route_request(user_input: str) -> Tuple[RouteTier, Optional[str], Optional[Dict[str, Any]]]:
    """Three-tier router: returns (tier, tool_name, tool_args) or (tier, None, None)."""
    text = user_input.strip().lower()
    
    # Tier 1: Chat-only (greetings, small talk)
    for pattern in _COMPILED_CHAT_ONLY:
        if pattern.search(text):
            return RouteTier.CHAT_ONLY, None, None
    
    # Tier 2: Deterministic commands (direct tool mapping)
    for pattern, tool_name, default_args in _COMPILED_DETERMINISTIC:
        match = pattern.search(text)
        if match:
            args = dict(default_args)
            for key, value in args.items():
                if isinstance(value, str) and value.startswith("\\") and value[2:].isdigit():
                    group_idx = int(value[2:])
                    if group_idx <= len(match.groups()):
                        args[key] = match.group(group_idx)
            return RouteTier.DETERMINISTIC, tool_name, args
    
    # Tier 3: Planner (everything else)
    return RouteTier.PLANNER, None, None


# ─── Tool Result Templates ──────────────────────────────────────────────────

TOOL_SUCCESS_TEMPLATES = {
    "open_application": "Opened {name}.",
    "desktop_automation": "Done.",
    "clipboard_manager": "Copied to clipboard.",
    "system_monitor": "{result}",
    "file_manager": "Done.",
}

TOOL_FAILURE_TEMPLATES = {
    "open_application": "Couldn't open {name}. {error}",
    "desktop_automation": "Action failed: {error}",
    "clipboard_manager": "Clipboard error: {error}",
    "system_monitor": "Couldn't get system info: {error}",
    "file_manager": "File operation failed: {error}",
}


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are NEXUS AI, an autonomous desktop agent. Complete tasks by using tools. Be concise and direct.

CORE RULES:
1. NEVER REFUSE A TASK. If a tool doesn't exist, use capability_synthesizer to create it.
2. ALWAYS USE TOOLS FOR ACTIONS. Only skip tools for pure conversation.
3. PYTHON INTERPRETER is your superpower for any task lacking a specific tool.
4. BE PROACTIVE: propose logical next steps after completing tasks.
5. REMEMBER EVERYTHING: Use local_vector_db to recall past tasks and preferences.

When you need to use a tool, respond with a tool_calls array. When you have a final answer, respond directly. Never mention tools, function names, JSON, or internal steps in your final response."""


# ─── Agent Orchestrator ──────────────────────────────────────────────────────

@dataclass
class AgentOrchestrator:
    """
    Simplified LangGraph ReAct agent orchestrator.
    
    Uses LangGraph's create_react_agent directly for reliable tool calling.
    The agent decides when to use tools based on the system prompt.
    """
    
    _settings: Any = field(default_factory=get_settings)
    _audit_logger: Any = field(default_factory=get_audit_logger)
    _agent_executor: Any = field(default=None, init=False)
    _tool_executor: Any = field(default=None, init=False)
    _llm_router: Any = field(default=None, init=False)
    _output_queue: Optional[asyncio.Queue] = field(default=None, init=False)
    _progress_queue: Optional[asyncio.Queue] = field(default=None, init=False)
    _event_bus: Any = field(default_factory=get_event_bus)
    _memory: Any = field(default=None, init=False)
    
    # Concurrency control
    _run_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _max_concurrent_runs: int = 3
    _active_runs: int = 0
    
    def __post_init__(self):
        self._init_agent()
    
    def _init_agent(self):
        """Initialize the LangGraph ReAct agent."""
        from langchain_groq import ChatGroq
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        
        # Get tools from registry
        registry = get_tool_registry()
        registry.load_builtin_tools()
        
        # Convert registry tools to LangChain tools format
        # For planner, only include tools needed for complex tasks
        PLANNER_TOOLS = {
            "run_system_command", "file_manager", "web_search", "web_manager",
            "browser_ghost", "desktop_automation", "python_interpreter",
            "document_builder", "local_vector_db", "process_manager",
            "image_processor", "data_analyzer", "code_editor_control",
            "workflow_macro", "git_operations", "capability_synthesizer",
            "system_monitor", "window_manager", "clipboard_manager",
            "notification_sender", "calendar_manager", "email_client",
            "image_processor", "data_analyzer",
        }
        
        tools = []
        for name in registry.list_tools():
            if name in PLANNER_TOOLS:
                func = registry.get_tool(name)
                if func:
                    tools.append(func)
        
        logger.info(f"Initialized planner agent with {len(tools)} tools")
        
        # Create LLM (Groq primary)
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            max_tokens=8192,
        )
        
        # Create LLM router for chat-only path
        from nexus_brain.llm_router import get_llm_router
        self._llm_router = get_llm_router()
        
        # Create checkpointer for conversation memory
        self._memory = MemorySaver()
        
        # Initialize TTS engine
        self._tts_engine = get_tts_engine()
        
        # Create ReAct agent
        self._agent_executor = create_react_agent(
            llm,
            tools,
            messages_modifier=SYSTEM_PROMPT,
            checkpointer=self._memory,
        )
        
        # Also keep the tool executor for direct tool calls
        async def tool_executor(tool_name: str, tool_input: Dict[str, Any], is_ui: bool) -> str:
            try:
                result = await registry.execute(tool_name, tool_input, is_ui)
                return result
            except Exception as e:
                logger.error(f"Tool execution failed for '{tool_name}': {e}")
                return json.dumps({"success": False, "error": str(e)})
        
        self._tool_executor = tool_executor
    
    def set_queues(
        self,
        output_queue: asyncio.Queue,
        progress_queue: asyncio.Queue,
    ) -> None:
        self._output_queue = output_queue
        self._progress_queue = progress_queue
        self._event_bus.set_queues(output_queue, progress_queue)
    
    def set_tool_executor(self, executor):
        """Set a custom tool executor (for testing)."""
        self._tool_executor = executor
    
    async def run(
        self,
        user_input: str,
        session_context: Optional[Dict[str, Any]] = None,
        input_type: str = "text",
    ) -> NexusState:
        """
        Run a complete agent cycle from input to response.
        """
        tracer = get_tracer()
        trace = tracer.start_request(input_type=input_type, user_input=user_input)
        request_id = trace.request_id
        
        # Emit user message event
        await emit_user_message(request_id, user_input, source=input_type)
        
        # Concurrency control
        async with self._run_lock:
            while self._active_runs >= self._max_concurrent_runs:
                await asyncio.sleep(0.1)
            self._active_runs += 1
        
        try:
            # Create initial state
            working_memory = session_context.get("working_memory", []) if session_context else []
            conversation_history = session_context.get("history", []) if session_context else []
            
            state = make_initial_state(
                user_input=user_input,
                session_id=request_id,
                input_type=input_type,
                working_memory=working_memory,
                conversation_history=conversation_history,
            )
            
            # Three-tier router
            tracer.mark("routed")
            route_tier, det_tool, det_args = route_request(user_input)
            state["route_tier"] = route_tier.value
            
            if route_tier == RouteTier.CHAT_ONLY:
                # Fast path: stream response AI directly, no tools, no memory
                await self._run_chat_only_path(state, request_id, tracer)
                state["total_duration_ms"] = (time.perf_counter() - trace.spans[0].start_ns / 1_000_000_000) * 1000
                tracer.set_routed_intent(state.get("intent", "chat"))
                tracer.end_request()
                return state
            
            elif route_tier == RouteTier.DETERMINISTIC:
                # Fast path: direct tool execution with verification
                await self._run_deterministic_path(state, request_id, tracer, det_tool, det_args)
                state["total_duration_ms"] = (time.perf_counter() - trace.spans[0].start_ns / 1_000_000_000) * 1000
                tracer.set_routed_intent(state.get("intent", "simple_command"))
                tracer.end_request()
                return state
            
            # PLANNER path: full ReAct loop
            state["route_tier"] = "planner"
            
            # Set UI state to thinking
            await emit_ui_state(UIState.THINKING, "Processing...", request_id)
            
            # Run the planner agent
            result = await self._run_planner_path(state, request_id, tracer, user_input)
            
            state["final_response"] = result.get("final_response", "Task completed.")
            state["intent"] = result.get("intent", "general")
            state["llm_calls"] = result.get("llm_calls", 1)
            state["total_duration_ms"] = (time.perf_counter() - trace.spans[0].start_ns / 1_000_000_000) * 1000
            tracer.set_routed_intent(state.get("intent", "general"))
            tracer.end_request()
            
            return state
            
        finally:
            async with self._run_lock:
                self._active_runs = max(0, self._active_runs - 1)
    
    # ── Helper Methods ──────────────────────────────────────────────────────
    
    async def _run_chat_only_path(self, state: NexusState, request_id: str, tracer) -> None:
        """Fast path: stream response AI directly, no tools, no memory."""
        await emit_ui_state(UIState.THINKING, "Generating response...", request_id)
        tracer.mark("llm_request_sent")
        
        messages = [
            {"role": "system", "content": RESPONSE_AI_SYSTEM_PROMPT},
            {"role": "user", "content": state["user_input"]},
        ]
        
        first_token = True
        full_content = ""
        
        async for token in self._llm_router.stream(
            messages,
            role="voice",
            temperature=0.3,
            max_tokens=500,
        ):
            if first_token:
                tracer.mark("first_token")
                first_token = False
            await emit_ui_text_delta(request_id, token, is_first=first_token)
            full_content += token
        
        tracer.mark("ui_rendered")
        await emit_ui_text_final(request_id, full_content, intent="chat")
        await self._speak_response(full_content)
        await emit_ui_state(UIState.IDLE, "Ready", request_id)
        
        # Update state
        state["final_response"] = full_content
        state["intent"] = "chat"
        state["llm_calls"] = 1
    
    async def _run_deterministic_path(
        self,
        state: NexusState,
        request_id: str,
        tracer,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> None:
        """Fast path: direct tool execution with verification, no planner LLM."""
        await emit_ui_state(UIState.THINKING, f"Executing {tool_name}...", request_id)
        tracer.mark("tool_started")
        
        # Execute tool with verification
        tool_result = await self._execute_tool_verified(request_id, tool_name, tool_args)
        
        tracer.mark("tool_verified")
        
        # Generate response from template (no second LLM call)
        if tool_result.success and tool_result.verified:
            template = TOOL_SUCCESS_TEMPLATES.get(tool_name, "Done.")
            response = template.format(name=tool_args.get("name", ""), result=tool_result.result or "")
        else:
            template = TOOL_FAILURE_TEMPLATES.get(tool_name, "Action failed: {error}")
            response = template.format(name=tool_args.get("name", ""), error=tool_result.error or "Unknown error")
        
        # Stream the template response (instant)
        await emit_ui_text_delta(request_id, response, is_first=True)
        await emit_ui_text_final(request_id, response, intent="deterministic")
        await self._speak_response(response)
        await emit_ui_state(UIState.IDLE, "Ready", request_id)
        
        state["final_response"] = response
        state["intent"] = "simple_command"
        state["llm_calls"] = 0
    
    async def _execute_tool_verified(
        self,
        request_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> ToolResult:
        """Execute tool and verify effect (for deterministic path)."""
        start = time.perf_counter()
        
        await emit_tool_call(request_id, tool_name, tool_args, False)
        
        try:
            if self._tool_executor is None:
                raise RuntimeError("Tool executor not set")
            
            output = await self._tool_executor(tool_name, tool_args, False)
            
            try:
                parsed = json.loads(output)
                success = parsed.get("success", True)
                result_data = parsed.get("result", parsed.get("error", output))
                verified = parsed.get("verified", False)
                verification_detail = parsed.get("verification_detail", "")
            except (json.JSONDecodeError, TypeError):
                success = True
                result_data = output
                verified = False
                verification_detail = ""
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            return ToolResult(
                request_id=request_id,
                tool_name=tool_name,
                success=success,
                result=result_data,
                error=parsed.get("error") if not success else None,
                duration_ms=duration_ms,
                verified=verified,
                verification_detail=verification_detail,
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                request_id=request_id,
                tool_name=tool_name,
                success=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=duration_ms,
            )
    
    async def _speak_response(self, text: str) -> None:
        """Speak the response using TTS engine."""
        if not text or not text.strip():
            return
        try:
            await self._tts_engine.say(text.strip())
        except Exception as e:
            logger.warning(f"TTS failed: {e}")
    
    async def _run_planner_path(
        self,
        state: NexusState,
        request_id: str,
        tracer,
        user_input: str,
    ) -> Dict[str, Any]:
        """Full ReAct loop with planner LLM."""
        await emit_ui_state(UIState.THINKING, "Processing...", request_id)
        tracer.mark("llm_request_sent")
        
        try:
            config = {
                "configurable": {"thread_id": state["session_id"]},
                "recursion_limit": 50,
            }
            
            full_content = ""
            first_token = True
            
            async for chunk in self._agent_executor.astream(
                {"messages": [("user", user_input)]},
                config={"configurable": {"thread_id": state["session_id"]}, "recursion_limit": 50},
                stream_mode="values",
            ):
                messages = chunk.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, 'content') and last_msg.content:
                        content = last_msg.content
                        if content and not content.startswith("{"):  # Not a tool call JSON
                            if first_token:
                                tracer.mark("first_token")
                                first_token = False
                            
                            await emit_ui_text_delta(request_id, content, is_first=first_token)
                            full_content += content
            
            if not full_content:
                full_content = "Task completed."
            
            await emit_ui_text_final(request_id, full_content, intent="general")
            await self._speak_response(full_content)
            await emit_ui_state(UIState.IDLE, "Ready", request_id)
            
            return {
                "final_response": full_content,
                "intent": "general",
                "llm_calls": 1,
            }
        finally:
            pass


# ─── Singleton Accessor ──────────────────────────────────────────────────────

_agent_orchestrator_instance: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _agent_orchestrator_instance
    if _agent_orchestrator_instance is None:
        _agent_orchestrator_instance = AgentOrchestrator()
    return _agent_orchestrator_instance


def reset_orchestrator():
    global _agent_orchestrator_instance
    _agent_orchestrator_instance = None