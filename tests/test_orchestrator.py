"""
NEXUS AI v4.0 — 8 end-to-end agent tests for the AgentOrchestrator.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Covers:
1. Orchestrator singleton creation and initialization
2. Full run cycle with mocked dependencies
3. Intent classification node
4. Context loading node
5. Agent reasoning node
6. Tool execution and capability gap detection
7. Memory write integration
8. Error handling and fallback behavior
"""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any

from nexus_brain.agent_state import NexusState, make_initial_state, GraphNode


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def mock_tool_executor():
    """Mock tool executor that returns successful results."""
    async def executor(tool_name: str, tool_input: Dict[str, Any], is_ui: bool) -> str:
        return json.dumps({"success": True, "result": f"Executed {tool_name}"})
    return executor


@pytest.fixture(scope="function")
def mock_tool_executor_failing():
    """Mock tool executor that returns errors."""
    async def executor(tool_name: str, tool_input: Dict[str, Any], is_ui: bool) -> str:
        return json.dumps({"success": False, "error": f"ModuleNotFoundError: No module named '{tool_name}'"})
    return executor


# ─── Test Classes ──────────────────────────────────────────────────────────────

class TestOrchestratorInit:
    """Tests for orchestrator initialization and singleton."""

    def test_get_orchestrator_singleton(self):
        """Test that get_orchestrator returns a singleton."""
        from nexus_brain.orchestrator import get_orchestrator
        o1 = get_orchestrator()
        o2 = get_orchestrator()
        assert o1 is o2

    def test_orchestrator_initializes_with_defaults(self):
        """Test orchestrator has default attributes after init."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()
        assert orch._settings is not None
        assert orch._audit_logger is not None
        assert orch._llm_router is not None
        assert orch._classifier is not None
        assert orch._context_builder is not None
        assert orch._tool_executor is None  # Must be set explicitly

    def test_set_tool_executor(self):
        """Test that set_tool_executor sets the executor."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()

        async def dummy_executor(name, inp, ui):
            return '{"success": true}'
        
        orch.set_tool_executor(dummy_executor)
        assert orch._tool_executor is not None

    def test_set_queues(self, event_loop):
        """Test that set_queues configures output and progress queues."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()
        oq = asyncio.Queue()
        pq = asyncio.Queue()
        previous_queues = (orch._output_queue, orch._progress_queue)
        try:
            orch.set_queues(oq, pq)
            assert orch._output_queue is oq
            assert orch._progress_queue is pq
        finally:
            orch.set_queues(*previous_queues)


class TestOrchestratorRun:
    """Tests for the main run cycle."""

    @pytest.mark.asyncio
    async def test_run_basic_flow(self, mock_settings, mock_tool_executor):
        """Test basic run flow with intent classification and response."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.set_tool_executor(mock_tool_executor)

        # Mock classifier to return simple_command
        with patch.object(orch._classifier, 'classify', new_callable=AsyncMock) as mock_classify:
            from nexus_brain.intent_classifier import ClassificationResult
            mock_classify.return_value = ClassificationResult(
                intent="simple_command", confidence=0.95,
                reasoning="Rule match", latency_ms=1.0,
                model_used="rule_based", needs_full_agent=False,
            )

            state = await orch.run("Hello", session_context={})

        assert state is not None
        assert state.get("user_input") == "Hello"
        assert state.get("intent") == "simple_command"
        assert state.get("session_id") is not None
        assert state.get("total_duration_ms", 0) > 0

    @pytest.mark.asyncio
    async def test_run_with_session_context(self, mock_settings, mock_tool_executor):
        """Test run with session context including working memory and history."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.set_tool_executor(mock_tool_executor)

        session = {
            "working_memory": [{"key": "test", "value": "value"}],
            "history": [{"role": "user", "content": "Previous message"}],
        }

        with patch.object(orch._classifier, 'classify', new_callable=AsyncMock) as mock_classify:
            from nexus_brain.intent_classifier import ClassificationResult
            mock_classify.return_value = ClassificationResult(
                intent="chat", confidence=0.9, reasoning="Chat pattern",
                latency_ms=1.0, model_used="rule_based", needs_full_agent=False,
            )

            state = await orch.run("Hi", session_context=session)

        assert state is not None
        assert len(state.get("working_memory", [])) > 0

    @pytest.mark.asyncio
    async def test_run_intent_classify_node(self, mock_settings):
        """Test the intent classification node directly."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()

        state = make_initial_state(
            user_input="Open VS Code",
            session_id="test-session-001",
        )

        with patch.object(orch._classifier, 'classify', new_callable=AsyncMock) as mock_classify:
            from nexus_brain.intent_classifier import ClassificationResult
            mock_classify.return_value = ClassificationResult(
                intent="simple_command", confidence=0.95,
                reasoning="Start word match: open", latency_ms=1.2,
                model_used="rule_based", needs_full_agent=False,
            )

            result = await orch._run_intent_classify(state)

        assert result.get("intent") == "simple_command"
        assert result.get("confidence") == 0.95

    @pytest.mark.asyncio
    async def test_run_context_load_node(self, mock_settings, mock_tool_executor):
        """Test the context loading node."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.set_tool_executor(mock_tool_executor)

        state = make_initial_state(
            user_input="List my files",
            session_id="test-session-002",
        )
        state["intent"] = "simple_command"

        with patch.object(orch._context_builder, 'build_prompt', new_callable=AsyncMock) as mock_build:
            mock_build.return_value = "You are NEXUS AI. Test system prompt."
            result = await orch._run_context_load(state)

        assert result.get("system_prompt") is not None
        assert "NEXUS AI" in result.get("system_prompt", "")

    @pytest.mark.asyncio
    async def test_agent_reason_node(self, mock_settings, mock_tool_executor):
        """Test the agent reasoning node."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.set_tool_executor(mock_tool_executor)

        state = make_initial_state(
            user_input="Hello",
            session_id="test-session-003",
        )
        state["system_prompt"] = "You are NEXUS AI."

        with patch.object(orch._llm_router, 'generate', new_callable=AsyncMock) as mock_gen:
            from nexus_brain.llm_router import LLMResponse
            mock_gen.return_value = LLMResponse(
                content="Hello! How can I help you today?",
                provider="groq", model="llama-3.3-70b-versatile",
            )

            result = await orch._run_agent_reason(state)

        assert result.get("final_response") is not None
        assert "Hello" in result.get("final_response", "")

    @pytest.mark.asyncio
    async def test_tool_execute_node(self, mock_settings, mock_tool_executor):
        """Test the tool execution node."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.set_tool_executor(mock_tool_executor)

        state = make_initial_state(
            user_input="Run a command",
            session_id="test-session-004",
        )
        state["iteration_count"] = 1

        result = await orch._run_tool_execute(state, "t01_system_command", {"command": "dir"}, False)

        assert f"tool_{state['iteration_count']}" in result.get("tool_outputs", {})
        output = result["tool_outputs"][f"tool_{state['iteration_count']}"]
        assert "Executed" in output

    def test_detect_capability_gap(self, mock_settings):
        """Test capability gap detection logic."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()

        # No errors — no gap
        state = make_initial_state("test", "test-sid")
        assert orch._detect_capability_gap(state) is None

        # ModuleNotFoundError — gap detected
        state["errors"] = [{
            "node": GraphNode.TOOL_EXECUTE,
            "tool": "t99_missing",
            "error": "ModuleNotFoundError: No module named 't99_missing'",
            "timestamp": "00:00:00",
        }]
        gap = orch._detect_capability_gap(state)
        assert gap is not None
        assert "t99_missing" in gap

        # ImportError — gap detected
        state["errors"] = [{
            "node": GraphNode.TOOL_EXECUTE,
            "tool": "custom_tool",
            "error": "ImportError: cannot import name 'missing_dep'",
            "timestamp": "00:00:00",
        }]
        gap = orch._detect_capability_gap(state)
        assert gap is not None

    @pytest.mark.asyncio
    async def test_memory_write_node(self, mock_settings, mock_tool_executor):
        """Test the memory write node."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.set_tool_executor(mock_tool_executor)

        state = make_initial_state(
            user_input="Test memory",
            session_id="test-session-005",
        )
        state["final_response"] = "Task completed."
        state["total_duration_ms"] = 100.0

        with patch.object(orch._memory_manager, 'store_episodic') as mock_store:
            result = await orch._run_memory_write(state)
            mock_store.assert_called_once()

        assert result is not None


class TestToolCallExtraction:
    """Tests for tool call extraction from LLM responses."""

    def test_extract_structured_tool_call(self, mock_settings):
        """Test extraction of structured TOOL CALL: format."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()

        state = make_initial_state("test", "test-sid")
        state["_last_llm_response"] = "TOOL CALL: t02_file_manager\nPARAMS: {\"path\": \".\"}"

        tool_name, tool_input, is_ui = orch._extract_tool_call(state)
        assert tool_name == "t02_file_manager"
        assert tool_input is not None
        assert tool_input.get("path") == "."

    def test_extract_markdown_tool_call(self, mock_settings):
        """Test extraction of markdown ```tool format."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()

        state = make_initial_state("test", "test-sid")
        state["_last_llm_response"] = '```tool\n{"tool": "t03_web_search", "input": {"query": "test"}}\n```'

        tool_name, tool_input, is_ui = orch._extract_tool_call(state)
        assert tool_name == "t03_web_search"
        assert tool_input is not None
        assert tool_input.get("query") == "test"

    def test_no_tool_call(self, mock_settings):
        """Test when no tool call is present."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()

        state = make_initial_state("test", "test-sid")
        state["_last_llm_response"] = "I'll answer directly."

        tool_name, tool_input, is_ui = orch._extract_tool_call(state)
        assert tool_name is None

    def test_extract_with_invalid_json(self, mock_settings):
        """Test extraction with malformed JSON."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()

        state = make_initial_state("test", "test-sid")
        state["_last_llm_response"] = "TOOL CALL: t01_system_command\nPARAMS: not-json"

        tool_name, tool_input, is_ui = orch._extract_tool_call(state)
        assert tool_name == "t01_system_command"
        assert tool_input is not None
        assert "raw" in tool_input


class TestResponseFormatting:
    """Tests for response formatting logic."""

    @pytest.mark.asyncio
    async def test_response_format_with_output(self, mock_settings):
        """Test formatting response from tool outputs."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()

        state = make_initial_state("test", "test-sid")
        state["tool_outputs"] = {
            "tool_1": json.dumps({"success": True, "result": "File list retrieved"}),
        }

        result = await orch._run_response_format(state)
        assert result.get("final_response") is not None
        assert "File list" in result["final_response"]

    @pytest.mark.asyncio
    async def test_response_format_empty(self, mock_settings):
        """Test formatting response when no outputs exist."""
        from nexus_brain.orchestrator import get_orchestrator
        orch = get_orchestrator()

        state = make_initial_state("test", "test-sid")
        result = await orch._run_response_format(state)
        assert result.get("final_response") is not None
        assert "completed" in result["final_response"].lower()