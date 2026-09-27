"""Regression coverage for onboarding navigation and chat context integration."""

import ast
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_brain.agent_state import make_initial_state
from nexus_brain.context_builder import ContextBuilder
from nexus_brain.llm_router import LLMResponse
from nexus_brain.orchestrator import AgentOrchestrator
from nexus_memory.memory_manager import MemoryManager
from nexus_ui.custom_hud import NexusHUD
from nexus_ui.onboarding import OnboardingWizard


def test_wizard_buttons_visit_all_screens():
    import customtkinter as ctk

    wizard = OnboardingWizard()
    completed = MagicMock()
    wizard.set_on_complete(completed)

    callback_errors = []

    def exercise(window):
        window.withdraw()
        window.tk.createcommand("bgerror", callback_errors.append)
        try:
            assert wizard._current_screen == 0
            wizard._next_button.invoke()
            assert wizard._current_screen == 1
            entry = wizard._fields["GROQ_API_KEY"]
            entry.delete(0, "end")
            entry.insert(0, "test-key")
            wizard._next_button.invoke()
            assert wizard._current_screen == 2
            wizard._back_button.invoke()
            assert wizard._current_screen == 1
            assert wizard._fields["GROQ_API_KEY"].get() == "test-key"
            wizard._next_button.invoke()
            wizard._next_button.invoke()
            assert wizard._current_screen == 3
            assert wizard._progress_label.cget("text") == "Step 4 of 4"
            saved.assert_not_called()
            completed.assert_not_called()
            # A physical click schedules a widget-owned animation callback;
            # invoke() alone does not exercise that lifecycle.
            wizard._next_button._clicked()
            saved.assert_called_once()
            completed.assert_called_once()
            assert wizard._closed
            assert not window.tk.call("after", "info")
            wizard._next_screen()
            wizard._back_screen()
            wizard._close_window()
            saved.assert_called_once()
            completed.assert_called_once()
        finally:
            try:
                if window.winfo_exists():
                    wizard._close_window()
            except Exception:
                pass

    with patch.object(wizard, "_save_config") as saved, patch.object(ctk.CTk, "mainloop", exercise):
        wizard._run_ctk_wizard()

    # Process events in the next root, as happens when the HUD starts.
    import time
    next_window = ctk.CTk()
    next_window.withdraw()
    try:
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            next_window.update()
            time.sleep(0.01)
        assert callback_errors == []
    finally:
        for timer in next_window.tk.call("after", "info"):
            next_window.tk.call("after", "cancel", timer)
        next_window.destroy()


@pytest.mark.parametrize("failed", [False, True])
def test_chat_context_and_actual_input_callback(failed):
    # Keep the real memory adapter, prompt builder, and orchestrator nodes.
    store = MagicMock()
    store.query.side_effect = [
        {"metadatas": [[{"task": "greet", "outcome": "said hello"}, None]]},
        {"metadatas": [[{"fact": "prefers concise replies"}]]},
    ]
    with patch("nexus_memory.memory_manager.get_vector_store", return_value=store), \
         patch("nexus_memory.memory_manager.get_session_context"):
        memory = MemoryManager()
    with patch("nexus_brain.context_builder.get_memory_manager", return_value=memory):
        builder = ContextBuilder()
    orchestrator = AgentOrchestrator()
    orchestrator._context_builder = builder
    orchestrator._tool_executor = None
    orchestrator._session_context = MagicMock()
    orchestrator._session_context.get.return_value = []
    orchestrator._llm_router = MagicMock()
    orchestrator._llm_router.generate = AsyncMock(return_value=LLMResponse(
        content="" if failed else "Hello! How can I help?",
        provider="test", model="test", success=not failed,
        error="Ollama is unavailable" if failed else None,
    ))
    states = []

    async def run(text):
        state = await orchestrator._run_context_load(make_initial_state(user_input=text, session_id="test-session"))
        assert not state["errors"]
        assert "said hello" in state["system_prompt"]
        assert "prefers concise replies" in state["system_prompt"]
        state = await orchestrator._run_agent_reason(state)
        state = await orchestrator._run_response_format(state)
        states.append(state)
        return state

    # Compile the actual nested main callback without running boot/UI side effects.
    source = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    callback = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "on_user_input")
    hud = NexusHUD()
    namespace = {"orchestrator": MagicMock(run=run), "hud": hud,
                 "logger": logging.getLogger("test.chat")}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), "exec"), namespace)
    hud.set_input_callback(namespace["on_user_input"])
    hud.submit_input("hello")
    response = hud.output_queue.get_nowait()
    assert response == states[0]["final_response"]
    assert "Ollama is unavailable" in response if failed else response == "Hello! How can I help?"
    assert "conversation_history" not in response
    store.query.assert_any_call("agent_memory", "hello", 3)
    store.query.assert_any_call("user_preferences", "hello", 3)


@pytest.mark.parametrize("results", [{}, {"metadatas": None}, {"metadatas": [[]]},
                                     {"metadatas": [[None, {}, {"fact": None}]]}])
def test_empty_and_incomplete_memory_records(results):
    assert MemoryManager._memory_records(results, ("fact",)) == []


def test_save_preserves_other_settings(tmp_path, monkeypatch):
    from dotenv import dotenv_values
    import nexus_ui.onboarding as onboarding

    wizard = OnboardingWizard()
    wizard._config = {"GROQ_API_KEY": "", "OLLAMA_MODEL": "test-model",
                      "UI_THEME": "midnight_blue"}
    env_file = tmp_path / ".env"
    env_file.write_text("# Keep this comment\nOTHER_SETTING=keep\nGROQ_API_KEY=old\n", encoding="utf-8")
    monkeypatch.setattr(onboarding, "APP_ROOT", tmp_path)
    with patch.dict("os.environ"), patch.object(onboarding, "get_settings") as settings, \
         patch.object(wizard._theme, "set_theme") as theme:
        wizard._save_config()
        settings.cache_clear.assert_called_once()
        theme.assert_called_once_with("midnight_blue")
    saved = dotenv_values(env_file)
    assert saved["OTHER_SETTING"] == "keep"
    assert saved["GROQ_API_KEY"] == ""
    assert saved["OLLAMA_MODEL"] == "test-model"
    assert "# Keep this comment" in env_file.read_text(encoding="utf-8")

