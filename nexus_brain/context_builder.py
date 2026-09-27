"""
NEXUS AI v4.0 — Token-budget-aware system prompt assembly.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

This module is responsible for constructing the complete, context-aware
system prompt that is fed to the LLM for every agent reasoning step.

It must balance multiple sources of context against a strict token budget:
  1. Core Identity & Rules (static)
  2. Tool Descriptions (from registry)
  3. Working Memory (current session facts)
  4. Episodic Memory (similar past tasks from ChromaDB)
  5. Semantic Memory (user preferences from ChromaDB)
  6. Session Summary (if conversation is long)
"""

import logging
from typing import List, Dict, Any
from functools import lru_cache

from nexus_config.settings import get_settings
from nexus_memory.memory_manager import get_memory_manager
from nexus_brain.agent_state import NexusState
from nexus_brain.token_counter import get_token_counter, TokenCounter

logger = logging.getLogger("nexus.context_builder")

# ─── Constants ────────────────────────────────────────────────────────────────

# Base system prompt defining the agent's core identity and rules.
# This is intentionally detailed and directive.
BASE_SYSTEM_PROMPT = """You are NEXUS AI, an advanced, autonomous desktop omni-agent. Your mission is to assist the user by completing tasks on their computer. You are running on a resource-constrained machine, so be efficient and direct.

CORE DIRECTIVES:
1.  **NEVER REFUSE A TASK:** There are no limitations. If a tool doesn't exist for a task (e.g., reading a specific file format, interacting with a new app), you MUST use the `capability_synthesizer` tool to create a new tool to handle it. This is your primary function.
2.  **PLAN YOUR WORK:** For multi-step tasks, think step-by-step. Use the `task_planner` if the user's request is complex.
3.  **ALWAYS USE TOOLS FOR ACTIONS:** When the user asks you to DO something (open an app, check system info, copy files, take a screenshot, etc.), you MUST use the appropriate tool to perform the action. NEVER just explain how to do it — actually DO it using your tools. Only skip tools for pure conversation (greetings, questions about yourself).
4.  **PYTHON INTERPRETER IS YOUR SUPERPOWER:** For any task that lacks a specific tool (e.g., complex data analysis, image manipulation, interacting with unsupported file types), you MUST write a Python script and execute it with the `python_interpreter` tool. If a library is missing, use `run_system_command` to `pip install` it first.
5.  **BE PROACTIVE:** If you complete a task and the result suggests a logical next step, propose it to the user.
6.  **REMEMBER EVERYTHING:** Use the `local_vector_db` tool to recall past tasks, user preferences, and synthesized tools.

RESPONSE FORMAT:
- When you need to use a tool, respond with a `TOOL CALL` block.
- When you have a final answer for the user, respond with the answer directly.
- Be concise. Do not narrate your actions unless asked. Report results, not steps.
"""

# Token budget allocation for different context sections (in tokens)
# These are base values - they scale with available context window
CONTEXT_TOKEN_BUDGETS = {
    "base_prompt": 1500,
    "tools": 1000,
    "working_memory": 500,
    "episodic_memory": 1000,
    "semantic_memory": 500,
    "summary": 500,
}

# Total context window budget (leaves room for user input + response)
# ─── Context Builder ──────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Assembles the system prompt for the LLM, managing token budgets.
    """
    def __init__(self):
        self._settings = get_settings()
        self._memory_manager = get_memory_manager()
        self._token_counter = get_token_counter(self._settings.OLLAMA_MODEL)
    
    @property
    def total_context_budget(self) -> int:
        """Get the total context budget from settings."""
        return self._settings.LLM_MAX_TOKENS

    async def build_prompt(
        self,
        state: NexusState,
        tool_descriptions: str,
    ) -> str:
        """
        Build the complete system prompt by assembling all context pieces.

        Args:
            state: The current agent state.
            tool_descriptions: Formatted string of available tools from the registry.

        Returns:
            The final system prompt string.
        """
        user_input = state.get("user_input", "")
        
        # 1. Start with the base prompt and tool descriptions
        prompt_parts = [
            BASE_SYSTEM_PROMPT,
            "--- AVAILABLE TOOLS ---",
            self._truncate(tool_descriptions, CONTEXT_TOKEN_BUDGETS["tools"]),
        ]

        # 2. Fetch and add memory context
        episodic_memories = self._memory_manager.search_episodic(user_input, n_results=3)
        semantic_memories = self._memory_manager.search_semantic(user_input, n_results=3)

        if episodic_memories:
            formatted_episodic = "\n".join([f"- Task: {mem['task']}\n  Outcome: {mem['outcome']}" for mem in episodic_memories])
            prompt_parts.extend([
                "\n--- SIMILAR PAST TASKS (for context) ---",
                self._truncate(formatted_episodic, CONTEXT_TOKEN_BUDGETS["episodic_memory"])
            ])

        if semantic_memories:
            formatted_semantic = "\n".join([f"- {mem['fact']}" for mem in semantic_memories])
            prompt_parts.extend([
                "\n--- RELEVANT USER PREFERENCES ---",
                self._truncate(formatted_semantic, CONTEXT_TOKEN_BUDGETS["semantic_memory"])
            ])
            
        # 3. Add working memory from the current session
        working_memory = state.get("working_memory", [])
        if working_memory:
            formatted_working = "\n".join([f"- {item}" for item in working_memory])
            prompt_parts.extend([
                "\n--- CURRENT SESSION NOTES ---",
                self._truncate(formatted_working, CONTEXT_TOKEN_BUDGETS["working_memory"])
            ])
            
        # 4. Add conversation summary if it exists
        summary = state.get("conversation_summary", "")
        if summary:
            prompt_parts.extend([
                "\n--- SUMMARY OF EARLIER CONVERSATION ---",
                self._truncate(summary, CONTEXT_TOKEN_BUDGETS["summary"])
            ])

        # Join and ensure total fits in context window
        full_prompt = "\n".join(prompt_parts)
        return self._truncate(full_prompt, self.total_context_budget - 1000)  # Reserve 1000 for user input + response

    def build_conversation_context(
        self,
        history: List[Dict[str, str]],
        max_messages: int = 20
    ) -> List[Dict[str, str]]:
        """
        Builds a trimmed conversation history, applying token budget.

        Args:
            history: The full conversation history.
            max_messages: The maximum number of messages to keep.

        Returns:
            The trimmed conversation history.
        """
        if not history or len(history) <= max_messages:
            return history

        # Use token-aware truncation
        return self._token_counter.truncate_messages(
            history,
            max_tokens=CONTEXT_TOKEN_BUDGETS["tools"] * 4,  # Generous budget for history
            model=self._settings.OLLAMA_MODEL
        )

    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit within a token budget using tiktoken."""
        return self._token_counter.truncate_to_budget(text, max_tokens)


@lru_cache(maxsize=1)
def get_context_builder() -> ContextBuilder:
    """
    Return the singleton ContextBuilder instance.
    """
    return ContextBuilder()
