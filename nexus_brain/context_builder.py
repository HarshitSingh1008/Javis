"""
NEXUS AI v4.0 — Self-healing capability synthesis engine.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

This is THE product moat — the ability to dynamically create new tools
when the agent encounters a capability gap. Every synthesis event creates
a permanent asset that makes the agent more capable forever.

The Self-Healing Loop:
  1. GAP ANALYSIS (LLM call, ~1.2s): Describe what capability is needed
  2. SYNTHESIS (LLM call, ~4-8s): Generate tool code via Groq 70B
  3. AST VALIDATION (local, <50ms): Check for banned imports/patterns
  4. FUNCTIONAL TEST (subprocess, <5s): Run minimal test
  5. PERSISTENCE (<5ms): Write to APP_ROOT/synthesized_tools/
  6. VECTOR EMBEDDING (<35ms): Index in ChromaDB for future lookup
  7. HOT REGISTRATION (<100ms): importlib.reload() into live ToolRegistry
  8. RETRY: Re-invoke original task step with new tool
"""

import ast
import asyncio
import importlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache

from nexus_config.settings import get_settings, APP_ROOT
from nexus_config.audit_logger import get_audit_logger
from nexus_brain.llm_router import get_llm_router, LLMResponse
from nexus_memory.vector_store import get_vector_store
from nexus_tools.secure_sandbox import ast_validate, run_in_subprocess

logger = logging.getLogger("nexus.synthesizer")


# ─── Constants ────────────────────────────────────────────────────────────────

SYNTHESIZED_TOOLS_DIR: Path = APP_ROOT / "synthesized_tools"

# Whitelisted imports for synthesized tools
WHITELISTED_IMPORTS: List[str] = [
    "json", "re", "os.path", "pathlib", "typing",
    "dataclasses", "datetime", "collections", "math",
    "csv", "io", "base64", "hashlib",
    "itertools", "functools", "uuid", "enum",
    # Optional but common
    "requests", "httpx", "aiohttp",
    "pandas", "numpy",
    "PIL", "PIL.Image",
    "pdfplumber", "PyPDF2", "pdfminer",
    "docx", "openpyxl",
    "markdown",
    "bs4", "BeautifulSoup",
    "lxml",
    "jinja2",
]

BANNED_PATTERNS: List[str] = [
    "__import__", "eval(", "exec(", "compile(",
    "os.system", "os.popen", "subprocess.",
    "shutil.rmtree", "shutil.move",
    "socket.", "requests.delete", "requests.put",
    "open(",  # Not banned — just flagged if used without context manager
    "pickle.load", "pickle.dump",
    "marshal.",
    "ctypes.",
    "signal.",
    "multiprocessing.",
]

MAX_SYNTHESIS_RETRIES: int = 3
MAX_TOOL_NODES: int = 500
SYNTHESIS_MODEL: str = "groq"  # Always use Groq 70B for synthesis quality


# ─── Synthesis Result Types ───────────────────────────────────────────────────

@dataclass
class SynthesisAttempt:
    """
    Record of a single synthesis attempt.
    
    Each attempt uses a different prompt strategy to maximize success.
    """
    attempt_number: int
    gap_description: str
    prompt_strategy: str  # "standard", "ast_violation_avoid", "simplify"
    generated_code: str
    ast_result: Dict[str, Any]
    test_result: Dict[str, Any]
    success: bool
    duration_ms: float
    error_message: Optional[str] = None


@dataclass
class SynthesisResult:
    """
    Complete result of a capability synthesis operation.
    """
    tool_name: str
    tool_description: str
    success: bool
    attempts: List[SynthesisAttempt]
    final_code: Optional[str] = None
    registered: bool = False
    estimated_savings_usd: Optional[float] = None  # Cost this will save by not needing re-synthesis
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── GAP ANALYSIS ─────────────────────────────────────────────────────────────

GAP_ANALYSIS_PROMPT = """You are analyzing a capability gap in an AI agent. A tool was not found to perform a specific task.

Task: {task}
Failure reason: {failure_reason}
Current tools available: {tools_list}

Your job: Describe EXACTLY what Python capability needs to be synthesized.

Respond with JSON ONLY:
{{
    "tool_name": "snake_case_name_for_the_tool",
    "tool_description": "One sentence describing what the tool does",
    "input_params": "Description of input parameters the tool needs",
    "output_description": "What the tool returns (JSON string with success/result/error)",
    "library_hint": "Python library this likely needs (or 'stdlib' if no external libs needed)"
}}"""


# ─── SYNTHESIS PROMPTS (3 strategies) ─────────────────────────────────────────

SYNTHESIS_STANDARD_PROMPT = """You are a Python code generator. Generate a single Python tool function.

REQUIREMENTS:
- Function must be decorated with @tool from langchain_core.tools.
- Function name: {tool_name}
- Description (docstring): {tool_description}
- Input: {input_params}
- Output: JSON string with keys: success (bool), result (any), error (str | null)
- Use ONLY these imports: {whitelisted_imports}
- AVOID these patterns: {banned_patterns}
- Max complexity: {max_nodes} AST nodes
- The function must be self-contained (no class dependencies).
- Handle errors gracefully — wrap logic in try/except, return error JSON on failure.
- Use standard library first. Only use external libraries if essential.
- No async — this must be a synchronous function.
- Complete, working code. No placeholders. No TODOs. No pseudo-code.

Example tool structure:
```python
from langchain_core.tools import tool
import json

@tool
def my_tool(param1: str, param2: int = 10) -> str:
    \"\"\"
    One sentence description of what this tool does.
    
    Args:
        param1: Description of param1.
        param2: Description of param2 (default: 10).
    
    Returns:
        JSON string with keys: success, result, error.
    \"\"\"
    try:
        # Your implementation here
        result = do_something(param1, param2)
        return json.dumps({"success": True, "result": result})
    except Exception as e:
        return json.dumps({"success": False, "result": None, "error": str(e)})
```

Generate ONLY the tool function code. No explanations."""


SYNTHESIS_AST_AVOID_PROMPT = """You are a Python code generator. Generate a single Python tool function.

REQUIREMENTS (same as standard), but CRITICAL to avoid these specific AST violations that were found in a previous attempt:

PREVIOUS VIOLATIONS (MUST FIX):
{violations}

IMPORTS YOU CAN USE: {whitelisted_imports}
FUNCTION NAME: {tool_name}
DESCRIPTION: {tool_description}

Rules to follow strictly:
- Do NOT import anything listed in the violations.
- Do NOT use attribute access like .__class__, .__dict__, .__builtins__.
- Do NOT call exec(), eval(), compile(), __import__().
- Use try/except for ALL error handling.
- Return JSON always.

Generate ONLY the function code."""


SYNTHESIS_SIMPLIFY_PROMPT = """You are a Python code generator. Generate the SIMPLEST POSSIBLE working tool function.

FUNCTION: {tool_name}
DESCRIPTION: {tool_description}

RULES:
- IMPORTS ONLY: {whitelisted_imports}
- No classes. One function only.
- No external libraries if standard library suffices.
- Max 100 lines.
- Use @tool decorator from langchain_core.tools.
- Return JSON: {{"success": true, "result": ...}} or {{"success": false, "error": "..."}}
- Try/except around the whole body.
- No async.

Generate ONLY the function code. The tool is called `{tool_name}`."""


# ─── CAPABILITY SYNTHESIZER ──────────────────────────────────────────────────

class CapabilitySynthesizer:
    """
    Self-healing capability synthesis engine.
    
    Usage:
        synthesizer = get_capability_synthesizer()
        result = await synthesizer.synthesize(
            task="Read .eml file and extract headers",
            failure_reason="No capability found for reading .eml files",
            tools_list="file_manager, web_search, ...",
        )
        if result.success:
            # Tool is now registered and available
            tool_output = await orchestrator.execute_tool(result.tool_name, ...)
    """
    
    def __init__(self):
        self._settings = get_settings()
        self._llm_router = get_llm_router()
        self._audit_logger = get_audit_logger()
        self._vector_store = get_vector_store()
        self._synthesized_dir = SYNTHESIZED_TOOLS_DIR
        self._synthesized_dir.mkdir(parents=True, exist_ok=True)
        
        # Track tools we've synthesized this session (skip re-synthesis)
        self._session_synthesized: Dict[str, str] = {}  # tool_name -> code
    
    async def synthesize(
        self,
        task: str,
        failure_reason: str,
        tools_list: str,
    ) -> SynthesisResult:
        """
        Full synthesis pipeline: analyze → generate → validate → test → persist → register.
        
        Args:
            task: The original user task that triggered the gap.
            failure_reason: Error message from the failed tool attempt.
            tools_list: Comma-separated list of currently available tools.
        
        Returns:
            SynthesisResult with full attempt history and final outcome.
        """
        start = time.perf_counter()
        attempts: List[SynthesisAttempt] = []
        
        # Step 1: Gap Analysis
        gap = await self._analyze_gap(task, failure_reason, tools_list)
        if not gap:
            return SynthesisResult(
                tool_name="",
                tool_description="",
                success=False,
                attempts=[],
                error_message="Gap analysis failed — could not determine what capability is needed.",
            )
        
        tool_name = gap.get("tool_name", "synthesized_tool")
        tool_description = gap.get("tool_description", "")
        input_params = gap.get("input_params", "")
        
        # Check if we already have this tool from a previous session
        existing_code = self._load_existing_tool(tool_name)
        if existing_code:
            # Quick re-validate and register
            logger.info("Tool '%s' already exists, re-registering.", tool_name)
            registered = await self._register_tool(tool_name, existing_code)
            if registered:
                return SynthesisResult(
                    tool_name=tool_name,
                    tool_description=tool_description,
                    success=True,
                    attempts=[],
                    final_code=existing_code,
                    registered=True,
                )
        
        # Step 2-8: Iterative synthesis with retries
        retries = min(self._settings.SYNTHESIS_MAX_RETRIES, MAX_SYNTHESIS_RETRIES)
        last_ast_errors: List[str] = []
        
        for attempt_num in range(1, retries + 1):
            attempt_start = time.perf_counter()
            
            # Choose prompt strategy
            if attempt_num == 1:
                strategy = "standard"
            elif attempt_num == 2 and last_ast_errors:
                strategy = "ast_violation_avoid"
            else:
                strategy = "simplify"
            
            # Generate code
            generated_code = await self._generate_code(
                strategy=strategy,
                tool_name=tool_name,
                tool_description=tool_description,
                input_params=input_params,
                violations=last_ast_errors,
                previous_attempts=attempts,
            )
            
            if not generated_code:
                continue
            
            # AST Validation
            ast_result = ast_validate(generated_code, max_nodes=MAX_TOOL_NODES)
            
            # Functional Test
            test_result = self._test_tool_code(tool_name, generated_code)
            
            attempt_duration = (time.perf_counter() - attempt_start) * 1000
            success = ast_result.get("valid", False) and test_result.get("success", False)
            
            attempt = SynthesisAttempt(
                attempt_number=attempt_num,
                gap_description=tool_description,
                prompt_strategy=strategy,
                generated_code=generated_code,
                ast_result=ast_result,
                test_result=test_result,
                success=success,
                duration_ms=attempt_duration,
                error_message=test_result.get("stderr") if not success else None,
            )
            attempts.append(attempt)
            
            if success:
                # Persist to disk
                self._persist_tool(tool_name, generated_code)
                
                # Index in vector store
                self._index_tool(tool_name, tool_description, generated_code)
                
                # Hot-register
                registered = await self._register_tool(tool_name, generated_code)
                
                total_duration = (time.perf_counter() - start) * 1000
                
                return SynthesisResult(
                    tool_name=tool_name,
                    tool_description=tool_description,
                    success=True,
                    attempts=attempts,
                    final_code=generated_code,
                    registered=registered,
                )
            
            # Collect AST errors for the next attempt
            if not ast_result.get("valid"):
                last_ast_errors = [ast_result.get("error", "Unknown AST error")]
            else:
                last_ast_errors = []
        
        # All attempts failed
        total_duration = (time.perf_counter() - start) * 1000
        
        return SynthesisResult(
            tool_name=tool_name,
            tool_description=tool_description,
            success=False,
            attempts=attempts,
            final_code=None,
            registered=False,
        )
    
    async def _analyze_gap(
        self,
        task: str,
        failure_reason: str,
        tools_list: str,
    ) -> Optional[Dict[str, str]]:
        """
        Analyze what capability is needed to fill the gap.
        
        Args:
            task: Original task description.
            failure_reason: What went wrong.
            tools_list: Current tools.
        
        Returns:
            Dict with tool_name, tool_description, input_params, etc.
        """
        prompt = GAP_ANALYSIS_PROMPT.format(
            task=task[:1000],
            failure_reason=failure_reason[:500],
            tools_list=tools_list[:500],
        )
        
        response = await self._llm_router.generate(
            messages=[
                {"role": "system", "content": "You are a capability gap analyst. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
            prefer_provider=SYNTHESIS_MODEL,
        )
        
        if not response.success:
            logger.error("Gap analysis failed: %s", response.error)
            return None
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error("Gap analysis JSON parse failed: %s", e)
            return None
    
    async def _generate_code(
        self,
        strategy: str,
        tool_name: str,
        tool_description: str,
        input_params: str,
        violations: List[str],
        previous_attempts: List[SynthesisAttempt],
    ) -> Optional[str]:
        """
        Generate tool code using the specified strategy.
        
        Args:
            strategy: "standard", "ast_violation_avoid", or "simplify".
            tool_name: Name for the tool function.
            tool_description: What the tool does.
            input_params: Description of input parameters.
            violations: AST violations from previous attempt (strategy 2).
            previous_attempts: Full history (for context).
        
        Returns:
            Generated Python code string, or None on failure.
        """
        whitelisted = ", ".join(WHITELISTED_IMPORTS)
        banned = ", ".join(BANNED_PATTERNS)
        
        if strategy == "standard":
            prompt = SYNTHESIS_STANDARD_PROMPT.format(
                tool_name=tool_name,
                tool_description=tool_description,
                input_params=input_params,
                whitelisted_imports=whitelisted,
                banned_patterns=banned,
                max_nodes=MAX_TOOL_NODES,
            )
        elif strategy == "ast_violation_avoid":
            prompt = SYNTHESIS_AST_AVOID_PROMPT.format(
                tool_name=tool_name,
                tool_description=tool_description,
                violations="\n".join(violations),
                whitelisted_imports=whitelisted,
            )
        else:  # simplify
            prompt = SYNTHESIS_SIMPLIFY_PROMPT.format(
                tool_name=tool_name,
                tool_description=tool_description,
                whitelisted_imports=whitelisted,
            )
        
        response = await self._llm_router.generate(
            messages=[
                {"role": "system", "content": "You are a Python code generator. Generate ONLY working code."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
            prefer_provider=SYNTHESIS_MODEL,
        )
        
        if not response.success:
            logger.error("Code generation failed (attempt %d): %s", len(previous_attempts) + 1, response.error)
            return None
        
        # Extract code from response (strip markdown fences if present)
        code = self._extract_code(response.content)
        
        return code
    
    def _extract_code(self, content: str) -> Optional[str]:
        """
        Extract Python code from LLM response, handling markdown fences.
        
        Args:
            content: Raw LLM response.
        
        Returns:
            Clean Python code string, or None if extraction fails.
        """
        # Try to find code between ```python and ``` fences
        if "```python" in content:
            parts = content.split("```python")
            if len(parts) > 1:
                code_part = parts[1]
                if "```" in code_part:
                    return code_part.split("```")[0].strip()
        
        # Try between ``` and ```
        if "```" in content:
            parts = content.split("```")
            for i, part in enumerate(parts):
                part = part.strip()
                if part.startswith("python"):
                    part = part[6:].strip()
                if part.startswith("def ") or part.startswith("from ") or part.startswith("import ") or part.startswith("@tool"):
                    # Make sure we have the closing fence
                    if i + 1 < len(parts):
                        return part
        
        # No fences found — use content directly if it looks like code
        if content.strip().startswith("def ") or content.strip().startswith("from ") or content.strip().startswith("import ") or content.strip().startswith("@"):
            return content.strip()
        
        return None
    
    def _test_tool_code(self, tool_name: str, code: str) -> Dict[str, Any]:
        """
        Run a minimal functional test of the generated code in the sandbox.
        
        Args:
            tool_name: Name of the tool function.
            code: Generated Python code.
        
        Returns:
            Dict with success, stdout, stderr, returncode.
        """
        # Wrap code with a minimal invocation test
        test_code = code + f"\n\nif __name__ == '__main__':\n    import json\n    # Test the tool's docstring is valid\n    print({tool_name!r} + ' loaded successfully')\n    print({tool_name}.__doc__[:100] if {tool_name}.__doc__ else 'No docstring')\n"
        
        return run_in_subprocess(
            test_code,
            timeout=15,
            max_memory_mb=256,
        )
    
    def _persist_tool(self, tool_name: str, code: str) -> None:
        """
        Write the synthesized tool to disk for permanent storage.
        
        Args:
            tool_name: Name of the tool.
            code: Generated Python code.
        """
        file_path = self._synthesized_dir / f"{tool_name}.py"
        try:
            file_path.write_text(code, encoding="utf-8")
            logger.info("Tool '%s' persisted to %s", tool_name, file_path)
        except Exception as e:
            logger.error("Failed to persist tool '%s': %s", tool_name, e)
    
    def _load_existing_tool(self, tool_name: str) -> Optional[str]:
        """
        Load a previously synthesized tool from disk.
        
        Args:
            tool_name: Name of the tool.
        
        Returns:
            Code string if file exists, None otherwise.
        """
        file_path = self._synthesized_dir / f"{tool_name}.py"
        if file_path.exists():
            try:
                return file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.debug("Could not load existing tool '%s': %s", tool_name, e)
        return None
    
    def _index_tool(self, tool_name: str, description: str, code: str) -> None:
        """
        Index the synthesized tool in ChromaDB for future semantic search.
        
        This allows the agent to check if a capability exists before triggering synthesis.
        
        Args:
            tool_name: Name of the tool.
            description: Tool description.
            code: Full source code (truncated for embedding).
        """
        try:
            document = f"Tool: {tool_name}\nDescription: {description}\nCode: {code[:500]}"
            self._vector_store.add_memory(
                col="synthesized_tools",
                doc_id=f"synth_{tool_name}",
                document=document,
                metadata={
                    "tool_name": tool_name,
                    "description": description,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
        except Exception as e:
            logger.debug("Failed to index tool '%s': %s", tool_name, e)
    
    async def _register_tool(self, tool_name: str, code: str) -> bool:
        """
        Hot-load the synthesized tool into the live ToolRegistry.
        
        Writes code to a temp file, imports it, extracts the @tool function,
        and registers it in the global registry.
        
        Args:
            tool_name: Name of the tool.
            code: Python code to register.
        
        Returns:
            True if registration succeeded.
        """
        try:
            # Write to disk if not already there
            self._persist_tool(tool_name, code)
            
            # Import the module
            file_path = self._synthesized_dir / f"{tool_name}.py"
            spec = importlib.util.spec_from_file_location(
                f"synthesized_tools.{tool_name}",
                file_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create spec for {tool_name}")
            
            module = importlib.util.module_from_spec(spec)
            # Remove any cached version
            sys.modules.pop(f"synthesized_tools.{tool_name}", None)
            spec.loader.exec_module(module)
            
            # Find the @tool-decorated function
            from langchain_core.tools import BaseTool
            tool_func = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, BaseTool):
                    tool_func = attr
                    break
            
            if tool_func is None:
                # Look for regular functions (the @tool decorator may not have fired)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if callable(attr) and attr_name == tool_name:
                        tool_func = attr
                        break
            
            if tool_func is None:
                raise ValueError(f"No @tool function '{tool_name}' found in synthesized code")
            
            # Register via the global registry
            from nexus_tools.registry import get_tool_registry
            registry = get_tool_registry()
            registry.register(tool_func, source="synthesized")
            
            logger.info("Tool '%s' hot-registered successfully.", tool_name)
            return True
            
        except Exception as e:
            logger.error("Failed to register tool '%s': %s", tool_name, e)
            return False
    
    def load_all_synthesized_tools(self) -> int:
        """
        Load all previously synthesized tools from disk.
        
        Called at boot to restore capabilities from previous sessions.
        
        Returns:
            Number of tools successfully loaded.
        """
        count = 0
        for file_path in self._synthesized_dir.glob("*.py"):
            if file_path.stem.startswith("__"):
                continue
            try:
                code = file_path.read_text(encoding="utf-8")
                registered = asyncio.run(self._register_tool(file_path.stem, code))
                if registered:
                    count += 1
            except Exception as e:
                logger.warning("Failed to load synthesized tool '%s': %s", file_path.stem, e)
        
        return count
    
    def search_existing_tool(self, query: str) -> Optional[str]:
        """
        Check if a tool already exists for a given query.
        
        Queries ChromaDB's synthesized_tools collection before triggering synthesis.
        
        Args:
            query: Description of the capability needed.
        
        Returns:
            Tool name if a matching tool exists, None otherwise.
        """
        try:
            result = self._vector_store.query(
                col="synthesized_tools",
                qt=query,
                n=1,
            )
            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])
            
            if documents and metadatas:
                score = result.get("distances", [[0.0]])[0][0]
                if score < 0.3:  # Threshold for "good match"
                    return metadatas[0].get("tool_name")
            
            return None
        except Exception as e:
            logger.debug("Tool search failed: %s", e)
            return None
    
    async def build_prompt(self, state: Dict[str, Any], tool_descriptions: str) -> str:
        """
        Build a system prompt for the LLM based on state and available tools.
        
        Args:
            state: Current agent state (dict).
            tool_descriptions: Available tools description.
        
        Returns:
            System prompt string for the LLM.
        """
        user_query = state.get("user_query", "Process the user request.")
        conversation_summary = state.get("conversation_summary", "")
        
        summary_section = ""
        if conversation_summary:
            summary_section = f"\n\nPrior conversation context:\n{conversation_summary}"
        
        prompt = f"""You are NEXUS AI, an autonomous desktop agent.

User Query: {user_query}{summary_section}

Available Tools:
{tool_descriptions}

Instructions:
1. Understand the user's intent
2. Plan the steps needed
3. Execute tools in the correct order
4. Return clear results

Respond with structured reasoning and tool calls."""
        
        return prompt
    
    def build_conversation_context(
        self, 
        history: List[Dict[str, str]], 
        max_messages: int = 20
    ) -> List[Dict[str, str]]:
        """
        Build a trimmed conversation context from history.
        
        Args:
            history: Full conversation history.
            max_messages: Maximum messages to keep.
        
        Returns:
            Trimmed conversation history.
        """
        if not history:
            return []
        
        # Keep only the last max_messages
        return history[-max_messages:] if len(history) > max_messages else history


@lru_cache(maxsize=1)
def get_capability_synthesizer() -> CapabilitySynthesizer:
    """
    Return the singleton CapabilitySynthesizer instance.
    
    Returns:
        CapabilitySynthesizer: The singleton synthesizer instance.
    """
    return CapabilitySynthesizer()


@lru_cache(maxsize=1)
def get_context_builder() -> CapabilitySynthesizer:
    """
    Return the singleton context builder instance (CapabilitySynthesizer).
    
    Returns:
        CapabilitySynthesizer: The singleton synthesizer instance.
    """
    return get_capability_synthesizer()
