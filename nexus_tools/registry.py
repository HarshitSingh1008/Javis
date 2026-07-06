"""
NEXUS AI v4.0 — 22-tool LangChain registry + plugin loader + watchdog.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Central tool registry managing all 22 built-in tools and dynamically
loaded plugins. Supports hot-reload via watchdog file monitoring.

Key responsibilities:
  1. Register all 22 tools at boot
  2. Load and register plugins from APP_ROOT/plugins/
  3. Watch for file changes and hot-reload plugins
  4. Provide tool listing for the LLM system prompt
  5. Route tool calls to the correct implementation
  6. Enforce tool-level security checks
"""

import asyncio
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Awaitable, Set, Tuple
from functools import lru_cache

from nexus_config.settings import get_settings, APP_ROOT
from nexus_config.audit_logger import get_audit_logger, audited
from nexus_billing.tier_gate import requires_tier, check_tier_access, TIER_RANK
from nexus_billing.license_manager import get_license_manager

logger = logging.getLogger("nexus.registry")


# ─── Plugin Metadata ──────────────────────────────────────────────────────────

@dataclass
class PluginMetadata:
    """
    Metadata for a registered plugin or tool.
    
    Fields:
        name: Tool function name (snake_case).
        description: One-sentence description for LLM selection.
        source: "builtin" | "plugin" | "synthesized".
        version: Semantic version string.
        author: Plugin author name.
        tags: List of category keywords.
        module_path: Python module path.
        is_ui_operation: True if this tool uses PyAutoGUI (needs serialization).
        requires_confirmation: True if destructive (delete, send email, etc.).
        needs_network: True if this tool makes network requests.
        added_at: ISO timestamp of registration.
    """
    name: str
    description: str
    source: str  # "builtin" | "plugin" | "synthesized"
    version: str = "1.0.0"
    author: str = "NEXUS AI"
    tags: List[str] = field(default_factory=list)
    module_path: str = ""
    is_ui_operation: bool = False
    requires_confirmation: bool = False
    needs_network: bool = False
    added_at: str = ""


# ─── Tool Registry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central tool registry for NEXUS AI.
    
    All 22 built-in tools, plugins, and synthesized tools register here.
    The registry provides:
      - Tool lookup by name
      - Tool execution with error handling
      - Tool listing formatted for LLM system prompt
      - Plugin hot-reload via watchdog
    
    Usage:
        registry = get_tool_registry()
        registry.register(my_tool_func, source="plugin")
        result = await registry.execute("file_manager", {"path": "/tmp"})
        tools_list = registry.format_for_prompt()
    """
    
    def __init__(self):
        self._settings = get_settings()
        self._audit_logger = get_audit_logger()
        
        # Tool storage
        self._tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        self._lock = threading.RLock()
        
        # Watchdog for plugin hot-reload
        self._watchdog_running = False
        self._watchdog_thread: Optional[threading.Thread] = None
        self._last_reload_times: Dict[str, float] = {}
        
        # UI automation lock (shared with TaskPlanner)
        self._ui_lock = asyncio.Lock()
        
        # Track tool sources for audit
        self._tool_sources: Dict[str, str] = {}
        
        # Lazy import state for tools/tool modules
        self._tool_modules: Dict[str, Any] = {}
    
    # ── Tool Registration ─────────────────────────────────────────────────
    
    def register(
        self,
        tool_func: Callable,
        source: str = "builtin",
        metadata: Optional[PluginMetadata] = None,
    ) -> None:
        """
        Register a tool function in the registry.
        
        Args:
            tool_func: The tool function (preferably @tool-decorated from langchain).
            source: "builtin" | "plugin" | "synthesized".
            metadata: Optional PluginMetadata. Generated if not provided.
        
        Raises:
            ValueError: If tool_func has no name or a tool with that name is
                       already registered from a different source.
        """
        import datetime
        
        # If metadata is provided with a name, use that (most authoritative)
        if metadata and metadata.name:
            name = metadata.name
        else:
            # For LangChain @tool decorated functions (StructuredTool), check name first
            name = getattr(tool_func, "name", None) or getattr(tool_func, "__name__", None)
        
        if not name:
            raise ValueError("Tool function must have a __name__ attribute or metadata with a name.")
        
        doc = (tool_func.__doc__ or "No description available.").strip()
        # Take first line/sentence of docstring
        description = doc.split("\n")[0].strip()[:200]
        
        with self._lock:
            # Check conflicts
            existing = self._metadata.get(name)
            if existing and existing.source != source and source != "synthesized":
                logger.warning(
                    "Tool '%s' already registered from source '%s'. Overwriting from '%s'.",
                    name, existing.source, source,
                )
            
            self._tools[name] = tool_func
            self._tool_sources[name] = source
            
            if metadata:
                self._metadata[name] = metadata
            else:
                # Generate metadata
                is_ui = "click" in description.lower() or "type" in description.lower() or "screenshot" in description.lower() or "mouse" in description.lower()
                needs_confirm = "delete" in name or "remove" in name or "send" in name or "email" in name
                
                self._metadata[name] = PluginMetadata(
                    name=name,
                    description=description[:200],
                    source=source,
                    module_path=getattr(tool_func, "__module__", "unknown"),
                    is_ui_operation=is_ui,
                    requires_confirmation=needs_confirm,
                    added_at=datetime.datetime.now().isoformat(),
                )
            
            logger.info("Registered tool '%s' (source=%s)", name, source)
    
    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            name: Tool name to unregister.
        
        Returns:
            True if tool was found and removed.
        """
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                self._metadata.pop(name, None)
                self._tool_sources.pop(name, None)
                logger.info("Unregistered tool '%s'", name)
                return True
            return False
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool function by name."""
        with self._lock:
            return self._tools.get(name)
    
    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get metadata for a tool by name."""
        with self._lock:
            return self._metadata.get(name)
    
    def list_tools(self, source: Optional[str] = None) -> List[str]:
        """
        List registered tool names.
        
        Args:
            source: Filter by source ("builtin", "plugin", "synthesized").
                   None returns all tools.
        
        Returns:
            Sorted list of tool names.
        """
        with self._lock:
            if source:
                return sorted([
                    name for name, src in self._tool_sources.items()
                    if src == source
                ])
            return sorted(self._tools.keys())
    
    def tool_count(self) -> int:
        """Return the total number of registered tools."""
        with self._lock:
            return len(self._tools)
    
    # ── Tool Execution ────────────────────────────────────────────────────
    
    async def execute(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        is_ui: bool = False,
    ) -> str:
        """
        Execute a tool by name with the given input.
        
        Handles:
          1. Tool lookup
          2. UI serialization (lock for UI operations)
          3. Execution with timeout
          4. Error handling and audit logging
        
        Args:
            tool_name: Name of the tool to execute.
            tool_input: Dict of arguments for the tool.
            is_ui: If True, acquires the UI automation lock.
        
        Returns:
            JSON string with "success", "result", "error" keys.
        
        Raises:
            KeyError: If tool_name is not found.
            TimeoutError: If execution exceeds SANDBOX_TIMEOUT_SECONDS.
        """
        # Special internal commands
        if tool_name == "__list_tools__":
            return self.format_for_prompt()
        
        tool_func = self.get_tool(tool_name)
        if tool_func is None:
            return json.dumps({
                "success": False,
                "result": None,
                "error": f"Tool '{tool_name}' not found in registry. "
                         f"Available: {', '.join(self.list_tools()[:10])}...",
            })
        
        metadata = self.get_metadata(tool_name)
        requires_confirm = metadata and metadata.requires_confirmation
        is_ui_op = metadata and metadata.is_ui_operation
        
        audit_data = {
            "tool": tool_name,
            "input_keys": list(tool_input.keys()),
            "requires_confirmation": requires_confirm,
        }
        
        start = time.perf_counter()
        
        try:
            # UI serialization
            if is_ui or is_ui_op:
                async with self._ui_lock:
                    result = await self._call_tool(tool_func, tool_input)
            else:
                result = await self._call_tool(tool_func, tool_input)
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            # Audit success
            self._audit_logger.log(
                event_type="TOOL_SUCCESS",
                data=audit_data,
                module=f"nexus_tools.registry",
                function_name=tool_name,
                duration_ms=duration_ms,
                success=True,
            )
            
            return result
        
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            error_msg = f"{type(e).__name__}: {e}"
            
            # Audit failure
            self._audit_logger.log(
                event_type="TOOL_FAILURE",
                data=audit_data,
                module="nexus_tools.registry",
                function_name=tool_name,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )
            
            return json.dumps({
                "success": False,
                "result": None,
                "error": error_msg,
            })
    
    async def _call_tool(
        self,
        tool_func: Callable,
        tool_input: Dict[str, Any],
    ) -> str:
        """
        Call a tool function with the given input.
        
        Handles both sync and async tools, as well as @tool-decorated
        LangChain tools vs plain functions.
        
        Args:
            tool_func: The tool callable.
            tool_input: Dict of arguments.
        
        Returns:
            Tool output as a string.
        """
        # If it's a LangChain BaseTool, use its .invoke() method
        if hasattr(tool_func, "invoke") and not inspect.iscoroutinefunction(tool_func.invoke):
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: tool_func.invoke(tool_input)
            )
        
        # If it has an .ainvoke() method (async LangChain tool)
        if hasattr(tool_func, "ainvoke"):
            return await tool_func.ainvoke(tool_input)
        
        # Plain async function
        if asyncio.iscoroutinefunction(tool_func):
            return await tool_func(**tool_input)
        
        # Plain sync function — run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: tool_func(**tool_input)
        )
    
    def execute_sync(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> str:
        """
        Synchronous tool execution (for use in thread pool).
        
        Args:
            tool_name: Name of the tool.
            tool_input: Dict of arguments.
        
        Returns:
            JSON string result.
        """
        tool_func = self.get_tool(tool_name)
        if tool_func is None:
            return json.dumps({
                "success": False,
                "error": f"Tool '{tool_name}' not found.",
            })
        
        try:
            if asyncio.iscoroutinefunction(tool_func):
                return asyncio.run(tool_func(**tool_input))
            return tool_func(**tool_input)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    # ── Prompt Formatting ────────────────────────────────────────────────
    
    def format_for_prompt(self) -> str:
        """
        Format all registered tools for inclusion in the LLM system prompt.
        
        Returns:
            Formatted string like:
            file_manager: Read/write/copy/move/delete files on the local filesystem.
            web_search: Search the web using DuckDuckGo and Brave Search engines.
            ...
        """
        with self._lock:
            lines = []
            for name in sorted(self._tools.keys()):
                meta = self._metadata.get(name)
                if meta:
                    desc = meta.description
                else:
                    func = self._tools[name]
                    doc = (func.__doc__ or "No description").strip()
                    desc = doc.split("\n")[0][:200]
                lines.append(f"  • {name}: {desc}")
            
            return "\n".join(lines)
    
    def format_for_llm_json(self) -> str:
        """
        Format tools as JSON for LLM tool use.
        
        Returns:
            JSON array of tool descriptors.
        """
        import datetime
        
        with self._lock:
            tools = []
            for name in sorted(self._tools.keys()):
                meta = self._metadata.get(name)
                func = self._tools[name]
                
                # Get parameter info from signature or schema
                params: Dict[str, str] = {}
                try:
                    sig = inspect.signature(func)
                    for pname, param in sig.parameters.items():
                        if pname in ("kwargs", "args"):
                            continue
                        annotation = str(param.annotation) if param.annotation != inspect.Parameter.empty else "any"
                        default = f"={param.default}" if param.default != inspect.Parameter.empty else ""
                        params[pname] = f"{annotation}{default}"
                except (ValueError, TypeError):
                    params = {}
                
                tools.append({
                    "name": name,
                    "description": (meta.description if meta else (func.__doc__ or "").strip()[:200]),
                    "parameters": params,
                    "source": self._tool_sources.get(name, "unknown"),
                    "is_ui": meta.is_ui_operation if meta else False,
                    "requires_confirmation": meta.requires_confirmation if meta else False,
                })
            
            return json.dumps(tools, indent=2)
    
    # ── Plugin Loading ───────────────────────────────────────────────────
    
    def load_plugins_from_directory(self, plugins_dir: Optional[Path] = None) -> int:
        """
        Load all plugin modules from a directory.
        
        Each plugin module should define a `register(registry)` function.
        
        Args:
            plugins_dir: Directory containing plugin .py files.
                        Defaults to APP_ROOT/plugins/.
        
        Returns:
            Number of plugins successfully loaded.
        """
        if plugins_dir is None:
            plugins_dir = APP_ROOT / "plugins"
        
        if not plugins_dir.exists():
            plugins_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created plugins directory: %s", plugins_dir)
            return 0
        
        count = 0
        for file_path in sorted(plugins_dir.glob("*.py")):
            if file_path.stem.startswith("_"):
                continue
            
            try:
                self._load_plugin_file(file_path)
                count += 1
            except Exception as e:
                logger.warning("Failed to load plugin '%s': %s", file_path.stem, e)
        
        logger.info("Loaded %d plugins from %s", count, plugins_dir)
        return count
    
    def _load_plugin_file(self, file_path: Path) -> None:
        """
        Load a single plugin file.
        
        The plugin file must define a `register(registry)` function that
        returns a list of PluginMetadata.
        
        Args:
            file_path: Path to the plugin .py file.
        """
        module_name = f"nexus_plugins.{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        # Remove from cache if previously loaded
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        
        # Call register function
        if hasattr(module, "register"):
            register_func = getattr(module, "register")
            if callable(register_func):
                metadatas = register_func(self)
                if isinstance(metadatas, list):
                    for meta in metadatas:
                        if isinstance(meta, PluginMetadata):
                            self.register(
                                self.get_tool(meta.name),
                                source="plugin",
                                metadata=meta,
                            )
                            logger.info("Plugin tool registered: %s", meta.name)
    
    # ── Plugin Watchdog ──────────────────────────────────────────────────
    
    def start_watchdog(self, plugins_dir: Optional[Path] = None) -> None:
        """
        Start a background thread that watches for plugin file changes.
        
        When a plugin file changes, it is automatically reloaded.
        Uses file modification time polling (no inotify dependency).
        
        Args:
            plugins_dir: Directory to watch. Defaults to APP_ROOT/plugins/.
        """
        if self._watchdog_running:
            return
        
        if plugins_dir is None:
            plugins_dir = APP_ROOT / "plugins"
        
        plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            args=(plugins_dir,),
            daemon=True,
            name="nexus-plugin-watchdog",
        )
        self._watchdog_thread.start()
        logger.info("Plugin watchdog started for %s", plugins_dir)
    
    def stop_watchdog(self) -> None:
        """Stop the plugin watchdog thread."""
        self._watchdog_running = False
        logger.info("Plugin watchdog stopped.")
    
    def _watchdog_loop(self, plugins_dir: Path) -> None:
        """
        Polling loop for plugin file changes.
        
        Checks file modification times every PLUGIN_RELOAD_DEBOUNCE_SECONDS.
        When a change is detected, reloads the plugin.
        """
        debounce = self._settings.PLUGIN_RELOAD_DEBOUNCE_SECONDS
        
        while self._watchdog_running:
            time.sleep(debounce)
            
            try:
                if not plugins_dir.exists():
                    continue
                
                for file_path in plugins_dir.glob("*.py"):
                    if file_path.stem.startswith("_"):
                        continue
                    
                    mtime = file_path.stat().st_mtime
                    last_time = self._last_reload_times.get(file_path.stem, 0)
                    
                    if mtime > last_time:
                        logger.info("Plugin file changed: %s", file_path.stem)
                        try:
                            self._load_plugin_file(file_path)
                            self._last_reload_times[file_path.stem] = mtime
                        except Exception as e:
                            logger.error("Plugin reload failed for '%s': %s", file_path.stem, e)
            
            except Exception as e:
                logger.debug("Watchdog loop error: %s", e)
    
    # ─── Built-in Tool Loading ────────────────────────────────────────────
    
    def load_builtin_tools(self) -> int:
        """
        Load all 22 built-in tools from nexus_tools.tools package.
        
        Each tool module should define a tool function decorated with @tool.
        The function name should match the module's purpose.
        
        Returns:
            Number of tools successfully loaded.
        """
        import nexus_tools.tools as tools_package
        
        tools_dir = Path(tools_package.__file__).parent
        
        count = 0
        for file_path in sorted(tools_dir.glob("t*.py")):
            if file_path.stem == "__init__":
                continue
            
            try:
                module_name = f"nexus_tools.tools.{file_path.stem}"
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    continue
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find @tool-decorated functions
                tools_found = 0
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    # LangChain @tool decorator creates a BaseTool instance
                    if hasattr(attr, "__name__") and callable(attr) and not attr_name.startswith("_"):
                        # Check if it's a regular function (not a class)
                        if inspect.isfunction(attr) or hasattr(attr, "invoke"):
                            # Use attr_name as the canonical tool name (most reliable)
                            self.register(attr, source="builtin", metadata=PluginMetadata(
                                name=attr_name,  # Force use of module-level variable name
                                description=getattr(attr, "__doc__", "").split("\n")[0][:200] if getattr(attr, "__doc__") else "No description",
                                source="builtin"
                            ))
                            tools_found += 1
                
                if tools_found == 0:
                    logger.debug("No tools found in %s (stub file)", file_path.stem)
                
                count += tools_found
            
            except Exception as e:
                logger.warning("Failed to load tools from '%s': %s", file_path.stem, e)
        logger.info("Loaded %d built-in tools", count)
        return count


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    """
    Return the singleton ToolRegistry instance.
    
    Returns:
        ToolRegistry: The singleton registry instance.
    """
    return ToolRegistry()
