"""
NEXUS AI v4.0 — Memory manager with enhanced vector store integration.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

High-level memory operations:
- Episodic memory (task outcomes)
- Semantic memory (user preferences)
- Synthesized tool documentation
- Working memory (session context)
"""

import logging
from typing import Any, Dict, List, Optional
from functools import lru_cache

from nexus_memory.vector_store import get_vector_store, SearchResult
from nexus_memory.session_context import get_session_context

logger = logging.getLogger("nexus.memory")


class MemoryManager:
    """
    High-level memory manager for NEXUS AI.
    
    Provides a clean API over the vector store for common memory operations.
    """
    
    def __init__(self):
        self._vs = get_vector_store()
        self._sc = get_session_context()
    
    # ── Episodic Memory ───────────────────────────────────────────────────────
    
    def store_episodic(
        self,
        task: str,
        outcome: str,
        tools: List[str],
        duration_ms: float,
        success: bool,
        session_id: str,
        complexity: str = "simple",
    ) -> str:
        """
        Store an episodic memory of a completed task.
        
        Args:
            task: Description of the task.
            outcome: Result/outcome of the task.
            tools: List of tools used.
            duration_ms: Task duration in milliseconds.
            success: Whether the task succeeded.
            session_id: Session identifier.
            complexity: Task complexity level.
        
        Returns:
            Memory document ID.
        """
        document = f"Task: {task}\nOutcome: {outcome}\nTools: {', '.join(tools)}\nDuration: {duration_ms}ms"
        
        metadata = {
            "task": task,
            "outcome": outcome,
            "success": success,
            "tools": str(tools),
            "duration_ms": duration_ms,
            "session_id": session_id,
            "complexity": complexity,
            "type": "episodic",
        }
        
        doc_id = f"ep_{session_id}_{hash(task) & 0xFFFFFFFF:08x}"
        return self._vs.add_memory("agent_memory", document, metadata, doc_id)
    
    def search_episodic(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search episodic memories for context assembly.
        
        Args:
            query: Search query.
            n_results: Number of results to return.
        
        Returns:
            List of memory records with task/outcome fields.
        """
        results = self._vs.search("agent_memory", query, n_results)
        return self._format_results(results, ["task", "outcome"])
    
    # ── Semantic Memory (User Preferences) ────────────────────────────────────
    
    def store_preference(self, fact: str) -> str:
        """
        Store a user preference fact.
        
        Args:
            fact: Preference fact to store.
        
        Returns:
            Memory document ID.
        """
        metadata = {
            "fact": fact,
            "type": "preference",
        }
        
        doc_id = f"pref_{hash(fact) & 0xFFFFFFFF:08x}"
        return self._vs.add_memory("user_preferences", f"Fact: {fact}", metadata, doc_id)
    
    def search_semantic(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search user preferences relevant to the current input.
        
        Args:
            query: Search query.
            n_results: Number of results to return.
        
        Returns:
            List of preference records.
        """
        results = self._vs.search("user_preferences", query, n_results)
        return self._format_results(results, ["fact"])
    
    # ── Synthesized Tools ─────────────────────────────────────────────────────
    
    def store_synthesized_tool(
        self,
        tool_name: str,
        description: str,
        code: str,
        requirements: List[str],
    ) -> str:
        """
        Store documentation for a synthesized tool.
        
        Args:
            tool_name: Name of the synthesized tool.
            description: Tool description.
            code: Tool implementation code.
            requirements: List of required packages.
        
        Returns:
            Memory document ID.
        """
        document = f"Tool: {tool_name}\nDescription: {description}\nRequirements: {', '.join(requirements)}\nCode:\n{code}"
        
        metadata = {
            "tool_name": tool_name,
            "description": description,
            "requirements": str(requirements),
            "type": "synthesized_tool",
        }
        
        doc_id = f"tool_{tool_name}"
        return self._vs.add_memory("synthesized_tools", document, metadata, doc_id)
    
    def search_synthesized_tools(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search synthesized tool documentation.
        
        Args:
            query: Search query.
            n_results: Number of results.
        
        Returns:
            List of tool documentation records.
        """
        results = self._vs.search("synthesized_tools", query, n_results)
        return self._format_results(results, ["tool_name", "description"])
    
    # ── Conversation History ──────────────────────────────────────────────────
    
    def store_conversation_summary(self, session_id: str, summary: str, key_topics: List[str]) -> str:
        """
        Store a conversation summary.
        
        Args:
            session_id: Session identifier.
            summary: Conversation summary text.
            key_topics: Key topics discussed.
        
        Returns:
            Memory document ID.
        """
        metadata = {
            "session_id": session_id,
            "key_topics": str(key_topics),
            "type": "conversation_summary",
        }
        
        doc_id = f"conv_{session_id}_{int(time.time())}"
        return self._vs.add_memory("conversation_history", summary, metadata, doc_id)
    
    def search_conversation_history(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search conversation summaries."""
        results = self._vs.search("conversation_history", query, n_results)
        return self._format_results(results, ["summary"])
    
    # ── General Query ─────────────────────────────────────────────────────────
    
    def query_memories(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query all episodic memories."""
        results = self._vs.search("agent_memory", query, n_results)
        return self._format_results(results, ["task", "outcome"])
    
    def query_all_collections(self, query: str, n_results: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Query across all memory collections."""
        collections = ["agent_memory", "user_preferences", "synthesized_tools", "conversation_history"]
        results = {}
        
        for coll in collections:
            try:
                search_results = self._vs.search(coll, query, n_results)
                results[coll] = self._format_results(search_results)
            except Exception as e:
                logger.debug("Query failed for %s: %s", coll, e)
                results[coll] = []
        
        return results
    
    # ── Context Assembly ──────────────────────────────────────────────────────
    
    def get_context(self) -> Dict[str, Any]:
        """Get current memory context for prompt assembly."""
        return {
            "working_memory": self._sc.get("working_memory", []),
            "last_task": self._sc.get_last_task_summary(),
            "task_history_count": len(self._sc.get("task_history", [])),
        }
    
    # ── Health & Stats ────────────────────────────────────────────────────────
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get counts for all collections."""
        stats = {}
        for coll_name in ["agent_memory", "user_preferences", "synthesized_tools", "conversation_history"]:
            stats[coll_name] = self._vs.count(coll_name)
        return stats
    
    def health_check(self) -> bool:
        """Check memory system health."""
        return self._vs.heartbeat()
    
    # ── Internal ──────────────────────────────────────────────────────────────
    
    def _format_results(
        self,
        results: List[SearchResult],
        required_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Format search results into clean dictionaries."""
        formatted = []
        
        for result in results:
            record = dict(result.metadata)
            record["document"] = result.document
            record["score"] = result.score
            record["id"] = result.id
            
            # Filter by required fields if specified
            if required_fields:
                if not all(isinstance(record.get(key), str) for key in required_fields):
                    continue
            
            formatted.append(record)
        
        return formatted


@lru_cache(maxsize=1)
def get_memory_manager() -> MemoryManager:
    """Get the singleton MemoryManager instance."""
    return MemoryManager()


def reset_memory_manager() -> None:
    """Reset memory manager (for testing)."""
    from nexus_memory.vector_store import reset_vector_store
    reset_vector_store()
    get_memory_manager.cache_clear()