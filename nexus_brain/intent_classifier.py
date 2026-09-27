"""
NEXUS AI v4.0 — Intent pre-classification via Groq voice model.

Uses the lightweight Groq voice model (llama-3.1-8b-instant) for fast,
cheap intent classification. Rule-based pre-filter handles ~30% of requests
at zero cost.

Classification prompt: ~50 tokens in, ~5 tokens out, cost ≈ $0.000001.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from functools import lru_cache

from nexus_config.settings import get_settings
from nexus_brain.agent_state import AgentRoute

logger = logging.getLogger("nexus.classifier")


# ─── Intent Categories ────────────────────────────────────────────────────────

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "simple_command": "Quick one-step actions: file ops, system info, open app, clipboard.",
    "complex_task": "Multi-step workflows needing planning, sequencing, tool chaining.",
    "ui_automation": "GUI operations: clicking, typing, screenshots, window management.",
    "capability_synthesis": "User requests a new capability the agent clearly doesn't have.",
    "chat": "General conversation, greetings, questions not requiring action.",
    "clarify": "Request is ambiguous — need to ask the user for more details.",
}


# ─── Classification Result ────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Result of intent classification.
    
    Fields:
        intent: The classified intent route.
        confidence: Confidence score (0.0-1.0) from the local model.
        reasoning: One-sentence explanation from the classifier.
        latency_ms: Wall-clock time for classification.
        model_used: Which model performed the classification.
        needs_full_agent: True if this request requires the full Groq 70B agent.
    """
    intent: AgentRoute
    confidence: float
    reasoning: str
    latency_ms: float
    model_used: str
    needs_full_agent: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "latency_ms": round(self.latency_ms, 1),
            "model_used": self.model_used,
            "needs_full_agent": self.needs_full_agent,
        }


# ─── Rule-Based Pre-Filter (zero-cost for obvious cases) ──────────────────────

# Patterns that immediately classify as simple_command (no LLM call needed)
_SIMPLE_COMMAND_PATTERNS: List[str] = [
    "open ", "launch ", "start ",
    "what", "who", "when", "where", "why", "how",
    "my cpu", "my ram", "my disk", "battery",
    "copy ", "paste", "clipboard",
    "list ", "show ", "find ",
    "shutdown", "restart", "sleep",
    "screenshot", "take a",
]

# Patterns that immediately classify as ui_automation
_UI_AUTOMATION_PATTERNS: List[str] = [
    "click", "double click", "right click",
    "type ", "keyboard", "press ",
    "scroll", "drag", "drop",
    "switch to window", "focus ",
    "move mouse", "move cursor",
]

# Patterns that immediately classify as chat
_CHAT_PATTERNS: List[str] = [
    "hello", "hi ", "hey", "good morning", "good evening",
    "who are you", "what can you do", "tell me about",
    "thank", "thanks", "you're welcome",
    "how are you", "nice to meet",
]

# Patterns that clearly require capability synthesis
_SYNTHESIS_PATTERNS: List[str] = [
    "build a tool", "create a tool", "write a tool",
    "teach yourself", "learn how to",
    "new capability", "new skill",
    "i need you to be able to",
    "can you [^ ]+ that you can't",  # partial
]


def _rule_based_classify(text: str) -> Optional[AgentRoute]:
    """
    Attempt to classify using regex patterns before calling the LLM.
    
    This is free (zero token cost) and handles ~30% of all requests.
    
    Args:
        text: The user's input text (lowercased).
    
    Returns:
        AgentRoute if matched by rules, None if ambiguous.
    """
    text_lower = text.lower().strip()
    
    # Check synthesis patterns (highest priority — safety)
    for pattern in _SYNTHESIS_PATTERNS:
        if pattern in text_lower:
            return "capability_synthesis"
    
    # Check simple command patterns
    for pattern in _SIMPLE_COMMAND_PATTERNS:
        if text_lower.startswith(pattern):
            return "simple_command"
    
    # Check UI automation patterns
    for pattern in _UI_AUTOMATION_PATTERNS:
        if pattern in text_lower:
            return "ui_automation"
    
    # Check chat patterns
    for pattern in _CHAT_PATTERNS:
        if text_lower.startswith(pattern) or text_lower.strip() == pattern.strip():
            return "chat"
    
    return None  # Ambiguous — must use LLM


# ─── LLM-Based Classifier ─────────────────────────────────────────────────────

CLASSIFICATION_PROMPT = """You are NEXUS AI's intent classifier. Your ONLY job is to classify the user's request into ONE category.

Categories:
- simple_command: Quick one-step actions (file ops, system info, open app, clipboard, copy/paste, listing). Ends in <5 seconds with one tool call.
- complex_task: Multi-step workflows needing planning, sequencing, tool chaining. Takes 5+ seconds and multiple tool calls.
- ui_automation: GUI operations — clicking, typing, dragging, window management, mouse movement, screenshots.
- capability_synthesis: User explicitly asks the agent to do something it clearly needs a new capability for.
- chat: General conversation, greetings, questions about the agent itself, "what can you do", thank yous.
- clarify: Request is ambiguous, missing information, or the intent is unclear.

User request: {user_input}

Respond with EXACTLY one JSON object — no markdown, no prose, no explanation:
{{"intent": "<category>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


class IntentClassifier:
    """
    Two-stage intent classifier: rule-based pre-filter → Groq voice model.

    Uses the lightweight Groq voice model (llama-3.1-8b-instant) for fast,
    cheap intent classification when rules don't match.

    Usage:
        classifier = get_intent_classifier()
        result = await classifier.classify("Open VS Code")
        if result.needs_full_agent:
            # Route to planner agent
        else:
            # Route to local action
    """

    def __init__(self):
        self._settings = get_settings()
        self._llm_router = None  # Lazy import to avoid circular deps

    def _get_llm_router(self):
        if self._llm_router is None:
            from nexus_brain.llm_router import get_llm_router
            self._llm_router = get_llm_router()
        return self._llm_router

    async def classify(self, user_input: str) -> ClassificationResult:
        """
        Classify a user request into an intent category.

        Two-stage pipeline:
        1. Rule-based pre-filter (zero cost, ~0.01ms)
        2. If ambiguous → Groq voice model call (~100ms)

        Args:
            user_input: The raw user message.

        Returns:
            ClassificationResult with intent, confidence, and routing hint.
        """
        start = time.perf_counter()

        # Stage 1: Rule-based pre-filter
        rule_match = _rule_based_classify(user_input)
        if rule_match:
            latency = (time.perf_counter() - start) * 1000
            return ClassificationResult(
                intent=rule_match,
                confidence=0.85,
                reasoning=f"Rule-based match: input starts with known pattern",
                latency_ms=latency,
                model_used="rule_based",
                needs_full_agent=rule_match in ("complex_task", "capability_synthesis", "clarify"),
            )

        # Stage 2: Groq voice model classification
        try:
            prompt = CLASSIFICATION_PROMPT.format(user_input=user_input[:500])
            router = self._get_llm_router()

            response = await router.generate(
                messages=[
                    {"role": "system", "content": "You are an intent classifier. Respond with ONLY the JSON object."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=100,
                role="voice",
            )

            if not response.success:
                raise RuntimeError(response.error or "Groq classification failed")

            content = response.content.strip()

            # Parse JSON response
            result = self._parse_response(content)
            latency = (time.perf_counter() - start) * 1000

            return ClassificationResult(
                intent=result.get("intent", "clarify"),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "Groq voice model classification"),
                latency_ms=latency,
                model_used=response.model,
                needs_full_agent=result.get("intent") in ("complex_task", "capability_synthesis", "clarify"),
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.warning("Intent classification failed: %s. Defaulting to complex_task.", e)
            return ClassificationResult(
                intent="complex_task",
                confidence=0.3,
                reasoning=f"Classifier error: {e}. Defaulted to complex_task for safety.",
                latency_ms=latency,
                model_used="fallback",
                needs_full_agent=True,
            )
    
    async def classify_batch(
        self,
        inputs: List[str],
    ) -> List[ClassificationResult]:
        """
        Classify multiple inputs in parallel.
        
        Args:
            inputs: List of user request strings.
        
        Returns:
            List of ClassificationResult in the same order.
        """
        tasks = [self.classify(inp) for inp in inputs]
        return await asyncio.gather(*tasks)
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """
        Parse the LLM's JSON response, handling common formatting issues.
        
        Args:
            content: Raw response string (may have markdown, extra text, etc.)
        
        Returns:
            Dict with "intent", "confidence", "reasoning" keys.
        """
        # Strip markdown code fences
        if "```" in content:
            # Extract JSON from between code fences
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{"):
                    content = part
                    break
                if part.startswith("json") and "{" in part:
                    content = part[part.index("{"):]
                    break
        
        # Strip leading/trailing whitespace
        content = content.strip()
        
        # Try to find JSON object boundaries
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            content = content[start_idx:end_idx]
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try to extract fields manually
            result = self._fallback_parse(content)
        
        # Validate intent
        valid_intents = {"simple_command", "complex_task", "ui_automation", "capability_synthesis", "chat", "clarify"}
        intent = result.get("intent", "clarify")
        if intent not in valid_intents:
            logger.warning("Invalid intent '%s' from classifier, defaulting to clarify", intent)
            result["intent"] = "clarify"
            result["confidence"] = 0.3
        
        return result
    
    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """
        Fallback parser for when JSON parsing fails.
        
        Uses simple string matching to extract intent, confidence, reasoning.
        """
        result: Dict[str, Any] = {"intent": "clarify", "confidence": 0.3, "reasoning": "Parse fallback"}
        
        valid_intents = {"simple_command", "complex_task", "ui_automation", "capability_synthesis", "chat", "clarify"}
        
        # Look for intent
        for intent in valid_intents:
            if intent in text.lower():
                result["intent"] = intent
                break
        
        # Look for confidence
        import re
        conf_match = re.search(r"confidence[\":\s]+([0-9.]+)", text.lower())
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass
        
        # Look for reasoning
        reasoning_match = re.search(r"reasoning[\":\s]+([^\"]+)", text)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()[:100]
        
        return result
    
    async def close(self) -> None:
        """No-op — Groq client is managed by LLMRouter."""


@lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    """
    Return the singleton IntentClassifier instance.
    
    Returns:
        IntentClassifier: The singleton classifier instance.
    """
    return IntentClassifier()
