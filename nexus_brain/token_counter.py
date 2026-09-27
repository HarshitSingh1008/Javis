"""
NEXUS AI v4.0 — Accurate token counting using tiktoken.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides precise token counting for context budget management,
replacing rough word-count estimation.
"""

import logging
from functools import lru_cache
from typing import List, Dict, Any, Optional

logger = logging.getLogger("nexus.token_counter")


class TokenCounter:
    """
    Accurate token counting using tiktoken.
    
    Supports multiple model encodings with fallback to cl100k_base.
    Caches encodings for performance.
    """
    
    # Model -> encoding mapping
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "llama3": "cl100k_base",  # Approximation
        "llama3.2": "cl100k_base",
        "llama3.1": "cl100k_base",
        "qwen": "cl100k_base",
        "gemma": "cl100k_base",
        "mistral": "cl100k_base",
        "mixtral": "cl100k_base",
    }
    
    def __init__(self, default_model: str = "llama3.2"):
        self._default_model = default_model
        self._encodings: Dict[str, Any] = {}
        self._init_default_encoding()
    
    def _init_default_encoding(self) -> None:
        """Initialize the default encoding."""
        try:
            import tiktoken
            encoding_name = self.MODEL_ENCODINGS.get(self._default_model, "cl100k_base")
            self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)
            logger.debug("Token counter initialized with encoding: %s", encoding_name)
        except Exception as e:
            logger.warning("Failed to initialize tiktoken: %s", e)
            self._encodings["cl100k_base"] = None
    
    def _get_encoding(self, model: str) -> Any:
        """Get or create encoding for a model."""
        encoding_name = self.MODEL_ENCODINGS.get(model, "cl100k_base")
        
        if encoding_name not in self._encodings:
            try:
                import tiktoken
                self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning("Failed to load encoding %s: %s", encoding_name, e)
                # Fallback to cl100k_base
                if "cl100k_base" not in self._encodings:
                    import tiktoken
                    self._encodings["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
                encoding_name = "cl100k_base"
        
        return self._encodings.get(encoding_name)
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: The text to count tokens for.
            model: Optional model name for model-specific encoding.
        
        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        
        model = model or self._default_model
        encoding = self._get_encoding(model)
        
        if encoding is None:
            # Rough fallback: ~4 chars per token
            return len(text) // 4
        
        try:
            return len(encoding.encode(text))
        except Exception as e:
            logger.debug("Token counting failed, using fallback: %s", e)
            return len(text) // 4
    
    def count_message_tokens(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> int:
        """
        Count tokens for a list of chat messages.
        
        Uses the same token counting logic as OpenAI's chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Optional model name.
        
        Returns:
            Total token count including message formatting overhead.
        """
        if not messages:
            return 0
        
        model = model or self._default_model
        encoding = self._get_encoding(model)
        
        if encoding is None:
            # Rough fallback
            total_chars = sum(len(m.get("content", "")) for m in messages)
            return total_chars // 4 + len(messages) * 4  # ~4 tokens per message overhead
        
        try:
            # OpenAI chat format: each message has role + content overhead
            # Based on: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
            tokens = 0
            for message in messages:
                tokens += 4  # message overhead (role, content structure)
                content = message.get("content", "")
                if content:
                    tokens += len(encoding.encode(content))
                role = message.get("role", "")
                if role:
                    tokens += len(encoding.encode(role))
            tokens += 2  # assistant message prefix
            return tokens
        except Exception as e:
            logger.debug("Message token counting failed: %s", e)
            total_chars = sum(len(m.get("content", "")) for m in messages)
            return total_chars // 4 + len(messages) * 4
    
    def truncate_to_budget(
        self,
        text: str,
        max_tokens: int,
        model: Optional[str] = None,
        suffix: str = "\n[...truncated...]"
    ) -> str:
        """
        Truncate text to fit within token budget.
        
        Args:
            text: Text to truncate.
            max_tokens: Maximum token budget.
            model: Optional model name.
            suffix: Suffix to append when truncated.
        
        Returns:
            Truncated text.
        """
        current_tokens = self.count_tokens(text, model)
        if current_tokens <= max_tokens:
            return text
        
        # Binary search for the right truncation point
        encoding = self._get_encoding(model or self._default_model)
        if encoding is None:
            # Rough fallback
            ratio = max_tokens / current_tokens
            char_limit = int(len(text) * ratio) - len(suffix)
            return text[:max(0, char_limit)] + suffix
        
        try:
            tokens = encoding.encode(text)
            # Reserve tokens for suffix
            suffix_tokens = len(encoding.encode(suffix))
            available = max_tokens - suffix_tokens
            
            if available <= 0:
                return suffix
            
            truncated_tokens = tokens[:available]
            truncated = encoding.decode(truncated_tokens)
            return truncated + suffix
        except Exception as e:
            logger.debug("Truncation failed: %s", e)
            ratio = max_tokens / current_tokens
            char_limit = int(len(text) * ratio) - len(suffix)
            return text[:max(0, char_limit)] + suffix
    
    def truncate_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        model: Optional[str] = None,
        keep_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        Truncate message list to fit within token budget.
        
        Preserves system message and most recent messages.
        
        Args:
            messages: List of messages.
            max_tokens: Token budget.
            model: Optional model name.
            keep_system: Whether to always keep system message.
        
        Returns:
            Truncated message list.
        """
        if not messages:
            return []
        
        current = self.count_message_tokens(messages, model)
        if current <= max_tokens:
            return messages
        
        # Separate system message
        system_msg = None
        other_msgs = messages
        
        if keep_system and messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            other_msgs = messages[1:]
        
        # Binary approach: keep removing oldest non-system messages
        while other_msgs and self.count_message_tokens(
            ([system_msg] if system_msg else []) + other_msgs, model
        ) > max_tokens:
            other_msgs = other_msgs[1:]  # Remove oldest
        
        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(other_msgs)
        return result


# Global instance
_default_counter: Optional[TokenCounter] = None


def get_token_counter(model: str = "llama3.2") -> TokenCounter:
    """Get or create the global token counter instance."""
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter(default_model=model)
    return _default_counter


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Convenience function for token counting."""
    return get_token_counter().count_tokens(text, model)


def count_message_tokens(messages: List[Dict[str, str]], model: Optional[str] = None) -> int:
    """Convenience function for message token counting."""
    return get_token_counter().count_message_tokens(messages, model)


def truncate_to_budget(text: str, max_tokens: int, model: Optional[str] = None) -> str:
    """Convenience function for truncation."""
    return get_token_counter().truncate_to_budget(text, max_tokens, model)


def truncate_messages(messages: List[Dict[str, str]], max_tokens: int, model: Optional[str] = None) -> List[Dict[str, str]]:
    """Convenience function for message truncation."""
    return get_token_counter().truncate_messages(messages, max_tokens, model)