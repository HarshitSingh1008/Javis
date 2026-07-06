# ==================================================================
# NEXUS AI v4.0 - Sliding window context management
# Module: nexus_brain/conversation_summarizer.py
# ==================================================================
"""
Sliding window context management
See NEXUS AI Master Prompt v4.0 for complete spec.
"""
from functools import lru_cache
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("nexus.summarizer")


class ConversationSummarizer:
    """Manages sliding window context and conversation summarization."""
    
    def __init__(self):
        """Initialize the conversation summarizer."""
        self.window_size = 10
        self.summary_cache: Dict[str, str] = {}
        logger.debug("ConversationSummarizer initialized")
    
    def summarize(self, conversation: List[Dict[str, str]]) -> str:
        """
        Summarize a conversation.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            Summarized conversation string
        """
        logger.debug(f"Summarizing {len(conversation)} messages")
        return f"Summary of {len(conversation)} messages"
    
    def get_window(self, conversation: List[Dict[str, str]], size: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get the sliding window of recent messages.
        
        Args:
            conversation: Full conversation history
            size: Window size (defaults to self.window_size)
            
        Returns:
            Recent messages within the window
        """
        if size is None:
            size = self.window_size
        return conversation[-size:] if len(conversation) > size else conversation


@lru_cache(maxsize=1)
def get_conversation_summarizer() -> ConversationSummarizer:
    """
    Return the singleton ConversationSummarizer instance.
    
    Returns:
        ConversationSummarizer: The singleton summarizer instance.
    """
    return ConversationSummarizer()
