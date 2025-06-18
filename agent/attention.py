"""
Simple attention trigger system for Discord bot.

This module provides fuzzy matching for attention triggers loaded from YAML
system prompts. When triggers match, the bot responds as if mentioned.
"""

from fuzzywuzzy import fuzz
from typing import List
import logging

logger = logging.getLogger(__name__)

def check_attention_triggers_fuzzy(content: str, persona_triggers: List[str], threshold: int = 80) -> bool:
    """
    Check if message contains any attention trigger words using fuzzy matching.
    
    Args:
        content: Message content to check
        persona_triggers: List of attention trigger strings from YAML
        threshold: Minimum fuzzy match score (0-100) for trigger activation
        
    Returns:
        True if any trigger matches, False otherwise
    """
    if not persona_triggers or not content:
        return False
    
    content_lower = content.lower().strip()
    words = content_lower.split()
    
    for trigger in persona_triggers:
        if not trigger:  # Skip empty triggers
            continue
            
        trigger_lower = trigger.lower().strip()
        
        # Exact substring match first (fastest)
        if trigger_lower in content_lower:
            logger.debug(f"Exact attention trigger matched: '{trigger}'")
            return True
        
        # Fuzzy match against individual words for single-word triggers
        if len(trigger.split()) == 1:
            for word in words:
                if fuzz.ratio(word, trigger_lower) >= threshold:
                    logger.debug(f"Fuzzy word attention trigger matched: '{trigger}' -> '{word}' (score: {fuzz.ratio(word, trigger_lower)})")
                    return True
        
        # Fuzzy match against phrases for multi-word triggers
        else:
            if fuzz.partial_ratio(content_lower, trigger_lower) >= threshold:
                logger.debug(f"Fuzzy phrase attention trigger matched: '{trigger}' (score: {fuzz.partial_ratio(content_lower, trigger_lower)})")
                return True
    
    return False