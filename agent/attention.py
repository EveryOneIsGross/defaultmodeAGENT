"""
Simple attention trigger system for Discord bot.

This module provides fuzzy matching for attention triggers loaded from YAML
system prompts. When triggers match, the bot responds as if mentioned.
"""

from fuzzywuzzy import fuzz
from typing import List
import logging

logger = logging.getLogger(__name__)

def check_attention_triggers_fuzzy(content: str, persona_triggers: List[str], threshold: int = 60) -> bool:
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
    
    # Number of words needed before checking triggers
    if len(words) < 4:
        return False
    
    for trigger in persona_triggers:
        if not trigger:  # Skip empty triggers
            continue
            
        trigger_lower = trigger.lower().strip()
        
        # Exact substring match first (fastest)
        if trigger_lower in content_lower:
            logger.debug(f"Exact attention trigger matched: '{trigger}'")
            return True
        
        # Check individual words from trigger against content with scoring
        trigger_words = [word for word in trigger_lower.split() if len(word) >= 3]
        if trigger_words:
            word_scores = []
            matched_words = []
            
            for trigger_word in trigger_words:
                # Find best fuzzy match between trigger word and any content word
                word_score = max((fuzz.ratio(content_word, trigger_word) for content_word in words), default=0)
                if word_score >= threshold:
                    word_scores.append(word_score)
                    matched_words.append(trigger_word)
            
            if word_scores:
                # Coverage ratio and average fuzzy score
                coverage_ratio = len(matched_words) / len(trigger_words)
                avg_fuzzy_score = sum(word_scores) / len(word_scores)
                
                # Coverage requirement based on trigger length
                min_coverage = 0.5 + (0.5 / len(trigger_words))  # Single: 100%, Two: 75%, Three: 67%, etc.
                
                # Trigger if: sufficient coverage AND good fuzzy scores (threshold unchanged)
                if coverage_ratio >= min_coverage and avg_fuzzy_score >= threshold:
                    logger.debug(f"Attention trigger matched: {matched_words} from '{trigger}' "
                               f"(scores: {word_scores}, coverage: {coverage_ratio:.2f} >= {min_coverage:.2f}, "
                               f"avg_score: {avg_fuzzy_score:.1f} >= {threshold})")
                    return True
                else:
                    logger.debug(f"Attention trigger insufficient: {matched_words} from '{trigger}' "
                               f"(scores: {word_scores}, coverage: {coverage_ratio:.2f} < {min_coverage:.2f} OR "
                               f"avg_score: {avg_fuzzy_score:.1f} < {threshold})")
    
    return False