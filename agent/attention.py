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
    
    # Number of words needed before checking triggers
    if len(words) < 8:
        return False
    
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
            # Check full phrase in both directions
            score1 = fuzz.partial_ratio(content_lower, trigger_lower)
            score2 = fuzz.partial_ratio(trigger_lower, content_lower)
            max_phrase_score = max(score1, score2)
            
            if max_phrase_score >= threshold:
                logger.debug(f"Fuzzy phrase attention trigger matched: '{trigger}' (score: {max_phrase_score})")
                return True
            
            # Check individual words from trigger against content with scoring
            trigger_words = [word for word in trigger_lower.split() if len(word) >= 3]
            if trigger_words:
                word_scores = []
                matched_words = []
                
                for trigger_word in trigger_words:
                    word_score = fuzz.partial_ratio(content_lower, trigger_word)
                    if word_score >= threshold:
                        word_scores.append(word_score)
                        matched_words.append(trigger_word)
                
                if word_scores:
                    # Calculate composite score: average score * word coverage bonus
                    avg_score = sum(word_scores) / len(word_scores)
                    coverage_bonus = len(matched_words) / len(trigger_words)  # 0.0 to 1.0
                    composite_score = avg_score * (0.7 + 0.3 * coverage_bonus)  # Boost for more words
                    
                    logger.debug(f"Fuzzy word set from phrase attention trigger matched: {matched_words} from '{trigger}' "
                               f"(avg_score: {avg_score:.1f}, coverage: {coverage_bonus:.1f}, composite: {composite_score:.1f})")
                    return True
    
    return False