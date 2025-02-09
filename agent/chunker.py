import logging
from tokenizer import get_tokenizer, encode_text, decode_tokens

def truncate_middle(text, max_tokens=64):
    """
    Truncates text to a maximum number of tokens while preserving content from both ends.
    Special tokens are disallowed and scrubbed from input text.
    Ensures proper handling of code blocks and markdown formatting.
    
    Args:
        text (str): The input text to truncate
        max_tokens (int, optional): Maximum number of tokens to keep. Defaults to 64.
        
    Returns:
        str: The truncated text with ... in the middle if truncation was needed,
             or the original text if it was already within the token limit.
    """
    # Clean special tokens first
    text = clean_response(text)
    
    try:
        # Use global tokenizer functions
        tokens = encode_text(text)
        
        if len(tokens) <= max_tokens:
            return text
            
        # Find code block boundaries
        code_blocks = []
        lines = text.split('\n')
        in_code_block = False
        start_idx = 0
        
        for i, line in enumerate(lines):
            if '```' in line:
                if not in_code_block:
                    start_idx = i
                    in_code_block = True
                else:
                    code_blocks.append((start_idx, i))
                    in_code_block = False
        
        # If we're in the middle of a code block, preserve it
        keep_tokens = max_tokens - 1  # Account for ellipsis
        side_tokens = keep_tokens // 2
        end_tokens = side_tokens + (keep_tokens % 2)
        
        # Special handling for code blocks
        if code_blocks:
            # Ensure we don't split inside code blocks
            block_tokens = []
            for start, end in code_blocks:
                block_text = '\n'.join(lines[start:end+1])
                block_tokens.extend(encode_text(block_text))
            
            if len(block_tokens) <= max_tokens:
                # If code blocks fit within limit, preserve them entirely
                return text
        
        ellipsis_token = encode_text('...')
        truncated_tokens = tokens[:side_tokens] + ellipsis_token + tokens[-end_tokens:]
        result = decode_tokens(truncated_tokens)
        
        # Ensure we haven't broken any markdown formatting
        if '```' in result:
            # Count occurrences of code block markers
            if result.count('```') % 2 != 0:
                # Add closing code block if needed
                result += '\n```'
                
        return result
        
    except Exception as e:
        logging.error(f"Tokenization error in truncate_middle: {str(e)}")
        # Fallback to simple string truncation if tokenization fails
        if len(text) <= max_tokens * 4:  # Rough estimate of chars per token
            return text
        mid = len(text) // 2
        keep_chars = (max_tokens * 4) // 2
        return text[:keep_chars] + "..." + text[-keep_chars:]

def clean_response(response: str) -> str:
    """Clean common LLM tokens and artifacts from response text."""
    if not response:
        return response
        
    # Common tokens to clean
    tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|assistant|>",
        "<|system|>",
        "<|human|>"
    ]
    
    cleaned = response
    for token in tokens:
        cleaned = cleaned.replace(token, "")
    
    return cleaned.strip()

def balance_wraps(text: str, wraps: dict[str, str] = {"{": "}", "(": ")", "[": "]", "<": ">"}) -> str:
    """Auto-corrects unbalanced wraps in text by adding missing closers or removing orphaned wraps.
    
    Args:
        text: String to fix
        wraps: Dict mapping opening chars to their closing chars
    
    Returns:
        str: Text with balanced wraps
    """
    stack = []
    chars = list(text)
    reverse_wraps = {v: k for k, v in wraps.items()}
    
    # Forward pass - handle orphaned closing wraps
    i = 0
    while i < len(chars):
        if chars[i] in wraps:
            stack.append(i)
        elif chars[i] in reverse_wraps:
            if not stack:
                chars.pop(i)
                continue
            stack.pop()
        i += 1
    
    # Add missing closing wraps
    while stack:
        pos = stack.pop()
        chars.append(wraps[chars[pos]])
        
    return ''.join(chars)