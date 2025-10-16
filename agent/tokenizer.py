import tiktoken

_tokenizer = None

def get_tokenizer():
    """Get or create the global tokenizer instance with standard configuration."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
            special_tokens = _tokenizer.special_tokens_set
            _tokenizer._special_tokens_set = special_tokens - {'<|endoftext|>'}
        except Exception as e:
            raise
    return _tokenizer

def encode_text(text: str, **kwargs):
    """Encode text using the global tokenizer with consistent configuration."""
    tokenizer = get_tokenizer()
    try:
        return tokenizer.encode(text, **kwargs)
    except Exception as e:

        return tokenizer.encode(text, disallowed_special=())

def decode_tokens(tokens, **kwargs):
    """Decode tokens using the global tokenizer."""
    tokenizer = get_tokenizer()
    return tokenizer.decode(tokens, **kwargs)

def count_tokens(text: str) -> int:
    """Count tokens in text using the global tokenizer."""
    return len(encode_text(text))

def calculate_image_tokens(width: int, height: int):
    """
    Calculate image tokens using OpenAI's vision API formula.
    Adapted from https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision/492318/2 
    """
    # Compute number of 512x512 tiles that can fit into the image
    tiles_width = -(-width // 512)  # Ceiling division without importing math
    tiles_height = -(-height // 512)  # Ceiling division without importing math

    # See https://platform.openai.com/docs/guides/vision/calculating-costs
    #   - 85 is the "base token" that will always be added
    #   - 1 tiles = 170 tokens 
    total_tokens = 85 + 170 * (tiles_width * tiles_height)
    
    return total_tokens
