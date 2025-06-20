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
