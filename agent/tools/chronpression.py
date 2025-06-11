"""
Chronomic Text Filter - Simple version with deduplication
Filters text to keep only statistically novel and significant words.
"""

from collections import Counter
import string
import re
import os
import argparse

# Common English words frequency list (simplified replacement for brown corpus)
COMMON_WORDS = {
    "the": 0.0616, "of": 0.0292, "and": 0.0284, "to": 0.0275, "a": 0.0235,
    "in": 0.0234, "is": 0.0119, "that": 0.0109, "for": 0.0107, "it": 0.0099,
    "with": 0.0095, "as": 0.0093, "was": 0.0087, "be": 0.0083, "on": 0.0075,
    "not": 0.0072, "he": 0.0069, "by": 0.0064, "are": 0.0062, "this": 0.0061,
    "at": 0.0056, "from": 0.0055, "but": 0.0053, "have": 0.0052, "an": 0.0049,
    "they": 0.0048, "which": 0.0047, "or": 0.0046, "his": 0.0045, "had": 0.0043,
    "we": 0.0042, "there": 0.0041, "can": 0.0040, "were": 0.0039, "been": 0.0038,
    "has": 0.0037, "their": 0.0035, "more": 0.0035, "will": 0.0034, "would": 0.0034,
    "about": 0.0033, "if": 0.0033, "no": 0.0032, "when": 0.0032, "who": 0.0031,
    "so": 0.0031, "all": 0.0030, "she": 0.0029, "you": 0.0027, "said": 0.0025,
}

def tokenize_text(text):
    """
    Simple tokenizer to split text into words and separators.
    
    Returns:
    - List of tokens (words and separators)
    """
    return re.findall(r'\b\w+\b|[^\w\s]|\s+', text)

def remove_consecutive_duplicates(text):
    """Remove consecutive duplicate words while preserving punctuation and spacing."""
    # Pattern matches word boundaries and captures the word, then looks for the same word
    pattern = r'\b(\w+)\b(\s+\1\b)+'
    return re.sub(pattern, r'\1', text)

def chronomic_filter(text, alpha=0.3, beta=3.0):
    """
    Process text using a bimodal significance model.
    
    Parameters:
    - text: Input text to process
    - alpha: Threshold for novel words (below this is novel)
    - beta: Threshold for significant words (above this is significant)
    
    Returns:
    - Processed text with middle-band words replaced and consecutive duplicates removed
    """
    # Use built-in common words as background distribution
    bg_probs = COMMON_WORDS
    bg_total = sum(bg_probs.values())
    
    # Tokenize input text
    tokens = tokenize_text(text)
    
    # Track word positions in original token list
    word_positions = []
    word_tokens = []
    
    # Separate words and build position mapping
    for i, token in enumerate(tokens):
        if token.isalpha():
            word_positions.append(i)
            word_tokens.append(token.lower())
    
    # Calculate document word frequencies
    doc_counts = Counter(word_tokens)
    doc_total = max(1, len(word_tokens))  # Avoid division by zero
    
    # Process each word
    filtered_tokens = tokens.copy()
    for i, word in enumerate(word_tokens):
        doc_freq = doc_counts[word] / doc_total
        bg_freq = bg_probs.get(word, 0.00001)  # Small default for unknown words
        norm_freq = doc_freq / bg_freq
        
        # Apply bimodal filter
        if not (norm_freq < alpha or norm_freq > beta):
            # Word is in middle band - remove it
            pos = word_positions[i]
            filtered_tokens[pos] = ""
    
    # Reconstruct text while preserving punctuation and spacing
    # Join tokens, handling consecutive spaces correctly
    processed = ""
    last_was_space = False
    
    for token in filtered_tokens:
        if not token:
            continue
        
        # Handle consecutive spaces
        if token.isspace():
            if not last_was_space and processed:
                processed += " "
            last_was_space = True
        else:
            processed += token
            last_was_space = token.isspace()
    
    # Remove consecutive duplicate words
    processed = remove_consecutive_duplicates(processed)
    
    return processed

# Command-line interface with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronomic Text Compression Tool")
    parser.add_argument("-i", "--input", help="Input file path to process")
    parser.add_argument("-o", "--output", help="Output file path (default: input_compressed.ext)")
    parser.add_argument("-a", "--alpha", type=float, default=0.3, help="Alpha threshold (default: 0.3)")
    parser.add_argument("-b", "--beta", type=float, default=3.0, help="Beta threshold (default: 3.0)")
    args = parser.parse_args()
    
    if args.input:
        # Read from file
        with open(args.input, 'r') as f:
            text = f.read()
            
        # Create output filename if not provided
        if not args.output:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_compressed{ext}"
    else:
        # Example text
        text = """
        The quick brown fox jumps over the lazy dog. Information theory concepts
        are applied in many fields including compression, cryptography, and linguistics.
        Common words are often filtered out while rare terms are highlighted.
        """
        args.output = None
    
    # Process with parameters
    result = chronomic_filter(text, args.alpha, args.beta)
    
    if args.output:
        # Write to output file
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"Processed text written to {args.output}")
    else:
        # Print to console
        print(result)