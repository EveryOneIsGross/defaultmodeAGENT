#!/usr/bin/env python3
import sys
import os
import re

def clean_yaml_file(filepath):
    """
    Clean a YAML file of illegal Unicode characters.
    Focuses on removing control characters and other non-printable characters.
    """
    print(f"Cleaning file: {filepath}")
    
    # Read the file content
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original_size = len(content)
    
    # Track replaced characters
    replaced_chars = {}
    
    # Replace common problematic characters
    for i, char in enumerate(content):
        cp = ord(char)
        # Control characters and other non-printable characters
        if (0x00 <= cp <= 0x1F and cp != 0x09 and cp != 0x0A and cp != 0x0D) or (0x7F <= cp <= 0x9F):
            replaced_chars[char] = replaced_chars.get(char, 0) + 1
            # Replace with space
            content = content[:i] + ' ' + content[i+1:]
    
    # Check for "smart" or curly quotes and replace them
    content = re.sub(r'[\u2018\u2019]', "'", content)  # Replace smart single quotes
    content = re.sub(r'[\u201C\u201D]', '"', content)  # Replace smart double quotes
    
    # Write back the cleaned content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Print summary
    new_size = len(content)
    if replaced_chars:
        print(f"Replaced {sum(replaced_chars.values())} illegal characters:")
        for char, count in replaced_chars.items():
            print(f"  U+{ord(char):04X} ({count} occurrences)")
    else:
        print("No illegal characters found.")
    
    print(f"Original size: {original_size} bytes")
    print(f"New size: {new_size} bytes")
    print(f"File cleaned and saved: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_yaml.py <yaml_file_path>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"Error: File not found - {filepath}")
        sys.exit(1)
    
    clean_yaml_file(filepath) 