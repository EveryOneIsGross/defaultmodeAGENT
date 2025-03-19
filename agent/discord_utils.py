import re
import discord
import logging

def strip_role_prefixes(username: str) -> str:
    """Strip all role prefix characters from username."""
    return username.lstrip('')  # Common role prefix characters

def sanitize_mentions(content: str, mentions: list) -> str:
    """Convert Discord mention IDs to readable usernames, preserving code blocks."""
    if not content or not mentions:
        return content
        
    lines = content.split('\n')
    formatted_lines = []
    in_code_block = False
    
    for line in lines:
        if '```' in line:
            in_code_block = not in_code_block
            
        # Create mention pattern map for current line state
        mention_map = {
            f'<@{m.id}>': f'@{strip_role_prefixes(m.name)}' if not in_code_block else strip_role_prefixes(m.name)
            for m in mentions if hasattr(m, 'name')
        }
        mention_map.update({
            f'<@{m.id}>.': f'@{strip_role_prefixes(m.name)}.' if not in_code_block else f'{strip_role_prefixes(m.name)}.'
            for m in mentions if hasattr(m, 'name')
        })
        mention_map.update({
            f'<@!{m.id}>': f'@{strip_role_prefixes(m.name)}' if not in_code_block else strip_role_prefixes(m.name)
            for m in mentions if hasattr(m, 'name')
        })
        mention_map.update({
            f'<@&{m.id}>': f'@{strip_role_prefixes(m.name)}' if not in_code_block else strip_role_prefixes(m.name)
            for m in mentions if hasattr(m, 'name') and hasattr(m, 'guild_permissions')
        })
            
        # Log transformations
        for pattern, replacement in mention_map.items():
            logging.debug(f"Sanitize transform: {pattern} -> {replacement}")
            
        current_line = line
        for pattern, replacement in mention_map.items():
            current_line = current_line.replace(pattern, replacement)
            
        formatted_lines.append(current_line)
    
    result = '\n'.join(formatted_lines)
    logging.info(f"Sanitized mentions result: {result[:100]}...")
    return result

def format_discord_mentions(content: str, guild: discord.Guild, mentions_enabled: bool = True) -> str:
    """Convert readable usernames to either Discord mentions or display names."""
    if not content or not guild:
        return content
        
    lines = content.split('\n')
    formatted_lines = []
    in_code_block = False
    
    for line in lines:
        if '```' in line:
            in_code_block = not in_code_block
            
        current_line = line
        if not in_code_block:
            if mentions_enabled:
                # Convert to Discord mentions, handling longer names first
                for member in sorted(guild.members, key=lambda m: len(m.name), reverse=True):
                    clean_name = re.escape(strip_role_prefixes(member.name))
                    if f"@{member.name}" in current_line:
                        current_line = current_line.replace(f"@{member.name}", f"<@{member.id}>")
            else:
                # Transform @username to display name but preserve code-related @ symbols
                def format_mentions_disabled(match):
                    username = match.group(1)
                    # Keep @ for code annotations like @property, @staticmethod, etc.
                    code_annotations = ['property', 'staticmethod', 'classmethod', 'decorator', 
                                      'param', 'return', 'override', 'abstractmethod']
                    if username.lower() in code_annotations:
                        return f"@{username}"
                        
                    # For user mentions, use display name without @
                    member = discord.utils.get(guild.members, name=username)
                    if member:
                        return member.display_name
                    return username
                
                current_line = re.sub(r'@([\w.]+)', format_mentions_disabled, current_line)
                
        formatted_lines.append(current_line)
    
    result = '\n'.join(formatted_lines)
    logging.info(f"Format mentions result (enabled={mentions_enabled}): {result[:100]}...")
    return result
