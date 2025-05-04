import re
import discord
import logging

def strip_role_prefixes(username: str) -> str:
    """Strip all role prefix characters from username."""
    return username.lstrip('')  # Common role prefix characters

def sanitize_mentions(content: str, mentions: list) -> str:
    """Convert Discord mention IDs to readable usernames and channel names, preserving code blocks."""
    if not content or not mentions:
        return content
        
    lines = content.split('\n')
    formatted_lines = []
    in_code_block = False
    
    # Common punctuation and tokens that might follow a mention
    punctuation_marks = ['.', ',', '!', '?', ';', ':', ')', ']', '}', '"', "'"]
    
    for line in lines:
        if '```' in line:
            in_code_block = not in_code_block
            
        # Create mention pattern map for current line state
        mention_map = {
            f'<@{m.id}>': f'@{strip_role_prefixes(m.name)}' if not in_code_block else strip_role_prefixes(m.name)
            for m in mentions if hasattr(m, 'name')
        }
        
        # Add punctuation-aware patterns
        for m in mentions:
            if hasattr(m, 'name'):
                for punct in punctuation_marks:
                    mention_map[f'<@{m.id}>{punct}'] = (
                        f'@{strip_role_prefixes(m.name)}{punct}' if not in_code_block 
                        else f'{strip_role_prefixes(m.name)}{punct}'
                    )
                    mention_map[f'<@!{m.id}>{punct}'] = (
                        f'@{strip_role_prefixes(m.name)}{punct}' if not in_code_block 
                        else f'{strip_role_prefixes(m.name)}{punct}'
                    )
        
        # Add standard mention patterns
        mention_map.update({
            f'<@!{m.id}>': f'@{strip_role_prefixes(m.name)}' if not in_code_block else strip_role_prefixes(m.name)
            for m in mentions if hasattr(m, 'name')
        })
        mention_map.update({
            f'<@&{m.id}>': f'@{strip_role_prefixes(m.name)}' if not in_code_block else strip_role_prefixes(m.name)
            for m in mentions if hasattr(m, 'name') and hasattr(m, 'guild_permissions')
        })
        
        # Add channel mentions with punctuation
        for m in mentions:
            if hasattr(m, 'name') and isinstance(m, discord.TextChannel):
                mention_map[f'<#{m.id}>'] = f'#{m.name}' if not in_code_block else m.name
                for punct in punctuation_marks:
                    mention_map[f'<#{m.id}>{punct}'] = (
                        f'#{m.name}{punct}' if not in_code_block 
                        else f'{m.name}{punct}'
                    )
            
        # Log transformations
        for pattern, replacement in mention_map.items():
            logging.debug(f"Sanitize transform: {pattern} -> {replacement}")
            
        current_line = line
        # Sort patterns by length (longest first) to avoid partial replacements
        patterns = sorted(mention_map.keys(), key=len, reverse=True)
        for pattern in patterns:
            current_line = current_line.replace(pattern, mention_map[pattern])
            
        formatted_lines.append(current_line)
    
    result = '\n'.join(formatted_lines)
    logging.debug(f"Sanitized mentions result: {result[:100]}...")
    return result

def format_discord_mentions(content: str, guild: discord.Guild, mentions_enabled: bool = True, bot=None) -> str:
    """Convert readable usernames to either Discord mentions or display names."""
    if not content:
        return content
        
    # In DMs (guild is None), use display_name when possible
    if not guild:
        logging.info(f"DM message detected - guild is None. Content preview: {content[:100]}...")
        
        # First log how many @ symbols we have in the content
        at_symbols = content.count('@')
        channel_symbols = content.count('#')
        logging.info(f"Found {at_symbols} '@' symbols and {channel_symbols} '#' symbols in DM message")
        
        # Check for code annotations and preserve them
        code_annotations = ['property', 'staticmethod', 'classmethod', 'decorator', 
                          'param', 'return', 'override', 'abstractmethod']
        
        # If bot is provided, we can directly search all guild members
        if bot:
            # Collect all guild members for direct replacement
            all_members = {}
            all_channels = {}
            
            for bot_guild in bot.guilds:
                # Collect members
                for member in bot_guild.members:
                    # Use member.name as key, mapped to display_name
                    all_members[member.name] = member.display_name
                    # Also add lowercase version for case-insensitive matching
                    all_members[member.name.lower()] = member.display_name
                    # Add display_name as well if different
                    if member.name != member.display_name:
                        all_members[member.display_name] = member.display_name
                
                # Collect channels
                for channel in bot_guild.channels:
                    if hasattr(channel, 'name'):
                        all_channels[channel.name] = channel
                        all_channels[channel.name.lower()] = channel
            
            # Sort member names by length (longer first) to avoid partial replacements
            sorted_names = sorted(all_members.keys(), key=len, reverse=True)
            sorted_channels = sorted(all_channels.keys(), key=len, reverse=True)
            
            # Split content into lines and check each line
            lines = content.split('\n')
            formatted_lines = []
            
            for line in lines:
                # Skip code blocks (marked by ```)
                if '```' in line:
                    formatted_lines.append(line)
                    continue
                
                # Check for code annotations first
                for annotation in code_annotations:
                    if f"@{annotation}" in line:
                        # Preserve the code annotation
                        line = line.replace(f"@{annotation}", f"__CODE_ANNOTATION_{annotation}__")
                
                # Replace each name with display_name
                for name in sorted_names:
                    if f"@{name}" in line:
                        replacement = all_members[name]
                        if mentions_enabled:
                            line = line.replace(f"@{name}", f"@{replacement}")
                        else:
                            line = line.replace(f"@{name}", replacement)
                
                # Replace each channel with channel mention or name
                for name in sorted_channels:
                    if f"#{name}" in line:
                        channel = all_channels[name]
                        if mentions_enabled:
                            line = line.replace(f"#{name}", f"<#{channel.id}>")
                        else:
                            line = line.replace(f"#{name}", f"#{channel.name}")
                
                # Restore code annotations
                for annotation in code_annotations:
                    if f"__CODE_ANNOTATION_{annotation}__" in line:
                        line = line.replace(f"__CODE_ANNOTATION_{annotation}__", f"@{annotation}")
                
                formatted_lines.append(line)
            
            result = '\n'.join(formatted_lines)
            logging.info(f"DM format result: {result[:100]}...")
            return result
        else:
            # If no bot is provided, we can't do much except basic replacement
            logging.warning("Bot instance not provided - can't look up users across guilds")
            
            # Simple code annotation preservation
            for annotation in code_annotations:
                content = content.replace(f"@{annotation}", f"__CODE_ANNOTATION_{annotation}__")
            
            # Keep @ symbols if mentions_enabled, otherwise strip them
            if not mentions_enabled:
                # Remove @ symbols that aren't part of code annotations
                content = re.sub(r'@([\w\.\-_]+)', r'\1', content)
            
            # Restore code annotations
            for annotation in code_annotations:
                content = content.replace(f"__CODE_ANNOTATION_{annotation}__", f"@{annotation}")
            
            logging.info(f"DM format result (basic): {content[:100]}...")
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
                
                # Handle channel name to channel mention conversion
                for channel in sorted(guild.channels, key=lambda c: len(c.name), reverse=True):
                    if hasattr(channel, 'name') and f"#{channel.name}" in current_line:
                        current_line = current_line.replace(f"#{channel.name}", f"<#{channel.id}>")
            else:
                # Transform @username to display name but preserve code-related @ symbols
                def format_mentions_disabled(match):
                    username = match.group(1)
                    # Keep @ for code annotations like @property, @staticmethod, etc.
                    code_annotations = ['property', 'staticmethod', 'classmethod', 'decorator', 
                                      'param', 'return', 'override', 'abstractmethod']
                    if username.lower() in code_annotations:
                        return f"@{username}"
                        
                    # For user mentions, use display_name without @
                    member = discord.utils.get(guild.members, name=username)
                    if member:
                        return member.display_name  # No @ symbol when mentions disabled
                    return username  # No @ symbol when mentions disabled
                
                current_line = re.sub(r'@([\w.]+)', format_mentions_disabled, current_line)
                
        formatted_lines.append(current_line)
    
    result = '\n'.join(formatted_lines)
    logging.info(f"Format mentions result (enabled={mentions_enabled}): {result[:100]}...")
    return result
