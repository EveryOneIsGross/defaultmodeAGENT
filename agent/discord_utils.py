import re
import discord
import logging
import asyncio

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
            
        # Create mention pattern map for current line state - directly use member.name
        mention_map = {
            f'<@{m.id}>': f'@{m.name}' if not in_code_block else m.name
            for m in mentions if hasattr(m, 'name')
        }
        
        # Add punctuation-aware patterns
        for m in mentions:
            if hasattr(m, 'name'):
                for punct in punctuation_marks:
                    mention_map[f'<@{m.id}>{punct}'] = (
                        f'@{m.name}{punct}' if not in_code_block 
                        else f'{m.name}{punct}'
                    )
                    mention_map[f'<@!{m.id}>{punct}'] = (
                        f'@{m.name}{punct}' if not in_code_block 
                        else f'{m.name}{punct}'
                    )
        
        # Add standard mention patterns
        mention_map.update({
            f'<@!{m.id}>': f'@{m.name}' if not in_code_block else m.name
            for m in mentions if hasattr(m, 'name')
        })
        mention_map.update({
            f'<@&{m.id}>': f'@{m.name}' if not in_code_block else m.name
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
    
    # Simple regex for DM fallback case only
    MENTION_RE = re.compile(r'@([\w.\-_]+)')
    
    # Check for code annotations to preserve them
    code_annotations = ['property', 'staticmethod', 'classmethod', 'decorator', 
                      'param', 'return', 'override', 'abstractmethod']
    
    # Preserve code annotations before any processing
    for annotation in code_annotations:
        content = content.replace(f"@{annotation}", f"__CODE_ANNOTATION_{annotation}__")
    
    # In DMs (guild is None), use display_name when possible
    if not guild:
        logging.info(f"DM message detected - guild is None. Content preview: {content[:100]}...")
        
        # If bot is provided, we can directly search all guild members
        if bot:
            # Collect all guild members for direct replacement
            all_members = {}
            all_channels = {}
            
            for bot_guild in bot.guilds:
                # Collect members
                for member in bot_guild.members:
                    # Use member.name as key, mapped to member object (not just display_name)
                    all_members[member.name] = member
                    # Also add lowercase version for case-insensitive matching
                    all_members[member.name.lower()] = member
                    # Add display_name as well if different
                    if member.name != member.display_name:
                        all_members[member.display_name] = member
                
                # Collect channels
                for channel in bot_guild.channels:
                    if hasattr(channel, 'name'):
                        all_channels[channel.name] = channel
                        all_channels[channel.name.lower()] = channel
            
            # Sort member names by length (longer first) to avoid partial replacements
            sorted_names = sorted(all_members.keys(), key=len, reverse=True)
            sorted_channels = sorted(all_channels.keys(), key=len, reverse=True)
            
            # Replace each name with display_name or mention
            for name in sorted_names:
                if f"@{name}" in content:
                    member = all_members[name]
                    # In DMs, always use display_name regardless of mentions_enabled setting
                    # because Discord doesn't render <@id> format properly in DMs
                    content = content.replace(f"@{name}", member.display_name)
            
            # Also handle bare usernames (without @) in DMs and convert to display names
            for name in sorted_names:
                if name in content and name.lower() not in [a.lower() for a in code_annotations]:
                    member = all_members[name]
                    #content = content.replace(name, member.display_name)
                    content = re.sub(r'\b' + re.escape(name) + r'\b', member.display_name, content)
            
            # Replace each channel with channel mention or name
            for name in sorted_channels:
                if f"#{name}" in content:
                    channel = all_channels[name]
                    # In DMs, always use channel name format instead of mentions
                    # for consistency with how user mentions are handled
                    content = content.replace(f"#{name}", f"#{channel.name}")
        else:
            # If no bot is provided, we can't do much except basic replacement
            logging.warning("Bot instance not provided - can't look up users across guilds")
            
            # Keep @ symbols if mentions_enabled, otherwise strip them
            if not mentions_enabled:
                # Remove @ symbols that aren't part of code annotations
                content = MENTION_RE.sub(r'\1', content)
    else:
        # For guild channels, build a dictionary of members for direct lookup
        members_dict = {}
        channels_dict = {}
        
        # Collect members from this guild
        for member in guild.members:
            members_dict[member.name] = member
            members_dict[member.name.lower()] = member
            # Also add display_name if different
            if member.name != member.display_name:
                members_dict[member.display_name] = member
        
        # Collect channels from this guild
        for channel in guild.channels:
            if hasattr(channel, 'name'):
                channels_dict[channel.name] = channel
                channels_dict[channel.name.lower()] = channel
        
        if mentions_enabled:
            # Process line by line with code block detection
            lines = content.split('\n')
            formatted_lines = []
            in_code_block = False
            
            for line in lines:
                if '```' in line:
                    in_code_block = not in_code_block
                
                current_line = line
                
                # Skip conversion for inline code spans and code blocks
                # First, temporarily replace inline code spans to protect them
                inline_code_spans = []
                while '`' in current_line:
                    start_idx = current_line.find('`')
                    if start_idx == -1:
                        break
                    end_idx = current_line.find('`', start_idx + 1)
                    if end_idx == -1:
                        break
                    # Extract the inline code span
                    code_span = current_line[start_idx:end_idx + 1]
                    inline_code_spans.append(code_span)
                    # Replace with placeholder
                    current_line = current_line[:start_idx] + f"__INLINE_CODE_{len(inline_code_spans)-1}__" + current_line[end_idx + 1:]
                
                # Only convert mentions if not in code block
                if not in_code_block:
                    # Convert to Discord mentions using consistent regex pattern
                    def replace_mention(match):
                        username = match.group(1)
                        if username in members_dict:
                            return f"<@{members_dict[username].id}>"
                        elif username.lower() in members_dict:
                            return f"<@{members_dict[username.lower()].id}>"
                        else:
                            return match.group(0)  # Keep original if no match
                    
                    current_line = MENTION_RE.sub(replace_mention, current_line)
                    
                    # Also handle bare usernames (without @) and convert to Discord mentions
                    for member_name, member in sorted(members_dict.items(), key=lambda x: len(x[0]), reverse=True):
                        if member_name in current_line and member_name.lower() not in [a.lower() for a in code_annotations]:
                            current_line = re.sub(r'\b' + re.escape(member_name) + r'\b', f"<@{member.id}>", current_line)
                    
                    # Handle channel name to channel mention conversion
                    for channel_name, channel in sorted(channels_dict.items(), key=lambda x: len(x[0]), reverse=True):
                        if f"#{channel_name}" in current_line:
                            current_line = current_line.replace(f"#{channel_name}", f"<#{channel.id}>")
                
                # Restore inline code spans
                for i, code_span in enumerate(inline_code_spans):
                    current_line = current_line.replace(f"__INLINE_CODE_{i}__", code_span)
                
                formatted_lines.append(current_line)
            
            content = '\n'.join(formatted_lines)
        else:
            # Convert @mentions to display names using direct string matching
            # Sort by length (longest first) to avoid partial replacements
            for member_name, member in sorted(members_dict.items(), key=lambda x: len(x[0]), reverse=True):
                if member_name.lower() in [a.lower() for a in code_annotations]:
                    continue  # Skip code annotations
                
                # Replace @username with display_name
                if f"@{member_name}" in content:
                    content = content.replace(f"@{member_name}", member.display_name)
            
            # Also handle bare usernames (without @) and convert to display names
            for member_name, member in sorted(members_dict.items(), key=lambda x: len(x[0]), reverse=True):
                if member_name in content and member_name.lower() not in [a.lower() for a in code_annotations]:
                    #content = content.replace(member_name, member.display_name)
                    content = re.sub(r'\b' + re.escape(member_name) + r'\b', member.display_name, content)
            
            # Process channel mentions
            channel_mentions = re.findall(r'#([\w.\-_]+)', content)
            
            for channel_mention in sorted(channel_mentions, key=len, reverse=True):
                if channel_mention in channels_dict:
                    channel = channels_dict[channel_mention]
                    content = content.replace(f"#{channel_mention}", f"#{channel.name}")
    
    # Restore code annotations
    for annotation in code_annotations:
        content = content.replace(f"__CODE_ANNOTATION_{annotation}__", f"@{annotation}")
    
    logging.info(f"Format mentions result (enabled={mentions_enabled}): {content[:100]}...")
    return content