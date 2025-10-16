#discord
import discord
from discord import TextChannel
from discord.ext import commands
# standard libraries
import asyncio
import os
import contextvars
import requests
from requests.exceptions import RequestException, Timeout, SSLError
from bs4 import BeautifulSoup
import mimetypes
import json
import yaml
from datetime import datetime
import time
import argparse
import threading
import re
import importlib.util
import sys
# api import and hyperparameter handlers
from api_client import initialize_api_client, call_api, update_api_temperature, api
from hippocampus import Hippocampus, HippocampusConfig
# image handling
from PIL import Image
import io
import traceback
# import tools
from tools.discordSUMMARISER import ChannelSummarizer
from tools.discordGITHUB import GitHubRepo, RepoIndex, process_repo_contents, repo_processing_event
from tools.webSCRAPE import scrape_webpage
# import memory module
from memory import UserMemoryIndex, CacheManager
from defaultmode import DMNProcessor
from chunker import truncate_middle, clean_response, balance_wraps
from temporality import TemporalParser
# Discord Format Handling
from discord_utils import sanitize_mentions, format_discord_mentions
from attention import check_attention_triggers_fuzzy, get_user_themes, get_current_themes, force_rebuild_theme_cache, force_rebuild_user_theme_cache, _get_user_count, format_themes_for_prompt
# Configuration imports
from bot_config import (
    config,
    APIConfig,
    DiscordConfig,
    FileConfig,
    SearchConfig,
    ConversationConfig,
    PersonaConfig,
    NotionConfig,
    TwitterConfig,
    SystemConfig,
    BotConfig,
    init_logging
)
# libraries logging import for jsonl, sqlite and info logging
from logger import BotLogger

init_logging()

script_dir = os.path.dirname(os.path.abspath(__file__))

_themes_ctx = contextvars.ContextVar('themes_ctx', default={})

def format_themes_for_prompt_memoized(mi, uid, mode="sections"):
    d=_themes_ctx.get()
    k=(uid,mode)
    if k in d:return d[k]
    s=format_themes_for_prompt(mi,uid,mode=mode)
    d[k]=s
    _themes_ctx.set(d)
    return s

# Access config values
HIPPOCAMPUS_BANDWIDTH = config.persona.hippocampus_bandwidth
MAX_CONVERSATION_HISTORY = config.conversation.max_history
MINIMAL_CONVERSATION_HISTORY = config.conversation.minimal_history
TRUNCATION_LENGTH = config.conversation.truncation_length
WEB_CONTENT_TRUNCATION_LENGTH = config.conversation.web_content_truncation_length
HARSH_TRUNCATION_LENGTH = config.conversation.harsh_truncation_length
TEMPERATURE = config.persona.temperature
DEFAULT_AMYGDALA_RESPONSE = config.persona.default_amygdala_response
ALLOWED_EXTENSIONS = config.files.allowed_extensions
ALLOWED_IMAGE_EXTENSIONS = config.files.allowed_image_extensions
DISCORD_BOT_MANAGER_ROLE = config.discord.bot_manager_role
TICK_RATE = config.system.tick_rate
MEMORY_CAPACITY = config.persona.memory_capacity
MOOD_COEFF = config.persona.mood_coefficient

# Attention system control
attention_enabled = True

def log_to_jsonl(data, bot_id=None):
    """Log data to JSONL file and SQLite database with consistent timestamp format.
    
    Args:
        data (dict): Data to log
        bot_id (str, optional): Bot identifier for log filename. If None, uses current bot's name.
    """
    # Get or create logger instance using bot's name if not specified
    if not hasattr(log_to_jsonl, '_logger'):
        log_to_jsonl._logger = None    
    # Update logger if bot_id changes or not initialized
    current_bot_id = bot_id or (bot.user.name if bot and bot.user else "default")
    if not log_to_jsonl._logger or log_to_jsonl._logger.bot_id != current_bot_id:
        log_to_jsonl._logger = BotLogger(current_bot_id)
    # Use logger to handle both JSONL and SQLite logging
    log_to_jsonl._logger.log(data)

def update_temperature(intensity: int) -> None:
    """Update the bot's temperature based on intensity value. Manages Amagdala response and API client temperature/top p."""
    TEMPERATURE = intensity / 100.0
    bot.update_api_temperature(intensity)
    if hasattr(bot, 'dmn_processor') and bot.dmn_processor:
        bot.dmn_processor.amygdala_response = intensity
        bot.dmn_processor.temperature = TEMPERATURE
    bot.logger.info(f"Updated bot temperature to {TEMPERATURE} across all components")

def currentmoment():
    return datetime.now().strftime("%H:%M [%d/%m/%y]")

def start_background_processing_thread(repo, memory_index, max_depth=None, branch='main', channel=None):
    thread = threading.Thread(target=run_background_processing, args=(repo, memory_index, max_depth, branch))
    thread.start()
    bot.logger.info(f"Started background processing of repository contents in a separate thread (Branch: {branch}, Max Depth: {max_depth if max_depth is not None else 'Unlimited'})")

def run_background_processing(repo, memory_index, max_depth=None, branch='main', channel=None):
    global repo_processing_event
    repo_processing_event.clear()
    try:
        asyncio.run(process_repo_contents(repo, '', memory_index, max_depth, branch))
        memory_index.save_cache()  # Save the cache after indexing
        if channel:
            asyncio.get_event_loop().create_task(channel.send(f"Repository indexing completed for branch '{branch}'"))
    except Exception as e:
        bot.logger.error(f"Error in background processing for branch '{branch}': {str(e)}")
    finally:
        repo_processing_event.set()

async def maintain_typing_state(channel):
    """Maintains typing state in channel by refreshing before timeout."""
    try:
        async with channel.typing():
            # Keep typing state active for up to 5 minutes
            await asyncio.sleep(300)
    except Exception as e:
        bot.logger.debug(f"Typing state maintenance ended: {str(e)}")

async def process_message(message, memory_index, prompt_formats, system_prompts, github_repo, is_command=False):
    '''Main plan text message processing function for Discord bot.'''
    bot.logger.debug(f"Processing message from {message.author.name}")
    #bot.logger.debug(f"Raw content: {message.content}")
    #bot.logger.debug(f"Mentions: {[m.name for m in message.mentions]}")
    """Process an incoming Discord message and generate an appropriate response."""
    if not getattr(bot, 'processing_enabled', True):
        await message.channel.send("BBL... ☕")
        return
    user_id = str(message.author.id)
    user_name = message.author.name
    # Extract URLs early in the flow
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message.content)
    # Use memory_index for first interaction detection
    is_first_interaction = not bool(memory_index.user_memories.get(user_id, []))
    
    # Process content and handle mentions
    if is_command:
        parts = message.content.split(maxsplit=1)
        content = parts[1] if len(parts) > 1 else ""
    else:
        if message.guild and message.guild.me:
            content = message.content.replace(f'<@!{message.guild.me.id}>', '').replace(f'<@{message.guild.me.id}>', '').strip()
        else:
            content = message.content.strip()
        reply_context = None
        if message.reference and not is_command:
            try:
                original = await message.channel.fetch_message(message.reference.message_id)
                original_content = original.content.strip()
                original_author = original.author.name
                replier = message.author.name
                if original_content:
                    for mention in original.mentions:
                        original_content = original_content.replace(f'<@{mention.id}>', f'@{mention.name}')
                        original_content = original_content.replace(f'<@!{mention.id}>', f'@{mention.name}')
                    for channel in original.channel_mentions:
                        original_content = original_content.replace(f'<#{channel.id}>', f'#{channel.name}')
                    #reply_context = f"@{original_author}: {original_content}"
                    reply_context = f"{original_content}"
                    content = f"[@{replier} replying to @{original_author}: {original_content}]\n\n@{replier}: {content}"
            except (discord.NotFound, discord.Forbidden):
                pass

    
    '''
    # Process content and handle mentions
    if is_command:
        parts = message.content.split(maxsplit=1)
        content = parts[1] if len(parts) > 1 else ""
    else:
        if message.guild and message.guild.me:
            content = message.content.replace(f'<@!{message.guild.me.id}>', '').replace(f'<@{message.guild.me.id}>', '').strip()
        else:
            content = message.content.strip()
        # Add reply context for non-command messages
        reply_context = None
        if message.reference and not is_command:
            try:
                original = await message.channel.fetch_message(message.reference.message_id)
                original_content = original.content.strip()
                original_author = original.author.name
                if original_content:
                    # Use utilities for consistent mention handling
                    for mention in original.mentions:
                        original_content = original_content.replace(f'<@{mention.id}>', f'@{mention.name}')
                        original_content = original_content.replace(f'<@!{mention.id}>', f'@{mention.name}')
                    for channel in original.channel_mentions:
                        original_content = original_content.replace(f'<#{channel.id}>', f'#{channel.name}')
                    reply_context = f"@{original_author}: {original_content}"
                    content = f"[@{original_author} replying to @{original_author}: {original_content}]\n\n @{original_author}: {content}"
            except (discord.NotFound, discord.Forbidden):
                pass
    '''
    combined_mentions = list(message.mentions) + list(message.channel_mentions)
    sanitized_content = sanitize_mentions(content, combined_mentions)
    #bot.logger.info(f"Received message from {user_name} (ID: {user_id}): {sanitized_content}")
    try:
        response_content = None
        try:
            is_dm = isinstance(message.channel, discord.DMChannel)
            # Get conversation context first
            conversation_context = ""
            async for msg in message.channel.history(limit=MAX_CONVERSATION_HISTORY):
                if msg.id != message.id:  # Skip the current message
                    clean_name = msg.author.name
                    combined_mentions = list(msg.mentions) + list(msg.channel_mentions)
                    msg_content = sanitize_mentions(msg.content, combined_mentions)
                    conversation_context += f"@{clean_name}: {msg_content}\n"
            # Enhance search query with contextual information
            context_parts = [sanitized_content]
            sanitized_user_name = sanitize_mentions(user_name, combined_mentions)
            context_parts.append(f"@{sanitized_user_name}")
            if not is_dm:
                channel_name = message.channel.name if hasattr(message.channel, 'name') else 'DM'
                sanitized_channel_name = sanitize_mentions(channel_name, combined_mentions)
                context_parts.append(f"#{sanitized_channel_name}")
            search_query = " ".join(context_parts)
            # Get initial candidate memories using the async method
            candidate_memories = await memory_index.search_async(
                search_query, 
                k=MEMORY_CAPACITY,  
                user_id=(user_id if is_dm else None)
            )
            # Process memories based on reranking setting
            if candidate_memories:
                #bot.logger.info(f"Candidate memories: {candidate_memories}")
                if config.persona.use_hippocampus_reranking:
                    # Initialize Hippocampus for reranking
                    hippocampus_config = HippocampusConfig(blend_factor=config.persona.reranking_blend_factor)
                    hippocampus = Hippocampus(hippocampus_config)
                    # Calculate and log threshold - high amygdala = lower threshold (more permissive)
                    amygdala_scale = bot.amygdala_response / 100.0  # 0-1 scale
                    threshold = max(config.persona.minimum_reranking_threshold, HIPPOCAMPUS_BANDWIDTH - (MOOD_COEFF * amygdala_scale))  # Fixed: subtract amygdala influence
                    bot.logger.info(f"Memory reranking threshold: {threshold:.3f} (bandwidth: {HIPPOCAMPUS_BANDWIDTH}, amygdala: {bot.amygdala_response}%, influence: {MOOD_COEFF * amygdala_scale:.3f})")
                    # Rerank memories with blended weights
                    relevant_memories = await hippocampus.rerank_memories(
                        query=search_query,  
                        memories=candidate_memories,
                        threshold=threshold,
                        blend_factor=config.persona.reranking_blend_factor
                    )
                    bot.logger.info(f"Reranked memories: {relevant_memories}")
                    # Log the memories that passed the threshold
                    bot.logger.info(f"Found {len(relevant_memories)} memories above threshold {threshold:.3f}:")
                    for memory, score in relevant_memories:
                        bot.logger.info(f"Memory score {score:.3f}: {memory[:100]}...")
                else:
                    # Skip reranking, use candidate memories directly
                    relevant_memories = candidate_memories
                    bot.logger.info(f"Using memories without reranking (reranking disabled): {len(relevant_memories)} memories")
            else:
                relevant_memories = []
                bot.logger.info("No candidate memories found for reranking")
            # Build memory context
            if hasattr(message.channel, 'name'):
                context = f"Current Discord server: {message.guild.name}, channel: #{message.channel.name}\n"
            else:
                context = "Current channel: Direct Message\n"
            #context += "When referring to users, include their @ symbol with their username.\n\n"
            if relevant_memories:
                context += f"{len(relevant_memories)} Potentially Relevant Memories:\n"
                context += "<memories>\n"
                for memory, score in relevant_memories:
                    # Convert timestamps in memory to temporal expressions
                    timestamp_pattern = r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)'
                    parsed_memory = re.sub(
                        timestamp_pattern,
                        lambda m: f"({bot.temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})",
                        memory
                    )
                    truncated_memory = truncate_middle(parsed_memory, max_tokens=TRUNCATION_LENGTH)
                    context += f"[Relevance: {score:.2f}] {truncated_memory}\n"
                context += "</memories>\n\n"
            # Process URLs if present
            if urls:
                url_contents = []
                for url in urls:
                    webpage_data = await scrape_webpage(url)
                    if webpage_data['content_type'] != 'error':
                        url_contents.append(f"URL Content: {url}\nTitle: {webpage_data['title']}\nDescription: {webpage_data['description']}\n\nContent:\n{webpage_data['content']}")
                    else:
                        await message.channel.send(f"Error scraping URL {url}: {webpage_data['error']}")
                if url_contents:
                    context += "\nWeb Page Content:\n<web_content>\n"
                    for content in url_contents:
                        context += f"{truncate_middle(content, max_tokens=WEB_CONTENT_TRUNCATION_LENGTH)}\n"
                    context += "</web_content>\n\n"
            context += "**Ongoing Channel Conversation:**\n\n"
            context += "<conversation>\n"
            messages = []
            async for msg in message.channel.history(limit=MAX_CONVERSATION_HISTORY):
                if msg.id != message.id:  # Skip the current message
                    clean_name = msg.author.name
                    combined_mentions = list(msg.mentions) + list(msg.channel_mentions)
                    msg_content = sanitize_mentions(msg.content, combined_mentions)
                    truncated_content = truncate_middle(msg_content, max_tokens=TRUNCATION_LENGTH)
                    # Convert Discord timestamp (UTC) to local time, then to pipeline format
                    local_timestamp = msg.created_at.astimezone().replace(tzinfo=None)
                    msg_timestamp_str = local_timestamp.strftime("%H:%M [%d/%m/%y]")
                    temporal_expr = bot.temporal_parser.get_temporal_expression(msg_timestamp_str)
                    formatted_msg = f"@{clean_name} ({temporal_expr.base_expression}): {truncated_content}"
                    # Add reactions if present
                    if msg.reactions:
                        reaction_parts = []
                        for reaction in msg.reactions:
                            reaction_emoji = str(reaction.emoji)
                            async for user in reaction.users():
                                reaction_user_name = user.name
                                reaction_parts.append(f"@{reaction_user_name}: {reaction_emoji}")
                        
                        if reaction_parts:
                            formatted_msg += f"\n(Discord User Reactions: {' '.join(reaction_parts)})"
                    
                    messages.append(formatted_msg)
            
            for msg in reversed(messages):
                context += f"{msg}\n"
            context += "</conversation>\n"
            
            prompt_key = 'introduction' if is_first_interaction else 'chat_with_memory'
            prompt = prompt_formats[prompt_key].format(
                context=sanitize_mentions(context, combined_mentions),
                user_name=user_name,
                user_message=sanitize_mentions(sanitized_content, combined_mentions)
            )

            themes=format_themes_for_prompt_memoized(bot.memory_index,user_id,mode="sections")

            system_prompt_key = 'default_chat'
            system_prompt = system_prompts[system_prompt_key].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)

            typing_task = asyncio.create_task(maintain_typing_state(message.channel))
            try:
                response_content = await bot.call_api(prompt, context=context, system_prompt=system_prompt, temperature=bot.amygdala_response/100)
                response_content = clean_response(response_content)
            finally:
                typing_task.cancel()
        finally:
            pass
            
        if response_content:
            # Add logging to help debug DM issues
            is_dm = isinstance(message.channel, discord.DMChannel)
            #bot.logger.info(f"Formatting response for channel type: {'DM' if is_dm else 'Guild'}")
            #bot.logger.info(f"Channel has guild: {message.guild is not None}, Guild name: {message.guild.name if message.guild else 'None'}")
            formatted_content = format_discord_mentions(response_content, message.guild, bot.mentions_enabled, bot)
            await send_long_message(message.channel, formatted_content, bot=bot)
            #bot.logger.info(f"Sent response to {user_name} (ID: {user_id}): {response_content[:1000]}...")
            timestamp = currentmoment()
            channel_name = message.channel.name if hasattr(message.channel, 'name') else 'DM'
            # Create complete interaction memory including both user input and bot response
            if hasattr(message.channel, 'name'):
                memory_text = (
                    f"@{user_name} in {message.guild.name} #{channel_name} ({timestamp}): "
                    f"{sanitize_mentions(sanitized_content, combined_mentions)}\n"
                    f"@{bot.user.name}: {response_content}"
                )
            else:
                memory_text = (
                    f"@{user_name} in DM ({timestamp}): "
                    f"{sanitize_mentions(sanitized_content, combined_mentions)}\n"
                    f"@{bot.user.name}: {response_content}"
                )
            #memory_index.add_memory(user_id, memory_text)

            await memory_index.add_memory_async(user_id, memory_text)
            
            asyncio.create_task(generate_and_save_thought(
                memory_index=memory_index,
                user_id=user_id,
                user_name=user_name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot,
                conversation_context=context
            ))
            # Maintain only JSONL logging for persistence
            log_to_jsonl({
                'event': 'chat_interaction',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'user_name': user_name,
                'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM',
                'user_message': sanitized_content,
                'reply_to': reply_context,
                'ai_response': response_content,
                'system_prompt': system_prompt,
                'prompt': prompt,
                'temperature': bot.amygdala_response/100
            }, bot_id=bot.user.name)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await send_long_message(message.channel, error_message, bot=bot)
        bot.logger.error(f"Error in message processing for {user_name} (ID: {user_id}): {str(e)}")
        log_to_jsonl({
            'event': 'chat_error',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM',
            'error': str(e)
        }, bot_id=bot.user.name)

async def process_files(message, memory_index, prompt_formats, system_prompts, user_message="", bot=None, temperature=TEMPERATURE):
    """Secondary entry for multiple files from a Discord message, handling combinations of images and text files, including resizing large images."""
    if not getattr(bot, 'processing_enabled', True):
        await message.channel.send("Processing currently disabled.")
        return
    user_id = str(message.author.id)
    user_name = message.author.name
    # Extract URLs from message content
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message.content)
    if not message.attachments and not urls:
        await message.channel.send("No attachments or URLs found.") 
        return
    # Track files for combined analysis
    image_files = []
    text_contents = []
    temp_paths = []
    response_content = None
    # Track detected types for prompt selection
    has_images = False
    has_text = False
    # Process URLs first
    if urls:
        for url in urls:
            webpage_data = await scrape_webpage(url)
            if webpage_data['content_type'] != 'error':
                text_contents.append({
                    'filename': f"webpage_{webpage_data['title']}",
                    'content': f"URL: {webpage_data['url']}\nTitle: {webpage_data['title']}\nDescription: {webpage_data['description']}\n\nContent:\n{webpage_data['content']}"
                })
                has_text = True
            else:
                await message.channel.send(f"Error scraping URL {url}: {webpage_data['error']}")
    # Standardize mention handling
    if message.guild and message.guild.me:
        user_message = message.content.replace(f'<@!{message.guild.me.id}>', '').replace(f'<@{message.guild.me.id}>', '').strip()
    combined_mentions = list(message.mentions) + list(message.channel_mentions)  
    user_message = sanitize_mentions(user_message, combined_mentions)
    bot.logger.info(f"Processing {len(message.attachments)} files from {user_name} (ID: {user_id}) with message: {user_message}")
    try:
        amygdala_response = str(bot.amygdala_response if bot else DEFAULT_AMYGDALA_RESPONSE)
        themes = ", ".join(get_current_themes(bot.memory_index))
        #bot.logger.info(f"Using amygdala arousal: {amygdala_response}")
        context = f"Current channel: #{message.channel.name if hasattr(message.channel, 'name') else 'Direct Message'}\n\n"
        #context += "Ongoing Chatroom Conversation:\n\n"
        context += "<conversation>\n"
        messages = []
        async for msg in message.channel.history(limit=MINIMAL_CONVERSATION_HISTORY):
            combined_mentions = list(msg.mentions) + list(msg.channel_mentions)
            msg_content = sanitize_mentions(msg.content, combined_mentions)
            truncated_content = truncate_middle(msg_content, max_tokens=HARSH_TRUNCATION_LENGTH)
            author_name = msg.author.name
            # appends @ for chatroom user names -- this needs fixing....
            display_text = f"@{('@' + author_name) if not msg.author.bot else author_name}: {truncated_content}"
            # Add reactions if present
            if msg.reactions:
                reaction_parts = []
                for reaction in msg.reactions:
                    reaction_emoji = str(reaction.emoji)
                    async for user in reaction.users():
                        reaction_user_name = user.name
                        reaction_parts.append(f"@{reaction_user_name}: {reaction_emoji}")
                if reaction_parts:
                    display_text += f" ({ ' '.join(reaction_parts) })"
            messages.append(display_text)
        for msg in reversed(messages):
            context += f"{msg}\n"
        context += "</conversation>\n"
        # Main processing block
        try:
            # Loop through attachments, handle size check and potential resizing here
            for attachment in message.attachments:
                ext = os.path.splitext(attachment.filename.lower())[1]
                is_potentially_image = (attachment.content_type and 
                                      attachment.content_type.startswith('image/') and 
                                      ext in ALLOWED_IMAGE_EXTENSIONS)
                is_potentially_text = ext in ALLOWED_EXTENSIONS
                data_to_save = None
                processed_as_image = False
                processed_as_text = False
                if attachment.size > 1000000:
                    # --- Handling for OVERSIZED attachments --- 
                    if is_potentially_image:
                        try:
                            image_data = await attachment.read()
                            #bot.logger.info(f"Downloaded large image data: {len(image_data)} bytes")                         
                            img = Image.open(io.BytesIO(image_data))
                            img.load()
                            #bot.logger.info(f"Large image opened: {img.format}, {img.size}, {img.mode}")
                            MAX_DIMENSION = 512 # Still use a dimension cap
                            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
                            #bot.logger.info(f"Resized large image to fit within {MAX_DIMENSION}x{MAX_DIMENSION}")
                            output_buffer = io.BytesIO()
                            save_format = 'PNG' if img.mode == 'RGBA' else 'JPEG'
                            if img.mode == 'P': img = img.convert('RGB'); save_format = 'JPEG'
                            elif img.mode == 'LA': img = img.convert('RGBA'); save_format = 'PNG'
                            img.save(output_buffer, format=save_format)
                            resized_data = output_buffer.getvalue()
                            if len(resized_data) > 1000000: # Check size *after* resize
                                bot.logger.warning(f"Image {attachment.filename} still too large after resizing.")
                                await message.channel.send(f"Sorry, could not resize {attachment.filename} sufficiently. Skipping.")
                                continue 
                            else:
                                data_to_save = resized_data
                                processed_as_image = True # Mark as successfully processed image
                        except Exception as resize_error:
                            bot.logger.error(f"Error resizing image {attachment.filename}: {str(resize_error)}")
                            bot.logger.error(traceback.format_exc())
                            await message.channel.send(f"Error processing large image {attachment.filename}. Skipping.")
                            continue # Skip on error
                    else:
                        # Oversized and not an image
                        bot.logger.warning(f"Skipping oversized non-image file: {attachment.filename}")
                        await message.channel.send(f"Skipping {attachment.filename} - file is over 1MB and not a resizable image.")
                        continue # Skip this attachment
                else:
                    # --- Handling for attachments UNDER OR EQUAL 1MB --- 
                    if is_potentially_image:
                        try:
                            image_data = await attachment.read()
                            data_to_save = image_data 
                            processed_as_image = True
                            try:
                                img = Image.open(io.BytesIO(data_to_save))
                                img.verify() # Quick check
                            except Exception as verify_err:
                                bot.logger.warning(f"Small image {attachment.filename} failed verification: {verify_err}. Still attempting to use.")

                        except Exception as img_error:
                            bot.logger.error(f"Error processing small image {attachment.filename}: {str(img_error)}")
                            continue 
                    elif is_potentially_text:
                        try:
                            content_bytes = await attachment.read()
                            text_content = content_bytes.decode('utf-8')
                            text_contents.append({
                                'filename': attachment.filename,
                                'content': text_content
                            })
                            processed_as_text = True
                        except UnicodeDecodeError as e:
                            bot.logger.error(f"Error decoding text file {attachment.filename}: {str(e)}")
                            await message.channel.send(
                                f"Warning: {attachment.filename} couldn't be decoded as UTF-8. Skipping."
                            )
                            continue
                        except Exception as e:
                            bot.logger.error(f"Error reading text file {attachment.filename}: {str(e)}")
                            continue
                    else:
                         await message.channel.send(
                            f"Skipping {attachment.filename} - unsupported type. "
                            f"Supported types: { ', '.join(ALLOWED_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS) }"
                         )
                         continue # Skip unsupported type
                # --- Save processed data to temp file --- 
                if processed_as_image and data_to_save:
                    try:
                        file_cache = bot.cache_managers['file']
                        temp_path, file_id = file_cache.create_temp_file(
                            user_id=user_id,
                            prefix="img_",
                            suffix=os.path.splitext(attachment.filename)[1],
                            content=data_to_save
                        )
                        if not os.path.exists(temp_path):
                             bot.logger.error(f"Failed to save image to temp file: {temp_path}")
                             continue # Skip if saving failed
                        image_files.append(attachment.filename)
                        temp_paths.append(temp_path)
                        has_images = True 
                    except Exception as e:
                         bot.logger.error(f"Error saving temp image file {attachment.filename}: {str(e)}")
                         continue 
                elif processed_as_text:
                     has_text = True
            # --- End of attachment loop --- 
            if not (has_images or has_text):
                if not message.channel.last_message or message.channel.last_message.author != bot.user:
                     await message.channel.send("No valid files found to analyze after processing.")
                return 
            if has_images and has_text:
                if 'analyze_combined' not in prompt_formats or 'combined_analysis' not in system_prompts:
                    raise ValueError("Missing required combined analysis prompts")
            elif has_images:
                if 'analyze_image' not in prompt_formats or 'image_analysis' not in system_prompts:
                    raise ValueError("Missing required image analysis prompts")
            else:  # has_text
                if 'analyze_file' not in prompt_formats or 'file_analysis' not in system_prompts:
                    raise ValueError("Missing required file analysis prompts")
            if image_files and text_contents:
                prompt = prompt_formats['analyze_combined'].format(
                    context=context,
                    image_files="\n".join(image_files),
                    text_files="\n".join(f"{t['filename']}: {truncate_middle(t['content'], 1000)}" for t in text_contents),
                    user_message=user_message if user_message else "Please analyze these files.",
                    user_name=user_name
                )
                system_prompt = system_prompts['combined_analysis'].replace(
                    '{amygdala_response}',
                    amygdala_response
                ).replace('{themes}', themes)
            elif image_files:
                prompt = prompt_formats['analyze_image'].format(
                    context=context,
                    filename=", ".join(image_files),
                    user_message=user_message if user_message else "Please analyze these images.",
                    user_name=user_name
                )
                system_prompt = system_prompts['image_analysis'].replace(
                    '{amygdala_response}',
                    amygdala_response
                ).replace('{themes}', themes)
            else:
                combined_text = "\n\n".join(f"=== {t['filename']} ===\n{t['content']}" for t in text_contents)
                prompt = prompt_formats['analyze_file'].format(
                    context=context,
                    filename=", ".join(t['filename'] for t in text_contents),
                    file_content=combined_text,
                    user_message=user_message,
                    user_name=user_name
                )
                system_prompt = system_prompts['file_analysis'].replace(
                    '{amygdala_response}',
                    amygdala_response
                ).replace('{themes}', themes)
            #bot.logger.info(f"Using prompt type: {'combined' if has_images and has_text else 'image' if has_images else 'text'}")
            typing_task = asyncio.create_task(maintain_typing_state(message.channel))
            try:
                response_content = await bot.call_api(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    image_paths=temp_paths if temp_paths else None,
                    temperature=bot.amygdala_response/100
                )
                response_content = clean_response(response_content)
            finally:
                typing_task.cancel()

            if response_content:
                formatted_content = format_discord_mentions(response_content, message.guild, bot.mentions_enabled, bot)
                await send_long_message(message.channel, formatted_content, bot=bot)

        finally:
            pass
        if response_content:
            # Save memory and generate thought
            files_description = []
            if image_files:
                files_description.append(f"{len(image_files)} images: {', '.join(image_files)}")
            if text_contents:
                files_description.append(f"{len(text_contents)} text files: {', '.join(t['filename'] for t in text_contents)}")
            timestamp = currentmoment()
            channel_name = message.channel.name if hasattr(message.channel, 'name') else 'DM'
            memory_text = f"({timestamp}) Grokking {' and '.join(files_description)} for User @{user_name} in #{channel_name}. User's message: {sanitize_mentions(user_message, combined_mentions)}\n@{bot.user.name}: {response_content}"
            await memory_index.add_memory_async(user_id, memory_text)
            file_context = ""
            if text_contents:
                file_context += "File Contents:\n"
                for file_data in text_contents:
                    truncated_content = truncate_middle(file_data['content'], max_tokens=TRUNCATION_LENGTH)
                    file_context += f"--- {file_data['filename']} ---\n{truncated_content}\n\n"
            if image_files:
                file_context += f"Images analyzed: {', '.join(image_files)}\n"
            def cleanup_temp_files():
                if hasattr(bot, 'cache_managers') and 'file' in bot.cache_managers:
                    bot.cache_managers['file'].cleanup_temp_files(force=True)
                else:
                    for temp_path in temp_paths:
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                                bot.logger.info(f"Removed temporary file: {temp_path}")
                        except Exception as e:
                            bot.logger.error(f"Error removing temporary file {temp_path}: {str(e)}")

            asyncio.create_task(generate_and_save_thought(
                memory_index=memory_index,
                user_id=user_id,
                user_name=user_name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot,
                file_context=file_context,
                image_paths=temp_paths if temp_paths else None,
                cleanup_callback=cleanup_temp_files
            ))

            log_to_jsonl({
                'event': 'file_analysis',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'user_name': user_name,
                'files_processed': {
                    'images': image_files,
                    'text_files': [t['filename'] for t in text_contents]
                },
                'user_message': user_message,
                'ai_response': response_content
            }, bot_id=bot.user.name)

    except Exception as e:
        error_message = f"An error occurred while analyzing files: {str(e)}"
        await send_long_message(message.channel, error_message, bot=bot)
        bot.logger.error(f"Error in file analysis for {user_name} (ID: {user_id}): {str(e)}")
        bot.logger.error(traceback.format_exc())
        
    finally:
        # Cleanup will happen after thought generation completes
        pass

async def send_long_message(channel: discord.TextChannel, text: str, max_length=1800, bot=None):
    '''Send a long message to a Discord channel, splitting it into chunks if necessary while preserving formatting.'''
    if not text:
        return
    guild = getattr(channel, 'guild', None)
    is_dm = isinstance(channel, discord.DMChannel)
    formatted_text = text
    segments = []
    lines = formatted_text.split('\n')
    current_segment = []
    in_code_block = False
    tag_stack = []
    for line in lines:
        if '```' in line:
            if not in_code_block:
                if current_segment:
                    segments.append(('\n'.join(current_segment), False))
                    current_segment = []
                in_code_block = True
            else:
                in_code_block = False
                current_segment.append(line)
                segments.append(('\n'.join(current_segment), True))
                current_segment = []
                continue
        if not in_code_block:
            opens = line.count('<')
            closes = line.count('>')
            if opens > closes:
                tag_stack.extend(['<'] * (opens - closes))
            elif closes > opens and tag_stack:
                tag_stack = tag_stack[:(opens - closes)]
                
            if tag_stack and not current_segment:
                if current_segment:
                    segments.append(('\n'.join(current_segment), False))
                    current_segment = []
            elif not tag_stack and current_segment and any('<' in s or '>' in s for s in current_segment):
                current_segment.append(line)
                segments.append(('\n'.join(current_segment), True))
                current_segment = []
                continue
        current_segment.append(line)
    if current_segment:
        segments.append(('\n'.join(current_segment), in_code_block or bool(tag_stack)))
    chunks = []
    current_chunk = []
    current_length = 0
    for content, is_wrapped in segments:
        if is_wrapped:
            if len(content) > max_length:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                if '\n' not in content:
                    remaining = content
                    while remaining:
                        chunk_size = max_length - 6
                        if remaining.startswith('```'):
                            chunk = remaining[:chunk_size] + '\n```'
                            remaining = '```\n' + remaining[chunk_size:] if remaining[chunk_size:] else ''
                        else:
                            chunk = remaining[:chunk_size]
                            remaining = remaining[chunk_size:]
                        chunks.append(chunk)
                else:
                    balanced = balance_wraps(content)
                    while balanced:
                        original_length = len(balanced)
                        split_point = balanced.rfind('\n', 0, max_length)
                        if split_point == -1:
                            split_point = max_length - 6
                        chunk = balanced[:split_point]
                        if '```' in chunk and chunk.count('```') % 2 != 0:
                            chunk += '\n```'
                        chunks.append(chunk)
                        balanced = balanced[split_point:].lstrip()
                        # Safety check - ensure we're making progress
                        if len(balanced) >= original_length:
                            # Force split if we're stuck
                            chunks.append(balanced[:max_length-6])
                            balanced = balanced[max_length-6:].lstrip()
                        # Emergency break if balanced is not getting smaller
                        if len(balanced) < 6:  # Minimum viable remainder
                            if balanced:
                                chunks.append(balanced)
                            break
                        if '```' in chunk and chunk.endswith('```') and balanced:
                            balanced = '```\n' + balanced
            else:
                if current_length + len(content) + 1 > max_length:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [content]
                    current_length = len(content)
                else:
                    current_chunk.append(content)
                    current_length += len(content) + 1
        else:
            lines = content.split('\n')
            for line in lines:
                while len(line) > max_length:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                    chunks.append(line[:max_length])
                    line = line[max_length:]
                    current_length = 0
                if current_length + len(line) + 1 > max_length:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_length = len(line)
                else:
                    current_chunk.append(line)
                    current_length += len(line) + 1
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    # Send chunks with rate limit handling
    for chunk in chunks:
        if not chunk.strip():  # Skip empty chunks
            continue
        max_retries = 3
        retry_count = 0
        base_delay = 0.5
        while retry_count < max_retries:
            try:
                await channel.send(chunk.strip())
                await asyncio.sleep(0.1)  
                break
            except discord.HTTPException as e:
                if e.status == 429:  # Rate limit hit
                    retry_count += 1
                    if retry_count == max_retries:
                        if bot and bot.logger:
                            bot.logger.error("Max retries reached for message chunk. Skipping.")
                        break
                        
                    retry_after = getattr(e, 'retry_after', base_delay * (2 ** retry_count))
                    if bot and bot.logger:
                        bot.logger.warning(f"Rate limited. Waiting {retry_after:.2f}s before retry {retry_count}/{max_retries}")
                    await asyncio.sleep(retry_after)
                else:
                    if bot and bot.logger:
                        bot.logger.error(f"Error sending message chunk: {str(e)}")
                    break

async def generate_and_save_thought(memory_index, user_id, user_name, memory_text, prompt_formats, system_prompts, bot, file_context=None, image_paths=None, cleanup_callback=None, conversation_context=None):
    """
    Generates a thought about a memory and saves both to the memory index.
    """
    current_time = datetime.now()
    storage_timestamp = current_time.strftime("%H:%M [%d/%m/%y]")
    temporal_expr = bot.temporal_parser.get_temporal_expression(current_time)
    temporal_timestamp = temporal_expr.base_expression
    if temporal_expr.time_context:
        temporal_timestamp = f"{temporal_timestamp} in the {temporal_expr.time_context}"
    timestamp_pattern = r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)'
    temporal_memory_text = re.sub(
        timestamp_pattern,
        lambda m: f"({bot.temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})",
        memory_text
    )
    thought_prompt = prompt_formats['generate_thought'].format(
        user_name=user_name,
        memory_text=temporal_memory_text,
        timestamp=temporal_timestamp,
        conversation_context=conversation_context if conversation_context else ""
    )
    if file_context:
        thought_prompt += f"\n\nAdditional File Context:\n{file_context}"
    context = ""
    themes=format_themes_for_prompt_memoized(bot.memory_index,user_id,mode="sections")
    thought_system_prompt = system_prompts['thought_generation'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)
    thought_response = await bot.call_api(
        thought_prompt,
        context=context,
        system_prompt=thought_system_prompt,
        image_paths=image_paths,
        temperature=bot.amygdala_response/100
    )
    thought_response = clean_response(thought_response)
    memory_string = f"Reflections on interactions with @{user_name} ({storage_timestamp}):\n {thought_response}"
    bot.logger.debug(f"Pre-memory addition string: {memory_string}")
    await memory_index.add_memory_async(user_id, memory_string)
    bot.logger.debug(f"Post-memory addition: {memory_index.user_memories[user_id][-1]}")
    log_to_jsonl({
        'event': 'thought_generation',
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'user_name': user_name,
        'memory_text': memory_text,
        'thought_response': thought_response
    }, bot_id=bot.user.name)

    if cleanup_callback:
        cleanup_callback()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = os.path.basename(sanitized)
    return sanitized

class CustomHelpCommand(commands.HelpCommand):
    async def send_bot_help(self, mapping):
        embed = discord.Embed(title=f"🤖 {self.context.bot.user.name} Commands", description="Here are all available commands:", color=discord.Color.blue())
        # Check permissions
        is_manager = False
        if isinstance(self.context.channel, discord.DMChannel):
            for guild in self.context.bot.guilds:
                member = guild.get_member(self.context.author.id)
                if member and (member.guild_permissions.administrator or member.guild_permissions.manage_guild or any(role.name == DISCORD_BOT_MANAGER_ROLE for role in member.roles)):
                    is_manager = True
                    break
        else:
            is_manager = (self.context.author.guild_permissions.administrator or self.context.author.guild_permissions.manage_guild or any(role.name == DISCORD_BOT_MANAGER_ROLE for role in self.context.author.roles))
        if is_manager:
            api_settings = [
                f"**API Type**: {self.context.bot.api.api_type}",
                f"**Model**: {self.context.bot.api.model_name}",
                f"**Amygdala Response**: {self.context.bot.amygdala_response}%"]
            embed.add_field(name="🔧 Current Settings", value="\n".join(api_settings), inline=False)
            embed.add_field(name="📚 GitHub Integration " + ("✅" if getattr(self.context.bot, 'github_enabled', False) else "❌"), value="", inline=False)
            embed.add_field(name="🧠 DMN Processor " + ("✅" if self.context.bot.dmn_processor.enabled else "❌"), value="", inline=False)
            embed.add_field(name="⚡ Processing " + ("✅" if getattr(self.context.bot, 'processing_enabled', True) else "❌"), value="", inline=False)
            embed.add_field(name="🔗 Mentions " + ("✅" if getattr(self.context.bot, 'mentions_enabled', True) else "❌"), value="", inline=False)
            embed.add_field(name="👁️ Attention " + ("✅" if getattr(self.context.bot, 'attention_enabled', True) else "❌"), value="", inline=False)
        for cog, commands in mapping.items():
            filtered = await self.filter_commands(commands, sort=True)
            if filtered:
                category = "General" if cog is None else cog.qualified_name
                command_list = []
                for cmd in filtered:
                    if not cmd.hidden or is_manager:
                        brief = cmd.help.split('\n')[0] if cmd.help else "No description"
                        if len(brief) > 60:
                            brief = brief[:57] + "..."
                        command_list.append(f"`!{cmd.name}` - {brief}")
                if command_list:
                    embed.add_field(name=f"📑 {category}", value="\n".join(command_list), inline=False)
        await self.get_destination().send(embed=embed)
    async def send_command_help(self, command):
        """Handles help for a specific command."""
        embed = discord.Embed(title=f"Command: {command.name}", description=command.help or "No description available.", color=discord.Color.green())
        signature = self.get_command_signature(command)
        embed.add_field(name="Usage", value=f"```{signature}```", inline=False)
        if command.aliases:
            embed.add_field(name="Aliases", value=", ".join(f"`{alias}`" for alias in command.aliases),  inline=False)
        if command.checks:
            checks = []
            for check in command.checks:
                check_name = check.__qualname__.split('.')[0]
                if 'has_guild_permissions' in check_name:
                    checks.append("Requires server management permissions")
                elif 'is_owner' in check_name:
                    checks.append("Bot owner only")
                else:
                    checks.append(check_name)           
            if checks:
                embed.add_field(name="Requirements", value="\n".join(f"• {check}" for check in checks), inline=False)
        await self.get_destination().send(embed=embed)

def load_private_api_client(bot_id: str, args):
    """
    Return an independent copy of api_client (api object + helpers)
    without touching the original module.
    """
    module_name = f"api_client_{bot_id}"
    if module_name in sys.modules:               # already loaded?
        return sys.modules[module_name]          # reuse

    spec = importlib.util.find_spec("api_client")
    if spec is None:
        raise ImportError("Could not locate api_client.py")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[module_name] = mod

    mod.initialize_api_client(args)

    return mod
    
async def initialize_themes_cache(memory_index, logger):
    """Initialize themes cache in background to avoid blocking startup."""
    try:
        themes = await asyncio.to_thread(get_current_themes, memory_index)
        logger.info(f"Startup themes cache loaded with {len(themes)} existing themes")
    except Exception as e:
        logger.error(f"Failed to initialize themes cache: {e}")

def setup_bot(prompt_path=None, bot_id=None):
    """Initialize the Discord bot with specified configuration."""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    help_command = CustomHelpCommand()
    
    bot = commands.Bot(
        command_prefix='!', 
        intents=intents, 
        status=discord.Status.online,
        help_command=help_command
    )
    # Initialize central logger for this bot instance
    bot.logger = BotLogger(bot_id if bot_id else "default")
    # Create base cache directory for this bot
    bot_cache_dir = bot_id if bot_id else "default"
    # Initialize with specific cache types under the bot's directory
    memory_index = UserMemoryIndex(f'{bot_cache_dir}/memory_index', logger=bot.logger)


    repo_index = RepoIndex(f'{bot_cache_dir}/repo_index')
    
    # Create a temp file cache for media shared from Discord 
    files_root = os.path.join('cache', (bot_id if bot_id else 'default'), 'files')
    os.makedirs(files_root, exist_ok=True)
    bot.cache_managers = {'file': CacheManager(files_root)}
    
    # Initialize GitHub repository with validation
    try:
        if bot_id:
            github_token = os.getenv(f'GITHUB_TOKEN_{bot_id.upper()}')
            github_repo_name = os.getenv(f'GITHUB_REPO_{bot_id.upper()}')
            bot.logger.info(f"Attempting GitHub init with token env: GITHUB_TOKEN_{bot_id.upper()}")
        else:
            github_token = os.getenv('GITHUB_TOKEN')
            github_repo_name = os.getenv('GITHUB_REPO')
            bot.logger.info("Attempting GitHub init with default token env")

        if not github_token or not github_repo_name:
            bot.github_enabled = False
            github_repo = None
            bot.logger.warning(f"GitHub credentials not found for bot {bot_id or 'default'}. Required env vars: " + 
                          f"GITHUB_TOKEN_{bot_id.upper() if bot_id else ''}, " +
                          f"GITHUB_REPO_{bot_id.upper() if bot_id else ''}")
        else:
            github_repo = GitHubRepo(github_token, github_repo_name)
            github_repo.repo.get_contents('/')
            bot.github_enabled = True
            bot.github_repo = github_repo
            bot.logger.info(f"GitHub integration enabled for repository: {github_repo_name}")
    except Exception as e:
        bot.github_enabled = False
        github_repo = None
        bot.logger.warning(f"GitHub initialization failed for bot {bot_id or 'default'}: {str(e)}. GitHub features will be disabled.")

    try:
        with open(os.path.join(prompt_path, 'prompt_formats.yaml'), 'r', encoding='utf-8') as file:
            prompt_formats = yaml.safe_load(file)
        
        with open(os.path.join(prompt_path, 'system_prompts.yaml'), 'r', encoding='utf-8') as file:
            system_prompts = yaml.safe_load(file)
    except Exception as e:
        bot.logger.error(f"Error loading prompt files from {prompt_path}: {str(e)}")
        raise

    bot.amygdala_response = DEFAULT_AMYGDALA_RESPONSE

    bot.memory_index = memory_index
    bot.prompt_formats = prompt_formats
    bot.system_prompts = system_prompts
    bot.temporal_parser = TemporalParser()
    bot.processing_enabled = True
    bot.mentions_enabled = False
    bot.attention_enabled = True

    @bot.event
    async def on_ready():
        bot.logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')

        bot.dmn_processor.logger = BotLogger(bot.user.name)
        bot.loop.create_task(bot.dmn_processor.start())
        bot.logger.info('DMN processor started')
        
        log_to_jsonl({
            'event': 'bot_ready',
            'timestamp': datetime.now().isoformat(),
            'bot_name': bot.user.name,
            'bot_id': bot.user.id
        }, bot_id=bot.user.name)

    @bot.event
    async def on_message(message):
        if message.author == bot.user: return
        command_content=None
        if not isinstance(message.channel, discord.DMChannel):
            if message.content.startswith(f'<@{bot.user.id}>') or message.content.startswith(f'<@!{bot.user.id}>'):
                parts=message.content.split(maxsplit=1)
                if len(parts)>1: command_content=parts[1]
        if command_content and command_content.startswith('!'):
            message.content=command_content; await bot.process_commands(message); return
        elif isinstance(message.channel, discord.DMChannel) and message.content.startswith('!'):
            await bot.process_commands(message); return

        uid=str(message.author.id)
        attn=False
        if bot.attention_enabled:
            attn=await asyncio.to_thread(
                check_attention_triggers_fuzzy,
                message.content,
                system_prompts.get('attention_triggers', []),
                memory_index=memory_index,
                user_id=uid
            )

        if isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions or attn:
            has_supported_files=False
            if message.attachments:
                for attachment in message.attachments:
                    ext=os.path.splitext(attachment.filename.lower())[1]
                    if (ext in ALLOWED_EXTENSIONS) or (attachment.content_type and attachment.content_type.startswith('image/') and ext in ALLOWED_IMAGE_EXTENSIONS):
                        has_supported_files=True; break

            if message.attachments and has_supported_files:
                try:
                    await process_files(
                        message=message,
                        memory_index=memory_index,
                        prompt_formats=prompt_formats,
                        system_prompts=system_prompts,
                        user_message=message.content,
                        bot=bot
                    )
                except Exception as e:
                    await message.channel.send(f"Error processing file(s): {str(e)}")
                    bot.logger.error(f"Error during process_files call from on_message: {str(e)}")
                    bot.logger.error(traceback.format_exc())
            else:
                await process_message(message, memory_index, prompt_formats, system_prompts, github_repo, is_command=False)

        
    @bot.command(name='persona')
    @commands.check(lambda ctx: config.discord.has_command_permission('persona', ctx))
    async def set_amygdala_response(ctx, intensity: int = None):
        """Set or get the AI's amygdala arousal (0-100). The intensity can be steered through in context prompts and it also adjusts the temperature of the API calls."""
        if intensity is None:
            await ctx.send(f"Current amygdala arousal is {bot.amygdala_response}%.")
        elif 0 <= intensity <= 100:
            bot.amygdala_response = intensity
            update_temperature(intensity)
            success_msg = f"Amygdala arousal set to {intensity}%"
            if hasattr(bot, 'dmn_processor') and bot.dmn_processor:
                success_msg += ". DMN processor synchronized."
            await ctx.send(success_msg)
        else:
            await ctx.send("Please provide a valid intensity between 0 and 100.")
            bot.logger.warning(f"Invalid amygdala arousal attempted: {intensity}")

    @bot.command(name='attention')
    @commands.check(lambda ctx: config.discord.has_command_permission('attention', ctx))
    async def toggle_attention(ctx, state: str = None):
        """Enable or disable attention trigger responses. Usage: !attention on/off"""
        if state is None:
            status = "enabled" if bot.attention_enabled else "disabled"
            await ctx.send(f"Attention triggers are currently **{status}**")
            bot.logger.info(f"Attention status queried: {status}")
            return
        if state.lower() in ['on', 'enable', 'true', '1']:
            bot.attention_enabled = True
            await ctx.send("✅ Attention triggers **enabled** - I'll respond to topic-based triggers")
        elif state.lower() in ['off', 'disable', 'false', '0']:
            bot.attention_enabled = False
            await ctx.send("❌ Attention triggers **disabled** - I'll only respond to mentions and DMs")
        else:
            await ctx.send("Usage: `!attention on` or `!attention off`")
            bot.logger.warning(f"Invalid attention command attempted: {state}")

    @bot.command(name='add_memory')
    @commands.check(lambda ctx: config.discord.has_command_permission('add_memory', ctx))
    async def add_memory(ctx, *, memory_text):
        """Add a new memory to the AI."""
        memory_index.add_memory(str(ctx.author.id), memory_text)
        await ctx.send("Memory added successfully.")
        log_to_jsonl({
            'event': 'add_memory',
            'timestamp': datetime.now().isoformat(),
            'user_id': str(ctx.author.id),
            'user_name': ctx.author.name,
            'memory_text': memory_text
        }, bot_id=bot.user.name)

    @bot.command(name='clear_memories')
    @commands.check(lambda ctx: config.discord.has_command_permission('clear_memories', ctx))
    async def clear_memories(ctx):
        """Clear all memories of the invoking user."""
        user_id = str(ctx.author.id)
        memory_index.clear_user_memories(user_id)
        await ctx.send("Your memories have been cleared.")
        log_to_jsonl({
            'event': 'clear_user_memories',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': ctx.author.name
        }, bot_id=bot.user.name)

    @bot.command(name='summarize')
    @commands.check(lambda ctx: config.discord.has_command_permission('summarize', ctx))
    async def summarize(ctx, *, args=None):
        """Summarize the last n messages in a specified channel and send the summary to DM."""
        try:
            n = MAX_CONVERSATION_HISTORY
            channel = None
            if args:
                parts = args.split()
                if len(parts) >= 1:
                    if parts[0].startswith('<#') and parts[0].endswith('>'):
                        channel_id = int(parts[0][2:-1])
                    elif parts[0].isdigit():
                        channel_id = int(parts[0])
                    else:
                        await ctx.send("Please provide a valid channel ID or mention.")
                        return
                    
                    channel = bot.get_channel(channel_id)
                    if channel is None:
                        await ctx.send(f"Invalid channel. Channel ID: {channel_id}")
                        return
                    parts = parts[1:]  

                    if parts:
                        try:
                            n = int(parts[0])
                        except ValueError:
                            await ctx.send("Invalid input. Please provide a number for the amount of messages to summarize.")
                            return
            else:
                await ctx.send("Please specify a channel ID or mention to summarize.")
                return
            member = channel.guild.get_member(ctx.author.id)
            if member is None or not channel.permissions_for(member).read_messages:
                await ctx.send(f"You don't have permission to read messages in the specified channel.")
                return
            if not channel.permissions_for(channel.guild.me).read_message_history:
                await ctx.send(f"I don't have permission to read message history in the specified channel.")
                return
            typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
            try:
                summarizer = ChannelSummarizer(bot, prompt_formats, system_prompts, max_entries=n)
                summary = await summarizer.summarize_channel(channel.id)
            finally:
                typing_task.cancel()
            try:
                await send_long_message(ctx.author, f"**Channel Summary for #{channel.name} (Last {n} messages)**\n\n{summary}", bot=bot)
                if isinstance(ctx.channel, discord.DMChannel):
                    await ctx.send(f"I've sent you the summary of #{channel.name}.")
                else:
                    await ctx.send(f"{ctx.author.mention}, I've sent you a DM with the summary of #{channel.name}.")
            except discord.Forbidden:
                await ctx.send("I couldn't send you a DM. Please check your privacy settings and try again.")
            memory_text = f"Summarized {n} messages from #{channel.name}. Summary: {summary}"
            await generate_and_save_thought(
                memory_index=memory_index,
                user_id=str(ctx.author.id),
                user_name=ctx.author.name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot
            )
        except discord.Forbidden as e:
            await ctx.send(f"I don't have permission to perform this action. Error: {str(e)}")
        except Exception as e:
            error_message = f"An error occurred while summarizing the channel: {str(e)}"
            await ctx.send(error_message)
            bot.logger.error(f"Error in channel summarization: {str(e)}")

    @bot.command(name='index_repo')
    @commands.check(lambda ctx: config.discord.has_command_permission('index_repo', ctx))
    async def index_repo(ctx, option: str = None, branch: str = 'main'):
        """Index the GitHub repository contents, list indexed files, or check indexing status."""
        if not bot.github_enabled:
            await ctx.send("GitHub integration is currently disabled. Please check bot logs for details.")
            return
        global repo_processing_event
        if option == 'list':
            if repo_processing_event.is_set():
                indexed_files = set()
                for file_paths in repo_index.repo_index.values():
                    indexed_files.update(file_paths)
                if indexed_files:
                    file_list = f"# Indexed Repository Files (Branch: {branch})\n\n"
                    for file in sorted(indexed_files):
                        file_list += f"- `{file}`\n"
                    temp_path, _ = bot.cache_managers['file'].create_temp_file(
                        user_id=str(ctx.author.id),
                        prefix="repo_index_",
                        suffix=".md", 
                        content=file_list
                    )
                    await ctx.send(f"Here's the list of indexed files from the '{branch}' branch:", file=discord.File(temp_path))
                else:
                    await ctx.send(f"No files have been indexed yet on the '{branch}' branch.")
            else:
                await ctx.send(f"Repository indexing has not been completed for the '{branch}' branch. Please run `!index_repo` first.")
        elif option == 'status':
            if repo_processing_event.is_set():
                await ctx.send("Repository indexing is complete.")
            else:
                await ctx.send("Repository indexing is still in progress.")
        else:
            try:
                repo_processing_event.clear()
                await ctx.send(f"Starting to index the repository on the '{branch}' branch... This may take a while.")
                repo_index.clear_cache()
                start_background_processing_thread(github_repo.repo, repo_index, max_depth=None, branch=branch)
                await ctx.send(f"Repository indexing has started in the background for the '{branch}' branch.")
            except Exception as e:
                error_message = f"An error occurred while starting the repository indexing on the '{branch}' branch: {str(e)}"
                await ctx.send(error_message)
                bot.logger.error(error_message)
                
    @bot.command(name='repo_file_chat')
    @commands.check(lambda ctx: config.discord.has_command_permission('repo_file_chat', ctx))
    async def repo_file_chat(ctx, *, input_text: str = None):
        """Specific file in the GitHub repo chat"""
        if not input_text:
            await ctx.send("Usage: !repo_file_chat <file_path> <task_description>")
            return
        if not bot.github_enabled:
            await ctx.send("GitHub integration is currently disabled. Please check bot logs for details.")
            return
        parts = input_text.split(maxsplit=1)
        if len(parts) < 2:
            await ctx.send("Error: Please provide both a file path and a task description.")
            return
        file_path = parts[0]
        user_task_description = parts[1]
        if not repo_processing_event.is_set():
            await ctx.send("Repository indexing is not complete. Please run !index_repo first.")
            return
        try:
            file_path = file_path.strip().replace('\\', '/')
            if file_path.startswith('/'):
                file_path = file_path[1:]  # Remove leading slash if present
            indexed_files = set()
            for file_set in repo_index.repo_index.values():
                indexed_files.update(file_set)
            if file_path not in indexed_files:
                await ctx.send(f"Error: The file '{file_path}' is not in the indexed repository.")
                return
            response_content = None
            typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
            try:
                repo_code = github_repo.get_file_content(file_path)
                if repo_code.startswith("Error fetching file:"):
                    await ctx.send(f"Error: {repo_code}")
                    return
                elif repo_code == "File is too large to fetch content directly.":
                    await ctx.send(repo_code)
                    return
                _, file_extension = os.path.splitext(file_path)
                code_type = mimetypes.types_map.get            
                if 'repo_file_chat' not in prompt_formats or 'repo_file_chat' not in system_prompts:
                    await ctx.send("Error: Required prompt templates are missing.")
                    return
                # Build context
                context = f"Current discord channel: #{ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'}\n\n"
                context += "Ongoing Chatroom Conversation:\n\n"
                context += "<conversation>\n"
                messages = []
                async for msg in ctx.channel.history(limit=MAX_CONVERSATION_HISTORY):
                    if msg.id != ctx.message.id:  # Skip the command message
                        combined_mentions = list(msg.mentions) + list(msg.channel_mentions)
                        msg_content = sanitize_mentions(msg.content, combined_mentions)
                        truncated_content = truncate_middle(msg_content, max_tokens=TRUNCATION_LENGTH)
                        clean_name = msg.author.name
                        formatted_msg = f" @{clean_name}: {truncated_content}"
                        # Add reactions if present
                        if msg.reactions:
                            reaction_parts = []
                            for reaction in msg.reactions:
                                reaction_emoji = str(reaction.emoji)
                                async for user in reaction.users():
                                    reaction_user_name = user.name
                                    reaction_parts.append(f"@{reaction_user_name}: {reaction_emoji}")
                            if reaction_parts:
                                formatted_msg += f" (Discord Member Reactions: {' '.join(reaction_parts)})"
                        messages.append(formatted_msg)

                for msg in reversed(messages):
                    context += f"{msg}\n"
                context += "</conversation>\n"

                prompt = prompt_formats['repo_file_chat'].format(
                    file_path=file_path,
                    code_type=code_type,
                    repo_code=repo_code,
                    user_task_description=user_task_description,
                    context=context
                )

                #themes = ", ".join(get_current_themes(bot.memory_index))
                themes=format_themes_for_prompt_memoized(bot.memory_index,str(ctx.author.id),mode="sections")
                system_prompt = system_prompts['repo_file_chat'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)
                response_content = await bot.call_api(prompt, system_prompt=system_prompt)
                response_content = clean_response(response_content)
                response_content = balance_wraps(response_content)
            finally:
                typing_task.cancel()
            if response_content:
                formatted_response = f"# Analysis for {file_path}\n\n"
                formatted_response += f"**Task**: {user_task_description}\n\n"
                formatted_response += response_content
                await send_long_message(ctx.channel, formatted_response, bot=bot)
                memory_text = f"Recollection of'{file_path}' discussing '{user_task_description}'.\n {response_content}"
                asyncio.create_task(generate_and_save_thought(
                    memory_index=memory_index,
                    user_id=str(ctx.author.id),
                    user_name=ctx.author.name,
                    memory_text=memory_text,
                    prompt_formats=prompt_formats,
                    system_prompts=system_prompts,
                    bot=bot
                ))
        except Exception as e:
            error_message = f"Error generating file summary and querying AI: {str(e)}"
            await ctx.send(error_message)
            bot.logger.error(error_message)

    @bot.command(name='ask_repo')
    @commands.check(lambda ctx: isinstance(ctx.channel, discord.DMChannel) or ctx.author.guild_permissions.manage_messages)
    async def ask_repo(ctx, *, question: str = None):
        """RAG GitHub repo chat"""
        if not question:
            await ctx.send("Usage: !ask_repo <question>")
            return
        if not bot.github_enabled:
            await ctx.send("GitHub integration is currently disabled. Please check bot logs for details.")
            return
        response = None  
        try:
            if not repo_processing_event.is_set():
                await ctx.send("Repository indexing is not complete. Please wait or run !index_repo first.")
                return
            typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
            try:
                relevant_files = await repo_index.search_repo_async(question)
            finally:
                typing_task.cancel()
            if not relevant_files:
                await ctx.send("No relevant files found in the repository for this question.")
                return
            context = "Relevant files in the repository:\n"
            file_links = []  
            for file_path, score in relevant_files:
                context += f"- {file_path} (Relevance: {score:.2f})\n"
                file_content = github_repo.get_file_content(file_path)
                context += f"Content preview: {truncate_middle(file_content, 1000)}\n\n"
                file_links.append(f"{file_path}")
            context += f"\nCurrent channel: #{ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'}\n\n"
            context += "**Ongoing Chatroom Conversation:**\n\n"
            context += "<conversation>\n"
            messages = []
            async for msg in ctx.channel.history(limit=MAX_CONVERSATION_HISTORY):
                if msg.id != ctx.message.id:  # Skip the question message
                    combined_mentions = list(msg.mentions) + list(msg.channel_mentions)
                    msg_content = sanitize_mentions(msg.content, combined_mentions)
                    truncated_content = truncate_middle(msg_content, max_tokens=TRUNCATION_LENGTH)
                    clean_name = msg.author.name
                    formatted_msg = f" @{clean_name}: {truncated_content}"
                    if msg.reactions:
                        reaction_parts = []
                        for reaction in msg.reactions:
                            reaction_emoji = str(reaction.emoji)
                            async for user in reaction.users():
                                reaction_user_name = user.name
                                reaction_parts.append(f"@{reaction_user_name}: {reaction_emoji}")
                        
                        if reaction_parts:
                            formatted_msg += f"\n(Message Reactions: {' '.join(reaction_parts)})"
                    
                    messages.append(formatted_msg)
            for msg in reversed(messages):
                context += f"{msg}\n"
            context += "</conversation>\n"
            prompt = prompt_formats['ask_repo'].format(
                context=context,
                question=question
            )
            #themes = ", ".join(get_current_themes(bot.memory_index))
            themes=format_themes_for_prompt_memoized(bot.memory_index,str(ctx.author.id),mode="sections")
            system_prompt = system_prompts['ask_repo'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)
            typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
            try:
                response = await bot.call_api(prompt, context=context, system_prompt=system_prompt)
                response = clean_response(response)
            finally:
                typing_task.cancel()
            if response:
                response += "\n\nReferenced Files:\n```md\n" + "\n".join(file_links) + "\n```"
                await send_long_message(ctx, response, bot=bot)
        except Exception as e:
            error_message = f"An error occurred while processing the repo chat: {str(e)}"
            await ctx.send(error_message)
            bot.logger.error(f"Error in repo chat: {str(e)}")
            return
        if response:
            timestamp = currentmoment()
            memory_text = f"({timestamp}) Asked repo question '{question}'. Response: {response}"
            await generate_and_save_thought(
                memory_index=memory_index,
                user_id=str(ctx.author.id),
                user_name=ctx.author.name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot
            )

    @bot.command(name='search_memories')
    @commands.check(lambda ctx: config.discord.has_command_permission('search_memories', ctx))
    async def search_memories(ctx, *, query):
        """Test the memory search function."""
        is_dm = isinstance(ctx.channel, discord.DMChannel)
        user_id = str(ctx.author.id) if is_dm else None
        typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
        try:
            results = await memory_index.search_async(query, user_id=user_id)
        finally:
            typing_task.cancel()
        if not results:
            await ctx.send("No results found.")
            return
        current_chunk = f"Search results for '{query}':\n"
        for memory, score in results:
            truncated_memory = truncate_middle(memory, 800)
            result_line = f"[Relevance: {score:.2f}] {truncated_memory}\n"
            if len(result_line) > 1800:
                result_line = result_line[:1896] + "...\n"
            if len(current_chunk) + len(result_line) > 1800:
                await ctx.send(current_chunk)
                current_chunk = result_line
            else:
                current_chunk += result_line
        if current_chunk:
            await ctx.send(current_chunk)

    @bot.command(name='dmn')
    @commands.check(lambda ctx: config.discord.has_command_permission('dmn', ctx))
    async def dmn_control(ctx, action: str = None):
        """Control the DMN processor. Usage: !dmn <start|stop|status>"""
        if not action:
            await ctx.send("Please specify an action: start, stop, or status")
            return
        action = action.lower()
        if action == "start":
            if not bot.dmn_processor.enabled:
                await bot.dmn_processor.start()
                await ctx.send("DMN processor started. Bot will now generate periodic reflective thoughts.")
            else:
                await ctx.send("DMN processor is already running.")
        elif action == "stop":
            if bot.dmn_processor.enabled:
                await bot.dmn_processor.stop()
                await ctx.send("DMN processor stopped. Bot will no longer generate background thoughts.")
            else:
                await ctx.send("DMN processor is not currently running.")
        elif action == "status":
            status = "running" if bot.dmn_processor.enabled else "stopped"
            await ctx.send(f"DMN processor is currently {status}.")
        else:
            await ctx.send("Invalid action. Please use: start, stop, or status")

    @bot.command(name='kill')
    @commands.check(lambda ctx: config.discord.has_command_permission('kill', ctx))
    async def kill_tasks(ctx):
        """Gracefully terminate API processing while maintaining Discord connection"""
        try:
            bot.processing_enabled = False
            if bot.dmn_processor.enabled:
                await bot.dmn_processor.stop()
            await ctx.send("Processing disabled. Ongoing API calls will complete but no new calls will be initiated.")
            bot.logger.info(f"Kill command initiated by {ctx.author.name} (ID: {ctx.author.id})")
        except Exception as e:
            await ctx.send(f"Error in kill command: {str(e)}")
            bot.logger.error(f"Kill command error: {str(e)}")

    @bot.command(name='resume')
    @commands.check(lambda ctx: config.discord.has_command_permission('resume', ctx))
    async def resume_tasks(ctx):
        """Resume API processing"""
        bot.processing_enabled = True
        await ctx.send("Processing resumed.")
        bot.logger.info(f"Processing resumed by {ctx.author.name} (ID: {ctx.author.id})")

    @bot.command(name='mentions')
    @commands.check(lambda ctx: config.discord.has_command_permission('mentions', ctx))
    async def toggle_mentions(ctx, state: str = None):
        """Toggle or check mention conversion state. Usage: !mentions <on|off|status>"""
        if state is None or state.lower() == 'status':
            await ctx.send(f"Mention conversion is currently {'enabled' if bot.mentions_enabled else 'disabled'}.")
            return
        state = state.lower()
        if state in ('on', 'true', 'enable'):
            bot.mentions_enabled = True
            await ctx.send("Mention conversion enabled - usernames will be converted to mentions.")
        elif state in ('off', 'false', 'disable'):
            bot.mentions_enabled = False
            await ctx.send("Mention conversion disabled - usernames will remain as plain text.")
        else:
            await ctx.send("Invalid state. Use: on/off/status")

    @bot.command(name='get_logs')
    @commands.check(lambda ctx: config.discord.has_command_permission('get_logs', ctx))
    async def get_logs(ctx):
        """Download bot logs (Permissions required)."""
        try:
            log_dir = os.path.join('cache', bot.user.name, 'logs')
            log_path = os.path.join(
                log_dir,
                config.logging.jsonl_pattern.format(bot_id=bot.user.name)
            )
            temp_path = os.path.join(
                log_dir,
                f'temp_{config.logging.jsonl_pattern.format(bot_id=bot.user.name)}'
            )
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                MAX_SIZE = 1 * 1024 * 1024  # 1MB size limit
                with open(log_path, 'r', encoding='utf-8') as source:
                    lines = source.readlines()
                    lines.reverse()
                    size = 0
                    recent_lines = []
                    for line in lines:
                        line_size = len(line.encode('utf-8'))
                        if size + line_size > MAX_SIZE:
                            break
                        recent_lines.append(line)
                        size += line_size
                if recent_lines:
                    with open(temp_path, 'w', encoding='utf-8') as temp:
                        temp.writelines(recent_lines)
                    try:
                        await ctx.author.send(
                            f"Most recent logs ({len(recent_lines)} entries)",
                            file=discord.File(temp_path, filename=f"{bot.user.name}_recent_logs.jsonl")
                        )
                        if not isinstance(ctx.channel, discord.DMChannel):
                            await ctx.send(f"{ctx.author.mention}, I've sent you the logs via DM.")
                    except discord.Forbidden:
                        await ctx.send("I couldn't send you a DM. Please check your privacy settings and try again.")
                    finally:
                        # Cleanup temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    await ctx.send("No logs available within size limit.")
            else:
                await ctx.send("No logs available.")
        except Exception as e:
            bot.logger.error(f"Error retrieving logs: {str(e)}")
            await ctx.send(f"An error occurred while retrieving the logs: {str(e)}")

    @bot.command(name='reranking')
    @commands.check(lambda ctx: config.discord.has_command_permission('reranking', ctx))
    async def toggle_reranking(ctx, setting: str = None):
        """
        Control hippocampus memory reranking.
        
        Usage:
        !reranking - Show current reranking status
        !reranking <on|off> - Enable/disable memory reranking
        """
        if setting is None:
            status = "on" if config.persona.use_hippocampus_reranking else "off"
            await ctx.send(f"**Memory Reranking:** {status}")
            return
        if setting.lower() in ('on', 'true', 'enable'):
            config.persona.use_hippocampus_reranking = True
            await ctx.send("✅ Memory reranking enabled")
        elif setting.lower() in ('off', 'false', 'disable'):
            config.persona.use_hippocampus_reranking = False
            await ctx.send("❌ Memory reranking disabled")
        else:
            await ctx.send("❓ Invalid value. Use: on/off")
    return bot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Discord bot with selected API and model')
    parser.add_argument('--api', choices=['ollama', 'openai', 'anthropic', 'vllm', 'gemini', 'openrouter'], 
                        default='ollama', help='Choose the API to use (default: ollama)')
    parser.add_argument('--model', type=str, 
                        help='Specify the model to use. If not provided, defaults will be used based on the API.')
    parser.add_argument('--prompt-path', type=str, 
                        default='agent/prompts',
                        help='Path to prompt files directory (default: agent/prompts)')
    parser.add_argument('--bot-name', type=str,
                        help='Name of the bot to run (used for token and cache management)')
    parser.add_argument('--dmn-api', choices=['ollama', 'openai', 'anthropic', 'vllm', 'gemini', 'openrouter'], 
                        help='Choose the API to use for DMN processor (default: use main API)')
    parser.add_argument('--dmn-model', type=str,
                        help='Specify the model to use for DMN processor (default: use main model)')

    args = parser.parse_args()
    # Initialize global logger
    logger = BotLogger(args.bot_name if args.bot_name else "default")
    # Get base prompt path and combine with bot name if provided
    base_prompt_path = os.path.abspath(args.prompt_path)
    prompt_path = os.path.join(base_prompt_path, args.bot_name.lower() if args.bot_name else 'default')
    if not os.path.exists(prompt_path):
        logger.critical(f"Prompt path does not exist: {prompt_path}")
        exit(1)
    #logger.info(f"Using prompt path: {prompt_path}")
    # Select appropriate token based on bot name
    if args.bot_name:
        token_env_var = f'DISCORD_TOKEN_{args.bot_name.upper()}'
        TOKEN = os.getenv(token_env_var)
        if not TOKEN:
            logger.critical(f"No token found for bot '{args.bot_name}' (Environment variable: {token_env_var})")
            exit(1)
        logger.info(f"Running as {args.bot_name}")
    else:
        TOKEN = os.getenv('DISCORD_TOKEN')  # Fall back to default token
        if not TOKEN:
            logger.critical("No Discord token found in environment variables")
            exit(1)
    # Create private API client for this bot
    private_api = load_private_api_client(args.bot_name or "default", args)
    # Override DMN config with command line arguments if provided
    if args.dmn_api or args.dmn_model:
        config.dmn.dmn_api_type = args.dmn_api or config.dmn.dmn_api_type
        config.dmn.dmn_model = args.dmn_model or config.dmn.dmn_model
        #logger.info(f"DMN API overridden: {config.dmn.dmn_api_type}, Model: {config.dmn.dmn_model}")
    bot = setup_bot(prompt_path=prompt_path, bot_id=args.bot_name)
    # Attach per-bot API handles
    bot.api = private_api.api
    bot.call_api = private_api.call_api
    bot.update_api_temperature = private_api.update_api_temperature
    bot.update_api_top_p = private_api.update_api_top_p
    # Initialize DMN processor after API client is attached
    bot.dmn_processor = DMNProcessor(
        memory_index=bot.memory_index,
        prompt_formats=bot.prompt_formats,
        system_prompts=bot.system_prompts,
        bot=bot,
        dmn_api_type=config.dmn.dmn_api_type,
        dmn_model=config.dmn.dmn_model
    )
    # Sync initial amygdala arousal
    bot.dmn_processor.set_amygdala_response(bot.amygdala_response)
    # Run the configured bot; discord.py handles reconnect internally
    try:
        bot.run(TOKEN, reconnect=True)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down gracefully...")
    except discord.errors.LoginFailure as e:
        logger.critical(f"Login failed (invalid token): {str(e)}")
    except Exception as e:
        logger.critical(f"Critical error occurred: {str(e)}")
        raise