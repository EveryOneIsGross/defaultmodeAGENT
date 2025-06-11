import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from api_client import call_api
from discord_utils import sanitize_mentions, format_discord_mentions

# Channel summarization
class ChannelSummarizer:
    """A class for summarizing Discord channel messages and threads.

    This class provides functionality to analyze and summarize messages from Discord channels,
    including both main channel messages and thread messages. It tracks participant activity,
    shared file types, and generates content summaries using AI.

    Attributes:
        bot: The Discord bot instance
        max_entries (int): Maximum number of messages to analyze
        prompt_formats (dict): Dictionary of prompt templates
        system_prompts (dict): Dictionary of system prompt templates
    """

    def __init__(self, bot, prompt_formats, system_prompts, max_entries=100):
        """Initialize the ChannelSummarizer.

        Args:
            bot: The Discord bot instance
            prompt_formats (dict): Dictionary of prompt templates
            system_prompts (dict): Dictionary of system prompt templates
            max_entries (int, optional): Maximum messages to analyze. Defaults to 100.
        """
        self.bot = bot
        self.max_entries = max_entries
        self.prompt_formats = prompt_formats
        self.system_prompts = system_prompts

    async def summarize_channel(self, channel_id):
        """Summarize messages from a Discord channel and its threads.

        Analyzes messages from both the main channel and any threads, tracking participant
        activity and generating summaries of the content.

        Args:
            channel_id: ID of the Discord channel to summarize

        Returns:
            str: A formatted summary of the channel activity and content
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return "Channel not found."

        main_messages = []
        threads = defaultdict(list)

        async for message in channel.history(limit=self.max_entries):
            if message.thread:
                threads[message.thread.id].append(message)
            else:
                main_messages.append(message)

        summary = f"Summary of #{channel.name}:\n\n"
        summary += await self._summarize_messages(main_messages, "Main Channel")

        for thread_id, thread_messages in threads.items():
            thread = channel.get_thread(thread_id)
            if thread:
                thread_summary = await self._summarize_messages(thread_messages, f"Thread: {thread.name}")
                summary += f"\n{thread_summary}"

        return summary

    async def _summarize_messages(self, messages, context):
        """Generate a summary of a set of Discord messages.

        Analyzes messages to track participant activity, shared file types,
        and generate a content summary using AI.

        Args:
            messages (list): List of Discord message objects to analyze
            context (str): Context string describing the message source

        Returns:
            str: A formatted summary of the messages
        """
        user_message_counts = defaultdict(int)
        file_types = defaultdict(int)
        content_chunks = []
        
        for message in messages:
            # Use display_name instead of name for better user identification
            user_message_counts[message.author.display_name] += 1
            for attachment in message.attachments:
                file_type = attachment.filename.split('.')[-1].lower()
                file_types[file_type] += 1
            
            # Sanitize the message content to convert mentions to readable names
            sanitized_content = sanitize_mentions(
                message.content,
                message.mentions + message.channel_mentions + message.role_mentions
            )
            
            # Add the sanitized message to chunks with author's display name
            content_chunks.append(f"{message.author.display_name}: {sanitized_content}")

        summary = f"{context}\n"
        summary += "Participants:\n"
        for user, count in user_message_counts.items():
            summary += f"- {user}: {count} messages\n"

        if file_types:
            summary += "\nShared Files:\n"
            for file_type, count in file_types.items():
                summary += f"- {file_type}: {count} files\n"

        content_summary = await self._process_chunks(content_chunks, context)
        summary += f"\nContent Summary:\n{content_summary}\n"

        return summary

    async def _process_chunks(self, chunks, context):
        """Process message chunks through the AI to generate a summary.

        Args:
            chunks (list): List of message content chunks to summarize
            context (str): Context string describing the message source

        Returns:
            str: AI-generated summary of the message content

        Raises:
            Exception: If there is an error calling the AI API
        """
        prompt = self.prompt_formats['summarize_channel'].format(
            context=context,
            content="\n".join(reversed(chunks)) #reversed order of the entries from the channel
        )
        
        system_prompt = self.system_prompts['channel_summarization'].replace('{amygdala_response}', str(self.bot.amygdala_response))
        

        try:
            return await call_api(prompt, context="", system_prompt=system_prompt)
        except Exception as e:
            return f"Error in generating summary: {str(e)}"
