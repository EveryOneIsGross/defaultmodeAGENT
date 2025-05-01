import asyncio
import unittest
from unittest.mock import MagicMock, patch
import discord

# Local imports - no need for sys.path manipulation since we're in the same directory
from discord_bot import send_long_message
from discord_utils import strip_role_prefixes, sanitize_mentions, format_discord_mentions

class MockChannel:
    def __init__(self):
        self.sent_messages = []
        self.guild = None
        
    async def send(self, content):
        self.sent_messages.append(content)
        return MagicMock()

class TestMessageWrapping(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.channel = MockChannel()
        self.bot = MagicMock()
        self.bot.mentions_enabled = True
        self.bot.logger = MagicMock()
        
    def assertMessagesValid(self, messages, wrapped_content_intact=True):
        """Verify that messages maintain proper wrapping and don't exceed length"""
        for msg in messages:
            # Check message length
            self.assertLessEqual(len(msg), 1800, f"Message exceeds max length: {msg[:50]}...")
            
            if wrapped_content_intact:
                # Check code block balance
                self.assertEqual(msg.count('```') % 2, 0, 
                               f"Unbalanced code blocks in message: {msg[:50]}...")
                
                # Check tag balance
                opens = msg.count('<')
                closes = msg.count('>')
                self.assertEqual(opens, closes, 
                               f"Unbalanced tags in message: {msg[:50]}...")
    
    async def test_basic_message_splitting(self):
        """Test basic message splitting without any wrapping"""
        long_text = "Test line\n" * 1000
        await send_long_message(self.channel, long_text, max_length=100, bot=self.bot)
        self.assertMessagesValid(self.channel.sent_messages)
        
    async def test_code_block_preservation(self):
        """Test that code blocks are kept intact when possible"""
        code_block = "```python\ndef test_function():\n    print('test')\n```"
        text = f"Some text before\n{code_block}\nSome text after"
        await send_long_message(self.channel, text, max_length=50, bot=self.bot)
        
        found_block = False
        for msg in self.channel.sent_messages:
            if "```python" in msg:
                self.assertIn("```", msg.strip()[-3:], "Code block not properly closed")
                found_block = True
        self.assertTrue(found_block, "Code block not found in any message")
        
    async def test_nested_code_blocks(self):
        """Test handling of nested code blocks and backticks"""
        nested_code = """
        Here's a code block with nested backticks:
        ```python
        def test():
            print('Here is a `quoted` word')
            print("And a ``double quoted`` word")
        ```
        And some more text
        """
        await send_long_message(self.channel, nested_code, max_length=50, bot=self.bot)
        self.assertMessagesValid(self.channel.sent_messages)
        
    async def test_tag_preservation(self):
        """Test that XML-like tags are kept together"""
        tagged_content = """
        <user_info>
        This is some user info
        with multiple lines
        </user_info>
        """
        await send_long_message(self.channel, tagged_content, max_length=50, bot=self.bot)
        
        # Verify tags stayed together
        found_tag = False
        for msg in self.channel.sent_messages:
            if "<user_info>" in msg:
                self.assertIn("</user_info>", msg, "Tag not kept together")
                found_tag = True
        self.assertTrue(found_tag, "Tagged content not found in any message")
        
    async def test_mixed_content(self):
        """Test mixture of code blocks, tags, and normal text"""
        mixed_content = """
        Here's some regular text
        <tag>
        Here's a tag with a code block:
        ```python
        def test():
            print('test')
        ```
        </tag>
        More regular text
        """
        await send_long_message(self.channel, mixed_content, max_length=100, bot=self.bot)
        self.assertMessagesValid(self.channel.sent_messages)
        
    async def test_long_wrapped_content(self):
        """Test handling of wrapped content that exceeds max_length"""
        # Create a code block that's definitely too long
        long_code = "```python\n" + "print('test')\n" * 100 + "```"
        await send_long_message(self.channel, long_code, max_length=100, bot=self.bot)
        
        # Verify each chunk is properly wrapped
        for msg in self.channel.sent_messages:
            if msg.startswith("```"):
                self.assertTrue(msg.endswith("```"), 
                              "Split code block not properly closed")
                
    async def test_edge_cases(self):
        """Test various edge cases"""
        edge_cases = [
            # Empty message
            "",
            # Just a code block marker
            "```",
            # Unclosed tags
            "<test>",
            # Mixed incomplete wrapping
            "```python\n<tag>\nsome code",
            # Multiple types of wrapping
            "```<tag>```",
            # Very long line without spaces
            "x" * 2000,
            # Long line with unicode
            "🌟" * 1000
        ]
        
        for case in edge_cases:
            self.channel.sent_messages = []  # Reset messages
            await send_long_message(self.channel, case, max_length=100, bot=self.bot)
            self.assertMessagesValid(self.channel.sent_messages, wrapped_content_intact=False)
            
    async def test_mention_handling(self):
        """Test that mentions are properly handled"""
        mention_text = "@user1 Here's a message for @user2 with some <wrapped>content</wrapped>"
        
        # Mock the mention formatting
        with patch('discord_utils.format_discord_mentions') as mock_format:
            mock_format.return_value = mention_text.replace("@user", "<@123>")
            await send_long_message(self.channel, mention_text, max_length=50, bot=self.bot)
            
        self.assertMessagesValid(self.channel.sent_messages)

if __name__ == '__main__':
    unittest.main(verbosity=2) 