"""
This is the start of the abstractions I have defined in the discord_bot flow, this is the path to buidling out other platforms as the lattice for these wee dreamers. 

Maybe this will become a pydantic model called 'bones' and other platforms shape can be defined in discord_flesh, x_flesh, irc_flesh, etc.
"""


class ChatInterface:
    # Core message handling
    async def send_message(self, channel_id, content):
        """Send a simple text message to a channel"""
        pass
        
    async def send_long_message(self, channel_id, content, max_length=1800):
        """Split and send a long message while preserving formatting"""
        pass
        
    async def send_direct_message(self, user_id, content):
        """Send a private message to a user"""
        pass
        
    async def send_file(self, channel_id, file_path, message=""):
        """Send a file with optional message"""
        pass
        
    async def edit_message(self, channel_id, message_id, new_content):
        """Edit an existing message"""
        pass
        
    # Message retrieval
    async def fetch_message_history(self, channel_id, limit=100):
        """Get recent messages from a channel"""
        pass
        
    async def fetch_message(self, channel_id, message_id):
        """Get a specific message by ID"""
        pass
        
    # Typing indicators
    async def start_typing(self, channel_id):
        """Show typing indicator in channel"""
        pass
        
    async def stop_typing(self, channel_id):
        """Stop typing indicator in channel"""
        pass
        
    # Rich content
    async def send_embed(self, channel_id, title, description, fields=None, color=None):
        """Send a rich formatted message"""
        pass
        
    # User and member information
    def get_user_name(self, user_id):
        """Get username for a user ID"""
        pass
        
    def get_user_display_name(self, user_id, guild_id=None):
        """Get display name (nickname if available)"""
        pass
        
    def get_user_id_from_name(self, name, guild_id=None):
        """Look up user ID from name"""
        pass
        
    def is_bot_mentioned(self, message_content):
        """Check if the bot is mentioned in a message"""
        pass
        
    # Attachment handling
    async def download_attachment(self, attachment_info):
        """Download a file attachment"""
        pass
        
    def get_attachment_info(self, message):
        """Get attachment metadata (name, type, size)"""
        pass
        
    # Channel and guild information
    def get_channel_name(self, channel_id):
        """Get channel name from ID"""
        pass
        
    def is_dm_channel(self, channel_id):
        """Check if channel is a direct message"""
        pass
        
    def get_channel_by_name(self, guild_id, channel_name):
        """Get channel ID from name"""
        pass
        
    # Permission and role handling
    def user_has_permission(self, user_id, guild_id, permission_name):
        """Check if user has a specific permission"""
        pass
        
    def user_has_role(self, user_id, guild_id, role_name):
        """Check if user has a specific role"""
        pass
        
    # Mention formatting
    def format_mentions(self, content, guild_id=None):
        """Convert @username to proper mention format"""
        pass
        
    def sanitize_mentions(self, content):
        """Convert mentions to plain text"""
        pass
        
    # Command system
    async def register_command(self, command_name, handler_function, help_text=""):
        """Register a new command"""
        pass
        
    async def parse_command(self, message_content):
        """Parse a message for commands"""
        pass
        
    # Bot lifecycle
    async def start_bot(self, token=None):
        """Start the bot or terminal interface"""
        pass
        
    async def stop_bot(self):
        """Gracefully shut down"""
        pass
        
    def set_status(self, status_text, status_type="online"):
        """Set bot's status/presence"""
        pass
        
    # Error handling
    def is_permission_error(self, error):
        """Check if error is permission-related"""
        pass
        
    def is_rate_limit_error(self, error):
        """Check if error is rate-limit related"""
        pass
        
    async def handle_rate_limit(self, error, retry_function, *args, **kwargs):
        """Implement rate limit retry logic"""
        pass