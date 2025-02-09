import os
from dotenv import load_dotenv
from prettier import ColoredFormatter
import logging

# Force reload of .env file
if os.path.exists('.env'):
    load_dotenv(override=True)

# Configure logging first
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(os.getenv('LOGLEVEL', 'INFO'))

# Bot configuration
TOKEN = os.getenv('DISCORD_TOKEN')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = os.getenv('GITHUB_REPO')

OLLAMA_API_BASE = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')

# Discord configuration
DISCORD_CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')
DISCORD_BOT_MANAGER_ROLE = 'Developer'


# File extensions
ALLOWED_EXTENSIONS = {'.py', '.js', '.html', '.css', '.json', '.md', '.txt'}

# Add allowed image types
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}


# Inverted Index Search configuration
MAX_TOKENS = 1000
CONTEXT_CHUNKS = 4
CHUNK_PERCENTAGE = 10

# Conversation history
MAX_CONVERSATION_HISTORY = 5
TRUNCATION_LENGTH = 256
HARSH_TRUNCATION_LENGTH = 64

# Persona intensity handling
DEFAULT_AMYGDALA_RESPONSE = 70
TEMPERATURE = DEFAULT_AMYGDALA_RESPONSE / 100.0

# Notion configuration
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
CALENDAR_DB_ID = os.getenv('CALENDAR_DB_ID')
PROJECTS_DB_ID = os.getenv('PROJECTS_DB_ID')
TASKS_DB_ID = os.getenv('TASKS_DB_ID')
KANBAN_DB_ID = os.getenv('KANBAN_DB_ID')

# Polling configuration
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', 120))


###############################################
# Added for experimental multi-platform support

# Twitter Configuration
TWITTER_USERNAME = os.getenv('TWITTER_USERNAME')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')

# Twitter Limits
TWITTER_CHAR_LIMIT = 280
TWITTER_MEDIA_LIMIT = 4
TWITTER_GIF_LIMIT = 1
TWITTER_VIDEO_LIMIT = 1
TWITTER_REPLY_DEPTH_LIMIT = 25  # Maximum thread depth

# Twitter Rate Limits
TWITTER_TWEET_RATE_LIMIT = 300  # per 3 hours
TWITTER_DM_RATE_LIMIT = 1000  # per 24 hours

# Tick rate
TICK_RATE = 800
