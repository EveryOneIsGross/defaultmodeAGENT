import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Set, Dict
from logger import BotLogger
from datetime import datetime

# Force reload of .env file
if os.path.exists('.env'):
    load_dotenv(override=True)

class LogConfig(BaseModel):
    """Logging configuration and paths"""
    base_log_dir: str = Field(default="logs")  # Simple flat directory
    jsonl_pattern: str = Field(default="bot_log_{bot_id}.jsonl")
    db_pattern: str = Field(default="bot_log_{bot_id}.db")
    log_level: str = Field(default=os.getenv('LOGLEVEL', 'INFO'))
    log_format: str = Field(default='%(asctime)s - %(levelname)s - %(message)s')

class APIConfig(BaseModel):
    """API authentication and endpoint configurations"""
    discord_token: str = Field(default=os.getenv('DISCORD_TOKEN'))
    github_token: str = Field(default=os.getenv('GITHUB_TOKEN'))
    github_repo: str = Field(default=os.getenv('GITHUB_REPO'))
    notion_api_key: str = Field(default=os.getenv('NOTION_API_KEY'))
    ollama_api_base: str = Field(default=os.getenv('OLLAMA_API_BASE', 'http://localhost:11434'))
    ollama_model: str = Field(default=os.getenv('OLLAMA_MODEL'))

class DiscordConfig(BaseModel):
    """Discord-specific configuration"""
    channel_id: str = Field(default=os.getenv('DISCORD_CHANNEL_ID'))
    bot_manager_role: str = Field(default='Developer')

class FileConfig(BaseModel):
    """File handling configuration"""
    allowed_extensions: Set[str] = Field(default={'.py', '.js', '.html', '.css', '.json', '.md', '.txt'})
    allowed_image_extensions: Set[str] = Field(default={'.jpg', '.jpeg', '.png', '.gif', '.bmp'})

class SearchConfig(BaseModel):
    """Search and indexing configuration"""
    max_tokens: int = Field(default=1000)
    context_chunks: int = Field(default=4)
    chunk_percentage: int = Field(default=10)

class ConversationConfig(BaseModel):
    """Conversation handling configuration"""
    max_history: int = Field(default=8)
    truncation_length: int = Field(default=256)
    harsh_truncation_length: int = Field(default=64)

class PersonaConfig(BaseModel):
    """Persona and response configuration"""
    default_amygdala_response: int = Field(default=70)
    temperature: float = Field(default_factory=lambda: 70/100.0)
    hippocampus_bandwidth: float = Field(default=0.6)
    memory_capacity: int = Field(default=16)

class NotionConfig(BaseModel):
    """Notion database configuration"""
    calendar_db_id: str = Field(default=os.getenv('CALENDAR_DB_ID'))
    projects_db_id: str = Field(default=os.getenv('PROJECTS_DB_ID'))
    tasks_db_id: str = Field(default=os.getenv('TASKS_DB_ID'))
    kanban_db_id: str = Field(default=os.getenv('KANBAN_DB_ID'))

class TwitterConfig(BaseModel):
    """Twitter API and limits configuration"""
    username: str = Field(default=os.getenv('TWITTER_USERNAME'))
    api_key: str = Field(default=os.getenv('TWITTER_API_KEY'))
    api_secret: str = Field(default=os.getenv('TWITTER_API_SECRET'))
    access_token: str = Field(default=os.getenv('TWITTER_ACCESS_TOKEN'))
    access_secret: str = Field(default=os.getenv('TWITTER_ACCESS_SECRET'))
    bearer_token: str = Field(default=os.getenv('TWITTER_BEARER_TOKEN'))
    char_limit: int = Field(default=280)
    media_limit: int = Field(default=4)
    gif_limit: int = Field(default=1)
    video_limit: int = Field(default=1)
    reply_depth_limit: int = Field(default=25)
    tweet_rate_limit: int = Field(default=300)
    dm_rate_limit: int = Field(default=1000)

class SystemConfig(BaseModel):
    """System-wide configuration"""
    poll_interval: int = Field(default=int(os.getenv('POLL_INTERVAL', 120)))
    tick_rate: int = Field(default=800)

class BotConfig(BaseModel):
    """Main configuration container"""
    api: APIConfig = Field(default_factory=APIConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    files: FileConfig = Field(default_factory=FileConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    notion: NotionConfig = Field(default_factory=NotionConfig)
    twitter: TwitterConfig = Field(default_factory=TwitterConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LogConfig = Field(default_factory=LogConfig)

# Create global config instance
config = BotConfig()

def init_logging():
    """Initialize global logging after config is fully loaded."""
    BotLogger.setup_global_logging()
