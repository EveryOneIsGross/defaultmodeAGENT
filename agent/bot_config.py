import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Set, Dict
from logger import BotLogger
from datetime import datetime, timedelta
import discord

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
    enable_jsonl: bool = Field(default=True, description="Enable JSONL file logging")
    enable_sql: bool = Field(default=False, description="Enable SQLite database logging")

class APIConfig(BaseModel):
    """API authentication and endpoint configurations"""
    discord_token: str = Field(default=os.getenv('DISCORD_TOKEN'))
    github_token: str = Field(default=os.getenv('GITHUB_TOKEN'))
    github_repo: str = Field(default=os.getenv('GITHUB_REPO'))
    notion_api_key: str = Field(default=os.getenv('NOTION_API_KEY'))
    ollama_api_base: str = Field(default=os.getenv('OLLAMA_API_BASE', 'http://localhost:11434'))
    ollama_model: str = Field(default=os.getenv('OLLAMA_MODEL'))


class FileConfig(BaseModel):
    """File handling configuration"""
    allowed_extensions: Set[str] = Field(default={'.py', '.js', '.html', '.css', '.json', '.md', '.txt'})
    allowed_image_extensions: Set[str] = Field(default={'.jpg', '.jpeg', '.png', '.gif', '.bmp'})
    # add audio extension for voice message module expansion
    allowed_audio_extensions: Set[str] = Field(default={'.mp3', '.wav', '.ogg', '.m4a'})

class SearchConfig(BaseModel):
    """Search and indexing configuration"""
    max_tokens: int = Field(default=8000)
    context_chunks: int = Field(default=4)
    chunk_percentage: int = Field(default=10)

class ConversationConfig(BaseModel):
    """Conversation handling configuration"""
    max_history: int = Field(default=24)
    minimal_history: int = Field(default=6)
    truncation_length: int = Field(default=512)
    harsh_truncation_length: int = Field(default=128)
    web_content_truncation_length: int = Field(default=8000)

class PersonaConfig(BaseModel):
    """Persona and response configuration"""
    default_amygdala_response: int = Field(default=70)
    temperature: float = Field(default_factory=lambda: 70/100.0)
    hippocampus_bandwidth: float = Field(default=0.70) 
    memory_capacity: int = Field(default=24)
    use_hippocampus_reranking: bool = Field(default=True)
    reranking_blend_factor: float = Field(default=0.20, description="Weight for blending initial search scores with reranking similarity (0-1)") 
    minimum_reranking_threshold: float = Field(default=0.60, description="Minimum threshold for reranked memories") 
    mood_coefficient: float = Field(default=0.30, description="Coefficient (0-1) that controls how strongly amygdala state lowers or raises the memory-selection threshold")

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

class AttentionConfig(BaseModel):
    """Attention mechanism configuration"""
    threshold: int = Field(default=70, description="Fuzzy match threshold for attention triggers (0-100)")
    default_top_n: int = Field(default=32, description="Default number of top trigrams to extract from memory")
    default_min_occ: int = Field(default=8, description="Minimum occurrence count for trigrams to be considered")
    refresh_interval_hours: int = Field(default=1, description="Hours between trigram cache refreshes")
    cooldown_minutes: float = Field(default=0.30, description="Minutes between attention trigger activations")

    stop_words: Set[str] = Field(default_factory=lambda: {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
        'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'ours',
        'theirs', 'a', 'an', 'some', 'any', 'all', 'each', 'every', 'no',
        'none', 'one', 'two', 'three', 'first', 'second', 'last', 'next',
        'other', 'another', 'more', 'most', 'much', 'many', 'few', 'little',
        'less', 'least', 'only', 'just', 'even', 'also', 'too', 'very',
        'quite', 'rather', 'so', 'such', 'how', 'what', 'when', 'where',
        'why', 'who', 'which', 'whose', 'whom', 'if', 'unless', 'until',
        'while', 'since', 'because', 'as', 'than', 'then', 'now', 'here',
        'there', 'yes', 'no', 'not', 'dont', 'doesnt', 'didnt', 'wont',
        'wouldnt', 'couldnt', 'shouldnt', 'cant', 'isnt', 'arent', 'wasnt',
        'werent', 'hasnt', 'havent', 'hadnt'
    })

    @property
    def refresh_interval(self) -> timedelta:
        """Get refresh interval as timedelta"""
        return timedelta(hours=self.refresh_interval_hours)

    @property
    def cooldown(self) -> timedelta:
        """Get cooldown as timedelta"""
        return timedelta(minutes=self.cooldown_minutes)

class DMNConfig(BaseModel):
    """DMN configuration"""
    tick_rate: int = Field(default=1200, description="Time between thought generations in seconds")
    temperature: float = Field(default=0.7, description="Base creative temperature")
    combination_threshold: float = Field(default=0.2, description="Minimum relevance score for memory combinations")
    decay_rate: float = Field(default=0.1, description="Rate at which used memory weights decrease")
    top_k: int = Field(default=24, description="Top k memories to consider for combination")
    density_multiplier: float = Field(default=2.1, description="Multiplier for density-based temperature scaling")
    fuzzy_overlap_threshold: int = Field(default=80, description="Minimum fuzzy overlap threshold for memory combination")
    fuzzy_search_threshold: int = Field(default=90, description="Minimum fuzzy search threshold for term matching")
    max_memory_length: int = Field(default=64, description="Maximum length of a memory based on truncate_middle function")
    similarity_threshold: float = Field(default=0.5, description="Minimum similarity score for memory relevance")
    top_p_min_clamp: float = Field(default=0.8, description="Minimum clamp value for top_p scaling (0.0-1.0)")
    
    # DMN-specific API settings
    dmn_api_type: str = Field(default=None, description="API type for DMN processor (ollama, openai, anthropic, etc.)")
    dmn_model: str = Field(default=None, description="Model name for DMN processor")
    
    # Mode presets
    modes: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        "forgetful": {
            "combination_threshold": 0.02,
            "similarity_threshold": 0.2,
            "decay_rate": 0.8,
            "top_k": 24,
            "fuzzy_overlap_threshold": 70,
            "fuzzy_search_threshold": 80
        },
        "homeostatic": {
            "combination_threshold": 0.3,
            "similarity_threshold": 0.3,
            "decay_rate": 0.1,
            "top_k": 16,
            "fuzzy_overlap_threshold": 80,
            "fuzzy_search_threshold": 90
        },
        "conservative": {
            "combination_threshold": 0.8,
            "similarity_threshold": 0.4,
            "decay_rate": 0.05,
            "top_k": 8,
            "fuzzy_overlap_threshold": 90,
            "fuzzy_search_threshold": 95
        }
    })

class EmbeddingConfig(BaseModel):
    """Pydantic model for embedding configuration."""
    provider: str = Field( default='ollama', description="Provider for embedding service" )
    model: str = Field( default='all-minilm', description="Specific model for embeddings" )
    api_base: str = Field( default='http://localhost:11434', description="Base URL for Ollama API" )
    max_embed_tokens: int = Field(default=160, description="Maximum tokens per embedding request")
    dimensions: int = Field( default=384, description="Expected embedding dimensions" )

class HippocampusConfig(BaseModel):
    """Pydantic model for Hippocampus configuration - provides vector embeddings for downstream search."""
    embedding_provider: str = Field(default='ollama', description="Provider for embedding service")
    embedding_model: str = Field(default='all-minilm:latest', description="Model to use for embeddings")
    blend_factor: float = Field(default=0.7, description="Weight for blending initial search scores with embedding similarity (0-1)")

class DiscordConfig(BaseModel):
    """Discord-specific configuration"""
    channel_id: str = Field(default=os.getenv('DISCORD_CHANNEL_ID'))
    bot_manager_role: str = Field(default='Ally')
    
    # Command permission groups - properly tiered
    system_commands: Set[str] = Field(default={ 'kill', 'resume', 'get_logs', 'dmn', 'mentions', 'persona', 'search_memories' })
    management_commands: Set[str] = Field(default={ 'add_memory', 'index_repo', 'reranking', 'clear_memories', 'attention' })
    general_commands: Set[str] = Field(default={ 'summarize', 'ask_repo', 'repo_file_chat', 'analyze_file' })

    def has_command_permission(self, command_name: str, ctx) -> bool:
        """Check if user has permission to use a command.
        
        Args:
            command_name: Name of the command being checked
            ctx: Discord command context
            
        Returns:
            bool: True if user has permission to use command
        """
        # Check if command exists in any permission group
        if command_name not in (
            self.system_commands | 
            self.management_commands | 
            self.general_commands
        ):
            return False
        # General commands are always allowed
        if command_name in self.general_commands:
            return True
        # For DM channels, check permissions across all mutual guilds
        if isinstance(ctx.channel, discord.DMChannel):
            has_admin = False
            has_ally = False
            for guild in ctx.bot.guilds:
                member = guild.get_member(ctx.author.id)
                if not member:
                    continue
                # Check admin permissions
                if (member.guild_permissions.administrator or 
                    member.guild_permissions.manage_guild):
                    has_admin = True
                    break
                # Check ally role
                if any(role.name == self.bot_manager_role for role in member.roles):
                    has_ally = True
            # System commands require admin permissions
            if command_name in self.system_commands:
                return has_admin
            # Management commands require either admin or ally role
            if command_name in self.management_commands:
                return has_admin or has_ally
            return False

        # For guild channels, check current guild permissions
        if (ctx.author.guild_permissions.administrator or 
            ctx.author.guild_permissions.manage_guild):
            return True
        # Check ally role for management commands
        if (command_name in self.management_commands and
            any(role.name == self.bot_manager_role for role in ctx.author.roles)):
            return True
        return False
    
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
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    dmn: DMNConfig = Field(default_factory=DMNConfig)

# Create global config instance
config = BotConfig()

def init_logging():
    """Initialize global logging after config is fully loaded."""
    BotLogger.setup_global_logging()
