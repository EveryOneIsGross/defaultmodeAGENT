import json
import sqlite3
from datetime import datetime
import os
from typing import Dict, Any, Set
import logging
from prettier import ColoredFormatter
from collections import defaultdict

class BotLogger:
    """Handles both JSONL and SQLite logging with dynamic table creation."""
    
    DEFAULT_LOG_LEVEL = 'INFO'
    DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    _initialized = False
    _schema_cache = defaultdict(set)
    _config = None
    
    COMMON_FIELDS = {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
        'timestamp': 'TEXT NOT NULL',
        'user_id': 'TEXT',
        'user_name': 'TEXT',
        'channel': 'TEXT',
        'data': 'JSON',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        'event_type': 'TEXT NOT NULL'
    }
    
    TABLE_SCHEMAS = {
        'error': {
            'error_type': 'TEXT',
            'error_message': 'TEXT',
            'stack_trace': 'TEXT'
        },
        'dmn': {
            'thought_type': 'TEXT',
            'seed_memory': 'TEXT',
            'generated_thought': 'TEXT'
        },
        'memory': {
            'memory_id': 'TEXT',
            'memory_text': 'TEXT',
            'operation': 'TEXT'
        }
    }
    
    SQL_TYPE_MAP = {
        str: 'TEXT',
        int: 'INTEGER',
        float: 'REAL',
        bool: 'INTEGER',
        dict: 'JSON',
        list: 'JSON',
        type(None): 'TEXT'
    }

    @classmethod
    def setup_global_logging(cls, level: str = None, format: str = None):
        """Initialize global logging configuration.
        
        Args:
            level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format (str): Log format string
        """
        if cls._initialized:
            return
            
        # Import config here to avoid circular imports
        from bot_config import config
        cls._config = config
            
        level = level or cls._config.logging.log_level
        format = format or cls._config.logging.log_format
        
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers:
                root.removeHandler(handler)
                
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter(format))
        root.addHandler(handler)
        root.setLevel(level)
        cls._initialized = True

    def __new__(cls, bot_id: str = None):
        """Ensure single logger instance per bot."""
        bot_id = bot_id or "default"
        if not hasattr(cls, '_instances'):
            cls._instances = {}
        if bot_id not in cls._instances:
            cls._instances[bot_id] = super(BotLogger, cls).__new__(cls)
        return cls._instances[bot_id]
    
    def __init__(self, bot_id: str = None):
        if hasattr(self, 'initialized'):
            return
            
        self.bot_id = bot_id or "default"
        
        # Ensure log directory exists
        os.makedirs(self._config.logging.base_log_dir, exist_ok=True)
        
        # Build log paths using config patterns
        self.db_path = os.path.join(
            self._config.logging.base_log_dir,
            self._config.logging.db_pattern.format(bot_id=self.bot_id)
        )
        self.jsonl_path = os.path.join(
            self._config.logging.base_log_dir,
            self._config.logging.jsonl_pattern.format(bot_id=self.bot_id)
        )
        
        # Create a logger instance for this bot
        self._logger = logging.getLogger(f'bot.{self.bot_id}')
        self._logger.propagate = False
        
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(ColoredFormatter(self._config.logging.log_format))
            self._logger.addHandler(handler)
            self._logger.setLevel(self._config.logging.log_level)
        
        self._init_db()
        self.initialized = True

    def _get_sql_type(self, value: Any) -> str:
        """Determine SQL type from Python value."""
        return self.SQL_TYPE_MAP.get(type(value), 'TEXT')

    def _create_table(self, table_name: str, fields: Dict[str, Any]):
        """Dynamically create table with given fields."""
        # Start with common fields
        columns = {k: v for k, v in self.COMMON_FIELDS.items()}
        
        # Add specific fields with their SQL types
        for field, value in fields.items():
            if field not in columns:
                columns[field] = self._get_sql_type(value)
        
        # Create table
        columns_sql = ', '.join(f'{name} {sql_type}' for name, sql_type in columns.items())
        create_table_sql = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                {columns_sql}
            )
        '''
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(create_table_sql)
            
            # Create indexes
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp)')
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_type ON {table_name}(event_type)')
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_user ON {table_name}(user_id)')
            
        # Update schema cache
        self._schema_cache[table_name].update(columns.keys())

    def _add_columns(self, table_name: str, new_fields: Dict[str, Any]):
        """Add new columns to existing table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get existing columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # Add new columns
            for field, value in new_fields.items():
                if field not in existing_columns and field not in self.COMMON_FIELDS:
                    sql_type = self._get_sql_type(value)
                    try:
                        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {field} {sql_type}")
                        self._schema_cache[table_name].add(field)
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            raise

    def _init_db(self):
        """Initialize SQLite database with schema tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def _get_table_name(self, event_type: str) -> str:
        """Determine table name from event type."""
        if not event_type:
            return 'events'
            
        category = event_type.lower().split('_')[0]
        return f"{category}_events"

    def _ensure_table_exists(self, table_name: str, event_type: str):
        """Ensure table exists with proper schema."""
        category = event_type.lower().split('_')[0]
        
        # Start with common fields
        columns = {k: v for k, v in self.COMMON_FIELDS.items()}
        
        # Add category-specific fields if they exist
        if category in self.TABLE_SCHEMAS:
            columns.update(self.TABLE_SCHEMAS[category])
            
        # Create or update table
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                columns_sql = ', '.join(f'{name} {sql_type}' for name, sql_type in columns.items())
                create_table_sql = f'''
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        {columns_sql}
                    )
                '''
                conn.execute(create_table_sql)
                
                # Create indexes
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp)')
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_type ON {table_name}(event_type)')
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_user ON {table_name}(user_id)')
                
                # Update schema cache
                self._schema_cache[table_name].update(columns.keys())

    def log(self, data: Dict[Any, Any]):
        """Log data to JSONL and dynamically managed SQLite tables."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()

        with open(self.jsonl_path, 'a') as f:
            json.dump(data, f)
            f.write('\n')

        event_type = data.get('event', '').lower()
        table_name = self._get_table_name(event_type)
        
        # Ensure table exists with proper schema
        self._ensure_table_exists(table_name, event_type)
        
        # Prepare all fields
        all_values = {
            'timestamp': data.get('timestamp'),
            'user_id': data.get('user_id'),
            'user_name': data.get('user_name'),
            'channel': data.get('channel'),
            'data': json.dumps(data),
            'event_type': event_type
        }
        
        # Add all other fields from data
        for key, value in data.items():
            if key not in all_values and key not in ['event']:
                all_values[key] = value if not isinstance(value, (dict, list)) else json.dumps(value)

        # Create or update table schema if needed
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                # Table doesn't exist, create it with all fields
                columns = {k: v for k, v in self.COMMON_FIELDS.items()}
                for field, value in all_values.items():
                    if field not in columns:
                        columns[field] = self._get_sql_type(value)
                
                columns_sql = ', '.join(f'{name} {sql_type}' for name, sql_type in columns.items())
                create_table_sql = f'''
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        {columns_sql}
                    )
                '''
                conn.execute(create_table_sql)
                
                # Create indexes
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp)')
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_type ON {table_name}(event_type)')
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_user ON {table_name}(user_id)')
                
                # Update schema cache
                self._schema_cache[table_name].update(columns.keys())
            else:
                # Table exists, check for new columns
                cursor.execute(f"PRAGMA table_info({table_name})")
                existing_columns = {row[1] for row in cursor.fetchall()}
                
                # Add any missing columns
                for field, value in all_values.items():
                    if field not in existing_columns and field not in self.COMMON_FIELDS:
                        sql_type = self._get_sql_type(value)
                        try:
                            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {field} {sql_type}")
                            self._schema_cache[table_name].add(field)
                        except sqlite3.OperationalError as e:
                            if "duplicate column name" not in str(e):
                                raise

            # Insert data
            placeholders = ', '.join(['?' for _ in all_values])
            columns = ', '.join(all_values.keys())
            query = f'''
                INSERT INTO {table_name} ({columns})
                VALUES ({placeholders})
            '''
            conn.execute(query, list(all_values.values()))

    def query_events(self, 
                    table_name: str = None,
                    event_type: str = None, 
                    user_id: str = None,
                    start_time: str = None,
                    end_time: str = None,
                    limit: int = 100) -> list:
        """Query events from specific table or all tables."""
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Get list of all event tables
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_events'")
            available_tables = [row[0] for row in cursor.fetchall()]
            
            tables = [table_name] if table_name and table_name in available_tables else available_tables
            
            for table in tables:
                query = f"SELECT * FROM {table} WHERE 1=1"
                params = []

                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                results.extend([dict(row) for row in cursor.fetchall()])
        
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)[:limit]

    def get_user_stats(self, user_id: str) -> dict:
        """Get statistics for a specific user across all tables."""
        stats = {
            'total_events': 0,
            'unique_events': set(),
            'first_seen': None,
            'last_seen': None,
            'tables': {}
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_events'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as count,
                        COUNT(DISTINCT event_type) as unique_events,
                        MIN(timestamp) as first_seen,
                        MAX(timestamp) as last_seen
                    FROM {table} 
                    WHERE user_id = ?
                ''', (user_id,))
                
                table_stats = dict(cursor.fetchone())
                if table_stats['count'] > 0:
                    stats['tables'][table] = table_stats
                    stats['total_events'] += table_stats['count']
                    stats['unique_events'].update(self._get_unique_events(table, user_id))
                    
                    if not stats['first_seen'] or table_stats['first_seen'] < stats['first_seen']:
                        stats['first_seen'] = table_stats['first_seen']
                    if not stats['last_seen'] or table_stats['last_seen'] > stats['last_seen']:
                        stats['last_seen'] = table_stats['last_seen']
        
        stats['unique_events'] = len(stats['unique_events'])
        return stats

    def _get_unique_events(self, table: str, user_id: str) -> Set[str]:
        """Helper to get unique event types for a user in a table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f'''
                SELECT DISTINCT event_type 
                FROM {table} 
                WHERE user_id = ?
            ''', (user_id,))
            return {row[0] for row in cursor.fetchall()}

    def get_event_stats(self) -> dict:
        """Get overall event statistics across all tables."""
        stats = {
            'total_events': 0,
            'unique_events': set(),
            'unique_users': set(),
            'first_event': None,
            'last_event': None,
            'tables': {}
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_events'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as count,
                        COUNT(DISTINCT event_type) as unique_events,
                        COUNT(DISTINCT user_id) as unique_users,
                        MIN(timestamp) as first_event,
                        MAX(timestamp) as last_event
                    FROM {table}
                ''')
                
                table_stats = dict(cursor.fetchone())
                if table_stats['count'] > 0:
                    stats['tables'][table] = table_stats
                    stats['total_events'] += table_stats['count']
                    stats['unique_events'].update(self._get_all_events(table))
                    stats['unique_users'].update(self._get_all_users(table))
                    
                    if not stats['first_event'] or table_stats['first_event'] < stats['first_event']:
                        stats['first_event'] = table_stats['first_event']
                    if not stats['last_event'] or table_stats['last_event'] > stats['last_event']:
                        stats['last_event'] = table_stats['last_event']
        
        stats['unique_events'] = len(stats['unique_events'])
        stats['unique_users'] = len(stats['unique_users'])
        return stats

    def _get_all_events(self, table: str) -> Set[str]:
        """Helper to get all unique event types in a table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f'SELECT DISTINCT event_type FROM {table}')
            return {row[0] for row in cursor.fetchall()}

    def _get_all_users(self, table: str) -> Set[str]:
        """Helper to get all unique users in a table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f'SELECT DISTINCT user_id FROM {table} WHERE user_id IS NOT NULL')
            return {row[0] for row in cursor.fetchall()}

    def debug(self, message: str):
        """Log debug message."""
        self._logger.debug(message)
        self.log({'event': 'debug', 'message': message, 'level': 'DEBUG'})

    def info(self, message: str):
        """Log info message."""
        self._logger.info(message)
        self.log({'event': 'info', 'message': message, 'level': 'INFO'})

    def warning(self, message: str):
        """Log warning message."""
        self._logger.warning(message)
        self.log({'event': 'warning', 'message': message, 'level': 'WARNING'})

    def error(self, message: str):
        """Log error message."""
        self._logger.error(message)
        self.log({'event': 'error', 'message': message, 'level': 'ERROR'})

    def critical(self, message: str):
        """Log critical message."""
        self._logger.critical(message)
        self.log({'event': 'critical', 'message': message, 'level': 'CRITICAL'}) 