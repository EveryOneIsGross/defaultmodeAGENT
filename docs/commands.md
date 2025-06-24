# Discord Bot Commands

## Command Permissions

| Command | Description | Admin | Ally | General |
|---------|-------------|-------|------|---------|
| `!summarize` | Summarize the last n messages in a specified channel and send to DM | ✓ | ✓ | ✓ |
| `!ask_repo` | RAG GitHub repo chat | ✓ | ✓ | ✓ |
| `!repo_file_chat` | Chat about a specific file in a GitHub repository | ✓ | ✓ | ✓ |
| `!analyze_file` | Analyze an uploaded file | ✓ | ✓ | ✓ |
| `!add_memory` | Add a memory to the bot's memory system | ✓ | ✓ | ✗ |
| `!index_repo` | Index GitHub repository contents, list files, or check status | ✓ | ✓ | ✗ |
| `!reranking` | Control hippocampus memory reranking | ✓ | ✓ | ✗ |
| `!clear_memories` | Clear all stored memories for the user | ✓ | ✓ | ✗ |
| `!attention` | Enable/disable attention trigger responses | ✓ | ✓ | ✗ |
| `!kill` | Gracefully terminate API processing | ✓ | ✗ | ✗ |
| `!resume` | Resume API processing | ✓ | ✗ | ✗ |
| `!get_logs` | Download bot logs | ✓ | ✗ | ✗ |
| `!dmn` | Control DMN processor | ✓ | ✗ | ✗ |
| `!mentions` | Toggle mention conversion state | ✓ | ✗ | ✗ |
| `!persona` | Set or get AI's amygdala arousal | ✓ | ✗ | ✗ |
| `!search_memories` | Search through stored memories | ✓ | ✗ | ✗ |

## Command Details

### General Commands (Available to All Users)

#### `!summarize <channel_id> [count]`
- **Description**: Summarize the last n messages in a specified channel and send the summary to DM
- **Usage**: 
  - `!summarize #general` - Summarize recent messages in #general
  - `!summarize 123456789 50` - Summarize last 50 messages in channel ID 123456789
- **Note**: Requires read permissions in the target channel

#### `!ask_repo <question>`
- **Description**: RAG GitHub repo chat - ask questions about the indexed repository
- **Usage**: `!ask_repo How does the authentication system work?`
- **Note**: Repository must be indexed first with `!index_repo`

#### `!repo_file_chat <file_path> <description>`
- **Description**: Chat about a specific file in the GitHub repository
- **Usage**: `!repo_file_chat src/main.py explain the main function`
- **Note**: Repository must be indexed and file must exist

#### `!analyze_file`
- **Description**: Analyze an uploaded file (supports images, text files, code files)
- **Usage**: Upload a file and use `!analyze_file` in the same message
- **Note**: Files over 1MB will be resized if they are images

### Management Commands (Admin + Ally Role)

#### `!add_memory <memory_text>`
- **Description**: Add a new memory to the AI's memory system
- **Usage**: `!add_memory The user prefers technical explanations`

#### `!index_repo [option] [branch]`
- **Description**: Index GitHub repository contents, list indexed files, or check indexing status
- **Usage**:
  - `!index_repo` - Start indexing repository
  - `!index_repo list` - List indexed files
  - `!index_repo status` - Check indexing status
  - `!index_repo list dev` - List files from dev branch

#### `!reranking [setting]`
- **Description**: Control hippocampus memory reranking system
- **Usage**:
  - `!reranking` - Show current status
  - `!reranking on` - Enable memory reranking
  - `!reranking off` - Disable memory reranking

#### `!clear_memories`
- **Description**: Clear all memories associated with the invoking user
- **Usage**: `!clear_memories`

#### `!attention [state]`
- **Description**: Enable or disable attention trigger responses
- **Usage**:
  - `!attention` - Check current status
  - `!attention on` - Enable attention triggers
  - `!attention off` - Disable attention triggers
- **Note**: When enabled, bot responds to topic-based triggers in addition to mentions and DMs

### System Commands (Admin Only)

#### `!kill`
- **Description**: Gracefully terminate API processing while maintaining Discord connection
- **Usage**: `!kill`

#### `!resume`
- **Description**: Resume API processing after being killed
- **Usage**: `!resume`

#### `!get_logs`
- **Description**: Download bot logs (recent entries up to 1MB)
- **Usage**: `!get_logs`

#### `!dmn [action]`
- **Description**: Control the DMN (Default Mode Network) processor for background thought generation
- **Usage**:
  - `!dmn status` - Check DMN status
  - `!dmn start` - Start DMN processor
  - `!dmn stop` - Stop DMN processor

#### `!mentions [state]`
- **Description**: Toggle or check mention conversion state (whether @usernames are converted to Discord mentions)
- **Usage**:
  - `!mentions` - Check current status
  - `!mentions on` - Enable mention conversion
  - `!mentions off` - Disable mention conversion

#### `!persona [intensity]`
- **Description**: Set or get the AI's amygdala arousal level (0-100). This affects emotional intensity and API temperature
- **Usage**: 
  - `!persona` - Get current arousal level
  - `!persona 75` - Set arousal to 75%

#### `!search_memories <query>`
- **Description**: Search through stored memories using semantic search
- **Usage**: `!search_memories python programming`
- **Note**: Also available to users with Manage Messages permission

## Permission Requirements

- **Admin**: Server Administrator or Server Management permission
- **Ally**: Has the 'Ally' role (configurable via `bot_manager_role`)
- **General**: All users

## Notes

- Commands are available in both guild channels and DMs
- DMs respect guild roles and permissions across all mutual guilds
- Higher permission levels inherit access to lower-level commands
- The `bot_manager_role` is set to 'Ally' by default but can be configured
- Some commands require GitHub integration to be enabled
- File analysis supports multiple formats with automatic resizing for large images
- Memory and repository search use semantic similarity for better results
- Two commands (`ask_repo` and `search_memories`) have special permission handling and are also available to users with Manage Messages permission