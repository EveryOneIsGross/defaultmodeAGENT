# Discord Bot Commands

## Command Permissions

| Command | Description | Admin | Ally | General |
|---------|-------------|-------|------|---------|
| `!analyze_file` | Analyze a file from GitHub repository | ✓ | ✓ | ✓ |
| `!ask_repo` | Chat about a GitHub repository | ✓ | ✓ | ✓ |
| `!repo_file_chat` | Chat about a specific file in a repository | ✓ | ✓ | ✓ |
| `!summarize` | Generate a summary of the conversation | ✓ | ✓ | ✓ |
| `!add_memory` | Add a memory to the bot's memory system | ✓ | ✓ | ✗ |
| `!clear_memories` | Clear all stored memories | ✓ | ✓ | ✗ |
| `!search_memories` | Search through stored memories | ✓ | ✓ | ✗ |
| `!mentions` | Get mentions of a user | ✓ | ✓ | ✗ |
| `!persona` | Set or get bot persona settings | ✓ | ✗ | ✗ |
| `!dmn` | Process DMN files | ✓ | ✗ | ✗ |
| `!kill` | Stop the bot | ✓ | ✗ | ✗ |
| `!resume` | Resume the bot | ✓ | ✗ | ✗ |
| `!get_logs` | Get bot logs | ✓ | ✗ | ✗ |
| `!reranking` | Disable hippocampus embedding | ✓ | ✗ | ✗ |

## Permission Requirements

- **Admin**: Server Administrator or Server Management permission
- **Ally**: Has the 'Ally' role
- **General**: All users

## Notes

- Commands are available in both guild channels and DMs
- DMs respect guild roles and permissions
- Higher roles inherit permissions from lower roles
- The `bot_manager_role` is set to 'Ally' by default 