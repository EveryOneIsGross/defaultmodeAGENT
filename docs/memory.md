# "Memory"

## Core Structure

Memory is organized using three key data structures:

```python
cache_structure = {
    'memories': List[str],  # All memory texts stored sequentially
    'user_memories': Dict[str, List[int]],  # User ID -> Memory IDs mapping
    'inverted_index': Dict[str, List[int]],  # Word -> Memory IDs for search
    'metadata': {
        'last_modified': str,
        'version': str,
        'memory_stats': Dict[int, Dict]
    }
}
```

## Core Features

- **Memory Storage**: Memories are saved with user attribution
- **Memory Indexing**: Text is processed to create an inverted index for fast retrieval
- **Efficient Search**: Hybrid TF-IDF and BM25-like scoring for relevance-based retrieval
- **Deduplication**: N-gram similarity detection avoids redundant memories
- **User Isolation**: Memories can be filtered by user for privacy

## Integration with Agent

The agent uses memories in three key ways:

1. **Contextual Retrieval**: During chat processing, relevant memories are fetched:
   ```python
   candidate_memories = memory_index.search(
       sanitized_content, 
       k=MEMORY_CAPACITY,  
       user_id=(user_id if is_dm else None)
   )
   ```

2. **Memory Formation**: After each interaction, new memories are created and indexed:
   ```python
   memory_text = (
       f"User @{user_name} in #{channel_name} ({timestamp}): "
       f"{sanitize_mentions(sanitized_content, message.mentions)}\n"
       f"@{bot.user.name}: {response_content}"
   )
   memory_index.add_memory(user_id, memory_text)
   ```

3. **Background Reflection**: The agent generates additional thought memories about interactions:
   ```python
   asyncio.create_task(generate_and_store_thought(
       memory_index=memory_index,
       user_id=user_id,
       user_name=user_name,
       memory_text=memory_text,
       prompt_formats=prompt_formats,
       system_prompts=system_prompts,
       bot=bot
   ))
   ```

## Technical Implementation

- **Tokenization**: Text is normalized, cleaned and converted to searchable tokens
- **Disk Persistence**: Memory cache is saved to pickle files for persistence
- **Cache Validation**: Index structure is validated on load with automatic repair options
- **Token-aware**: Respects token limits for context construction
- **Temporal Enrichment**: Timestamps are converted to humanized expressions

## Memory Scoring

The system uses a hybrid scoring approach:
- TF-IDF for keyword extraction and basic relevance
- BM25-like scoring for search results with length normalization
- N-gram similarity for deduplication

## Index Maintenance

- Automatic index rebuilding on updates
- Validation of index structure on load
- Cleanup of orphaned entries and null values
- User memory mapping maintenance

## Usage

```python
# Initialize memory system
memory_index = UserMemoryIndex("bot_name/memory_index")

# Add new memories
memory_index.add_memory(user_id="user123", memory_text="Memory content")

# Retrieve relevant memories
results = memory_index.search(
    query="search terms",
    k=5,  # Number of results
    user_id="user123",  # Optional user filter
    similarity_threshold=0.85  # Deduplication threshold
)

# Clear user memories
memory_index.clear_user_memories(user_id="user123")
```

## Cache Directory Structure 

```bash
cache/
├── bot_name/
│ ├── memory_index/
│ │ ├── memory_cache.pkl
│ ├── temp/
│ │ └── user_specific_temp_files/
```

## Best Practices

1. Regular cache saves after modifications
2. Periodic index validation and cleanup
3. User-specific memory isolation 
4. Token-aware context management
5. Proper error handling and logging

## Dependencies

- Python 3.7+
- Custom tokenizer implementation