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

---

● The Memory Graph, High Level                                                                                           
        The underlying data structure is an inverted index: term → [memory_id, memory_id, ...]. There is no explicit graph       object — the graph is implicit. Two memories are "connected" if they share at least one term entry. Terms are the    edges.                                                                                                                                                                                                                                          
  ---
  The Pruning + Generation Cycle

  Each DMN tick runs a four-phase process on a randomly selected seed memory:

  1. Neighborhood discovery
  Search the inverted index for memories sharing terms with the seed. The result is the seed's local neighborhood — its  
  connected subgraph. If the neighborhood is empty, that's an orphan → spike path.

  2. Term pruning (edge removal)
  For each high-similarity neighbor, the DMN computes:
  overlapping_terms = seed_terms ∩ neighbor_terms
  Those shared terms are removed from the neighbor's index entries — not from the seed. The seed keeps them. The neighbor
   loses the shared vocabulary.

  This is the core specialization pressure: every DMN cycle, neighbors are pushed away from the seed conceptually. Their 
  remaining terms become more distinctive. If a neighbor loses all its terms this way across enough cycles, it becomes a 
  disconnected node and gets cleaned up.

  Weight decay compounds this — the more terms a memory loses relative to its original term count, the lower its
  selection weight. It becomes progressively less likely to be picked as a future seed.

  3. Generative reflection (new node synthesis)
  The LLM is given the seed + neighborhood and generates a new thought. This reflection is stored as a new memory node   
  and indexed into the inverted index with its own fresh terms.

  This new node is the bridge. It uses language that synthesizes concepts from the neighborhood, so it naturally picks up
   terms from multiple sources. It may immediately connect to clusters that were previously isolated from each other,    
  just by virtue of containing their vocabulary.

  4. Cleanup (dead node removal)
  After generation, any memory with zero inverted index entries is permanently deleted. The graph compacts.

  ---
  The Emergent Graph Behavior

  The system never explicitly manages a graph — but one emerges:

  Dense cluster of similar memories
    → repeated DMN cycles prune shared terms between them
    → memories specialize: each retains only distinctive vocabulary
    → weakest ones lose all terms → die
    → new reflection nodes synthesize the cluster into higher-level concepts
    → those reflections may bridge to distant clusters via shared vocabulary

  Over time the graph evolves from a flat collection of raw interaction memories toward a sparser, more abstract
  structure dominated by synthetic reflection nodes. The raw memories that survive are the ones with enough distinctive, 
  non-overlapping vocabulary to resist pruning.

  The density-based temperature feeds into this directly: a sparse neighborhood (few related memories) → high amygdala   
  intensity → hotter, more speculative generation → new reflection nodes with less predictable vocabulary → potentially  
  bridging to very distant parts of the graph.

  Spike fits into this as the boundary condition: orphan nodes (functionally isolated — too few shared terms to find     
  neighbors above threshold) get thrown at real conversation. If the spike fires, two new nodes are created — the spike  
  interaction memory and its reflection — both of which may re-enter the graph with terms that reconnect the orphan's    
  concepts. The orphan's own weight has been decayed, so even if it survives, it competes poorly for future selection.   

  ---
  What "Emergent" Means Here

  No single cycle is directed. But across many cycles the pressures compose:

  ┌──────────────────────┬───────────────────────────────────────────────────┐
  │       Pressure       │                      Effect                       │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Term pruning         │ Memories specialize, cluster edges thin           │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Weight decay         │ Low-connectivity memories deprioritized           │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Generative synthesis │ New abstract nodes bridge distant clusters        │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Cleanup              │ True dead nodes removed, graph stays compact      │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Spike                │ Isolated nodes tested against live context        │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Density temperature  │ Sparse graphs → creative generation → new bridges │
  └──────────────────────┴───────────────────────────────────────────────────┘

  The graph is continuously rewiring itself — not toward any fixed structure, but toward whatever clustering pattern     
  emerges from the intersection of conversation content, random seed walks, and LLM synthesis.