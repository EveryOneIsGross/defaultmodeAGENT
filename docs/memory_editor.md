# Memory Editor

## Editor Features

- **Visual Interface**: Web-based UI for memory examination
- **Memory Editing**: Update or delete specific memories
- **Memory Search**: Find memories using keyword search
- **Visualization**: Interactive network graph of memory relationships
- **Cache Management**: Load and save memory caches

<div align="center">

![alt text](/docs/assests/memory_editor_screenshot.png)

</div>

## Editor Implementation

The editor is a FastAPI application that operates on the same memory cache structure used by the agent. Key components:

1. **Cache Loading**: Loads agent memory caches
   ```python
   @app.post("/load/{bot_name}")
   async def load_cache(bot_name: str):
       """Load cache for specific bot"""
   ```

2. **Memory Viewing and Search**: Provides search with hybrid scoring
   ```python
   @app.get("/search/")
   async def search_memories(query: str = "", user_id: Optional[str] = None, per_page: Optional[int] = None):
       """Search memories with hybrid TF-IDF and BM25-like weighting"""
   ```

3. **Memory Editing**: Updates memories and rebuilds indices
   ```python
   @app.put("/memory/{memory_id}")
   async def update_memory(memory_id: int, update: MemoryUpdate):
       """Update specific memory"""
   ```

4. **Memory Visualization**: Creates interactive network visualization
   ```python
   @app.get("/visualize/")
   async def visualize_network(query: str = "", user_id: Optional[str] = None):
       """Visualize memory network as simple SVG based on current search results"""
   ```

## Visualization Algorithm

- Force-directed layout with simulated annealing
- Node sizing based on keyword density
- Edge weights from shared keywords
- Interactive node highlighting

## Memory Cache Structure

The editor works with the same cache structure used by the agent:

```python
cache_structure = {
    'memories': List[str],  # All memory texts
    'user_memories': Dict[str, List[int]],  # User ID -> Memory IDs
    'inverted_index': Dict[str, List[int]],  # Word -> Memory IDs
    'metadata': {
        'last_modified': str,
        'version': str,
        'memory_stats': Dict[int, Dict]
    }
}
```

## Interaction with Agent Memory

The Memory Editor and Agent Memory System interact through the shared cache file structure:

1. The agent continuously writes to the memory cache file during operation
2. The editor can load this cache file to provide visualization and management
3. Any edits made through the editor are saved back to the cache file
4. The agent loads the updated cache file when restarted or periodically

This separation of concerns allows:
- Agent to focus on real-time memory utilization
- Editor to provide human oversight and management
- Both systems to operate on the same underlying data structure

## Usage

1. Start the FastAPI server:
```bash
uvicorn memory_editor:app --host localhost --port 8000
```

2. Access the web interface at `http://localhost:8000`
3. Select a bot cache to load
4. View, edit, or delete memories
5. Visualize memory networks
6. Save changes

## Key Features

### Memory Search
The editor provides a sophisticated search interface with:
- Relevance-based ranking of results
- Keyword highlighting
- User-specific filtering
- Pagination controls

### Memory Visualization
The network visualization shows:
- Memory nodes sized by importance
- Connections based on shared keywords
- Interactive highlighting
- Tooltips showing memory content

### Memory Management
The editor allows:
- Deleting obsolete memories
- Updating memory content
- Rebuilding indices
- Cleaning up broken references

## Dependencies

- FastAPI
- Pydantic
- Python 3.7+
- Modern web browser with SVG support 