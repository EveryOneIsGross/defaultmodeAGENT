from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import pickle
from collections import defaultdict
import os
import glob
import math
import re
import string
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive operations
executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global executor
    executor = ThreadPoolExecutor(max_workers=2)
    logger.info("Started thread pool executor")
    yield
    # Shutdown
    if executor:
        executor.shutdown(wait=True)
        logger.info("Shutdown thread pool executor")

app = FastAPI(title="Memory Cache Editor", lifespan=lifespan)

# Mount static files directory
app.mount("/static", StaticFiles(directory="gui/static"), name="static")

class MemoryUpdate(BaseModel):
    text: str

class FindReplaceRequest(BaseModel):
    find_text: str
    replace_text: str
    case_sensitive: bool = False
    whole_words: bool = False
    user_id: Optional[str] = None

# Global state
CACHE_DIR = "cache"
current_file = None
cache = {
    'memories': [],
    'user_memories': defaultdict(list),
    'inverted_index': defaultdict(list),
    'metadata': {
        'last_modified': None,
        'version': '1.0',
        'memory_stats': defaultdict(dict)
    }
}

def get_available_bots():
    """Get list of available bot caches"""
    if not os.path.exists(CACHE_DIR):
        return []
    
    bots = []
    for bot_dir in os.listdir(CACHE_DIR):
        bot_path = os.path.join(CACHE_DIR, bot_dir)
        if os.path.isdir(bot_path):
            memory_dir = os.path.join(bot_path, "memory_index")
            cache_file = os.path.join(memory_dir, "memory_cache.pkl")
            if os.path.exists(cache_file):
                bots.append(bot_dir)
    
    return sorted(bots)


@app.get("/bots")
async def list_bots():
    """List available bot caches"""
    return {
        "bots": get_available_bots()
    }

@app.post("/load/{bot_name}")
async def load_cache(bot_name: str):
    """Load cache for specific bot"""
    global cache, current_file
    
    cache_path = os.path.join(CACHE_DIR, bot_name, "memory_index", "memory_cache.pkl")
    if not os.path.exists(cache_path):
        raise HTTPException(status_code=404, detail=f"Cache not found for bot: {bot_name}")
        
    try:
        with open(cache_path, 'rb') as f:
            loaded_data = pickle.load(f)
            
        # Validate basic structure
        if not isinstance(loaded_data, dict):
            raise ValueError("Invalid pickle format: root must be dict")
            
        required_keys = ['memories', 'user_memories', 'inverted_index']
        for key in required_keys:
            if key not in loaded_data:
                raise ValueError(f"Missing required key: {key}")

        # Convert dictionaries to defaultdict where needed
        cache = {
            'memories': loaded_data['memories'],
            'user_memories': defaultdict(list, loaded_data['user_memories']),
            'inverted_index': defaultdict(list, loaded_data['inverted_index']),
            'metadata': loaded_data.get('metadata', {
                'last_modified': datetime.now().isoformat(),
                'version': '1.0',
                'memory_stats': defaultdict(dict)
            })
        }
        
        # Ensure metadata has defaultdict
        if 'memory_stats' not in cache['metadata']:
            cache['metadata']['memory_stats'] = defaultdict(dict)
        elif not isinstance(cache['metadata']['memory_stats'], defaultdict):
            cache['metadata']['memory_stats'] = defaultdict(dict, cache['metadata']['memory_stats'])
        
        current_file = cache_path
        
        # Count only active memories
        active_memories = len([m for m in cache['memories'] if m is not None])
        
        return {
            "message": f"Cache loaded successfully for {bot_name}", 
            "stats": {
                "memories": active_memories,
                "users": len(cache['user_memories']),
                "index_terms": len(cache['inverted_index'])
            }
        }
    except Exception as e:
        logger.error(f"Failed to load cache: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/save")
async def save_cache():
    """Save current cache"""
    global current_file
    
    if not cache['memories']:
        raise HTTPException(status_code=400, detail="No cache data loaded")
    if not current_file:
        raise HTTPException(status_code=400, detail="No file loaded")
        
    try:
        # Create a temporary file
        temp_file = f"{current_file}.tmp"
        with open(temp_file, 'wb') as f:
            pickle.dump(cache, f)
            
        # Atomic replace
        os.replace(temp_file, current_file)
        
        # Return current stats with save confirmation
        active_memories = len([m for m in cache['memories'] if m is not None])
        return {
            "message": "Changes saved successfully",
            "stats": {
                "memories": active_memories,
                "users": len(cache['user_memories']),
                "index_terms": len(cache['inverted_index'])
            }
        }
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        logger.error(f"Failed to save cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _tokenize(text: str) -> List[str]:
    """Convert text to searchable tokens"""
    # Clean rogue LLM tokens
    text = text.replace("<|endoftext|>", "").replace("<|im_start|>", "").replace("<|im_end|>", "")
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    # Filter stopwords
    stopwords = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'nor', 'yet', 'so',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
        'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
        'this', 'that', 'these', 'those'
    ])
    words = [word for word in words if word not in stopwords]
    return words

def _rebuild_memory_index(memory_id: int, text: str):
    """Rebuild inverted index for a single memory"""
    # Remove old entries for this memory
    for word, memory_list in cache['inverted_index'].items():
        while memory_id in memory_list:
            memory_list.remove(memory_id)
            
    # Add new entries
    tokens = _tokenize(text)
    for token in tokens:
        cache['inverted_index'][token].append(memory_id)

def _rebuild_entire_index():
    """Rebuild the entire inverted index from scratch"""
    cache['inverted_index'].clear()
    
    for memory_id, memory_text in enumerate(cache['memories']):
        if memory_text is not None:
            tokens = _tokenize(memory_text)
            for token in tokens:
                cache['inverted_index'][token].append(memory_id)
    
    # Clean up empty entries
    empty_words = [word for word, memory_list in cache['inverted_index'].items() if not memory_list]
    for word in empty_words:
        del cache['inverted_index'][word]
    
@app.put("/memory/{memory_id}")
async def update_memory(memory_id: int, update: MemoryUpdate):
    """Update specific memory"""
    if not cache['memories']:
        raise HTTPException(status_code=400, detail="No cache data loaded")
    try:
        if memory_id >= len(cache['memories']):
            raise ValueError("Memory ID not found")
            
        # Update the memory
        cache['memories'][memory_id] = update.text
        
        # Rebuild index for this memory
        _rebuild_memory_index(memory_id, update.text)
        
        # Update metadata
        now = datetime.now().isoformat()
        cache['metadata']['last_modified'] = now
        cache['metadata']['memory_stats'][memory_id] = {
            'last_modified': now,
            'action': 'updated'
        }
        
        return {"message": "Memory updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: int, user_id: Optional[str] = None):
    """Delete specific memory"""
    if not cache['memories']:
        raise HTTPException(status_code=400, detail="No cache data loaded")
    try:
        if memory_id >= len(cache['memories']):
            raise ValueError("Memory ID not found")
            
        # If user_id provided, verify ownership
        if user_id:
            if memory_id not in cache['user_memories'].get(user_id, []):
                raise ValueError("Memory does not belong to specified user")
        
        # Update metadata before deletion
        now = datetime.now().isoformat()
        cache['metadata']['last_modified'] = now
        cache['metadata']['memory_stats'][memory_id] = {
            'last_modified': now,
            'action': 'deleted'
        }
        
        # Remove from inverted index first - safely iterate over a copy
        words_to_delete = []
        for word, memory_list in list(cache['inverted_index'].items()):
            while memory_id in memory_list:
                memory_list.remove(memory_id)
            if not memory_list:
                words_to_delete.append(word)
                
        # Clean up empty entries
        for word in words_to_delete:
            del cache['inverted_index'][word]
        
        # Remove the memory
        cache['memories'][memory_id] = None
        
        # Update user_memories
        for uid in list(cache['user_memories'].keys()):
            if memory_id in cache['user_memories'][uid]:
                cache['user_memories'][uid].remove(memory_id)
            # Clean up empty user entries
            if not cache['user_memories'][uid]:
                del cache['user_memories'][uid]
        
        # Return updated stats with response
        active_memories = len([m for m in cache['memories'] if m is not None])
        return {
            "message": "Memory deleted successfully",
            "stats": {
                "memories": active_memories,
                "users": len(cache['user_memories']),
                "index_terms": len(cache['inverted_index'])
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/search/")
async def search_memories(query: str = "", user_id: Optional[str] = None, page: int = 1, per_page: int = 10):
    """Search memories with hybrid TF-IDF and BM25-like weighting"""
    if not cache['memories']:
        raise HTTPException(status_code=400, detail="No cache data loaded")
    
    # Run search in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        _perform_search_sync, 
        query, user_id, page, per_page
    )
    
    return result

@app.get("/visualize/")
async def visualize_network(query: str = "", user_id: Optional[str] = None, page: int = 1, per_page: int = 10):
    """Visualize memory network as simple SVG based on current search results"""
    if not cache['memories']:
        raise HTTPException(status_code=400, detail="No cache data loaded")
    
    # Run visualization generation in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    svg_content = await loop.run_in_executor(
        executor,
        _generate_visualization_sync,
        query, user_id, page, per_page
    )
    
    return HTMLResponse(svg_content)

@app.get("/stats")
async def get_stats():
    """Get current cache stats"""
    if not cache['memories']:
        raise HTTPException(status_code=400, detail="No cache data loaded")
        
    active_memories = len([m for m in cache['memories'] if m is not None])
    return {
        "memories": active_memories,
        "users": len(cache['user_memories']),
        "index_terms": len(cache['inverted_index'])
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head>
        <title>Memory Cache Editor</title>
        <script src="https://unpkg.com/htmx.org@1.9.10"></script>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <div class="container">
            <!-- Visualization container always visible -->
            <div id="visualization"></div>

            <!-- Search input on its own line -->
            <div style="display: flex; gap: 10px; align-items: center;">
                <select id="bot-select" class="bot-select" onchange="loadSelectedCache()">
                    <option value="">Loading bots...</option>
                </select>
            </div>
            <div id="stats"></div>

            <button onclick="saveChanges()" 
                    class="button save-button">
                Save Changes
            </button>
            <div id="save-status"></div>
                        
            <!--  Find & Replace — compact two-row layout -->
            <div class="find-replace-section"
                style="margin:15px 0;padding:15px;border:2px solid #000;background:rgba(255,156,156,.1);">

            <h3 style="margin:0 0 10px 0;">Find and Replace</h3>

            <!-- Row 1 · options + scope ----------------------------------------->
            <div style="display:flex;flex-wrap:wrap;align-items:center;gap:15px;margin-bottom:10px;">
                <label style="display:flex;align-items:center;gap:6px;">
                Case sensitive
                <input type="checkbox" id="case_sensitive" style="width:auto;margin:0;">
                </label>

                <label style="display:flex;align-items:center;gap:6px;">
                Whole words only
                <input type="checkbox" id="whole_words" style="width:auto;margin:0;">
                </label>

                <span id="replace-scope" style="font-weight:bold;">Scope: All Users</span>
            </div>

            <!-- Row 2 · text fields + button ----------------------------------->
            <div style="display:flex;gap:10px;margin-bottom:10px;">
                <input type="text" id="find_text"    placeholder="Find text…"    style="flex:1 1 0;padding:5px;">
                <input type="text" id="replace_text" placeholder="Replace with…" style="flex:1 1 0;padding:5px;">
                <button onclick="findAndReplace()" class="button" style="white-space:nowrap;">Find &amp; Replace</button>
            </div>

            <div id="replace-status"></div>
            </div>

            
            <div style="margin-bottom: 15px;">
                <input type="text" 
                       id="query_input" 
                       placeholder="Search memories..." 
                       oninput="debounce(handleSearchInput, 800)()">
            </div>
            
            <!-- Controls on separate line -->
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 15px;">
                <select id="user_id_select" 
                        onchange="fetchMemories(); updateReplaceScope()">
                    <option value="">All Users</option>
                </select>
                <button onclick="deleteUser()" 
                        class="button delete-button"
                        id="delete_user_button"
                        style="display: none;">
                    Delete User
                </button>
                <div id="pagination_controls" style="display: flex; gap: 5px; align-items: center;">
                    <input type="number" 
                           id="per_page_input" 
                           placeholder="Per Page" 
                           value="10" 
                           min="1" 
                           style="width: 80px;"
                           onchange="fetchMemories()">
                    <button onclick="changePage(-1)" class="button">&lt;</button>
                    <span id="page_info">Page 1</span>
                    <button onclick="changePage(1)" class="button">&gt;</button>
                </div>
            </div>
            
            <div id="results"></div>
            
            <script>
                let isModified = false;
                let currentPage = 1;

                // Load available bots on startup
                fetch('/bots')
                    .then(response => response.json())
                    .then(data => {
                        const select = document.getElementById('bot-select');
                        if (data.bots.length === 0) {
                            select.innerHTML = '<option value="">No bots found</option>';
                            return;
                        }
                        
                        select.innerHTML = '<option value="">Select a bot...</option>' +
                            data.bots.map(bot => `
                                <option value="${bot}">${bot}</option>
                            `).join('');
                    })
                    .catch(error => {
                        const select = document.getElementById('bot-select');
                        select.innerHTML = '<option value="">Error loading bots</option>';
                    });

                function loadSelectedCache() {
                    const select = document.getElementById('bot-select');
                    const botName = select.value;
                    
                    if (!botName) {
                        return;
                    }
                    
                    loadCache(botName);
                }

                function loadCache(botName) {
                    fetch(`/load/${botName}`, {
                        method: 'POST'
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('stats').innerHTML = `
                            <div class="success">
                                ${data.message}<br>
                                Memories: ${data.stats.memories}<br>
                                Users: ${data.stats.users}<br>
                                Index Terms: ${data.stats.index_terms}
                            </div>`;
                        isModified = false;
                        updateUserIds();
                        fetchMemories();  // Fetch and display memories after loading cache
                    })
                    .catch(error => {
                        document.getElementById('stats').innerHTML = `
                            <div class="error">Error loading cache: ${error}</div>`;
                    });
                }

                // Debounce function to limit API calls
                function debounce(func, wait) {
                    let timeout;
                    return function executedFunction(...args) {
                        const later = () => {
                            clearTimeout(timeout);
                            func(...args);
                        };
                        clearTimeout(timeout);
                        timeout = setTimeout(later, wait);
                    };
                }

                function handleSearchInput() {
                    const query = document.getElementById('query_input').value.trim();
                    fetchMemories();
                }

                function changePage(delta) {
                    currentPage = Math.max(1, currentPage + delta);
                    fetchMemories();
                }

                // Add auto-resize function for textareas
                function autoResize(textarea) {
                    textarea.style.height = 'auto';
                    textarea.style.height = textarea.scrollHeight + 'px';
                }

                // Initialize auto-resize for all textareas
                function initAutoResize() {
                    document.querySelectorAll('.memory-text').forEach(textarea => {
                        autoResize(textarea);
                        textarea.addEventListener('input', () => autoResize(textarea));
                    });
                }

                // Call initAutoResize after loading memories
                function fetchMemories() {
                    const query = document.getElementById('query_input').value.trim();
                    const userId = document.getElementById('user_id_select').value;
                    const perPage = document.getElementById('per_page_input').value || 10;
                    let url = "";

                    // Always use pagination for both search and list views
                    if (query) {
                        // For search results, use the search endpoint with pagination
                        url = `/search/?query=${encodeURIComponent(query)}&page=${currentPage}&per_page=${perPage}` + 
                              (userId ? `&user_id=${encodeURIComponent(userId)}` : '');
                    } else {
                        // For all memories, use the list endpoint with pagination
                        url = `/list/?page=${currentPage}&per_page=${perPage}` + 
                              (userId ? `&user_id=${encodeURIComponent(userId)}` : '');
                    }

                    fetch(url)
                    .then(response => response.json())
                    .then(data => {
                        const resultsDiv = document.getElementById('results');
                        resultsDiv.innerHTML = "";
                       
                        // Update pagination info for both search and list views
                        if (data.pagination) {
                            document.getElementById('page_info').textContent = 
                                `Page ${data.pagination.current_page} of ${data.pagination.total_pages}`;
                        }

                        if(data.memories) {
                            data.memories.forEach(function(memory) {
                                const memoryDiv = document.createElement('div');
                                memoryDiv.className = 'memory-item';
                                const sortedKeywords = Object.entries(memory.keywords || {})
                                    .sort((a, b) => b[1] - a[1])
                                    .slice(0, 10);
                                const keywordsHtml = sortedKeywords.length ? `                                    <div class="keywords">
                                        ${sortedKeywords.map(([word, weight]) => `
                                            <span class="keyword">
                                                ${word}
                                                <span class="weight">${weight}</span>
                                            </span>
                                        `).join('')}
                                    </div>
                                ` : '';
                                memoryDiv.innerHTML = `
                                    <div>
                                        <strong>ID:</strong> ${memory.id} 
                                        <strong>User:</strong> ${memory.user_id || 'N/A'}
                                    </div>
                                    <div class="metadata">
                                        ${memory.last_modified ? `<div>Last Modified: ${new Date(memory.last_modified).toLocaleString()}</div>` : ''}
                                        ${memory.last_action ? `<div>Last Action: ${memory.last_action}</div>` : ''}
                                    </div>
                                    ${keywordsHtml}
                                    <textarea class="memory-text" id="text-${memory.id}" oninput="autoResize(this)">${memory.text || ''}</textarea>
                                    <div class="actions">
                                        <button class="button edit-button" onclick="updateMemory(${memory.id})">Update</button>
                                        <button class="button delete-button" onclick="deleteMemory(${memory.id}, '${memory.user_id || ''}')">Delete</button>
                                    </div>
                                `;
                                resultsDiv.appendChild(memoryDiv);
                            });
                        } else if(Array.isArray(data)) {
                            data.forEach(function(memory) {
                                const memoryDiv = document.createElement('div');
                                memoryDiv.className = 'memory-item';
                                const sortedKeywords = Object.entries(memory.keywords || {})
                                    .sort((a, b) => b[1] - a[1])
                                    .slice(0, 10);
                                const keywordsHtml = sortedKeywords.length ? `                                    <div class="keywords">
                                        ${sortedKeywords.map(([word, weight]) => `
                                            <span class="keyword">
                                                ${word}
                                                <span class="weight">${weight}</span>
                                            </span>
                                        `).join('')}
                                    </div>
                                ` : '';
                                memoryDiv.innerHTML = `
                                    <div>
                                        <strong>ID:</strong> ${memory.id} 
                                        <strong>User:</strong> ${memory.user_id || 'N/A'}
                                    </div>
                                    <div class="metadata">
                                        ${memory.last_modified ? `<div>Last Modified: ${new Date(memory.last_modified).toLocaleString()}</div>` : ''}
                                        ${memory.last_action ? `<div>Last Action: ${memory.last_action}</div>` : ''}
                                    </div>
                                    ${keywordsHtml}
                                    <textarea class="memory-text" id="text-${memory.id}" oninput="autoResize(this)">${memory.text || ''}</textarea>
                                    <div class="actions">
                                        <button class="button edit-button" onclick="updateMemory(${memory.id})">Update</button>
                                        <button class="button delete-button" onclick="deleteMemory(${memory.id}, '${memory.user_id || ''}')">Delete</button>
                                    </div>
                                `;
                                resultsDiv.appendChild(memoryDiv);
                            });
                        }
                        
                        // Initialize auto-resize for all textareas
                        initAutoResize();
                        
                        // Always update visualization with current results
                        updateVisualization();
                    })
                    .catch(error => alert('Error fetching memories: ' + error));
                }

                function updateMemory(id) {
                    const text = document.getElementById(`text-${id}`).value;
                    fetch(`/memory/${id}`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.createElement('div');
                        statusDiv.className = 'success';
                        statusDiv.textContent = data.message;
                        const resultDiv = document.getElementById(`text-${id}`).closest('.memory-item');
                        resultDiv.insertBefore(statusDiv, resultDiv.querySelector('.actions'));
                        setTimeout(() => statusDiv.remove(), 3000);
                        isModified = true;
                    })
                    .catch(error => alert('Error updating memory: ' + error));
                }
                
                function deleteMemory(id, userId) {
                    if (!confirm('Are you sure you want to delete this memory?')) return;
                    
                    const url = userId ? 
                        `/memory/${id}?user_id=${userId}` : 
                        `/memory/${id}`;

                    fetch(url, {
                        method: 'DELETE',
                    })
                    .then(async response => {
                        if (!response.ok) {
                            const errorText = await response.text();
                            throw new Error(errorText);
                        }
                        return response.json();
                    })
                    .then(data => {
                        const element = document.querySelector(`#text-${id}`).closest('.memory-item');
                        element.style.animation = 'fadeOut 0.5s';
                        setTimeout(() => element.remove(), 500);
                        isModified = true;
                        
                        // Update stats display
                        if (data.stats) {
                            document.getElementById('stats').innerHTML = `
                                <div class="success">
                                    ${data.message}<br>
                                    Memories: ${data.stats.memories}<br>
                                    Users: ${data.stats.users}<br>
                                    Index Terms: ${data.stats.index_terms}
                                </div>`;
                        }
                        
                        // Refresh the view to update keyword frequencies
                        const searchInput = document.querySelector('input[name="query"]');
                        if (searchInput) {
                            const event = new Event('keyup');
                            searchInput.dispatchEvent(event);
                        }
                    })
                    .catch(error => {
                        console.error('Delete error:', error);
                        alert('Error deleting memory: ' + error.message);
                    });
                }

                function saveChanges() {
                    if (!isModified) {
                        alert('No changes to save');
                        return;
                    }

                    fetch('/save', {
                        method: 'POST'
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('save-status').innerHTML = `
                            <div class="success">${data.message}</div>`;
                        setTimeout(() => {
                            document.getElementById('save-status').innerHTML = '';
                        }, 3000);
                        
                        // Update stats display
                        if (data.stats) {
                            document.getElementById('stats').innerHTML = `
                                <div class="success">
                                    ${data.message}<br>
                                    Memories: ${data.stats.memories}<br>
                                    Users: ${data.stats.users}<br>
                                    Index Terms: ${data.stats.index_terms}
                                </div>`;
                        }
                        
                        // Refresh the view to update keyword frequencies
                        const searchInput = document.querySelector('input[name="query"]');
                        if (searchInput) {
                            const event = new Event('keyup');
                            searchInput.dispatchEvent(event);
                        }
                        
                        isModified = false;
                    })
                    .catch(error => {
                        document.getElementById('save-status').innerHTML = `
                            <div class="error">Error saving changes: ${error}</div>`;
                    });
                }

                function updateVisualization() {
                    const query = document.getElementById('query_input').value;
                    const userId = document.getElementById('user_id_select').value;
                    const perPage = document.getElementById('per_page_input').value || 10;
                    
                    const params = new URLSearchParams();
                    if (query) params.append('query', query);
                    if (userId) params.append('user_id', userId);
                    params.append('page', currentPage);
                    params.append('per_page', perPage);
                    
                    fetch('/visualize/?' + params.toString())
                        .then(response => response.text())
                        .then(svg => {
                            document.getElementById('visualization').innerHTML = svg;
                        })
                        .catch(error => console.error('Error updating visualization:', error));
                }
                
                function highlightMemory(id) {
                    // Find the memory item in the results
                    const memoryItem = document.querySelector(`#text-${id}`);
                    if (memoryItem) {
                        memoryItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        memoryItem.classList.add('highlighted');
                        setTimeout(() => {
                            memoryItem.classList.remove('highlighted');
                        }, 2000);
                    }
                }

                // Add animation for deletion
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes fadeOut {
                        from { opacity: 1; }
                        to { opacity: 0; }
                    }
                `;
                document.head.appendChild(style);

                // Warn about unsaved changes
                window.addEventListener('beforeunload', function (e) {
                    if (isModified) {
                        e.preventDefault();
                        e.returnValue = '';
                    }
                });

                // Remove initial population on page load since cache may be empty

                function updateUserIds() {
                    fetch('/user_ids')
                    .then(response => response.json())
                    .then(data => {
                        const select = document.getElementById('user_id_select');
                        const deleteButton = document.getElementById('delete_user_button');
                        if(data.user_ids && data.user_ids.length > 0) {
                            select.innerHTML = '<option value="">All Users</option>' +
                                data.user_ids.map(uid => `<option value="${uid}">${uid}</option>`).join('');
                            deleteButton.style.display = 'inline-block';
                        } else {
                            select.innerHTML = '<option value="">All Users</option>';
                            deleteButton.style.display = 'none';
                        }
                        updateReplaceScope(); // Update replace scope when user list changes
                    })
                    .catch(error => console.error('Error fetching user IDs:', error));
                }

                function updateReplaceScope() {
                    const userId = document.getElementById('user_id_select').value;
                    const scopeSpan = document.getElementById('replace-scope');
                    if (userId) {
                        scopeSpan.textContent = `Scope: User ${userId}`;
                    } else {
                        scopeSpan.textContent = 'Scope: All Users';
                    }
                }

                function findAndReplace() {
                    const findText = document.getElementById('find_text').value.trim();
                    const replaceText = document.getElementById('replace_text').value;
                    const caseSensitive = document.getElementById('case_sensitive').checked;
                    const wholeWords = document.getElementById('whole_words').checked;
                    const userId = document.getElementById('user_id_select').value;
                    
                    if (!findText) {
                        alert('Please enter text to find');
                        return;
                    }
                    
                    if (!confirm(`Are you sure you want to replace "${findText}" with "${replaceText}" across ${userId ? `user ${userId}'s` : 'all'} memories? This action cannot be undone.`)) {
                        return;
                    }
                    
                    const requestData = {
                        find_text: findText,
                        replace_text: replaceText,
                        case_sensitive: caseSensitive,
                        whole_words: wholeWords
                    };
                    
                    if (userId) {
                        requestData.user_id = userId;
                    }
                    
                    fetch('/find-replace', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestData)
                    })
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('replace-status');
                        statusDiv.innerHTML = `
                            <div class="success" style="margin-top: 10px;">
                                ${data.message}<br>
                                Changes made: ${data.changes_made}<br>
                                Memories processed: ${data.memories_processed}<br>
                                Scope: ${data.user_scope}
                            </div>`;
                        
                        // Update stats display
                        if (data.stats) {
                            document.getElementById('stats').innerHTML = `
                                <div class="success">
                                    Cache updated after find/replace<br>
                                    Memories: ${data.stats.memories}<br>
                                    Users: ${data.stats.users}<br>
                                    Index Terms: ${data.stats.index_terms}
                                </div>`;
                        }
                        
                        // Refresh the current view
                        fetchMemories();
                        updateVisualization();
                        
                        // Clear status after 5 seconds
                        setTimeout(() => {
                            statusDiv.innerHTML = '';
                        }, 5000);
                        
                        // Mark as modified for save prompt
                        isModified = true;
                    })
                    .catch(error => {
                        const statusDiv = document.getElementById('replace-status');
                        statusDiv.innerHTML = `
                            <div class="error" style="margin-top: 10px;">
                                Error: ${error.message || error}
                            </div>`;
                    });
                }

                function deleteUser() {
                    const userId = document.getElementById('user_id_select').value;
                    if (!userId) {
                        alert('Please select a user to delete');
                        return;
                    }
                    
                    if (!confirm(`Are you sure you want to delete user ${userId} and all their memories? This action cannot be undone.`)) {
                        return;
                    }
                    
                    fetch(`/user/${userId}`, {
                        method: 'DELETE'
                    })
                    .then(async response => {
                        if (!response.ok) {
                            const errorText = await response.text();
                            throw new Error(errorText);
                        }
                        return response.json();
                    })
                    .then(data => {
                        // Update stats display
                        if (data.stats) {
                            document.getElementById('stats').innerHTML = `
                                <div class="success">
                                    ${data.message}<br>
                                    Memories: ${data.stats.memories}<br>
                                    Users: ${data.stats.users}<br>
                                    Index Terms: ${data.stats.index_terms}
                                </div>`;
                        }
                        
                        // Reset user selection and refresh the view
                        document.getElementById('user_id_select').value = '';
                        updateUserIds();
                        fetchMemories();
                        isModified = true;
                    })
                    .catch(error => {
                        console.error('Delete user error:', error);
                        alert('Error deleting user: ' + error.message);
                    });
                }
            </script>
        </div>
    </body>
</html>
    """

def _get_memory_keywords(memory_id: int) -> Dict[str, float]:
    """Get keywords with TF-IDF weights for a memory"""
    if memory_id >= len(cache['memories']) or cache['memories'][memory_id] is None:
        return {}
        
    memory = cache['memories'][memory_id]
    total_docs = len([m for m in cache['memories'] if m is not None])
    doc_freqs = {}
    for word, memory_list in cache['inverted_index'].items():
        doc_freqs[word] = len(set(memory_list))
    
    keywords = {}
    word_counts = defaultdict(int)
    tokens = _tokenize(memory)
    total_tokens = len(tokens)
    
    for token in tokens:
        word_counts[token] += 1
        
    for word, freq in word_counts.items():
        if word in cache['inverted_index']:
            tf = freq / total_tokens
            idf = math.log(total_docs / (doc_freqs[word] or 1))
            tfidf = tf * idf
            keywords[word] = round(tfidf, 3)
            
    return keywords

def _perform_search_sync(query: str, user_id: Optional[str], page: int, per_page: int):
    """Synchronous search function to run in thread pool"""
    results = []
    memory_ids = set(range(len(cache['memories'])))
    
    # Filter for specific user if provided
    if user_id:
        memory_ids = set(cache['user_memories'].get(user_id, []))
    
    # Pre-calculate document frequencies and lengths
    total_docs = len([m for m in cache['memories'] if m is not None])
    doc_freqs = {}
    doc_lengths = {}
    avg_length = 0
    
    # Single pass to gather statistics
    for word, memory_list in cache['inverted_index'].items():
        doc_freqs[word] = len(set(memory_list))
    
    for mid in memory_ids:
        if mid < len(cache['memories']) and cache['memories'][mid] is not None:
            tokens = _tokenize(cache['memories'][mid])
            doc_lengths[mid] = len(tokens)
            avg_length += len(tokens)
    
    avg_length = avg_length / len(doc_lengths) if doc_lengths else 1
    
    # Calculate scores for each memory
    for mid in sorted(memory_ids):
        if mid >= len(cache['memories']) or cache['memories'][mid] is None:
            continue
            
        memory = cache['memories'][mid]
        word_counts = defaultdict(int)
        tokens = _tokenize(memory)
        
        for token in tokens:
            word_counts[token] += 1
            
        # Calculate hybrid score
        score = 0.0
        if query:
            query_tokens = set(_tokenize(query))
            k1, b = 1.5, 0.75  # BM25 parameters
            
            for word in query_tokens:
                if word in word_counts and word in doc_freqs:
                    tf = word_counts[word]
                    df = doc_freqs[word]
                    doc_len = doc_lengths[mid]
                    
                    # BM25-like score component
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    tf_normalized = ((k1 + 1) * tf) / (k1 * (1 - b + b * doc_len / avg_length) + tf)
                    score += idf * tf_normalized
        
        memory_metadata = cache['metadata']['memory_stats'].get(mid, {})
        
        # Get keywords for display (keep existing TF-IDF for keyword highlighting)
        keywords = {}
        for word, freq in word_counts.items():
            if word in cache['inverted_index']:
                tf = freq / len(tokens)
                idf = math.log(total_docs / (doc_freqs[word] or 1))
                keywords[word] = round(tf * idf, 3)
        
        results.append({
            "id": mid,
            "text": memory,
            "user_id": next((uid for uid, mids in cache['user_memories'].items() if mid in mids), None),
            "score": score,
            "last_modified": memory_metadata.get('last_modified'),
            "last_action": memory_metadata.get('action'),
            "keywords": keywords
        })
    
    # Sort by score if query provided
    if query:
        results.sort(key=lambda x: x["score"], reverse=True)
    
    # Calculate pagination
    total_items = len(results)
    total_pages = math.ceil(total_items / per_page)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get paginated results
    paginated_results = results[start_idx:end_idx]
    
    return {
        "memories": paginated_results,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_items": total_items,
            "per_page": per_page
        }
    }

def _generate_visualization_sync(query: str, user_id: Optional[str], page: int, per_page: int):
    """Synchronous visualization generation to run in thread pool"""
    # Get memories from search results with exact pagination
    search_data = _perform_search_sync(query, user_id, page, per_page)
    search_results = search_data["memories"]
    
    if not search_results:
        return '<div class="error">No memories to visualize</div>'
    
    # Build efficient keyword index
    keyword_to_memories = defaultdict(set)
    memory_to_keywords = {}
    max_keywords = 1
    
    # O(n) single pass to build indexes
    for result in search_results:
        mid = result["id"]
        keywords = result["keywords"]
        if keywords:
            memory_to_keywords[mid] = {
                "keywords": keywords,
                "score": result["score"]
            }
            max_keywords = max(max_keywords, len(keywords))
            # Index keywords for fast intersection
            for word in keywords:
                keyword_to_memories[word].add(mid)
    
    if not memory_to_keywords:
        return '<div class="error">No memories with keywords to visualize</div>'
    
    # Calculate links using inverted index, respecting user_id filter
    links = defaultdict(float)  # (mem1,mem2) -> weight
    max_shared = 1
    
    # Get set of valid memory IDs for the current user
    valid_memory_ids = set(memory_to_keywords.keys())
    
    for mid, data in memory_to_keywords.items():
        for word in data["keywords"]:
            # Only consider memories that are in the current results
            sharing_memories = keyword_to_memories[word] & valid_memory_ids
            for other_mid in sharing_memories:
                if other_mid > mid:
                    key = (mid, other_mid)
                    links[key] += 1
                    max_shared = max(max_shared, links[key])

    # Force-directed layout calculation
    iterations = 50  # Number of iterations for force-directed layout
    k = 50  # Optimal distance between nodes
    temperature = 0.9  # Initial temperature for simulated annealing
    
    # Initialize random positions
    positions = {}
    node_sizes = {}  # Store node sizes for overlap calculations
    for mid in memory_to_keywords:
        positions[mid] = {
            'x': random.uniform(100, 700),
            'y': random.uniform(100, 500),
            'dx': 0,
            'dy': 0
        }
        # Calculate and store node size
        base_size = 5 + (15 * len(memory_to_keywords[mid]["keywords"]) / max_keywords)
        score_boost = memory_to_keywords[mid]["score"] if query else 1
        node_sizes[mid] = min(25, base_size * (1 + score_boost / 2))
    
    # Run force-directed layout
    for _ in range(iterations):
        # Calculate repulsive forces between all nodes
        for mid1 in positions:
            positions[mid1]['dx'] = 0
            positions[mid1]['dy'] = 0
            for mid2 in positions:
                if mid1 != mid2:
                    dx = positions[mid1]['x'] - positions[mid2]['x']
                    dy = positions[mid1]['y'] - positions[mid2]['y']
                    distance = max(0.1, math.sqrt(dx * dx + dy * dy))
                    
                    # Calculate minimum required distance based on node sizes
                    min_distance = (node_sizes[mid1] + node_sizes[mid2]) * 1.5
                    
                    # Stronger repulsion when nodes are too close
                    if distance < min_distance:
                        force = k * k * min_distance / (distance * distance)
                        force *= 2.0  # Extra repulsion for overlapping nodes
                    else:
                        force = k * k / distance
                    
                    positions[mid1]['dx'] += dx / distance * force
                    positions[mid1]['dy'] += dy / distance * force

        # Calculate attractive forces for linked nodes
        for (mid1, mid2), weight in links.items():
            dx = positions[mid1]['x'] - positions[mid2]['x']
            dy = positions[mid1]['y'] - positions[mid2]['y']
            distance = max(0.1, math.sqrt(dx * dx + dy * dy))
            
            # Limit attraction based on minimum node distance
            min_distance = (node_sizes[mid1] + node_sizes[mid2]) * 1.2
            if distance > min_distance:
                force = (distance - min_distance) * distance / (k * weight)
                
                positions[mid1]['dx'] -= dx / distance * force
                positions[mid1]['dy'] -= dy / distance * force
                positions[mid2]['dx'] += dx / distance * force
                positions[mid2]['dy'] += dy / distance * force

        # Update positions with temperature cooling
        for mid in positions:
            dx = max(-5, min(5, positions[mid]['dx'])) * temperature
            dy = max(-5, min(5, positions[mid]['dy'])) * temperature
            
            # Ensure nodes stay within bounds with padding based on node size
            padding = node_sizes[mid] * 1.2
            positions[mid]['x'] = max(padding, min(800 - padding, positions[mid]['x'] + dx))
            positions[mid]['y'] = max(padding, min(600 - padding, positions[mid]['y'] + dy))
        
        temperature *= 0.9  # Cool down

    # Generate SVG with improved styling
    svg = [
        f'<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">',
        '<style>',
        '.memory-link { stroke: #000000; transition: all 0.3s; }',
        '.memory-link:hover { stroke: #000000; opacity: 0.8; }',
        '.memory-node { fill: #ff9c9c; stroke: #000000; stroke-width: 2; transition: all 0.3s; cursor: pointer; }',
        '.memory-node.active { fill: #ff9c9c; opacity: 0.8; }',
        '.memory-node:hover { fill: #ff9c9c; opacity: 0.7; }',
        '.memory-label { font-family: "Courier New", Courier, monospace; font-weight: bold; font-size: 10px; fill: #000000; pointer-events: none; }',
        '.memory-score { font-family: "Courier New", Courier, monospace; font-weight: bold; font-size: 8px; fill: #000000; pointer-events: none; }',
        '.memory-keyword { font-family: "Courier New", Courier, monospace; font-style: italic; font-size: 9px; fill: #000000; pointer-events: none; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ff9c9c"/>'
    ]
    
    # Draw links with gradient opacity
    for (mid1, mid2), weight in links.items():
        opacity = 0.2 + weight / max_shared * 0.6
        width = 1 + weight / max_shared * 3
        svg.append(
            f'<line class="memory-link" x1="{positions[mid1]["x"]}" y1="{positions[mid1]["y"]}" '
            f'x2="{positions[mid2]["x"]}" y2="{positions[mid2]["y"]}" '
            f'stroke-width="{width}" opacity="{opacity}"/>'
        )
    
    # Draw nodes
    for mid, data in memory_to_keywords.items():
        # Node size based on normalized keyword count and search score
        base_size = 5 + (15 * len(data["keywords"]) / max_keywords)
        score_boost = data["score"] if query else 1
        size = min(25, base_size * (1 + score_boost / 2))
        
        # Get top keyword
        sorted_keywords = sorted(data["keywords"].items(), key=lambda x: x[1], reverse=True)
        top_keyword = sorted_keywords[0][0] if sorted_keywords else ""
        
        # Node class with active state based on score
        node_class = 'memory-node' + (' active' if data["score"] > 1 else '')
        
        svg.append(
            f'<circle class="{node_class}" cx="{positions[mid]["x"]}" cy="{positions[mid]["y"]}" '
            f'r="{size}" data-id="{mid}" onclick="highlightMemory({mid})"/>'
        )
        svg.append(
            f'<text class="memory-label" x="{positions[mid]["x"]}" y="{positions[mid]["y"] + size + 10}" '
            f'text-anchor="middle">{mid}</text>'
        )
        if query:
            svg.append(
                f'<text class="memory-score" x="{positions[mid]["x"]}" y="{positions[mid]["y"] - size - 5}" '
                f'text-anchor="middle">{data["score"]:.2f}</text>'
            )
        if top_keyword:
            svg.append(
                f'<text class="memory-keyword" x="{positions[mid]["x"]}" y="{positions[mid]["y"]}" '
                f'text-anchor="middle" dominant-baseline="middle">{top_keyword}</text>'
            )
    
    svg.append('</svg>')
    
    return '\n'.join(svg)

@app.get("/list/")
async def list_memories(
    page: int = 1,
    per_page: int = 10,
    user_id: Optional[str] = None
):
    """List memories with pagination"""
    if not cache['memories']:
        raise HTTPException(status_code=400, detail="No cache data loaded")
        
    # Filter memories
    memory_ids = range(len(cache['memories']))
    if user_id:
        memory_ids = cache['user_memories'].get(user_id, [])
    
    # Get valid memories
    valid_memories = [
        mid for mid in memory_ids 
        if mid < len(cache['memories']) and cache['memories'][mid] is not None
    ]
    
    # Calculate pagination
    total_memories = len(valid_memories)
    total_pages = math.ceil(total_memories / per_page)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get paginated memories
    paginated_ids = valid_memories[start_idx:end_idx]
    
    results = []
    for mid in paginated_ids:
        memory = cache['memories'][mid]
        memory_metadata = cache['metadata']['memory_stats'].get(mid, {})
        
        results.append({
            "id": mid,
            "text": memory,
            "user_id": next((uid for uid, mids in cache['user_memories'].items() 
                           if mid in mids), None),
            "last_modified": memory_metadata.get('last_modified'),
            "last_action": memory_metadata.get('action'),
            "keywords": _get_memory_keywords(mid)
        })
    
    return {
        "memories": results,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_items": total_memories,
            "per_page": per_page
        }
    }

@app.get("/user_ids")
async def get_user_ids():
    """Return the list of user IDs from the loaded cache."""
    return {"user_ids": list(cache['user_memories'].keys())}

@app.delete("/user/{user_id}")
async def remove_user(user_id: str):
    """Remove a user and all their associated memories from the index.
    
    Args:
        user_id (str): ID of the user to remove
        
    Returns:
        dict: Stats about the operation
    """
    if not cache['memories']:
        raise HTTPException(status_code=400, detail="No cache data loaded")
        
    try:
        if user_id not in cache['user_memories']:
            raise ValueError("User not found in memory index")
            
        # Create a backup of the current state
        backup = {
            'memories': cache['memories'].copy(),
            'user_memories': {k: v.copy() for k, v in cache['user_memories'].items()},
            'inverted_index': {k: v.copy() for k, v in cache['inverted_index'].items()},
            'metadata': {
                'last_modified': cache['metadata']['last_modified'],
                'version': cache['metadata']['version'],
                'memory_stats': {k: v.copy() for k, v in cache['metadata']['memory_stats'].items()}
            }
        }
        
        try:
            # Get all memory IDs for this user
            memory_ids_to_remove = sorted(cache['user_memories'][user_id], reverse=True)
            
            # Update metadata before removal
            now = datetime.now().isoformat()
            cache['metadata']['last_modified'] = now
            
            # Single pass: Remove memories and update all indices simultaneously
            for memory_id in memory_ids_to_remove:
                # Mark for deletion in metadata
                cache['metadata']['memory_stats'][memory_id] = {
                    'last_modified': now,
                    'action': 'deleted'
                }
                
                # Remove the memory
                cache['memories'].pop(memory_id)
                
                # Update all higher indices in user_memories
                for uid, mems in cache['user_memories'].items():
                    cache['user_memories'][uid] = [
                        mid if mid < memory_id else mid - 1 
                        for mid in mems 
                        if mid != memory_id
                    ]
                
                # Update inverted index
                for word in list(cache['inverted_index'].keys()):
                    cache['inverted_index'][word] = [
                        mid if mid < memory_id else mid - 1 
                        for mid in cache['inverted_index'][word] 
                        if mid != memory_id
                    ]
                    # Remove empty word entries
                    if not cache['inverted_index'][word]:
                        del cache['inverted_index'][word]
            
            # Remove user from memory mapping
            del cache['user_memories'][user_id]
            
            # Clean up any orphaned memories
            all_user_memories = set()
            for user_memories in cache['user_memories'].values():
                all_user_memories.update(user_memories)
                
            orphaned = [i for i in range(len(cache['memories'])) 
                       if i not in all_user_memories]
            
            if orphaned:
                # Remove orphaned memories in reverse order
                for memory_id in sorted(orphaned, reverse=True):
                    cache['memories'].pop(memory_id)
                    
                    # Update inverted index
                    for word in list(cache['inverted_index'].keys()):
                        cache['inverted_index'][word] = [
                            mid if mid < memory_id else mid - 1 
                            for mid in cache['inverted_index'][word] 
                            if mid != memory_id
                        ]
                        if not cache['inverted_index'][word]:
                            del cache['inverted_index'][word]
            
            # Return updated stats
            active_memories = len([m for m in cache['memories'] if m is not None])
            return {
                "message": f"Successfully removed user {user_id} and {len(memory_ids_to_remove)} memories",
                "stats": {
                    "memories": active_memories,
                    "users": len(cache['user_memories']),
                    "index_terms": len(cache['inverted_index']),
                    "memories_removed": len(memory_ids_to_remove),
                    "orphaned_removed": len(orphaned)
                }
            }
            
        except Exception as e:
            # Restore from backup on any error
            cache['memories'] = backup['memories']
            cache['user_memories'] = backup['user_memories']
            cache['inverted_index'] = backup['inverted_index']
            cache['metadata'] = backup['metadata']
            raise ValueError(f"Error during user removal, restored from backup: {str(e)}")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error removing user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-replace")
async def find_replace_memories(request: FindReplaceRequest):
    """Find and replace text across memories with optional user filtering"""
    if not cache['memories']:
        raise HTTPException(status_code=400, detail="No cache data loaded")
    
    if not request.find_text:
        raise HTTPException(status_code=400, detail="Find text cannot be empty")
    
    try:
        # Determine which memories to process
        memory_ids_to_process = set()
        if request.user_id:
            # Process only memories for specific user
            memory_ids_to_process = set(cache['user_memories'].get(request.user_id, []))
        else:
            # Process all memories
            memory_ids_to_process = set(range(len(cache['memories'])))
        
        # Filter out None memories
        valid_memory_ids = [
            mid for mid in memory_ids_to_process 
            if mid < len(cache['memories']) and cache['memories'][mid] is not None
        ]
        
        if not valid_memory_ids:
            return {
                "message": "No memories found to process",
                "changes_made": 0,
                "memories_processed": 0
            }
        
        # Prepare regex pattern for replacement
        find_pattern = request.find_text
        if request.whole_words:
            find_pattern = r'\b' + re.escape(request.find_text) + r'\b'
        else:
            find_pattern = re.escape(request.find_text)
        
        flags = 0 if request.case_sensitive else re.IGNORECASE
        
        # Track changes
        changes_made = 0
        updated_memory_ids = []
        
        # Process each memory
        for memory_id in valid_memory_ids:
            original_text = cache['memories'][memory_id]
            
            # Perform replacement
            new_text = re.sub(find_pattern, request.replace_text, original_text, flags=flags)
            
            # Check if text actually changed
            if new_text != original_text:
                cache['memories'][memory_id] = new_text
                updated_memory_ids.append(memory_id)
                changes_made += 1
        
        # If changes were made, rebuild the entire index
        if changes_made > 0:
            _rebuild_entire_index()
            
            # Update metadata
            now = datetime.now().isoformat()
            cache['metadata']['last_modified'] = now
            
            # Mark all updated memories in metadata
            for memory_id in updated_memory_ids:
                cache['metadata']['memory_stats'][memory_id] = {
                    'last_modified': now,
                    'action': 'find_replace'
                }
        
        # Return statistics
        active_memories = len([m for m in cache['memories'] if m is not None])
        return {
            "message": f"Find and replace completed. Updated {changes_made} memories.",
            "changes_made": changes_made,
            "memories_processed": len(valid_memory_ids),
            "user_scope": request.user_id or "ALL USERS",
            "stats": {
                "memories": active_memories,
                "users": len(cache['user_memories']),
                "index_terms": len(cache['inverted_index'])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in find and replace: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Find and replace failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

