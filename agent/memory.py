import os
import json
import pickle
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Dict, Any
import string
import re
import math
import asyncio

from bot_config import config
from tokenizer import get_tokenizer, count_tokens

import os
import json
from collections import deque
import tempfile
import shutil

from datetime import datetime, timedelta
import uuid

from logger import BotLogger, logging

MAX_TOKENS = config.search.max_tokens
CONTEXT_CHUNKS = config.search.context_chunks

class CacheManager:
    def __init__(self, bot_name, temp_file_ttl=3600):

        """Initialize cache manager with bot name and conversation history limit."""
        self.bot_name = bot_name
        self.temp_file_ttl = temp_file_ttl
        self.base_cache_dir = os.path.join('cache', self.bot_name)
        
        # Set up logger
        self.logger = logging.getLogger(f'bot.{self.bot_name}.cache')
        
        # Only create the base bot directory
        os.makedirs(self.base_cache_dir, exist_ok=True)

    def get_cache_dir(self, cache_type):
        """Creates and returns a cache directory for a given type."""
        cache_dir = os.path.join(self.base_cache_dir, cache_type)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def get_temp_dir(self):
        """Gets the temp directory, creating it if needed."""
        temp_dir = self.get_cache_dir('temp')
        self.cleanup_temp_files()  # Only clean temp files when temp dir is actually used
        return temp_dir

    def get_user_temp_dir(self, user_id):
        """Get or create a temporary directory for a specific user.
        
        Args:
            user_id (str): Discord user ID
            
        Returns:
            str: Path to the user's temporary directory
        """
        user_temp_dir = os.path.join(self.get_temp_dir(), str(user_id))
        os.makedirs(user_temp_dir, exist_ok=True)
        return user_temp_dir

    def create_temp_file(self, user_id, prefix=None, suffix=None, content=None):
        """Creates a temporary file with optional content and returns its path.
        
        Args:
            user_id (str): Discord user ID
            prefix (str, optional): Prefix for the temporary filename
            suffix (str, optional): Suffix for the temporary filename (e.g., '.txt')
            content (str/bytes, optional): Content to write to the temporary file
            
        Returns:
            tuple: (str: Path to the created temporary file, str: Unique file ID)
        """
        # Generate a unique ID for this file
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Construct filename with all components
        filename_parts = []
        if prefix:
            filename_parts.append(prefix)
        filename_parts.extend([timestamp, file_id])
        filename = '_'.join(filename_parts)
        if suffix:
            filename += suffix
            
        # Get user's temp directory and create full path
        user_temp_dir = self.get_user_temp_dir(user_id)
        temp_path = os.path.join(user_temp_dir, filename)
        
        # Write content to file
        mode = 'wb' if isinstance(content, bytes) else 'w'
        encoding = None if isinstance(content, bytes) else 'utf-8'
        
        try:
            with open(temp_path, mode, encoding=encoding) as f:
                if content is not None:
                    f.write(content)
            
            try:
                self.logger.info(f"Created temporary file for user {user_id}: {temp_path}")
            except AttributeError:
                # Fallback in case logger is not available
                pass
            
            # Create metadata file to store creation time and other info
            metadata = {
                'created_at': datetime.now().isoformat(),
                'user_id': user_id,
                'file_id': file_id,
                'original_filename': filename
            }
            metadata_path = f"{temp_path}.meta"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            return temp_path, file_id
            
        except Exception as e:
            try:
                self.logger.error(f"Error creating temporary file for user {user_id}: {str(e)}")
            except AttributeError:
                # Fallback if logger is not available
                pass
            raise

    def get_temp_file(self, user_id, file_id):
        """Retrieve a temporary file path by its ID and verify user ownership.
        
        Args:
            user_id (str): Discord user ID
            file_id (str): Unique file ID
            
        Returns:
            str: Path to the temporary file if found and owned by user, None otherwise
        """
        user_temp_dir = self.get_user_temp_dir(user_id)
        
        try:
            for filename in os.listdir(user_temp_dir):
                if filename.endswith('.meta'):
                    metadata_path = os.path.join(user_temp_dir, filename)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    if metadata['file_id'] == file_id and metadata['user_id'] == user_id:
                        file_path = metadata_path[:-5]  # Remove .meta extension
                        if os.path.exists(file_path):
                            return file_path
                            
            return None
            
        except Exception as e:
            try:
                self.logger.error(f"Error retrieving temporary file {file_id} for user {user_id}: {str(e)}")
            except AttributeError:
                # Fallback if logger is not available
                pass
            return None

    def cleanup_temp_files(self, force=False):
        """Removes temporary files older than TTL."""
        current_time = datetime.now()
        temp_dir = self.get_cache_dir('temp')  # Get the path once
        
        try:
            # Iterate through user directories
            for user_id in os.listdir(temp_dir):
                user_temp_dir = os.path.join(temp_dir, user_id)
                if not os.path.isdir(user_temp_dir):
                    continue
                    
                for filename in os.listdir(user_temp_dir):
                    if filename.endswith('.meta'):
                        continue
                        
                    file_path = os.path.join(user_temp_dir, filename)
                    metadata_path = f"{file_path}.meta"
                    
                    try:
                        if os.path.exists(metadata_path):
                            # Use metadata to get file info
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            created_at = datetime.fromisoformat(metadata['created_at'])
                            file_age = current_time - created_at
                            
                            if force or file_age > timedelta(seconds=self.temp_file_ttl):
                                self.remove_temp_file(metadata['user_id'], metadata['file_id'])
                        else:
                            # If no metadata exists, use file creation time directly
                            file_age = current_time - datetime.fromtimestamp(os.path.getctime(file_path))
                            
                            if force or file_age > timedelta(seconds=self.temp_file_ttl):
                                # Direct file removal since we don't have metadata
                                try:
                                    os.remove(file_path)
                                    try:
                                        self.logger.info(f"Removed orphaned temporary file {filename} for user {user_id}")
                                    except AttributeError:
                                        pass
                                except Exception as e:
                                    try:
                                        self.logger.error(f"Error removing orphaned file {file_path}: {str(e)}")
                                    except AttributeError:
                                        pass
                            
                    except Exception as e:
                        try:
                            self.logger.error(f"Error processing temporary file {file_path}: {str(e)}")
                        except AttributeError:
                            # Fallback if logger is not available
                            pass
                        
                # Remove empty user directories
                if not os.listdir(user_temp_dir):
                    os.rmdir(user_temp_dir)
                    
        except Exception as e:
            try:
                self.logger.error(f"Error during temp file cleanup: {str(e)}")
            except AttributeError:
                # Fallback if logger is not available
                pass

    def remove_temp_file(self, user_id, file_id):
        """Safely removes a specific temporary file.
        
        Args:
            user_id (str): Discord user ID
            file_id (str): Unique file ID
        """
        file_path = self.get_temp_file(user_id, file_id)
        if not file_path:
            return
            
        try:
            # Remove the main file
            if os.path.exists(file_path):
                os.remove(file_path)
                
            # Remove the metadata file
            metadata_path = f"{file_path}.meta"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            try:
                self.logger.info(f"Removed temporary file {file_id} for user {user_id}")
            except AttributeError:
                # Fallback if logger is not available
                pass
            
        except Exception as e:
            try:
                self.logger.error(f"Error removing temporary file {file_id} for user {user_id}: {str(e)}")
            except AttributeError:
                # Fallback if logger is not available
                pass

# Memory

class UserMemoryIndex:
    """A class for indexing and searching user memories with efficient caching.
    
    This class provides functionality to store, index, and search through user memories
    using an inverted index approach. It handles caching of memories to disk and 
    supports per-user memory isolation.

    Attributes:
        cache_manager (CacheManager): Manager for handling cache operations
        cache_dir (str): Directory path for storing cache files
        max_tokens (int): Maximum number of tokens allowed in search results
        context_chunks (int): Number of context chunks to maintain
        tokenizer: Tokenizer for counting tokens in text
        inverted_index (defaultdict): Inverted index mapping words to memory IDs
        memories (list): List of all memory texts
        stopwords (set): Set of common words to ignore during indexing
        user_memories (defaultdict): Mapping of user IDs to their memory IDs
    """

    def __init__(self, cache_type, max_tokens=MAX_TOKENS, context_chunks=CONTEXT_CHUNKS, logger=None):
        """Initialize the UserMemoryIndex.

        Args:
            cache_type (str): Type of cache to use (e.g., 'user_memory_index')
            max_tokens (int, optional): Maximum tokens in search results. Defaults to MAX_TOKENS.
            context_chunks (int, optional): Number of context chunks. Defaults to CONTEXT_CHUNKS.
            logger: Logger instance to use. If None, uses default bot logger.
        """
        # Split the cache_type to get the bot name and cache type
        parts = cache_type.split('/')
        if len(parts) >= 2:
            self.bot_name = parts[0]
            cache_subtype = parts[-1]  # Use the last part as the cache subtype
        else:
            self.bot_name = 'default'
            cache_subtype = cache_type

        self.cache_manager = CacheManager(self.bot_name)
        self.cache_dir = self.cache_manager.get_cache_dir(cache_subtype)
        self.max_tokens = max_tokens
        self.context_chunks = context_chunks
        self.tokenizer = get_tokenizer()  # Use global tokenizer
        self.inverted_index = defaultdict(list)
        self.memories = []
        self.stopwords = set([
            # Common articles and conjunctions
            'the', 'a', 'an', 'and', 'or', 'but', 'nor', 'yet', 'so',
            # Common prepositions
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
            # Common pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            # Common auxiliary verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            # Common modals
            'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
            # Common demonstratives
            'this', 'that', 'these', 'those'
        ])
        self.user_memories = defaultdict(list)
        self.logger = logger or logging.getLogger('bot.default')
        self.load_cache()

    def clean_text(self, text):
        """Clean and normalize text for indexing/searching.

        Args:
            text (str): Text to clean

        Returns:
            str: Cleaned text with punctuation removed, numbers removed, and stopwords filtered
        """
        # Clean rogue LLM tokens
        text = text.replace("<|endoftext|>", "").replace("<|im_start|>", "").replace("<|im_end|>", "")
        text = text.lower()
        # Replace punctuation with spaces to preserve word boundaries
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        text = re.sub(r'\d+', '', text)
        # Split on whitespace and filter empty strings
        words = [w for w in text.split() if w]
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)

    def add_memory(self, user_id, memory_text):
        """Add a new memory for a user.

        Args:
            user_id (str): ID of the user this memory belongs to
            memory_text (str): Text content of the memory
        """
        memory_id = len(self.memories)
        self.memories.append(memory_text)
        self.user_memories[user_id].append(memory_id)
        
        cleaned_text = self.clean_text(memory_text)
        words = cleaned_text.split()
        for word in words:
            self.inverted_index[word].append(memory_id)
        
        self.logger.info(f"Added new memory for user {user_id}: {memory_text[:100]}...")
        self.save_cache()

    def clear_user_memories(self, user_id):
        """Clear all memories for a specific user and rebuild index.

        Args:
            user_id (str): ID of user whose memories should be cleared
        """
        if user_id in self.user_memories:
            memory_ids_to_remove = sorted(self.user_memories[user_id], reverse=True)
            
            # Remove memories from main list and adjust all indices
            for memory_id in memory_ids_to_remove:
                self.memories.pop(memory_id)
                
                # Update all higher indices in user_memories
                for uid, mems in self.user_memories.items():
                    self.user_memories[uid] = [
                        mid if mid < memory_id else mid - 1 
                        for mid in mems 
                        if mid != memory_id
                    ]
                
                # Update inverted index
                for word in list(self.inverted_index.keys()):
                    self.inverted_index[word] = [
                        mid if mid < memory_id else mid - 1 
                        for mid in self.inverted_index[word] 
                        if mid != memory_id
                    ]
                    # Remove empty word entries
                    if not self.inverted_index[word]:
                        del self.inverted_index[word]
            
            # Remove user from memory mapping
            del self.user_memories[user_id]
            
            self.logger.info(f"Cleared and rebuilt index after removing {len(memory_ids_to_remove)} memories for user {user_id}")
            self.save_cache()

    def search(self, query, k=5, user_id=None, dedup_threshold=0.95):
        """Search for relevant memories matching a query, removing duplicates.

        Args:
            query (str): Search query text
            k (int): Maximum number of results to return
            user_id (str, optional): If provided, only search this user's memories
            dedup_threshold (float): Threshold for considering memories as duplicates
            
        Returns:
            list: List of tuples containing (memory_text, relevance_score), where score is normalized to 0.00-1.00
        """
        cleaned_query = self.clean_text(query)
        query_words = cleaned_query.split()
        memory_scores = Counter()

        # Calculate IDF components once
        total_memories = len([m for m in self.memories if m is not None])
        doc_freqs = {word: len(self.inverted_index.get(word, [])) for word in query_words}

        # If user_id is provided, only search that user's memories
        if user_id:
            relevant_memory_ids = self.user_memories.get(user_id, [])
        else:
            relevant_memory_ids = range(len(self.memories))

        # BM25 parameters
        k1 = 1.2  # tf saturation curve
        b = 0.75  # length normalization parameter
        
        # Pre-compute document lengths only for relevant memories
        doc_lengths = {}
        total_length = 0
        valid_count = 0
        
        for mid in relevant_memory_ids:
            if mid < len(self.memories) and self.memories[mid] is not None:
                # Simple word count without full text cleaning for length calculation
                doc_len = len(self.memories[mid].split())
                doc_lengths[mid] = doc_len
                total_length += doc_len
                valid_count += 1
        
        avgdl = total_length / valid_count if valid_count > 0 else 1.0

        # Score memories with full BM25 weighting
        # Convert relevant_memory_ids to set for O(1) lookup
        relevant_set = set(relevant_memory_ids) if user_id else None
        for word in query_words:
            posting_list = self.inverted_index.get(word, [])
            if not posting_list:
                continue
            
            # Filter to relevant memories efficiently
            if relevant_set:
                posting = [mid for mid in posting_list if mid in relevant_set]
            else:
                posting = posting_list
                
            if not posting:
                continue

            tf_counts = Counter(posting)  # collapse duplicates → tf per doc
            df = len(tf_counts)  # docs containing word
            idf = math.log((total_memories - df + 0.5) / (df + 0.5) + 1.0)

            for mid, tf in tf_counts.items():
                # Use pre-computed document length
                doc_len = doc_lengths.get(mid, 1)
                length_norm = k1 * ((1 - b) + b * (doc_len / avgdl))
                bm25_weight = idf * ((k1 + 1) * tf) / (length_norm + tf)
                memory_scores[mid] += bm25_weight
        # Light normalization using pre-computed lengths
        for memory_id, score in memory_scores.items():
            doc_len = doc_lengths.get(memory_id, 1)
            memory_scores[memory_id] = score / math.log(1 + doc_len)
        
        # Get max score for normalization
        max_score = max(memory_scores.values()) if memory_scores else 1.0
        
        # Normalize all scores to 0.00-1.00
        for memory_id in memory_scores:
            memory_scores[memory_id] /= max_score
        
        sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Deduplication process
        results = []
        total_tokens = 0
        seen_content = set()  # Track unique content fingerprints
        
        for memory_id, score in sorted_memories:
            memory = self.memories[memory_id]
            memory_tokens = self.count_tokens(memory)

            if total_tokens + memory_tokens > self.max_tokens:
                break

            # Create a content fingerprint by cleaning and normalizing the text
            cleaned_memory = self.clean_text(memory)
            
            # Check for similar content using n-gram comparison
            is_duplicate = False
            for seen in seen_content:
                similarity = self._calculate_similarity(cleaned_memory, seen)
                if similarity > dedup_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                results.append((memory, score))
                seen_content.add(cleaned_memory)
                total_tokens += memory_tokens
                
                if len(results) >= k:
                    break

        self.logger.info(f"Found {len(results)} unique memories for query: {query[:100]}...")
      
        return results

    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts using character n-grams.
        
        Args:
            text1 (str): First text to compare
            text2 (str): Second text to compare
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Use 3-character n-grams for comparison
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text)-n+1))
        
        # Get n-grams for both texts
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        # Calculate Jaccard similarity
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0

    def count_tokens(self, text):
        """Count the number of tokens in text using the global tokenizer."""
        return count_tokens(text)
    
    def save_cache(self):
        """Save the current state to cache file."""
        cache_data = {
            'inverted_index': self.inverted_index,
            'memories': self.memories,
            'user_memories': self.user_memories
        }
        with open(os.path.join(self.cache_dir, 'memory_cache.pkl'), 'wb') as f:
            pickle.dump(cache_data, f)
        self.logger.info("Memory cache saved successfully.")

    def load_cache(self, cleanup_orphans=False, cleanup_nulls=True):
        """Load the state from cache file if it exists and validate index structure.

        Args:
            cleanup_orphans (bool): Whether to remove orphaned memories during load.
                                Default False to preserve all memories.
            cleanup_nulls (bool): Whether to remove null entries and repair indices.
                                Default True to clean up null entries.

        Returns:
            bool: True if cache was loaded successfully, False otherwise
        """
        cache_file = os.path.join(self.cache_dir, 'memory_cache.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.inverted_index = cache_data.get('inverted_index', defaultdict(list))
            self.memories = cache_data.get('memories', [])
            self.user_memories = cache_data.get('user_memories', defaultdict(list))
            
            # Validate and repair index state
            memory_count = len(self.memories)
            
            # Check for and fix any invalid memory references
            for word in list(self.inverted_index.keys()):
                # Remove any memory IDs that are out of range
                self.inverted_index[word] = [
                    mid for mid in self.inverted_index[word]
                    if mid < memory_count and self.memories[mid] is not None
                ]
                # Remove empty word entries
                if not self.inverted_index[word]:
                    del self.inverted_index[word]
            
            # Fix any user memory references that are out of range
            for user_id in list(self.user_memories.keys()):
                self.user_memories[user_id] = [
                    mid for mid in self.user_memories[user_id]
                    if mid < memory_count and self.memories[mid] is not None
                ]
                # Remove users with no valid memories
                if not self.user_memories[user_id]:
                    del self.user_memories[user_id]
            
            if cleanup_orphans:
                # Find truly orphaned memories (no user references)
                all_user_memories = set()
                for user_memories in self.user_memories.values():
                    all_user_memories.update(user_memories)
                
                orphaned = [(i, mem) for i, mem in enumerate(self.memories) 
                        if i not in all_user_memories]
                
                if orphaned:
                    # Log orphaned memories before removal
                    for i, mem in orphaned:
                        self.logger.warning(f"Found orphaned memory {i}: {mem[:100]}...")
                    
                    # Ask for confirmation if in interactive mode
                    self.logger.warning(f"Found {len(orphaned)} orphaned memories. Set cleanup_orphans=True to remove them.")
            
            # Validate and visualize index structure
            index_stats = {
                'total_memories': len(self.memories),
                'active_memories': len([m for m in self.memories if m is not None]),
                'total_users': len(self.user_memories),
                'vocabulary_size': len(self.inverted_index),
                'index_distribution': {
                    word: len(memories) for word, memories in self.inverted_index.items()
                }
            }

            # Calculate memory distribution per user
            index_stats['memories_per_user'] = {
                user_id: len(memories) for user_id, memories in self.user_memories.items()
            }

            # Log index structure
            self.logger.info("Memory Index Structure:")
            self.logger.info(f"Total Memories: {index_stats['total_memories']}")
            self.logger.info(f"Active Memories: {index_stats['active_memories']}")
            self.logger.info(f"Total Users: {index_stats['total_users']}")
            self.logger.info(f"Vocabulary Size: {index_stats['vocabulary_size']}")
            self.logger.info(f"Average memories per word: {sum(index_stats['index_distribution'].values()) / len(self.inverted_index) if self.inverted_index else 0:.2f}")
            self.logger.info(f"Average memories per user: {sum(index_stats['memories_per_user'].values()) / len(self.user_memories) if self.user_memories else 0:.2f}")

            # Handle null memories
            if index_stats['total_memories'] != index_stats['active_memories']:
                null_entries = [(i, mem) for i, mem in enumerate(self.memories) if mem is None]
                self.logger.warning(f"Found {len(null_entries)} null memories in index:")
                for idx, _ in null_entries:
                    self.logger.warning(f"Null entry at index {idx}")
                
                if cleanup_nulls:
                    # Remove nulls in reverse order to maintain index validity
                    for idx, _ in sorted(null_entries, reverse=True):
                        self.memories.pop(idx)
                        
                        # Update user memory indices
                        for user_id in self.user_memories:
                            self.user_memories[user_id] = [
                                mid if mid < idx else mid - 1
                                for mid in self.user_memories[user_id]
                                if mid != idx
                            ]
                        
                        # Update inverted index
                        for word in list(self.inverted_index.keys()):
                            self.inverted_index[word] = [
                                mid if mid < idx else mid - 1
                                for mid in self.inverted_index[word]
                                if mid != idx
                            ]
                            if not self.inverted_index[word]:
                                del self.inverted_index[word]
                    
                    self.logger.info(f"Removed {len(null_entries)} null entries")
                    self.save_cache()

            self.logger.info("Memory cache loaded successfully.")
            return True
        return False

    async def search_async(self, query, k=32, user_id=None, dedup_threshold=0.95):
        """Asynchronous version of search that runs in a thread pool to avoid blocking the event loop.
        
        Args:
            query (str): Search query text
            k (int): Maximum number of results to return
            user_id (str, optional): If provided, only search this user's memories
            dedup_threshold (float): Threshold for considering memories as duplicates
            
        Returns:
            list: List of tuples containing (memory_text, relevance_score), where score is normalized to 0.00-1.00
        """
        # Use a thread pool executor to run the CPU-bound search operation
        loop = asyncio.get_event_loop()
        self.logger.info(f"Starting async search for query: {query[:100]}...")
        
        # Run the expensive search operation in a thread pool
        results = await loop.run_in_executor(
            None,  # Use default executor
            lambda: self.search(query, k, user_id, dedup_threshold)
        )
        
        self.logger.info(f"Async search completed with {len(results)} results")
        return results

class RepoIndex:
    """A class for indexing and searching GitHub repository contents."""
    
    def __init__(self, cache_dir):
        """Initialize the repository indexing and searching class.
        
        Args:
            cache_dir (str): Path to the cache directory
        """
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.repo_index = defaultdict(set)
        self.load_cache()
        
    def index_file(self, file_path, content, branch='main'):
        """Index a file for search.
        
        Args:
            file_path (str): Path to the file
            content (str): Content of the file
            branch (str): Git branch
        """
        # Use only words longer than 3 characters, split by non-alphanumeric chars
        words = set(re.findall(r'\b\w{3,}\b', content.lower()))
        self.repo_index[branch].add(file_path)
        
        with open(os.path.join(self.cache_dir, f'{branch}_{file_path.replace("/", "_")}.txt'), 'w', encoding='utf-8') as f:
            f.write(content)
            
    def search_repo(self, query, branch='main', limit=5):
        """Search for relevant files matching a query.
        
        Args:
            query (str): Search query
            branch (str): Git branch to search
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of tuples containing (file_path, relevance_score)
        """
        # Search implementation based on term frequency
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_words:
            return []
            
        results = []
        for file_path in self.repo_index.get(branch, set()):
            try:
                with open(os.path.join(self.cache_dir, f'{branch}_{file_path.replace("/", "_")}.txt'), 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count query words in content
                file_words = set(re.findall(r'\b\w{3,}\b', content.lower()))
                match_count = sum(1 for word in query_words if word in file_words)
                
                if match_count > 0:
                    # Calculate simple relevance score
                    score = match_count / len(query_words)
                    results.append((file_path, score))
            except Exception as e:
                # Skip files that can't be read
                continue
                
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def search_repo_async(self, query, branch='main', limit=5):
        """Asynchronous version of search_repo that runs in a thread pool.
        
        Args:
            query (str): Search query
            branch (str): Git branch to search
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of tuples containing (file_path, relevance_score)
        """
        # Run the CPU-bound search in a thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,  # Use default executor
            lambda: self.search_repo(query, branch, limit)
        )
        return results
        
    def clear_cache(self):
        """Clear the index cache."""
        self.repo_index = defaultdict(set)
        for f in os.listdir(self.cache_dir):
            if f.endswith('.txt'):
                os.remove(os.path.join(self.cache_dir, f))
                
    def save_cache(self):
        """Save the current index to a cache file."""
        with open(os.path.join(self.cache_dir, 'repo_index.json'), 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable_index = {k: list(v) for k, v in self.repo_index.items()}
            json.dump(serializable_index, f)
            
    def load_cache(self):
        """Load the index from a cache file if it exists."""
        cache_file = os.path.join(self.cache_dir, 'repo_index.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                try:
                    loaded_index = json.load(f)
                    # Convert lists back to sets
                    self.repo_index = defaultdict(set, {k: set(v) for k, v in loaded_index.items()})
                except:
                    # If there's an error loading the cache, start fresh
                    self.repo_index = defaultdict(set)