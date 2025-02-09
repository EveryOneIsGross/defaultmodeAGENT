import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import base64
import logging
import asyncio
import pickle
import string
import re
from collections import defaultdict, Counter

from github import Github

from memory import *

from bot_config import *

# Global variable for repository processing status
repo_processing_event = asyncio.Event()

# GitHub Integration
"""GitHub repository integration and indexing functionality.

This module provides classes for interacting with GitHub repositories, including fetching
file contents, indexing repository files, and searching through indexed content.

The main components are:

- GitHubRepo: Handles direct interaction with GitHub repositories via the GitHub API
- RepoIndex: Provides indexing and search functionality for repository contents

The integration supports:
- Fetching and caching repository file contents
- Building searchable indexes of repository content
- Directory structure traversal
- File content search and retrieval

Key features:
- Efficient caching of repository contents
- Configurable depth for directory traversal 
- Text cleaning and normalization for search
- Relevance-based file search
"""

class GitHubRepo:
    def __init__(self, token, repo_name):
        self.g = Github(token)
        self.repo = self.g.get_repo(repo_name)

    def get_file_content(self, file_path):
        try:
            file_content = self.repo.get_contents(file_path)
            if file_content.size > 1000000:  # 1MB limit
                return "File is too large to fetch content directly."
            content = base64.b64decode(file_content.content).decode('utf-8')
            return content
        except Exception as e:
            return f"Error fetching file: {str(e)}"

    def get_directory_structure(self, path="", prefix="", max_depth=2, current_depth=0):
        if current_depth > max_depth:
            return []

        contents = self.repo.get_contents(path)
        structure = []
        for content in contents:
            if content.type == "dir":
                structure.append(f"{prefix}{content.name}/")
                if current_depth < max_depth:
                    structure.extend(self.get_directory_structure(
                        content.path, 
                        prefix + "  ", 
                        max_depth, 
                        current_depth + 1
                    ))
            else:
                structure.append(f"{prefix}{content.name}")
        return structure

class RepoIndex:
    def __init__(self, bot_name, max_depth=3):
        """Initialize the RepoIndex.
        
        Args:
            bot_name (str): Name of the bot for cache organization
            max_depth (int, optional): Maximum depth for repository traversal. Defaults to 3.
        """
        self.cache_manager = CacheManager(bot_name)
        self.cache_dir = self.cache_manager.get_cache_dir('repo_index')
        self.max_depth = max_depth
        self.repo_index = defaultdict(list)
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        self.load_cache()

    def index_repo_file(self, file_path, content):
        cleaned_content = self.clean_text(content)
        words = cleaned_content.split()
        for word in words:
            if file_path not in self.repo_index[word]:
                self.repo_index[word].append(file_path)

    def search_repo(self, query, k=5):
        cleaned_query = self.clean_text(query)
        query_words = cleaned_query.split()
        file_scores = Counter()

        for word in query_words:
            for file_path in self.repo_index.get(word, []):
                file_scores[file_path] += 1

        return file_scores.most_common(k)

    def clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)

    def clear_cache(self):
        self.repo_index.clear()
        cache_file = os.path.join(self.cache_dir, 'repo_index.pkl')
        if os.path.exists(cache_file):
            os.remove(cache_file)
        logging.info("Repository index cache cleared")

    def save_cache(self):
        with open(os.path.join(self.cache_dir, 'repo_index.pkl'), 'wb') as f:
            pickle.dump(dict(self.repo_index), f)
        logging.info("Repository index cache saved successfully.")

    def load_cache(self):
        """Load repository index from cache.
        Returns True if cache was successfully loaded, False otherwise."""
        global repo_processing_event
        repo_index_path = os.path.join(self.cache_dir, 'repo_index.pkl')
        if os.path.exists(repo_index_path):
            try:
                with open(repo_index_path, 'rb') as f:
                    self.repo_index = defaultdict(list, pickle.load(f))
                logging.info("Repository index cache loaded successfully.")
                repo_processing_event.set()  # Set the event to indicate repo is ready
                return True
            except Exception as e:
                logging.error(f"Error loading repository cache: {str(e)}")
        return False

async def fetch_and_chunk_repo_contents(repo, memory_index, max_depth=None):
    contents = repo.get_contents("")
    if contents is None:
        logging.error("Failed to fetch repository contents.")
        return

    logging.info(f"Starting to fetch contents for repo: {repo.full_name}")

    async def process_contents(contents, current_depth=0):
        tasks = []
        for content in contents:
            if content.type == "dir":
                if max_depth is None or current_depth < max_depth:
                    dir_contents = repo.get_contents(content.path)
                    await process_contents(dir_contents, current_depth + 1)
            elif content.type == "file":
                tasks.append(asyncio.create_task(process_repofile(content)))

            if len(tasks) >= 10:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

    async def process_repofile(file_content):
        try:
            _, file_extension = os.path.splitext(file_content.path)
            if file_extension.lower() in ALLOWED_EXTENSIONS:
                logging.debug(f"Processing file: {file_content.path}")
                try:
                    file_data = file_content.decoded_content.decode('utf-8')
                except UnicodeDecodeError:
                    logging.warning(f"UTF-8 decoding failed for {file_content.path}, trying latin-1")
                    try:
                        file_data = file_content.decoded_content.decode('latin-1')
                    except Exception as e:
                        logging.error(f"Failed to decode {file_content.path}: {str(e)}")
                        return
                memory_index.index_repo_file(file_content.path, file_data)
                logging.info(f"Successfully processed file: {file_content.path}")
            else:
                logging.debug(f"Skipping file with unsupported extension: {file_content.path}")
        except Exception as e:
            logging.error(f"Unexpected error processing {file_content.path}: {str(e)}")

    await process_contents(contents)

    memory_index.save_cache()
    logging.info(f"Finished processing repo: {repo.full_name}")

    # set the event to indicate that the repo has been processed

async def start_background_processing(repo, memory_index, max_depth=None, branch='main'):
    global repo_processing_event
    repo_processing_event.clear()
    
    try:
        await process_repo_contents(repo, '', memory_index, max_depth, branch)
        memory_index.save_cache()  # Save the cache after indexing
    except Exception as e:
        logging.error(f"Error in background processing for branch '{branch}': {str(e)}")
    finally:
        repo_processing_event.set()

async def process_repo_contents(repo, path, memory_index, max_depth=None, branch='main', current_depth=0):
    if max_depth is not None and current_depth > max_depth:
        return

    try:
        contents = repo.get_contents(path, ref=branch)
        for content in contents:
            if content.type == 'dir':
                await process_repo_contents(repo, content.path, memory_index, max_depth, branch, current_depth + 1)
            elif content.type == 'file':
                _, file_extension = os.path.splitext(content.path)
                if file_extension.lower() in ALLOWED_EXTENSIONS:
                    try:
                        file_content = content.decoded_content.decode('utf-8')
                        memory_index.index_repo_file(content.path, file_content)
                        logging.info(f"Indexed file: {content.path} (Branch: {branch})")
                    except UnicodeDecodeError:
                        logging.warning(f"Skipped binary file: {content.path} (Branch: {branch})")
                    except Exception as file_error:
                        logging.error(f"Error processing file {content.path} on branch '{branch}': {str(file_error)}")
                else:
                    logging.debug(f"Skipping file with unsupported extension: {content.path}")
    except Exception as e:
        logging.error(f"Error processing directory {path} on branch '{branch}': {str(e)}")