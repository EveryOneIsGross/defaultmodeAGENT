from typing import List, Tuple, Optional, Dict
import numpy as np
import aiohttp
from logger import logging
from pydantic import BaseModel, Field
from api_client import get_embeddings
import asyncio

class EmbeddingConfig(BaseModel):
    """Pydantic model for embedding configuration."""
    provider: str = Field(
        default='ollama',
        description="Provider for embedding service"
    )
    model: str = Field(
        default='nomic-embed-text',  # Ollama's default embedding model
        description="Specific model for embeddings"
    )
    api_base: str = Field(
        default='http://localhost:11434',
        description="Base URL for Ollama API"
    )
    dimensions: int = Field(
        default=768,  # nomic-embed-text default
        description="Expected embedding dimensions"
    )

class HippocampusConfig(BaseModel):
    """Pydantic model for Hippocampus configuration - provides vector embeddings for downstream search."""
    embedding_provider: str = Field(default='ollama', description="Provider for embedding service")
    embedding_model: str = Field(default='all-minilm:latest', description="Model to use for embeddings")

class Hippocampus:
    def __init__(self, config: HippocampusConfig, logger=None):
        self.config = config
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self.embedding_config = EmbeddingConfig()
        self.logger = logger or logging.getLogger('bot.default')
        
    async def _get_ollama_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embeddings specifically from Ollama API."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.embedding_config.api_base}/api/embeddings",
                    json={
                        "model": self.embedding_config.model,
                        "prompt": text,
                        "options": {
                            "temperature": 0.0  # Embeddings should be deterministic
                        }
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Ollama API returned status {response.status}: {error_text}")
                        raise Exception(f"Ollama API returned status {response.status}: {error_text}")
                    
                    result = await response.json()
                    embedding = np.array(result['embedding'])
                    
                    # Verify embedding dimensions
                    if embedding.shape[0] != self.embedding_config.dimensions:
                        raise ValueError(
                            f"Unexpected embedding dimensions: got {embedding.shape[0]}, "
                            f"expected {self.embedding_config.dimensions}"
                        )
                    
                    return embedding
                    
            except Exception as e:
                self.logger.error(f"Ollama embedding error: {str(e)}")
                return None

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embeddings with provider-specific handling."""
        if text not in self._embedding_cache:
            try:
                if self.config.embedding_provider == 'ollama':
                    embedding = await self._get_ollama_embedding(text)
                else:
                    # Fallback to existing get_embeddings for other providers
                    embedding = await get_embeddings(
                        text,
                        provider=self.config.embedding_provider,
                        model=self.config.embedding_model
                    )
                    embedding = np.array(embedding)
                
                if embedding is not None:
                    # Normalize the embedding vector
                    embedding = embedding / np.linalg.norm(embedding)
                    self._embedding_cache[text] = embedding
                    
            except Exception as e:
                self.logger.error(f"Embedding generation failed: {str(e)}")
                return None
                
        return self._embedding_cache[text] 

    async def _get_ollama_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for multiple texts in a single batch from Ollama API."""
        async with aiohttp.ClientSession() as session:
            try:
                # Create tasks for parallel embedding requests
                tasks = []
                for text in texts:
                    tasks.append(
                        session.post(
                            f"{self.embedding_config.api_base}/api/embeddings",
                            json={
                                "model": self.embedding_config.model,
                                "prompt": text,
                                "options": {"temperature": 0.0}
                            }
                        )
                    )
                
                # Execute requests concurrently
                responses = await asyncio.gather(*tasks)
                embeddings = []
                
                for response in responses:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Ollama API returned status {response.status}: {error_text}")
                        continue
                    
                    result = await response.json()
                    embedding = np.array(result['embedding'])
                    
                    # Verify embedding dimensions
                    if embedding.shape[0] != self.embedding_config.dimensions:
                        self.logger.error(f"Unexpected embedding dimensions: {embedding.shape[0]}")
                        continue
                        
                    embeddings.append(embedding)
                
                return np.array(embeddings)
                    
            except Exception as e:
                self.logger.error(f"Batch embedding error: {str(e)}")
                return None

    async def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get cached embeddings for multiple texts with batch processing."""
        uncached_texts = []
        uncached_indices = []
        embeddings = np.zeros((len(texts), self.embedding_config.dimensions))
        
        # Check cache and collect uncached texts
        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                embeddings[i] = self._embedding_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if uncached_texts:
            try:
                if self.config.embedding_provider == 'ollama':
                    new_embeddings = await self._get_ollama_embeddings_batch(uncached_texts)
                else:
                    # Batch request for other providers
                    new_embeddings = await get_embeddings(
                        uncached_texts,
                        provider=self.config.embedding_provider,
                        model=self.config.embedding_model
                    )
                    new_embeddings = np.array(new_embeddings)
                
                if new_embeddings is not None:
                    # Normalize and cache new embeddings
                    for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        normalized_embedding = embedding / np.linalg.norm(embedding)
                        self._embedding_cache[text] = normalized_embedding
                        embeddings[uncached_indices[i]] = normalized_embedding
                        
            except Exception as e:
                self.logger.error(f"Batch embedding generation failed: {str(e)}")
                
        return embeddings

    async def rerank_memories(self, query: str, memories: List[Tuple[str, float]], threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Rerank memories using batched vector similarity."""
        self.logger.info(f"Starting batch reranking for query: {query[:100]}... with {len(memories)} candidates")
        
        # Extract memory texts
        memory_texts = [memory[0] for memory in memories]
        
        # Get embeddings for query and all memories in batch
        query_embedding = await self._get_embedding(query)
        if query_embedding is None:
            self.logger.error("Failed to generate query embedding")
            return []
            
        memory_embeddings = await self._get_embeddings_batch(memory_texts)
        
        # Calculate similarities in one operation
        similarities = np.dot(memory_embeddings, query_embedding)
        
        # Create reranked list with threshold filtering
        reranked = [
            (memory_text, float(similarity))
            for memory_text, similarity in zip(memory_texts, similarities)
            if similarity >= threshold
        ]
        
        # Sort by similarity scores
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Batch reranking complete - {len(reranked)}/{len(memories)} memories above threshold")
        return reranked 