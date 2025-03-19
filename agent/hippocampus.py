from typing import List, Tuple, Optional, Dict
import numpy as np
import aiohttp
from logger import logging
from pydantic import BaseModel, Field
from api_client import get_embeddings

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
    embedding_model: str = Field(default='nomic-embed-text', description="Model to use for embeddings")

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

    async def rerank_memories(self, query: str, memories: List[Tuple[str, float]], threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Rerank memories using vector similarity.
        
        Args:
            query: The query text to compare against
            memories: List of (memory_text, score) tuples from initial search
            threshold: Minimum cosine similarity to include in results
            
        Returns:
            List of (memory_text, new_score) tuples reranked by vector similarity
        """
        self.logger.info(f"Starting reranking for query: {query[:100]}... with {len(memories)} candidate memories")
        
        query_embedding = await self._get_embedding(query)
        if query_embedding is None:
            self.logger.error("Failed to generate query embedding")
            return []
            
        self.logger.info(f"Generated query embedding with shape: {query_embedding.shape}")
        
        reranked = []
        for memory_text, initial_score in memories:
            self.logger.debug(f"Processing memory (initial score {initial_score:.3f}): {memory_text[:100]}...")
            memory_embedding = await self._get_embedding(memory_text)
            
            if memory_embedding is not None:
                similarity = np.dot(query_embedding, memory_embedding)
                self.logger.debug(f"Memory embedding shape: {memory_embedding.shape}, Cosine similarity: {similarity:.3f}")
                
                if similarity >= threshold:
                    reranked.append((memory_text, float(similarity)))
                    self.logger.info(f"Memory accepted - Initial: {initial_score:.3f}, New: {similarity:.3f}")
                else:
                    self.logger.debug(f"Memory rejected - similarity {similarity:.3f} below threshold {threshold}")
            else:
                self.logger.warning(f"Failed to generate embedding for memory: {memory_text[:100]}...")
                    
        # Sort by new similarity scores
        reranked.sort(key=lambda x: x[1], reverse=True)
        self.logger.info(f"Reranking complete - {len(reranked)}/{len(memories)} memories above threshold")
        return reranked 