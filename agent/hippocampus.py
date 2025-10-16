from typing import List, Tuple, Optional, Dict
import numpy as np
import aiohttp
from logger import logging
from pydantic import BaseModel, Field
from api_client import get_embeddings
import asyncio

from bot_config import HippocampusConfig, EmbeddingConfig

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

    async def rerank_memories(
        self,
        query: str,
        memories: List[Tuple[str, float]],
        threshold: float = 0.6,
        blend_factor: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank memories using batched vector similarity and blend with original scores.
        All intermediate and final scores are normalised to the range [0.0, 1.0].
        """
        self.logger.info(f"Starting batch reranking for query: {query[:100]}... with {len(memories)} candidates")
        # 1 · blend factor
        blend = self.config.blend_factor if blend_factor is None else blend_factor
        self.logger.info(
            f"Using blend factor: {blend:.2f} (initial:{blend:.2f}/embedding:{1 - blend:.2f})"
        )
        # 2 · split memories into text + initial score
        memory_texts   = [m[0] for m in memories]
        initial_scores = np.array([m[1] for m in memories])
        # 3 · get embeddings
        query_embedding = await self._get_embedding(query)
        if query_embedding is None:
            self.logger.error("Failed to generate query embedding")
            return []

        memory_embeddings = await self._get_embeddings_batch(memory_texts)
        # 4 · cosine similarities  →  [-1, 1]  →  [0, 1]
        cosine = np.dot(memory_embeddings, query_embedding)          # raw cosine
        embedding_similarities = 0.5 * (cosine + 1.0)                # rescaled
        # 5 · clip lexical scores to [0, 1] just in case
        initial_scores = np.clip(initial_scores, 0.0, 1.0)
        # 6 · blended score  →  clip to [0, 1]
        combined_scores = (blend * initial_scores) + ((1 - blend) * embedding_similarities)
        combined_scores = np.clip(combined_scores, 0.0, 1.0)
        # 7 · debug logging
        for text, init_s, emb_s, comb_s in zip(
            memory_texts, initial_scores, embedding_similarities, combined_scores
        ):
            self.logger.debug(
                f"Memory score: initial={init_s:.4f}, embedding={emb_s:.4f}, "
                f"combined={comb_s:.4f} for {text[:50]}..."
            )
        # 8 · filter + sort
        reranked = [
            (text, float(score))
            for text, score in zip(memory_texts, combined_scores)
            if score >= threshold
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(
            f"Batch reranking complete – {len(reranked)}/{len(memories)} memories above threshold"
        )
        return reranked

