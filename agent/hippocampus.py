from typing import List, Tuple, Optional, Dict
import numpy as np
import aiohttp
from logger import logging
from pydantic import BaseModel, Field
from api_client import get_embeddings
import asyncio
from chunker import truncate_middle
from bot_config import HippocampusConfig, EmbeddingConfig


class Hippocampus:
    def __init__(self, config: HippocampusConfig, logger=None):
        self.config = config
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self.embedding_config = EmbeddingConfig()
        self.logger = logger or logging.getLogger("bot.default")

    async def _get_ollama_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embeddings specifically from Ollama API."""
        # Truncate text to fit model's context window with safety margin
        max_tokens = getattr(self.embedding_config, "max_embed_tokens", 160)
        truncated_text = truncate_middle(text, max_tokens=max_tokens)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.embedding_config.api_base}/api/embeddings",
                    json={
                        "model": self.embedding_config.model,
                        "prompt": truncated_text,
                        "options": {"temperature": 0.0, "num_ctx": 256},
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(
                            f"Ollama API returned status {response.status}: {error_text}"
                        )
                        raise Exception(
                            f"Ollama API returned status {response.status}: {error_text}"
                        )
                    result = await response.json()
                    embedding = np.array(result["embedding"])
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
                if self.config.embedding_provider == "ollama":
                    embedding = await self._get_ollama_embedding(text)
                else:
                    max_tokens = getattr(self.embedding_config, "max_embed_tokens", 256)
                    embedding = await get_embeddings(
                        text,
                        provider=self.config.embedding_provider,
                        model=self.config.embedding_model,
                        max_tokens=max_tokens,
                    )
                    embedding = np.array(embedding)

                if embedding is not None:
                    embedding = embedding / np.linalg.norm(embedding)
                    self._embedding_cache[text] = embedding
            except Exception as e:
                self.logger.error(f"Embedding generation failed: {str(e)}")
                return None
        return self._embedding_cache.get(text)

    async def _get_ollama_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for multiple texts in a single batch from Ollama API."""
        # Truncate texts to fit model's context window with safety margin
        max_tokens = getattr(self.embedding_config, "max_embed_tokens", 160)
        truncated_texts = [truncate_middle(text, max_tokens=max_tokens) for text in texts]

        async with aiohttp.ClientSession() as session:
            try:
                tasks = []
                for text in truncated_texts:
                    tasks.append(
                        session.post(
                            f"{self.embedding_config.api_base}/api/embeddings",
                            json={
                                "model": self.embedding_config.model,
                                "prompt": text,
                                "options": {"temperature": 0.0, "num_ctx": 256},
                            },
                        )
                    )
                responses = await asyncio.gather(*tasks)
                embeddings = []
                for response in responses:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(
                            f"Ollama API returned status {response.status}: {error_text}"
                        )
                        continue
                    result = await response.json()
                    embedding = np.array(result["embedding"])
                    if embedding.shape[0] != self.embedding_config.dimensions:
                        self.logger.error(
                            f"Unexpected embedding dimensions: {embedding.shape[0]}"
                        )
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

        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                embeddings[i] = self._embedding_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            try:
                if self.config.embedding_provider == "ollama":
                    new_embeddings = await self._get_ollama_embeddings_batch(uncached_texts)
                else:
                    max_tokens = getattr(self.embedding_config, "max_embed_tokens", 256)
                    new_embeddings = await get_embeddings(
                        uncached_texts,
                        provider=self.config.embedding_provider,
                        model=self.config.embedding_model,
                        max_tokens=max_tokens,
                    )
                    new_embeddings = np.array(new_embeddings)

                if new_embeddings is not None and len(new_embeddings) > 0:
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
        blend_factor: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank memories using batched vector similarity and blend with original scores."""
        self.logger.info(
            f"Starting batch reranking for query: {query[:100]}... with {len(memories)} candidates"
        )
        blend = self.config.blend_factor if blend_factor is None else blend_factor
        self.logger.info(
            f"Using blend factor: {blend:.2f} (initial:{blend:.2f}/embedding:{1 - blend:.2f})"
        )

        # Truncate memories to fit embedding model token limits with safety margin
        max_tokens = getattr(self.embedding_config, "max_embed_tokens", 160)
        memory_texts = [truncate_middle(str(m[0]), max_tokens=max_tokens) for m in memories]
        initial_scores = np.array([m[1] for m in memories])

        # Truncate query to fit embedding model token limits with safety margin
        truncated_query = truncate_middle(query, max_tokens=max_tokens)
        query_embedding = await self._get_embedding(truncated_query)
        if query_embedding is None:
            self.logger.error("Failed to generate query embedding")
            return []

        memory_embeddings = await self._get_embeddings_batch(memory_texts)
        cosine = np.dot(memory_embeddings, query_embedding)
        embedding_similarities = 0.5 * (cosine + 1.0)
        initial_scores = np.clip(initial_scores, 0.0, 1.0)
        combined_scores = np.clip(
            (blend * initial_scores) + ((1 - blend) * embedding_similarities), 0.0, 1.0
        )

        reranked = [(t, float(s)) for t, s in zip(memory_texts, combined_scores) if s >= threshold]
        reranked.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(
            f"Batch reranking complete – {len(reranked)}/{len(memories)} memories above threshold"
        )
        return reranked
