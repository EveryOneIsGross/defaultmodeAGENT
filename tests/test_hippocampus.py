import unittest
import asyncio
import numpy as np
from agent.hippocampus import Hippocampus, HippocampusConfig, EmbeddingConfig

class TestHippocampus(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = HippocampusConfig()
        self.hippocampus = Hippocampus(self.config)

    async def test_embedding_dimensions(self):
        text = "Test embedding dimensions"
        embedding = await self.hippocampus._get_embedding(text)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape[0], self.hippocampus.embedding_config.dimensions)
        self.assertTrue(np.isclose(np.linalg.norm(embedding), 1.0))

    async def test_embedding_cache(self):
        text = "Test caching behavior"
        # First call should compute embedding
        embedding1 = await self.hippocampus._get_embedding(text)
        # Second call should return cached value
        embedding2 = await self.hippocampus._get_embedding(text)
        self.assertTrue(np.array_equal(embedding1, embedding2))

    async def test_vector_similarity(self):
        text1 = "The cat sat on the mat"
        text2 = "A feline rested on a rug"
        text3 = "Python is a programming language"
        
        emb1 = await self.hippocampus._get_embedding(text1)
        emb2 = await self.hippocampus._get_embedding(text2)
        emb3 = await self.hippocampus._get_embedding(text3)
        
        # Similar texts should have higher cosine similarity
        sim_1_2 = np.dot(emb1, emb2)
        sim_1_3 = np.dot(emb1, emb3)
        self.assertGreater(sim_1_2, sim_1_3)  # semantically similar texts should be closer

if __name__ == '__main__':
    unittest.main() 