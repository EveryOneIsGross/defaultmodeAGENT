import pytest
from agent.hypocampus import Hypocampus, HypocampusConfig

@pytest.mark.asyncio
async def test_default_hypocampus_workflow():
    # Initialize with default settings
    config = HypocampusConfig()
    hypocampus = Hypocampus(config)
    
    # Test texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A swift canine leaps across a sleeping hound"
    ]
    
    # Get embeddings for both texts
    embeddings = []
    for text in texts:
        embedding = await hypocampus._get_embedding(text)
        assert embedding is not None
        embeddings.append(embedding)
    
    # Verify we got valid embeddings for both texts
    assert len(embeddings) == 2
    assert all(e is not None for e in embeddings) 