import pytest
from agent.hippocampus import Hippocampus, HippocampusConfig

@pytest.mark.asyncio
async def test_default_hippocampus_workflow():
    # Initialize with default settings
    config = HippocampusConfig()
    hippocampus = Hippocampus(config)
    
    # Test texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A swift canine leaps across a sleeping hound"
    ]
    
    # Get embeddings for both texts
    embeddings = []
    for text in texts:
        embedding = await hippocampus._get_embedding(text)
        assert embedding is not None
        embeddings.append(embedding)
    
    # Verify we got valid embeddings for both texts
    assert len(embeddings) == 2
    assert all(e is not None for e in embeddings) 