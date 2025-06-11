import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_youtube_scrape():
    # Test different YouTube URL formats
    test_urls = [
        "https://www.youtube.com/watch?v=uqRF4IszorU",
        "https://youtu.be/uqRF4IszorU",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll for testing
        "https://www.google.com",  # Non-YouTube URL
    ]
    
    # First, check what's installed
    print("Checking YouTube library availability...")
    try:
        import pytube
        print(f"✓ pytube version: {pytube.__version__}")
    except ImportError as e:
        print(f"✗ pytube not installed: {e}")
    
    try:
        import youtube_transcript_api
        print(f"✓ youtube_transcript_api installed")
    except ImportError as e:
        print(f"✗ youtube_transcript_api not installed: {e}")
    
    print("\nTesting URL scraping...")
    
    # Import the scraper
    from agent.tools.webSCRAPE import scrape_webpage
    
    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"Testing: {url}")
        print(f"{'='*60}")
        
        result = await scrape_webpage(url)
        
        print(f"Content Type: {result['content_type']}")
        print(f"Title: {result['title'][:100]}...")
        print(f"Description: {result['description'][:100]}...")
        print(f"Content Length: {len(result['content'])} chars")
        print(f"Error Info: {result['error_info']}")
        
        if result['content_type'] == 'youtube':
            print("\nFirst 500 chars of content:")
            print(result['content'][:500])

if __name__ == "__main__":
    asyncio.run(test_youtube_scrape())