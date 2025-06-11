# for webscraping
import re, html, unicodedata, logging, asyncio, aiohttp
from typing import TypedDict
from bs4 import BeautifulSoup
from yt_dlp import YoutubeDL

from .chronpression import chronomic_filter

# Content limits (in characters)
MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 500
MAX_CONTENT_LENGTH = 50000  # ~12,500 tokens
MAX_TRANSCRIPT_LENGTH = 30000  # ~7,500 tokens for YouTube
MAX_HTML_PREVIEW = 1000  # For error fallback
PRESERVE_RATIO = 0.4  # Preserve 40% at start and 40% at end (80% total)

# YouTube URL patterns
YOUTUBE_PATTERNS = [
    r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})'
]

def is_youtube_url(url: str) -> tuple[bool, str]:
    """Check if URL is YouTube and extract video ID"""
    for pattern in YOUTUBE_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return True, match.group(1)
    return False, None

async def scrape_webpage(url: str) -> dict:
    """
    Robust, single-entry async scraper that always returns the same keys:
    {url, title, description, content, content_type, error_info}
    """
    logging.info(f"Starting to scrape URL: {url}")
    
    class Result(TypedDict):
        url: str
        title: str
        description: str
        content: str
        content_type: str
        error_info: dict

    # Helper functions
    def truncate_middle(text: str, max_length: int, preserve_ratio: float = PRESERVE_RATIO) -> str:
        """Truncate text from the middle, preserving start and end."""
        if not text or len(text) <= max_length:
            return text
        
        preserve_chars = int(max_length * preserve_ratio)
        start_chars = preserve_chars // 2
        end_chars = preserve_chars // 2
        
        start_break = start_chars
        end_break = len(text) - end_chars
        
        # Find sentence boundaries
        for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n']:
            pos = text.find(punct, start_chars, min(start_chars + 500, len(text)))
            if pos != -1:
                start_break = pos + len(punct)
                break
        
        truncation_notice = "\n\n[... content truncated ...]\n\n"
        return text[:start_break].rstrip() + truncation_notice + text[end_break:].lstrip()

    def truncate_field(text: str, max_length: int) -> str:
        """Simple truncation for short fields."""
        if not text or len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."

    def partial_result(error_code: str, error_msg: str, phase: str, **kwargs) -> Result:
        """Return partial results with error info"""
        logging.warning(f"Error in {phase} for {url}: {error_msg}")
        return Result(
            url=url,
            title=truncate_field(kwargs.get('title', url), MAX_TITLE_LENGTH),
            description=truncate_field(kwargs.get('description', f"Error: {error_msg}"), MAX_DESCRIPTION_LENGTH),
            content=kwargs.get('content', ''),
            content_type=kwargs.get('content_type', "error"),
            error_info={
                'code': error_code,
                'message': error_msg,
                'phase': phase
            }
        )

    # Check if YouTube URL
    is_yt, video_id = is_youtube_url(url)
    
    if is_yt:
        logging.info(f"Detected YouTube video ID: {video_id}")
        
        # Use yt-dlp for YouTube processing
        async def scrape_youtube() -> dict:
            """Extract YouTube video metadata and transcript using yt-dlp."""
            try:
                loop = asyncio.get_event_loop()
                
                # Get video info + transcript using yt-dlp
                def get_video_data():
                    ydl_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': False,
                        'skip_download': True,
                        'writesubtitles': True,
                        'writeautomaticsub': True,
                        'subtitleslangs': ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'],
                        'subtitlesformat': 'vtt'
                    }
                    
                    with YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        
                        # Extract transcript from available subtitles
                        transcript_text = ""
                        
                        # First try manual subtitles, then automatic
                        subtitle_sources = [
                            info.get('subtitles', {}),
                            info.get('automatic_captions', {})
                        ]
                        
                        for subtitle_dict in subtitle_sources:
                            if transcript_text:  # Already found subtitles
                                break
                                
                            # Try different language codes
                            for lang in ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']:
                                if lang in subtitle_dict:
                                    subtitle_formats = subtitle_dict[lang]
                                    
                                    # Find a VTT format
                                    for sub_format in subtitle_formats:
                                        if sub_format.get('ext') == 'vtt' and 'url' in sub_format:
                                            try:
                                                # Download the subtitle content
                                                import urllib.request
                                                with urllib.request.urlopen(sub_format['url']) as response:
                                                    vtt_content = response.read().decode('utf-8')
                                                    
                                                # Parse VTT content to extract just the text
                                                lines = vtt_content.split('\n')
                                                text_lines = []
                                                
                                                for line in lines:
                                                    line = line.strip()
                                                    # Skip VTT metadata, timestamps, and empty lines
                                                    if (line and 
                                                        not line.startswith('WEBVTT') and 
                                                        '-->' not in line and 
                                                        not line.startswith('NOTE') and
                                                        not line.isdigit()):
                                                        # Remove VTT formatting tags
                                                        clean_line = re.sub(r'<[^>]+>', '', line)
                                                        if clean_line and clean_line not in text_lines:
                                                            text_lines.append(clean_line)
                                                
                                                transcript_text = ' '.join(text_lines)
                                                break
                                                
                                            except Exception as e:
                                                logging.warning(f"Failed to download subtitle from {sub_format['url']}: {e}")
                                                continue
                                    
                                    if transcript_text:
                                        break
                        
                        return {
                            'title': info.get('title', 'Unknown Title'),
                            'channel': info.get('channel', info.get('uploader', 'Unknown Channel')),
                            'duration': info.get('duration', 0),
                            'views': info.get('view_count', 0),
                            'description': info.get('description', ''),
                            'upload_date': info.get('upload_date', ''),
                            'video_id': info.get('id', video_id),
                            'transcript': transcript_text
                        }
                
                video_data = await loop.run_in_executor(None, get_video_data)
                logging.info(f"Successfully retrieved video data: {video_data['title']}")
                
                # Apply chronomic filtering to transcript if available
                transcript_text = video_data['transcript']
                transcript_error = None
                
                if transcript_text:
                    try:
                        transcript_text = chronomic_filter(
                            transcript_text,
                            alpha=0.3,
                            beta=3.0
                        )
                    except Exception as e:
                        transcript_error = {
                            'code': 'CHRONOMIC_ERROR',
                            'message': str(e),
                            'phase': 'filtering'
                        }
                        logging.warning(f"Chronomic filtering error: {e}")
                else:
                    transcript_text = "[Transcript not available]"
                
                # Format content
                content_parts = [
                    f"# {truncate_field(video_data['title'], MAX_TITLE_LENGTH)}",
                    "",
                    f"**Channel**: {video_data['channel']}",
                    f"**Duration**: {video_data['duration']//60}:{video_data['duration']%60:02d}" if video_data['duration'] else "**Duration**: Unknown",
                    f"**Views**: {video_data['views']:,}" if video_data['views'] else "**Views**: Unknown",
                    f"**URL**: {url}",
                    "",
                    "## Description",
                    truncate_middle(video_data['description'] or 'No description', 5000),
                    "",
                    "## Transcript",
                    truncate_middle(transcript_text, MAX_TRANSCRIPT_LENGTH)
                ]
                
                content = "\n".join(content_parts)
                
                return Result(
                    url=url,
                    title=truncate_field(video_data['title'], MAX_TITLE_LENGTH),
                    description=truncate_field(
                        f"YouTube video by {video_data['channel']}" + 
                        (f" - {video_data['duration']//60}m" if video_data['duration'] else ""),
                        MAX_DESCRIPTION_LENGTH
                    ),
                    content=content,
                    content_type='youtube',
                    error_info=transcript_error or {}
                )
                
            except Exception as e:
                logging.exception(f"YouTube scraping failed")
                return partial_result('YT_ERROR', str(e), 'youtube', title=f"Video: {video_id}")
        
        return await scrape_youtube()

    # Regular HTML scraping for non-YouTube URLs
    BAD_UI = re.compile(
        r"(cookie|accept|subscribe|sign[\s-]?up|sign[\s-]?in|login|password|username|"
        r"newsletter|privacy\s+policy|terms\s+of\s+service|copyright)",
        flags=re.I,
    )

    def is_sig(block: str) -> bool:
        block = block.strip()
        if len(block) < 30 or BAD_UI.search(block):
            return False
        if not any(c in block for c in ".!?"):
            return False
        spec_ratio = sum(1 for c in block if not (c.isalnum() or c.isspace())) / len(block)
        if spec_ratio > 0.33:
            return False
        w = block.lower().split()
        return not (len(w) > 5 and len(set(w)) / len(w) < 0.5)

    def clean(txt: str) -> str:
        txt = html.unescape(txt)
        txt = unicodedata.normalize("NFKC", txt)
        return re.sub(r"\s+", " ", txt).strip()

    # Fetch HTML
    try:
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=hdrs, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status >= 400:
                    return partial_result(f'HTTP_{r.status}', f"HTTP {r.status}", 'fetch')
                
                html_text = await r.text()
                
        # Parse HTML
        soup = BeautifulSoup(html_text, "html.parser")
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'form']):
            element.decompose()
        
        # Extract content
        main_content = None
        for selector in ['article', 'main', '[role="main"]', '.content', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if main_content:
            content = clean(main_content.get_text(" ", strip=True))
        else:
            # Fallback to paragraph extraction
            blocks = []
            for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = elem.get_text(" ", strip=True)
                if is_sig(text) and text not in blocks:
                    blocks.append(text)
            content = "\n\n".join(blocks)
        
        if not content:
            return partial_result('NO_CONTENT', 'No content found', 'extraction')

        # Apply chronomic filtering to web content
        try:
            content = chronomic_filter(content, alpha=0.3, beta=3.0)
        except Exception as e:
            logging.warning(f"Chronomic filtering failed for web content: {e}")
            # Continue with unfiltered content
        
        title = soup.title.string if soup.title else url
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc.get("content", "") if meta_desc else ""
        
        return Result(
            url=url,
            title=truncate_field(title.strip(), MAX_TITLE_LENGTH),
            description=truncate_field(description.strip(), MAX_DESCRIPTION_LENGTH),
            content=truncate_middle(content, MAX_CONTENT_LENGTH),
            content_type="text/html",
            error_info={}
        )
        
    except Exception as e:
        logging.exception(f"Scraping failed for {url}")
        return partial_result('SCRAPE_ERROR', str(e), 'general')