"""
A single-entry async scraper.
It guarantees the same six keys in the returned dict:

    {url, title, description, content, content_type, error_info}

Where:
* `content_type` is **never** "error". It is one of:
    - "text/html"      -> normal extraction
    - "youtube"        -> YouTube with transcript/metadata
    - "twitter"        -> Twitter/X.com tweet
    - "html_preview"   -> raw HTML preview when structured extraction failed
    - "none"           -> nothing could be fetched (e.g. HTTP 404)
* `error_info` is always an **empty dict** so downstream clients never see
  scraper internals.

The module still logs everything (`logging.warning`/`logging.exception`) to aid
server-side debugging.
"""

import re
import html
import unicodedata
import logging
import asyncio
import aiohttp
from typing import TypedDict
from bs4 import BeautifulSoup
from yt_dlp import YoutubeDL
from concurrent.futures import ThreadPoolExecutor

from .chronpression import chronomic_filter
from tokenizer import count_tokens, encode_text, decode_tokens

# Optional Playwright import for JS-heavy sites
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.info("Playwright not installed - JS rendering unavailable. Install with: pip install playwright && playwright install chromium")

MAX_TITLE_TOKENS = 50
MAX_DESCRIPTION_TOKENS = 125
MAX_CONTENT_TOKENS = 12500
MAX_TRANSCRIPT_TOKENS = 15000
MAX_HTML_PREVIEW_TOKENS = 2500
PRESERVE_RATIO = 0.4
CHUNK_TARGET_CHARS = 8000
MAX_PARALLEL_CHUNKS = 8
CHARS_PER_TOKEN_ESTIMATE = 4
MIN_COMPRESSION = 0.3
MAX_COMPRESSION = 0.92
MIN_CONTENT_LENGTH = 200  # Minimum chars to consider content valid

_compress_executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS)

# yt-dlp supported patterns (these use a different extraction method entirely)
YOUTUBE_PATTERNS = [
    r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
]

TWITTER_PATTERNS = [
    r"(?:https?://)?(?:www\.)?(twitter\.com|x\.com)/\w+/status/(\d+)",
]

def is_youtube_url(url: str) -> tuple[bool, str | None]:
    for pattern in YOUTUBE_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return True, match.group(1)
    return False, None

def is_twitter_url(url: str) -> tuple[bool, str | None]:
    for pattern in TWITTER_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return True, match.group(2)
    return False, None

def needs_playwright_fallback(html_text: str, extracted_content: str) -> bool:
    """
    Detect if we need to retry with Playwright based on response content.
    Returns True if the page appears to require JavaScript rendering.
    """
    if not html_text:
        return True

    lower = html_text.lower()

    # Check for explicit JS-required messages
    js_indicators = [
        "requires javascript",
        "enable javascript",
        "javascript is required",
        "please enable javascript",
        "needs javascript",
        "javascript must be enabled",
        "this site requires javascript",
        "you need to enable javascript",
    ]
    if any(indicator in lower for indicator in js_indicators):
        return True

    # Check for empty/minimal content after extraction
    if len(extracted_content.strip()) < MIN_CONTENT_LENGTH:
        # Page loaded but no real content - likely JS-rendered
        return True

    return False

def _sync_playwright_fetch(url: str, timeout: int = 20) -> str | None:
    """Synchronous Playwright fetch - runs in thread to avoid blocking event loop."""
    if not PLAYWRIGHT_AVAILABLE:
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                )
            )
            page = context.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                # Smart wait: check for content rather than fixed delay
                try:
                    page.wait_for_function(
                        "document.body && document.body.innerText.length > 500",
                        timeout=3000
                    )
                except:
                    page.wait_for_timeout(1500)
                content = page.content()
            finally:
                browser.close()

            return content
    except Exception as e:
        logging.warning("Playwright fetch failed for %s: %s", url, e)
        return None

async def fetch_with_playwright(url: str, timeout: int = 20) -> str | None:
    """Fetch page content using Playwright - runs in executor to avoid blocking Discord."""
    if not PLAYWRIGHT_AVAILABLE:
        return None

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _sync_playwright_fetch, url, timeout)
    except Exception as e:
        logging.warning("Playwright executor failed for %s: %s", url, e)
        return None

async def fetch_html(url: str) -> tuple[str | None, str | None]:
    """
    Fetch HTML content. Returns (html_text, error_description).
    Tries aiohttp first, falls back to Playwright on failure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
    }

    # Try aiohttp first (fast path)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 403:
                    logging.info("HTTP 403 - site blocking bots: %s", url)
                    # Fallback to Playwright
                    if PLAYWRIGHT_AVAILABLE:
                        html_text = await fetch_with_playwright(url)
                        return html_text, None if html_text else "HTTP 403 - blocked"
                    return None, "HTTP 403 - site blocks bots"

                if resp.status >= 400:
                    return None, f"HTTP {resp.status}"

                return await resp.text(), None

    except aiohttp.ClientResponseError as e:
        if "Header value is too long" in str(e):
            logging.info("Oversized headers, trying Playwright: %s", url)
            if PLAYWRIGHT_AVAILABLE:
                html_text = await fetch_with_playwright(url)
                return html_text, None if html_text else "Site has oversized headers"
            return None, "Site has oversized headers"
        raise

async def download_image(image_url: str, cache, user_id: str) -> str | None:
    """
    Download an image and save it to the cache.
    Returns the local file path or None if download failed.
    """
    if not cache or not user_id:
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    logging.warning("Failed to download image %s: HTTP %s", image_url, resp.status)
                    return None

                content_type = resp.headers.get("Content-Type", "")
                # Determine file extension
                if "png" in content_type:
                    suffix = ".png"
                elif "gif" in content_type:
                    suffix = ".gif"
                elif "webp" in content_type:
                    suffix = ".webp"
                else:
                    suffix = ".jpg"

                image_bytes = await resp.read()
                if len(image_bytes) < 100:  # Too small, probably an error
                    return None

                file_path, _ = cache.create_temp_file(
                    user_id,
                    prefix="img",
                    suffix=suffix,
                    content=image_bytes
                )
                logging.info("Downloaded image to %s (%d bytes)", file_path, len(image_bytes))
                return file_path

    except Exception as e:
        logging.warning("Failed to download image %s: %s", image_url, e)
        return None

async def download_images(image_urls: list[str], cache, user_id: str, max_images: int = 4) -> list[str]:
    """Download multiple images in parallel, return list of local file paths."""
    if not cache or not user_id or not image_urls:
        return []

    tasks = [download_image(url, cache, user_id) for url in image_urls[:max_images]]
    results = await asyncio.gather(*tasks)
    return [path for path in results if path]

def get_compression_for_target(raw_tokens: int, target_tokens: int) -> float:
    if raw_tokens <= target_tokens:
        return MIN_COMPRESSION
    needed_reduction = 1.0 - (target_tokens / raw_tokens)
    compression = needed_reduction + 0.15
    return min(MAX_COMPRESSION, max(MIN_COMPRESSION, compression))

def transcript_quality(text: str) -> float:
    words = text.split()
    if len(words) < 50:
        return 0.0
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if unique_ratio < 0.3 or avg_word_len < 3.0:
        return 0.5
    return 1.0

def chunk_text_by_chars(text: str, target_chars: int = CHUNK_TARGET_CHARS) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_chars = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_chars = len(sentence)
        if current_chars + sent_chars > target_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chars = sent_chars
        else:
            current_chunk.append(sentence)
            current_chars += sent_chars
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def compress_chunk(chunk: str, compression: float, fuzzy_strength: float) -> str:
    try:
        return chronomic_filter(chunk, compression=compression, fuzzy_strength=fuzzy_strength, horizon=6)
    except Exception:
        logging.warning("Chronomic filtering failed on chunk")
        return chunk

def sync_compress_all(text: str, compression: float, fuzzy_strength: float) -> str:
    if len(text) < CHUNK_TARGET_CHARS * 1.5:
        try:
            return chronomic_filter(text, compression=compression, fuzzy_strength=fuzzy_strength, horizon=6)
        except Exception:
            logging.warning("Chronomic filtering failed")
            return text
    chunks = chunk_text_by_chars(text, CHUNK_TARGET_CHARS)
    if len(chunks) == 1:
        try:
            return chronomic_filter(chunks[0], compression=compression, fuzzy_strength=fuzzy_strength, horizon=6)
        except Exception:
            return text
    compressed = []
    for chunk in chunks:
        compressed.append(compress_chunk(chunk, compression, fuzzy_strength))
    return ' '.join(c for c in compressed if c)

async def parallel_compress(text: str, compression: float = 0.5, fuzzy_strength: float = 1.0) -> tuple[str, bool]:
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _compress_executor,
            sync_compress_all,
            text,
            compression,
            fuzzy_strength
        )
        return result, True
    except Exception:
        logging.warning("Parallel compression failed")
        return text, False

async def scrape_webpage(url: str, cache=None, user_id=None) -> dict:
    logging.info("Starting to scrape URL: %s", url)

    class Result(TypedDict):
        url: str
        title: str
        description: str
        content: str
        content_type: str
        error_info: dict
        image_paths: list[str]  # Local paths to downloaded images

    def truncate_middle_tokens(text: str, max_tokens: int, preserve_ratio: float = PRESERVE_RATIO) -> str:
        if not text:
            return text
        current_tokens = count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        preserve_tokens = int(max_tokens * preserve_ratio)
        head_tokens = preserve_tokens // 2
        tail_tokens = preserve_tokens // 2
        tokens = encode_text(text)
        head_end = head_tokens
        try:
            search_start = max(0, head_tokens - 50)
            search_end = min(len(tokens), head_tokens + 50)
            for i in range(search_end - 1, search_start - 1, -1):
                token_text = decode_tokens([tokens[i]])
                if any(punct in token_text for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']):
                    head_end = i + 1
                    break
        except:
            pass
        head_tokens_actual = tokens[:head_end]
        tail_tokens_actual = tokens[-tail_tokens:] if tail_tokens > 0 else []
        head_text = decode_tokens(head_tokens_actual).rstrip()
        tail_text = decode_tokens(tail_tokens_actual).lstrip() if tail_tokens_actual else ""
        return head_text + "\n\n[...content truncated...]\n\n" + tail_text

    def truncate_field_tokens(text: str, max_tokens: int) -> str:
        if not text:
            return text
        current_tokens = count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        tokens = encode_text(text)
        truncated_tokens = tokens[:max_tokens - 1]
        return decode_tokens(truncated_tokens) + "..."

    def clean(txt: str) -> str:
        txt = html.unescape(txt)
        txt = unicodedata.normalize("NFKC", txt)
        return re.sub(r"\s+", " ", txt).strip()

    def partial_result(*, title: str | None = None, description: str | None = None, content: str = "", image_paths: list[str] | None = None) -> Result:
        return Result(
            url=url,
            title=truncate_field_tokens(title or url, MAX_TITLE_TOKENS),
            description=truncate_field_tokens(description or "", MAX_DESCRIPTION_TOKENS),
            content=content,
            content_type="html_preview" if content else "none",
            error_info={},
            image_paths=image_paths or [],
        )

    def extract_content(soup: BeautifulSoup) -> str:
        """Extract main content from parsed HTML."""
        # Remove junk
        for tag in soup(["script", "style", "nav", "header", "footer", "iframe", "form"]):
            tag.decompose()

        # Try semantic selectors
        for sel in ("article", "main", '[role="main"]', ".content", "#content"):
            main = soup.select_one(sel)
            if main:
                return clean(main.get_text(" ", strip=True))

        # Fallback: collect significant paragraphs and headings
        BAD_UI = re.compile(
            r"(cookie|accept|subscribe|sign[\s-]?up|sign[\s-]?in|login|password|username|newsletter|privacy\s+policy|terms\s+of\s+service|copyright)",
            flags=re.I,
        )

        def is_sig(block: str) -> bool:
            block = block.strip()
            if len(block) < 30 or BAD_UI.search(block):
                return False
            if not any(p in block for p in ".!?"):
                return False
            spec_ratio = sum(1 for c in block if not (c.isalnum() or c.isspace())) / len(block)
            if spec_ratio > 0.33:
                return False
            words = block.lower().split()
            return not (len(words) > 5 and len(set(words)) / len(words) < 0.5)

        blocks = []
        for elem in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
            txt = elem.get_text(" ", strip=True)
            if is_sig(txt) and txt not in blocks:
                blocks.append(txt)
        return "\n\n".join(blocks)

    # === yt-dlp handlers (YouTube, Twitter) ===

    is_yt, video_id = is_youtube_url(url)
    if is_yt:
        logging.info("Detected YouTube video ID: %s", video_id)

        async def fetch_vtt_segment(session: aiohttp.ClientSession, segment_url: str) -> str:
            try:
                async with session.get(segment_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    return ""
            except Exception:
                return ""

        async def scrape_youtube() -> Result:
            try:
                loop = asyncio.get_event_loop()

                def get_video_data():
                    opts = {
                        "quiet": True,
                        "no_warnings": True,
                        "extract_flat": False,
                        "skip_download": True,
                        "writesubtitles": True,
                        "writeautomaticsub": True,
                        "subtitleslangs": ["en", "en-US", "en-GB", "en-CA", "en-AU"],
                        "subtitlesformat": "vtt",
                    }
                    if cache and user_id:
                        opts["paths"] = {"temp": cache.get_user_temp_dir(user_id)}
                    with YoutubeDL(opts) as ydl:
                        return ydl.extract_info(url, download=False)

                info = await loop.run_in_executor(None, get_video_data)

                transcript = ""
                async with aiohttp.ClientSession() as session:
                    for source in (info.get("subtitles", {}), info.get("automatic_captions", {})):
                        if transcript:
                            break
                        for lang in ("en", "en-US", "en-GB", "en-CA", "en-AU"):
                            if lang not in source:
                                continue
                            for fmt in source[lang]:
                                if fmt.get("ext") != "vtt" or "url" not in fmt:
                                    continue
                                try:
                                    async with session.get(fmt["url"], timeout=aiohttp.ClientTimeout(total=10)) as resp:
                                        if resp.status != 200:
                                            continue
                                        content = await resp.text()
                                    if content.startswith("#EXTM3U") or "#EXT-X-VERSION" in content:
                                        segment_urls = [
                                            line.strip()
                                            for line in content.split("\n")
                                            if line.strip() and not line.startswith("#") and "youtube.com" in line
                                        ]
                                        segments = await asyncio.gather(*[
                                            fetch_vtt_segment(session, seg_url) for seg_url in segment_urls
                                        ])
                                        vtt = "\n".join(seg for seg in segments if seg)
                                    else:
                                        vtt = content
                                    lines = [
                                        re.sub(r"<[^>]+>", "", ln.strip())
                                        for ln in vtt.split("\n")
                                        if ln.strip()
                                        and not ln.startswith(("WEBVTT", "NOTE", "Kind:", "Language:"))
                                        and "-->" not in ln
                                        and not ln.strip().isdigit()
                                    ]
                                    transcript = " ".join(dict.fromkeys(lines))
                                    break
                                except Exception:
                                    continue
                            if transcript:
                                break

                if transcript:
                    raw_tokens = len(transcript) // CHARS_PER_TOKEN_ESTIMATE
                    logging.info("Transcript estimated tokens: %d", raw_tokens)
                    quality = transcript_quality(transcript)
                    compression = get_compression_for_target(raw_tokens, MAX_TRANSCRIPT_TOKENS)
                    if quality < 0.7:
                        compression = min(compression + 0.05, MAX_COMPRESSION)
                    transcript, _ = await parallel_compress(transcript, compression=compression, fuzzy_strength=1.0)
                else:
                    transcript = "[Transcript not available]"

                # Get best thumbnail URL
                thumbnail_url = info.get("thumbnail") or ""
                if not thumbnail_url and info.get("thumbnails"):
                    thumbs = sorted(info["thumbnails"], key=lambda x: x.get("width", 0), reverse=True)
                    thumbnail_url = thumbs[0].get("url", "") if thumbs else ""

                # Download thumbnail if cache available
                thumbnail_path = None
                if thumbnail_url and cache and user_id:
                    thumbnail_path = await download_image(thumbnail_url, cache, user_id)

                # Build content - let content dictate metadata
                title = info.get('title', 'Untitled')
                channel = info.get('channel') or info.get('uploader', 'Unknown')
                description = info.get("description") or ""

                content_parts = []
                if description:
                    content_parts.append(truncate_middle_tokens(description, 1250))
                if transcript and transcript != "[Transcript not available]":
                    content_parts.append(f"\n\n---\nTranscript:\n{truncate_middle_tokens(transcript, MAX_TRANSCRIPT_TOKENS)}")
                if thumbnail_path:
                    content_parts.append(f"\n\n[Thumbnail: {thumbnail_path}]")
                elif thumbnail_url:
                    content_parts.append(f"\n\n[Thumbnail URL: {thumbnail_url}]")

                full_content = "".join(content_parts) if content_parts else "[No content available]"

                return Result(
                    url=url,
                    title=truncate_field_tokens(title, MAX_TITLE_TOKENS),
                    description=truncate_field_tokens(channel, MAX_DESCRIPTION_TOKENS),
                    content=full_content,
                    content_type="youtube",
                    error_info={},
                    image_paths=[thumbnail_path] if thumbnail_path else [],
                )
            except Exception:
                logging.exception("YouTube scraping failed")
                return partial_result(title=f"Video: {video_id}")

        return await scrape_youtube()

    is_tweet, tweet_id = is_twitter_url(url)
    if is_tweet:
        logging.info("Detected Twitter/X.com tweet ID: %s", tweet_id)

        def extract_urls_from_text(text: str) -> list[str]:
            """Extract URLs from text, excluding twitter/x.com links."""
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, text)
            # Filter out twitter/x.com URLs and t.co shortlinks
            return [u for u in urls if not re.search(r'(twitter\.com|x\.com|t\.co)/', u)]

        async def scrape_linked_content(urls: list[str]) -> str:
            """Scrape and compress content from linked URLs."""
            if not urls:
                return ""

            linked_parts = []
            for link_url in urls[:2]:  # Limit to 2 links
                try:
                    logging.info("Following link from tweet: %s", link_url)
                    link_html, error = await fetch_html(link_url)
                    if link_html and not error:
                        soup = BeautifulSoup(link_html, "html.parser")
                        content = extract_content(soup)
                        if needs_playwright_fallback(link_html, content) and PLAYWRIGHT_AVAILABLE:
                            pw_html = await fetch_with_playwright(link_url)
                            if pw_html:
                                soup = BeautifulSoup(pw_html, "html.parser")
                                pw_content = extract_content(soup)
                                if len(pw_content) > len(content):
                                    content = pw_content

                        if content and len(content) > 100:
                            raw_tokens = len(content) // CHARS_PER_TOKEN_ESTIMATE
                            compression = get_compression_for_target(raw_tokens, 2000)
                            compressed, _ = await parallel_compress(content, compression=compression, fuzzy_strength=1.0)
                            title = soup.title.string.strip() if soup.title and soup.title.string else link_url
                            linked_parts.append(f"### Linked: {title}\n{truncate_middle_tokens(compressed, 2000)}")
                except Exception as e:
                    logging.warning("Failed to scrape linked URL %s: %s", link_url, e)

            return "\n\n".join(linked_parts)

        async def scrape_twitter() -> Result:
            tweet_text = ""
            author = "Unknown"
            images = []

            # Use Playwright directly for Twitter (yt-dlp is for video, not tweets)
            if PLAYWRIGHT_AVAILABLE:
                logging.info("Fetching Twitter with Playwright: %s", url)
                try:
                    html_text = await fetch_with_playwright(url)
                    if html_text:
                        logging.info("Playwright got %d chars of HTML", len(html_text))
                        soup = BeautifulSoup(html_text, "html.parser")

                        # Extract author from tweet
                        author_elem = soup.select_one('[data-testid="User-Name"] a, article a[href*="/status/"]')
                        if author_elem:
                            href = author_elem.get("href", "")
                            if href and "/" in href:
                                author = href.split("/")[1] if href.startswith("/") else href.split("/")[3]

                        # Extract tweet text
                        for sel in ['[data-testid="tweetText"]', 'article [lang]', '[role="article"]']:
                            elem = soup.select_one(sel)
                            if elem:
                                tweet_text = clean(elem.get_text(" ", strip=True))
                                logging.info("Found tweet text with selector %s: %d chars", sel, len(tweet_text))
                                if len(tweet_text) > 20:
                                    break

                        # Extract images from tweet
                        for img in soup.select('[data-testid="tweetPhoto"] img, article img[src*="pbs.twimg.com"]'):
                            src = img.get("src", "")
                            if src and "pbs.twimg.com" in src and src not in images:
                                images.append(src)
                    else:
                        logging.warning("Playwright returned no HTML for Twitter")
                except Exception as e:
                    logging.warning("Playwright Twitter fetch error: %s", e)
            else:
                logging.warning("Playwright not available for Twitter scraping")

            if not tweet_text:
                return partial_result(description="Could not extract tweet - login may be required")

            # Download images if cache available
            image_paths = []
            if images and cache and user_id:
                image_paths = await download_images(images, cache, user_id, max_images=4)
                logging.info("Downloaded %d/%d images for tweet", len(image_paths), len(images))

            # Extract and follow links from tweet text
            external_urls = extract_urls_from_text(tweet_text)
            linked_content = await scrape_linked_content(external_urls)

            # Build content - let content dictate metadata
            content_parts = [tweet_text]

            if image_paths:
                content_parts.append(f"\n[Images: {', '.join(image_paths)}]")
            elif images:
                # Fallback to URLs if download failed
                content_parts.append(f"\n[Image URLs: {', '.join(images[:4])}]")

            if linked_content:
                content_parts.append(f"\n\n---\n{linked_content}")

            full_content = "".join(content_parts)

            # Derive metadata from content
            first_line = tweet_text.split('\n')[0][:100] if tweet_text else url

            return Result(
                url=url,
                title=truncate_field_tokens(first_line, MAX_TITLE_TOKENS),
                description=truncate_field_tokens(f"@{author}" if author != "Unknown" else "", MAX_DESCRIPTION_TOKENS),
                content=full_content,
                content_type="twitter",
                error_info={},
                image_paths=image_paths,
            )

        return await scrape_twitter()

    # === General HTML scraping ===

    try:
        # Step 1: Try fast aiohttp fetch
        html_text, error = await fetch_html(url)

        if error:
            logging.warning("%s for %s", error, url)
            return partial_result(description=error)

        if not html_text:
            return partial_result(description="Could not fetch page")

        # Step 2: Parse and extract content
        soup = BeautifulSoup(html_text, "html.parser")
        content = extract_content(soup)

        # Step 3: Check if we need Playwright fallback
        if needs_playwright_fallback(html_text, content) and PLAYWRIGHT_AVAILABLE:
            logging.info("Content insufficient, retrying with Playwright: %s", url)
            playwright_html = await fetch_with_playwright(url)
            if playwright_html:
                soup = BeautifulSoup(playwright_html, "html.parser")
                playwright_content = extract_content(soup)
                # Only use Playwright result if it's better
                if len(playwright_content) > len(content):
                    html_text = playwright_html
                    content = playwright_content

        # Step 4: Process content
        if not content:
            content = truncate_middle_tokens(clean(html_text), MAX_HTML_PREVIEW_TOKENS)
            ctype = "html_preview"
        else:
            ctype = "text/html"
            raw_tokens = len(content) // CHARS_PER_TOKEN_ESTIMATE
            compression = get_compression_for_target(raw_tokens, MAX_CONTENT_TOKENS)
            content, chronomic_succeeded = await parallel_compress(content, compression=compression, fuzzy_strength=1.0)
            if not chronomic_succeeded:
                content = truncate_middle_tokens(content, int(MAX_CONTENT_TOKENS * 0.6))

        title_tag = soup.title.string if soup.title else url
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc.get("content", "") if meta_desc else ""

        return Result(
            url=url,
            title=truncate_field_tokens(title_tag.strip(), MAX_TITLE_TOKENS),
            description=truncate_field_tokens(description.strip(), MAX_DESCRIPTION_TOKENS),
            content=truncate_middle_tokens(content, MAX_CONTENT_TOKENS),
            content_type=ctype,
            error_info={},
            image_paths=[],
        )

    except Exception:
        logging.exception("Scraping failed for %s", url)
        return partial_result(description="Unexpected scraping error")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape and parse websites to console")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--full", action="store_true", help="Show full content (default: preview only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    result = asyncio.run(scrape_webpage(args.url))

    print(f"URL:          {result['url']}")
    print(f"Title:        {result['title']}")
    print(f"Description:  {result['description']}")
    print(f"Content Type: {result['content_type']}")
    print(f"Tokens:       {count_tokens(result['content'])}")
    print("\n" + "=" * 80)

    if args.full:
        print(result['content'])
    else:
        preview = result['content'][:1000]
        print(preview)
        if len(result['content']) > 1000:
            print(f"\n... (use --full to see all {len(result['content'])} characters)")
