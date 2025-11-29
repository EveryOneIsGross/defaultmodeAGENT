"""
A single-entry async scraper.
It guarantees the same six keys in the returned dict:

    {url, title, description, content, content_type, error_info}

Where:
* `content_type` is **never** "error". It is one of:
    - "text/html"      -> normal extraction
    - "youtube"        -> YouTube with transcript/metadata
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

_compress_executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS)

YOUTUBE_PATTERNS = [
    r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
]

def is_youtube_url(url: str) -> tuple[bool, str | None]:
    for pattern in YOUTUBE_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return True, match.group(1)
    return False, None

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

def safe_chronomic(text: str, compression: float = 0.5, fuzzy_strength: float = 1.0) -> tuple[str, bool]:
    try:
        return chronomic_filter(text, compression=compression, fuzzy_strength=fuzzy_strength, horizon=6), True
    except Exception:
        logging.warning("Chronomic filtering failed")
        return text, False

async def scrape_webpage(url: str) -> dict:
    logging.info("Starting to scrape URL: %s", url)

    class Result(TypedDict):
        url: str
        title: str
        description: str
        content: str
        content_type: str
        error_info: dict

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

    def partial_result(*, title: str | None = None, description: str | None = None, content: str = "") -> Result:
        return Result(
            url=url,
            title=truncate_field_tokens(title or url, MAX_TITLE_TOKENS),
            description=truncate_field_tokens(description or "", MAX_DESCRIPTION_TOKENS),
            content=content,
            content_type="html_preview" if content else "none",
            error_info={},
        )

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
                    logging.info("Using compression: %.2f (pidgin factor: %.2f)", compression, max(0, (compression - 0.8) / 0.2) if compression > 0.8 else 0)
                    transcript, _ = await parallel_compress(transcript, compression=compression, fuzzy_strength=1.0)
                    compressed_tokens = count_tokens(transcript)
                    logging.info("Compressed tokens: %d", compressed_tokens)
                else:
                    transcript = "[Transcript not available]"

                md_parts = [
                    f"# {truncate_field_tokens(info.get('title', 'Untitled'), MAX_TITLE_TOKENS)}",
                    "",
                    f"**Channel**: {info.get('channel') or info.get('uploader', 'Unknown')}",
                    (
                        f"**Duration**: {info.get('duration', 0) // 60}:{info.get('duration', 0) % 60:02d}"
                        if info.get("duration")
                        else "**Duration**: Unknown"
                    ),
                    f"**Views**: {info.get('view_count', 0):,}" if info.get("view_count") else "**Views**: Unknown",
                    f"**URL**: {url}",
                    "",
                    "## Description",
                    truncate_middle_tokens(info.get("description") or "No description", 1250),
                    "",
                    "## Transcript",
                    truncate_middle_tokens(transcript, MAX_TRANSCRIPT_TOKENS),
                ]

                return Result(
                    url=url,
                    title=truncate_field_tokens(info.get("title", "Untitled"), MAX_TITLE_TOKENS),
                    description=truncate_field_tokens(f"YouTube video by {info.get('channel', 'Unknown')}", MAX_DESCRIPTION_TOKENS),
                    content="\n".join(md_parts),
                    content_type="youtube",
                    error_info={},
                )
            except Exception:
                logging.exception("YouTube scraping failed")
                return partial_result(title=f"Video: {video_id}")

        return await scrape_youtube()

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

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            )
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status >= 400:
                    logging.warning("HTTP %s for %s", resp.status, url)
                    return partial_result(description=f"HTTP {resp.status}")
                html_text = await resp.text()

        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "iframe", "form"]):
            tag.decompose()

        main = None
        for sel in ("article", "main", '[role="main"]', ".content", "#content"):
            main = soup.select_one(sel)
            if main:
                break

        if main:
            content = clean(main.get_text(" ", strip=True))
        else:
            blocks = []
            for elem in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
                txt = elem.get_text(" ", strip=True)
                if is_sig(txt) and txt not in blocks:
                    blocks.append(txt)
            content = "\n\n".join(blocks)

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