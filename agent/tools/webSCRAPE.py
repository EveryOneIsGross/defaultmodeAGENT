# for webscraping with fallback and silent‑error policy
"""
A single‑entry async scraper that **never forwards internal error details** to the
caller.  It guarantees the same six keys in the returned dict:

    {url, title, description, content, content_type, error_info}

Where:
* `content_type` is **never** "error". It is one of:
    - "text/html"      → normal extraction
    - "youtube"        → YouTube with transcript/metadata
    - "html_preview"   → raw HTML preview when structured extraction failed
    - "none"           → nothing could be fetched (e.g. HTTP 404)
* `error_info` is always an **empty dict** so downstream clients never see
  scraper internals.

The module still logs everything (`logging.warning`/`logging.exception`) to aid
server‑side debugging.
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

from .chronpression import chronomic_filter
from tokenizer import count_tokens, encode_text, decode_tokens

# ---------------------------------------------------------------------------
# Global limits (tokens - more accurate for LLM processing)
# ---------------------------------------------------------------------------
MAX_TITLE_TOKENS = 50       # ~200 chars
MAX_DESCRIPTION_TOKENS = 125 # ~500 chars
MAX_CONTENT_TOKENS = 12500   # ~50k chars   
MAX_TRANSCRIPT_TOKENS = 15000 # ~60k chars
MAX_HTML_PREVIEW_TOKENS = 2500 # ~10k chars
PRESERVE_RATIO = 0.4  # keep 40 % head + 40 % tail when truncating

# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------
YOUTUBE_PATTERNS = [
    r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
]

def is_youtube_url(url: str) -> tuple[bool, str | None]:
    """Return (True, video_id) if *url* looks like YouTube."""
    for pattern in YOUTUBE_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return True, match.group(1)
    return False, None

# ---------------------------------------------------------------------------
# Main public coroutine
# ---------------------------------------------------------------------------
async def scrape_webpage(url: str) -> dict:
    """Fetch *url* and return a uniform result dict (see module doc‑string)."""

    logging.info("Starting to scrape URL: %s", url)

    class Result(TypedDict):
        url: str
        title: str
        description: str
        content: str
        content_type: str
        error_info: dict

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------
    def truncate_middle_tokens(text: str, max_tokens: int, preserve_ratio: float = PRESERVE_RATIO) -> str:
        """Keep head/tail, drop middle so total ≤ *max_tokens*."""
        if not text:
            return text
        
        current_tokens = count_tokens(text)
        if current_tokens <= max_tokens:
            return text
            
        # Calculate how many tokens to preserve from head/tail
        preserve_tokens = int(max_tokens * preserve_ratio)
        head_tokens = preserve_tokens // 2
        tail_tokens = preserve_tokens // 2
        
        # Encode text to tokens
        tokens = encode_text(text)
        
        # Find sentence boundary near the cut point if possible
        head_end = head_tokens
        try:
            # Look for sentence ending punctuation in a small window
            search_start = max(0, head_tokens - 50)
            search_end = min(len(tokens), head_tokens + 50)
            
            for i in range(search_end - 1, search_start - 1, -1):
                token_text = decode_tokens([tokens[i]])
                if any(punct in token_text for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']):
                    head_end = i + 1
                    break
        except:
            pass  # fallback to original position
        
        # Combine head + truncation marker + tail
        head_tokens_actual = tokens[:head_end]
        tail_tokens_actual = tokens[-tail_tokens:] if tail_tokens > 0 else []
        
        head_text = decode_tokens(head_tokens_actual).rstrip()
        tail_text = decode_tokens(tail_tokens_actual).lstrip() if tail_tokens_actual else ""
        
        return head_text + "\n\n[…content truncated…]\n\n" + tail_text

    def truncate_field_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens with ellipsis if needed."""
        if not text:
            return text
            
        current_tokens = count_tokens(text)
        if current_tokens <= max_tokens:
            return text
            
        # Encode, truncate, decode
        tokens = encode_text(text)
        # Reserve 1 token for ellipsis
        truncated_tokens = tokens[:max_tokens - 1]
        return decode_tokens(truncated_tokens) + "…"

    def clean(txt: str) -> str:
        txt = html.unescape(txt)
        txt = unicodedata.normalize("NFKC", txt)
        return re.sub(r"\s+", " ", txt).strip()

    def partial_result(*, title: str | None = None, description: str | None = None, content: str = "") -> Result:
        """Return a minimal, *non‑error* Result suitable for forwarding."""
        return Result(
            url=url,
            title=truncate_field_tokens(title or url, MAX_TITLE_TOKENS),
            description=truncate_field_tokens(description or "", MAX_DESCRIPTION_TOKENS),
            content=content,
            content_type="html_preview" if content else "none",
            error_info={},  # never forward internal errors
        )

    # ---------------------------------------------------------------------
    # YouTube branch
    # ---------------------------------------------------------------------
    is_yt, video_id = is_youtube_url(url)
    if is_yt:
        logging.info("Detected YouTube video ID: %s", video_id)

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

                # --- captions to plain text --------------------------------
                transcript = ""
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
                                import urllib.request

                                with urllib.request.urlopen(fmt["url"]) as resp:
                                    vtt = resp.read().decode("utf-8")
                                lines = [
                                    re.sub(r"<[^>]+>", "", ln.strip())
                                    for ln in vtt.split("\n")
                                    if ln.strip()
                                    and not ln.startswith(("WEBVTT", "NOTE", "Kind:", "Language:"))
                                    and "-->" not in ln
                                    and not ln.strip().isdigit()
                                ]
                                transcript = " ".join(dict.fromkeys(lines))  # de‑dup lines
                                break
                            except Exception:  # pragma: no cover
                                continue
                        if transcript:
                            break

                if transcript:
                    try:
                        transcript = chronomic_filter(transcript, alpha=0.3, beta=3.0)
                    except Exception:
                        logging.warning("Chronomic filtering failed on YouTube transcript.")
                else:
                    transcript = "[Transcript not available]"

                # --- build markdown content -------------------------------
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
                    truncate_middle_tokens(info.get("description") or "No description", 1250),  # ~5k chars
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
                    error_info={},  # hide internals
                )
            except Exception as e:  # pragma: no cover
                logging.exception("YouTube scraping failed")
                # Do not forward details – just mark as none
                return partial_result(title=f"Video: {video_id}")

        return await scrape_youtube()

    # ---------------------------------------------------------------------
    # Plain HTML branch
    # ---------------------------------------------------------------------
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
            # fallback to raw HTML preview
            content = truncate_middle_tokens(clean(html_text), MAX_HTML_PREVIEW_TOKENS)
            ctype = "html_preview"
        else:
            ctype = "text/html"
            try:
                content = chronomic_filter(content, alpha=0.3, beta=3.0)
            except Exception:
                logging.warning("Chronomic filtering failed on web content")

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

    except Exception:  # pragma: no cover
        logging.exception("Scraping failed for %s", url)
        return partial_result(description="Unexpected scraping error")
