# for webscraping
import re, html, unicodedata, logging, asyncio, aiohttp
from typing import TypedDict
from bs4 import BeautifulSoup

async def scrape_webpage(url: str) -> dict:
    """
    Robust, single-entry async scraper that always returns the same keys:
    {url, title, description, content, content_type, error_info}
    Error info contains any error details while still returning partial content
    """
    logging.info(f"Starting to scrape URL: {url}")
    
    class Result(TypedDict):
        url: str
        title: str
        description: str
        content: str
        content_type: str
        error_info: dict  # Contains error details if any: {code, message, phase}

    # ── tiny helpers ──────────────────────────────────────────────────────────
    BAD_UI = re.compile(
        r"(cookie|accept|subscribe|sign[\s-]?up|sign[\s-]?in|login|password|username|"
        r"newsletter|privacy\s+policy|terms\s+of\s+service|copyright)",
        flags=re.I,
    )

    def is_sig(block: str) -> bool:
        block = block.strip()
        if len(block) < 30 or BAD_UI.search(block):          # too short / boiler-plate
            return False
        if not any(c in block for c in ".!?"):               # no sentences → nav
            return False
        spec_ratio = sum(1 for c in block if not (c.isalnum() or c.isspace())) / len(block)
        if spec_ratio > 0.33:                                # code/glyph soup
            return False
        w = block.lower().split()
        return not (len(w) > 5 and len(set(w)) / len(w) < 0.5)  # repeated words

    def clean(txt: str) -> str:
        txt = html.unescape(txt)
        txt = unicodedata.normalize("NFKC", txt)
        return re.sub(r"\s+", " ", txt).strip()

    def partial_result(error_code: str, error_msg: str, phase: str, **kwargs) -> Result:
        """Return partial results with error info instead of failing completely"""
        logging.warning(f"Error in {phase} for {url}: {error_msg}")
        return Result(
            url=url,
            title=kwargs.get('title', url),
            description=kwargs.get('description', f"Partial content - {error_msg}"),
            content=kwargs.get('content', ''),
            content_type=kwargs.get('content_type', "partial"),
            error_info={
                'code': error_code,
                'message': error_msg,
                'phase': phase
            }
        )

    # ── fetch with back-off ───────────────────────────────────────────────────
    async def fetch_html() -> tuple[str, str, dict | None]:
        hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0"}
        last_error = None
        for delay in (0, 2, 5):
            try:
                logging.debug(f"Attempt {delay+1} to fetch {url}")
                async with aiohttp.ClientSession() as s:
                    async with s.get(url, headers=hdrs, timeout=15) as r:
                        if r.status >= 400:
                            last_error = {'code': str(r.status), 'message': f"HTTP {r.status}", 'phase': 'fetch'}
                            if delay == 4:  # Last attempt
                                return None, None, last_error
                            continue
                        ctype = r.headers.get("content-type", "")
                        logging.debug(f"Content-Type: {ctype}")
                        return ctype, await r.text(), None
            except Exception as e:
                last_error = {'code': type(e).__name__, 'message': str(e), 'phase': 'fetch'}
                logging.warning(f"Fetch attempt {delay+1} failed: {str(e)}")
                if delay:
                    await asyncio.sleep(delay)
                continue
        return None, None, last_error

    try:
        ctype, html_text, fetch_error = await fetch_html()
        if not html_text:
            return partial_result(
                fetch_error['code'],
                fetch_error['message'],
                'fetch'
            )
        
        if "text/html" not in (ctype or "").lower():
            return partial_result(
                'INVALID_CONTENT_TYPE',
                f"Non-HTML content type ({ctype})",
                'content_check',
                content_type=ctype
            )

        # ── main extraction via BeautifulSoup ───────────────────────────────────
        try:
            logging.debug("Starting BeautifulSoup parsing")
            soup = BeautifulSoup(html_text, "html.parser")
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'form']):
                element.decompose()
            
            # Extract main content - prioritize article or main tags
            main_content = None
            for tag in ['article', 'main', 'div[role="main"]', '.content', '#content', '.post', '.article']:
                main_content = soup.select_one(tag)
                if main_content:
                    logging.debug(f"Found main content using selector: {tag}")
                    break
            
            content = ""
            extraction_error = None
            
            if main_content:
                content = main_content.get_text(" ", strip=True)
                logging.debug("Extracted content from main content area")
            else:
                logging.debug("No main content found, falling back to block extraction")
                blocks = [t.get_text(" ", strip=True) for t in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
                seen, sig_blocks = set(), []
                for b in blocks:
                    if b not in seen and is_sig(b):
                        seen.add(b)
                        sig_blocks.append(b)
                content = "\n\n".join(sig_blocks)
                if not sig_blocks:
                    extraction_error = {
                        'code': 'NO_MAIN_CONTENT',
                        'message': 'No main content container found',
                        'phase': 'extraction'
                    }
                logging.debug(f"Extracted {len(sig_blocks)} significant blocks")

            content = clean(content)
            title = (soup.title.string if soup.title else url).strip()
            meta_desc = soup.find("meta", attrs={"name": "description"})
            description = (meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else "")

            if not content:
                return partial_result(
                    'NO_CONTENT',
                    "No meaningful content extracted",
                    'extraction',
                    title=title,
                    description=description
                )

            logging.info(f"Successfully scraped {url}")
            return Result(
                url=url,
                title=title,
                description=description,
                content=content,
                content_type="text/html",
                error_info=extraction_error or {}
            )

        except Exception as e:
            logging.error(f"BeautifulSoup parsing failed for {url}: {str(e)}")
            # Try fallback parsing
            logging.debug("Attempting fallback parsing")
            try:
                soup_all = BeautifulSoup(html_text, "html.parser")
                for t in soup_all(["script", "style", "nav", "footer"]):
                    t.decompose()
                blocks = [t.get_text(" ", strip=True) for t in soup_all.find_all(text=True)]
                seen, sig_blocks = set(), []
                for b in blocks:
                    if b not in seen and is_sig(b):
                        seen.add(b)
                        sig_blocks.append(b)
                content = "\n\n".join(sig_blocks)
                logging.debug(f"Fallback parsing found {len(sig_blocks)} blocks")

                content = clean(content)
                title = (soup_all.title.string if soup_all.title else url).strip()
                meta_desc = soup_all.find("meta", attrs={"name": "description"})
                description = (meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else "")

                if not content:
                    return partial_result(
                        'FALLBACK_NO_CONTENT',
                        "No content from fallback parsing",
                        'fallback',
                        title=title,
                        description=description
                    )

                return Result(
                    url=url,
                    title=title,
                    description=description,
                    content=content,
                    content_type="text/html",
                    error_info={'code': 'USED_FALLBACK', 'message': str(e), 'phase': 'parsing'}
                )
            except Exception as fallback_e:
                return partial_result(
                    'FALLBACK_FAILED',
                    str(fallback_e),
                    'fallback',
                    content=html_text[:1000] if html_text else ''  # Return raw HTML snippet as last resort
                )

    except Exception as exc:
        logging.exception(f"scrape_webpage failure for {url}")
        return partial_result(
            'CRITICAL_ERROR',
            str(exc),
            'unknown',
            content=html_text[:1000] if html_text else ''
        )