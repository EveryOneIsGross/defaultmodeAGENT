import os, json, base64, asyncio, logging, mimetypes
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Tuple

import aiohttp
import openai
import anthropic
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
from colorama import Fore, init as color_init

from tokenizer import count_tokens, calculate_image_tokens

# ───────────────────────────  constants & init  ────────────────────────────
MAX_IMAGE_DIM = 1_568

color_init(autoreset=True)
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ───────────────────────────  global config  ───────────────────────────────
class APIState:
    """Global state for API configuration."""
    def __init__(self) -> None:
        self.api_type:   str   | None = None
        self.api_base:   str   | None = None
        self.api_key:    str   | None = None
        self.model_name: str   | None = None
        self.temperature:float = 0.7
        self.top_p:      float = 0.9
        self.frequency_penalty: float = 0.8
        self.presence_penalty: float = 0.5
        self.gemini_client: genai.Client | None = None

api = APIState()

# ───────────────────────────  helpers  ─────────────────────────────────────
def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise EnvironmentError(f"Environment variable {var} is required")
    return val

def build_chat_messages(system: str, context: str, user_content):
    msgs = []
    if system:  msgs.append({"role": "system", "content": system})
    if context: msgs.append({"role": "system", "content": context})
    msgs.append({"role": "user", "content": user_content})
    return msgs

def encode_image(path: str) -> Tuple[str, Tuple[int, int]]:
    """Resize (if >MAX_IMAGE_DIM), convert to JPEG, return (b64, size)."""
    with Image.open(path) as img:
        if max(img.size) > MAX_IMAGE_DIM:
            ratio  = MAX_IMAGE_DIM / max(img.size)
            new_sz = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_sz, Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode(), img.size

def prepare_image_content(prompt: str,
                          image_paths: List[str],
                          api_type: str
                          ) -> Tuple[object, list]:
    """Return (provider‑ready content, [ (w,h), ... ])"""
    if not image_paths:
        return prompt, []

    # Gemini expects PIL Images; everyone else wants base64 URLs / dicts
    if api_type == "gemini":
        content = [prompt]
        dims = []
        for p in image_paths:
            img = Image.open(p)
            img.load()
            if max(img.size) > MAX_IMAGE_DIM:
                ratio = MAX_IMAGE_DIM / max(img.size)
                img = img.resize(tuple(int(d * ratio) for d in img.size),
                                 Image.Resampling.LANCZOS)
            content.append(img)
            dims.append(img.size)
        return content, dims

    b64s, dims = zip(*(encode_image(p) for p in image_paths))
    if api_type == "anthropic":
        parts = [{"type": "text", "text": prompt}]
        for b in b64s:
            parts.append({"type": "image",
                          "source": {"type": "base64",
                                     "media_type": "image/jpeg",
                                     "data": b}})
        return parts, list(dims)

    if api_type in ("openai", "ollama", "openrouter"):
        parts = [{"type": "text", "text": prompt}]
        for b in b64s:
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
        return parts, list(dims)

    raise ValueError(f"Unsupported image provider: {api_type}")

def log_to_jsonl(data: dict, path: str = "api_calls.jsonl") -> None:
    with open(path, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

def get_api_config(api_type: str, model_override: str | None = None) -> dict:
    """Return {model_name, api_key?, api_base?}"""
    cfg = {}
    if api_type == "ollama":
        cfg.update(api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
                   api_key="ollama",
                   model_name=model_override or
                              os.getenv("OLLAMA_MODEL_NAME", "gemma3:12b"))
    elif api_type == "openai":
        cfg.update(api_key=_require_env("OPENAI_API_KEY"),
                   model_name=model_override or
                              os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini"))
    elif api_type == "anthropic":
        cfg.update(api_key=_require_env("ANTHROPIC_API_KEY"),
                   model_name=model_override or
                              os.getenv("ANTHROPIC_MODEL_NAME",
                                        "claude-3-5-haiku-latest"))
    elif api_type == "vllm":
        cfg.update(api_base=os.getenv("VLLM_API_BASE", "http://localhost:4000"),
                   api_key=_require_env("VLLM_API_KEY"),
                   model_name=model_override or
                              os.getenv("VLLM_MODEL_NAME",
                                        "google/gemma-3-4b-it"))
    elif api_type == "gemini":
        cfg.update(api_key=_require_env("GEMINI_API_KEY"),
                   model_name=model_override or
                              os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
    elif api_type == "openrouter":
        cfg.update(api_base="https://openrouter.ai/api/v1",
                   api_key=_require_env("OPENROUTER_API_KEY"),
                   model_name=model_override or
                              os.getenv("OPENROUTER_MODEL_NAME", "moonshotai/kimi-k2:free"))
    else:
        raise ValueError(f"Unsupported API type: {api_type}")
    return cfg

# ───────────────────────────  initialize_api_client  ──────────────────────
def initialize_api_client(args):
    """
    Original one‑shot boot‑strap helper kept for backward‑compat.
    Typical call site:
        >>> from api_client import initialize_api_client, call_api, api
        >>> initialize_api_client(args)          # args.api / args.model
    """
    api.api_type = args.api
    
    # Determine which model to use (arg overrides provider default)
    model_override = args.model  # may be None
    cfg = get_api_config(api.api_type, model_override)

    # Now populate fresh values, overwriting any prior session state
    api.model_name = cfg["model_name"]
    api.api_base   = cfg.get("api_base")
    api.api_key    = cfg.get("api_key")

    # Provider‑specific SDK priming
    if api.api_type == "openai":
        openai.api_key = api.api_key
    elif api.api_type == "gemini":
        api.gemini_client = genai.Client(api_key=api.api_key)

    logging.info("Initialized API client (%s, model=%s)", api.api_type,
                 api.model_name)


# ───────────────────────────  public setters  ──────────────────────────────
def update_api_temperature(intensity: int) -> None:
    if not 0 <= intensity <= 100:
        raise ValueError("intensity 0‑100 expected")
    api.temperature = intensity / 100
    logging.info("temperature set to %.2f", api.temperature)

def update_api_top_p(p: float) -> None:
    if not 0.0 < p <= 1.0:
        raise ValueError("top_p must be 0‑1")
    api.top_p = p
    logging.info("top_p set to %.2f", api.top_p)

def update_api_frequency_penalty(penalty: float) -> None:
    if not -2.0 <= penalty <= 2.0:
        raise ValueError("frequency_penalty must be between -2.0 and 2.0")
    api.frequency_penalty = penalty
    logging.info("frequency_penalty set to %.2f", api.frequency_penalty)

def update_api_presence_penalty(penalty: float) -> None:
    if not -2.0 <= penalty <= 2.0:
        raise ValueError("presence_penalty must be between -2.0 and 2.0")
    api.presence_penalty = penalty
    logging.info("presence_penalty set to %.2f", api.presence_penalty)

# ───────────────────────────  retry wrapper  ───────────────────────────────
async def retry_api_call(func, *a, max_retries=3, retry_delay=1, **kw):
    for attempt in range(max_retries):
        try:
            return await func(*a, **kw)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600 and attempt < max_retries - 1:
                logging.warning("5xx from provider, retry %d/%d", attempt + 1,
                                max_retries)
                await asyncio.sleep(retry_delay)
                continue
            raise

# ───────────────────────────  main entry  ──────────────────────────────────
async def call_api(prompt: str,
                   *,
                   context: str = "",
                   system_prompt: str = "",
                   conversation_id=None,
                   temperature: float | None = None,
                   top_p: float | None = None,
                   frequency_penalty: float | None = None,
                   presence_penalty: float | None = None,
                   image_paths: List[str] | None = None,
                   api_type_override: str | None = None,
                   model_override: str | None = None):
    # decide hyper‑params & provider
    temp = temperature if temperature is not None else api.temperature
    p_val = top_p if top_p is not None else api.top_p
    freq_pen = frequency_penalty if frequency_penalty is not None else api.frequency_penalty
    pres_pen = presence_penalty if presence_penalty is not None else api.presence_penalty
    provider = api_type_override or api.api_type
    model    = model_override or api.model_name
    print(Fore.LIGHTMAGENTA_EX + system_prompt)
    print(Fore.LIGHTCYAN_EX + prompt)

    # format images / content
    content, dims = prepare_image_content(prompt, image_paths or [], provider)

    # pick provider coroutine
    async def dispatch():
        # Use the current API model if no override specified, or get config for the provider
        if api_type_override and not model_override:
            # API type overridden but no model - get that API type's default
            cfg = get_api_config(provider, None)
        elif model_override:
            # Explicit model specified
            cfg = get_api_config(provider, model_override)
        else:
            # No overrides - use current API configuration
            cfg = {
                "model_name": api.model_name,
                "api_key": api.api_key,
                "api_base": getattr(api, 'api_base', None)
            }
            # Ensure we have the right API key for the current provider
            if provider != api.api_type:
                cfg = get_api_config(provider, api.model_name)
        
        logging.info("Call → %s | model=%s T=%.2f P=%.2f FP=%.2f PP=%.2f",
                     provider, cfg["model_name"], temp, p_val, freq_pen, pres_pen)
        kwargs = dict(system_prompt=system_prompt, context=context,
                      temperature=temp, top_p=p_val, 
                      frequency_penalty=freq_pen, presence_penalty=pres_pen,
                      config=cfg)
        if provider == "openai":
            return await _call_openai(content, **kwargs)
        if provider == "ollama":
            return await _call_ollama(content, **kwargs)
        if provider == "anthropic":
            return await _call_anthropic(content, **kwargs)
        if provider == "vllm":
            return await _call_vllm(content, **kwargs)
        if provider == "openrouter":
            return await _call_openrouter(content, **kwargs)
        if provider == "gemini":
            return await _call_gemini(content, **kwargs)
        raise ValueError(provider)

    response = await retry_api_call(dispatch)

    print(Fore.MAGENTA + response)

    # token accounting
    txt  = f"{system_prompt}\n{context}\n{prompt}" if (system_prompt or context)\
          else prompt
    input_tok  = count_tokens(txt) + sum(
                 calculate_image_tokens(w, h) for w, h in dims)
    output_tok = count_tokens(response)
    
    logging.info("Tokens → Input: %d | Output: %d | Total: %d", 
                 input_tok, output_tok, input_tok + output_tok)

    user_field = f"[Image] {prompt}" if image_paths else prompt
    log_to_jsonl({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "api_type": provider,
        "model": model,                   # extra field – harmless for old reader
        "system_prompt": system_prompt,
        "context": context,
        "user_input": user_field,
        "ai_output": response,
        "is_image": bool(image_paths),
        "num_images": len(image_paths or []),
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "total_tokens": input_tok + output_tok
    })
    return response

# ─────────────────────── provider‑specific implementations ─────────────────
async def _call_openai(content, *, system_prompt, context,
                       temperature, top_p, frequency_penalty, presence_penalty, config):
    client = openai.AsyncOpenAI(api_key=config["api_key"])
    msgs = build_chat_messages(system_prompt, context, content)
    res = await client.chat.completions.create(
        model=config["model_name"],
        messages=msgs,
        max_completion_tokens=12_000,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty)
    return res.choices[0].message.content.strip()

async def _call_ollama(content, *, system_prompt, context,
                       temperature, top_p, frequency_penalty, presence_penalty, config):
    client = openai.AsyncOpenAI(base_url=f'{config["api_base"]}/v1',
                                api_key="ollama")
    msgs = build_chat_messages(system_prompt, context, content)
    res = await client.chat.completions.create(
        model=config["model_name"], messages=msgs,
        max_tokens=12_000, temperature=temperature, top_p=top_p,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    return res.choices[0].message.content.strip()

async def _call_anthropic(content, *, system_prompt, context,
                          temperature, top_p, frequency_penalty, presence_penalty, config):
    client = anthropic.AsyncAnthropic(api_key=config["api_key"])
    res = await client.messages.create(
        model=config["model_name"],
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
        max_tokens=4096,
        temperature=temperature,
        top_p=top_p)
    return res.content[0].text.strip()

async def _call_vllm(content, *, system_prompt, context,
                     temperature, top_p, frequency_penalty, presence_penalty, config):
    prompt = "\n".join(filter(None, [system_prompt, context, content]))
    async with aiohttp.ClientSession() as s:
        res = await s.post(f'{config["api_base"]}/v1/completions',
                           headers={"Authorization": f'Bearer {config["api_key"]}',
                                    "Content-Type": "application/json"},
                           json={"model": config["model_name"],
                                 "prompt": prompt,
                                 "max_tokens": 4096,
                                 "temperature": temperature,
                                 "top_p": top_p})
        if res.status != 200:
            raise Exception(f"vLLM status {res.status}: {await res.text()}")
        data = await res.json()
        return data["choices"][0]["text"].strip()

async def _call_openrouter(content, *, system_prompt, context,
                           temperature, top_p, frequency_penalty, presence_penalty, config):
    client = openai.AsyncOpenAI(base_url=config["api_base"],
                                api_key=config["api_key"])
    msgs = build_chat_messages(system_prompt, context, content)
    res = await client.chat.completions.create(
        model=config["model_name"],
        messages=msgs,
        max_tokens=12000,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty)
    return res.choices[0].message.content.strip()
'''
async def _call_gemini(content, *, system_prompt, context,
                       temperature, top_p, frequency_penalty, presence_penalty, config):
    client = genai.Client(api_key=config["api_key"])
    parts = [p for p in (system_prompt, context) if p] + (
            content if isinstance(content, list) else [content])
    gen_cfg = types.GenerateContentConfig(temperature=temperature,
                                          top_p=top_p,
                                          top_k=64,
                                          max_output_tokens=8192,
                                          response_mime_type="text/plain")
    try:
        res = await asyncio.to_thread(
            client.models.generate_content,
            model=config["model_name"],
            contents=parts,
            config=gen_cfg)
        return res.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        raise Exception(f"Gemini API call failed: {str(e)}")
'''
async def _call_gemini(content,*,system_prompt,context,temperature,top_p,frequency_penalty,presence_penalty,config):
    c=api.gemini_client or genai.Client(api_key=config["api_key"])
    sys="\n\n".join([s for s in (system_prompt,context) if s])
    if isinstance(content,list): utext,imgs=(content[0],content[1:])
    else: utext,imgs=(str(content),[])
    parts=[types.Part.from_text(text=utext)]+[types.Part.from_image(image=i) for i in imgs]
    cfg=types.GenerateContentConfig(system_instruction=(sys or None),temperature=temperature,top_p=top_p,top_k=64,max_output_tokens=8192,response_mime_type="text/plain")
    r=await asyncio.to_thread(c.models.generate_content,model=config["model_name"],contents=[types.Content(role="user",parts=parts)],config=cfg)
    return r.text.strip()


# ─────────────────────── embeddings helper (unchanged) ────────────────────
async def get_embeddings(text: str,
                         provider: str | None = None,
                         model: str | None = None):
    provider = provider or api.api_type
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=_require_env("OPENAI_API_KEY"))
        model  = model or "text-embedding-3-small"
        res = await client.embeddings.create(model=model, input=text)
        return res.data[0].embedding
    if provider == "ollama":
        client = openai.AsyncOpenAI(base_url=f"{api.api_base}/v1",
                                    api_key="ollama")
        model = model or "all-minilm:latest"
        res = await client.embeddings.create(model=model, input=text)
        return res.data[0].embedding
    if provider == "vllm":
        model = model or os.getenv("VLLM_EMBED_MODEL",
                                   "jinaai/jina-embeddings-v2-base-en")
        async with aiohttp.ClientSession() as s:
            r = await s.post("http://localhost:8080/embed",
                             json={"model": model, "inputs": text})
            if r.status != 200:
                raise RuntimeError(await r.text())
            data = await r.json()
            return data[0] if isinstance(data, list) else data["embeddings"][0]
    raise ValueError(f"Embeddings not supported for {provider}")

# ───────────────────────────  cli  ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse, asyncio as _aio
    ap = argparse.ArgumentParser(description="Multi‑API LLM client")
    ap.add_argument("--api", required=True,
                    choices=["ollama", "openai", "anthropic", "vllm", "openrouter", "gemini"])
    ap.add_argument("--model", help="model override")
    args = ap.parse_args()

    # initialise global api state using proper defaults
    cfg = get_api_config(args.api, args.model)
    api.api_type = args.api
    api.model_name = cfg["model_name"]
    api.api_base = cfg.get("api_base")
    api.api_key = cfg.get("api_key")

    # simple REPL
    while True:
        try:
            user_in = input(">>> ")
            if user_in.lower() in ("quit", "exit"):
                break
            _aio.run(call_api(user_in))
        except KeyboardInterrupt:
            break