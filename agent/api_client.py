import os, json, base64, asyncio, logging
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
from pydantic import BaseModel, Field

from tokenizer import count_tokens, calculate_image_tokens

# ───────────────────────────  constants & init  ────────────────────────────
MAX_IMAGE_DIM = 640
color_init(autoreset=True)
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───────────────────────────  pydantic models  ─────────────────────────────
class ProviderConfig(BaseModel):
    model_name: str
    api_key: str | None = None
    api_base: str | None = None

class APIState(BaseModel):
    api_type:   str | None = None
    api_base:   str | None = None
    api_key:    str | None = None
    model_name: str | None = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p:       float = Field(0.9, gt=0.0, le=1.0)
    frequency_penalty: float = Field(0.8, ge=-2.0, le=2.0)
    presence_penalty:  float = Field(0.5, ge=-2.0, le=2.0)

api = APIState()

# ───────────────────────────  helpers  ─────────────────────────────────────
def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val: raise EnvironmentError(f"Environment variable {var} is required")
    return val

def build_chat_messages(system: str, context: str, user_content):
    msgs = []
    if system:  msgs.append({"role": "system", "content": system})
    if context: msgs.append({"role": "system", "content": context})
    msgs.append({"role": "user", "content": user_content})
    return msgs

def _is_gpt5(name: str | None) -> bool:
    return (name or "").lower().startswith("gpt-5")

def encode_image(path: str) -> Tuple[str, Tuple[int, int]]:
    with Image.open(path) as img:
        if max(img.size) > MAX_IMAGE_DIM:
            ratio  = MAX_IMAGE_DIM / max(img.size)
            new_sz = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_sz, Image.Resampling.LANCZOS)
        if img.mode != "RGB": img = img.convert("RGB")
        buf = BytesIO(); img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode(), img.size

def prepare_image_content(prompt: str, image_paths: List[str], api_type: str) -> Tuple[object, list]:
    if not image_paths: return prompt, []
    if api_type == "gemini":
        content, dims = [prompt], []
        for p in image_paths:
            img = Image.open(p); img.load()
            if max(img.size) > MAX_IMAGE_DIM:
                ratio = MAX_IMAGE_DIM / max(img.size)
                img = img.resize(tuple(int(d * ratio) for d in img.size), Image.Resampling.LANCZOS)
            content.append(img); dims.append(img.size)
        return content, dims
    b64s, dims = zip(*(encode_image(p) for p in image_paths))
    if api_type == "anthropic":
        parts = [{"type": "text", "text": prompt}]
        for b in b64s:
            parts.append({"type": "image","source":{"type":"base64","media_type":"image/jpeg","data":b}})
        return parts, list(dims)
    if api_type in ("openai", "ollama", "openrouter"):
        parts = [{"type": "text", "text": prompt}]
        for b in b64s:
            parts.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b}"}})
        return parts, list(dims)
    raise ValueError(f"Unsupported image provider: {api_type}")

def log_to_jsonl(data: dict, path: str = "api_calls.jsonl") -> None:
    with open(path, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False); f.write("\n")

def get_api_config(api_type: str, model_override: str | None = None) -> ProviderConfig:
    if api_type == "ollama":
        return ProviderConfig(api_base=os.getenv("OLLAMA_API_BASE","http://localhost:11434"),
                              api_key="ollama",
                              model_name=model_override or os.getenv("OLLAMA_MODEL_NAME","gemma3:12b"))
    if api_type == "openai":
        return ProviderConfig(api_key=_require_env("OPENAI_API_KEY"),
                              model_name=model_override or os.getenv("OPENAI_MODEL_NAME","gpt-4.1-mini"))
    if api_type == "anthropic":
        return ProviderConfig(api_key=_require_env("ANTHROPIC_API_KEY"),
                              model_name=model_override or os.getenv("ANTHROPIC_MODEL_NAME","claude-haiku-4-5"))
    if api_type == "vllm":
        return ProviderConfig(api_base=os.getenv("VLLM_API_BASE","http://localhost:4000"),
                              api_key=_require_env("VLLM_API_KEY"),
                              model_name=model_override or os.getenv("VLLM_MODEL_NAME","google/gemma-3-4b-it"))
    if api_type == "gemini":
        return ProviderConfig(api_key=_require_env("GEMINI_API_KEY"),
                              model_name=model_override or os.getenv("GEMINI_MODEL_NAME","gemini-2.5-flash"))
    if api_type == "openrouter":
        return ProviderConfig(api_base="https://openrouter.ai/api/v1",
                              api_key=_require_env("OPENROUTER_API_KEY"),
                              model_name=model_override or os.getenv("OPENROUTER_MODEL_NAME","moonshotai/kimi-k2:free"))
    raise ValueError(f"Unsupported API type: {api_type}")

# ───────────────────────────  initialize_api_client  ──────────────────────
def initialize_api_client(args):
    api.api_type  = args.api
    cfg = get_api_config(api.api_type, args.model)
    api.model_name = cfg.model_name
    api.api_base   = cfg.api_base
    api.api_key    = cfg.api_key
    if api.api_type == "openai": openai.api_key = api.api_key
    logging.info("Initialized API client (%s, model=%s)", api.api_type, api.model_name)

# ───────────────────────────  public setters  ──────────────────────────────
def update_api_temperature(temperature: float) -> None:
    if not 0.0 <= temperature <= 2.0: raise ValueError("temperature must be between 0.0 and 2.0")
    api.temperature = temperature; logging.info("temperature set to %.2f", api.temperature)

def update_api_top_p(p: float) -> None:
    if not 0.0 < p <= 1.0: raise ValueError("top_p must be 0-1")
    api.top_p = p; logging.info("top_p set to %.2f", api.top_p)

def update_api_frequency_penalty(penalty: float) -> None:
    if not -2.0 <= penalty <= 2.0: raise ValueError("frequency_penalty must be between -2.0 and 2.0")
    api.frequency_penalty = penalty; logging.info("frequency_penalty set to %.2f", api.frequency_penalty)

def update_api_presence_penalty(penalty: float) -> None:
    if not -2.0 <= penalty <= 2.0: raise ValueError("presence_penalty must be between -2.0 and 2.0")
    api.presence_penalty = penalty; logging.info("presence_penalty set to %.2f", api.presence_penalty)

# ───────────────────────────  retry wrapper  ───────────────────────────────
async def retry_api_call(func, *a, max_retries=3, retry_delay=1, **kw):
    for attempt in range(max_retries):
        try:
            return await func(*a, **kw)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600 and attempt < max_retries - 1:
                logging.warning("5xx from provider, retry %d/%d", attempt + 1, max_retries)
                await asyncio.sleep(retry_delay); continue
            raise

# ───────────────────────────  main entry  ──────────────────────────────────
async def call_api(prompt: str, *, context: str = "", system_prompt: str = "",
                   conversation_id=None, temperature: float | None = None,
                   top_p: float | None = None, frequency_penalty: float | None = None,
                   presence_penalty: float | None = None, image_paths: List[str] | None = None,
                   api_type_override: str | None = None, model_override: str | None = None):
    temp = temperature if temperature is not None else api.temperature
    p_val = top_p if top_p is not None else api.top_p
    freq_pen = frequency_penalty if frequency_penalty is not None else api.frequency_penalty
    pres_pen = presence_penalty if presence_penalty is not None else api.presence_penalty
    provider = api_type_override or api.api_type
    model    = model_override or api.model_name
    print(Fore.LIGHTMAGENTA_EX + system_prompt)
    print(Fore.LIGHTCYAN_EX + prompt)
    content, dims = prepare_image_content(prompt, image_paths or [], provider)

    async def dispatch():
        if api_type_override and not model_override:
            cfg = get_api_config(provider, None)
        elif model_override:
            cfg = get_api_config(provider, model_override)
        else:
            cfg = ProviderConfig(model_name=api.model_name, api_key=api.api_key, api_base=getattr(api,'api_base',None))
            if provider != api.api_type:
                cfg = get_api_config(provider, api.model_name)

        logging.info("Call → %s | model=%s T=%.2f P=%.2f FP=%.2f PP=%.2f",
                     provider, cfg.model_name, temp, p_val, freq_pen, pres_pen)
        kwargs = dict(system_prompt=system_prompt, context=context,
                      temperature=temp, top_p=p_val,
                      frequency_penalty=freq_pen, presence_penalty=pres_pen,
                      config=cfg)
        if provider == "openai":     return await _call_openai(content, **kwargs)
        if provider == "ollama":     return await _call_ollama(content, **kwargs)
        if provider == "anthropic":  return await _call_anthropic(content, **kwargs)
        if provider == "vllm":       return await _call_vllm(content, **kwargs)
        if provider == "openrouter": return await _call_openrouter(content, **kwargs)
        if provider == "gemini":     return await _call_gemini(content, **kwargs)
        raise ValueError(provider)

    response = await retry_api_call(dispatch)
    print(Fore.MAGENTA + response)

    txt = f"{system_prompt}\n{context}\n{prompt}" if (system_prompt or context) else prompt
    input_tok  = count_tokens(txt) + sum(calculate_image_tokens(w, h) for w, h in dims)
    output_tok = count_tokens(response)
    logging.info("Tokens → Input: %d | Output: %d | Total: %d", input_tok, output_tok, input_tok + output_tok)

    user_field = f"[Image] {prompt}" if image_paths else prompt
    log_to_jsonl({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "api_type": provider,
        "model": model,
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

# ─────────────────────── provider-specific implementations ─────────────────
async def _call_openai(content, *, system_prompt, context,
                       temperature, top_p, frequency_penalty, presence_penalty, config: ProviderConfig):
    client = openai.AsyncOpenAI(api_key=config.api_key)
    msgs = build_chat_messages(system_prompt, context, content)
    kwargs = {"model":config.model_name, "messages":msgs, "max_completion_tokens":12_000}
    if not _is_gpt5(config.model_name):
        kwargs.update(temperature=temperature, top_p=top_p,
                      frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    res = await client.chat.completions.create(**kwargs)
    return res.choices[0].message.content.strip()

async def _call_ollama(content, *, system_prompt, context,
                       temperature, top_p, frequency_penalty, presence_penalty, config: ProviderConfig):
    client = openai.AsyncOpenAI(base_url=f'{config.api_base}/v1', api_key="ollama")
    msgs = build_chat_messages(system_prompt, context, content)
    res = await client.chat.completions.create(
        model=config.model_name, messages=msgs,
        max_tokens=12_000, temperature=temperature, top_p=top_p,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    return res.choices[0].message.content.strip()

async def _call_anthropic(content, *, system_prompt, context,
                          temperature, top_p, frequency_penalty, presence_penalty, config: ProviderConfig):
    client = anthropic.AsyncAnthropic(api_key=config.api_key)
    res = await client.messages.create(
        model=config.model_name,
        system=system_prompt,
        messages=[{"role":"user","content":content}],
        max_tokens=4096,
        temperature=temperature)  # ignore top_p per Anthropic constraint
    return res.content[0].text.strip()

async def _call_vllm(content, *, system_prompt, context,
                     temperature, top_p, frequency_penalty, presence_penalty, config: ProviderConfig):
    prompt = "\n".join(filter(None, [system_prompt, context, content]))
    async with aiohttp.ClientSession() as s:
        res = await s.post(f'{config.api_base}/v1/completions',
                           headers={"Authorization": f'Bearer {config.api_key}',
                                    "Content-Type": "application/json"},
                           json={"model": config.model_name,
                                 "prompt": prompt,
                                 "max_tokens": 4096,
                                 "temperature": temperature,
                                 "top_p": top_p})
        if res.status != 200: raise Exception(f"vLLM status {res.status}: {await res.text()}")
        data = await res.json()
        return data["choices"][0]["text"].strip()

async def _call_openrouter(content, *, system_prompt, context,
                           temperature, top_p, frequency_penalty, presence_penalty, config: ProviderConfig):
    client = openai.AsyncOpenAI(base_url=config.api_base, api_key=config.api_key)
    msgs = build_chat_messages(system_prompt, context, content)
    res = await client.chat.completions.create(
        model=config.model_name,
        messages=msgs,
        max_tokens=12000,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty)
    return res.choices[0].message.content.strip()

async def _call_gemini(content,*,system_prompt,context,temperature,top_p,frequency_penalty,presence_penalty,config: ProviderConfig):
    client = genai.Client(api_key=config.api_key)
    sys = "\n\n".join([s for s in (system_prompt,context) if s])
    if isinstance(content,list): utext,imgs=(content[0],content[1:])
    else: utext,imgs=(str(content),[])
    parts=[types.Part.from_text(text=utext)]+[types.Part.from_image(image=i) for i in imgs]
    cfg=types.GenerateContentConfig(system_instruction=(sys or None),
                                    temperature=temperature, top_p=top_p, top_k=64,
                                    max_output_tokens=8192, response_mime_type="text/plain")
    r=await asyncio.to_thread(client.models.generate_content,
                              model=config.model_name,
                              contents=[types.Content(role="user",parts=parts)],
                              config=cfg)
    return r.text.strip()

# ─────────────────────── embeddings helper (unchanged) ────────────────────
async def get_embeddings(text: str | list[str],
                         provider: str | None = None,
                         model: str | None = None,
                         max_tokens: int = 256):
    max_chars = max_tokens * 3
    if isinstance(text, str):
        if len(text) > max_chars: text = text[:max_chars // 2] + "..." + text[-max_chars // 2:]
    else:
        text = [t[:max_chars // 2] + "..." + t[-max_chars // 2:] if len(t) > max_chars else t for t in text]

    provider = provider or api.api_type
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=_require_env("OPENAI_API_KEY"))
        model  = model or "text-embedding-3-small"
        res = await client.embeddings.create(model=model, input=text)
        return res.data[0].embedding if isinstance(text, str) else [d.embedding for d in res.data]
    if provider == "ollama":
        client = openai.AsyncOpenAI(base_url=f"{api.api_base}/v1", api_key="ollama")
        model = model or "all-minilm:latest"
        res = await client.embeddings.create(model=model, input=text)
        return res.data[0].embedding if isinstance(text, str) else [d.embedding for d in res.data]
    if provider == "vllm":
        model = model or os.getenv("VLLM_EMBED_MODEL", "jinaai/jina-embeddings-v2-base-en")
        async with aiohttp.ClientSession() as s:
            r = await s.post("http://localhost:8080/embed", json={"model": model, "inputs": text})
            if r.status != 200: raise RuntimeError(await r.text())
            data = await r.json()
            return data[0] if isinstance(data, list) else data["embeddings"][0]
    raise ValueError(f"Embeddings not supported for {provider}")

# ───────────────────────────  cli  ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse, asyncio as _aio
    ap = argparse.ArgumentParser(description="Multi-API LLM client")
    ap.add_argument("--api", required=True,
                    choices=["ollama", "openai", "anthropic", "vllm", "openrouter", "gemini"])
    ap.add_argument("--model", help="model override")
    args = ap.parse_args()

    cfg = get_api_config(args.api, args.model)
    api.api_type   = args.api
    api.model_name = cfg.model_name
    api.api_base   = cfg.api_base
    api.api_key    = cfg.api_key

    while True:
        try:
            user_in = input(">>> ")
            if user_in.lower() in ("quit", "exit"): break
            _aio.run(call_api(user_in))
        except KeyboardInterrupt:
            break
