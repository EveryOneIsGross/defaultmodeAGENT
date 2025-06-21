"""
api_client has it's own logging to ensure it remains project agnostic.
"""

import os
import logging
import aiohttp
import json
from dotenv import load_dotenv
from colorama import Fore, Back, Style, init
from datetime import datetime

import openai
import anthropic
from google import genai
from google.genai import types

import base64
from io import BytesIO
from PIL import Image

import mimetypes
import asyncio

from tokenizer import count_tokens

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Global state for API configuration
class APIState:
    """Global state for API configuration."""
    def __init__(self):
        self.api_type = None
        self.api_base = None
        self.api_version = None
        self.api_key = None
        self.deployment_name = None
        self.model_name = None
        self.temperature = 0.7  # Default temperature
        self.top_p = 0.9  # Default top_p

# Create global instance
api = APIState()

def initialize_api_client(args):
    """Initialize API client with configuration."""
    api.api_type = args.api
    
    if api.api_type == 'ollama':
        api.api_base = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
        api.model_name = args.model or os.getenv('OLLAMA_MODEL_NAME', 'llama3.2-vision')
    elif api.api_type == 'openai':
        api.api_key = os.getenv('OPENAI_API_KEY')
        api.model_name = args.model or os.getenv('OPENAI_MODEL_NAME', 'gpt-4.1')
        openai.api_key = api.api_key
    elif api.api_type == 'anthropic':
        api.api_key = os.getenv('ANTHROPIC_API_KEY')
        api.model_name = args.model or os.getenv('ANTHROPIC_MODEL_NAME', 'claude-3-5-sonnet-20241022')
    elif api.api_type == 'vllm':
        api.api_base = os.getenv('VLLM_API_BASE', 'http://localhost:4000')
        api.model_name = args.model or os.getenv('VLLM_MODEL_NAME', 'hermes-testing/Hermes-3-Pro-RC2-e4')
    elif api.api_type == 'gemini':
        api.api_key = os.getenv('GEMINI_API_KEY')
        api.model_name = args.model or os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-pro-exp-03-25')
        api.gemini_client = genai.Client(api_key=api.api_key)
    else:
        raise ValueError(f"Unsupported API type: {api.api_type}")

    logging.info(f"Initialized API client with {api.api_type}")

def log_to_jsonl(data):
    with open('api_calls.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

def encode_image(image_path):
    """Encode image to base64 with proper resizing if needed"""
    with Image.open(image_path) as img:
        # Resize if needed (max dimension 1568 for compatibility)
        max_dim = 1568
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def prepare_image_content(prompt: str, image_paths: list, api_type: str) -> list:
    """Prepare image content based on API requirements. Returns a list.

    For Gemini, returns a list containing the prompt string and PIL Image objects.
    For other APIs, returns a formatted list/dictionary as required by their respective call functions.
    """
    if not image_paths:
        # Return prompt in a list if no images, to maintain list type consistency? No, call_api expects str if no images.
        return prompt

    if api_type == 'gemini':
        # Gemini: Load images as PIL objects
        content_parts = [prompt] # Start with the text prompt
        for path in image_paths:
            try:
                img = Image.open(path)
                img.load() # Load image data to catch errors early
                 # Optional: Add resizing or format conversion here if needed globally for Gemini
                 # Example: Convert to RGB if not already
                 # if img.mode != 'RGB':
                 #     img = img.convert('RGB')
                content_parts.append(img)
            except Exception as e:
                logging.error(f"Error opening or loading image {path} for Gemini: {e}")
                # Decide how to handle errors: skip image, raise exception, etc.
                # Skipping for now:
                continue
        return content_parts

    # --- Handling for other APIs (remains the same) --- 
    base64_images = [encode_image(path) for path in image_paths] # Encode only if not Gemini
    
    if api_type == 'anthropic':
        content = [{"type": "text", "text": prompt}]
        for img_data in base64_images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_data
                }
            })
        return content
        
    elif api_type == 'openai':
        content = [{"type": "text", "text": prompt}]
        for img_data in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_data}"
                }
            })
        return content

    elif api_type == 'ollama':
        content = [{"type": "text", "text": prompt}]
        for img_data in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_data}"
                }
            })
        return content
        
    else:
        # This case should ideally not be reached if checks are done earlier,
        # but serves as a fallback.
        raise ValueError(f"Unsupported API type for image handling: {api_type}")

def update_api_temperature(intensity: int) -> None:
    """Update the API temperature based on persona intensity.
    
    Args:
        intensity (int): Value between 0-100 representing persona intensity
    """
    api.temperature = intensity / 100.0
    logging.info(f"API temperature updated to {api.temperature}")

def update_api_top_p(top_p: float) -> None:
    """Update the API top_p value.
    
    Args:
        top_p (float): Top-p value between 0.0-1.0
    """
    api.top_p = top_p
    logging.info(f"API top_p updated to {api.top_p}")

async def retry_api_call(api_func, *args, max_retries=3, retry_delay=1, **kwargs):
    """Generic retry wrapper for API calls that handles 500 errors.
    
    Args:
        api_func: The API function to call
        *args: Positional arguments for the API function
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1)
        **kwargs: Keyword arguments for the API function
    """
    for attempt in range(max_retries):
        try:
            return await api_func(*args, **kwargs)
            
        except Exception as e:
            # Check if it's a 500 error (either APIError with status_code or error message containing 500)
            is_500_error = (
                (hasattr(e, 'status_code') and e.status_code == 500) or
                ('500' in str(e)) or
                ('Internal Server Error' in str(e)) or
                ('model runner has unexpectedly stopped' in str(e))
            )
            
            if is_500_error and attempt < max_retries - 1:
                logging.warning(f"API returned 500 error, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                # Re-raise the original exception
                raise

async def call_api(prompt, context="", system_prompt="", conversation_id=None, temperature=None, top_p=None, image_paths=None):
    """
    Args:
        temperature (float, optional): Override for global temperature. 
            If None, uses global temperature value.
        top_p (float, optional): Override for global top_p value.
    """
    # Use provided temperature or fall back to global
    current_temp = temperature if temperature is not None else api.temperature
    current_top_p = top_p if top_p is not None else api.top_p
    
    is_image = bool(image_paths)
    
    logging.info(f"API Call Settings - Model: {api.model_name}, Temperature: {current_temp:.2f}")
    print(f"{Fore.YELLOW}System Prompt: {system_prompt}")
    print(f"{Fore.CYAN}Input: {f'[Image] {prompt}' if is_image else prompt}")

    try:
        if is_image:
            formatted_content = prepare_image_content(prompt, image_paths, api.api_type)
        else:
            formatted_content = prompt

        # Use retry wrapper for all API calls
        async def make_api_call():
            if api.api_type == 'ollama':
                return await call_ollama_api(formatted_content, context, system_prompt, current_temp, current_top_p, is_image)
            elif api.api_type == 'openai':
                return await call_openai_api(formatted_content, context, system_prompt, current_temp, current_top_p, is_image)
            elif api.api_type == 'anthropic':
                return await call_anthropic_api(formatted_content, context, system_prompt, current_temp, current_top_p, is_image)
            elif api.api_type == 'vllm':
                return await call_vllm_api(formatted_content, context, system_prompt, current_temp, current_top_p)
            elif api.api_type == 'gemini':
                return await call_gemini_api(formatted_content, context, system_prompt, current_temp, current_top_p, is_image)
            else:
                raise ValueError(f"Unsupported API type: {api.api_type}")
        
        response = await retry_api_call(make_api_call)

        print(f"{Fore.GREEN}Output: {response}")

        # Calculate token counts for input and output
        input_text = f"{system_prompt}\n{context}\n{prompt}" if system_prompt or context else prompt
        input_tokens = count_tokens(input_text) if not is_image else 0  # Skip token count for images
        output_tokens = count_tokens(response)
        
        logging.info(f"Token Usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "api_type": api.api_type,
            "system_prompt": system_prompt,
            "context": context,
            "user_input": f"[Image] {prompt}" if is_image else prompt,
            "ai_output": response,
            "is_image": is_image,
            "num_images": len(image_paths) if image_paths else 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
        log_to_jsonl(log_data)

        return response
    except Exception as e:
        logging.error(f"Error in API call: {str(e)}")
        raise

async def call_vllm_api(content, context, system_prompt, temperature, top_p):
    logging.info(f"Calling vLLM API - Model: {api.model_name}, Temperature: {temperature:.2f}")
    async with aiohttp.ClientSession() as session:
        # Combine prompts if present
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n"
        if context:
            full_prompt += f"{context}\n"
        full_prompt += content

        # Add API key to headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('VLLM_API_KEY')}"  # Add API key from environment
        }
        
        data = {
            "model": api.model_name,
            "prompt": full_prompt,
            "max_tokens": 4096,
            "temperature": temperature
        }
        
        try:
            async with session.post(
                f"{api.api_base}/v1/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"vLLM API returned status {response.status}: {error_text}")
                
                result = await response.json()
                return result["choices"][0]["text"].strip()
        except aiohttp.ClientError as e:
            error_message = f"vLLM API request failed: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

async def call_ollama_api(content, context, system_prompt, temperature, top_p, is_image, max_retries=3, retry_delay=1):
    logging.info(f"Calling Ollama API - Model: {api.model_name}, Temperature: {temperature:.2f}")
    client = openai.AsyncOpenAI(
        base_url=f"{api.api_base}/v1",
        api_key="ollama"  # Required but ignored
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": content})
    
    try:
        response = await client.chat.completions.create(
            model=api.model_name,
            messages=messages,
            max_tokens=16384,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Ollama API error: {str(e)}")
        raise Exception(f"Ollama API call failed: {str(e)}")

async def call_openai_api(content, context, system_prompt, temperature, top_p, is_image):
    logging.info(f"Calling OpenAI API - Model: {api.model_name}, Temperature: {temperature:.2f}")
    client = openai.AsyncOpenAI(api_key=api.api_key)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": content})
    
    try:
        response = await client.chat.completions.create(
            model=api.model_name,
            messages=messages,
            max_tokens=16384,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        raise Exception(f"OpenAI API call failed: {str(e)}")

async def call_anthropic_api(content, context, system_prompt, temperature, top_p, is_image):
    """Call Anthropic API.
    
    Args:
        content: Message content
        context: Context string
        system_prompt: System prompt
        temperature: Temperature setting
        top_p: Top-p setting
        is_image: Whether content includes images
    """
    logging.info(f"Calling Anthropic API - Model: {api.model_name}, Temperature: {temperature:.2f}")
    client = anthropic.AsyncAnthropic(api_key=api.api_key)
    
    try:
        response = await client.messages.create(
            model=api.model_name,
            system=system_prompt,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=4096,
            temperature=temperature
        )
        return response.content[0].text.strip()
        
    except Exception as e:
        logging.error(f"Anthropic API error: {str(e)}")
        raise Exception(f"Anthropic API call failed: {str(e)}")

async def call_gemini_api(content, context, system_prompt, temperature, top_p, is_image):
    """Call Google Gemini API with support for text and image content.
    
    Args:
        content: Message content or image content
        context: Context string
        system_prompt: System prompt
        temperature: Temperature setting
        is_image: Whether content includes images
    """
    logging.info(f"Calling Gemini API - Model: {api.model_name}, Temperature: {temperature:.2f}")
    
    try:
        # Prepare content parts - The 'content' argument is now expected to be 
        # a list containing the prompt string and PIL.Image objects for Gemini.
        parts = []
        
        # Add system prompt and context if provided (as text parts)
        if system_prompt:
            parts.append(system_prompt)
        if context:
            parts.append(context)
            
        # Add main content (text and images from the list)
        if isinstance(content, list):
            for item in content:
                # Directly append text strings or PIL Image objects
                if isinstance(item, str) or isinstance(item, Image.Image):
                    parts.append(item)
                else:
                    # Log or handle unexpected item types in the list
                    logging.warning(f"Unexpected item type in Gemini content list: {type(item)}")
        elif isinstance(content, str):
             # Handle case where only text prompt was passed (no images)
             parts.append(content)
        else:
             raise ValueError("Invalid content format for Gemini API call")
            
        # Create content configuration
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=64,
            max_output_tokens=65536,
            response_mime_type="text/plain"
        )
        
        # Generate content asynchronously using asyncio.to_thread
        response = await asyncio.to_thread(
            api.gemini_client.models.generate_content,
            model=api.model_name,
            contents=parts,
            config=config
        )
        
        return response.text.strip()
        
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        raise Exception(f"Gemini API call failed: {str(e)}")

async def get_embeddings(text, provider=None, model=None):
    """Get embeddings from various API providers.
    
    Args:
        text (str): Text to get embeddings for
        provider (str, optional): Specific provider for embeddings ('openai', 'ollama', 'vllm')
            If None, uses global API_TYPE
        model (str, optional): Override default embedding model
        
    Returns:
        list: Vector embeddings
    """
    # Use provided provider or fall back to global API_TYPE
    embed_provider = provider or api.api_type

    if embed_provider == 'openai':
        client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        embed_model = model or "text-embedding-3-small"
        try:
            response = await client.embeddings.create(
                model=embed_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"OpenAI embeddings error: {str(e)}")
            raise Exception(f"OpenAI embeddings failed: {str(e)}")
            
    elif embed_provider == 'ollama':
        client = openai.AsyncOpenAI(
            base_url=f"{api.api_base}/v1",
            api_key="ollama"  # Required but ignored
        )
        embed_model = model or "all-minilm:latest"
        try:
            response = await client.embeddings.create(
                model=embed_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Ollama embeddings error: {str(e)}")
            raise Exception(f"Ollama embeddings failed: {str(e)}")
    
    elif embed_provider == 'vllm':
        embed_model = model or os.getenv('VLLM_EMBED_MODEL', 'jinaai/jina-embeddings-v2-base-en')
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "http://localhost:8080/embed",
                    json={
                        "model": embed_model,
                        "inputs": text
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"vLLM embeddings API returned status {response.status}: {error_text}")
                    
                    result = await response.json()
                    logging.debug(f"VLLM API Response: {result}")
                    
                    if isinstance(result, list):
                        return result[0]
                    elif isinstance(result, dict) and "embeddings" in result:
                        return result["embeddings"][0]
                    else:
                        return result
                        
            except Exception as e:
                logging.error(f"vLLM embeddings error: {str(e)}")
                raise Exception(f"vLLM embeddings failed: {str(e)}")
    
    else:
        raise ValueError(f"Embeddings not supported for provider: {embed_provider}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-API LLM Client')
    parser.add_argument('--api', required=True, choices=['ollama', 'openai', 'anthropic', 'vllm', 'gemini'],
                      help='API type to use')
    parser.add_argument('--model', help='Model name (optional)')
    # Add other arguments as needed