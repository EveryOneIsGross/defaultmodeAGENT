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
import base64
from io import BytesIO
from PIL import Image
import mimetypes

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
        api.model_name = args.model or os.getenv('OPENAI_MODEL_NAME', 'chatgpt-4o-latest')
        openai.api_key = api.api_key
    elif api.api_type == 'anthropic':
        api.api_key = os.getenv('ANTHROPIC_API_KEY')
        api.model_name = args.model or os.getenv('ANTHROPIC_MODEL_NAME', 'claude-3-5-sonnet-20241022')
    elif api.api_type == 'vllm':
        api.api_base = os.getenv('VLLM_API_BASE', 'http://localhost:4000')
        api.model_name = args.model or os.getenv('VLLM_MODEL_NAME', 'hermes-testing/Hermes-3-Pro-RC2-e4')
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

def prepare_image_content(prompt: str, image_paths: list, api_type: str) -> dict:
    """Prepare image content based on API requirements"""
    if not image_paths:
        return prompt

    base64_images = [encode_image(path) for path in image_paths]
    
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
        if len(base64_images) > 1:
            logging.warning("Ollama only supports one image per request. Using first image.")
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[0]}"
                }
            }
        ]
    
    else:
        raise ValueError(f"Unsupported API type for image handling: {api_type}")

def update_api_temperature(intensity: int) -> None:
    """Update the API temperature based on persona intensity.
    
    Args:
        intensity (int): Value between 0-100 representing persona intensity
    """
    api.temperature = intensity / 100.0
    logging.info(f"API temperature updated to {api.temperature}")

async def call_api(prompt, context="", system_prompt="", conversation_id=None, temperature=None, image_paths=None):
    """
    Args:
        temperature (float, optional): Override for global temperature. 
            If None, uses global temperature value.
    """
    # Use provided temperature or fall back to global
    current_temp = temperature if temperature is not None else api.temperature
    
    is_image = bool(image_paths)
    
    logging.info(f"API Call Settings - Model: {api.model_name}, Temperature: {current_temp:.2f}")
    print(f"{Fore.YELLOW}System Prompt: {system_prompt}")
    print(f"{Fore.CYAN}Input: {f'[Image] {prompt}' if is_image else prompt}")

    try:
        if is_image:
            formatted_content = prepare_image_content(prompt, image_paths, api.api_type)
        else:
            formatted_content = prompt

        if api.api_type == 'ollama':
            response = await call_ollama_api(formatted_content, context, system_prompt, current_temp, is_image)
        elif api.api_type == 'openai':
            response = await call_openai_api(formatted_content, context, system_prompt, current_temp, is_image)
        elif api.api_type == 'anthropic':
            response = await call_anthropic_api(formatted_content, context, system_prompt, current_temp, is_image)
        elif api.api_type == 'vllm':
            response = await call_vllm_api(formatted_content, context, system_prompt, current_temp)
        else:
            raise ValueError(f"Unsupported API type: {api.api_type}")

        print(f"{Fore.GREEN}Output: {response}")

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "api_type": api.api_type,
            "system_prompt": system_prompt,
            "context": context,
            "user_input": f"[Image] {prompt}" if is_image else prompt,
            "ai_output": response,
            "is_image": is_image,
            "num_images": len(image_paths) if image_paths else 0
        }
        log_to_jsonl(log_data)

        return response
    except Exception as e:
        logging.error(f"Error in API call: {str(e)}")
        raise

async def call_vllm_api(content, context, system_prompt, temperature):
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

async def call_ollama_api(content, context, system_prompt, temperature, is_image):
    logging.info(f"Calling Ollama API - Model: {api.model_name}, Temperature: {temperature + 0.5:.2f}")
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
            temperature=temperature + 0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Ollama API error: {str(e)}")
        raise Exception(f"Ollama API call failed: {str(e)}")

async def call_openai_api(content, context, system_prompt, temperature, is_image):
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

async def call_anthropic_api(content, context, system_prompt, temperature, is_image, max_retries=3, retry_delay=1):
    """Call Anthropic API with retry logic.
    
    Args:
        content: Message content
        context: Context string
        system_prompt: System prompt
        temperature: Temperature setting
        is_image: Whether content includes images
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1)
    """
    logging.info(f"Calling Anthropic API - Model: {api.model_name}, Temperature: {temperature:.2f}")
    client = anthropic.AsyncAnthropic(api_key=api.api_key)
    
    for attempt in range(max_retries):
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
            
        except anthropic.APIError as e:
            if e.status_code == 500:  # Internal Server Error
                if attempt < max_retries - 1:  # If not the last attempt
                    logging.warning(f"Anthropic API returned 500 error, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    error_message = f"Anthropic API failed after {max_retries} attempts: {str(e)}"
                    logging.error(error_message)
                    raise Exception(error_message)
            else:
                error_message = f"Anthropic API error: {str(e)}"
                logging.error(error_message)
                raise Exception(error_message)
                
        except Exception as e:
            error_message = f"Unexpected error in Anthropic API call: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

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
        embed_model = model or "nomic-embed-text"
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
    import asyncio
    
    parser = argparse.ArgumentParser(description='Multi-API LLM Client')
    parser.add_argument('--api', required=True, choices=['ollama', 'openai', 'anthropic', 'vllm'],
                      help='API type to use')
    parser.add_argument('--model', help='Model name (optional)')
    # Add other arguments as needed