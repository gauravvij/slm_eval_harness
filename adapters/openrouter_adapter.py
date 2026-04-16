"""OpenRouter adapter for cloud model inference via OpenRouter API."""

import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from .base_adapter import BaseModelAdapter
from core.base import GenerationResult
from core.registry import register_adapter


@register_adapter("openrouter")
class OpenRouterAdapter(BaseModelAdapter):
    """
    OpenRouter adapter for cloud-based model inference.
    
    Features:
    - Connects to OpenRouter API (https://openrouter.ai)
    - Supports chat completions API
    - Handles various model providers through OpenRouter
    - Automatic retries with exponential backoff
    
    API Key should be provided via config file at:
    ~/.config/openrouter/config
    """
    
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        """
        Initialize OpenRouter adapter.
        
        Args:
            model_name: Model identifier (e.g., "google/gemma-4-26b-it", "qwen/qwen-3.5-9b")
            device: Ignored for cloud models (kept for API compatibility)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (seconds)
            api_key: OpenRouter API key (if None, reads from ~/.config/openrouter/config)
            base_url: OpenRouter API base URL
            timeout: Request timeout in seconds
        """
        super().__init__(
            model_name,
            device=device,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs
        )
        
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        
        # Load API key
        self.api_key = api_key or self._load_api_key()
        if not self.api_key:
            raise RuntimeError(
                "OpenRouter API key not found. Please provide api_key parameter "
                "or create ~/.config/openrouter/config with {'api_key': 'your-key'}"
            )
        
        # Verify API connectivity
        if not self._check_api():
            raise RuntimeError(
                f"OpenRouter API not accessible at {self.base_url}. "
                "Please check your internet connection and API key."
            )
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from config file."""
        import os
        config_path = os.path.expanduser("~/.config/openrouter/config")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("api_key")
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            # Try reading as plain text
            try:
                with open(config_path, 'r') as f:
                    content = f.read().strip()
                    # Try to extract from JSON-like format
                    if '"api_key"' in content:
                        import re
                        match = re.search(r'"api_key"\s*:\s*"([^"]+)"', content)
                        if match:
                            return match.group(1)
                    return content if content else None
            except Exception:
                return None
        except Exception:
            return None
    
    def _check_api(self) -> bool:
        """Check if OpenRouter API is accessible."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                method="GET"
            )
            req.add_header("Authorization", f"Bearer {self.api_key}")
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    def _generate_impl(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate text using OpenRouter chat completions API.
        
        OpenRouter uses OpenAI-compatible chat completions format.
        """
        # Build messages for chat API
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                    'HTTP-Referer': 'https://slm-eval-harness.local',  # Required by OpenRouter
                    'X-Title': 'SLM Evaluation Harness',  # Optional but recommended
                },
                method="POST"
            )
            
            start_time = time.time()
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                generation_time = time.time() - start_time
                
                # Extract response from OpenAI-compatible format
                choices = data.get('choices', [])
                if not choices:
                    raise RuntimeError("No choices returned from OpenRouter API")
                
                choice = choices[0]
                message = choice.get('message', {})
                content = message.get('content', '')
                finish_reason = choice.get('finish_reason', 'stop')
                
                # Estimate tokens (OpenRouter may not provide token counts)
                tokens_generated = len(content) // 4  # Rough estimate
                
                return GenerationResult(
                    text=content,
                    tokens_generated=tokens_generated,
                    generation_time=generation_time,
                    finish_reason=finish_reason,
                    logprobs=None,
                )
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            try:
                error_data = json.loads(error_body)
                error_msg = error_data.get('error', {}).get('message', error_body)
            except json.JSONDecodeError:
                error_msg = error_body
            raise RuntimeError(f"OpenRouter API error: {e.code} - {error_msg}")
        except Exception as e:
            raise RuntimeError(f"OpenRouter generation failed: {e}")
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate using chat API with message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Roles: 'system', 'user', 'assistant'
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                    'HTTP-Referer': 'https://slm-eval-harness.local',
                    'X-Title': 'SLM Evaluation Harness',
                },
                method="POST"
            )
            
            start_time = time.time()
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                generation_time = time.time() - start_time
                
                choices = data.get('choices', [])
                if not choices:
                    raise RuntimeError("No choices returned from OpenRouter API")
                
                choice = choices[0]
                message = choice.get('message', {})
                content = message.get('content', '')
                finish_reason = choice.get('finish_reason', 'stop')
                
                tokens_generated = len(content) // 4
                
                return GenerationResult(
                    text=content,
                    tokens_generated=tokens_generated,
                    generation_time=generation_time,
                    finish_reason=finish_reason,
                    logprobs=None,
                )
                
        except Exception as e:
            raise RuntimeError(f"OpenRouter chat generation failed: {e}")
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[GenerationResult]:
        """
        Generate text for a batch of prompts (sequential).
        
        Note: OpenRouter doesn't support true batching, so we process sequentially.
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
            )
            results.append(result)
        return results
    
    def is_available(self) -> bool:
        """Check if OpenRouter API is available."""
        return self._check_api()
    
    def health_check(self) -> bool:
        """Health check for OpenRouter API."""
        return self.is_available()
    
    def _fetch_model_info(self) -> Dict[str, Any]:
        """Fetch model information from OpenRouter."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                method="GET"
            )
            req.add_header("Authorization", f"Bearer {self.api_key}")
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                # Find our model in the list
                models = data.get('data', [])
                for model in models:
                    if model.get('id') == self.model_name:
                        return {
                            "model_name": self.model_name,
                            "context_length": model.get('context_length', 4096),
                            "pricing": model.get('pricing', {}),
                            "supports_chat_template": True,
                            "description": model.get('description', ''),
                            "architecture": model.get('architecture', {}),
                        }
                
                # Model not found in list, return defaults
                return {
                    "model_name": self.model_name,
                    "context_length": 4096,
                    "supports_chat_template": True,
                }
                
        except Exception:
            return {
                "model_name": self.model_name,
                "context_length": 4096,
                "supports_chat_template": True,
            }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models from OpenRouter."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                method="GET"
            )
            req.add_header("Authorization", f"Bearer {self.api_key}")
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data.get('data', [])
                
        except Exception:
            return []
    
    def get_credits(self) -> Optional[Dict[str, Any]]:
        """Get account credits information (if available)."""
        # OpenRouter doesn't have a standard credits endpoint
        # This is a placeholder for future API support
        return None
