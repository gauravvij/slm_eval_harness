"""Ollama adapter for local model inference."""

import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from .base_adapter import BaseModelAdapter
from core.base import GenerationResult
from core.registry import register_adapter


@register_adapter("ollama")
class OllamaAdapter(BaseModelAdapter):
    """
    Ollama adapter for local model inference.
    
    Features:
    - Connects to local Ollama server
    - Supports streaming and non-streaming generation
    - Handles chat and completion modes
    - Automatic model pulling
    """
    
    DEFAULT_HOST = "http://localhost:11434"
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        host: Optional[str] = None,
        timeout: float = 120.0,
        auto_pull: bool = True,
        **kwargs
    ):
        super().__init__(
            model_name,
            device=device,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs
        )
        
        self.host = host or self.DEFAULT_HOST
        self.timeout = timeout
        self.auto_pull = auto_pull
        
        # Check if Ollama is available
        if not self._check_server():
            raise RuntimeError(
                f"Ollama server not available at {self.host}. "
                "Please start Ollama with 'ollama serve'"
            )
        
        # Check/pull model if needed
        if auto_pull and not self._check_model():
            self._pull_model()
    
    def _check_server(self) -> bool:
        """Check if Ollama server is running."""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
    
    def _check_model(self) -> bool:
        """Check if the model is available locally."""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = data.get('models', [])
                return any(m.get('name') == self.model_name for m in models)
        except Exception:
            return False
    
    def _pull_model(self) -> bool:
        """Pull the model from Ollama registry."""
        try:
            print(f"Pulling model {self.model_name} from Ollama...")
            req = urllib.request.Request(
                f"{self.host}/api/pull",
                data=json.dumps({"name": self.model_name}).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=300) as response:
                # Stream the response to show progress
                for line in response:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'status' in data:
                            if 'completed' in data.get('status', '').lower():
                                print(f"Model {self.model_name} pulled successfully")
                                return True
                    except json.JSONDecodeError:
                        continue
            return True
        except Exception as e:
            print(f"Failed to pull model: {e}")
            return False
    
    def _generate_impl(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate text using Ollama API."""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method="POST"
            )
            
            start_time = time.time()
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                generation_time = time.time() - start_time
                
                return GenerationResult(
                    text=data.get('response', ''),
                    tokens_generated=data.get('eval_count', 0),
                    generation_time=generation_time,
                    finish_reason=data.get('done_reason', 'stop'),
                    logprobs=None,
                )
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise RuntimeError(f"Ollama API error: {e.code} - {error_body}")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate using chat API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        try:
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method="POST"
            )
            
            start_time = time.time()
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                generation_time = time.time() - start_time
                
                message = data.get('message', {})
                
                return GenerationResult(
                    text=message.get('content', ''),
                    tokens_generated=data.get('eval_count', 0),
                    generation_time=generation_time,
                    finish_reason=data.get('done_reason', 'stop'),
                    logprobs=None,
                )
                
        except Exception as e:
            raise RuntimeError(f"Ollama chat generation failed: {e}")
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[GenerationResult]:
        """Generate text for a batch of prompts (sequential)."""
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
        """Check if Ollama server and model are available."""
        return self._check_server() and self._check_model()
    
    def health_check(self) -> bool:
        """Health check for Ollama server and model availability."""
        return self.is_available()
    
    def health_check(self) -> bool:
        """Health check for Ollama server and model availability."""
        return self.is_available()
    
    def _fetch_model_info(self) -> Dict[str, Any]:
        """Fetch model information from Ollama."""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/show",
                data=json.dumps({"name": self.model_name}).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                return {
                    "model_name": self.model_name,
                    "modelfile": data.get('modelfile', ''),
                    "parameters": data.get('parameters', ''),
                    "template": data.get('template', ''),
                    "context_length": 4096,  # Default, may vary by model
                    "supports_chat_template": True,
                }
                
        except Exception:
            return {
                "model_name": self.model_name,
                "context_length": 4096,
                "supports_chat_template": True,
            }
    
    def list_local_models(self) -> List[Dict[str, Any]]:
        """List all locally available models."""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/tags",
                method="GET"
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data.get('models', [])
                
        except Exception:
            return []
    
    def delete_model(self) -> bool:
        """Delete the model from local Ollama."""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/delete",
                data=json.dumps({"name": self.model_name}).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method="DELETE"
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status == 200
                
        except Exception:
            return False
