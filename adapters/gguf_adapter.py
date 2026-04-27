"""GGUF adapter for local model inference using llama-cpp-python."""

import os
import time
from typing import Any, Dict, List, Optional

from .base_adapter import BaseModelAdapter
from core.base import GenerationResult
from core.registry import register_adapter


@register_adapter("gguf")
class GGUFAdapter(BaseModelAdapter):
    """
    GGUF adapter for local model inference using llama-cpp-python.
    
    Features:
    - Loads GGUF format models locally
    - CPU and GPU (CUDA) support
    - Configurable context length
    - Supports quantized models (Q4, Q8, etc.)
    
    Model should be a path to a .gguf file or HuggingFace model ID
    that contains GGUF files (e.g., "unsloth/Qwen3.6-35B-A3B-GGUF").
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize GGUF adapter.
        
        Args:
            model_name: Path to .gguf file or HuggingFace model ID
            device: "cpu", "cuda", or "auto"
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all, 0 = CPU only)
            verbose: Enable verbose output from llama.cpp
        """
        super().__init__(
            model_name,
            device=device,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs
        )
        
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers if device != "cpu" else 0
        self.verbose = verbose
        
        self._model = None
        self._model_path = None
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the GGUF model."""
        try:
            from llama_cpp import Llama
            
            # Check if model_name is a local path or HF model ID
            if os.path.isfile(self.model_name):
                # Local GGUF file
                self._model_path = self.model_name
            elif self.model_name.endswith('.gguf'):
                # Local path without checking existence (will fail on load if missing)
                self._model_path = self.model_name
            else:
                # HuggingFace model ID - need to download GGUF files
                self._model_path = self._download_from_hf(self.model_name)
            
            # Ensure n_ctx is an integer (CLI args may pass as string)
            self.n_ctx = int(self.n_ctx) if isinstance(self.n_ctx, str) else self.n_ctx
            
            # Ensure n_gpu_layers is an integer (CLI args may pass as string)
            self.n_gpu_layers = int(self.n_gpu_layers) if isinstance(self.n_gpu_layers, str) else self.n_gpu_layers
            
            # Detect if this is a BF16 model and adjust context accordingly
            # BF16 models often have very large n_ctx_train (e.g., 262144)
            # We need to use a larger n_ctx to avoid 0-token generation issues
            if 'BF16' in self._model_path.upper() or 'bf16' in self._model_path.lower():
                print(f"Detected BF16 model. Adjusting context from {self.n_ctx} to 32768...")
                self.n_ctx = max(self.n_ctx, 32768)  # Use at least 32K context for BF16
                print(f"Using n_ctx={self.n_ctx} for BF16 model")
            
            # Load the model
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )
            
            # Log model metadata for debugging
            if hasattr(self._model, 'n_ctx_train'):
                print(f"Model n_ctx_train: {self._model.n_ctx_train}, using n_ctx: {self.n_ctx}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model {self.model_name}: {e}")
    
    def _download_from_hf(self, model_id: str) -> str:
        """Download GGUF files from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # List files in the repo
            files = list_repo_files(model_id)
            
            # Find GGUF files
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                raise ValueError(f"No GGUF files found in {model_id}")
            
            # Prefer smaller quantized versions for CPU inference
            # Priority: Q4_K_M, Q4_K_S, Q5_K_M, Q8_0, or smallest file
            preferred_patterns = ['Q4_K_M', 'Q4_K_S', 'Q5_K_M', 'Q8_0']
            selected_file = None
            
            for pattern in preferred_patterns:
                matches = [f for f in gguf_files if pattern in f]
                if matches:
                    selected_file = matches[0]
                    break
            
            if not selected_file:
                # Fall back to smallest GGUF file
                selected_file = min(gguf_files, key=lambda f: self._get_file_size(model_id, f))
            
            # Download the file
            print(f"Downloading {selected_file} from {model_id}...")
            model_path = hf_hub_download(repo_id=model_id, filename=selected_file)
            print(f"Downloaded to: {model_path}")
            
            return model_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download GGUF model from {model_id}: {e}")
    
    def _get_file_size(self, model_id: str, filename: str) -> int:
        """Get file size from HuggingFace repo."""
        try:
            from huggingface_hub import get_hf_file_metadata
            metadata = get_hf_file_metadata(f"https://huggingface.co/{model_id}/resolve/main/{filename}")
            return metadata.size
        except:
            return float('inf')  # Unknown size, treat as large
    
    def _generate_impl(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate text using llama-cpp.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
        
        Returns:
            GenerationResult
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        # Prepare generation parameters
        generation_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 0.0,
        }
        
        if temperature > 0:
            generation_kwargs["top_p"] = top_p
        
        if stop_sequences:
            generation_kwargs["stop"] = stop_sequences
        
        try:
            # Generate
            output = self._model(prompt, **generation_kwargs)
            
            # Debug: print raw output structure
            # print(f"DEBUG: output type={type(output)}, keys={output.keys() if isinstance(output, dict) else 'N/A'}")
            
            # Handle different output formats from llama-cpp
            if isinstance(output, dict):
                # Standard format: {'choices': [{'text': '...', 'finish_reason': '...'}]}
                choices = output.get('choices', [])
                if choices and len(choices) > 0:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        generated_text = choice.get('text', '')
                        finish_reason = choice.get('finish_reason', 'stop')
                    else:
                        generated_text = str(choice) if choice else ''
                        finish_reason = 'stop'
                else:
                    generated_text = ''
                    finish_reason = 'error'
                
                # Count tokens from logprobs if available
                tokens_generated = 0
                if choices and len(choices) > 0:
                    logprobs = choices[0].get('logprobs', {})
                    if logprobs and isinstance(logprobs, dict):
                        tokens_generated = len(logprobs.get('tokens', []))
                
                if tokens_generated == 0 and generated_text:
                    tokens_generated = len(generated_text) // 4  # Rough estimate
                    
            elif isinstance(output, str):
                # Direct string output
                generated_text = output
                finish_reason = 'stop'
                tokens_generated = len(generated_text) // 4
            else:
                # Unknown format, convert to string
                generated_text = str(output) if output else ''
                finish_reason = 'stop'
                tokens_generated = len(generated_text) // 4
            
            return GenerationResult(
                text=generated_text,
                tokens_generated=tokens_generated,
                generation_time=0.0,  # Will be set by parent
                finish_reason=finish_reason,
                logprobs=None,
            )
            
        except Exception as e:
            # Return error result on exception
            return GenerationResult(
                text="",
                tokens_generated=0,
                generation_time=0.0,
                finish_reason="error",
                logprobs=None,
            )
    
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
        
        Note: llama-cpp doesn't support true batching, so we process sequentially.
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
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using the model's tokenizer."""
        if self._model:
            return len(self._model.tokenize(text.encode('utf-8')))
        return len(text) // 4
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self._model is not None
    
    def _fetch_model_info(self) -> Dict[str, Any]:
        """Fetch model information."""
        try:
            return {
                "model_name": self.model_name,
                "model_path": self._model_path,
                "context_length": self.n_ctx,
                "supports_chat_template": False,  # GGUF models don't have built-in chat templates
                "device": self.device,
                "n_gpu_layers": self.n_gpu_layers,
            }
        except Exception:
            return {
                "model_name": self.model_name,
                "context_length": self.n_ctx,
                "supports_chat_template": False,
                "device": self.device,
            }