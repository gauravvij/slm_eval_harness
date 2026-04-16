"""Base adapter interface for model adapters."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
import time

from core.base import ModelAdapter, GenerationResult


class BaseModelAdapter(ModelAdapter):
    """
    Base class for all model adapters.
    
    Provides common functionality like:
    - Request retry logic
    - Token counting estimates
    - Batch processing helpers
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.device = device
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._model_info: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    def _generate_impl(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Implementation of generation. Override in subclasses.
        
        This method should handle the actual model inference.
        """
        pass
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate text with retry logic.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
        
        Returns:
            GenerationResult
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                result = self._generate_impl(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                )
                result.generation_time = time.time() - start_time
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    # Return error result on final failure
                    return GenerationResult(
                        text="",
                        tokens_generated=0,
                        generation_time=0.0,
                        finish_reason="error",
                        logprobs=None,
                    )
        
        # Should not reach here, but just in case
        raise last_error or RuntimeError("Generation failed")
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[GenerationResult]:
        """
        Generate text for a batch of prompts.
        
        Default implementation processes sequentially.
        Override for parallel/batch processing.
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
        """
        Estimate the number of tokens in text.
        
        Default: rough estimate of 4 characters per token.
        Override for more accurate estimates.
        """
        return len(text) // 4
    
    def is_available(self) -> bool:
        """Check if the model is available. Override in subclasses."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        if self._model_info is None:
            self._model_info = self._fetch_model_info()
        return self._model_info
    
    @abstractmethod
    def _fetch_model_info(self) -> Dict[str, Any]:
        """Fetch model information. Override in subclasses."""
        pass
    
    def get_context_length(self) -> int:
        """Get the model's context length."""
        info = self.get_model_info()
        return info.get("context_length", 4096)
    
    def supports_chat_template(self) -> bool:
        """Check if the model supports chat templates."""
        info = self.get_model_info()
        return info.get("supports_chat_template", False)
