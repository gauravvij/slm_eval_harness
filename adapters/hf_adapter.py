"""HuggingFace Transformers adapter with chat template auto-detection."""

import os
import re
import time
from typing import Any, Dict, List, Optional
import warnings

from .base_adapter import BaseModelAdapter
from .chat_templates import ChatTemplateManager
from core.base import GenerationResult
from core.registry import register_adapter

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)


@register_adapter("hf")
class HuggingFaceAdapter(BaseModelAdapter):
    """
    HuggingFace Transformers adapter.
    
    Supports:
    - Auto chat template detection
    - Device placement (cuda/cpu)
    - Quantization support
    - Batch generation
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        trust_remote_code: bool = True,
        torch_dtype: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        super().__init__(
            model_name,
            device=device,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs
        )
        
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self._model = None
        self._tokenizer = None
        self._template_manager = ChatTemplateManager()
        self._chat_template = None
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="left",
            )
            
            # Set pad token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Detect chat template
            self._detect_chat_template()
            
            # Load model
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
            }
            
            # Handle quantization
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = "auto"
            elif self.load_in_8bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_kwargs["device_map"] = "auto"
            else:
                # Handle dtype
                if self.torch_dtype:
                    if self.torch_dtype == "float16":
                        model_kwargs["torch_dtype"] = torch.float16
                    elif self.torch_dtype == "bfloat16":
                        model_kwargs["torch_dtype"] = torch.bfloat16
                    elif self.torch_dtype == "float32":
                        model_kwargs["torch_dtype"] = torch.float32
                
                if self.device != "auto":
                    model_kwargs["device_map"] = self.device
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Set to eval mode
            self._model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def _detect_chat_template(self):
        """Detect and set up the appropriate chat template."""
        # Check if tokenizer has a built-in chat template
        if hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template:
            self._chat_template = "tokenizer_builtin"
            return
        
        # Auto-detect from model name
        template = self._template_manager.detect_template(self.model_name)
        self._chat_template = template.name
    
    def _format_prompt(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Format prompt using the appropriate chat template."""
        if self._chat_template == "tokenizer_builtin":
            # Use tokenizer's built-in chat template
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Use our template manager
            return self._template_manager.format_prompt(
                self.model_name,
                prompt,
                system_message
            )
    
    def _generate_impl(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate text using the model."""
        import torch
        
        # Format prompt
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokenize
        inputs = self._tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Move to device
        if self.device != "auto" and not (self.load_in_8bit or self.load_in_4bit):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        
        # Handle stop sequences
        if stop_sequences:
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopSequenceCriteria(StoppingCriteria):
                def __init__(self, tokenizer, stop_sequences):
                    self.tokenizer = tokenizer
                    self.stop_sequences = stop_sequences
                    self.stop_token_ids = [
                        tokenizer.encode(seq, add_special_tokens=False)
                        for seq in stop_sequences
                    ]
                
                def __call__(self, input_ids, scores, **kwargs):
                    for stop_ids in self.stop_token_ids:
                        if len(input_ids[0]) >= len(stop_ids):
                            if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                                return True
                    return False
            
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopSequenceCriteria(self._tokenizer, stop_sequences)
            ])
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                **generation_kwargs
            )
        
        # Decode
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        # Check finish reason
        finish_reason = "stop"
        if len(generated_tokens) >= max_tokens:
            finish_reason = "length"
        
        return GenerationResult(
            text=generated_text,
            tokens_generated=len(generated_tokens),
            generation_time=0.0,  # Will be set by parent
            finish_reason=finish_reason,
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
        """Generate text for a batch of prompts."""
        import torch
        
        # Format prompts
        formatted_prompts = [self._format_prompt(p) for p in prompts]
        
        # Tokenize
        inputs = self._tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Move to device
        if self.device != "auto" and not (self.load_in_8bit or self.load_in_4bit):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                **generation_kwargs
            )
        
        # Decode results
        results = []
        input_lengths = inputs["input_ids"].shape[1]
        
        for i, output in enumerate(outputs):
            input_length = (inputs["input_ids"][i] != self._tokenizer.pad_token_id).sum().item()
            generated_tokens = output[input_length:]
            generated_text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            finish_reason = "stop"
            if len(generated_tokens) >= max_tokens:
                finish_reason = "length"
            
            results.append(GenerationResult(
                text=generated_text,
                tokens_generated=len(generated_tokens),
                generation_time=0.0,
                finish_reason=finish_reason,
                logprobs=None,
            ))
        
        return results
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using the tokenizer."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return super().estimate_tokens(text)
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self._model is not None and self._tokenizer is not None
    
    def _fetch_model_info(self) -> Dict[str, Any]:
        """Fetch model information."""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            
            return {
                "model_name": self.model_name,
                "model_type": getattr(config, "model_type", "unknown"),
                "context_length": getattr(config, "max_position_embeddings", 4096),
                "vocab_size": getattr(config, "vocab_size", 0),
                "supports_chat_template": self._chat_template is not None,
                "chat_template": self._chat_template,
                "device": self.device,
            }
        except Exception:
            return {
                "model_name": self.model_name,
                "model_type": "unknown",
                "context_length": 4096,
                "vocab_size": 0,
                "supports_chat_template": False,
                "chat_template": None,
                "device": self.device,
            }
