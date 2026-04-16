"""Chat templates for major model families."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re


@dataclass
class ChatTemplate:
    """Chat template configuration for a model family."""
    name: str
    system_format: str
    user_format: str
    assistant_format: str
    system_in_user: bool = False
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []
    
    def format_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> str:
        """
        Format a list of messages into a prompt string.
        
        Args:
            messages: List of dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add the assistant prefix at the end
        
        Returns:
            Formatted prompt string
        """
        formatted = []
        system_content = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_content = content
            elif role == "user":
                if system_content and self.system_in_user:
                    # Some models expect system in first user message
                    content = f"{system_content}\n\n{content}"
                    system_content = None
                formatted.append(self.user_format.format(content=content))
            elif role == "assistant":
                formatted.append(self.assistant_format.format(content=content))
        
        # Handle system message if not already included
        if system_content:
            formatted.insert(0, self.system_format.format(content=system_content))
        
        result = "".join(formatted)
        
        if add_generation_prompt:
            # Add assistant prefix for generation
            result += self.assistant_format.split("{content}")[0]
        
        return result


# Chat templates for major model families
CHAT_TEMPLATES = {
    "llama": ChatTemplate(
        name="llama",
        system_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        user_format="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        assistant_format="<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        stop_sequences=["<|eot_id|>", "<|end_of_text|>"],
    ),
    
    "llama-2": ChatTemplate(
        name="llama-2",
        system_format="<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n",
        user_format="{content} [/INST] ",
        assistant_format="{content} </s><s>[INST] ",
        stop_sequences=["</s>"],
    ),
    
    "qwen": ChatTemplate(
        name="qwen",
        system_format="<|im_start|>system\n{content}<|im_end|>\n",
        user_format="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_format="<|im_start|>assistant\n{content}<|im_end|>\n",
        stop_sequences=["<|im_end|>"],
    ),
    
    "qwen2": ChatTemplate(
        name="qwen2",
        system_format="<|im_start|>system\n{content}<|im_end|>\n",
        user_format="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_format="<|im_start|>assistant\n{content}<|im_end|>\n",
        stop_sequences=["<|im_end|>"],
    ),
    
    "gemma": ChatTemplate(
        name="gemma",
        system_format="<start_of_turn>model\n{content}<end_of_turn>\n",
        user_format="<start_of_turn>user\n{content}<end_of_turn>\n",
        assistant_format="<start_of_turn>model\n{content}<end_of_turn>\n",
        stop_sequences=["<end_of_turn>"],
    ),
    
    "gemma-2": ChatTemplate(
        name="gemma-2",
        system_format="<start_of_turn>model\n{content}<end_of_turn>\n",
        user_format="<start_of_turn>user\n{content}<end_of_turn>\n",
        assistant_format="<start_of_turn>model\n{content}<end_of_turn>\n",
        stop_sequences=["<end_of_turn>"],
    ),
    
    "phi": ChatTemplate(
        name="phi",
        system_format="<|system|>\n{content}<|end|>\n",
        user_format="<|user|>\n{content}<|end|>\n",
        assistant_format="<|assistant|>\n{content}<|end|>\n",
        stop_sequences=["<|end|>"],
    ),
    
    "phi-3": ChatTemplate(
        name="phi-3",
        system_format="<|system|>\n{content}<|end|>\n",
        user_format="<|user|>\n{content}<|end|>\n",
        assistant_format="<|assistant|>\n{content}<|end|>\n",
        stop_sequences=["<|end|>"],
    ),
    
    "mistral": ChatTemplate(
        name="mistral",
        system_format="<s>[INST] {content}\n\n",
        user_format="{content} [/INST] ",
        assistant_format="{content} </s><s>[INST] ",
        stop_sequences=["</s>"],
    ),
    
    "mixtral": ChatTemplate(
        name="mixtral",
        system_format="<s>[INST] {content}\n\n",
        user_format="{content} [/INST] ",
        assistant_format="{content} </s><s>[INST] ",
        stop_sequences=["</s>"],
    ),
    
    "chatml": ChatTemplate(
        name="chatml",
        system_format="<|im_start|>system\n{content}<|im_end|>\n",
        user_format="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_format="<|im_start|>assistant\n{content}<|im_end|>\n",
        stop_sequences=["<|im_end|>"],
    ),
    
    "default": ChatTemplate(
        name="default",
        system_format="System: {content}\n\n",
        user_format="User: {content}\n",
        assistant_format="Assistant: {content}\n",
        stop_sequences=[],
    ),
}


# Model name patterns to template mappings
MODEL_PATTERN_MAPPINGS = [
    (r"llama-?3.*instruct", "llama"),
    (r"llama-?2.*chat", "llama-2"),
    (r"qwen-?2", "qwen2"),
    (r"qwen", "qwen"),
    (r"gemma-?2", "gemma-2"),
    (r"gemma", "gemma"),
    (r"phi-?3", "phi-3"),
    (r"phi", "phi"),
    (r"mistral", "mistral"),
    (r"mixtral", "mixtral"),
]


class ChatTemplateManager:
    """Manager for chat templates with auto-detection."""
    
    def __init__(self):
        self.templates = CHAT_TEMPLATES.copy()
    
    def get_template(self, name: str) -> ChatTemplate:
        """Get a template by name."""
        if name not in self.templates:
            raise ValueError(f"Unknown template: {name}. Available: {list(self.templates.keys())}")
        return self.templates[name]
    
    def detect_template(self, model_name: str) -> ChatTemplate:
        """
        Auto-detect the appropriate template for a model.
        
        Args:
            model_name: Name of the model (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        
        Returns:
            ChatTemplate for the model
        """
        model_lower = model_name.lower()
        
        for pattern, template_name in MODEL_PATTERN_MAPPINGS:
            if re.search(pattern, model_lower):
                return self.templates[template_name]
        
        # Default fallback
        return self.templates["default"]
    
    def register_template(self, name: str, template: ChatTemplate):
        """Register a new template."""
        self.templates[name] = template
    
    def format_prompt(
        self,
        model_name: str,
        prompt: str,
        system_message: Optional[str] = None,
    ) -> str:
        """
        Format a simple prompt using the appropriate template.
        
        Args:
            model_name: Name of the model
            prompt: User prompt
            system_message: Optional system message
        
        Returns:
            Formatted prompt string
        """
        template = self.detect_template(model_name)
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        return template.format_messages(messages, add_generation_prompt=True)
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())


# Global instance
template_manager = ChatTemplateManager()


def get_chat_template(name_or_model: str) -> ChatTemplate:
    """
    Get a chat template by name or auto-detect from model name.
    
    Args:
        name_or_model: Template name or model identifier
    
    Returns:
        ChatTemplate instance
    """
    if name_or_model in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[name_or_model]
    return template_manager.detect_template(name_or_model)


def format_prompt(
    model_name: str,
    prompt: str,
    system_message: Optional[str] = None,
) -> str:
    """Convenience function to format a prompt."""
    return template_manager.format_prompt(model_name, prompt, system_message)
