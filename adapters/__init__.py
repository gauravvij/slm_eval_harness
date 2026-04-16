"""Model adapters for different inference backends."""

from .base_adapter import BaseModelAdapter
from .hf_adapter import HuggingFaceAdapter
from .ollama_adapter import OllamaAdapter
from .chat_templates import ChatTemplateManager, get_chat_template

__all__ = [
    "BaseModelAdapter",
    "HuggingFaceAdapter",
    "OllamaAdapter",
    "ChatTemplateManager",
    "get_chat_template",
]
