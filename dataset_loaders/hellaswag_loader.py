"""HellaSwag dataset loader with multiple choice options handling."""

from typing import Any, Dict, List, Optional

from .base_loader import BaseDatasetLoader
from core.base import Sample, TaskConfig
from core.registry import register_dataset_loader


@register_dataset_loader("hellaswag")
class HellaSwagLoader(BaseDatasetLoader):
    """
    Loader for HellaSwag commonsense reasoning benchmark.
    
    Dataset fields:
    - ctx: Context (first part of the scenario)
    - ctx_a: Context before the event
    - ctx_b: Context after the event
    - endings: List of 4 possible endings (A, B, C, D)
    - label: Correct ending index (0-3)
    - activity_label: Category label
    - split_type: Train/val/test split indicator
    """
    
    def _get_field_mapping(self) -> Dict[str, str]:
        """Map standard fields to HellaSwag fields."""
        return {
            "prompt": "ctx",
            "context": "ctx",
            "endings": "endings",
            "reference": "label",
            "label": "label",
            "activity": "activity_label",
        }
    
    def _extract_sample(self, item: Dict[str, Any], index: int) -> Sample:
        """Extract a HellaSwag sample."""
        # Get context
        ctx = item.get("ctx", "")
        ctx_a = item.get("ctx_a", "")
        ctx_b = item.get("ctx_b", "")
        
        # Get endings (A, B, C, D)
        endings = item.get("endings", [])
        
        # Get correct label (0-3)
        label = item.get("label", 0)
        
        # Convert label to letter if needed
        if isinstance(label, int):
            label_letter = chr(ord('A') + label)
        elif isinstance(label, str) and label.isdigit():
            # Handle string numeric labels (e.g., "3" -> "D")
            label_letter = chr(ord('A') + int(label))
        else:
            label_letter = str(label).upper()
        
        # Build the prompt with options
        prompt = self._build_prompt(ctx, endings)
        
        # Build metadata
        metadata = {
            "ctx": ctx,
            "ctx_a": ctx_a,
            "ctx_b": ctx_b,
            "endings": endings,
            "activity_label": item.get("activity_label", ""),
            "split_type": item.get("split_type", ""),
            "label_index": label if isinstance(label, int) else ord(label_letter) - ord('A'),
        }
        
        return Sample(
            id=f"hellaswag_{index}",
            prompt=prompt,
            reference=label_letter,
            metadata=metadata,
        )
    
    def _build_prompt(self, ctx: str, endings: List[str]) -> str:
        """
        Build the multiple choice prompt.
        
        Args:
            ctx: Context text
            endings: List of 4 possible endings
        
        Returns:
            Formatted prompt with options
        """
        prompt = f"""{ctx}

Complete the scenario by choosing the most likely ending:

A) {endings[0] if len(endings) > 0 else 'N/A'}
B) {endings[1] if len(endings) > 1 else 'N/A'}
C) {endings[2] if len(endings) > 2 else 'N/A'}
D) {endings[3] if len(endings) > 3 else 'N/A'}

Answer:"""
        
        return prompt
    
    def _get_prompt(self, item: Dict[str, Any]) -> str:
        """Get the formatted prompt."""
        ctx = item.get("ctx", "")
        endings = item.get("endings", [])
        return self._build_prompt(ctx, endings)
    
    def _get_reference(self, item: Dict[str, Any]) -> str:
        """Get the correct answer letter."""
        label = item.get("label", 0)
        if isinstance(label, int):
            return chr(ord('A') + label)
        return str(label).upper()
    
    def get_endings(self, item: Dict[str, Any]) -> List[str]:
        """Get the list of endings."""
        return item.get("endings", [])
    
    def get_label_index(self, item: Dict[str, Any]) -> int:
        """Get the correct label index (0-3)."""
        label = item.get("label", 0)
        if isinstance(label, int):
            return label
        # Convert letter to index
        label_str = str(label).upper()
        if label_str in 'ABCD':
            return ord(label_str) - ord('A')
        return 0
