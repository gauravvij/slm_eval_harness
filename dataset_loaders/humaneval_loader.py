"""HumanEval dataset loader with field mapping and test extraction."""

from typing import Any, Dict, List, Optional

from .base_loader import BaseDatasetLoader
from core.base import Sample, TaskConfig
from core.registry import register_dataset_loader


@register_dataset_loader("humaneval")
class HumanEvalLoader(BaseDatasetLoader):
    """
    Loader for HumanEval coding benchmark.
    
    Dataset fields:
    - task_id: Unique identifier
    - prompt: Function signature and docstring
    - entry_point: Function name
    - canonical_solution: Reference solution
    - test: Test cases
    """
    
    def _get_field_mapping(self) -> Dict[str, str]:
        """Map standard fields to HumanEval fields."""
        return {
            "prompt": "prompt",
            "reference": "canonical_solution",
            "test": "test",
            "entry_point": "entry_point",
            "task_id": "task_id",
        }
    
    def _extract_sample(self, item: Dict[str, Any], index: int) -> Sample:
        """Extract a HumanEval sample."""
        task_id = item.get("task_id", f"humaneval_{index}")
        
        # Get the prompt (function signature + docstring)
        prompt = item.get("prompt", "")
        
        # Get canonical solution as reference
        reference = item.get("canonical_solution", "")
        
        # Get test code
        test_code = item.get("test", "")
        
        # Get entry point (function name)
        entry_point = item.get("entry_point", "")
        
        # Build metadata
        metadata = {
            "entry_point": entry_point,
            "test_code": test_code,
            "task_id": task_id,
            "language": "python",
        }
        
        return Sample(
            id=task_id,
            prompt=prompt,
            reference=reference,
            metadata=metadata,
        )
    
    def _get_prompt(self, item: Dict[str, Any]) -> str:
        """Get the prompt for code generation."""
        return item.get("prompt", "")
    
    def _get_reference(self, item: Dict[str, Any]) -> str:
        """Get the canonical solution."""
        return item.get("canonical_solution", "")
    
    def get_test_code(self, item: Dict[str, Any]) -> str:
        """Get the test code for execution."""
        return item.get("test", "")
    
    def get_entry_point(self, item: Dict[str, Any]) -> str:
        """Get the function entry point."""
        return item.get("entry_point", "")
    
    def build_test_script(self, completion: str, test_code: str, entry_point: str) -> str:
        """
        Build a complete test script from completion and test code.
        
        Args:
            completion: Generated code completion
            test_code: Test cases
            entry_point: Function name
        
        Returns:
            Complete Python script ready for execution
        """
        # Combine completion with test code
        # HumanEval format: completion is the function body, test uses entry_point
        script = f"""{completion}

{test_code}
"""
        return script
