"""Base dataset loader with common functionality."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional
from datasets import load_dataset

from core.base import DatasetLoader, Sample, TaskConfig


class BaseDatasetLoader(DatasetLoader):
    """
    Base class for dataset loaders.
    
    Provides common functionality for:
    - Loading datasets from HuggingFace
    - Field mapping
    - Sample limiting
    - Schema extraction
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self._dataset = None
        self._schema = None
    
    @abstractmethod
    def _extract_sample(self, item: Dict[str, Any], index: int) -> Sample:
        """
        Extract a Sample from a dataset item.
        
        Override in subclasses to handle specific dataset formats.
        """
        pass
    
    @abstractmethod
    def _get_field_mapping(self) -> Dict[str, str]:
        """
        Get field mapping from dataset fields to standard fields.
        
        Returns:
            Dict mapping standard field names to dataset field names
        """
        pass
    
    def load(self) -> List[Sample]:
        """
        Load the dataset and return samples.
        
        Returns:
            List of Sample objects
        """
        # Load dataset
        dataset = self._load_dataset()
        
        # Extract samples
        samples = []
        for idx, item in enumerate(dataset):
            try:
                sample = self._extract_sample(item, idx)
                samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to extract sample {idx}: {e}")
                continue
        
        # Apply sample limit if specified
        if self.config.num_samples is not None:
            samples = samples[:self.config.num_samples]
        
        return samples
    
    def _load_dataset(self):
        """Load dataset from HuggingFace or local path."""
        path = self.config.dataset_path
        split = self.config.split
        subset = self.config.subset
        
        try:
            if subset:
                dataset = load_dataset(path, subset, split=split)
            else:
                dataset = load_dataset(path, split=split)
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {path}: {e}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema of the dataset."""
        if self._schema is None:
            dataset = self._load_dataset()
            if hasattr(dataset, 'features'):
                self._schema = dict(dataset.features)
            else:
                # Try to infer from first item
                if len(dataset) > 0:
                    first_item = dataset[0]
                    self._schema = {k: type(v).__name__ for k, v in first_item.items()}
                else:
                    self._schema = {}
        return self._schema
    
    def _map_field(self, item: Dict[str, Any], standard_field: str) -> Any:
        """
        Map a dataset field to a standard field.
        
        Args:
            item: Dataset item
            standard_field: Standard field name
        
        Returns:
            The mapped value or None if not found
        """
        mapping = self._get_field_mapping()
        dataset_field = mapping.get(standard_field)
        
        if dataset_field and dataset_field in item:
            return item[dataset_field]
        
        # Try standard field name directly
        if standard_field in item:
            return item[standard_field]
        
        return None
    
    def _get_prompt(self, item: Dict[str, Any]) -> str:
        """Extract prompt from item. Override if needed."""
        return self._map_field(item, "prompt") or ""
    
    def _get_reference(self, item: Dict[str, Any]) -> Any:
        """Extract reference from item. Override if needed."""
        return self._map_field(item, "reference")
    
    def _get_test_code(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract test code from item (for coding tasks)."""
        return self._map_field(item, "test")
    
    def _get_entry_point(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract entry point from item (for coding tasks)."""
        return self._map_field(item, "entry_point")
    
    def _get_canonical_solution(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract canonical solution from item."""
        return self._map_field(item, "canonical_solution")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        dataset = self._load_dataset()
        return {
            "path": self.config.dataset_path,
            "split": self.config.split,
            "subset": self.config.subset,
            "num_samples": len(dataset),
            "schema": self.get_schema(),
        }
