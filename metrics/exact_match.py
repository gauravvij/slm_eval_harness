"""Exact match metric with normalization."""

import re
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass

from core.base import EvaluationResult
from core.registry import register_metric


@dataclass
class ExactMatchResult:
    """Result of exact match calculation."""
    accuracy: float
    total: int
    correct: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "total": self.total,
            "correct": self.correct,
        }


@register_metric("exact_match")
class ExactMatch:
    """
    Exact match metric with various normalization options.
    
    Supports:
    - Case insensitive matching
    - Whitespace normalization
    - Punctuation removal
    - Custom normalization functions
    """
    
    def __init__(
        self,
        case_insensitive: bool = True,
        strip_whitespace: bool = True,
        remove_punctuation: bool = False,
        normalize_unicode: bool = True,
        custom_normalizer: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize ExactMatch metric.
        
        Args:
            case_insensitive: Convert to lowercase before comparison
            strip_whitespace: Remove leading/trailing whitespace
            remove_punctuation: Remove punctuation characters
            normalize_unicode: Normalize unicode characters
            custom_normalizer: Custom normalization function
        """
        self.case_insensitive = case_insensitive
        self.strip_whitespace = strip_whitespace
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.custom_normalizer = custom_normalizer
    
    def compute(
        self,
        predictions: List[str],
        references: List[str]
    ) -> ExactMatchResult:
        """
        Compute exact match accuracy.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
        
        Returns:
            ExactMatchResult
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) "
                "must have same length"
            )
        
        if not predictions:
            return ExactMatchResult(0.0, 0, 0)
        
        correct = 0
        for pred, ref in zip(predictions, references):
            if self.match(pred, ref):
                correct += 1
        
        accuracy = correct / len(predictions)
        
        return ExactMatchResult(
            accuracy=accuracy,
            total=len(predictions),
            correct=correct
        )
    
    def compute_from_results(
        self,
        results: List[EvaluationResult]
    ) -> ExactMatchResult:
        """
        Compute exact match from evaluation results.
        
        Args:
            results: List of evaluation results
        
        Returns:
            ExactMatchResult
        """
        if not results:
            return ExactMatchResult(0.0, 0, 0)
        
        correct = sum(1 for r in results if r.passed)
        total = len(results)
        
        return ExactMatchResult(
            accuracy=correct / total if total > 0 else 0.0,
            total=total,
            correct=correct
        )
    
    def match(self, prediction: str, reference: str) -> bool:
        """
        Check if prediction matches reference after normalization.
        
        Args:
            prediction: Predicted string
            reference: Reference string
        
        Returns:
            True if match, False otherwise
        """
        norm_pred = self.normalize(prediction)
        norm_ref = self.normalize(reference)
        
        return norm_pred == norm_ref
    
    def normalize(self, text: str) -> str:
        """
        Normalize text according to configured options.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        if text is None:
            return ""
        
        result = str(text)
        
        # Unicode normalization
        if self.normalize_unicode:
            import unicodedata
            result = unicodedata.normalize('NFKC', result)
        
        # Strip whitespace
        if self.strip_whitespace:
            result = result.strip()
        
        # Remove punctuation
        if self.remove_punctuation:
            result = re.sub(r'[^\w\s]', '', result)
        
        # Case normalization
        if self.case_insensitive:
            result = result.lower()
        
        # Collapse whitespace
        result = re.sub(r'\s+', ' ', result)
        
        # Custom normalizer
        if self.custom_normalizer:
            result = self.custom_normalizer(result)
        
        return result.strip()
    
    def get_matches(
        self,
        predictions: List[str],
        references: List[str]
    ) -> List[Tuple[str, str, bool]]:
        """
        Get detailed match information.
        
        Args:
            predictions: List of predictions
            references: List of references
        
        Returns:
            List of (prediction, reference, matched) tuples
        """
        matches = []
        for pred, ref in zip(predictions, references):
            matched = self.match(pred, ref)
            matches.append((pred, ref, matched))
        return matches


class NormalizedExactMatch(ExactMatch):
    """Exact match with all normalizations enabled."""
    
    def __init__(self):
        super().__init__(
            case_insensitive=True,
            strip_whitespace=True,
            remove_punctuation=True,
            normalize_unicode=True,
        )


class StrictExactMatch(ExactMatch):
    """Exact match with no normalizations."""
    
    def __init__(self):
        super().__init__(
            case_insensitive=False,
            strip_whitespace=False,
            remove_punctuation=False,
            normalize_unicode=False,
        )


def compute_exact_match(
    predictions: List[str],
    references: List[str],
    **kwargs
) -> ExactMatchResult:
    """Convenience function to compute exact match."""
    metric = ExactMatch(**kwargs)
    return metric.compute(predictions, references)
