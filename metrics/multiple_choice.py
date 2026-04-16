"""Multiple choice accuracy metric."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from core.base import EvaluationResult
from core.registry import register_metric


@dataclass
class MultipleChoiceResult:
    """Result of multiple choice evaluation."""
    accuracy: float
    total: int
    correct: int
    per_choice_accuracy: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "total": self.total,
            "correct": self.correct,
            "per_choice_accuracy": self.per_choice_accuracy,
            "confusion_matrix": self.confusion_matrix,
        }


@register_metric("multiple_choice")
class MultipleChoiceAccuracy:
    """
    Metric for multiple choice evaluation.
    
    Computes:
    - Overall accuracy
    - Per-choice accuracy
    - Confusion matrix
    """
    
    def __init__(self, choices: Optional[List[str]] = None):
        """
        Initialize MultipleChoiceAccuracy metric.
        
        Args:
            choices: List of valid choices (e.g., ['A', 'B', 'C', 'D'])
        """
        self.choices = choices or ['A', 'B', 'C', 'D']
    
    def compute(
        self,
        predictions: List[str],
        references: List[str]
    ) -> MultipleChoiceResult:
        """
        Compute multiple choice accuracy.
        
        Args:
            predictions: List of predicted choices
            references: List of reference choices
        
        Returns:
            MultipleChoiceResult
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) "
                "must have same length"
            )
        
        if not predictions:
            return MultipleChoiceResult(
                accuracy=0.0,
                total=0,
                correct=0,
                per_choice_accuracy={},
                confusion_matrix={}
            )
        
        # Normalize choices
        norm_preds = [self._normalize_choice(p) for p in predictions]
        norm_refs = [self._normalize_choice(r) for r in references]
        
        # Overall accuracy
        correct = sum(1 for p, r in zip(norm_preds, norm_refs) if p == r)
        accuracy = correct / len(predictions)
        
        # Per-choice accuracy
        per_choice = self._compute_per_choice_accuracy(norm_preds, norm_refs)
        
        # Confusion matrix
        confusion = self._compute_confusion_matrix(norm_preds, norm_refs)
        
        return MultipleChoiceResult(
            accuracy=accuracy,
            total=len(predictions),
            correct=correct,
            per_choice_accuracy=per_choice,
            confusion_matrix=confusion
        )
    
    def compute_from_results(
        self,
        results: List[EvaluationResult]
    ) -> MultipleChoiceResult:
        """
        Compute accuracy from evaluation results.
        
        Args:
            results: List of evaluation results
        
        Returns:
            MultipleChoiceResult
        """
        if not results:
            return MultipleChoiceResult(
                accuracy=0.0,
                total=0,
                correct=0,
                per_choice_accuracy={},
                confusion_matrix={}
            )
        
        # Extract predictions and references from results
        # Note: This assumes results contain the necessary info in metadata
        correct = sum(1 for r in results if r.passed)
        total = len(results)
        
        return MultipleChoiceResult(
            accuracy=correct / total if total > 0 else 0.0,
            total=total,
            correct=correct,
            per_choice_accuracy={},
            confusion_matrix={}
        )
    
    def _normalize_choice(self, choice: str) -> str:
        """Normalize a choice to uppercase letter."""
        if choice is None:
            return ""
        
        choice_str = str(choice).strip().upper()
        
        # Extract first letter if there's extra text
        if len(choice_str) > 1:
            # Try to find a valid choice letter
            for char in choice_str:
                if char in self.choices:
                    return char
        
        return choice_str if choice_str in self.choices else choice_str[:1]
    
    def _compute_per_choice_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute accuracy for each choice."""
        choice_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for pred, ref in zip(predictions, references):
            choice_stats[ref]["total"] += 1
            if pred == ref:
                choice_stats[ref]["correct"] += 1
        
        per_choice = {}
        for choice, stats in choice_stats.items():
            per_choice[choice] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return per_choice
    
    def _compute_confusion_matrix(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Compute confusion matrix."""
        matrix = defaultdict(lambda: defaultdict(int))
        
        for pred, ref in zip(predictions, references):
            matrix[ref][pred] += 1
        
        # Convert to regular dict
        result = {}
        for ref in matrix:
            result[ref] = dict(matrix[ref])
        
        return result
    
    def get_choice_distribution(self, predictions: List[str]) -> Dict[str, int]:
        """
        Get distribution of predicted choices.
        
        Args:
            predictions: List of predictions
        
        Returns:
            Dictionary mapping choice to count
        """
        distribution = defaultdict(int)
        
        for pred in predictions:
            norm_pred = self._normalize_choice(pred)
            distribution[norm_pred] += 1
        
        return dict(distribution)
    
    def analyze_errors(
        self,
        predictions: List[str],
        references: List[str],
        questions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze errors in detail.
        
        Args:
            predictions: List of predictions
            references: List of references
            questions: Optional list of questions
        
        Returns:
            List of error details
        """
        errors = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if pred != ref:
                error_info = {
                    "index": i,
                    "predicted": pred,
                    "reference": ref,
                    "question": questions[i] if questions else None,
                }
                errors.append(error_info)
        
        return errors


def compute_multiple_choice_accuracy(
    predictions: List[str],
    references: List[str],
    choices: Optional[List[str]] = None
) -> MultipleChoiceResult:
    """Convenience function to compute multiple choice accuracy."""
    metric = MultipleChoiceAccuracy(choices=choices)
    return metric.compute(predictions, references)
