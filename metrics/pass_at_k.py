"""Pass@k metric with correct combinatorial formula."""

import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.base import EvaluationResult
from core.registry import register_metric


@dataclass
class PassAtKResult:
    """Result of pass@k calculation."""
    pass_at_k: float
    total: int
    correct: int
    k: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_at_k": self.pass_at_k,
            "total": self.total,
            "correct": self.correct,
            "k": self.k,
        }


@register_metric("pass_at_k")
class PassAtK:
    """
    Pass@k metric for code generation evaluation.
    
    Uses the correct combinatorial formula:
    pass@k = 1 - C(n-c, k) / C(n, k)
    
    Where:
    - n = total number of samples
    - c = number of correct samples
    - k = number of attempts
    """
    
    def __init__(self, k: int = 1):
        """
        Initialize Pass@k metric.
        
        Args:
            k: Number of attempts (samples per problem)
        """
        self.k = k
    
    def compute(
        self,
        results: List[EvaluationResult],
        group_by: Optional[str] = None
    ) -> PassAtKResult:
        """
        Compute pass@k metric.
        
        Args:
            results: List of evaluation results
            group_by: Optional field to group by (e.g., 'task_id')
        
        Returns:
            PassAtKResult
        """
        if not results:
            return PassAtKResult(0.0, 0, 0, self.k)
        
        if group_by:
            # Group results and compute pass@k per group
            groups = self._group_results(results, group_by)
            total_pass = 0
            total_groups = len(groups)
            
            for group_results in groups.values():
                group_pass = self._compute_pass_at_k_for_group(group_results)
                total_pass += group_pass
            
            pass_rate = total_pass / total_groups if total_groups > 0 else 0.0
            
            return PassAtKResult(
                pass_at_k=pass_rate,
                total=total_groups,
                correct=int(total_pass),
                k=self.k
            )
        else:
            # Simple pass@k over all results
            n = len(results)
            c = sum(1 for r in results if r.passed)
            
            pass_at_k = self._calculate_pass_at_k(n, c, self.k)
            
            return PassAtKResult(
                pass_at_k=pass_at_k,
                total=n,
                correct=c,
                k=self.k
            )
    
    def _calculate_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        Calculate pass@k using the combinatorial formula.
        
        pass@k = 1 - C(n-c, k) / C(n, k)
        
        Args:
            n: Total number of samples
            c: Number of correct samples
            k: Number of attempts
        
        Returns:
            pass@k value
        """
        if n == 0:
            return 0.0
        
        if c == 0:
            return 0.0
        
        if c == n:
            return 1.0
        
        # Use the unbiased estimator formula
        # pass@k ≈ 1 - (1 - c/n)^k for large n
        # But for small n, use exact calculation
        
        if n <= 100:
            # Exact calculation using combinatorics
            try:
                # C(n-c, k) / C(n, k)
                # = (n-c)! / (k! * (n-c-k)!) / (n! / (k! * (n-k)!))
                # = (n-c)! * (n-k)! / ((n-c-k)! * n!)
                
                if k > n - c:
                    # C(n-c, k) = 0 when k > n-c
                    return 1.0
                
                # Use log factorials for numerical stability
                log_c_n_c_k = self._log_combination(n - c, k)
                log_c_n_k = self._log_combination(n, k)
                
                ratio = math.exp(log_c_n_c_k - log_c_n_k)
                return 1.0 - ratio
                
            except (ValueError, OverflowError):
                # Fall back to approximation
                pass
        
        # Approximation for large n
        # pass@k ≈ 1 - (1 - c/n)^k
        p = c / n
        return 1.0 - (1.0 - p) ** k
    
    def _log_combination(self, n: int, k: int) -> float:
        """
        Calculate log(C(n, k)) using log gamma function.
        
        C(n, k) = n! / (k! * (n-k)!)
        log(C(n, k)) = log(n!) - log(k!) - log((n-k)!)
        """
        if k < 0 or k > n:
            raise ValueError(f"Invalid k={k} for n={n}")
        
        if k == 0 or k == n:
            return 0.0
        
        # Use log gamma: log(n!) = log_gamma(n+1)
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    
    def _group_results(
        self,
        results: List[EvaluationResult],
        group_by: str
    ) -> Dict[str, List[EvaluationResult]]:
        """Group results by a field."""
        groups = {}
        
        for result in results:
            # Extract group key from metadata
            key = result.sample_id
            if hasattr(result, 'diagnostics') and result.diagnostics:
                key = result.diagnostics.get(group_by, key)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        return groups
    
    def _compute_pass_at_k_for_group(self, group_results: List[EvaluationResult]) -> float:
        """Compute pass@k for a group of results."""
        n = len(group_results)
        c = sum(1 for r in group_results if r.passed)
        return 1.0 if c > 0 else 0.0
    
    def compute_multiple_k(
        self,
        results: List[EvaluationResult],
        k_values: List[int] = None,
        group_by: Optional[str] = None
    ) -> Dict[int, PassAtKResult]:
        """
        Compute pass@k for multiple k values.
        
        Args:
            results: List of evaluation results
            k_values: List of k values to compute
            group_by: Optional field to group by
        
        Returns:
            Dictionary mapping k to PassAtKResult
        """
        if k_values is None:
            k_values = [1, 10, 100]
        
        results_dict = {}
        for k in k_values:
            self.k = k
            results_dict[k] = self.compute(results, group_by)
        
        return results_dict
    
    @staticmethod
    def estimate_pass_at_k(n: int, c: int, k: int) -> float:
        """
        Static method to estimate pass@k without instantiation.
        
        Args:
            n: Total number of samples
            c: Number of correct samples
            k: Number of attempts
        
        Returns:
            Estimated pass@k
        """
        if n == 0:
            return 0.0
        if c == 0:
            return 0.0
        if c == n:
            return 1.0
        
        # Simple approximation
        p = c / n
        return 1.0 - (1.0 - p) ** k


def compute_pass_at_k(
    results: List[EvaluationResult],
    k: int = 1,
    group_by: Optional[str] = None
) -> PassAtKResult:
    """Convenience function to compute pass@k."""
    metric = PassAtK(k=k)
    return metric.compute(results, group_by)
