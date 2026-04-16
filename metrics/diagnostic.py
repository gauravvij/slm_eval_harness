"""Diagnostic analyzer for error pattern analysis."""

import re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from core.base import EvaluationResult


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report."""
    total_samples: int
    passed: int
    failed: int
    pass_rate: float
    
    # Error analysis
    error_patterns: Dict[str, int]
    error_categories: Dict[str, int]
    
    # Performance by category
    category_performance: Dict[str, Dict[str, float]]
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "error_patterns": self.error_patterns,
            "error_categories": self.error_categories,
            "category_performance": self.category_performance,
            "recommendations": self.recommendations,
        }


class DiagnosticAnalyzer:
    """
    Analyzer for diagnostic information and error patterns.
    
    Provides:
    - Error pattern detection
    - Category-based performance analysis
    - Actionable recommendations
    """
    
    # Error patterns to detect
    ERROR_PATTERNS = {
        "syntax_error": [
            r"syntax\s*error",
            r"unexpected\s+token",
            r"invalid\s+syntax",
            r"indentation\s*error",
        ],
        "timeout": [
            r"timeout",
            r"timed\s*out",
            r"exceeded",
        ],
        "runtime_error": [
            r"runtime\s*error",
            r"exception",
            r"traceback",
            r"error:\s*",
        ],
        "refusal": [
            r"i\s+cannot",
            r"i\s+can't",
            r"i\s+am\s+unable",
            r"i'm\s+unable",
        ],
        "empty_output": [
            r"^\s*$",
            r"no\s+output",
            r"empty",
        ],
        "wrong_format": [
            r"format",
            r"expected",
            r"invalid\s+format",
        ],
    }
    
    def __init__(self):
        self.error_patterns = defaultdict(int)
        self.error_categories = defaultdict(int)
    
    def analyze(
        self,
        results: List[EvaluationResult],
        categories: Optional[Dict[str, List[str]]] = None
    ) -> DiagnosticReport:
        """
        Analyze evaluation results and generate diagnostic report.
        
        Args:
            results: List of evaluation results
            categories: Optional mapping of category names to sample IDs
        
        Returns:
            DiagnosticReport
        """
        if not results:
            return DiagnosticReport(
                total_samples=0,
                passed=0,
                failed=0,
                pass_rate=0.0,
                error_patterns={},
                error_categories={},
                category_performance={},
                recommendations=[],
            )
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = passed / total if total > 0 else 0.0
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns(results)
        
        # Analyze error categories
        error_categories = self._categorize_errors(results)
        
        # Category performance
        category_performance = {}
        if categories:
            category_performance = self._analyze_category_performance(
                results, categories
            )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            pass_rate, error_patterns, error_categories
        )
        
        return DiagnosticReport(
            total_samples=total,
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            error_patterns=error_patterns,
            error_categories=error_categories,
            category_performance=category_performance,
            recommendations=recommendations,
        )
    
    def _analyze_error_patterns(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, int]:
        """Analyze error patterns in failed results."""
        patterns = defaultdict(int)
        
        for result in results:
            if not result.passed:
                error_text = self._get_error_text(result)
                
                for pattern_name, pattern_list in self.ERROR_PATTERNS.items():
                    for pattern in pattern_list:
                        if re.search(pattern, error_text, re.IGNORECASE):
                            patterns[pattern_name] += 1
                            break
        
        return dict(patterns)
    
    def _categorize_errors(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, int]:
        """Categorize errors by type."""
        categories = defaultdict(int)
        
        for result in results:
            if not result.passed:
                if result.error:
                    categories["execution_error"] += 1
                elif result.parsed_result and result.parsed_result.refusal_detected:
                    categories["refusal"] += 1
                elif result.parsed_result and result.parsed_result.error:
                    categories["parsing_error"] += 1
                else:
                    categories["incorrect_output"] += 1
        
        return dict(categories)
    
    def _analyze_category_performance(
        self,
        results: List[EvaluationResult],
        categories: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance by category."""
        performance = {}
        
        # Build sample ID to result mapping
        result_map = {r.sample_id: r for r in results}
        
        for category_name, sample_ids in categories.items():
            category_results = [
                result_map[sid] for sid in sample_ids
                if sid in result_map
            ]
            
            if category_results:
                total = len(category_results)
                passed = sum(1 for r in category_results if r.passed)
                performance[category_name] = {
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "accuracy": passed / total if total > 0 else 0.0,
                }
        
        return performance
    
    def _generate_recommendations(
        self,
        pass_rate: float,
        error_patterns: Dict[str, int],
        error_categories: Dict[str, int]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Overall performance
        if pass_rate < 0.3:
            recommendations.append(
                "CRITICAL: Very low pass rate. Model may not be suitable for this task. "
                "Consider using a larger model or fine-tuning."
            )
        elif pass_rate < 0.6:
            recommendations.append(
                "WARNING: Below average performance. Review prompt engineering and "
                "consider few-shot examples."
            )
        elif pass_rate < 0.8:
            recommendations.append(
                "MODERATE: Room for improvement. Analyze error patterns for specific issues."
            )
        else:
            recommendations.append(
                "GOOD: Solid performance. Focus on edge cases for further improvement."
            )
        
        # Error-specific recommendations
        if error_patterns.get("syntax_error", 0) > 0:
            recommendations.append(
                f"SYNTAX ERRORS: {error_patterns['syntax_error']} samples had syntax errors. "
                "Consider improving code parser or adding syntax validation."
            )
        
        if error_patterns.get("timeout", 0) > 0:
            recommendations.append(
                f"TIMEOUTS: {error_patterns['timeout']} samples timed out. "
                "Consider increasing timeout or optimizing generated code."
            )
        
        if error_patterns.get("refusal", 0) > 0:
            recommendations.append(
                f"REFUSALS: {error_patterns['refusal']} samples were refused. "
                "Review safety settings or use a different model."
            )
        
        if error_categories.get("parsing_error", 0) > 0:
            recommendations.append(
                f"PARSING ERRORS: {error_categories['parsing_error']} samples failed to parse. "
                "Consider using a more robust parser or adjusting extraction mode."
            )
        
        return recommendations
    
    def _get_error_text(self, result: EvaluationResult) -> str:
        """Extract error text from result."""
        texts = []
        
        if result.error:
            texts.append(result.error)
        
        if result.parsed_result:
            if result.parsed_result.error:
                texts.append(result.parsed_result.error)
            texts.append(result.parsed_result.raw_output)
        
        return " ".join(texts).lower()
    
    def get_failed_samples(
        self,
        results: List[EvaluationResult]
    ) -> List[EvaluationResult]:
        """Get list of failed samples."""
        return [r for r in results if not r.passed]
    
    def get_error_summary(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Get quick error summary."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "error_types": self._categorize_errors(results),
        }
