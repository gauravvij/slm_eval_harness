"""Detector for common refusal patterns in model outputs."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class RefusalCheckResult:
    """Result of refusal detection."""
    is_refusal: bool
    pattern_matched: Optional[str]
    confidence: float
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_refusal": self.is_refusal,
            "pattern_matched": self.pattern_matched,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class RefusalDetector:
    """
    Detector for common refusal patterns in model outputs.
    
    Detects:
    - Safety refusals ("I cannot", "I'm not able to")
    - Policy violations ("against my guidelines")
    - Capability limitations ("I don't know how")
    - Empty/placeholder responses
    """
    
    # Refusal patterns with confidence scores
    REFUSAL_PATTERNS: List[Tuple[str, float]] = [
        # High confidence refusals
        (r"\b(i cannot|i can't|i am unable to|i'm unable to)\b", 0.95),
        (r"\b(i'm not able to|i am not able to)\b", 0.95),
        (r"\b(i will not|i won't)\b", 0.95),
        (r"\b(i'm not comfortable|i am not comfortable)\b", 0.95),
        (r"\b(i refuse to|i'm refusing to)\b", 0.95),
        (r"\b(violates|against|violating)\s+(my|the)\s+(policy|guidelines|rules|terms)\b", 0.95),
        (r"\b(not appropriate|inappropriate)\b", 0.9),
        (r"\b(not ethical|unethical)\b", 0.9),
        (r"\b(not allowed|disallowed)\b", 0.9),
        (r"\b(cannot assist|can't assist|unable to assist)\b", 0.95),
        (r"\b(cannot help|can't help|unable to help)\b", 0.95),
        (r"^sorry[,\s]", 0.85),  # Sorry at start of sentence
        
        # Medium confidence - capability limitations
        (r"\b(i don't know|i do not know)\b", 0.7),
        (r"\b(i'm not sure|i am not sure)\b", 0.7),
        (r"\b(i don't understand|i do not understand)\b", 0.7),
        (r"\b(i'm not familiar|i am not familiar)\b", 0.7),
        (r"\b(beyond my|outside my)\s+(capabilities|abilities|knowledge)\b", 0.75),
        (r"\b(i don't have|i do not have)\s+(the ability|access to)\b", 0.75),
        
        # Lower confidence - hedging
        (r"\b(i would need|i'd need)\s+more\s+information\b", 0.5),
        (r"\b(i can't provide|i cannot provide)\s+(specific|detailed)\b", 0.6),
        (r"\b(i'm not able to|i am not able to)\s+provide\b", 0.7),
        
        # Code-specific refusals
        (r"\b(i cannot generate|i can't generate)\s+code\b", 0.95),
        (r"\b(i will not generate|i won't generate)\s+code\b", 0.95),
        (r"\b(code generation|generating code)\s+is\s+not\s+(allowed|permitted)\b", 0.95),
    ]
    
    # Empty/placeholder patterns
    EMPTY_PATTERNS: List[Tuple[str, float]] = [
        (r"^\s*$", 0.9),  # Completely empty
        (r"^\s*\.\.\.\s*$", 0.9),  # Just ellipsis
        (r"^\s*\[\s*\]\s*$", 0.8),  # Empty brackets
        (r"^\s*\{\s*\}\s*$", 0.8),  # Empty braces
        (r"^\s*null\s*$", 0.8),  # Just null
        (r"^\s*none\s*$", 0.8),  # Just none
        (r"^\s*n/a\s*$", 0.8),  # Just N/A
    ]
    
    def __init__(
        self,
        threshold: float = 0.7,
        custom_patterns: Optional[List[Tuple[str, float]]] = None,
    ):
        self.threshold = threshold
        self.patterns = self.REFUSAL_PATTERNS.copy()
        
        if custom_patterns:
            self.patterns.extend(custom_patterns)
    
    def detect(self, text: str) -> RefusalCheckResult:
        """
        Detect if text contains a refusal.
        
        Args:
            text: Model output text
        
        Returns:
            RefusalCheckResult
        """
        if not text or not isinstance(text, str):
            return RefusalCheckResult(
                is_refusal=True,
                pattern_matched="empty_input",
                confidence=1.0,
                reason="Empty or invalid input"
            )
        
        text_lower = text.lower().strip()
        
        # Check for empty/placeholder patterns
        for pattern, confidence in self.EMPTY_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return RefusalCheckResult(
                    is_refusal=True,
                    pattern_matched=pattern,
                    confidence=confidence,
                    reason="Empty or placeholder response"
                )
        
        # Check refusal patterns
        for pattern, confidence in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                if confidence >= self.threshold:
                    return RefusalCheckResult(
                        is_refusal=True,
                        pattern_matched=pattern,
                        confidence=confidence,
                        reason=f"Matched refusal pattern: {pattern[:50]}..."
                    )
        
        # Check for very short responses that might be refusals
        if len(text_lower) < 20:
            # Check if it contains refusal keywords
            refusal_keywords = ['no', 'sorry', 'apologize', 'cannot', "can't", 'unable']
            if any(kw in text_lower for kw in refusal_keywords):
                return RefusalCheckResult(
                    is_refusal=True,
                    pattern_matched="short_refusal",
                    confidence=0.6,
                    reason="Short response with refusal keywords"
                )
        
        return RefusalCheckResult(
            is_refusal=False,
            pattern_matched=None,
            confidence=0.0,
            reason="No refusal patterns detected"
        )
    
    def is_refusal(self, text: str) -> bool:
        """Quick check if text is a refusal."""
        result = self.detect(text)
        return result.is_refusal
    
    def get_refusal_reason(self, text: str) -> Optional[str]:
        """Get the reason for refusal if detected."""
        result = self.detect(text)
        if result.is_refusal:
            return result.reason
        return None
    
    def add_pattern(self, pattern: str, confidence: float):
        """Add a custom refusal pattern."""
        self.patterns.append((pattern, confidence))
    
    def remove_pattern(self, pattern: str):
        """Remove a refusal pattern."""
        self.patterns = [(p, c) for p, c in self.patterns if p != pattern]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Detailed analysis of potential refusal.
        
        Returns:
            Dictionary with detailed analysis
        """
        result = self.detect(text)
        
        # Find all matching patterns (even below threshold)
        all_matches = []
        text_lower = text.lower()
        
        for pattern, confidence in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                all_matches.append({
                    "pattern": pattern,
                    "confidence": confidence,
                })
        
        return {
            "is_refusal": result.is_refusal,
            "primary_match": result.pattern_matched,
            "primary_confidence": result.confidence,
            "reason": result.reason,
            "all_matches": all_matches,
            "text_length": len(text),
            "text_preview": text[:200] if text else "",
        }
    
    def filter_refusals(
        self,
        texts: List[str],
        return_indices: bool = False
    ) -> Union[List[str], List[int]]:
        """
        Filter out refusal texts from a list.
        
        Args:
            texts: List of texts to filter
            return_indices: If True, return indices instead of texts
        
        Returns:
            List of non-refusal texts or their indices
        """
        non_refusal_indices = []
        
        for i, text in enumerate(texts):
            if not self.is_refusal(text):
                non_refusal_indices.append(i)
        
        if return_indices:
            return non_refusal_indices
        
        return [texts[i] for i in non_refusal_indices]


def is_refusal(text: str, threshold: float = 0.7) -> bool:
    """Quick function to check if text is a refusal."""
    detector = RefusalDetector(threshold=threshold)
    return detector.is_refusal(text)


def detect_refusal(text: str) -> RefusalCheckResult:
    """Get detailed refusal detection result."""
    detector = RefusalDetector()
    return detector.detect(text)
