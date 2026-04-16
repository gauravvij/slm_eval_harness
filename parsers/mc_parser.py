"""Multiple choice parser for A/B/C/D extraction."""

import re
from typing import Any, Dict, List, Optional, Tuple

from .base_parser import BaseParser
from core.base import ParsedResult
from core.registry import register_parser


@register_parser("mc")
class MultipleChoiceParser(BaseParser):
    """
    Parser for multiple choice answers (A/B/C/D).
    
    Features:
    - Extracts single letter answers
    - Handles various formats (A), A., A), etc.
    - Confidence scoring based on format
    """
    
    name: str = "mc"
    
    # Valid choice letters
    CHOICES = ['A', 'B', 'C', 'D', 'E']
    
    # Patterns for answer extraction
    PATTERNS = [
        # Explicit answer patterns
        (r'\banswer\s*[:=]\s*([A-E])', 1.0),
        (r'\bthe answer is\s*[:=]?\s*([A-E])\b', 1.0),
        (r'\bcorrect answer\s*[:=]?\s*([A-E])\b', 1.0),
        (r'\boption\s*[:=]?\s*([A-E])\b', 1.0),
        
        # Choice patterns with various formats
        (r'\*\*([A-E])\*\*', 0.95),  # Bold: **A**
        (r'`([A-E])`', 0.95),         # Inline code: `A`
        (r'\[([A-E])\]', 0.9),        # Brackets: [A]
        (r'\(([A-E])\)', 0.9),        # Parentheses: (A)
        (r'([A-E])\)', 0.85),          # A), B), etc.
        (r'([A-E])\.', 0.85),          # A., B., etc.
        
        # Standalone letters
        (r'(?:^|\s)([A-E])(?:\s|$|\.|,)', 0.7),  # Standalone letter
        
        # Last occurrence (fallback) - only if text looks like MC format
        (r'(?:answer|option|choice)\s*:?\s*([A-E])', 0.5),
    ]
    
    def __init__(
        self,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.choices = choices or self.CHOICES
        self.case_sensitive = case_sensitive
    
    def _parse_impl(self, text: str, **kwargs) -> str:
        """
        Extract multiple choice answer.
        
        Args:
            text: Model output
            **kwargs: May contain 'choices' override
        
        Returns:
            Single letter answer (A, B, C, D, or E)
        """
        choices = kwargs.get('choices', self.choices)
        
        # Normalize text
        if not self.case_sensitive:
            text_upper = text.upper()
        else:
            text_upper = text
        
        # Try patterns in order of confidence
        for pattern, confidence in self.PATTERNS:
            matches = list(re.finditer(pattern, text_upper, re.IGNORECASE))
            if matches:
                # Get the last match (usually the final answer)
                match = matches[-1]
                answer = match.group(1).upper()
                
                # Validate against allowed choices
                if answer in [c.upper() for c in choices]:
                    return answer
        
        # No match found
        # Try to find any letter from choices
        for choice in reversed(choices):
            if choice.upper() in text_upper:
                # Check if it's a standalone letter
                if re.search(rf'\b{re.escape(choice.upper())}\b', text_upper):
                    return choice.upper()
        
        # Final fallback: return first letter found
        for char in reversed(text_upper):
            if char in 'ABCDE':
                return char
        
        return ""
    
    def _calculate_confidence(self, text: str, parsed: Any) -> float:
        """Calculate confidence based on pattern match."""
        if not parsed or not isinstance(parsed, str):
            return 0.0
        
        if not self.case_sensitive:
            text_upper = text.upper()
        else:
            text_upper = text
        
        # Check patterns in order
        for pattern, confidence in self.PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return confidence
        
        return 0.3  # Low confidence for fallback extraction
    
    def parse_with_alternatives(
        self,
        text: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Parse with top-k alternatives.
        
        Args:
            text: Model output
            top_k: Number of alternatives to return
        
        Returns:
            List of (answer, confidence) tuples
        """
        results = []
        
        if not self.case_sensitive:
            text_upper = text.upper()
        else:
            text_upper = text
        
        # Collect all matches with confidence
        for pattern, confidence in self.PATTERNS:
            for match in re.finditer(pattern, text_upper, re.IGNORECASE):
                answer = match.group(1).upper()
                if answer in 'ABCDE':
                    results.append((answer, confidence))
        
        # Sort by confidence and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for answer, conf in results:
            if answer not in seen:
                seen.add(answer)
                unique_results.append((answer, conf))
        
        return unique_results[:top_k]
    
    def extract_reasoning(self, text: str) -> str:
        """
        Extract reasoning/explanation from text.
        
        Args:
            text: Model output
        
        Returns:
            Reasoning text (without the answer)
        """
        # Remove the answer from text
        answer = self._parse_impl(text)
        
        if answer:
            # Remove answer patterns
            patterns = [
                rf'\banswer\s*[:=]\s*{answer}',
                rf'\bthe answer is\s*[:=]?\s*{answer}',
                rf'\*\*{answer}\*\*',
                rf'`{answer}`',
                rf'\[{answer}\]',
                rf'\({answer}\)',
            ]
            
            reasoning = text
            for pattern in patterns:
                reasoning = re.sub(pattern, '', reasoning, flags=re.IGNORECASE)
            
            return reasoning.strip()
        
        return text.strip()
    
    def validate_answer(self, answer: str) -> bool:
        """Validate that answer is a valid choice."""
        if not answer:
            return False
        
        answer_upper = answer.upper().strip()
        return answer_upper in [c.upper() for c in self.choices]
