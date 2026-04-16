"""Base parser with confidence scoring interface."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import re

from core.base import Parser, ParsedResult
from core.registry import register_parser


class BaseParser(Parser):
    """
    Base class for all output parsers.
    
    Provides:
    - Confidence scoring
    - Error handling
    - Pre/post processing hooks
    """
    
    name: str = "base"
    
    def __init__(
        self,
        strip_whitespace: bool = True,
        lowercase: bool = False,
        max_length: Optional[int] = None,
    ):
        self.strip_whitespace = strip_whitespace
        self.lowercase = lowercase
        self.max_length = max_length
    
    def parse(self, text: str, **kwargs) -> ParsedResult:
        """
        Parse model output.
        
        Args:
            text: Raw model output
            **kwargs: Additional parsing parameters
        
        Returns:
            ParsedResult with content and confidence
        """
        # Pre-process
        processed_text = self._preprocess(text)
        
        try:
            # Parse
            content = self._parse_impl(processed_text, **kwargs)
            
            # Calculate confidence
            confidence = self._calculate_confidence(processed_text, content)
            
            return ParsedResult(
                content=content,
                confidence=confidence,
                parser_used=self.name,
                raw_output=text,
                refusal_detected=False,
                error=None,
            )
            
        except Exception as e:
            return ParsedResult(
                content=None,
                confidence=0.0,
                parser_used=self.name,
                raw_output=text,
                refusal_detected=False,
                error=str(e),
            )
    
    @abstractmethod
    def _parse_impl(self, text: str, **kwargs) -> Any:
        """
        Implementation of parsing logic.
        
        Override in subclasses.
        """
        pass
    
    def _preprocess(self, text: str) -> str:
        """
        Pre-process text before parsing.
        
        Args:
            text: Raw text
        
        Returns:
            Processed text
        """
        if text is None:
            return ""
        
        result = text
        
        if self.strip_whitespace:
            result = result.strip()
        
        if self.lowercase:
            result = result.lower()
        
        if self.max_length and len(result) > self.max_length:
            result = result[:self.max_length]
        
        return result
    
    def _calculate_confidence(self, text: str, parsed: Any) -> float:
        """
        Calculate confidence score for parsing.
        
        Args:
            text: Processed text
            parsed: Parsed content
        
        Returns:
            Confidence score (0.0 - 1.0)
        """
        # Base confidence
        if parsed is None:
            return 0.0
        
        # Higher confidence for non-empty results
        if isinstance(parsed, str) and len(parsed) > 0:
            return 0.8
        
        if isinstance(parsed, (list, dict)) and len(parsed) > 0:
            return 0.9
        
        return 0.5
    
    def get_confidence(self, text: str, parsed: Any) -> float:
        """Get confidence score."""
        return self._calculate_confidence(text, parsed)
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing common artifacts."""
        # Remove common prefixes
        prefixes = [
            r'^\s*```\w*\s*',  # Code block start
            r'\s*```\s*$',      # Code block end
            r'^\s*["\']+',       # Leading quotes
            r'["\']+\s*$',       # Trailing quotes
        ]
        
        result = text
        for pattern in prefixes:
            result = re.sub(pattern, '', result, flags=re.MULTILINE)
        
        return result.strip()
    
    def extract_between(
        self,
        text: str,
        start: str,
        end: str,
        inclusive: bool = False
    ) -> Optional[str]:
        """
        Extract text between two markers.
        
        Args:
            text: Source text
            start: Start marker
            end: End marker
            inclusive: Include markers in result
        
        Returns:
            Extracted text or None
        """
        pattern = f"{re.escape(start)}(.*?){re.escape(end)}"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            result = match.group(0 if inclusive else 1)
            return result.strip()
        
        return None
