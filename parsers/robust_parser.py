"""Robust parser with multi-stage fallback chain."""

from typing import Any, Dict, List, Optional, Tuple, Type

from .base_parser import BaseParser
from .code_parser import CodeParser
from .mc_parser import MultipleChoiceParser
from .json_parser import JSONParser
from .refusal_detector import RefusalDetector
from core.base import ParsedResult
from core.registry import register_parser


@register_parser("robust")
class RobustParser(BaseParser):
    """
    Robust parser with multi-stage fallback chain.
    
    Features:
    - Tries multiple parsers in sequence
    - Refusal detection
    - Confidence scoring
    - Fallback to raw text
    """
    
    name: str = "robust"
    
    def __init__(
        self,
        parsers: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        detect_refusals: bool = True,
        fallback_to_raw: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.parser_names = parsers or ["code", "mc", "json"]
        self.min_confidence = min_confidence
        self.detect_refusals = detect_refusals
        self.fallback_to_raw = fallback_to_raw
        
        # Initialize parsers
        self._parsers: Dict[str, BaseParser] = {}
        self._refusal_detector = RefusalDetector() if detect_refusals else None
        
        self._init_parsers()
    
    def _init_parsers(self):
        """Initialize parser instances."""
        parser_map = {
            "code": CodeParser,
            "mc": MultipleChoiceParser,
            "json": JSONParser,
        }
        
        for name in self.parser_names:
            if name in parser_map:
                self._parsers[name] = parser_map[name]()
    
    def parse(self, text: str, **kwargs) -> ParsedResult:
        """
        Parse with multi-stage fallback.
        
        Args:
            text: Raw model output
            **kwargs: Parser-specific parameters
        
        Returns:
            ParsedResult with best available parsing
        """
        # Check for refusal first
        if self.detect_refusals and self._refusal_detector:
            refusal_result = self._refusal_detector.detect(text)
            if refusal_result.is_refusal:
                # Return raw text as fallback for refusals (don't return None)
                return ParsedResult(
                    content=text.strip() if text else None,
                    confidence=0.3,
                    parser_used="robust->refusal_fallback",
                    raw_output=text,
                    refusal_detected=True,
                    error=f"Refusal detected: {refusal_result.reason}",
                )
        
        # Try each parser in sequence
        results = []
        
        for parser_name in self.parser_names:
            parser = self._parsers.get(parser_name)
            if not parser:
                continue
            
            try:
                result = parser.parse(text, **kwargs)
                results.append((parser_name, result))
                
                # If we have a high-confidence result, use it
                if result.confidence >= self.min_confidence and result.content is not None:
                    return ParsedResult(
                        content=result.content,
                        confidence=result.confidence,
                        parser_used=f"robust->{parser_name}",
                        raw_output=text,
                        refusal_detected=False,
                        error=None,
                    )
                    
            except Exception as e:
                # Parser failed, continue to next
                results.append((parser_name, None))
                continue
        
        # No high-confidence result, pick the best one
        best_result = self._select_best_result(results)
        
        if best_result:
            parser_name, result = best_result
            return ParsedResult(
                content=result.content if result else None,
                confidence=result.confidence if result else 0.0,
                parser_used=f"robust->{parser_name}",
                raw_output=text,
                refusal_detected=False,
                error=None if result else "All parsers failed",
            )
        
        # Fallback to raw text
        if self.fallback_to_raw:
            return ParsedResult(
                content=text.strip() if text else None,
                confidence=0.3,
                parser_used="robust->raw",
                raw_output=text,
                refusal_detected=False,
                error=None,
            )
        
        # Complete failure
        return ParsedResult(
            content=None,
            confidence=0.0,
            parser_used="robust->failed",
            raw_output=text,
            refusal_detected=False,
            error="All parsing attempts failed",
        )
    
    def _parse_impl(self, text: str, **kwargs) -> Any:
        """Internal parse implementation."""
        result = self.parse(text, **kwargs)
        return result.content
    
    def _select_best_result(
        self,
        results: List[Tuple[str, Optional[ParsedResult]]]
    ) -> Optional[Tuple[str, ParsedResult]]:
        """Select the best result from multiple parsers."""
        valid_results = [
            (name, result) for name, result in results
            if result is not None and result.content is not None
        ]
        
        if not valid_results:
            return None
        
        # Sort by confidence
        valid_results.sort(key=lambda x: x[1].confidence, reverse=True)
        
        return valid_results[0]
    
    def parse_with_details(
        self,
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Parse with detailed information about all attempts.
        
        Returns:
            Dictionary with detailed parsing information
        """
        details = {
            "input_preview": text[:200] if text else "",
            "input_length": len(text) if text else 0,
            "attempts": [],
            "refusal_check": None,
            "final_result": None,
        }
        
        # Check refusal
        if self.detect_refusals and self._refusal_detector:
            refusal = self._refusal_detector.detect(text)
            details["refusal_check"] = refusal.to_dict()
        
        # Try each parser
        for parser_name in self.parser_names:
            parser = self._parsers.get(parser_name)
            if not parser:
                continue
            
            try:
                result = parser.parse(text, **kwargs)
                details["attempts"].append({
                    "parser": parser_name,
                    "success": result.content is not None,
                    "confidence": result.confidence,
                    "content_type": type(result.content).__name__ if result.content else None,
                })
            except Exception as e:
                details["attempts"].append({
                    "parser": parser_name,
                    "success": False,
                    "error": str(e),
                })
        
        # Get final result
        final = self.parse(text, **kwargs)
        details["final_result"] = {
            "parser_used": final.parser_used,
            "confidence": final.confidence,
            "refusal_detected": final.refusal_detected,
            "has_error": final.error is not None,
        }
        
        return details
    
    def add_parser(self, name: str, parser: BaseParser):
        """Add a parser to the chain."""
        self._parsers[name] = parser
        self.parser_names.append(name)
    
    def set_parser_order(self, order: List[str]):
        """Set the order of parsers to try."""
        self.parser_names = order
