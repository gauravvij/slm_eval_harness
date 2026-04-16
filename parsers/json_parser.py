"""JSON parser with error tolerance."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_parser import BaseParser
from core.base import ParsedResult
from core.registry import register_parser


@register_parser("json")
class JSONParser(BaseParser):
    """
    Parser for JSON output with error tolerance.
    
    Features:
    - Standard JSON parsing
    - JSON repair for common errors
    - Extraction from markdown code blocks
    - Partial parsing fallback
    """
    
    name: str = "json"
    
    def __init__(
        self,
        strict: bool = False,
        repair: bool = True,
        schema: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.strict = strict
        self.repair = repair
        self.schema = schema
    
    def _parse_impl(self, text: str, **kwargs) -> Any:
        """
        Parse JSON from text.
        
        Args:
            text: Raw model output
            **kwargs: May contain 'schema' override
        
        Returns:
            JSON string (not parsed object) to maintain consistency
        """
        schema = kwargs.get('schema', self.schema)
        
        # Try standard parsing first
        try:
            json.loads(text)
            return text.strip()
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if json_match:
            try:
                extracted = json_match.group(1).strip()
                json.loads(extracted)
                return extracted
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON array or object in text
        # Look for outermost brackets/braces
        for pattern in [r'(\[[^\[\]]*\])', r'(\{[^\{\}]*\})']:
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if matches:
                # Try the longest match (most complete)
                for match in sorted(matches, key=lambda m: len(m.group(0)), reverse=True):
                    try:
                        extracted = match.group(1)
                        json.loads(extracted)
                        return extracted
                    except json.JSONDecodeError:
                        continue
        
        # Try repair if enabled
        if self.repair:
            repaired = self._repair_json(text)
            if repaired:
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    pass
        
        # Final fallback: return raw text
        if not self.strict:
            return text
        
        raise ValueError(f"Could not parse JSON from text: {text[:100]}...")
    
    def _repair_json(self, text: str) -> Optional[str]:
        """
        Attempt to repair common JSON errors.
        
        Args:
            text: Potentially malformed JSON
        
        Returns:
            Repaired JSON string or None
        """
        repaired = text.strip()
        
        # Common repairs
        repairs = [
            # Remove trailing commas
            (r',\s*}', '}'),
            (r',\s*]', ']'),
            # Fix single quotes to double quotes
            ("'", '"'),
            # Fix unquoted keys (simple cases)
            (r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3'),
            # Remove comments
            (r'//.*?\n', '\n'),
            (r'/\*.*?\*/', '', re.DOTALL),
        ]
        
        for pattern, replacement, *flags in repairs:
            flag = flags[0] if flags else 0
            repaired = re.sub(pattern, replacement, repaired, flags=flag)
        
        # Try to parse
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None
    
    def _calculate_confidence(self, text: str, parsed: Any) -> float:
        """Calculate confidence based on parsing success."""
        if parsed is None:
            return 0.0
        
        # Try standard parsing
        try:
            json.loads(text)
            return 1.0  # Perfect parse
        except json.JSONDecodeError:
            pass
        
        # Check if we extracted from code block
        if re.search(r'```json', text):
            return 0.9
        
        # Check if we found JSON structure
        if re.search(r'[\{\[]', text):
            return 0.7
        
        # Repaired or fallback
        return 0.5
    
    def parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse function calls from JSON.
        
        Args:
            text: Model output
        
        Returns:
            List of function call dicts with 'name' and 'arguments'
        """
        parsed = self._parse_impl(text)
        
        if isinstance(parsed, list):
            return self._normalize_function_calls(parsed)
        elif isinstance(parsed, dict):
            # Single function call
            return self._normalize_function_calls([parsed])
        else:
            return []
    
    def _normalize_function_calls(self, calls: List[Dict]) -> List[Dict[str, Any]]:
        """Normalize function call format."""
        normalized = []
        
        for call in calls:
            if not isinstance(call, dict):
                continue
            
            # Handle different formats
            if 'name' in call and 'arguments' in call:
                # Standard format
                normalized.append({
                    'name': call['name'],
                    'arguments': call['arguments']
                })
            elif 'function' in call and 'arguments' in call:
                # Alternative format
                normalized.append({
                    'name': call['function'],
                    'arguments': call['arguments']
                })
            elif 'name' in call and 'args' in call:
                # args instead of arguments
                normalized.append({
                    'name': call['name'],
                    'arguments': call['args']
                })
            elif len(call) == 1:
                # Single key-value where key is function name
                name = list(call.keys())[0]
                normalized.append({
                    'name': name,
                    'arguments': call[name]
                })
        
        return normalized
    
    def validate_against_schema(self, data: Any, schema: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate parsed JSON against schema.
        
        Args:
            data: Parsed JSON data
            schema: JSON schema to validate against
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        schema = schema or self.schema
        
        if schema is None:
            return True, None
        
        try:
            from jsonschema import validate, ValidationError
            validate(instance=data, schema=schema)
            return True, None
        except ValidationError as e:
            return False, str(e)
        except ImportError:
            # jsonschema not available
            return True, None
    
    def extract_json_from_markdown(self, text: str) -> Optional[str]:
        """Extract JSON from markdown code block."""
        match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def is_valid_json(self, text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
