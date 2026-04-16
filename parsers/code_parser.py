"""Code parser with priority-based extraction and language detection."""

import re
from typing import Any, Dict, List, Optional, Tuple

from .base_parser import BaseParser
from core.base import ParsedResult
from core.registry import register_parser


@register_parser("code")
class CodeParser(BaseParser):
    """
    Parser for code extraction from model outputs.
    
    Features:
    - Priority-based extraction (last code block preferred)
    - Language detection
    - Multiple code block formats
    - Fallback to raw text
    """
    
    name: str = "code"
    
    # Common programming languages
    LANGUAGES = [
        "python", "py", "javascript", "js", "typescript", "ts",
        "java", "c", "cpp", "cxx", "c++", "csharp", "cs", "c#",
        "go", "golang", "rust", "rs", "ruby", "rb",
        "php", "swift", "kotlin", "scala", "r", "matlab",
        "sql", "bash", "sh", "shell", "powershell", "ps",
        "html", "xml", "css", "json", "yaml", "yml",
    ]
    
    def __init__(
        self,
        language: Optional[str] = None,
        extraction_mode: str = "last_block",
        strip_comments: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.language = language
        self.extraction_mode = extraction_mode
        self.strip_comments = strip_comments
    
    def _parse_impl(self, text: str, **kwargs) -> str:
        """
        Extract code from text.
        
        Args:
            text: Raw model output
            **kwargs: May contain 'language' override
        
        Returns:
            Extracted code
        """
        language = kwargs.get('language', self.language)
        extraction_mode = kwargs.get('extraction_mode', self.extraction_mode)
        
        # Try to extract code blocks
        code_blocks = self._extract_code_blocks(text)
        
        if code_blocks:
            if extraction_mode == "last_block":
                code = code_blocks[-1][0]
                detected_lang = code_blocks[-1][1]
            elif extraction_mode == "first_block":
                code = code_blocks[0][0]
                detected_lang = code_blocks[0][1]
            else:  # all
                code = "\n\n".join([cb[0] for cb in code_blocks])
                detected_lang = code_blocks[-1][1] if code_blocks else None
            
            # Validate language if specified
            if language and detected_lang:
                if detected_lang.lower() != language.lower():
                    # Language mismatch, but still return the code
                    pass
            
            return self._postprocess_code(code)
        
        # No code blocks found, return cleaned text
        return self._postprocess_code(self.clean_text(text))
    
    # Data formats that should be excluded when looking for "code"
    DATA_FORMATS = {"json", "yaml", "yml", "xml", "csv", "toml"}
    
    def _extract_code_blocks(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """
        Extract code blocks from text.
        
        Returns:
            List of (code, language) tuples
        """
        blocks = []
        
        # Pattern 1: Markdown code blocks with language (HIGHEST PRIORITY)
        # Split by ``` to find code block boundaries
        parts = text.split('```')
        # parts[0] = text before first ```
        # parts[1] = lang\ncode (if odd index)
        # parts[2] = text after closing ```
        for i in range(1, len(parts), 2):
            block_content = parts[i].strip()
            if not block_content:
                continue
            
            # Check if there's a language tag on first line
            lines = block_content.split('\n', 1)
            if len(lines) == 2:
                first_line, code = lines
                first_line = first_line.strip()
                # If first line looks like a language tag (single word, no spaces)
                if first_line and ' ' not in first_line and len(first_line) < 20:
                    lang = first_line.lower()
                    code = code.strip()
                else:
                    # No language tag, treat first line as code
                    lang = None
                    code = block_content
            else:
                # Single line block
                lang = None
                code = block_content
            
            if code:
                blocks.append((code, lang))
        
        # If we found markdown code blocks, filter out data formats for code extraction
        if blocks:
            # Filter to only programming language blocks (exclude json, yaml, etc)
            code_blocks = [(code, lang) for code, lang in blocks 
                          if lang not in self.DATA_FORMATS]
            if code_blocks:
                return code_blocks
            # If no code blocks after filtering, return all (fallback)
            return blocks
        
        # Pattern 2: Look for Python function definitions without fences
        # Match function definitions and their indented body
        func_blocks = []
        pattern2 = r'(def\s+\w+\s*\([^)]*\)\s*:(?:\n(?:    |\t).+)+)'
        for match in re.finditer(pattern2, text, re.MULTILINE):
            code = match.group(1).strip()
            if code:
                lang = self._detect_language(code)
                func_blocks.append((code, lang))
        
        # If we found function definitions, return those (higher priority than indented)
        if func_blocks:
            return func_blocks
        
        # Pattern 3: Indented code blocks (4 spaces or tab) - as fallback
        pattern3 = r'(?:^|\n)(?:    |\t)(.+?)(?=\n[^\s]|\Z)'
        for match in re.finditer(pattern3, text, re.DOTALL):
            code = match.group(0).strip()
            if code:
                # Detect language from content
                lang = self._detect_language(code)
                blocks.append((code, lang))
        
        # Pattern 4: Code between specific markers
        pattern4 = r'<code>(.*?)</code>'
        for match in re.finditer(pattern4, text, re.DOTALL | re.IGNORECASE):
            code = match.group(1).strip()
            if code:
                lang = self._detect_language(code)
                blocks.append((code, lang))
        
        return blocks
    
    def _detect_language(self, code: str) -> Optional[str]:
        """
        Detect programming language from code.
        
        Args:
            code: Code snippet
        
        Returns:
            Detected language or None
        """
        # Check for shebang
        shebang_match = re.match(r'^#!\s*/usr/bin/env\s+(\w+)', code)
        if shebang_match:
            return shebang_match.group(1)
        
        # Language-specific patterns
        patterns = {
            "python": [
                r'^\s*(def|class|import|from)\s+',
                r'print\s*\(',
                r':\s*\n\s+(if|for|while|def|class)',
            ],
            "javascript": [
                r'\b(const|let|var)\s+\w+\s*=',
                r'function\s*\w*\s*\(',
                r'=>\s*{',
                r'console\.log\s*\(',
            ],
            "java": [
                r'\bpublic\s+(class|static|void)\s+',
                r'System\.(out|err)\.print',
                r'import\s+java\.',
            ],
            "c": [
                r'#include\s*<',
                r'int\s+main\s*\(',
                r'printf\s*\(',
            ],
            "cpp": [
                r'#include\s*<\w+>',
                r'std::',
                r'cout\s*<<',
            ],
            "go": [
                r'^package\s+\w+',
                r'func\s+\w+\s*\(',
                r'fmt\.Print',
            ],
            "rust": [
                r'^fn\s+\w+\s*\(',
                r'let\s+mut\s+',
                r'println!\s*\(',
            ],
        }
        
        for lang, lang_patterns in patterns.items():
            for pattern in lang_patterns:
                if re.search(pattern, code, re.MULTILINE):
                    return lang
        
        return None
    
    def _postprocess_code(self, code: str) -> str:
        """
        Post-process extracted code.
        
        Args:
            code: Extracted code
        
        Returns:
            Cleaned code
        """
        if not code:
            return ""
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Note: Language tags are already stripped in _extract_code_blocks
        # Don't use generic regex here as it strips valid code like "def function():"
        
        # Strip comments if requested
        if self.strip_comments:
            code = self._strip_comments(code)
        
        return code
    
    def _strip_comments(self, code: str) -> str:
        """Remove comments from code."""
        # Python-style comments
        lines = code.split('\n')
        cleaned = []
        
        for line in lines:
            # Remove inline comments (simple version)
            if '#' in line:
                # Check if # is inside a string
                quote_chars = ['"', "'"]
                in_string = False
                string_char = None
                cleaned_line = []
                
                for char in line:
                    if not in_string and char in quote_chars:
                        in_string = True
                        string_char = char
                    elif in_string and char == string_char:
                        in_string = False
                        string_char = None
                    elif not in_string and char == '#':
                        break
                    
                    cleaned_line.append(char)
                
                line = ''.join(cleaned_line).rstrip()
            
            if line:  # Keep non-empty lines
                cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    def _calculate_confidence(self, text: str, parsed: Any) -> float:
        """Calculate confidence based on code block presence."""
        if not parsed or not isinstance(parsed, str):
            return 0.0
        
        # Check if we found actual code blocks
        code_blocks = self._extract_code_blocks(text)
        
        if code_blocks:
            # Higher confidence if we found markdown code blocks
            if re.search(r'```', text):
                return 0.95
            return 0.8
        
        # No code blocks, very low confidence - this is just a fallback
        if len(parsed.strip()) > 0:
            return 0.3
        
        return 0.0
    
    def extract_function(self, code: str, function_name: str) -> Optional[str]:
        """
        Extract a specific function from code.
        
        Args:
            code: Source code
            function_name: Name of function to extract
        
        Returns:
            Function code or None
        """
        # Python function pattern
        pattern = rf'(^|\n)(def\s+{re.escape(function_name)}\s*\([^)]*\)\s*:.*?)(?=\n\n|\Z)'
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            return match.group(2).strip()
        
        return None
    
    def has_syntax_errors(self, code: str, language: str = "python") -> Tuple[bool, Optional[str]]:
        """
        Check if code has syntax errors.
        
        Returns:
            Tuple of (has_errors, error_message)
        """
        if language.lower() in ["python", "py"]:
            import ast
            try:
                ast.parse(code)
                return False, None
            except SyntaxError as e:
                return True, str(e)
        
        # For other languages, basic check only
        return False, None
