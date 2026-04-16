"""Unit tests for all parser edge cases."""

import unittest
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.code_parser import CodeParser
from parsers.mc_parser import MultipleChoiceParser
from parsers.json_parser import JSONParser
from parsers.refusal_detector import RefusalDetector
from parsers.robust_parser import RobustParser
from core.base import ParsedResult


class TestCodeParser(unittest.TestCase):
    """Test code extraction parser."""
    
    def setUp(self):
        self.parser = CodeParser(language="python")
    
    def test_extract_code_block_markdown(self):
        """Test extracting code from markdown fences."""
        output = """
Here's the solution:

```python
def add(a, b):
    return a + b
```
"""
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        self.assertIn("def add", result.content)
        self.assertEqual(result.parser_used, "code")
    
    def test_extract_last_code_block(self):
        """Test extracting last code block when multiple present."""
        output = """
```python
# Old version
def old():
    pass
```

Here's the correct version:

```python
def new():
    return 42
```
"""
        result = self.parser.parse(output)
        self.assertIn("def new", result.content)
        self.assertNotIn("def old", result.content)
    
    def test_extract_code_without_fences(self):
        """Test extracting code without markdown fences."""
        output = """
def calculate(x):
    return x * 2

This is the function.
"""
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        self.assertIn("def calculate", result.content)
    
    def test_extract_code_with_indentation(self):
        """Test extracting code with proper indentation."""
        output = """
    def indented():
        return True
"""
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
    
    def test_no_code_found(self):
        """Test handling when no code is found."""
        output = "This is just text with no code."
        result = self.parser.parse(output)
        # Should return raw as fallback
        self.assertIsNotNone(result.content)
    
    def test_language_detection(self):
        """Test language detection in code blocks."""
        output = """
```javascript
function hello() {
    return "world";
}
```
"""
        parser = CodeParser(language="javascript")
        result = parser.parse(output)
        self.assertIn("function hello", result.content)


class TestMultipleChoiceParser(unittest.TestCase):
    """Test multiple choice parser."""
    
    def setUp(self):
        self.parser = MultipleChoiceParser()
    
    def test_extract_option_a(self):
        """Test extracting option A."""
        output = "The answer is A."
        result = self.parser.parse(output)
        self.assertEqual(result.content, "A")
    
    def test_extract_option_b(self):
        """Test extracting option B."""
        output = "I think the answer is (B)."
        result = self.parser.parse(output)
        self.assertEqual(result.content, "B")
    
    def test_extract_option_c(self):
        """Test extracting option C."""
        output = "Answer: C"
        result = self.parser.parse(output)
        self.assertEqual(result.content, "C")
    
    def test_extract_option_d(self):
        """Test extracting option D."""
        output = "D."
        result = self.parser.parse(output)
        self.assertEqual(result.content, "D")
    
    def test_extract_with_explanation(self):
        """Test extracting option with explanation."""
        output = "The answer is B because it makes the most sense."
        result = self.parser.parse(output)
        self.assertEqual(result.content, "B")
    
    def test_extract_lowercase(self):
        """Test extracting lowercase option."""
        output = "answer: a"
        result = self.parser.parse(output)
        self.assertEqual(result.content, "A")
    
    def test_extract_with_brackets(self):
        """Test extracting option in brackets."""
        output = "I choose [C]"
        result = self.parser.parse(output)
        self.assertEqual(result.content, "C")
    
    def test_no_option_found(self):
        """Test when no option is found."""
        output = "I don't know the answer."
        result = self.parser.parse(output)
        # Should return raw as fallback
        self.assertIsNotNone(result.content)
    
    def test_multiple_options(self):
        """Test when multiple options mentioned."""
        output = "It could be A or B, but I think C."
        result = self.parser.parse(output)
        # Should extract last one
        self.assertEqual(result.content, "C")


class TestJSONParser(unittest.TestCase):
    """Test JSON parser."""
    
    def setUp(self):
        self.parser = JSONParser()
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        output = '{"name": "test", "value": 42}'
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        data = json.loads(result.content)
        self.assertEqual(data["name"], "test")
        self.assertEqual(data["value"], 42)
    
    def test_parse_json_with_markdown(self):
        """Test parsing JSON in markdown fences."""
        output = """
```json
{"key": "value"}
```
"""
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        data = json.loads(result.content)
        self.assertEqual(data["key"], "value")
    
    def test_parse_json_with_single_quotes(self):
        """Test parsing JSON with single quotes (repair)."""
        output = "{'key': 'value'}"
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        data = json.loads(result.content)
        self.assertEqual(data["key"], "value")
    
    def test_parse_json_with_trailing_comma(self):
        """Test parsing JSON with trailing comma (repair)."""
        output = '{"a": 1, "b": 2,}'
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        data = json.loads(result.content)
        self.assertEqual(data["a"], 1)
    
    def test_parse_nested_json(self):
        """Test parsing nested JSON."""
        output = '{"outer": {"inner": "value"}}'
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        data = json.loads(result.content)
        self.assertEqual(data["outer"]["inner"], "value")
    
    def test_parse_json_array(self):
        """Test parsing JSON array."""
        output = '[1, 2, 3, "test"]'
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        data = json.loads(result.content)
        self.assertEqual(len(data), 4)
    
    def test_parse_invalid_json(self):
        """Test handling invalid JSON."""
        output = "This is not JSON"
        result = self.parser.parse(output)
        # Should return raw as fallback
        self.assertIsNotNone(result.content)


class TestRefusalDetector(unittest.TestCase):
    """Test refusal detection."""
    
    def setUp(self):
        self.detector = RefusalDetector()
    
    def test_detect_cannot(self):
        """Test detecting 'I cannot' refusal."""
        output = "I cannot help with that."
        result = self.detector.detect(output)
        self.assertTrue(result.is_refusal)
    
    def test_detect_sorry(self):
        """Test detecting 'I'm sorry' refusal."""
        output = "I'm sorry, but I can't do that."
        result = self.detector.detect(output)
        self.assertTrue(result.is_refusal)
    
    def test_detect_dont_know(self):
        """Test detecting 'I don't know' refusal."""
        output = "I don't know the answer to that."
        result = self.detector.detect(output)
        self.assertTrue(result.is_refusal)
    
    def test_detect_as_ai(self):
        """Test detecting 'As an AI' refusal."""
        output = "As an AI, I cannot provide that information."
        result = self.detector.detect(output)
        self.assertTrue(result.is_refusal)
    
    def test_detect_not_refusal(self):
        """Test non-refusal content."""
        output = "Here's the solution: def add(a, b): return a + b"
        result = self.detector.detect(output)
        self.assertFalse(result.is_refusal)
    
    def test_detect_case_insensitive(self):
        """Test case-insensitive detection."""
        output = "I CANNOT HELP WITH THIS"
        result = self.detector.detect(output)
        self.assertTrue(result.is_refusal)
    
    def test_detect_partial_match(self):
        """Test detecting partial match."""
        output = "Sorry, I don't have that information."
        result = self.detector.detect(output)
        self.assertTrue(result.is_refusal)


class TestRobustParser(unittest.TestCase):
    """Test robust parser with fallback chain."""
    
    def setUp(self):
        self.parser = RobustParser()
    
    def test_parse_code_with_fallback(self):
        """Test parsing code with fallback chain."""
        output = """
```python
def test():
    return True
```
"""
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        self.assertIn("def test", result.content)
    
    def test_parse_json_with_fallback(self):
        """Test parsing JSON with fallback chain."""
        output = '{"key": "value"}'
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        data = json.loads(result.content)
        self.assertEqual(data["key"], "value")
    
    def test_parse_mc_with_fallback(self):
        """Test parsing multiple choice with fallback chain."""
        output = "The answer is B."
        result = self.parser.parse(output)
        self.assertEqual(result.content, "B")
    
    def test_parse_refusal(self):
        """Test handling refusal."""
        output = "I cannot help with that request."
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        # Should mark as refusal
    
    def test_parse_raw_fallback(self):
        """Test raw fallback when all parsers fail."""
        output = "Just some random text without structure"
        result = self.parser.parse(output)
        self.assertIsNotNone(result.content)
        self.assertEqual(result.content, output.strip())
    
    def test_confidence_scoring(self):
        """Test confidence scoring."""
        # High confidence for clear code
        output = "```python\ndef test(): pass\n```"
        result = self.parser.parse(output)
        self.assertGreater(result.confidence, 0.7)


class TestParserEdgeCases(unittest.TestCase):
    """Test edge cases across all parsers."""
    
    def test_empty_string(self):
        """Test parsing empty string."""
        for parser in [CodeParser(), MultipleChoiceParser(), JSONParser()]:
            result = parser.parse("")
            self.assertIsNotNone(result)
    
    def test_whitespace_only(self):
        """Test parsing whitespace only."""
        for parser in [CodeParser(), MultipleChoiceParser(), JSONParser()]:
            result = parser.parse("   \n\t  ")
            self.assertIsNotNone(result)
    
    def test_very_long_output(self):
        """Test parsing very long output."""
        long_output = "x" * 10000
        result = CodeParser().parse(long_output)
        self.assertIsNotNone(result)
    
    def test_unicode_content(self):
        """Test parsing unicode content."""
        output = "```python\ndef 你好(): pass\n```"
        result = CodeParser().parse(output)
        self.assertIsNotNone(result.content)
    
    def test_mixed_content(self):
        """Test parsing mixed content types."""
        output = """
Here's some text.

```python
code here
```

And the answer is A.

```json
{"key": "value"}
```
"""
        # Code parser should extract code
        code_result = CodeParser().parse(output)
        self.assertIn("code here", code_result.content)
        
        # MC parser should extract option
        mc_result = MultipleChoiceParser().parse(output)
        self.assertEqual(mc_result.content, "A")
        
        # JSON parser should extract JSON
        json_result = JSONParser().parse(output)
        self.assertIn("key", json_result.content)


def run_tests():
    """Run all parser tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCodeParser))
    suite.addTests(loader.loadTestsFromTestCase(TestMultipleChoiceParser))
    suite.addTests(loader.loadTestsFromTestCase(TestJSONParser))
    suite.addTests(loader.loadTestsFromTestCase(TestRefusalDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustParser))
    suite.addTests(loader.loadTestsFromTestCase(TestParserEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
