"""Unit tests for sandbox safety."""

import unittest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.sandbox import SandboxExecutor, ExecutionResult
from execution.safety import RestrictedPythonExecutor, SafetyError


class TestSafetyChecker(unittest.TestCase):
    """Test safety checker."""
    
    def setUp(self):
        self.checker = RestrictedPythonExecutor()
    
    def test_safe_code(self):
        """Test that safe code passes."""
        code = "x = 1 + 2\nprint(x)"
        is_safe, error = self.checker.check_code(code)
        self.assertTrue(is_safe)
        self.assertIsNone(error)
    
    def test_import_os(self):
        """Test blocking os import."""
        code = "import os\nos.system('ls')"
        is_safe, error = self.checker.check_code(code)
        self.assertFalse(is_safe)
        self.assertIn("os", error.lower())
    
    def test_import_sys(self):
        """Test blocking sys import."""
        code = "import sys\nprint(sys.path)"
        is_safe, error = self.checker.check_code(code)
        self.assertFalse(is_safe)
        self.assertIn("sys", error.lower())
    
    def test_import_subprocess(self):
        """Test blocking subprocess import."""
        code = "import subprocess\nsubprocess.run(['ls'])"
        is_safe, error = self.checker.check_code(code)
        self.assertFalse(is_safe)
        self.assertIn("subprocess", error.lower())
    
    def test_from_import_os(self):
        """Test blocking 'from os import' pattern."""
        code = "from os import system\nsystem('ls')"
        is_safe, error = self.checker.check_code(code)
        self.assertFalse(is_safe)
    
    def test_safe_import_math(self):
        """Test allowing safe imports like math."""
        code = "import math\nprint(math.pi)"
        is_safe, error = self.checker.check_code(code)
        self.assertTrue(is_safe)
    
    def test_safe_import_random(self):
        """Test allowing safe imports like random."""
        code = "import random\nprint(random.randint(1, 10))"
        is_safe, error = self.checker.check_code(code)
        self.assertTrue(is_safe)
    
    def test_eval_call(self):
        """Test blocking eval()."""
        code = "eval('1 + 1')"
        is_safe, error = self.checker.check_code(code)
        self.assertFalse(is_safe)
        self.assertIn("eval", error.lower())
    
    def test_exec_call(self):
        """Test blocking exec()."""
        code = "exec('print(1)')"
        is_safe, error = self.checker.check_code(code)
        self.assertFalse(is_safe)
        self.assertIn("exec", error.lower())
    
    def test_compile_call(self):
        """Test blocking compile()."""
        code = "compile('print(1)', '<string>', 'exec')"
        is_safe, error = self.checker.check_code(code)
        self.assertFalse(is_safe)
        self.assertIn("compile", error.lower())
    
    def test_open_call(self):
        """Test blocking open()."""
        code = "open('/etc/passwd')"
        is_safe, error = self.checker.check_code(code)
        self.assertFalse(is_safe)
        self.assertIn("open", error.lower())
    
    def test_syntax_error(self):
        """Test handling syntax errors."""
        code = "def broken(\n    pass"
        is_safe, error = self.checker.check_code(code)
        self.assertFalse(is_safe)
        self.assertIn("syntax", error.lower())
    
    def test_empty_code(self):
        """Test handling empty code."""
        code = ""
        is_safe, error = self.checker.check_code(code)
        self.assertTrue(is_safe)


class TestSandbox(unittest.TestCase):
    """Test sandbox execution."""
    
    def setUp(self):
        self.sandbox = SandboxExecutor(timeout=5, max_memory_mb=128)
    
    def test_simple_execution(self):
        """Test simple code execution."""
        code = "x = 1 + 2\nprint(x)"
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
        self.assertIn("3", result.stdout)
    
    def test_function_execution(self):
        """Test function definition and execution."""
        code = """
def add(a, b):
    return a + b

result = add(2, 3)
print(result)
"""
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
        self.assertIn("5", result.stdout)
    
    def test_timeout(self):
        """Test timeout handling."""
        code = """
import time
time.sleep(10)
print("done")
"""
        result = self.sandbox.execute(code)
        self.assertFalse(result.success)
        self.assertIn("timeout", result.error.lower())
    
    def test_syntax_error(self):
        """Test syntax error handling."""
        code = "def broken(\n    pass"
        result = self.sandbox.execute(code)
        self.assertFalse(result.success)
        self.assertIn("syntax", result.error.lower())
    
    def test_runtime_error(self):
        """Test runtime error handling."""
        code = "1 / 0"
        result = self.sandbox.execute(code)
        self.assertFalse(result.success)
        self.assertIn("zero", result.error.lower())
    
    def test_name_error(self):
        """Test name error handling."""
        code = "print(undefined_variable)"
        result = self.sandbox.execute(code)
        self.assertFalse(result.success)
        self.assertIn("name", result.error.lower())
    
    def test_import_error(self):
        """Test import error handling."""
        code = "import nonexistent_module_xyz"
        result = self.sandbox.execute(code)
        self.assertFalse(result.success)
        self.assertIn("import", result.error.lower())
    
    def test_output_capture(self):
        """Test stdout capture."""
        code = """
print("line1")
print("line2")
"""
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
        self.assertIn("line1", result.stdout)
        self.assertIn("line2", result.stdout)
    
    def test_stderr_capture(self):
        """Test stderr capture."""
        code = """
import sys
sys.stderr.write("error message\n")
"""
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
        self.assertIn("error message", result.stderr)
    
    def test_return_value(self):
        """Test return value extraction."""
        code = """
result = {"key": "value"}
"""
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)


class TestSandboxEdgeCases(unittest.TestCase):
    """Test sandbox edge cases."""
    
    def setUp(self):
        self.sandbox = SandboxExecutor(timeout=5, max_memory_mb=128)
    
    def test_empty_code(self):
        """Test empty code."""
        result = self.sandbox.execute("")
        self.assertTrue(result.success)
    
    def test_whitespace_only(self):
        """Test whitespace-only code."""
        result = self.sandbox.execute("   \n\t  ")
        self.assertTrue(result.success)
    
    def test_very_long_output(self):
        """Test very long output."""
        code = "print('x' * 10000)"
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
        self.assertIn("x", result.stdout)
    
    def test_unicode_output(self):
        """Test unicode output."""
        code = "print('Hello 世界')"
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
    
    def test_multiple_prints(self):
        """Test multiple print statements."""
        code = """
for i in range(5):
    print(i)
"""
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
        for i in range(5):
            self.assertIn(str(i), result.stdout)
    
    def test_recursive_function(self):
        """Test recursive function."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
"""
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
        self.assertIn("120", result.stdout)
    
    def test_list_comprehension(self):
        """Test list comprehension."""
        code = "print([x**2 for x in range(5)])"
        result = self.sandbox.execute(code)
        self.assertTrue(result.success)
        self.assertIn("0", result.stdout)
        self.assertIn("1", result.stdout)
        self.assertIn("4", result.stdout)


class TestSandboxWithTests(unittest.TestCase):
    """Test sandbox with test cases."""
    
    def setUp(self):
        self.sandbox = Sandbox(timeout=5, max_memory_mb=128)
    
    def test_humaneval_style(self):
        """Test HumanEval-style code execution."""
        code = """
def has_close_elements(numbers: list, threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""
        test_code = """
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
"""
        result = self.sandbox.execute(code, test_code)
        self.assertTrue(result.success)
    
    def test_humaneval_failure(self):
        """Test HumanEval-style code with failure."""
        code = """
def broken_function(numbers: list, threshold: float) -> bool:
    return False  # Always returns False
"""
        test_code = """
assert has_close_elements([1.0, 2.8, 3.0], 0.5) == True
"""
        result = self.sandbox.execute(code, test_code)
        self.assertFalse(result.success)
        self.assertIn("assert", result.error.lower())


def run_tests():
    """Run all sandbox tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSafetyChecker))
    suite.addTests(loader.loadTestsFromTestCase(TestSandbox))
    suite.addTests(loader.loadTestsFromTestCase(TestSandboxEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestSandboxWithTests))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
