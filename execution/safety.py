"""Restricted Python execution with import restrictions."""

import ast
import builtins
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


class SafetyError(Exception):
    """Error raised when code violates safety constraints."""
    pass


@dataclass
class SafetyCheckResult:
    """Result of safety check."""
    is_safe: bool
    violations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "violations": self.violations,
        }


class RestrictedPythonExecutor:
    """
    Executor with restricted Python capabilities.
    
    Features:
    - AST-based code analysis
    - Import restrictions
    - Built-in function restrictions
    - Dangerous operation blocking
    """
    
    # Dangerous built-ins to block
    BLOCKED_BUILTINS: Set[str] = {
        'eval', 'exec', 'compile', '__import__', 'open',
        'input', 'raw_input', 'reload', 'exit', 'quit',
        'help', 'license', 'credits',
    }
    
    # Dangerous modules to block
    BLOCKED_MODULES: Set[str] = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'http',
        'ftplib', 'smtplib', 'email', 'ctypes', 'mmap',
        'pickle', 'shelve', 'dbm', 'sqlite3', 'threading',
        'multiprocessing', 'concurrent', 'asyncio',
        'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    }
    
    # Dangerous AST node types
    BLOCKED_AST_NODES: Set[str] = {
        'Import', 'ImportFrom',  # Handled separately
        'Exec',  # Python 2
        'Call',  # Checked for dangerous calls
    }
    
    def __init__(
        self,
        allowed_imports: Optional[List[str]] = None,
        blocked_imports: Optional[List[str]] = None,
        allowed_builtins: Optional[List[str]] = None,
    ):
        self.allowed_imports = set(allowed_imports or [])
        self.blocked_imports = set(blocked_imports or []) | self.BLOCKED_MODULES
        self.allowed_builtins = set(allowed_builtins or [])
        
        # Create restricted globals
        self._restricted_globals = self._build_restricted_globals()
    
    def _build_restricted_globals(self) -> Dict[str, Any]:
        """Build restricted globals dictionary."""
        restricted = {}
        
        # Copy safe builtins
        for name in dir(builtins):
            if name not in self.BLOCKED_BUILTINS:
                if not self.allowed_builtins or name in self.allowed_builtins:
                    restricted[name] = getattr(builtins, name)
        
        # Add safe modules
        safe_modules = ['math', 'random', 'datetime', 'json', 're', 'string']
        for mod_name in safe_modules:
            if mod_name not in self.blocked_imports:
                try:
                    restricted[mod_name] = __import__(mod_name)
                except ImportError:
                    pass
        
        # Add __builtins__
        restricted['__builtins__'] = restricted
        
        return restricted
    
    def check_safety(self, code: str) -> SafetyCheckResult:
        """
        Check if code is safe to execute.
        
        Args:
            code: Python code to check
        
        Returns:
            SafetyCheckResult
        """
        violations = []
        
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return SafetyCheckResult(
                is_safe=False,
                violations=[f"Syntax error: {e}"]
            )
        
        # Check for dangerous imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if self._is_blocked_import(module_name):
                        violations.append(
                            f"Blocked import: {module_name}"
                        )
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if self._is_blocked_import(module_name):
                        violations.append(
                            f"Blocked import from: {module_name}"
                        )
            
            elif isinstance(node, ast.Call):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        violations.append(
                            f"Blocked builtin call: {node.func.id}"
                        )
                
                # Check for getattr with dangerous attributes
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ('__subclasses__', '__bases__', '__class__', '__globals__'):
                        violations.append(
                            f"Potentially dangerous attribute access: {node.func.attr}"
                        )
            
            elif isinstance(node, ast.Attribute):
                # Check for dangerous attribute access
                if node.attr in ('__subclasses__', '__bases__', '__class__', '__globals__', '__code__'):
                    violations.append(
                        f"Potentially dangerous attribute: {node.attr}"
                    )
        
        return SafetyCheckResult(
            is_safe=len(violations) == 0,
            violations=violations
        )
    
    def _is_blocked_import(self, module_name: str) -> bool:
        """Check if a module import is blocked."""
        # Check if explicitly allowed
        if module_name in self.allowed_imports:
            return False
        
        # Check if blocked
        if module_name in self.blocked_imports:
            return True
        
        return False
    
    def execute(
        self,
        code: str,
        local_vars: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute code in restricted environment.
        
        Args:
            code: Python code to execute
            local_vars: Local variables to provide
        
        Returns:
            Tuple of (success, result, error_message)
        """
        # Check safety first
        safety = self.check_safety(code)
        if not safety.is_safe:
            return False, None, f"Safety violations: {safety.violations}"
        
        # Prepare locals
        exec_locals = local_vars or {}
        
        try:
            # Execute in restricted environment
            exec(code, self._restricted_globals, exec_locals)
            return True, exec_locals, None
        except Exception as e:
            return False, None, str(e)
    
    def compile_restricted(
        self,
        code: str,
        filename: str = '<inline>',
        mode: str = 'exec'
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Compile code with safety checks.
        
        Returns:
            Tuple of (success, compiled_code_or_None, error_message)
        """
        # Check safety
        safety = self.check_safety(code)
        if not safety.is_safe:
            return False, None, f"Safety violations: {safety.violations}"
        
        try:
            compiled = compile(code, filename, mode)
            return True, compiled, None
        except SyntaxError as e:
            return False, None, str(e)
    
    def get_allowed_imports(self) -> List[str]:
        """Get list of allowed imports."""
        return sorted(self.allowed_imports)
    
    def get_blocked_imports(self) -> List[str]:
        """Get list of blocked imports."""
        return sorted(self.blocked_imports)
    
    def add_allowed_import(self, module_name: str):
        """Add a module to the allowed imports list."""
        self.allowed_imports.add(module_name)
        if module_name in self.blocked_imports:
            self.blocked_imports.remove(module_name)
        # Rebuild globals
        self._restricted_globals = self._build_restricted_globals()
    
    def add_blocked_import(self, module_name: str):
        """Add a module to the blocked imports list."""
        self.blocked_imports.add(module_name)
        if module_name in self.allowed_imports:
            self.allowed_imports.remove(module_name)
        # Rebuild globals
        self._restricted_globals = self._build_restricted_globals()


def is_safe_code(code: str, blocked_modules: Optional[List[str]] = None) -> bool:
    """
    Quick check if code is safe.
    
    Args:
        code: Python code to check
        blocked_modules: Additional modules to block
    
    Returns:
        True if code appears safe
    """
    executor = RestrictedPythonExecutor(
        blocked_imports=blocked_modules or []
    )
    result = executor.check_safety(code)
    return result.is_safe


def safe_exec(
    code: str,
    local_vars: Optional[Dict[str, Any]] = None,
    allowed_imports: Optional[List[str]] = None,
) -> Tuple[bool, Any, Optional[str]]:
    """
    Safely execute Python code.
    
    Args:
        code: Python code to execute
        local_vars: Local variables
        allowed_imports: Modules allowed to import
    
    Returns:
        Tuple of (success, locals_dict, error_message)
    """
    executor = RestrictedPythonExecutor(allowed_imports=allowed_imports)
    return executor.execute(code, local_vars)
