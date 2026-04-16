"""Subprocess-based safe code execution with resource limits."""

import os
import sys
import subprocess
import tempfile
import signal
import resource
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import time


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    return_code: int = 0
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    timed_out: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "timed_out": self.timed_out,
        }


class SandboxExecutor:
    """
    Sandboxed code executor using subprocess.
    
    Features:
    - Resource limits (time, memory)
    - Process isolation
    - Temporary file cleanup
    - Output capture
    """
    
    def __init__(
        self,
        timeout: float = 5.0,
        max_memory_mb: int = 512,
        tmp_dir: Optional[str] = None,
        allowed_imports: Optional[List[str]] = None,
        blocked_imports: Optional[List[str]] = None,
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.tmp_dir = tmp_dir or tempfile.gettempdir()
        self.allowed_imports = allowed_imports or []
        self.blocked_imports = blocked_imports or [
            "os", "sys", "subprocess", "socket", "urllib", "http",
            "ftplib", "smtplib", "email", "ctypes", "mmap", "pickle",
        ]
    
    def execute(
        self,
        code: str,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute code in a sandboxed subprocess.
        
        Args:
            code: The code to execute
            test_code: Optional test code to run after the main code
            entry_point: Optional entry point function to call
        
        Returns:
            ExecutionResult
        """
        # Build the full script
        script = self._build_script(code, test_code, entry_point)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            dir=self.tmp_dir,
            delete=False
        ) as f:
            f.write(script)
            temp_path = f.name
        
        try:
            return self._run_subprocess(temp_path)
        finally:
            # Cleanup
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def _build_script(
        self,
        code: str,
        test_code: Optional[str] = None,
        entry_point: Optional[str] = None,
    ) -> str:
        """Build the execution script."""
        script_parts = []
        
        # Add safety wrapper
        script_parts.append(self._get_safety_wrapper())
        
        # Add the main code
        script_parts.append("# User code")
        script_parts.append(code)
        
        # Add test code if provided (this contains the actual assertions)
        if test_code:
            script_parts.append("\n# Test code")
            script_parts.append(test_code)
        
        # Only add entry point call if no test code and entry_point specified
        # HumanEval format: test_code already calls the function, no need for entry_point call
        elif entry_point:
            script_parts.append(f"\n# Entry point call")
            script_parts.append(f"if __name__ == '__main__':")
            script_parts.append(f"    {entry_point}()")
        
        return "\n\n".join(script_parts)
    
    def _get_safety_wrapper(self) -> str:
        """Get the safety wrapper code."""
        blocked = json.dumps(self.blocked_imports)
        return f"""# Safety wrapper
import sys
import builtins
import json

# Block dangerous imports
_blocked_modules = {blocked}

_original_import = builtins.__import__

def _safe_import(name, *args, **kwargs):
    # Check if module or any parent is blocked
    parts = name.split('.')
    for i in range(len(parts)):
        module_part = '.'.join(parts[:i+1])
        if module_part in _blocked_modules:
            raise ImportError(f"Import of '{{name}}' is not allowed")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _safe_import

# Limit recursion depth
sys.setrecursionlimit(1000)
"""
    
    def _run_subprocess(self, script_path: str) -> ExecutionResult:
        """Run the script in a subprocess with resource limits."""
        start_time = time.time()
        
        # Prepare the command
        cmd = [sys.executable, script_path]
        
        # Set up environment
        env = os.environ.copy()
        env['PYTHONPATH'] = ''  # Clear PYTHONPATH
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        
        try:
            # Run with resource limits
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                preexec_fn=self._set_resource_limits,
            )
            
            # Wait with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                
                # Check return code
                success = process.returncode == 0
                
                # Combine output
                output = stdout
                error = stderr if stderr else None
                
                return ExecutionResult(
                    success=success,
                    output=output,
                    error=error,
                    return_code=process.returncode,
                    execution_time=execution_time,
                    memory_usage_mb=0.0,  # Not tracked in simple mode
                    timed_out=False,
                )
                
            except subprocess.TimeoutExpired:
                # Kill the process
                process.kill()
                process.wait()
                
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout} seconds",
                    return_code=-1,
                    execution_time=self.timeout,
                    memory_usage_mb=0.0,
                    timed_out=True,
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
                execution_time=time.time() - start_time,
                memory_usage_mb=0.0,
                timed_out=False,
            )
    
    def _set_resource_limits(self):
        """Set resource limits for the subprocess (Unix only)."""
        try:
            # Limit CPU time (soft limit)
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(self.timeout) + 1, int(self.timeout) + 5)
            )
            
            # Limit memory
            max_memory_bytes = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_AS,
                (max_memory_bytes, max_memory_bytes * 2)
            )
            
            # Limit file size
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (10 * 1024 * 1024, 20 * 1024 * 1024)  # 10MB / 20MB
            )
            
            # Limit number of processes
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (10, 20)
            )
            
        except (OSError, ValueError):
            # Resource limits not available on this platform
            pass
    
    def execute_with_check(
        self,
        code: str,
        expected_output: Optional[str] = None,
        test_code: Optional[str] = None,
    ) -> Tuple[bool, ExecutionResult]:
        """
        Execute code and check if it produces expected output.
        
        Args:
            code: Code to execute
            expected_output: Expected output string
            test_code: Test code to run
        
        Returns:
            Tuple of (passed, ExecutionResult)
        """
        result = self.execute(code, test_code)
        
        if not result.success:
            return False, result
        
        if expected_output is not None:
            passed = expected_output.strip() in result.output.strip()
            return passed, result
        
        return result.success, result
    
    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if code has valid Python syntax.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        import ast
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
    
    @contextmanager
    def temp_environment(self):
        """Context manager for temporary execution environment."""
        old_cwd = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        
        try:
            os.chdir(temp_dir)
            yield temp_dir
        finally:
            os.chdir(old_cwd)
            # Cleanup temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass
