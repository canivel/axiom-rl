"""Sandboxed Python execution using subprocess."""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of executing code in sandbox."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


class PythonSandbox:
    """
    Simple sandboxed Python execution using subprocess.

    Security measures for MVP:
    - Subprocess isolation (separate process)
    - Timeout enforcement
    - No shell=True
    - Temporary file cleanup
    - Restricted environment variables

    NOTE: This is NOT secure against malicious code - MVP only!
    For production, use Docker or proper sandboxing.
    """

    def __init__(
        self,
        timeout: float = 5.0,
        python_executable: Optional[str] = None,
    ):
        """
        Initialize the sandbox.

        Args:
            timeout: Maximum execution time in seconds
            python_executable: Path to Python interpreter (default: "python")
        """
        self.timeout = timeout
        self.python_executable = python_executable or "python"

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code in a subprocess.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with stdout, stderr, returncode, timeout status
        """
        # Create temporary file for code
        fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="axiom_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)

            # Build restricted environment
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "PYTHONHASHSEED": "0",  # Deterministic hashing
            }
            # Add Windows-specific env vars if needed
            if os.name == "nt":
                env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
                env["TEMP"] = tempfile.gettempdir()
                env["TMP"] = tempfile.gettempdir()

            result = subprocess.run(
                [self.python_executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=False,
                env=env,
                cwd=tempfile.gettempdir(),
            )
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout}s",
                returncode=-1,
                timed_out=True,
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution error: {str(e)}",
                returncode=-1,
                timed_out=False,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
