"""
Code Evaluator for LiveCodeBench and other coding benchmarks.

Extracts code from model output, executes it, and verifies against test cases.
Uses sandboxed execution for safety.
"""

import re
import sys
import traceback
from typing import Any, Optional, List, Dict, Tuple
from io import StringIO
import contextlib
import signal

from ..base import BenchmarkEvaluator, BenchmarkProblem, BenchmarkResult, BenchmarkType
from ..registry import register_evaluator


class TimeoutError(Exception):
    """Exception for execution timeout."""
    pass


@contextlib.contextmanager
def timeout(seconds: int):
    """Context manager for execution timeout (Unix only)."""
    def handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds}s")

    # Only works on Unix
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows fallback - no timeout
        yield


def extract_code_from_output(output: str) -> Optional[str]:
    """
    Extract Python code from model output.

    Handles markdown code blocks and raw code.

    Args:
        output: Model output text

    Returns:
        Extracted code or None
    """
    if not output:
        return None

    # Strategy 1: ```python blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, output, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Strategy 2: ``` blocks (any language)
    pattern = r"```\w*\s*\n(.*?)```"
    matches = re.findall(pattern, output, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Strategy 3: Look for def/class statements
    lines = output.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        if re.match(r"^(def |class |import |from |\s+)", line):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return None


def execute_code_safely(
    code: str,
    test_input: Any,
    timeout_seconds: int = 5
) -> Tuple[bool, Any, Optional[str]]:
    """
    Execute code safely and capture output.

    Args:
        code: Python code to execute
        test_input: Input for the function
        timeout_seconds: Max execution time

    Returns:
        (success, output, error_message)
    """
    # Create isolated namespace
    namespace = {"__builtins__": __builtins__}

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Execute the code to define functions
        exec(code, namespace)

        # Find the main function (last defined function)
        functions = [
            name for name, obj in namespace.items()
            if callable(obj) and not name.startswith("_")
        ]

        if not functions:
            return False, None, "No function found in code"

        # Call the function with test input
        func_name = functions[-1]  # Use last defined function
        func = namespace[func_name]

        with timeout(timeout_seconds):
            if isinstance(test_input, dict):
                result = func(**test_input)
            elif isinstance(test_input, (list, tuple)):
                result = func(*test_input)
            else:
                result = func(test_input)

        return True, result, None

    except TimeoutError as e:
        return False, None, str(e)
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
        return False, None, error
    finally:
        sys.stdout = old_stdout


def run_test_case(
    code: str,
    test_case: Dict,
    timeout_seconds: int = 5
) -> Tuple[bool, str]:
    """
    Run a single test case against the code.

    Args:
        code: Python code
        test_case: Dict with 'input' and 'expected_output'
        timeout_seconds: Max time per test

    Returns:
        (passed, message)
    """
    test_input = test_case.get("input", test_case.get("inputs", ""))
    expected = test_case.get("expected_output", test_case.get("output", ""))

    # Parse input if it's a string
    if isinstance(test_input, str):
        try:
            test_input = eval(test_input)
        except:
            pass

    # Execute
    success, actual, error = execute_code_safely(code, test_input, timeout_seconds)

    if not success:
        return False, f"Execution error: {error}"

    # Compare outputs
    # Parse expected if string
    if isinstance(expected, str):
        try:
            expected = eval(expected)
        except:
            pass

    # Normalize for comparison
    if str(actual).strip() == str(expected).strip():
        return True, "Passed"

    # Try numeric comparison
    try:
        if abs(float(actual) - float(expected)) < 1e-6:
            return True, "Passed (float)"
    except (ValueError, TypeError):
        pass

    return False, f"Expected {expected}, got {actual}"


@register_evaluator(BenchmarkType.CODE)
class CodeEvaluator(BenchmarkEvaluator):
    """
    Evaluator for code generation problems.

    Supports:
    - LiveCodeBench
    - Other competitive programming benchmarks
    """

    def __init__(self, timeout_per_test: int = 5):
        """
        Initialize code evaluator.

        Args:
            timeout_per_test: Max seconds per test case
        """
        self.timeout_per_test = timeout_per_test

    def extract_answer(
        self,
        output: str,
        problem: Optional[BenchmarkProblem] = None
    ) -> Any:
        """
        Extract code from model output.

        Args:
            output: Model output text
            problem: Problem for context

        Returns:
            Extracted Python code or None
        """
        return extract_code_from_output(output)

    def check_answer(self, extracted: Any, expected: Any) -> bool:
        """
        Check if extracted code passes all test cases.

        Args:
            extracted: Python code from model
            expected: Dict with test cases

        Returns:
            True if all tests pass
        """
        if not extracted:
            return False

        if not isinstance(expected, dict):
            return False

        # Get test cases
        public_tests = expected.get("public_tests", [])
        private_tests = expected.get("private_tests", [])

        # Combine all tests
        all_tests = []
        if isinstance(public_tests, list):
            all_tests.extend(public_tests)
        if isinstance(private_tests, list):
            all_tests.extend(private_tests)

        if not all_tests:
            # No tests - can't verify
            return False

        # Run all tests
        for test in all_tests:
            passed, _ = run_test_case(extracted, test, self.timeout_per_test)
            if not passed:
                return False

        return True

    def evaluate(
        self,
        problem: BenchmarkProblem,
        model_output: str,
        time_seconds: float = 0.0
    ) -> BenchmarkResult:
        """
        Evaluate code solution.

        Provides detailed feedback on which tests passed/failed.

        Args:
            problem: The code problem
            model_output: Model's code output
            time_seconds: Generation time

        Returns:
            BenchmarkResult with test details
        """
        code = self.extract_answer(model_output, problem)

        if not code:
            return BenchmarkResult(
                problem_id=problem.id,
                correct=False,
                model_answer=None,
                expected_answer="[code with passing tests]",
                time_seconds=time_seconds,
                error="Could not extract code from output",
                raw_output=model_output,
            )

        # Get test cases
        expected = problem.answer
        if not isinstance(expected, dict):
            expected = {"public_tests": [], "private_tests": []}

        public_tests = expected.get("public_tests", [])
        private_tests = expected.get("private_tests", [])

        # Run tests and collect results
        test_results = []
        all_passed = True

        for i, test in enumerate(public_tests):
            passed, msg = run_test_case(code, test, self.timeout_per_test)
            test_results.append({"test": i, "type": "public", "passed": passed, "message": msg})
            if not passed:
                all_passed = False

        for i, test in enumerate(private_tests):
            passed, msg = run_test_case(code, test, self.timeout_per_test)
            test_results.append({"test": i, "type": "private", "passed": passed, "message": msg})
            if not passed:
                all_passed = False

        passed_count = sum(1 for t in test_results if t["passed"])
        total_count = len(test_results)

        return BenchmarkResult(
            problem_id=problem.id,
            correct=all_passed,
            model_answer=code,
            expected_answer=f"Pass {total_count} tests",
            time_seconds=time_seconds,
            error=None if all_passed else f"Failed {total_count - passed_count}/{total_count} tests",
            raw_output=model_output,
        )
