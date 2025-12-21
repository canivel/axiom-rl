"""Test harness for running code against test cases."""

import json
import time
from typing import Optional

from ..problems.base import Problem
from .result import TestResult, VerificationResult, VerificationStatus
from .sandbox import ExecutionResult, PythonSandbox


class TestHarness:
    """
    Runs generated code against test cases.

    Strategy:
    1. Combine solution code with test runner code
    2. Execute in sandbox
    3. Parse JSON results from stdout
    """

    def __init__(self, sandbox: Optional[PythonSandbox] = None):
        """
        Initialize the harness.

        Args:
            sandbox: PythonSandbox instance. If None, creates default.
        """
        self.sandbox = sandbox or PythonSandbox()

    def verify(self, solution_code: str, problem: Problem) -> VerificationResult:
        """
        Verify a solution against all test cases.

        Args:
            solution_code: The generated solution code (should define the function)
            problem: The problem with test cases

        Returns:
            VerificationResult with pass/fail status
        """
        start_time = time.time()

        # Build the full test script
        test_script = self._build_test_script(solution_code, problem)

        # Execute
        exec_result = self.sandbox.execute(test_script)
        execution_time = time.time() - start_time

        # Parse and return result
        return self._parse_result(exec_result, problem, execution_time)

    def _build_test_script(self, solution_code: str, problem: Problem) -> str:
        """Build a complete test script that outputs JSON results."""
        # Serialize test cases
        test_cases_json = json.dumps(
            [
                {
                    "input": getattr(tc, 'input', None) or getattr(tc, 'input_args', None),
                    "expected": tc.expected_output
                }
                for tc in problem.test_cases
            ]
        )

        func_name = problem.function_name

        # Build the test runner script
        # Note: We pass test_cases as a JSON string and parse it at runtime
        # to ensure proper boolean conversion (JSON true/false -> Python True/False)
        test_script = f'''# -*- coding: utf-8 -*-
import json
import sys
from typing import List, Optional, Tuple, Dict, Any, Set

# === SOLUTION CODE ===
{solution_code}
# === END SOLUTION ===

def run_tests():
    test_cases = json.loads('{test_cases_json}')
    results = []

    for i, tc in enumerate(test_cases):
        inp = tc["input"]
        expected = tc["expected"]

        try:
            # Call function with unpacked arguments
            if isinstance(inp, list):
                actual = {func_name}(*inp)
            else:
                actual = {func_name}(inp)

            # Compare results
            passed = actual == expected

            results.append({{
                "index": i,
                "input": inp,
                "expected": expected,
                "actual": actual,
                "passed": passed
            }})
        except Exception as e:
            results.append({{
                "index": i,
                "input": inp,
                "expected": expected,
                "actual": None,
                "passed": False,
                "error": str(e)
            }})

    print(json.dumps({{"results": results, "success": True}}))

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(json.dumps({{"results": [], "success": False, "error": str(e)}}))
'''
        return test_script

    def _parse_result(
        self, exec_result: ExecutionResult, problem: Problem, execution_time: float
    ) -> VerificationResult:
        """Parse execution result into VerificationResult."""
        total_count = len(problem.test_cases)

        # Handle timeout
        if exec_result.timed_out:
            return VerificationResult(
                status=VerificationStatus.TIMEOUT,
                passed_count=0,
                total_count=total_count,
                test_results=[],
                error_message=exec_result.stderr,
                execution_time=execution_time,
            )

        # Handle execution errors (non-zero return code with no valid output)
        if exec_result.returncode != 0 and not exec_result.stdout.strip():
            return VerificationResult(
                status=VerificationStatus.ERROR,
                passed_count=0,
                total_count=total_count,
                test_results=[],
                error_message=exec_result.stderr or "Unknown execution error",
                execution_time=execution_time,
            )

        # Try to parse JSON output
        try:
            data = json.loads(exec_result.stdout)

            if not data.get("success", False) and "error" in data:
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    passed_count=0,
                    total_count=total_count,
                    test_results=[],
                    error_message=data["error"],
                    execution_time=execution_time,
                )

            test_results = [
                TestResult(
                    test_index=r["index"],
                    input=r["input"],
                    expected=r["expected"],
                    actual=r.get("actual"),
                    passed=r["passed"],
                    error=r.get("error"),
                )
                for r in data["results"]
            ]

            passed_count = sum(1 for tr in test_results if tr.passed)

            status = (
                VerificationStatus.PASSED
                if passed_count == total_count
                else VerificationStatus.FAILED
            )

            return VerificationResult(
                status=status,
                passed_count=passed_count,
                total_count=total_count,
                test_results=test_results,
                execution_time=execution_time,
            )

        except json.JSONDecodeError as e:
            # Could be a syntax error or other issue that prevented JSON output
            error_msg = exec_result.stderr or exec_result.stdout or str(e)
            return VerificationResult(
                status=VerificationStatus.ERROR,
                passed_count=0,
                total_count=total_count,
                test_results=[],
                error_message=f"Failed to parse output: {error_msg}",
                execution_time=execution_time,
            )
