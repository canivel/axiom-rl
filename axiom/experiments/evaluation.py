"""Evaluation utilities for procedural problems."""

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from axiom.procedural import ProceduralProblem, GENERATORS


@dataclass
class EvaluationResult:
    """Result of evaluating a model on a set of problems."""

    accuracy: float
    correct: int
    total: int
    results: list[dict]  # Per-problem results

    def __str__(self) -> str:
        return f"Accuracy: {self.accuracy:.1%} ({self.correct}/{self.total})"


class ProceduralEvaluator:
    """Evaluates model solutions on procedural problems."""

    def __init__(self, timeout: float = 5.0):
        """
        Initialize the evaluator.

        Args:
            timeout: Execution timeout in seconds
        """
        self.timeout = timeout
        self._generators = {name: cls() for name, cls in GENERATORS.items()}

    def extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from model response."""
        # Try to find code in markdown blocks
        patterns = [
            r"```python\n(.*?)```",
            r"```\n(.*?)```",
            r"def \w+\([^)]*\):[^\n]*\n(?:[ \t]+[^\n]*\n)*",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()

        # If no code block, try to find function definition
        if "def " in response:
            lines = response.split("\n")
            code_lines = []
            in_function = False

            for line in lines:
                if line.strip().startswith("def "):
                    in_function = True
                if in_function:
                    code_lines.append(line)
                    # End function on blank line or new def
                    if line.strip() == "" and len(code_lines) > 1:
                        break

            if code_lines:
                return "\n".join(code_lines).strip()

        return None

    def execute_solution(
        self,
        code: str,
        input_data: Any,
        problem_type: str,
    ) -> tuple[bool, Any, Optional[str]]:
        """
        Execute a solution and return result.

        Args:
            code: Python code containing the function
            input_data: Input to pass to the function
            problem_type: Type of problem (to determine function name)

        Returns:
            (success, output, error_message)
        """
        # Determine expected function name based on problem type
        function_names = {
            "arithmetic": "evaluate_expression",
            "rpn": "evaluate_rpn",
            "parentheses": "is_valid_parentheses",
            "list_sort": "custom_sort",
            "list_filter": "filter_list",
            "list_aggregate": "aggregate",
        }

        func_name = function_names.get(problem_type)
        if not func_name:
            return False, None, f"Unknown problem type: {problem_type}"

        # Create execution namespace
        namespace = {}

        try:
            # Execute the code to define the function
            exec(code, namespace)

            if func_name not in namespace:
                # Try to find any function that was defined
                funcs = [k for k, v in namespace.items() if callable(v) and not k.startswith("_")]
                if funcs:
                    func_name = funcs[0]
                else:
                    return False, None, f"Function {func_name} not found in code"

            func = namespace[func_name]

            # Call the function with appropriate arguments
            if isinstance(input_data, dict):
                # Unpack dictionary as kwargs
                output = func(**input_data)
            else:
                output = func(input_data)

            return True, output, None

        except Exception as e:
            return False, None, str(e)

    def verify_solution(
        self,
        problem: ProceduralProblem,
        model_output: Any,
    ) -> bool:
        """
        Verify if model output is correct for a problem.

        Args:
            problem: The procedural problem
            model_output: The model's output

        Returns:
            True if correct, False otherwise
        """
        if problem.problem_type not in self._generators:
            # Fall back to direct comparison
            return model_output == problem.expected_output

        gen = self._generators[problem.problem_type]
        return gen.verify(problem.input_data, model_output, problem.expected_output)

    def evaluate_response(
        self,
        problem: ProceduralProblem,
        response: str,
    ) -> dict:
        """
        Evaluate a single model response.

        Args:
            problem: The problem being solved
            response: The model's full response

        Returns:
            Evaluation result dict
        """
        result = {
            "problem_id": problem.problem_id,
            "problem_type": problem.problem_type,
            "difficulty": problem.difficulty,
            "correct": False,
            "error": None,
            "model_output": None,
            "expected_output": problem.expected_output,
        }

        # Extract code
        code = self.extract_code(response)
        if not code:
            result["error"] = "No code found in response"
            return result

        # Execute solution
        success, output, error = self.execute_solution(
            code, problem.input_data, problem.problem_type
        )

        if not success:
            result["error"] = error
            return result

        result["model_output"] = output

        # Verify
        result["correct"] = self.verify_solution(problem, output)

        return result

    def evaluate_batch(
        self,
        problems: list[ProceduralProblem],
        generate_fn: Callable[[str], str],
        verbose: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate model on a batch of problems.

        Args:
            problems: List of problems to evaluate
            generate_fn: Function that takes a prompt and returns model response
            verbose: Print progress

        Returns:
            EvaluationResult with aggregate metrics
        """
        results = []
        correct = 0

        for i, problem in enumerate(problems):
            # Always show progress for batches
            print(f"\r    Progress: {i+1}/{len(problems)} ({correct} correct so far)", end="", flush=True)

            if verbose:
                print(f"\n  Evaluating {i+1}/{len(problems)}: {problem.problem_id}...", end=" ")

            # Generate prompt and get response
            prompt = problem.to_prompt()
            response = generate_fn(prompt)

            # Evaluate
            result = self.evaluate_response(problem, response)
            results.append(result)

            if result["correct"]:
                correct += 1
                if verbose:
                    print("CORRECT")
            else:
                if verbose:
                    error = result.get("error", "Wrong answer")
                    print(f"WRONG ({error})")

        # Clear progress line
        print(f"\r    Progress: {len(problems)}/{len(problems)} - Done!                    ")

        return EvaluationResult(
            accuracy=correct / len(problems) if problems else 0.0,
            correct=correct,
            total=len(problems),
            results=results,
        )
