"""Base classes for procedural problem generation - V2 (Correct Design).

Key differences from V1:
1. Problems have MULTIPLE test cases (not just one)
2. Functions take INPUT arguments (not just def solve())
3. This forces the model to learn ALGORITHMS, not memorize answers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, List
import random


@dataclass
class TestCase:
    """A single test case for verification."""
    input_args: List[Any]  # Arguments to pass to the function
    expected_output: Any   # Expected return value

    def to_dict(self) -> dict:
        return {
            "input": self.input_args,
            "expected": self.expected_output,
        }


@dataclass
class AlgorithmicProblem:
    """A problem that requires implementing an ALGORITHM.

    Key design principles:
    1. The function takes INPUT - it's not just def solve()
    2. Multiple test cases - can't pass by hardcoding
    3. Test cases share the same function signature

    Example:
        Problem: Implement an RPN evaluator
        Function: def evaluate_rpn(expression: str) -> int
        Test cases:
            - evaluate_rpn("3 4 +") -> 7
            - evaluate_rpn("5 2 *") -> 10
            - evaluate_rpn("10 3 -") -> 7

        A hardcoded solution like `return 7` would fail 2/3 test cases.
    """

    # Problem identification
    problem_type: str      # e.g., "rpn", "arithmetic", "parentheses"
    problem_id: str        # Unique instance ID

    # Problem description
    title: str
    description: str
    function_signature: str  # e.g., "def evaluate_rpn(expression: str) -> int:"

    # Test cases - MULTIPLE required
    test_cases: List[TestCase] = field(default_factory=list)

    # Difficulty
    difficulty: int = 5    # 1-10 scale
    complexity: str = "medium"  # "easy", "medium", "hard"

    @property
    def function_name(self) -> str:
        """Extract function name from signature."""
        # "def evaluate_rpn(expression: str) -> int:" -> "evaluate_rpn"
        return self.function_signature.split("(")[0].replace("def ", "").strip()

    def to_prompt(self) -> str:
        """Generate the prompt for the model."""
        # Show 2-3 example test cases in the prompt
        examples = self.test_cases[:3]
        example_text = "\n".join([
            f"  {self.function_name}({', '.join(repr(a) for a in tc.input_args)}) -> {repr(tc.expected_output)}"
            for tc in examples
        ])

        return f"""Solve the following problem by implementing the function.

## Problem: {self.title}

{self.description}

## Function Signature
```python
{self.function_signature}
    # Your implementation here
```

## Examples
{example_text}

Write ONLY the complete function implementation that works for ALL inputs, not just the examples."""

    def to_verification_dict(self) -> dict:
        """Convert to format expected by TestHarness."""
        return {
            "id": self.problem_id,
            "title": self.title,
            "description": self.description,
            "function_signature": self.function_signature,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }


class AlgorithmicGenerator(ABC):
    """Base class for generators that create ALGORITHMIC problems.

    Key requirements:
    1. generate_test_cases() - Generate MULTIPLE test cases for one problem
    2. All test cases use the SAME function signature
    3. Test cases should require actual algorithm, not memorization
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.generated_count = 0

    @property
    @abstractmethod
    def problem_type(self) -> str:
        """Unique identifier for this problem type (e.g., 'rpn')."""
        pass

    @property
    @abstractmethod
    def title(self) -> str:
        """Human-readable title."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Full problem description."""
        pass

    @property
    @abstractmethod
    def function_signature(self) -> str:
        """The function signature (e.g., 'def evaluate(expr: str) -> int:')."""
        pass

    @abstractmethod
    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """
        Generate MULTIPLE test cases for one problem instance.

        Args:
            difficulty: 1-10 scale
            count: Number of test cases to generate (default 5)

        Returns:
            List of TestCase objects - model must pass ALL to succeed
        """
        pass

    def generate(self, difficulty: int = 5, num_test_cases: int = 5) -> AlgorithmicProblem:
        """Generate a complete algorithmic problem with multiple test cases."""
        self.generated_count += 1

        test_cases = self.generate_test_cases(difficulty, num_test_cases)

        # Determine complexity label
        if difficulty <= 3:
            complexity = "easy"
        elif difficulty <= 6:
            complexity = "medium"
        else:
            complexity = "hard"

        return AlgorithmicProblem(
            problem_type=self.problem_type,
            problem_id=f"{self.problem_type}_{self.generated_count}",
            title=self.title,
            description=self.description,
            function_signature=self.function_signature,
            test_cases=test_cases,
            difficulty=difficulty,
            complexity=complexity,
        )

    def generate_batch(
        self,
        count: int,
        min_difficulty: int = 1,
        max_difficulty: int = 10,
        test_cases_per_problem: int = 5,
    ) -> List[AlgorithmicProblem]:
        """Generate multiple problems."""
        problems = []
        for _ in range(count):
            difficulty = self.rng.randint(min_difficulty, max_difficulty)
            problems.append(self.generate(difficulty, test_cases_per_problem))
        return problems
