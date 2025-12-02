"""Base classes for procedural problem generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional
import random


@dataclass
class ProceduralProblem:
    """A single procedurally generated problem instance.

    Unlike hand-crafted problems with multiple test cases,
    procedural problems are single input → output pairs
    that can be generated infinitely.
    """

    # Problem type identifier (e.g., "arithmetic", "rpn", "parentheses")
    problem_type: str

    # The specific problem instance
    problem_id: str  # Unique ID for this instance
    title: str  # Human-readable title
    description: str  # Full problem description
    function_signature: str  # The function to implement

    # The specific instance data
    input_data: Any  # The input to the function
    expected_output: Any  # The correct output

    # Difficulty metrics
    difficulty: int  # 1-10 scale
    complexity: str  # "easy", "medium", "hard"

    def to_test_case(self) -> dict:
        """Convert to test case format for verification."""
        return {
            "input": self.input_data,
            "expected_output": self.expected_output,
        }

    def to_prompt(self) -> str:
        """Generate the prompt for the model."""
        return f"""Solve the following problem by implementing the function.

## Problem: {self.title}

{self.description}

## Function Signature
```python
{self.function_signature}
    # Your implementation here
```

## Example
Input: {repr(self.input_data)}
Expected Output: {repr(self.expected_output)}

Write ONLY the complete function implementation."""


class ProceduralGenerator(ABC):
    """Base class for procedural problem generators.

    Subclasses implement specific problem types that can
    generate infinite unique instances with perfect ground truth.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility (None = random)
        """
        self.rng = random.Random(seed)
        self.generated_count = 0

    @property
    @abstractmethod
    def problem_type(self) -> str:
        """Unique identifier for this problem type."""
        pass

    @property
    @abstractmethod
    def title(self) -> str:
        """Human-readable title for this problem type."""
        pass

    @property
    @abstractmethod
    def description_template(self) -> str:
        """Template for problem description."""
        pass

    @property
    @abstractmethod
    def function_signature(self) -> str:
        """The function signature to implement."""
        pass

    @abstractmethod
    def generate_instance(self, difficulty: int = 5) -> tuple[Any, Any]:
        """
        Generate a single problem instance.

        Args:
            difficulty: 1-10 scale (1=easiest, 10=hardest)

        Returns:
            (input_data, expected_output)
        """
        pass

    @abstractmethod
    def verify(self, input_data: Any, output: Any, expected: Any) -> bool:
        """
        Verify if the output is correct.

        Args:
            input_data: The problem input
            output: The model's output
            expected: The expected output

        Returns:
            True if correct, False otherwise
        """
        pass

    def generate(self, difficulty: int = 5) -> ProceduralProblem:
        """
        Generate a complete problem instance.

        Args:
            difficulty: 1-10 scale

        Returns:
            ProceduralProblem instance
        """
        input_data, expected_output = self.generate_instance(difficulty)
        self.generated_count += 1

        # Determine complexity label
        if difficulty <= 3:
            complexity = "easy"
        elif difficulty <= 6:
            complexity = "medium"
        else:
            complexity = "hard"

        return ProceduralProblem(
            problem_type=self.problem_type,
            problem_id=f"{self.problem_type}_{self.generated_count}",
            title=self.title,
            description=self.description_template,
            function_signature=self.function_signature,
            input_data=input_data,
            expected_output=expected_output,
            difficulty=difficulty,
            complexity=complexity,
        )

    def generate_batch(
        self,
        count: int,
        min_difficulty: int = 1,
        max_difficulty: int = 10,
    ) -> list[ProceduralProblem]:
        """
        Generate a batch of problems with varying difficulty.

        Args:
            count: Number of problems to generate
            min_difficulty: Minimum difficulty level
            max_difficulty: Maximum difficulty level

        Returns:
            List of ProceduralProblem instances
        """
        problems = []
        for _ in range(count):
            difficulty = self.rng.randint(min_difficulty, max_difficulty)
            problems.append(self.generate(difficulty))
        return problems

    def generate_infinite(
        self,
        min_difficulty: int = 1,
        max_difficulty: int = 10,
    ) -> Iterator[ProceduralProblem]:
        """
        Generate an infinite stream of problems.

        Args:
            min_difficulty: Minimum difficulty level
            max_difficulty: Maximum difficulty level

        Yields:
            ProceduralProblem instances forever
        """
        while True:
            difficulty = self.rng.randint(min_difficulty, max_difficulty)
            yield self.generate(difficulty)
