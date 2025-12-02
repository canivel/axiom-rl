"""Integration between procedural generators and the training pipeline.

This module bridges procedural problem generation with the existing
trainer infrastructure, allowing infinite training data generation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .base import ProceduralGenerator, ProceduralProblem
from . import get_all_generators, GENERATORS


@dataclass
class ProceduralTrainingSample:
    """A training sample derived from a procedural problem.

    This format is compatible with the existing trainer's TrainingSample.
    """

    problem_id: str
    problem_title: str
    problem_description: str
    function_signature: str
    input_data: str  # String representation of input
    expected_output: str  # String representation of expected output
    difficulty: int
    problem_type: str

    def to_prompt(self) -> str:
        """Generate the prompt for the model."""
        return f"""Solve the following problem by implementing the function.

## Problem: {self.problem_title}

{self.problem_description}

## Function Signature
```python
{self.function_signature}
    # Your implementation here
```

## Example
Input: {self.input_data}
Expected Output: {self.expected_output}

Write ONLY the complete function implementation."""

    def to_jsonl_entry(self) -> dict:
        """Convert to JSONL-compatible dictionary."""
        return {
            "problem_id": self.problem_id,
            "problem_title": self.problem_title,
            "problem_description": self.problem_description,
            "function_signature": self.function_signature,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "difficulty": self.difficulty,
            "problem_type": self.problem_type,
        }


def procedural_to_training_sample(problem: ProceduralProblem) -> ProceduralTrainingSample:
    """Convert a ProceduralProblem to a training-compatible sample."""
    return ProceduralTrainingSample(
        problem_id=problem.problem_id,
        problem_title=problem.title,
        problem_description=problem.description,
        function_signature=problem.function_signature,
        input_data=repr(problem.input_data),
        expected_output=repr(problem.expected_output),
        difficulty=problem.difficulty,
        problem_type=problem.problem_type,
    )


class ProceduralDataStream:
    """Infinite stream of procedural training data.

    This can be used as a data source for online training,
    generating fresh problems on-the-fly.
    """

    def __init__(
        self,
        problem_types: list[str] | None = None,
        min_difficulty: int = 1,
        max_difficulty: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Initialize the data stream.

        Args:
            problem_types: Types to include (None = all)
            min_difficulty: Minimum difficulty (1-10)
            max_difficulty: Maximum difficulty (1-10)
            seed: Random seed for reproducibility
        """
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty

        # Initialize generators
        if problem_types:
            self.generators = {
                t: GENERATORS[t](seed=seed) for t in problem_types
            }
        else:
            self.generators = get_all_generators(seed=seed)

        self._gen_names = list(self.generators.keys())
        self._current_gen_idx = 0

    def __iter__(self) -> Iterator[ProceduralTrainingSample]:
        """Iterate over infinite training samples."""
        return self

    def __next__(self) -> ProceduralTrainingSample:
        """Get next training sample."""
        # Round-robin through generators
        gen_name = self._gen_names[self._current_gen_idx]
        gen = self.generators[gen_name]
        self._current_gen_idx = (self._current_gen_idx + 1) % len(self._gen_names)

        # Generate problem
        problem = gen.generate(
            difficulty=gen.rng.randint(self.min_difficulty, self.max_difficulty)
        )

        return procedural_to_training_sample(problem)

    def generate_batch(self, size: int) -> list[ProceduralTrainingSample]:
        """Generate a batch of training samples."""
        return [next(self) for _ in range(size)]


def generate_training_file(
    output_path: Path,
    count: int,
    problem_types: list[str] | None = None,
    min_difficulty: int = 1,
    max_difficulty: int = 10,
    seed: Optional[int] = None,
) -> int:
    """
    Generate a JSONL file of procedural training problems.

    Args:
        output_path: Path to output JSONL file
        count: Number of problems to generate
        problem_types: Types to include (None = all)
        min_difficulty: Minimum difficulty
        max_difficulty: Maximum difficulty
        seed: Random seed

    Returns:
        Number of problems generated
    """
    stream = ProceduralDataStream(
        problem_types=problem_types,
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty,
        seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(count):
            sample = next(stream)
            f.write(json.dumps(sample.to_jsonl_entry()) + "\n")

    return count


class ProceduralVerifier:
    """Verifier for procedural problem solutions.

    Unlike the main TestHarness which runs test cases,
    this verifier knows the exact expected output and can
    verify directly without code execution.
    """

    def __init__(self):
        self.generators = get_all_generators()

    def verify(
        self,
        problem_type: str,
        input_data: any,
        model_output: any,
        expected_output: any,
    ) -> bool:
        """
        Verify a model's output for a procedural problem.

        Args:
            problem_type: The type of problem
            input_data: The problem input
            model_output: The model's output
            expected_output: The expected output

        Returns:
            True if correct, False otherwise
        """
        if problem_type not in self.generators:
            raise ValueError(f"Unknown problem type: {problem_type}")

        gen = self.generators[problem_type]
        return gen.verify(input_data, model_output, expected_output)
