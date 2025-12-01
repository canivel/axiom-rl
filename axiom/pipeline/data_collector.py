"""Data collector for saving successful solutions."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..problems.base import Problem
from ..verifier.result import VerificationResult


@dataclass
class SyntheticExample:
    """A single synthetic training example."""

    problem_id: str
    problem_title: str
    problem_description: str
    function_signature: str
    solution_code: str
    passed_tests: int
    total_tests: int
    execution_time: float
    timestamp: str
    model_name: str


class DataCollector:
    """
    Collects successful solutions and saves to JSONL format.

    Output format is designed for future fine-tuning:
    - Each line is a JSON object
    - Contains problem context + verified solution
    """

    def __init__(
        self,
        output_path: Path,
        model_name: str = "unknown",
    ):
        """
        Initialize the collector.

        Args:
            output_path: Path to output JSONL file
            model_name: Name of the model that generated solutions
        """
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.examples_saved = 0

        # Ensure directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        problem: Problem,
        solution_code: str,
        result: VerificationResult,
    ) -> None:
        """
        Save a successful solution.

        Args:
            problem: The problem that was solved
            solution_code: The verified solution code
            result: The verification result
        """
        example = SyntheticExample(
            problem_id=problem.id,
            problem_title=problem.title,
            problem_description=problem.description,
            function_signature=problem.function_signature,
            solution_code=solution_code,
            passed_tests=result.passed_count,
            total_tests=result.total_count,
            execution_time=result.execution_time,
            timestamp=datetime.utcnow().isoformat(),
            model_name=self.model_name,
        )

        # Append to JSONL file
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

        self.examples_saved += 1

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "output_path": str(self.output_path),
            "examples_saved": self.examples_saved,
            "model_name": self.model_name,
        }

    def load_existing(self) -> list:
        """Load existing examples from the output file."""
        if not self.output_path.exists():
            return []

        examples = []
        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples
