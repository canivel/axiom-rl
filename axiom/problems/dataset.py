"""Problem dataset loader."""

import json
from pathlib import Path
from typing import Iterator, List, Optional

from .base import Problem, TestCase


class ProblemDataset:
    """Loads and manages the problem dataset."""

    def __init__(self, path: Optional[Path] = None):
        """
        Initialize the dataset.

        Args:
            path: Path to problems.json. If None, uses the default bundled problems.
        """
        self.path = path or Path(__file__).parent / "problems.json"
        self.problems: List[Problem] = []
        self._load()

    def _load(self) -> None:
        """Load problems from JSON file."""
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)

        for p in data["problems"]:
            test_cases = [
                TestCase(input=tc["input"], expected_output=tc["expected_output"])
                for tc in p["test_cases"]
            ]
            self.problems.append(
                Problem(
                    id=p["id"],
                    title=p["title"],
                    description=p["description"],
                    function_signature=p["function_signature"],
                    test_cases=test_cases,
                    difficulty=p.get("difficulty", "easy"),
                    tags=p.get("tags", []),
                )
            )

    def get_problem(self, problem_id: str) -> Optional[Problem]:
        """Get a specific problem by ID."""
        for p in self.problems:
            if p.id == problem_id:
                return p
        return None

    def get_by_difficulty(self, difficulty: str) -> List[Problem]:
        """Get all problems of a given difficulty."""
        return [p for p in self.problems if p.difficulty == difficulty]

    def __iter__(self) -> Iterator[Problem]:
        return iter(self.problems)

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, index: int) -> Problem:
        return self.problems[index]
