"""Base dataclasses for problems and test cases."""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class TestCase:
    """A single test case with input and expected output."""

    input: Any  # Can be a single value or list of args
    expected_output: Any


@dataclass
class Problem:
    """An algorithmic problem with test cases."""

    id: str
    title: str
    description: str
    function_signature: str  # e.g., "def two_sum(nums: List[int], target: int) -> List[int]:"
    test_cases: List[TestCase]
    difficulty: str = "easy"
    tags: List[str] = field(default_factory=list)

    @property
    def function_name(self) -> str:
        """Extract function name from signature."""
        # "def two_sum(nums: List[int], target: int) -> List[int]:" -> "two_sum"
        return self.function_signature.split("(")[0].replace("def ", "").strip()
