"""Verification result dataclasses."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class VerificationStatus(Enum):
    """Status of a verification attempt."""

    PASSED = "passed"  # All tests passed
    FAILED = "failed"  # Some tests failed
    ERROR = "error"  # Syntax error, runtime error, etc.
    TIMEOUT = "timeout"  # Execution timed out


@dataclass
class TestResult:
    """Result of running a single test case."""

    test_index: int
    input: Any
    expected: Any
    actual: Optional[Any]
    passed: bool
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """Complete result of verifying a solution."""

    status: VerificationStatus
    passed_count: int
    total_count: int
    test_results: List[TestResult] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time: float = 0.0

    @property
    def passed(self) -> bool:
        """Whether all tests passed."""
        return self.status == VerificationStatus.PASSED

    @property
    def pass_rate(self) -> float:
        """Fraction of tests that passed."""
        return self.passed_count / self.total_count if self.total_count > 0 else 0.0

    def __str__(self) -> str:
        return f"VerificationResult({self.status.value}, {self.passed_count}/{self.total_count})"
