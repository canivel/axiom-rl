"""Best-of-N sampling for code generation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..problems.base import Problem
from ..verifier.harness import TestHarness
from ..verifier.result import VerificationResult
from .model import CodeGenerator
from .prompts import build_messages, extract_code_from_response


@dataclass
class SamplingResult:
    """Result of Best-of-N sampling for a problem."""

    problem_id: str
    successful_solutions: List[Tuple[str, VerificationResult]] = field(
        default_factory=list
    )  # (code, result)
    failed_solutions: List[Tuple[str, VerificationResult]] = field(
        default_factory=list
    )
    total_samples: int = 0

    @property
    def success_rate(self) -> float:
        """Fraction of samples that passed all tests."""
        if self.total_samples == 0:
            return 0.0
        return len(self.successful_solutions) / self.total_samples

    @property
    def has_solution(self) -> bool:
        """Whether at least one solution passed all tests."""
        return len(self.successful_solutions) > 0

    @property
    def best_solution(self) -> Optional[Tuple[str, VerificationResult]]:
        """Return the first successful solution, or None."""
        if self.successful_solutions:
            return self.successful_solutions[0]
        return None


class BestOfNSampler:
    """
    Best-of-N sampling strategy.

    Generates N solutions and returns all that pass verification.
    """

    def __init__(
        self,
        generator: CodeGenerator,
        harness: TestHarness,
        num_samples: int = 8,
        verbose: bool = True,
    ):
        """
        Initialize the sampler.

        Args:
            generator: Code generator model
            harness: Test harness for verification
            num_samples: Number of samples to generate per attempt
            verbose: Whether to print progress
        """
        self.generator = generator
        self.harness = harness
        self.num_samples = num_samples
        self.verbose = verbose

    def sample(self, problem: Problem) -> SamplingResult:
        """
        Generate N solutions and verify each.

        Args:
            problem: The problem to solve

        Returns:
            SamplingResult with successful and failed solutions
        """
        messages = build_messages(problem)

        # Generate N responses
        if self.verbose:
            print(f"  Generating {self.num_samples} samples...")

        responses = self.generator.generate(messages, num_samples=self.num_samples)

        result = SamplingResult(problem_id=problem.id, total_samples=len(responses))

        # Verify each response
        for i, response in enumerate(responses):
            # Extract code from response
            code = extract_code_from_response(response)

            # Verify against test cases
            verification = self.harness.verify(code, problem)

            if verification.passed:
                result.successful_solutions.append((code, verification))
                if self.verbose:
                    print(f"    Sample {i + 1}: PASSED ({verification.pass_rate:.0%})")
            else:
                result.failed_solutions.append((code, verification))
                if self.verbose:
                    status = verification.status.value
                    print(
                        f"    Sample {i + 1}: {status.upper()} ({verification.pass_rate:.0%})"
                    )

        return result
