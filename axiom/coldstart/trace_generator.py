"""Generator for reasoning traces using a teacher model."""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..problems.base import Problem
from ..problems.dataset import ProblemDataset
from ..verifier.harness import TestHarness
from ..verifier.result import VerificationStatus
from .gemini_client import GeminiClient


@dataclass
class ReasoningTrace:
    """A single reasoning trace for training."""

    problem_id: str
    problem_title: str
    problem_description: str
    function_signature: str
    thinking: str  # The reasoning process
    solution_code: str  # The verified code
    teacher_model: str  # Model that generated this
    verified: bool  # Whether code passed tests
    passed_tests: int
    total_tests: int
    timestamp: str


class ReasoningTraceGenerator:
    """
    Generates reasoning traces for problems using a teacher model.

    This implements the "Cold Start" phase:
    1. Use a strong model (Gemini) to generate reasoning + code
    2. Verify the code actually works
    3. Save verified traces for training
    """

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        output_dir: Path = Path("data/coldstart"),
        output_file: str = "teacher_traces.jsonl",
    ):
        """
        Initialize the generator.

        Args:
            client: GeminiClient instance (creates default if None)
            output_dir: Directory to save traces
            output_file: Filename for output JSONL
        """
        self.client = client or GeminiClient()
        self.output_dir = Path(output_dir)
        self.output_file = output_file
        self.harness = TestHarness()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_for_problem(
        self,
        problem: Problem,
        num_traces: int = 5,
        delay: float = 1.0,
    ) -> list[ReasoningTrace]:
        """
        Generate multiple reasoning traces for a single problem.

        Args:
            problem: The problem to solve
            num_traces: Number of traces to generate
            delay: Delay between API calls (rate limiting)

        Returns:
            List of verified ReasoningTrace objects
        """
        traces = []

        for i in range(num_traces):
            try:
                # Generate reasoning trace
                response = self.client.generate_reasoning_trace(
                    problem_title=problem.title,
                    problem_description=problem.description,
                    function_signature=problem.function_signature,
                    temperature=0.7,  # High temperature for diverse reasoning paths
                )

                # Verify the generated code
                verification = self.harness.verify(
                    solution_code=response.code,
                    problem=problem,
                )

                is_verified = verification.status == VerificationStatus.PASSED

                trace = ReasoningTrace(
                    problem_id=problem.id,
                    problem_title=problem.title,
                    problem_description=problem.description,
                    function_signature=problem.function_signature,
                    thinking=response.thinking,
                    solution_code=response.code,
                    teacher_model=self.client.model_name,
                    verified=is_verified,
                    passed_tests=verification.passed_count,
                    total_tests=verification.total_count,
                    timestamp=datetime.now().isoformat(),
                )

                traces.append(trace)

                status = "VERIFIED" if is_verified else f"FAILED ({verification.passed_count}/{verification.total_count})"
                print(f"    Trace {i + 1}/{num_traces}: {status}")

                # Rate limiting
                if i < num_traces - 1:
                    time.sleep(delay)

            except Exception as e:
                print(f"    Trace {i + 1}/{num_traces}: ERROR - {e}")
                continue

        return traces

    def generate_dataset(
        self,
        problems: Optional[list[Problem]] = None,
        traces_per_problem: int = 10,
        delay: float = 1.0,
        only_verified: bool = True,
    ) -> list[ReasoningTrace]:
        """
        Generate a full teacher dataset.

        Args:
            problems: List of problems (uses default dataset if None)
            traces_per_problem: Number of traces per problem
            delay: Delay between API calls
            only_verified: Only save traces that pass verification

        Returns:
            All generated traces
        """
        if problems is None:
            dataset = ProblemDataset()
            problems = dataset.problems

        all_traces = []
        output_path = self.output_dir / self.output_file

        print(f"\nGenerating teacher dataset")
        print(f"Problems: {len(problems)}")
        print(f"Traces per problem: {traces_per_problem}")
        print(f"Output: {output_path}")
        print("=" * 50)

        for idx, problem in enumerate(problems):
            print(f"\n[{idx + 1}/{len(problems)}] {problem.title} ({problem.id})")

            traces = self.generate_for_problem(
                problem=problem,
                num_traces=traces_per_problem,
                delay=delay,
            )

            # Filter to verified only if requested
            if only_verified:
                traces = [t for t in traces if t.verified]

            # Save incrementally
            self._append_traces(traces, output_path)

            all_traces.extend(traces)

            verified_count = len([t for t in traces if t.verified])
            print(f"  >> Saved {verified_count} verified traces")

        print("\n" + "=" * 50)
        print(f"DATASET COMPLETE")
        print(f"Total traces: {len(all_traces)}")
        print(f"Verified traces: {len([t for t in all_traces if t.verified])}")
        print(f"Output: {output_path}")
        print("=" * 50)

        return all_traces

    def _append_traces(self, traces: list[ReasoningTrace], path: Path) -> None:
        """Append traces to JSONL file."""
        with open(path, "a", encoding="utf-8") as f:
            for trace in traces:
                f.write(json.dumps(asdict(trace)) + "\n")
