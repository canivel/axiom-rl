"""Main pipeline orchestrator."""

from pathlib import Path
from typing import List, Optional

from ..config import PipelineConfig
from ..generator.model import CodeGenerator
from ..generator.sampler import BestOfNSampler
from ..problems.base import Problem
from ..problems.dataset import ProblemDataset
from ..verifier.harness import TestHarness
from ..verifier.sandbox import PythonSandbox
from .data_collector import DataCollector


class Pipeline:
    """
    Main orchestrator for the axiom-rl MVP.

    Flow:
    1. Load problems
    2. For each problem:
       a. Generate N solutions
       b. Verify each solution
       c. Save successful solutions
    3. Report statistics
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize components
        self.dataset = ProblemDataset(config.problems_path)

        self.sandbox = PythonSandbox(
            timeout=config.verifier.timeout,
            python_executable=config.verifier.python_executable,
        )
        self.harness = TestHarness(self.sandbox)

        self.generator = CodeGenerator(config.generator)
        self.sampler = BestOfNSampler(
            self.generator,
            self.harness,
            num_samples=config.generator.num_samples,
        )

        output_path = config.output_dir / config.output_file
        self.collector = DataCollector(
            output_path,
            model_name=config.generator.model_name,
        )

        # Statistics
        self.stats = {
            "problems_attempted": 0,
            "problems_solved": 0,
            "total_solutions": 0,
            "total_samples": 0,
        }

    def run(
        self,
        problem_ids: Optional[List[str]] = None,
        skip_existing: bool = False,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            problem_ids: Optional list of specific problem IDs to run.
                        If None, runs all problems.
            skip_existing: Skip problems that already have solutions in output

        Returns:
            Statistics dictionary
        """
        # Load model
        print("=" * 50)
        print("AXIOM-RL MVP Pipeline")
        print("=" * 50)
        print(f"\nModel: {self.config.generator.model_name}")
        print(f"Samples per problem: {self.config.generator.num_samples}")
        print(f"Max attempts: {self.config.max_attempts_per_problem}")
        print(f"Output: {self.collector.output_path}")
        print()

        print("Loading generator model...")
        self.generator.load()
        print()

        # Determine which problems to run
        if problem_ids:
            problems = [p for p in self.dataset if p.id in problem_ids]
        else:
            problems = list(self.dataset)

        # Skip existing if requested
        if skip_existing:
            existing = self.collector.load_existing()
            existing_ids = {e["problem_id"] for e in existing}
            problems = [p for p in problems if p.id not in existing_ids]
            if existing_ids:
                print(f"Skipping {len(existing_ids)} already-solved problems")

        print(f"Running pipeline on {len(problems)} problems\n")
        print("-" * 50)

        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] {problem.title} ({problem.id})")
            self._process_problem(problem)

        # Final report
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE")
        print("=" * 50)
        print(f"Problems attempted: {self.stats['problems_attempted']}")
        print(f"Problems solved: {self.stats['problems_solved']}")
        print(
            f"Solve rate: {self.stats['problems_solved']/max(1,self.stats['problems_attempted']):.1%}"
        )
        print(f"Total solutions collected: {self.stats['total_solutions']}")
        print(f"Total samples generated: {self.stats['total_samples']}")
        print(f"Output saved to: {self.collector.output_path}")

        return self.stats

    def _process_problem(self, problem: Problem) -> bool:
        """
        Process a single problem.

        Args:
            problem: The problem to solve

        Returns:
            True if at least one solution was found
        """
        self.stats["problems_attempted"] += 1
        solved = False

        # Try multiple attempts if configured
        for attempt in range(self.config.max_attempts_per_problem):
            if attempt > 0:
                print(f"  Retry attempt {attempt + 1}...")

            result = self.sampler.sample(problem)
            self.stats["total_samples"] += result.total_samples

            if result.has_solution:
                # Deduplicate solutions by code content
                seen_codes = set()
                unique_solutions = []
                for code, verification in result.successful_solutions:
                    # Normalize whitespace for comparison
                    normalized = code.strip()
                    if normalized not in seen_codes:
                        seen_codes.add(normalized)
                        unique_solutions.append((code, verification))

                # Save unique solutions
                for code, verification in unique_solutions:
                    self.collector.save(problem, code, verification)
                    self.stats["total_solutions"] += 1

                self.stats["problems_solved"] += 1
                total = len(result.successful_solutions)
                unique = len(unique_solutions)
                if total == unique:
                    print(f"  >> SUCCESS: {unique} solution(s) saved")
                else:
                    print(f"  >> SUCCESS: {unique} unique solution(s) saved ({total - unique} duplicates filtered)")
                solved = True
                break
            else:
                print(f"  -- No passing solutions in this attempt")

        if not solved:
            print(
                f"  >> FAILED: Could not solve after {self.config.max_attempts_per_problem} attempt(s)"
            )

        return solved

    def run_single(self, problem_id: str) -> dict:
        """
        Run pipeline on a single problem (for testing).

        Args:
            problem_id: ID of the problem to solve

        Returns:
            Statistics dictionary
        """
        return self.run(problem_ids=[problem_id])
