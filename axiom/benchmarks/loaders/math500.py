"""
MATH500 Benchmark Loader.

MATH500 is a subset of 500 problems from the MATH dataset (Hendrycks et al., 2021).
Competition-level math problems across 7 subjects at 5 difficulty levels.

Subjects: Algebra, Counting & Probability, Geometry, Intermediate Algebra,
          Number Theory, Prealgebra, Precalculus

Levels: 1 (easiest) to 5 (hardest, Olympiad-level)

Dataset: hendrycks/competition_math or lighteval/MATH
"""

from typing import List
import random

from ..base import BenchmarkLoader, BenchmarkProblem, BenchmarkType
from ..registry import register_loader


@register_loader("math500")
class MATH500Loader(BenchmarkLoader):
    """
    Loader for MATH500 benchmark.

    500 competition math problems from the MATH dataset.
    Answers are in LaTeX \\boxed{} format.
    """

    @property
    def name(self) -> str:
        return "math500"

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.MATH

    @property
    def description(self) -> str:
        return (
            "500 competition math problems from MATH dataset. "
            "Levels 1-5 across 7 subjects (algebra, geometry, number theory, etc.)"
        )

    def load(self, split: str = "test", seed: int = 42) -> List[BenchmarkProblem]:
        """
        Load MATH500 problems.

        Args:
            split: Dataset split (default: "test")
            seed: Random seed for sampling 500 problems

        Returns:
            List of 500 BenchmarkProblem instances
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required. Install with: pip install datasets"
            )

        # Try different dataset sources
        dataset = None
        sources = [
            ("lighteval/MATH", None),
            ("hendrycks/competition_math", None),
            ("math_dataset", "default"),
        ]

        for source, config in sources:
            try:
                if config:
                    dataset = load_dataset(source, config, split=split)
                else:
                    dataset = load_dataset(source, split=split)
                break
            except Exception:
                continue

        if dataset is None:
            raise RuntimeError(
                "Could not load MATH dataset. Tried: lighteval/MATH, "
                "hendrycks/competition_math. Please ensure dataset is available."
            )

        # Sample 500 if more exist
        all_problems = list(dataset)
        if len(all_problems) > 500:
            rng = random.Random(seed)
            all_problems = rng.sample(all_problems, 500)

        problems = []
        for i, item in enumerate(all_problems):
            # Extract fields (handle different dataset formats)
            question = item.get("problem", item.get("question", ""))
            solution = item.get("solution", item.get("answer", ""))
            level = item.get("level", item.get("difficulty", "unknown"))
            subject = item.get("type", item.get("subject", "unknown"))

            # Clean level format if needed
            if isinstance(level, str) and "Level" in level:
                level = level.replace("Level ", "")

            problems.append(BenchmarkProblem(
                id=f"math500_{i}",
                question=question,
                answer=solution,
                metadata={
                    "level": level,
                    "subject": subject,
                    "source": "MATH",
                }
            ))

        return problems
