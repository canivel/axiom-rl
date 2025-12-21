"""
LiveCodeBench Loader.

LiveCodeBench is a benchmark of real-world competitive programming problems.
Problems are sourced from actual programming contests (LeetCode, Codeforces, etc.)
and include both public and hidden test cases.

Evaluation requires code execution and passing all test cases.

Dataset: livecodebench/code_generation
"""

from typing import List

from ..base import BenchmarkLoader, BenchmarkProblem, BenchmarkType
from ..registry import register_loader


@register_loader("livecodebench")
class LiveCodeBenchLoader(BenchmarkLoader):
    """
    Loader for LiveCodeBench.

    Competitive programming problems with test cases.
    """

    @property
    def name(self) -> str:
        return "livecodebench"

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.CODE

    @property
    def description(self) -> str:
        return (
            "Competitive programming problems from real contests. "
            "Code generation with test case verification."
        )

    def load(self, split: str = "test") -> List[BenchmarkProblem]:
        """
        Load LiveCodeBench problems.

        Args:
            split: Dataset split (default: "test")

        Returns:
            List of BenchmarkProblem instances with code problems
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required. Install with: pip install datasets"
            )

        # Try to load the dataset
        dataset = None
        sources = [
            ("livecodebench/code_generation_lite", split),
            ("livecodebench/code_generation", split),
            ("bigcode/livecodebench", split),
        ]

        for source, sp in sources:
            try:
                dataset = load_dataset(source, split=sp)
                break
            except Exception:
                continue

        if dataset is None:
            raise RuntimeError(
                "Could not load LiveCodeBench dataset. "
                "Please ensure livecodebench/code_generation is available."
            )

        problems = []
        for i, item in enumerate(dataset):
            # Extract problem description
            question_id = item.get("question_id", item.get("id", f"lcb_{i}"))
            question = item.get("question_content", item.get("prompt", ""))

            # Extract test cases
            public_tests = item.get("public_test_cases", item.get("test_cases", []))
            private_tests = item.get("private_test_cases", [])

            # Parse test cases if they're strings
            if isinstance(public_tests, str):
                import json
                try:
                    public_tests = json.loads(public_tests)
                except json.JSONDecodeError:
                    public_tests = []

            if isinstance(private_tests, str):
                import json
                try:
                    private_tests = json.loads(private_tests)
                except json.JSONDecodeError:
                    private_tests = []

            # Extract other metadata
            difficulty = item.get("difficulty", item.get("contest_level", "unknown"))
            platform = item.get("platform", item.get("source", "unknown"))
            starter_code = item.get("starter_code", item.get("signature", ""))

            # Build full problem with starter code if available
            full_question = question
            if starter_code:
                full_question += f"\n\n## Starter Code\n```python\n{starter_code}\n```"

            problems.append(BenchmarkProblem(
                id=str(question_id),
                question=full_question,
                answer={
                    "public_tests": public_tests,
                    "private_tests": private_tests,
                    "starter_code": starter_code,
                },
                metadata={
                    "difficulty": difficulty,
                    "platform": platform,
                    "num_public_tests": len(public_tests) if isinstance(public_tests, list) else 0,
                    "num_private_tests": len(private_tests) if isinstance(private_tests, list) else 0,
                    "source": "LiveCodeBench",
                }
            ))

        return problems
