"""
AIME Benchmark Loaders (2024 and 2025).

AIME (American Invitational Mathematics Examination) is a prestigious
high school math competition. Only the top ~2.5% of AMC test-takers qualify.

Each year has 30 problems (AIME I + AIME II, 15 each).
Answers are always integers from 000 to 999.

Datasets: AI-MO/aimo-validation-aime or similar AIME collections
"""

from typing import List

from ..base import BenchmarkLoader, BenchmarkProblem, BenchmarkType
from ..registry import register_loader


class AIMELoaderBase(BenchmarkLoader):
    """Base class for AIME loaders."""

    year: int = 2024

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.MATH

    def _load_from_huggingface(self) -> List[dict]:
        """Load AIME problems from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required. Install with: pip install datasets"
            )

        # Try different sources for AIME data
        sources = [
            ("AI-MO/aimo-validation-aime", None, "train"),
            ("lighteval/aime", None, "test"),
            ("agieval", "aime", "test"),
        ]

        dataset = None
        for source, config, split in sources:
            try:
                if config:
                    dataset = load_dataset(source, config, split=split)
                else:
                    dataset = load_dataset(source, split=split)
                break
            except Exception:
                continue

        if dataset is None:
            # Return empty list - will use fallback data
            return []

        return list(dataset)

    def _filter_by_year(self, items: List[dict]) -> List[dict]:
        """Filter items to only include problems from our target year."""
        filtered = []
        for item in items:
            # Check various fields for year info
            url = str(item.get("url", ""))
            source = str(item.get("source", ""))
            year_field = item.get("year", "")

            year_str = str(self.year)
            if (year_str in url or year_str in source or
                str(year_field) == year_str):
                filtered.append(item)

        return filtered

    def _get_fallback_problems(self) -> List[dict]:
        """
        Return fallback AIME problems if dataset not available.

        In production, this would contain actual AIME problems.
        For now, returns placeholder to indicate where problems would go.
        """
        # This would be populated with actual AIME problems
        # For now, return empty and let the user know
        return []

    def load(self, split: str = "test") -> List[BenchmarkProblem]:
        """
        Load AIME problems for the specified year.

        Args:
            split: Ignored (AIME has single test set)

        Returns:
            List of 30 BenchmarkProblem instances
        """
        # Try to load from HuggingFace
        all_items = self._load_from_huggingface()

        # Filter by year
        year_items = self._filter_by_year(all_items)

        # If no year-specific items, try fallback
        if not year_items:
            year_items = self._get_fallback_problems()

        if not year_items:
            print(f"Warning: No AIME {self.year} problems found in dataset. "
                  f"Using all available problems as placeholder.")
            year_items = all_items[:30] if all_items else []

        problems = []
        for i, item in enumerate(year_items[:30]):  # Cap at 30
            # Extract fields (handle different formats)
            question = item.get("problem", item.get("question", ""))
            answer = item.get("answer", item.get("solution", ""))

            # AIME answers should be integers 0-999
            try:
                if isinstance(answer, str):
                    # Extract numeric answer
                    import re
                    nums = re.findall(r'\d+', str(answer))
                    if nums:
                        answer = int(nums[-1]) % 1000  # Keep in 0-999 range
            except (ValueError, TypeError):
                pass

            problems.append(BenchmarkProblem(
                id=f"aime{self.year}_{i + 1}",
                question=question,
                answer=answer,
                metadata={
                    "year": self.year,
                    "problem_number": i + 1,
                    "source": f"AIME {self.year}",
                }
            ))

        return problems


@register_loader("aime24")
class AIME24Loader(AIMELoaderBase):
    """Loader for AIME 2024 (30 problems)."""

    year = 2024

    @property
    def name(self) -> str:
        return "aime24"

    @property
    def description(self) -> str:
        return (
            "30 problems from AIME 2024 (I + II). "
            "Advanced high school math, integer answers 0-999."
        )


@register_loader("aime25")
class AIME25Loader(AIMELoaderBase):
    """Loader for AIME 2025 (30 problems)."""

    year = 2025

    @property
    def name(self) -> str:
        return "aime25"

    @property
    def description(self) -> str:
        return (
            "30 problems from AIME 2025 (I + II). "
            "Advanced high school math, integer answers 0-999."
        )
