"""
GPQA Benchmark Loaders.

GPQA (Graduate-level Google-Proof Q&A) is a challenging benchmark of
multiple-choice questions written by domain experts. Questions are designed
to be difficult even for experts and resistant to simple web searches.

Domains: Physics, Chemistry, Biology
Format: 4-choice multiple choice questions

GPQA Diamond is the hardest subset (198 questions) with additional validation.

Dataset: Idavidrein/gpqa
"""

from typing import List
import random

from ..base import BenchmarkLoader, BenchmarkProblem, BenchmarkType
from ..registry import register_loader


class GPQALoaderBase(BenchmarkLoader):
    """Base class for GPQA loaders."""

    subset: str = "gpqa_main"  # or "gpqa_diamond"

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.MCQ

    def _shuffle_options(self, options: List[str], correct_idx: int, seed: int) -> tuple:
        """
        Shuffle options and return (shuffled_options, new_correct_idx).

        This prevents models from exploiting position bias.
        """
        rng = random.Random(seed)
        indexed = list(enumerate(options))
        rng.shuffle(indexed)

        shuffled = [opt for _, opt in indexed]
        new_idx = next(i for i, (orig_i, _) in enumerate(indexed) if orig_i == correct_idx)

        return shuffled, new_idx

    def load(self, split: str = "train") -> List[BenchmarkProblem]:
        """
        Load GPQA problems.

        Args:
            split: Dataset split (GPQA uses "train" as the main split)

        Returns:
            List of BenchmarkProblem instances
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required. Install with: pip install datasets"
            )

        try:
            dataset = load_dataset("Idavidrein/gpqa", self.subset, split=split)
        except Exception as e:
            raise RuntimeError(
                f"Could not load GPQA dataset ({self.subset}): {e}"
            )

        problems = []
        for i, item in enumerate(dataset):
            # Extract question
            question = item.get("Question", item.get("question", ""))

            # Extract options
            correct = item.get("Correct Answer", item.get("correct_answer", ""))
            incorrect1 = item.get("Incorrect Answer 1", item.get("incorrect_answer_1", ""))
            incorrect2 = item.get("Incorrect Answer 2", item.get("incorrect_answer_2", ""))
            incorrect3 = item.get("Incorrect Answer 3", item.get("incorrect_answer_3", ""))

            # Build options list (correct is always first before shuffle)
            options = [correct, incorrect1, incorrect2, incorrect3]
            options = [o for o in options if o]  # Remove empty

            # Shuffle options to randomize correct answer position
            if len(options) >= 4:
                shuffled_options, correct_idx = self._shuffle_options(
                    options, 0, seed=i
                )
                correct_letter = chr(ord("A") + correct_idx)
            else:
                shuffled_options = options
                correct_letter = "A"

            # Format question with options
            formatted_question = question + "\n\n"
            for j, opt in enumerate(shuffled_options):
                letter = chr(ord("A") + j)
                formatted_question += f"{letter}. {opt}\n"

            problems.append(BenchmarkProblem(
                id=item.get("Record ID", f"gpqa_{i}"),
                question=formatted_question,
                answer=correct_letter,  # The letter of the correct option
                metadata={
                    "domain": item.get("High-level domain", item.get("domain", "unknown")),
                    "subdomain": item.get("Subdomain", item.get("subdomain", "unknown")),
                    "options": shuffled_options,
                    "correct_text": correct,  # Original correct answer text
                    "source": self.subset,
                }
            ))

        return problems


@register_loader("gpqa")
class GPQALoader(GPQALoaderBase):
    """Loader for full GPQA benchmark (448 questions)."""

    subset = "gpqa_main"

    @property
    def name(self) -> str:
        return "gpqa"

    @property
    def description(self) -> str:
        return (
            "448 graduate-level science MCQ (physics, chemistry, biology). "
            "Expert-written, designed to be Google-proof."
        )


@register_loader("gpqa_diamond")
class GPQADiamondLoader(GPQALoaderBase):
    """Loader for GPQA Diamond (198 hardest questions)."""

    subset = "gpqa_diamond"

    @property
    def name(self) -> str:
        return "gpqa_diamond"

    @property
    def description(self) -> str:
        return (
            "198 hardest questions from GPQA. "
            "Additional expert validation, highest difficulty."
        )
