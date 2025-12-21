"""
Multiple Choice Question (MCQ) Evaluator.

Used for GPQA and other multiple choice benchmarks.
Extracts answer choice (A, B, C, D) from model output.
"""

import re
from typing import Any, Optional, List

from ..base import BenchmarkEvaluator, BenchmarkProblem, BenchmarkType
from ..registry import register_evaluator
from ..utils import extract_mcq_answer


@register_evaluator(BenchmarkType.MCQ)
class MCQEvaluator(BenchmarkEvaluator):
    """
    Evaluator for multiple choice questions.

    Supports:
    - GPQA (4-choice questions)
    - General MCQ with A/B/C/D/E options
    """

    def extract_answer(
        self,
        output: str,
        problem: Optional[BenchmarkProblem] = None
    ) -> Any:
        """
        Extract MCQ answer (letter) from model output.

        Tries multiple extraction strategies:
        1. Explicit "Answer: X" patterns
        2. Option letter at end of response
        3. Match option text content

        Args:
            output: Model output text
            problem: Problem with options in metadata

        Returns:
            Answer letter (A, B, C, D) or None
        """
        if not output:
            return None

        # Get options from problem if available
        options = []
        if problem and "options" in problem.metadata:
            options = problem.metadata["options"]

        # Strategy 1: Use utility function
        answer = extract_mcq_answer(output, options)
        if answer:
            return answer.upper()

        # Strategy 2: Look for letter patterns more aggressively
        # Check last few lines for standalone letters
        lines = output.strip().split("\n")
        for line in reversed(lines[-5:]):
            line = line.strip()
            # Pattern: just a letter, possibly with punctuation
            match = re.match(r"^[(\[]?([A-Da-d])[)\].\s]*$", line)
            if match:
                return match.group(1).upper()

            # Pattern: "X is correct"
            match = re.search(r"\b([A-Da-d])\b\s+is\s+(the\s+)?correct", line, re.I)
            if match:
                return match.group(1).upper()

        # Strategy 3: Find most confident letter mention
        # Count letter occurrences in reasoning
        letter_counts = {l: 0 for l in "ABCD"}
        for match in re.finditer(r"\b([A-Da-d])\b", output):
            letter = match.group(1).upper()
            if letter in letter_counts:
                letter_counts[letter] += 1

        # Return most frequent if there's a clear winner
        if letter_counts:
            max_count = max(letter_counts.values())
            if max_count > 0:
                winners = [l for l, c in letter_counts.items() if c == max_count]
                if len(winners) == 1:
                    return winners[0]

        # Strategy 4: Check if model output contains exact option text
        if options:
            output_lower = output.lower()
            for i, opt in enumerate(options):
                if opt.lower().strip() in output_lower:
                    return chr(ord("A") + i)

        return None

    def check_answer(self, extracted: Any, expected: Any) -> bool:
        """
        Check if extracted answer matches expected.

        Args:
            extracted: Letter from model output (A, B, C, D)
            expected: Correct answer letter

        Returns:
            True if answers match
        """
        if extracted is None or expected is None:
            return False

        # Normalize to uppercase letters
        ext = str(extracted).strip().upper()
        exp = str(expected).strip().upper()

        # Handle if expected is full text but should be a letter
        if len(exp) > 1:
            # Maybe expected is the option text, not letter
            # In this case, we can't directly compare
            return ext == exp

        return ext == exp
