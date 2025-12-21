"""
Math Evaluator for MATH500, AIME, and other math benchmarks.

Handles extraction of answers from LaTeX format (\\boxed{}) and
various "The answer is X" patterns. Normalizes answers for comparison.
"""

import re
from typing import Any, Optional

from ..base import BenchmarkEvaluator, BenchmarkProblem, BenchmarkType
from ..registry import register_evaluator
from ..utils import (
    extract_boxed_answer,
    extract_final_answer,
    extract_last_number,
    normalize_math_answer,
)


@register_evaluator(BenchmarkType.MATH)
class MathEvaluator(BenchmarkEvaluator):
    """
    Evaluator for math problems.

    Supports:
    - MATH dataset (LaTeX \\boxed{} answers)
    - AIME (integer answers 0-999)
    - General math problems with numeric/symbolic answers
    """

    def extract_answer(
        self,
        output: str,
        problem: Optional[BenchmarkProblem] = None
    ) -> Any:
        """
        Extract answer from model output.

        Tries multiple extraction strategies:
        1. LaTeX \\boxed{} format
        2. "The answer is X" patterns
        3. Last number in output

        Args:
            output: Model output text
            problem: Optional problem for context

        Returns:
            Extracted answer string
        """
        if not output:
            return ""

        # Strategy 1: Try boxed answer (most reliable for MATH)
        boxed = extract_boxed_answer(output)
        if boxed:
            return boxed.strip()

        # Strategy 2: Try "final answer" patterns
        final = extract_final_answer(output)
        if final:
            return final.strip()

        # Strategy 3: For AIME, look for integer answer
        if problem and problem.metadata.get("source", "").startswith("AIME"):
            # AIME answers are integers 0-999
            numbers = re.findall(r"\b(\d{1,3})\b", output)
            if numbers:
                return numbers[-1]

        # Strategy 4: Fall back to last number
        last_num = extract_last_number(output)
        if last_num:
            return last_num

        # Strategy 5: Return last non-empty line
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        if lines:
            return lines[-1]

        return output.strip()

    def check_answer(self, extracted: Any, expected: Any) -> bool:
        """
        Check if extracted answer matches expected.

        Normalizes both answers before comparison.

        Args:
            extracted: Answer from model output
            expected: Ground truth answer

        Returns:
            True if answers match
        """
        if extracted is None:
            return False

        # Convert to strings
        extracted_str = str(extracted)
        expected_str = str(expected)

        # If expected is a full solution, try to extract the boxed answer
        if "\\boxed" in expected_str:
            expected_boxed = extract_boxed_answer(expected_str)
            if expected_boxed:
                expected_str = expected_boxed

        # Normalize both
        norm_extracted = normalize_math_answer(extracted_str)
        norm_expected = normalize_math_answer(expected_str)

        # Direct comparison
        if norm_extracted == norm_expected:
            return True

        # Try numeric comparison for floating point tolerance
        try:
            ext_float = float(norm_extracted)
            exp_float = float(norm_expected)
            if abs(ext_float - exp_float) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass

        # For AIME-style integer answers
        try:
            ext_int = int(float(norm_extracted))
            exp_int = int(float(norm_expected))
            if ext_int == exp_int:
                return True
        except (ValueError, TypeError):
            pass

        return False
