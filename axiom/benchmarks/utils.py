"""
Utility functions for benchmark evaluation.

Includes answer normalization, extraction helpers, and formatting.
"""

import re
from typing import Optional, List, Any
from fractions import Fraction


def normalize_math_answer(answer: str) -> str:
    """
    Normalize a math answer for comparison.

    Handles:
    - Whitespace removal
    - LaTeX formatting
    - Fraction normalization
    - Common equivalences

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    # Convert to string if needed
    answer = str(answer)

    # Remove leading/trailing whitespace
    answer = answer.strip()

    # Remove dollar signs (LaTeX math mode)
    answer = answer.replace("$", "")

    # Remove \text{} wrappers
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)

    # Remove spaces
    answer = re.sub(r"\s+", "", answer)

    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)

    # Remove redundant parentheses around single terms
    # e.g., (5) -> 5
    answer = re.sub(r"^\(([^()]+)\)$", r"\1", answer)

    # Lowercase
    answer = answer.lower()

    # Try to evaluate simple numeric expressions
    try:
        # Check if it's a simple fraction like 3/4
        if "/" in answer and answer.replace("/", "").replace("-", "").isdigit():
            parts = answer.split("/")
            if len(parts) == 2:
                frac = Fraction(int(parts[0]), int(parts[1]))
                return str(float(frac))
    except (ValueError, ZeroDivisionError):
        pass

    return answer


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from LaTeX \\boxed{} format.

    Handles nested braces properly.

    Args:
        text: Text containing boxed answer

    Returns:
        Content of boxed answer, or None if not found
    """
    # Find \boxed{ and match braces
    patterns = [
        r"\\boxed\{",
        r"\\fbox\{",
        r"\\framebox\{",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            start = match.end()
            depth = 1
            pos = start

            while pos < len(text) and depth > 0:
                if text[pos] == "{":
                    depth += 1
                elif text[pos] == "}":
                    depth -= 1
                pos += 1

            if depth == 0:
                return text[start:pos - 1].strip()

    return None


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract answer from common "final answer" patterns.

    Looks for patterns like:
    - "The answer is X"
    - "Final answer: X"
    - "Therefore, X"

    Args:
        text: Model output text

    Returns:
        Extracted answer or None
    """
    patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
        r"[Ff]inal\s+[Aa]nswer[:\s]+([^\n.]+)",
        r"[Tt]herefore,?\s+(?:the\s+answer\s+is\s+)?([^\n.]+)",
        r"[Hh]ence,?\s+(?:the\s+answer\s+is\s+)?([^\n.]+)",
        r"[Aa]nswer[:\s]+([^\n.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return None


def extract_last_number(text: str) -> Optional[str]:
    """
    Extract the last number from text.

    Useful as fallback for numeric answers.

    Args:
        text: Text to search

    Returns:
        Last number found, or None
    """
    # Match integers and decimals, including negative
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_mcq_answer(text: str, options: List[str]) -> Optional[str]:
    """
    Extract multiple choice answer from text.

    Looks for option letters (A, B, C, D) or matches option content.

    Args:
        text: Model output
        options: List of option texts

    Returns:
        Matched option or letter, or None
    """
    text_lower = text.lower().strip()

    # Look for explicit letter choices
    # Patterns: "Answer: A", "The answer is B", "(C)", etc.
    letter_patterns = [
        r"[Aa]nswer[:\s]+\(?([A-Da-d])\)?",
        r"[Cc]hoose\s+\(?([A-Da-d])\)?",
        r"\(?([A-Da-d])\)?\s*is\s+(?:the\s+)?(?:correct|right|answer)",
        r"^([A-Da-d])[\.\)\s]",  # Just the letter at start
        r"\b([A-Da-d])\b\s*$",  # Just the letter at end
    ]

    for pattern in letter_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    # Check if any option text is contained in the response
    for i, option in enumerate(options):
        option_lower = option.lower().strip()
        if option_lower in text_lower:
            return chr(ord("A") + i)

    return None


def format_accuracy(accuracy: float, precision: int = 1) -> str:
    """Format accuracy as percentage string."""
    return f"{accuracy * 100:.{precision}f}%"


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
