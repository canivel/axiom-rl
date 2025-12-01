"""Prompt templates for code generation."""

import re
from typing import List

from ..problems.base import Problem

# System prompt for code generation
SYSTEM_PROMPT = """You are an expert Python programmer. Your task is to solve algorithmic problems by writing clean, efficient, and correct Python code.

Rules:
1. Write ONLY the function implementation - no explanations, no test code
2. The function signature is provided - implement the function body
3. Use standard Python libraries only (no external packages)
4. Write clear, readable code
5. Handle edge cases appropriately
6. Return the result as specified"""


def build_user_prompt(problem: Problem) -> str:
    """Build the user prompt for a problem."""
    return f"""Solve the following problem by implementing the function.

## Problem: {problem.title}

{problem.description}

## Function Signature
```python
{problem.function_signature}
    # Your implementation here
```

Write ONLY the complete function implementation. Do not include any explanations, examples, or test code."""


def build_messages(problem: Problem) -> List[dict]:
    """Build the chat messages for a problem."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(problem)},
    ]


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from the model's response.

    Handles:
    - Code wrapped in ```python ... ```
    - Code wrapped in ``` ... ```
    - Raw code without markers
    """
    # Try to extract from ```python blocks
    python_pattern = r"```python\s*(.*?)```"
    matches = re.findall(python_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try to extract from generic ``` blocks
    generic_pattern = r"```\s*(.*?)```"
    matches = re.findall(generic_pattern, response, re.DOTALL)
    if matches:
        code = matches[0].strip()
        # Remove language identifier if present on first line
        lines = code.split("\n")
        if lines and lines[0].strip().lower() in ["python", "py", ""]:
            code = "\n".join(lines[1:])
        return code.strip()

    # No code blocks - assume the whole response is code
    # But try to clean it up by removing common prefixes
    lines = response.strip().split("\n")
    cleaned_lines = []
    for line in lines:
        # Skip lines that look like explanations
        if line.strip().startswith(("#", "//", "/*", "Here", "This", "The", "Note")):
            if not line.strip().startswith("# "):  # Keep Python comments
                continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
