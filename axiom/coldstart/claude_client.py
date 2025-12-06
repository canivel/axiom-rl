"""Claude API client for generating reasoning traces."""

import os
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv


@dataclass
class ClaudeResponse:
    """Response from Claude API."""

    thinking: str  # The reasoning trace
    code: str  # The generated code
    raw_response: str  # Full raw response


class ClaudeClient:
    """
    Client for Claude API to generate reasoning traces.

    This is an alternative "Teacher" model that generates high-quality
    reasoning examples to prime the smaller model.

    Claude's extended thinking produces particularly detailed reasoning
    traces that can complement Gemini's traces for more robust distillation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the Claude client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model_name: Model to use (default: claude-sonnet-4-20250514)
        """
        # Load .env file
        load_dotenv()

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in .env or pass api_key parameter."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model_name = model_name

    def generate_reasoning_trace(
        self,
        problem_title: str,
        problem_description: str,
        function_signature: str,
        temperature: float = 0.7,
    ) -> ClaudeResponse:
        """
        Generate a reasoning trace for a problem.

        Args:
            problem_title: Title of the problem
            problem_description: Full problem description
            function_signature: The function signature to implement
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            ClaudeResponse with thinking trace and code
        """
        prompt = self._build_prompt(
            problem_title, problem_description, function_signature
        )

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        raw_text = response.content[0].text

        # Parse the response
        thinking, code = self._parse_response(raw_text)

        return ClaudeResponse(
            thinking=thinking,
            code=code,
            raw_response=raw_text,
        )

    def _build_prompt(
        self,
        problem_title: str,
        problem_description: str,
        function_signature: str,
    ) -> str:
        """Build the prompt for reasoning trace generation."""
        return f"""You are an expert Python programmer solving algorithmic problems.

Your task is to solve the problem below. You MUST:
1. First, show your reasoning process inside <think> tags
2. Then, provide the complete Python code inside ```python``` blocks

## Problem: {problem_title}

{problem_description}

## Function Signature
```python
{function_signature}
```

## Required Format

<think>
Step 1: [Understand the problem - what are the inputs/outputs?]
Step 2: [Consider edge cases]
Step 3: [Plan the approach - what algorithm/data structure?]
Step 4: [Analyze time/space complexity]
Step 5: [Implement the solution]
</think>

```python
# Your complete implementation here
{function_signature}
    # implementation
```

Remember:
- Show your complete reasoning process in <think> tags
- Write clean, efficient Python code
- Handle edge cases
- Use only standard library (no external packages)"""

    def _parse_response(self, raw_text: str) -> tuple[str, str]:
        """
        Parse the raw response to extract thinking and code.

        Returns:
            (thinking_text, code_text)
        """
        import re

        # Extract thinking
        thinking_match = re.search(
            r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE
        )
        thinking = thinking_match.group(1).strip() if thinking_match else ""

        # Extract code
        code_match = re.search(r"```python\s*(.*?)```", raw_text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else ""

        return thinking, code
