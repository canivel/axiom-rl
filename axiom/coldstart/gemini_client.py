"""Gemini API client for generating reasoning traces."""

import os
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv


@dataclass
class GeminiResponse:
    """Response from Gemini API."""

    thinking: str  # The reasoning trace
    code: str  # The generated code
    raw_response: str  # Full raw response


class GeminiClient:
    """
    Client for Gemini 2.5 API to generate reasoning traces.

    This is the "Teacher" model that generates high-quality
    reasoning examples to prime the smaller model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Model to use (default: gemini-2.5-flash)
        """
        # Load .env file
        load_dotenv()

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env or pass api_key parameter."
            )

        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def generate_reasoning_trace(
        self,
        problem_title: str,
        problem_description: str,
        function_signature: str,
    ) -> GeminiResponse:
        """
        Generate a reasoning trace for a problem.

        Args:
            problem_title: Title of the problem
            problem_description: Full problem description
            function_signature: The function signature to implement

        Returns:
            GeminiResponse with thinking trace and code
        """
        prompt = self._build_prompt(
            problem_title, problem_description, function_signature
        )

        response = self.model.generate_content(prompt)
        raw_text = response.text

        # Parse the response
        thinking, code = self._parse_response(raw_text)

        return GeminiResponse(
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
