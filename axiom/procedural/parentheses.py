"""Parentheses validation generator.

Generates problems like:
    Input: "({[]})"
    Output: True

    Input: "({[}])"
    Output: False

The model must implement a stack-based bracket matcher.
"""

from typing import Any, Optional

from .base import ProceduralGenerator


class ParenthesesGenerator(ProceduralGenerator):
    """
    Generates parentheses/bracket validation problems.

    This is a classic stack problem where the model must determine
    if brackets are properly matched and nested.

    Difficulty scales with:
    - String length (4-50 characters)
    - Number of bracket types (1-3)
    - Nesting depth
    - Mix of valid/invalid patterns
    """

    @property
    def problem_type(self) -> str:
        return "parentheses"

    @property
    def title(self) -> str:
        return "Valid Parentheses"

    @property
    def description_template(self) -> str:
        return """Determine if a string containing brackets is valid.

A string is valid if:
1. Open brackets must be closed by the same type of brackets
2. Open brackets must be closed in the correct order
3. Every close bracket has a corresponding open bracket

Valid bracket pairs: (), [], {}

Examples:
- "()" → True
- "()[]{}" → True
- "(]" → False
- "([)]" → False
- "{[]}" → True

Return True if valid, False otherwise."""

    @property
    def function_signature(self) -> str:
        return "def is_valid_parentheses(s: str) -> bool:"

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.bracket_pairs = [("(", ")"), ("[", "]"), ("{", "}")]

    def generate_instance(self, difficulty: int = 5) -> tuple[str, bool]:
        """
        Generate a parentheses validation problem.

        Difficulty mapping:
        1-2: Short strings (4-8), one bracket type, mostly valid
        3-4: Medium strings (8-16), two bracket types
        5-6: Longer strings (12-24), all bracket types
        7-8: Complex strings (20-35), tricky patterns
        9-10: Long complex strings (30-50), edge cases
        """
        # Determine parameters based on difficulty
        if difficulty <= 2:
            length = self.rng.randint(4, 8)
            num_types = 1
            valid_prob = 0.7
        elif difficulty <= 4:
            length = self.rng.randint(8, 16)
            num_types = 2
            valid_prob = 0.5
        elif difficulty <= 6:
            length = self.rng.randint(12, 24)
            num_types = 3
            valid_prob = 0.5
        elif difficulty <= 8:
            length = self.rng.randint(20, 35)
            num_types = 3
            valid_prob = 0.4
        else:
            length = self.rng.randint(30, 50)
            num_types = 3
            valid_prob = 0.3

        # Select bracket types to use
        pairs = self.bracket_pairs[:num_types]

        # Decide if we're generating a valid or invalid string
        generate_valid = self.rng.random() < valid_prob

        if generate_valid:
            s = self._generate_valid(length, pairs)
            return s, True
        else:
            s = self._generate_invalid(length, pairs)
            return s, False

    def _generate_valid(
        self,
        target_length: int,
        pairs: list[tuple[str, str]],
    ) -> str:
        """Generate a valid bracket string."""
        result = []
        stack = []

        while len(result) < target_length:
            remaining = target_length - len(result)

            # Decide: open new bracket or close existing
            can_open = remaining >= 2  # Need room for open + close
            can_close = len(stack) > 0

            if can_open and can_close:
                # Random choice, but ensure we can close all brackets
                if len(stack) >= remaining // 2:
                    # Must close to fit
                    action = "close"
                else:
                    action = self.rng.choice(["open", "close"])
            elif can_open:
                action = "open"
            elif can_close:
                action = "close"
            else:
                break

            if action == "open":
                pair = self.rng.choice(pairs)
                result.append(pair[0])
                stack.append(pair[1])
            else:
                # Close most recent bracket
                result.append(stack.pop())

        # Close any remaining open brackets
        while stack:
            result.append(stack.pop())

        return "".join(result)

    def _generate_invalid(
        self,
        target_length: int,
        pairs: list[tuple[str, str]],
    ) -> str:
        """Generate an invalid bracket string."""
        # Strategy: Generate valid-ish structure, then introduce an error
        error_type = self.rng.choice([
            "mismatched",  # Wrong closing bracket
            "unclosed",    # Open bracket never closed
            "extra_close", # Close without open
            "wrong_order", # Interleaved incorrectly
        ])

        if error_type == "mismatched":
            return self._generate_mismatched(target_length, pairs)
        elif error_type == "unclosed":
            return self._generate_unclosed(target_length, pairs)
        elif error_type == "extra_close":
            return self._generate_extra_close(target_length, pairs)
        else:
            return self._generate_wrong_order(target_length, pairs)

    def _generate_mismatched(
        self,
        target_length: int,
        pairs: list[tuple[str, str]],
    ) -> str:
        """Generate string with mismatched bracket types."""
        if len(pairs) < 2:
            # Need at least 2 types for mismatch
            return self._generate_extra_close(target_length, pairs)

        result = []
        stack = []

        while len(result) < target_length - 2:
            pair = self.rng.choice(pairs)
            result.append(pair[0])
            stack.append(pair)

        if stack:
            # Close with wrong bracket type
            correct_pair = stack.pop()
            wrong_pairs = [p for p in pairs if p != correct_pair]
            if wrong_pairs:
                wrong_pair = self.rng.choice(wrong_pairs)
                result.append(wrong_pair[1])

        # Close remaining correctly
        while stack:
            result.append(stack.pop()[1])

        return "".join(result)

    def _generate_unclosed(
        self,
        target_length: int,
        pairs: list[tuple[str, str]],
    ) -> str:
        """Generate string with unclosed bracket."""
        result = []

        # Add some valid pairs
        for _ in range((target_length - 1) // 2):
            pair = self.rng.choice(pairs)
            result.append(pair[0])
            result.append(pair[1])

        # Add one unclosed bracket
        pair = self.rng.choice(pairs)
        insert_pos = self.rng.randint(0, len(result))
        result.insert(insert_pos, pair[0])

        return "".join(result)

    def _generate_extra_close(
        self,
        target_length: int,
        pairs: list[tuple[str, str]],
    ) -> str:
        """Generate string with extra closing bracket."""
        result = []

        # Add some valid pairs
        for _ in range((target_length - 1) // 2):
            pair = self.rng.choice(pairs)
            result.append(pair[0])
            result.append(pair[1])

        # Add one extra close at the beginning
        pair = self.rng.choice(pairs)
        result.insert(0, pair[1])

        return "".join(result)

    def _generate_wrong_order(
        self,
        target_length: int,
        pairs: list[tuple[str, str]],
    ) -> str:
        """Generate string with incorrectly interleaved brackets."""
        if len(pairs) < 2:
            return self._generate_extra_close(target_length, pairs)

        # Classic wrong order: ([)] instead of ([])
        pair1, pair2 = self.rng.sample(pairs, 2)
        base = f"{pair1[0]}{pair2[0]}{pair1[1]}{pair2[1]}"

        # Pad to target length with valid pairs
        result = list(base)
        while len(result) < target_length:
            pair = self.rng.choice(pairs)
            result.append(pair[0])
            result.append(pair[1])

        return "".join(result[:target_length])

    def verify(self, input_data: str, output: bool, expected: bool) -> bool:
        """Verify the output matches expected."""
        return output == expected
