"""Arithmetic expression evaluation generator.

Generates problems like:
    Input: "2 + 3 * 4"
    Output: 14

The model must implement correct operator precedence.
"""

from typing import Any, Optional

from .base import ProceduralGenerator


class ArithmeticGenerator(ProceduralGenerator):
    """
    Generates arithmetic expression evaluation problems.

    Difficulty scales with:
    - Number of operands (2-10)
    - Types of operators (+, -, *, /, //, %)
    - Use of parentheses
    - Number magnitude
    """

    @property
    def problem_type(self) -> str:
        return "arithmetic"

    @property
    def title(self) -> str:
        return "Evaluate Arithmetic Expression"

    @property
    def description_template(self) -> str:
        return """Given a string containing an arithmetic expression with integers and operators (+, -, *, /, //, %), evaluate the expression and return the result as an integer.

Rules:
- Follow standard operator precedence (*, /, //, % before +, -)
- Division (/) should be integer division (truncate toward zero)
- The expression may contain parentheses
- All numbers are integers
- The result should be an integer"""

    @property
    def function_signature(self) -> str:
        return "def evaluate_expression(expr: str) -> int:"

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.basic_ops = ["+", "-", "*"]
        self.advanced_ops = ["+", "-", "*", "//", "%"]

    def generate_instance(self, difficulty: int = 5) -> tuple[str, int]:
        """
        Generate an arithmetic expression problem.

        Difficulty mapping:
        1-2: 2 operands, +/- only, small numbers
        3-4: 3-4 operands, +/-/* , medium numbers
        5-6: 4-5 operands, all ops, with some parentheses
        7-8: 5-7 operands, all ops, nested parentheses
        9-10: 7-10 operands, complex nesting
        """
        # Determine parameters based on difficulty
        if difficulty <= 2:
            num_operands = 2
            ops = ["+", "-"]
            max_num = 10
            use_parens = False
        elif difficulty <= 4:
            num_operands = self.rng.randint(3, 4)
            ops = ["+", "-", "*"]
            max_num = 50
            use_parens = False
        elif difficulty <= 6:
            num_operands = self.rng.randint(4, 5)
            ops = self.advanced_ops
            max_num = 100
            use_parens = self.rng.random() < 0.5
        elif difficulty <= 8:
            num_operands = self.rng.randint(5, 7)
            ops = self.advanced_ops
            max_num = 100
            use_parens = True
        else:
            num_operands = self.rng.randint(7, 10)
            ops = self.advanced_ops
            max_num = 200
            use_parens = True

        # Generate expression
        expr = self._generate_expression(num_operands, ops, max_num, use_parens)

        # Calculate expected result
        try:
            # Use Python's eval for ground truth
            # Replace // with integer division behavior we want
            result = int(eval(expr))
        except (ZeroDivisionError, ValueError):
            # If we hit division by zero, regenerate
            return self.generate_instance(difficulty)

        return expr, result

    def _generate_expression(
        self,
        num_operands: int,
        ops: list[str],
        max_num: int,
        use_parens: bool,
    ) -> str:
        """Generate a random arithmetic expression."""
        # Generate operands (avoiding zero for division safety)
        operands = []
        for i in range(num_operands):
            if i > 0 and ops and any(op in ["//", "%", "/"] for op in ops):
                # Avoid zero after division operators
                num = self.rng.randint(1, max_num)
            else:
                num = self.rng.randint(0, max_num)
            operands.append(str(num))

        # Generate operators
        operators = [self.rng.choice(ops) for _ in range(num_operands - 1)]

        # Build expression
        if not use_parens:
            # Simple: num op num op num ...
            parts = []
            for i, operand in enumerate(operands):
                parts.append(operand)
                if i < len(operators):
                    parts.append(operators[i])
            return " ".join(parts)
        else:
            # With parentheses
            return self._add_parentheses(operands, operators)

    def _add_parentheses(self, operands: list[str], operators: list[str]) -> str:
        """Add parentheses to an expression."""
        if len(operands) <= 2:
            # Too short for meaningful parentheses
            parts = []
            for i, operand in enumerate(operands):
                parts.append(operand)
                if i < len(operators):
                    parts.append(operators[i])
            return " ".join(parts)

        # Decide where to add parentheses
        # Pick a random contiguous subsequence to parenthesize
        start = self.rng.randint(0, len(operands) - 2)
        length = self.rng.randint(2, min(3, len(operands) - start))

        parts = []
        i = 0
        while i < len(operands):
            if i == start:
                # Start parenthesized group
                group_parts = []
                for j in range(length):
                    if i + j < len(operands):
                        group_parts.append(operands[i + j])
                        if i + j < len(operators):
                            group_parts.append(operators[i + j])
                # Remove trailing operator if present
                if group_parts and group_parts[-1] in self.advanced_ops + ["+", "-"]:
                    group_parts = group_parts[:-1]
                parts.append("(" + " ".join(group_parts) + ")")
                i += length
                if i < len(operands) and i - 1 < len(operators):
                    parts.append(operators[i - 1])
            else:
                parts.append(operands[i])
                if i < len(operators):
                    parts.append(operators[i])
                i += 1

        return " ".join(parts)

    def verify(self, input_data: str, output: int, expected: int) -> bool:
        """Verify the output matches expected."""
        return output == expected
