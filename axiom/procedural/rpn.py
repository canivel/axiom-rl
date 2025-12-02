"""Reverse Polish Notation (RPN) expression evaluator generator.

Generates problems like:
    Input: "3 4 + 2 *"
    Output: 14

The model must implement a stack-based RPN evaluator.
"""

from typing import Any, Optional

from .base import ProceduralGenerator


class RPNGenerator(ProceduralGenerator):
    """
    Generates RPN (Reverse Polish Notation) evaluation problems.

    RPN is a mathematical notation where operators follow their operands.
    Example: "3 4 +" means 3 + 4 = 7

    Difficulty scales with:
    - Number of operands (2-10)
    - Types of operators (+, -, *, //)
    - Number magnitude
    - Expression complexity
    """

    @property
    def problem_type(self) -> str:
        return "rpn"

    @property
    def title(self) -> str:
        return "Evaluate RPN Expression"

    @property
    def description_template(self) -> str:
        return """Evaluate a Reverse Polish Notation (RPN) expression.

In RPN, operators follow their operands. For example:
- "3 4 +" means 3 + 4 = 7
- "3 4 + 2 *" means (3 + 4) * 2 = 14

Rules:
- Tokens are separated by spaces
- Valid operators: +, -, *, // (integer division)
- All numbers are integers
- Division should truncate toward zero
- The expression is always valid (enough operands for each operator)
- Return the final result as an integer"""

    @property
    def function_signature(self) -> str:
        return "def evaluate_rpn(expression: str) -> int:"

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.basic_ops = ["+", "-", "*"]
        self.all_ops = ["+", "-", "*", "//"]

    def generate_instance(self, difficulty: int = 5) -> tuple[str, int]:
        """
        Generate an RPN expression problem.

        Difficulty mapping:
        1-2: 2-3 operands, +/- only, small numbers
        3-4: 3-4 operands, +/-/*, medium numbers
        5-6: 4-5 operands, all ops, larger numbers
        7-8: 5-7 operands, all ops, complex expressions
        9-10: 7-10 operands, all ops, large numbers
        """
        # Determine parameters based on difficulty
        if difficulty <= 2:
            num_operands = self.rng.randint(2, 3)
            ops = ["+", "-"]
            max_num = 10
        elif difficulty <= 4:
            num_operands = self.rng.randint(3, 4)
            ops = self.basic_ops
            max_num = 50
        elif difficulty <= 6:
            num_operands = self.rng.randint(4, 5)
            ops = self.all_ops
            max_num = 100
        elif difficulty <= 8:
            num_operands = self.rng.randint(5, 7)
            ops = self.all_ops
            max_num = 100
        else:
            num_operands = self.rng.randint(7, 10)
            ops = self.all_ops
            max_num = 200

        # Generate valid RPN expression
        expr, result = self._generate_rpn_expression(num_operands, ops, max_num)

        return expr, result

    def _generate_rpn_expression(
        self,
        num_operands: int,
        ops: list[str],
        max_num: int,
    ) -> tuple[str, int]:
        """
        Generate a valid RPN expression and compute its result.

        Strategy: Build expression by simulating a stack.
        We track how many values are on the stack and only
        add operators when there are at least 2 values.
        """
        tokens = []
        stack = []  # Simulated stack for computing result

        operands_added = 0
        operators_needed = num_operands - 1

        while operands_added < num_operands or operators_needed > 0:
            # Decide whether to add operand or operator
            can_add_operand = operands_added < num_operands
            can_add_operator = len(stack) >= 2 and operators_needed > 0

            if can_add_operand and can_add_operator:
                # Choose randomly, but bias toward operands early
                add_operand = self.rng.random() < 0.6
            elif can_add_operand:
                add_operand = True
            elif can_add_operator:
                add_operand = False
            else:
                break

            if add_operand:
                # Add a number (avoid zero for division safety)
                if "//" in ops and len(stack) >= 1:
                    num = self.rng.randint(1, max_num)
                else:
                    num = self.rng.randint(0, max_num)
                tokens.append(str(num))
                stack.append(num)
                operands_added += 1
            else:
                # Add an operator
                op = self.rng.choice(ops)

                # Pop two values, compute, push result
                b = stack.pop()
                a = stack.pop()

                # Handle division by zero
                if op == "//" and b == 0:
                    # Re-add values and try different operator
                    stack.append(a)
                    stack.append(b)
                    op = self.rng.choice([o for o in ops if o != "//"])
                    b = stack.pop()
                    a = stack.pop()

                # Compute result
                if op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                elif op == "*":
                    result = a * b
                elif op == "//":
                    # Integer division truncating toward zero
                    result = int(a / b) if b != 0 else 0
                else:
                    result = a + b  # fallback

                tokens.append(op)
                stack.append(result)
                operators_needed -= 1

        # The final result is the only value left on the stack
        final_result = stack[0] if stack else 0

        return " ".join(tokens), final_result

    def verify(self, input_data: str, output: int, expected: int) -> bool:
        """Verify the output matches expected."""
        return output == expected
