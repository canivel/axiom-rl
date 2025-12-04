"""Algorithmic problem generators - V2 (Correct Design).

Each generator creates problems that:
1. Require implementing an ALGORITHM (not just returning a constant)
2. Have MULTIPLE test cases (can't pass by memorization)
3. Take INPUT arguments (can't hardcode)
"""

from typing import List, Optional
from .base_v2 import AlgorithmicGenerator, AlgorithmicProblem, TestCase


class RPNEvaluatorGenerator(AlgorithmicGenerator):
    """
    Generates RPN (Reverse Polish Notation) evaluation problems.

    The model must implement a stack-based evaluator that works
    for ANY valid RPN expression, not just specific examples.

    Example test cases:
        evaluate_rpn("3 4 +") -> 7
        evaluate_rpn("5 2 *") -> 10
        evaluate_rpn("10 3 -") -> 7
        evaluate_rpn("2 3 + 4 *") -> 20
    """

    @property
    def problem_type(self) -> str:
        return "rpn"

    @property
    def title(self) -> str:
        return "RPN Expression Evaluator"

    @property
    def description(self) -> str:
        return """Implement a Reverse Polish Notation (RPN) expression evaluator.

In RPN, operators come AFTER their operands:
- "3 4 +" means 3 + 4 = 7
- "3 4 + 2 *" means (3 + 4) * 2 = 14
- "5 1 2 + 4 * + 3 -" means 5 + ((1 + 2) * 4) - 3 = 14

Rules:
- Tokens are separated by spaces
- Valid operators: +, -, *
- All numbers are single-digit positive integers (1-9)
- Return the final result as an integer

Your function must work for ANY valid RPN expression, not just the examples."""

    @property
    def function_signature(self) -> str:
        return "def evaluate_rpn(expression: str) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate multiple RPN expressions with their correct answers."""
        test_cases = []

        for _ in range(count):
            # Generate RPN expression based on difficulty
            expr, result = self._generate_rpn_expression(difficulty)
            test_cases.append(TestCase(
                input_args=[expr],
                expected_output=result,
            ))

        return test_cases

    def _generate_rpn_expression(self, difficulty: int) -> tuple:
        """Generate a valid RPN expression and its result."""
        # More operators for higher difficulty
        num_ops = min(1 + difficulty // 2, 5)
        ops = ['+', '-', '*']

        # Build expression using a stack simulation
        stack = []
        tokens = []

        # Start with two numbers
        n1 = self.rng.randint(1, 9)
        n2 = self.rng.randint(1, 9)
        stack.append(n1)
        stack.append(n2)
        tokens.append(str(n1))
        tokens.append(str(n2))

        # Add operations
        for _ in range(num_ops):
            if len(stack) >= 2:
                op = self.rng.choice(ops)
                b = stack.pop()
                a = stack.pop()

                if op == '+':
                    result = a + b
                elif op == '-':
                    result = a - b
                else:  # '*'
                    result = a * b

                stack.append(result)
                tokens.append(op)

            # Sometimes add another number
            if self.rng.random() < 0.4 and len(tokens) < difficulty * 2:
                n = self.rng.randint(1, 9)
                stack.append(n)
                tokens.append(str(n))

        # Reduce to single result
        while len(stack) > 1:
            op = self.rng.choice(ops)
            b = stack.pop()
            a = stack.pop()

            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            else:
                result = a * b

            stack.append(result)
            tokens.append(op)

        return " ".join(tokens), stack[0]


class ArithmeticEvaluatorGenerator(AlgorithmicGenerator):
    """
    Generates arithmetic expression evaluation problems.

    The model must implement a parser/evaluator for infix expressions.
    This is HARDER than RPN because of operator precedence and parentheses.

    Example test cases:
        evaluate("3 + 4") -> 7
        evaluate("3 + 4 * 2") -> 11 (not 14!)
        evaluate("(3 + 4) * 2") -> 14
    """

    @property
    def problem_type(self) -> str:
        return "arithmetic"

    @property
    def title(self) -> str:
        return "Arithmetic Expression Evaluator"

    @property
    def description(self) -> str:
        return """Implement an arithmetic expression evaluator.

The expression contains:
- Single-digit positive integers (1-9)
- Operators: +, -, *
- Parentheses for grouping

Rules:
- Follow standard operator precedence (* before + and -)
- Parentheses override precedence
- Spaces separate tokens
- Return the result as an integer

Your function must work for ANY valid expression, not just the examples."""

    @property
    def function_signature(self) -> str:
        return "def evaluate(expression: str) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate multiple arithmetic expressions with correct answers."""
        test_cases = []

        for _ in range(count):
            expr, result = self._generate_expression(difficulty)
            test_cases.append(TestCase(
                input_args=[expr],
                expected_output=result,
            ))

        return test_cases

    def _generate_expression(self, difficulty: int) -> tuple:
        """Generate an expression and evaluate it."""
        # Simple expressions for low difficulty
        if difficulty <= 3:
            # Just "a op b"
            a = self.rng.randint(1, 9)
            b = self.rng.randint(1, 9)
            op = self.rng.choice(['+', '-', '*'])
            expr = f"{a} {op} {b}"
            result = eval(expr)
            return expr, result

        # Medium: "a op b op c" with precedence
        if difficulty <= 6:
            a = self.rng.randint(1, 9)
            b = self.rng.randint(1, 9)
            c = self.rng.randint(1, 9)
            op1 = self.rng.choice(['+', '-', '*'])
            op2 = self.rng.choice(['+', '-', '*'])
            expr = f"{a} {op1} {b} {op2} {c}"
            result = eval(expr)
            return expr, result

        # Hard: With parentheses
        a = self.rng.randint(1, 9)
        b = self.rng.randint(1, 9)
        c = self.rng.randint(1, 9)
        op1 = self.rng.choice(['+', '-'])
        op2 = self.rng.choice(['+', '-', '*'])

        if self.rng.random() < 0.5:
            expr = f"( {a} {op1} {b} ) {op2} {c}"
        else:
            expr = f"{a} {op2} ( {b} {op1} {c} )"

        result = eval(expr.replace(' ', ''))
        return expr, result


class ParenthesesValidatorGenerator(AlgorithmicGenerator):
    """
    Generates parentheses validation problems.

    The model must implement a stack-based validator.

    Example test cases:
        is_valid("()") -> True
        is_valid("()[]{}") -> True
        is_valid("(]") -> False
        is_valid("([)]") -> False
    """

    @property
    def problem_type(self) -> str:
        return "parentheses"

    @property
    def title(self) -> str:
        return "Valid Parentheses Checker"

    @property
    def description(self) -> str:
        return """Implement a function to check if a string of brackets is valid.

A string is valid if:
1. Open brackets are closed by the same type of brackets
2. Open brackets are closed in the correct order
3. Every close bracket has a corresponding open bracket

Valid brackets: (), [], {}

Your function must work for ANY bracket string, not just the examples."""

    @property
    def function_signature(self) -> str:
        return "def is_valid(s: str) -> bool:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate multiple bracket strings with correct validity."""
        test_cases = []

        # Ensure mix of valid and invalid
        num_valid = count // 2 + 1
        num_invalid = count - num_valid

        for _ in range(num_valid):
            s = self._generate_valid(difficulty)
            test_cases.append(TestCase(input_args=[s], expected_output=True))

        for _ in range(num_invalid):
            s = self._generate_invalid(difficulty)
            test_cases.append(TestCase(input_args=[s], expected_output=False))

        self.rng.shuffle(test_cases)
        return test_cases

    def _generate_valid(self, difficulty: int) -> str:
        """Generate a valid bracket string."""
        length = 2 * (1 + difficulty // 2)
        pairs = [("(", ")"), ("[", "]"), ("{", "}")]

        if difficulty <= 3:
            pairs = pairs[:1]
        elif difficulty <= 6:
            pairs = pairs[:2]

        result = []
        stack = []

        while len(result) < length:
            remaining = length - len(result)

            if len(stack) >= remaining // 2:
                # Must close
                result.append(stack.pop())
            elif len(stack) == 0:
                # Must open
                pair = self.rng.choice(pairs)
                result.append(pair[0])
                stack.append(pair[1])
            else:
                # Can do either
                if self.rng.random() < 0.5:
                    pair = self.rng.choice(pairs)
                    result.append(pair[0])
                    stack.append(pair[1])
                else:
                    result.append(stack.pop())

        while stack:
            result.append(stack.pop())

        return "".join(result)

    def _generate_invalid(self, difficulty: int) -> str:
        """Generate an invalid bracket string."""
        pairs = [("(", ")"), ("[", "]"), ("{", "}")]

        if difficulty <= 3:
            pairs = pairs[:1]
        elif difficulty <= 6:
            pairs = pairs[:2]

        # Various invalid patterns
        pattern = self.rng.choice(["mismatch", "unclosed", "extra_close", "wrong_order"])

        if pattern == "mismatch" and len(pairs) > 1:
            # Different bracket types
            p1, p2 = self.rng.sample(pairs, 2)
            return p1[0] + p2[1]

        if pattern == "unclosed":
            pair = self.rng.choice(pairs)
            return pair[0] * 2 + pair[1]

        if pattern == "extra_close":
            pair = self.rng.choice(pairs)
            return pair[1] + pair[0] + pair[1]

        if pattern == "wrong_order" and len(pairs) > 1:
            p1, p2 = self.rng.sample(pairs, 2)
            return p1[0] + p2[0] + p1[1] + p2[1]

        # Fallback
        return self.rng.choice(pairs)[1]


# Registry of V2 generators
GENERATORS_V2 = {
    "rpn": RPNEvaluatorGenerator,
    "arithmetic": ArithmeticEvaluatorGenerator,
    "parentheses": ParenthesesValidatorGenerator,
}


def get_generator_v2(problem_type: str, seed: Optional[int] = None) -> AlgorithmicGenerator:
    """Get a V2 generator by type."""
    if problem_type not in GENERATORS_V2:
        raise ValueError(f"Unknown type: {problem_type}. Available: {list(GENERATORS_V2.keys())}")
    return GENERATORS_V2[problem_type](seed=seed)


def get_all_generators_v2(seed: Optional[int] = None) -> dict:
    """Get all V2 generators."""
    return {name: cls(seed=seed) for name, cls in GENERATORS_V2.items()}
