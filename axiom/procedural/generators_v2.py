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


class FizzBuzzGenerator(AlgorithmicGenerator):
    """
    Generates FizzBuzz problems with varying rules.

    Classic FizzBuzz: Return "Fizz" for multiples of 3, "Buzz" for 5,
    "FizzBuzz" for both, otherwise the number as string.
    """

    @property
    def problem_type(self) -> str:
        return "fizzbuzz"

    @property
    def title(self) -> str:
        return "FizzBuzz"

    @property
    def description(self) -> str:
        return """Implement the FizzBuzz function.

Given an integer n, return:
- "FizzBuzz" if n is divisible by both 3 and 5
- "Fizz" if n is divisible by 3 (but not 5)
- "Buzz" if n is divisible by 5 (but not 3)
- The number as a string otherwise

Your function must work for ANY positive integer."""

    @property
    def function_signature(self) -> str:
        return "def fizzbuzz(n: int) -> str:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate diverse FizzBuzz test cases."""
        test_cases = []

        # Ensure we cover all cases
        categories = {
            'fizzbuzz': [15, 30, 45, 60, 75, 90],  # divisible by both
            'fizz': [3, 6, 9, 12, 18, 21, 24, 27],  # divisible by 3 only
            'buzz': [5, 10, 20, 25, 35, 40],  # divisible by 5 only
            'number': [1, 2, 4, 7, 8, 11, 13, 14, 16, 17, 19, 22, 23],  # neither
        }

        # Pick from each category
        for category, values in categories.items():
            n = self.rng.choice(values)
            if category == 'fizzbuzz':
                expected = "FizzBuzz"
            elif category == 'fizz':
                expected = "Fizz"
            elif category == 'buzz':
                expected = "Buzz"
            else:
                expected = str(n)
            test_cases.append(TestCase(input_args=[n], expected_output=expected))

        # Add more random cases if needed
        while len(test_cases) < count:
            n = self.rng.randint(1, 100)
            if n % 15 == 0:
                expected = "FizzBuzz"
            elif n % 3 == 0:
                expected = "Fizz"
            elif n % 5 == 0:
                expected = "Buzz"
            else:
                expected = str(n)
            test_cases.append(TestCase(input_args=[n], expected_output=expected))

        self.rng.shuffle(test_cases)
        return test_cases[:count]


class ReverseStringGenerator(AlgorithmicGenerator):
    """
    Generates string reversal problems.
    """

    @property
    def problem_type(self) -> str:
        return "reverse_string"

    @property
    def title(self) -> str:
        return "Reverse String"

    @property
    def description(self) -> str:
        return """Implement a function to reverse a string.

Given a string s, return it reversed.

Examples:
- "hello" -> "olleh"
- "world" -> "dlrow"

Your function must work for ANY string."""

    @property
    def function_signature(self) -> str:
        return "def reverse_string(s: str) -> str:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate string reversal test cases."""
        test_cases = []

        # Various test strings
        words = ["hello", "world", "python", "algorithm", "testing",
                 "abc", "a", "ab", "racecar", "level", "programming",
                 "data", "science", "machine", "learning"]

        for _ in range(count):
            word = self.rng.choice(words)
            # Sometimes modify the word
            if self.rng.random() < 0.3:
                word = word + str(self.rng.randint(1, 9))
            test_cases.append(TestCase(
                input_args=[word],
                expected_output=word[::-1]
            ))

        return test_cases


class IsPalindromeGenerator(AlgorithmicGenerator):
    """
    Generates palindrome checking problems.
    """

    @property
    def problem_type(self) -> str:
        return "is_palindrome"

    @property
    def title(self) -> str:
        return "Is Palindrome"

    @property
    def description(self) -> str:
        return """Implement a function to check if a string is a palindrome.

A palindrome reads the same forwards and backwards.
Consider only alphanumeric characters and ignore case.

Examples:
- "racecar" -> True
- "hello" -> False
- "A man a plan a canal Panama" -> True (ignoring spaces and case)

Your function must work for ANY string."""

    @property
    def function_signature(self) -> str:
        return "def is_palindrome(s: str) -> bool:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate palindrome test cases."""
        test_cases = []

        palindromes = ["racecar", "level", "radar", "civic", "rotor",
                       "kayak", "madam", "refer", "noon", "deed",
                       "A man a plan a canal Panama", "Was it a car or a cat I saw"]
        non_palindromes = ["hello", "world", "python", "testing", "algorithm",
                          "abc", "programming", "data", "science"]

        # Mix of palindromes and non-palindromes
        num_palindromes = count // 2 + 1

        for _ in range(num_palindromes):
            s = self.rng.choice(palindromes)
            test_cases.append(TestCase(input_args=[s], expected_output=True))

        for _ in range(count - num_palindromes):
            s = self.rng.choice(non_palindromes)
            test_cases.append(TestCase(input_args=[s], expected_output=False))

        self.rng.shuffle(test_cases)
        return test_cases[:count]


class FibonacciGenerator(AlgorithmicGenerator):
    """
    Generates Fibonacci number problems.
    """

    @property
    def problem_type(self) -> str:
        return "fibonacci"

    @property
    def title(self) -> str:
        return "Fibonacci Number"

    @property
    def description(self) -> str:
        return """Implement a function to compute the nth Fibonacci number.

The Fibonacci sequence is: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

Your function must work for any non-negative integer n."""

    @property
    def function_signature(self) -> str:
        return "def fibonacci(n: int) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate Fibonacci test cases."""
        # Precompute Fibonacci numbers
        fib = [0, 1]
        for i in range(2, 30):
            fib.append(fib[-1] + fib[-2])

        test_cases = []
        # Always include base cases
        test_cases.append(TestCase(input_args=[0], expected_output=0))
        test_cases.append(TestCase(input_args=[1], expected_output=1))

        # Add more cases based on difficulty
        max_n = min(5 + difficulty * 2, 25)
        indices = list(range(2, max_n))
        self.rng.shuffle(indices)

        for n in indices[:count - 2]:
            test_cases.append(TestCase(input_args=[n], expected_output=fib[n]))

        self.rng.shuffle(test_cases)
        return test_cases[:count]


class BinarySearchGenerator(AlgorithmicGenerator):
    """
    Generates binary search problems.
    """

    @property
    def problem_type(self) -> str:
        return "binary_search"

    @property
    def title(self) -> str:
        return "Binary Search"

    @property
    def description(self) -> str:
        return """Implement binary search to find a target in a sorted array.

Given a sorted array of integers and a target value, return the index
of the target if found, or -1 if not found.

The array is sorted in ascending order and contains no duplicates.

Your function must work for any sorted array and target."""

    @property
    def function_signature(self) -> str:
        return "def binary_search(nums: list, target: int) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate binary search test cases."""
        test_cases = []

        for _ in range(count):
            # Generate sorted array
            size = self.rng.randint(5, 10 + difficulty)
            nums = sorted(self.rng.sample(range(1, 100), size))

            # Sometimes target exists, sometimes not
            if self.rng.random() < 0.6:
                # Target exists
                idx = self.rng.randint(0, len(nums) - 1)
                target = nums[idx]
                expected = idx
            else:
                # Target doesn't exist
                target = self.rng.randint(1, 100)
                while target in nums:
                    target = self.rng.randint(1, 100)
                expected = -1

            test_cases.append(TestCase(
                input_args=[nums, target],
                expected_output=expected
            ))

        return test_cases


class TwoSumGenerator(AlgorithmicGenerator):
    """
    Generates Two Sum problems.
    """

    @property
    def problem_type(self) -> str:
        return "two_sum"

    @property
    def title(self) -> str:
        return "Two Sum"

    @property
    def description(self) -> str:
        return """Given an array of integers and a target sum, return indices of two numbers that add up to target.

You may assume that each input has exactly one solution, and you may not use the same element twice.
Return the indices in any order.

Example:
- nums = [2, 7, 11, 15], target = 9 -> [0, 1] (because nums[0] + nums[1] = 2 + 7 = 9)

Your function must work for any valid input."""

    @property
    def function_signature(self) -> str:
        return "def two_sum(nums: list, target: int) -> list:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate Two Sum test cases."""
        test_cases = []

        for _ in range(count):
            # Generate array with guaranteed solution
            size = self.rng.randint(4, 8 + difficulty)

            # Pick two positions and values that sum to target
            i, j = self.rng.sample(range(size), 2)
            a = self.rng.randint(1, 50)
            b = self.rng.randint(1, 50)
            target = a + b

            # Fill rest of array with random values that don't create other solutions
            nums = [self.rng.randint(1, 100) for _ in range(size)]
            nums[i] = a
            nums[j] = b

            # Expected output (sorted indices)
            expected = sorted([i, j])

            test_cases.append(TestCase(
                input_args=[nums, target],
                expected_output=expected
            ))

        return test_cases


class MaxSubarrayGenerator(AlgorithmicGenerator):
    """
    Generates Maximum Subarray problems (Kadane's algorithm).
    """

    @property
    def problem_type(self) -> str:
        return "max_subarray"

    @property
    def title(self) -> str:
        return "Maximum Subarray Sum"

    @property
    def description(self) -> str:
        return """Find the contiguous subarray with the largest sum.

Given an integer array, find the contiguous subarray (containing at least one number)
which has the largest sum and return that sum.

Example:
- [-2, 1, -3, 4, -1, 2, 1, -5, 4] -> 6 (subarray [4, -1, 2, 1])

Your function must work for any integer array."""

    @property
    def function_signature(self) -> str:
        return "def max_subarray(nums: list) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate max subarray test cases."""
        test_cases = []

        for _ in range(count):
            size = self.rng.randint(5, 10 + difficulty)
            nums = [self.rng.randint(-10, 10) for _ in range(size)]

            # Compute expected using Kadane's algorithm
            max_sum = nums[0]
            current_sum = nums[0]
            for n in nums[1:]:
                current_sum = max(n, current_sum + n)
                max_sum = max(max_sum, current_sum)

            test_cases.append(TestCase(
                input_args=[nums],
                expected_output=max_sum
            ))

        return test_cases


class RemoveDuplicatesGenerator(AlgorithmicGenerator):
    """
    Generates remove duplicates from sorted array problems.
    """

    @property
    def problem_type(self) -> str:
        return "remove_duplicates"

    @property
    def title(self) -> str:
        return "Remove Duplicates from Sorted List"

    @property
    def description(self) -> str:
        return """Given a sorted list, remove duplicates and return the new list.

The input list is sorted in ascending order.
Return a new list with duplicates removed, maintaining sorted order.

Example:
- [1, 1, 2, 2, 3] -> [1, 2, 3]
- [1, 1, 1] -> [1]

Your function must work for any sorted list."""

    @property
    def function_signature(self) -> str:
        return "def remove_duplicates(nums: list) -> list:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        """Generate remove duplicates test cases."""
        test_cases = []

        for _ in range(count):
            # Generate sorted list with duplicates
            unique_count = self.rng.randint(3, 6 + difficulty)
            unique_values = sorted(self.rng.sample(range(1, 50), unique_count))

            # Add duplicates
            nums = []
            for v in unique_values:
                repeat = self.rng.randint(1, 3)
                nums.extend([v] * repeat)

            test_cases.append(TestCase(
                input_args=[nums],
                expected_output=unique_values
            ))

        return test_cases


# Registry of V2 generators
GENERATORS_V2 = {
    "rpn": RPNEvaluatorGenerator,
    "arithmetic": ArithmeticEvaluatorGenerator,
    "parentheses": ParenthesesValidatorGenerator,
    "fizzbuzz": FizzBuzzGenerator,
    "reverse_string": ReverseStringGenerator,
    "is_palindrome": IsPalindromeGenerator,
    "fibonacci": FibonacciGenerator,
    "binary_search": BinarySearchGenerator,
    "two_sum": TwoSumGenerator,
    "max_subarray": MaxSubarrayGenerator,
    "remove_duplicates": RemoveDuplicatesGenerator,
}


def get_generator_v2(problem_type: str, seed: Optional[int] = None) -> AlgorithmicGenerator:
    """Get a V2 generator by type."""
    if problem_type not in GENERATORS_V2:
        raise ValueError(f"Unknown type: {problem_type}. Available: {list(GENERATORS_V2.keys())}")
    return GENERATORS_V2[problem_type](seed=seed)


def get_all_generators_v2(seed: Optional[int] = None) -> dict:
    """Get all V2 generators."""
    return {name: cls(seed=seed) for name, cls in GENERATORS_V2.items()}
