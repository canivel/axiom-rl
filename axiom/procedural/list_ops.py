"""List operations generator.

Generates problems involving list manipulations with verifiable outputs:
- Sorting
- Filtering
- Transformations
- Aggregations

Each operation has a clear algorithmic solution with perfect ground truth.
"""

from typing import Any, Optional

from .base import ProceduralGenerator


class ListSortGenerator(ProceduralGenerator):
    """
    Generates list sorting problems with various criteria.

    Difficulty scales with:
    - List length (5-100 elements)
    - Element types (integers, tuples)
    - Sorting criteria (simple to complex)
    """

    @property
    def problem_type(self) -> str:
        return "list_sort"

    @property
    def title(self) -> str:
        return "Custom List Sort"

    @property
    def description_template(self) -> str:
        return """Sort a list of integers according to a custom criterion.

The sorting criterion is specified in each problem instance.
Return the sorted list.

Common criteria include:
- Sort by absolute value
- Sort by digit sum
- Sort by number of divisors
- Sort even numbers before odd
- Sort by distance from a target value"""

    @property
    def function_signature(self) -> str:
        return "def custom_sort(nums: list[int], criterion: str) -> list[int]:"

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.criteria = [
            ("absolute", lambda x: abs(x)),
            ("digit_sum", lambda x: sum(int(d) for d in str(abs(x)))),
            ("even_first", lambda x: (x % 2, x)),
            ("odd_first", lambda x: (1 - x % 2, x)),
            ("reverse", lambda x: -x),
        ]

    def generate_instance(self, difficulty: int = 5) -> tuple[dict, list[int]]:
        """
        Generate a list sorting problem.

        Difficulty mapping:
        1-2: Small lists (5-10), simple criteria
        3-4: Medium lists (10-20), basic criteria
        5-6: Larger lists (15-30), mixed criteria
        7-8: Large lists (25-50), complex criteria
        9-10: Very large lists (40-100), all criteria
        """
        if difficulty <= 2:
            length = self.rng.randint(5, 10)
            max_num = 20
            criteria_pool = self.criteria[:2]
        elif difficulty <= 4:
            length = self.rng.randint(10, 20)
            max_num = 50
            criteria_pool = self.criteria[:3]
        elif difficulty <= 6:
            length = self.rng.randint(15, 30)
            max_num = 100
            criteria_pool = self.criteria[:4]
        elif difficulty <= 8:
            length = self.rng.randint(25, 50)
            max_num = 200
            criteria_pool = self.criteria
        else:
            length = self.rng.randint(40, 100)
            max_num = 500
            criteria_pool = self.criteria

        # Generate list with some negative numbers at higher difficulties
        nums = []
        for _ in range(length):
            num = self.rng.randint(1, max_num)
            if difficulty >= 5 and self.rng.random() < 0.3:
                num = -num
            nums.append(num)

        # Select criterion
        criterion_name, criterion_func = self.rng.choice(criteria_pool)

        # Compute expected output
        expected = sorted(nums, key=criterion_func)

        return {"nums": nums, "criterion": criterion_name}, expected

    def verify(self, input_data: dict, output: list, expected: list) -> bool:
        """Verify the output matches expected."""
        return output == expected


class ListFilterGenerator(ProceduralGenerator):
    """
    Generates list filtering problems.

    Difficulty scales with:
    - List length
    - Filter complexity
    - Multiple conditions
    """

    @property
    def problem_type(self) -> str:
        return "list_filter"

    @property
    def title(self) -> str:
        return "Filter List"

    @property
    def description_template(self) -> str:
        return """Filter a list of integers based on a specified condition.

Return a new list containing only elements that satisfy the condition.
Maintain the original order of elements.

Common conditions include:
- Keep only even/odd numbers
- Keep numbers greater than threshold
- Keep numbers divisible by N
- Keep prime numbers
- Keep numbers with digit sum > N"""

    @property
    def function_signature(self) -> str:
        return "def filter_list(nums: list[int], condition: str, param: int) -> list[int]:"

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def _is_prime(self, n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def _digit_sum(self, n: int) -> int:
        """Calculate sum of digits."""
        return sum(int(d) for d in str(abs(n)))

    def generate_instance(self, difficulty: int = 5) -> tuple[dict, list[int]]:
        """Generate a list filtering problem."""
        if difficulty <= 2:
            length = self.rng.randint(5, 10)
            max_num = 20
            conditions = ["even", "odd"]
        elif difficulty <= 4:
            length = self.rng.randint(10, 20)
            max_num = 50
            conditions = ["even", "odd", "greater_than", "divisible_by"]
        elif difficulty <= 6:
            length = self.rng.randint(15, 30)
            max_num = 100
            conditions = ["even", "odd", "greater_than", "divisible_by", "prime"]
        elif difficulty <= 8:
            length = self.rng.randint(25, 50)
            max_num = 200
            conditions = ["even", "odd", "greater_than", "divisible_by", "prime", "digit_sum_gt"]
        else:
            length = self.rng.randint(40, 100)
            max_num = 500
            conditions = ["even", "odd", "greater_than", "divisible_by", "prime", "digit_sum_gt"]

        # Generate list
        nums = [self.rng.randint(1, max_num) for _ in range(length)]

        # Select condition
        condition = self.rng.choice(conditions)

        # Determine parameter and filter function
        if condition == "even":
            param = 0
            filter_func = lambda x: x % 2 == 0
        elif condition == "odd":
            param = 0
            filter_func = lambda x: x % 2 == 1
        elif condition == "greater_than":
            param = self.rng.randint(max_num // 4, max_num // 2)
            filter_func = lambda x: x > param
        elif condition == "divisible_by":
            param = self.rng.randint(2, 7)
            filter_func = lambda x: x % param == 0
        elif condition == "prime":
            param = 0
            filter_func = self._is_prime
        elif condition == "digit_sum_gt":
            param = self.rng.randint(5, 15)
            filter_func = lambda x: self._digit_sum(x) > param
        else:
            param = 0
            filter_func = lambda x: True

        # Compute expected output
        expected = [x for x in nums if filter_func(x)]

        return {"nums": nums, "condition": condition, "param": param}, expected

    def verify(self, input_data: dict, output: list, expected: list) -> bool:
        """Verify the output matches expected."""
        return output == expected


class ListAggregateGenerator(ProceduralGenerator):
    """
    Generates list aggregation problems.

    Problems like: find second largest, count inversions, etc.
    """

    @property
    def problem_type(self) -> str:
        return "list_aggregate"

    @property
    def title(self) -> str:
        return "List Aggregation"

    @property
    def description_template(self) -> str:
        return """Compute an aggregate value from a list of integers.

Common aggregations include:
- Find the Nth largest/smallest element
- Count elements satisfying a condition
- Find the most frequent element
- Calculate running statistics
- Find longest increasing subsequence length"""

    @property
    def function_signature(self) -> str:
        return "def aggregate(nums: list[int], operation: str, param: int) -> int:"

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def generate_instance(self, difficulty: int = 5) -> tuple[dict, int]:
        """Generate a list aggregation problem."""
        if difficulty <= 2:
            length = self.rng.randint(5, 10)
            max_num = 20
            operations = ["second_max", "count_even"]
        elif difficulty <= 4:
            length = self.rng.randint(10, 20)
            max_num = 50
            operations = ["second_max", "count_even", "nth_largest", "count_greater"]
        elif difficulty <= 6:
            length = self.rng.randint(15, 30)
            max_num = 100
            operations = ["second_max", "nth_largest", "count_greater", "mode"]
        elif difficulty <= 8:
            length = self.rng.randint(25, 50)
            max_num = 200
            operations = ["nth_largest", "count_greater", "mode", "count_unique"]
        else:
            length = self.rng.randint(40, 100)
            max_num = 500
            operations = ["nth_largest", "mode", "count_unique", "range"]

        # Generate list
        nums = [self.rng.randint(1, max_num) for _ in range(length)]

        # Select operation
        operation = self.rng.choice(operations)

        # Compute expected output based on operation
        if operation == "second_max":
            param = 2
            sorted_unique = sorted(set(nums), reverse=True)
            expected = sorted_unique[1] if len(sorted_unique) > 1 else sorted_unique[0]
        elif operation == "nth_largest":
            param = self.rng.randint(1, min(5, length))
            sorted_nums = sorted(nums, reverse=True)
            expected = sorted_nums[param - 1]
        elif operation == "count_even":
            param = 0
            expected = sum(1 for x in nums if x % 2 == 0)
        elif operation == "count_greater":
            param = self.rng.randint(max_num // 4, max_num // 2)
            expected = sum(1 for x in nums if x > param)
        elif operation == "mode":
            param = 0
            from collections import Counter
            expected = Counter(nums).most_common(1)[0][0]
        elif operation == "count_unique":
            param = 0
            expected = len(set(nums))
        elif operation == "range":
            param = 0
            expected = max(nums) - min(nums)
        else:
            param = 0
            expected = sum(nums)

        return {"nums": nums, "operation": operation, "param": param}, expected

    def verify(self, input_data: dict, output: int, expected: int) -> bool:
        """Verify the output matches expected."""
        return output == expected
