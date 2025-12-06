"""Hard Algorithmic Problem Generators - LeetCode Hard Style.

These problems require advanced algorithmic thinking:
1. Dynamic Programming (LCS, Edit Distance, Knapsack)
2. Graph Algorithms (Shortest Path, Topological Sort)
3. Advanced Data Structures (Trie operations, LRU Cache logic)
4. Complex String Manipulation (Regex matching, Parsing)
5. Mathematical Reasoning (Number theory, Combinatorics)
"""

from typing import List, Optional, Tuple
from .base_v2 import AlgorithmicGenerator, TestCase
from collections import defaultdict
import heapq


class LongestCommonSubsequenceGenerator(AlgorithmicGenerator):
    """
    LCS - Classic DP problem.

    Find the length of the longest subsequence common to two strings.
    This requires 2D dynamic programming.

    Example:
        lcs("ABCDGH", "AEDFHR") -> 3 ("ADH")
        lcs("AGGTAB", "GXTXAYB") -> 4 ("GTAB")
    """

    @property
    def problem_type(self) -> str:
        return "lcs"

    @property
    def title(self) -> str:
        return "Longest Common Subsequence"

    @property
    def description(self) -> str:
        return """Find the length of the longest common subsequence of two strings.

A subsequence is a sequence that can be derived from another sequence by deleting
some or no elements without changing the order of remaining elements.

For example:
- "ACE" is a subsequence of "ABCDE"
- "AEC" is NOT a subsequence of "ABCDE" (order changed)

Given two strings s1 and s2, return the LENGTH of their longest common subsequence.
If there is no common subsequence, return 0.

Examples:
- lcs("ABCDGH", "AEDFHR") -> 3 (subsequence "ADH")
- lcs("ABC", "DEF") -> 0 (no common subsequence)

Your function must work for any two strings."""

    @property
    def function_signature(self) -> str:
        return "def lcs(s1: str, s2: str) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        test_cases = []

        for _ in range(count):
            # Generate strings based on difficulty
            if difficulty <= 3:
                len1, len2 = self.rng.randint(3, 6), self.rng.randint(3, 6)
            elif difficulty <= 6:
                len1, len2 = self.rng.randint(5, 10), self.rng.randint(5, 10)
            else:
                len1, len2 = self.rng.randint(8, 15), self.rng.randint(8, 15)

            # Generate strings with some common characters
            chars = "ABCDEFGH"
            s1 = "".join(self.rng.choice(chars) for _ in range(len1))
            s2 = "".join(self.rng.choice(chars) for _ in range(len2))

            # Compute LCS using DP
            expected = self._compute_lcs(s1, s2)
            test_cases.append(TestCase(input_args=[s1, s2], expected_output=expected))

        return test_cases

    def _compute_lcs(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]


class EditDistanceGenerator(AlgorithmicGenerator):
    """
    Edit Distance (Levenshtein Distance) - Classic DP problem.

    Find minimum number of operations to convert one string to another.
    Operations: insert, delete, replace.

    Example:
        edit_distance("horse", "ros") -> 3
        edit_distance("intention", "execution") -> 5
    """

    @property
    def problem_type(self) -> str:
        return "edit_distance"

    @property
    def title(self) -> str:
        return "Edit Distance"

    @property
    def description(self) -> str:
        return """Calculate the minimum number of operations to convert word1 to word2.

You have three operations:
1. Insert a character
2. Delete a character
3. Replace a character

Examples:
- edit_distance("horse", "ros") -> 3
  - horse -> rorse (replace 'h' with 'r')
  - rorse -> rose (remove 'r')
  - rose -> ros (remove 'e')

- edit_distance("", "abc") -> 3 (insert 3 characters)

Your function must work for any two strings."""

    @property
    def function_signature(self) -> str:
        return "def edit_distance(word1: str, word2: str) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        test_cases = []

        words = ["horse", "ros", "intention", "execution", "cat", "cut",
                 "abc", "def", "kitten", "sitting", "saturday", "sunday",
                 "algorithm", "altruistic", "plasma", "altruism"]

        for _ in range(count):
            if difficulty <= 3:
                len1, len2 = self.rng.randint(2, 4), self.rng.randint(2, 4)
            elif difficulty <= 6:
                len1, len2 = self.rng.randint(4, 7), self.rng.randint(4, 7)
            else:
                len1, len2 = self.rng.randint(6, 10), self.rng.randint(6, 10)

            # Either use preset words or generate random
            if self.rng.random() < 0.5:
                word1 = self.rng.choice(words)[:len1]
                word2 = self.rng.choice(words)[:len2]
            else:
                chars = "abcdefgh"
                word1 = "".join(self.rng.choice(chars) for _ in range(len1))
                word2 = "".join(self.rng.choice(chars) for _ in range(len2))

            expected = self._compute_edit_distance(word1, word2)
            test_cases.append(TestCase(input_args=[word1, word2], expected_output=expected))

        return test_cases

    def _compute_edit_distance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]


class KnapsackGenerator(AlgorithmicGenerator):
    """
    0/1 Knapsack Problem - Classic DP.

    Given weights and values of items, find maximum value that fits in capacity.

    Example:
        knapsack(capacity=50, weights=[10,20,30], values=[60,100,120]) -> 220
    """

    @property
    def problem_type(self) -> str:
        return "knapsack"

    @property
    def title(self) -> str:
        return "0/1 Knapsack"

    @property
    def description(self) -> str:
        return """Solve the 0/1 Knapsack problem.

Given:
- A knapsack with maximum capacity W
- N items, each with a weight and value
- Each item can only be taken once (0/1 choice)

Find the maximum total value that can fit in the knapsack.

Example:
- capacity=50, weights=[10,20,30], values=[60,100,120]
- Take items with weights 20 and 30 (total weight 50)
- Total value = 100 + 120 = 220

Your function must work for any valid input."""

    @property
    def function_signature(self) -> str:
        return "def knapsack(capacity: int, weights: list, values: list) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        test_cases = []

        for _ in range(count):
            if difficulty <= 3:
                n_items = self.rng.randint(3, 4)
                capacity = self.rng.randint(20, 40)
            elif difficulty <= 6:
                n_items = self.rng.randint(4, 6)
                capacity = self.rng.randint(30, 60)
            else:
                n_items = self.rng.randint(5, 8)
                capacity = self.rng.randint(50, 100)

            weights = [self.rng.randint(5, 30) for _ in range(n_items)]
            values = [self.rng.randint(10, 100) for _ in range(n_items)]

            expected = self._compute_knapsack(capacity, weights, values)
            test_cases.append(TestCase(
                input_args=[capacity, weights, values],
                expected_output=expected
            ))

        return test_cases

    def _compute_knapsack(self, capacity: int, weights: List[int], values: List[int]) -> int:
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
                else:
                    dp[i][w] = dp[i-1][w]

        return dp[n][capacity]


class LongestIncreasingSubsequenceGenerator(AlgorithmicGenerator):
    """
    LIS - Find length of longest increasing subsequence.

    Requires O(n log n) solution for hard difficulty.

    Example:
        lis([10,9,2,5,3,7,101,18]) -> 4 ([2,3,7,101])
    """

    @property
    def problem_type(self) -> str:
        return "lis"

    @property
    def title(self) -> str:
        return "Longest Increasing Subsequence"

    @property
    def description(self) -> str:
        return """Find the length of the longest strictly increasing subsequence.

A subsequence is derived by deleting some or no elements without changing order.

Examples:
- lis([10,9,2,5,3,7,101,18]) -> 4 (subsequence [2,3,7,101] or [2,5,7,101])
- lis([0,1,0,3,2,3]) -> 4 (subsequence [0,1,2,3])
- lis([7,7,7,7,7]) -> 1 (all same, so just one element)

Your function must work for any integer array."""

    @property
    def function_signature(self) -> str:
        return "def lis(nums: list) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        test_cases = []

        for _ in range(count):
            if difficulty <= 3:
                size = self.rng.randint(5, 8)
            elif difficulty <= 6:
                size = self.rng.randint(8, 12)
            else:
                size = self.rng.randint(12, 20)

            nums = [self.rng.randint(0, 50) for _ in range(size)]
            expected = self._compute_lis(nums)
            test_cases.append(TestCase(input_args=[nums], expected_output=expected))

        return test_cases

    def _compute_lis(self, nums: List[int]) -> int:
        if not nums:
            return 0

        # O(n^2) DP solution (simple and correct)
        n = len(nums)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)


class CoinChangeGenerator(AlgorithmicGenerator):
    """
    Coin Change - Minimum coins to make amount.

    Classic unbounded knapsack / DP problem.

    Example:
        coin_change([1,2,5], 11) -> 3 (5+5+1)
        coin_change([2], 3) -> -1 (impossible)
    """

    @property
    def problem_type(self) -> str:
        return "coin_change"

    @property
    def title(self) -> str:
        return "Coin Change"

    @property
    def description(self) -> str:
        return """Find the minimum number of coins needed to make up an amount.

Given an array of coin denominations and a target amount, return the fewest
number of coins needed. If impossible, return -1.

You have infinite supply of each coin denomination.

Examples:
- coin_change([1,2,5], 11) -> 3 (5+5+1)
- coin_change([2], 3) -> -1 (impossible with only 2s)
- coin_change([1], 0) -> 0 (no coins needed)

Your function must work for any valid input."""

    @property
    def function_signature(self) -> str:
        return "def coin_change(coins: list, amount: int) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        test_cases = []

        coin_sets = [
            [1, 2, 5],
            [1, 5, 10, 25],
            [2, 5, 10],
            [1, 3, 4],
            [1, 2, 3],
            [2, 3, 7],
        ]

        for _ in range(count):
            coins = self.rng.choice(coin_sets)

            if difficulty <= 3:
                amount = self.rng.randint(5, 15)
            elif difficulty <= 6:
                amount = self.rng.randint(10, 30)
            else:
                amount = self.rng.randint(20, 50)

            expected = self._compute_coin_change(coins, amount)
            test_cases.append(TestCase(
                input_args=[coins, amount],
                expected_output=expected
            ))

        return test_cases

    def _compute_coin_change(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0

        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i and dp[i - coin] != float('inf'):
                    dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] != float('inf') else -1


class WordBreakGenerator(AlgorithmicGenerator):
    """
    Word Break - Can string be segmented into dictionary words?

    DP with string matching.

    Example:
        word_break("leetcode", ["leet", "code"]) -> True
        word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) -> False
    """

    @property
    def problem_type(self) -> str:
        return "word_break"

    @property
    def title(self) -> str:
        return "Word Break"

    @property
    def description(self) -> str:
        return """Determine if a string can be segmented into dictionary words.

Given a string s and a dictionary of words, return True if s can be segmented
into a space-separated sequence of one or more dictionary words.

The same word in the dictionary may be reused multiple times.

Examples:
- word_break("leetcode", ["leet", "code"]) -> True ("leet code")
- word_break("applepenapple", ["apple", "pen"]) -> True ("apple pen apple")
- word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) -> False

Your function must work for any valid input."""

    @property
    def function_signature(self) -> str:
        return "def word_break(s: str, word_dict: list) -> bool:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        test_cases = []

        # Predefined cases that are known to work
        cases = [
            ("leetcode", ["leet", "code"], True),
            ("applepenapple", ["apple", "pen"], True),
            ("catsandog", ["cats", "dog", "sand", "and", "cat"], False),
            ("aaaaaaa", ["aaa", "aaaa"], True),
            ("cars", ["car", "ca", "rs"], True),
            ("abcd", ["a", "abc", "b", "cd"], True),
            ("goalspecial", ["go", "goal", "goals", "special"], True),
        ]

        # Always include some known cases
        for s, words, expected in cases[:count]:
            test_cases.append(TestCase(input_args=[s, words], expected_output=expected))

        return test_cases[:count]


class MergeIntervalsGenerator(AlgorithmicGenerator):
    """
    Merge Overlapping Intervals.

    Example:
        merge_intervals([[1,3],[2,6],[8,10],[15,18]]) -> [[1,6],[8,10],[15,18]]
    """

    @property
    def problem_type(self) -> str:
        return "merge_intervals"

    @property
    def title(self) -> str:
        return "Merge Intervals"

    @property
    def description(self) -> str:
        return """Merge all overlapping intervals.

Given an array of intervals where intervals[i] = [start, end], merge all
overlapping intervals and return an array of non-overlapping intervals
that cover all the intervals in the input.

Examples:
- merge_intervals([[1,3],[2,6],[8,10],[15,18]]) -> [[1,6],[8,10],[15,18]]
- merge_intervals([[1,4],[4,5]]) -> [[1,5]]
- merge_intervals([[1,4],[0,4]]) -> [[0,4]]

Your function must work for any valid input."""

    @property
    def function_signature(self) -> str:
        return "def merge_intervals(intervals: list) -> list:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        test_cases = []

        for _ in range(count):
            if difficulty <= 3:
                n = self.rng.randint(2, 4)
            elif difficulty <= 6:
                n = self.rng.randint(4, 6)
            else:
                n = self.rng.randint(5, 8)

            intervals = []
            for _ in range(n):
                start = self.rng.randint(0, 20)
                end = start + self.rng.randint(1, 5)
                intervals.append([start, end])

            expected = self._merge_intervals(intervals)
            test_cases.append(TestCase(
                input_args=[intervals],
                expected_output=expected
            ))

        return test_cases

    def _merge_intervals(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []

        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]

        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        return merged


class MedianTwoSortedArraysGenerator(AlgorithmicGenerator):
    """
    Find Median of Two Sorted Arrays - LeetCode Hard #4.

    Requires O(log(min(m,n))) solution for optimal.

    Example:
        find_median([1,3], [2]) -> 2.0
        find_median([1,2], [3,4]) -> 2.5
    """

    @property
    def problem_type(self) -> str:
        return "median_sorted_arrays"

    @property
    def title(self) -> str:
        return "Median of Two Sorted Arrays"

    @property
    def description(self) -> str:
        return """Find the median of two sorted arrays.

Given two sorted arrays nums1 and nums2, return the median of the combined sorted array.

The median is the middle value. If the combined array has even length,
the median is the average of the two middle values.

Examples:
- find_median([1,3], [2]) -> 2.0 (merged: [1,2,3], median is 2)
- find_median([1,2], [3,4]) -> 2.5 (merged: [1,2,3,4], median is (2+3)/2)

Your function must work for any two sorted arrays."""

    @property
    def function_signature(self) -> str:
        return "def find_median(nums1: list, nums2: list) -> float:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        test_cases = []

        for _ in range(count):
            if difficulty <= 3:
                size1, size2 = self.rng.randint(1, 3), self.rng.randint(1, 3)
            elif difficulty <= 6:
                size1, size2 = self.rng.randint(2, 5), self.rng.randint(2, 5)
            else:
                size1, size2 = self.rng.randint(3, 7), self.rng.randint(3, 7)

            nums1 = sorted(self.rng.sample(range(1, 50), size1))
            nums2 = sorted(self.rng.sample(range(1, 50), size2))

            # Compute median
            merged = sorted(nums1 + nums2)
            n = len(merged)
            if n % 2 == 1:
                expected = float(merged[n // 2])
            else:
                expected = (merged[n // 2 - 1] + merged[n // 2]) / 2.0

            test_cases.append(TestCase(
                input_args=[nums1, nums2],
                expected_output=expected
            ))

        return test_cases


class TrappingRainWaterGenerator(AlgorithmicGenerator):
    """
    Trapping Rain Water - LeetCode Hard #42.

    Classic two-pointer or DP problem.

    Example:
        trap([0,1,0,2,1,0,1,3,2,1,2,1]) -> 6
    """

    @property
    def problem_type(self) -> str:
        return "trapping_rain_water"

    @property
    def title(self) -> str:
        return "Trapping Rain Water"

    @property
    def description(self) -> str:
        return """Calculate how much rain water can be trapped.

Given n non-negative integers representing an elevation map where the width
of each bar is 1, compute how much water it can trap after raining.

Example:
- trap([0,1,0,2,1,0,1,3,2,1,2,1]) -> 6
  The elevation map looks like:
       #
   #   ##
  _##_####

  Water fills the valleys.

Your function must work for any elevation map."""

    @property
    def function_signature(self) -> str:
        return "def trap(height: list) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        test_cases = []

        for _ in range(count):
            if difficulty <= 3:
                size = self.rng.randint(5, 8)
                max_h = 4
            elif difficulty <= 6:
                size = self.rng.randint(8, 12)
                max_h = 6
            else:
                size = self.rng.randint(10, 15)
                max_h = 8

            height = [self.rng.randint(0, max_h) for _ in range(size)]
            expected = self._compute_trap(height)
            test_cases.append(TestCase(input_args=[height], expected_output=expected))

        return test_cases

    def _compute_trap(self, height: List[int]) -> int:
        if not height:
            return 0

        n = len(height)
        left_max = [0] * n
        right_max = [0] * n

        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])

        right_max[n-1] = height[n-1]
        for i in range(n-2, -1, -1):
            right_max[i] = max(right_max[i+1], height[i])

        water = 0
        for i in range(n):
            water += min(left_max[i], right_max[i]) - height[i]

        return water


class NQueensCountGenerator(AlgorithmicGenerator):
    """
    N-Queens - Count number of solutions.

    Classic backtracking problem.

    Example:
        n_queens(4) -> 2
        n_queens(8) -> 92
    """

    @property
    def problem_type(self) -> str:
        return "n_queens"

    @property
    def title(self) -> str:
        return "N-Queens Count"

    @property
    def description(self) -> str:
        return """Count the number of solutions to the N-Queens puzzle.

The N-Queens puzzle is placing N chess queens on an N×N chessboard such
that no two queens attack each other.

Queens can attack horizontally, vertically, and diagonally.

Examples:
- n_queens(1) -> 1
- n_queens(4) -> 2
- n_queens(8) -> 92

Your function must work for any positive integer n."""

    @property
    def function_signature(self) -> str:
        return "def n_queens(n: int) -> int:"

    def generate_test_cases(self, difficulty: int, count: int = 5) -> List[TestCase]:
        # Precomputed N-Queens solutions
        solutions = {1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724}

        test_cases = []

        if difficulty <= 3:
            ns = [1, 2, 3, 4]
        elif difficulty <= 6:
            ns = [1, 4, 5, 6]
        else:
            ns = [4, 5, 6, 7, 8]

        for _ in range(count):
            n = self.rng.choice(ns)
            test_cases.append(TestCase(input_args=[n], expected_output=solutions[n]))

        return test_cases


# Registry of hard generators
GENERATORS_HARD = {
    "lcs": LongestCommonSubsequenceGenerator,
    "edit_distance": EditDistanceGenerator,
    "knapsack": KnapsackGenerator,
    "lis": LongestIncreasingSubsequenceGenerator,
    "coin_change": CoinChangeGenerator,
    "word_break": WordBreakGenerator,
    "merge_intervals": MergeIntervalsGenerator,
    "median_sorted_arrays": MedianTwoSortedArraysGenerator,
    "trapping_rain_water": TrappingRainWaterGenerator,
    "n_queens": NQueensCountGenerator,
}


def get_hard_generator(problem_type: str, seed: int = None):
    """Get a hard problem generator by type."""
    if problem_type not in GENERATORS_HARD:
        raise ValueError(f"Unknown type: {problem_type}. Available: {list(GENERATORS_HARD.keys())}")
    return GENERATORS_HARD[problem_type](seed=seed)


def get_all_hard_generators(seed: int = None) -> dict:
    """Get all hard problem generators."""
    return {name: cls(seed=seed) for name, cls in GENERATORS_HARD.items()}
