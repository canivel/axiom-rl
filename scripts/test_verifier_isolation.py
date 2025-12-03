from axiom.verifier import TestHarness
from axiom.problems.dataset import ProblemDataset

def main():
    dataset = ProblemDataset()
    problem = dataset.get_problem("two_sum")
    harness = TestHarness()

    print(f"Testing Verifier on problem: {problem.title}")

    # 1. Correct Solution
    correct_code = """
from typing import List
def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""
    result = harness.verify(correct_code, problem)
    print(f"Correct Solution: {'PASSED' if result.passed_count == result.total_count else 'FAILED'}")

    # 2. Incorrect Solution
    incorrect_code = """
from typing import List
def two_sum(nums: List[int], target: int) -> List[int]:
    return [0, 0]
"""
    result = harness.verify(incorrect_code, problem)
    print(f"Incorrect Solution: {'PASSED' if result.passed_count == result.total_count else 'FAILED'}")

if __name__ == "__main__":
    main()
