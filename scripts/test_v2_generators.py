#!/usr/bin/env python
"""Test the V2 algorithmic problem generators.

This script validates that:
1. Problems have MULTIPLE test cases
2. Functions take INPUT arguments
3. Test cases require actual algorithms (can't just hardcode)
"""

import sys
sys.path.insert(0, ".")

from axiom.procedural.base_v2 import AlgorithmicProblem, TestCase
from axiom.procedural.generators_v2 import (
    RPNEvaluatorGenerator,
    ArithmeticEvaluatorGenerator,
    ParenthesesValidatorGenerator,
    get_all_generators_v2,
)


def test_rpn_generator():
    """Test the RPN evaluator generator."""
    print("\n" + "="*60)
    print("TEST: RPN Evaluator Generator")
    print("="*60)

    gen = RPNEvaluatorGenerator(seed=42)
    problem = gen.generate(difficulty=5, num_test_cases=5)

    print(f"\nProblem: {problem.title}")
    print(f"Function: {problem.function_signature}")
    print(f"Number of test cases: {len(problem.test_cases)}")

    print("\nTest cases:")
    for i, tc in enumerate(problem.test_cases):
        print(f"  {i+1}. evaluate_rpn({tc.input_args[0]!r}) -> {tc.expected_output}")

    # Verify test cases have DIFFERENT outputs (can't hardcode)
    outputs = [tc.expected_output for tc in problem.test_cases]
    unique_outputs = len(set(outputs))
    print(f"\nUnique outputs: {unique_outputs}/{len(outputs)}")

    if unique_outputs < 3:
        print("WARNING: Too few unique outputs - hardcoding might work!")
    else:
        print("GOOD: Diverse outputs - requires real algorithm")

    # Show the prompt
    print("\n" + "-"*40)
    print("PROMPT PREVIEW:")
    print("-"*40)
    print(problem.to_prompt()[:500] + "...")


def test_arithmetic_generator():
    """Test the arithmetic evaluator generator."""
    print("\n" + "="*60)
    print("TEST: Arithmetic Evaluator Generator")
    print("="*60)

    gen = ArithmeticEvaluatorGenerator(seed=42)
    problem = gen.generate(difficulty=7, num_test_cases=5)

    print(f"\nProblem: {problem.title}")
    print(f"Function: {problem.function_signature}")
    print(f"Number of test cases: {len(problem.test_cases)}")

    print("\nTest cases:")
    for i, tc in enumerate(problem.test_cases):
        print(f"  {i+1}. evaluate({tc.input_args[0]!r}) -> {tc.expected_output}")


def test_parentheses_generator():
    """Test the parentheses validator generator."""
    print("\n" + "="*60)
    print("TEST: Parentheses Validator Generator")
    print("="*60)

    gen = ParenthesesValidatorGenerator(seed=42)
    problem = gen.generate(difficulty=5, num_test_cases=6)

    print(f"\nProblem: {problem.title}")
    print(f"Function: {problem.function_signature}")
    print(f"Number of test cases: {len(problem.test_cases)}")

    print("\nTest cases:")
    for i, tc in enumerate(problem.test_cases):
        print(f"  {i+1}. is_valid({tc.input_args[0]!r}) -> {tc.expected_output}")

    # Count True vs False
    true_count = sum(1 for tc in problem.test_cases if tc.expected_output)
    false_count = len(problem.test_cases) - true_count
    print(f"\nTrue cases: {true_count}, False cases: {false_count}")


def test_hardcoding_prevention():
    """Demonstrate that hardcoding can't pass all test cases."""
    print("\n" + "="*60)
    print("TEST: Hardcoding Prevention")
    print("="*60)

    gen = RPNEvaluatorGenerator(seed=42)
    problem = gen.generate(difficulty=5, num_test_cases=5)

    print("\nTest cases:")
    for tc in problem.test_cases:
        print(f"  evaluate_rpn({tc.input_args[0]!r}) -> {tc.expected_output}")

    # Simulate a hardcoded solution
    first_answer = problem.test_cases[0].expected_output
    print(f"\nHardcoded solution: return {first_answer}")

    passed = sum(1 for tc in problem.test_cases if tc.expected_output == first_answer)
    print(f"Would pass: {passed}/{len(problem.test_cases)} test cases")

    if passed < len(problem.test_cases):
        print("GOOD: Hardcoding fails - model must implement algorithm!")
    else:
        print("WARNING: All outputs are the same - need more diversity")


def test_correct_solution():
    """Test that a correct algorithm would pass all test cases."""
    print("\n" + "="*60)
    print("TEST: Correct Algorithm Verification")
    print("="*60)

    gen = RPNEvaluatorGenerator(seed=42)
    problem = gen.generate(difficulty=5, num_test_cases=5)

    # Correct RPN evaluator
    def evaluate_rpn(expression: str) -> int:
        stack = []
        for token in expression.split():
            if token.lstrip('-').isdigit():
                stack.append(int(token))
            else:
                b = stack.pop()
                a = stack.pop()
                if token == '+':
                    stack.append(a + b)
                elif token == '-':
                    stack.append(a - b)
                elif token == '*':
                    stack.append(a * b)
        return stack[0]

    print("\nTesting correct RPN evaluator:")
    all_passed = True
    for tc in problem.test_cases:
        expr = tc.input_args[0]
        expected = tc.expected_output
        actual = evaluate_rpn(expr)
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            all_passed = False
        print(f"  {status}: evaluate_rpn({expr!r}) = {actual} (expected {expected})")

    if all_passed:
        print("\nSUCCESS: Correct algorithm passes all test cases!")
    else:
        print("\nFAILURE: Something wrong with test cases or evaluator")


def main():
    print("="*60)
    print("V2 GENERATOR VALIDATION")
    print("="*60)
    print("\nThis test validates the new problem design that:")
    print("1. Has MULTIPLE test cases per problem")
    print("2. Functions take INPUT arguments")
    print("3. Can't be solved by hardcoding")

    test_rpn_generator()
    test_arithmetic_generator()
    test_parentheses_generator()
    test_hardcoding_prevention()
    test_correct_solution()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
