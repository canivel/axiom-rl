#!/usr/bin/env python3
"""
Generate teacher traces for hard problems using Claude or Gemini API.

This script generates high-quality reasoning traces for LeetCode-hard style
problems that the small model struggles with after GRPO training:
- Coin Change (1D DP with coin iteration)
- Knapsack (1D DP with weight iteration)
- N-Queens (backtracking with constraint checking)

The traces include detailed <think> reasoning that can be used for
SFT distillation to bootstrap the model before GRPO refinement.

Usage:
    uv run python scripts/generate_hard_traces.py --problems coin_change knapsack
    uv run python scripts/generate_hard_traces.py --teacher gemini --all --count 5
    uv run python scripts/generate_hard_traces.py --teacher claude --problems n_queens
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.procedural.generators_hard import get_all_hard_generators, GENERATORS_HARD

# Problems that GRPO failed to solve (need teacher distillation)
WEAK_PROBLEMS = ["coin_change", "knapsack", "n_queens"]

# All hard problems for comparison
ALL_HARD_PROBLEMS = list(GENERATORS_HARD.keys())


def verify_solution(code: str, problem) -> tuple[bool, int, int]:
    """
    Verify solution against all test cases.

    Returns:
        (all_passed, num_passed, total_tests)
    """
    func_name = problem.function_name
    namespace = {}

    try:
        exec(code, namespace)
    except Exception as e:
        return False, 0, len(problem.test_cases)

    # Find the function
    if func_name not in namespace:
        funcs = [k for k, v in namespace.items() if callable(v) and not k.startswith("_")]
        if funcs:
            func_name = funcs[0]
        else:
            return False, 0, len(problem.test_cases)

    func = namespace[func_name]
    passed = 0
    total = len(problem.test_cases)

    for tc in problem.test_cases:
        try:
            result = func(*tc.input_args)
            if result == tc.expected_output:
                passed += 1
        except Exception:
            pass

    return passed == total, passed, total


def get_teacher_client(teacher: str):
    """Get the appropriate teacher client based on teacher name."""
    if teacher.lower() == "gemini":
        from axiom.coldstart.gemini_client import GeminiClient
        try:
            return GeminiClient(), "gemini-2.5-flash"
        except ValueError as e:
            print(f"Error: {e}")
            print("Please set GEMINI_API_KEY in your .env file")
            return None, None
    elif teacher.lower() == "claude":
        from axiom.coldstart.claude_client import ClaudeClient
        try:
            return ClaudeClient(), "claude-sonnet-4"
        except ValueError as e:
            print(f"Error: {e}")
            print("Please set ANTHROPIC_API_KEY in your .env file")
            return None, None
    else:
        print(f"Error: Unknown teacher '{teacher}'. Use 'gemini' or 'claude'")
        return None, None


def generate_traces(
    problem_types: list[str],
    count_per_type: int,
    difficulties: list[int],
    output_path: Path,
    teacher: str = "gemini",
    seed: int = 42,
    delay: float = 1.0,
    verbose: bool = False,
):
    """Generate teacher traces for hard problems."""

    # Initialize teacher client
    client, model_name = get_teacher_client(teacher)
    if client is None:
        return []

    print(f"Using teacher model: {model_name}")

    # Initialize generators
    generators = get_all_hard_generators(seed=seed)
    generators = {k: v for k, v in generators.items() if k in problem_types}

    if not generators:
        print(f"Error: No valid problem types from {problem_types}")
        print(f"Available: {ALL_HARD_PROBLEMS}")
        return []

    traces = []
    total = len(problem_types) * count_per_type * len(difficulties)
    current = 0

    print(f"Generating {total} traces for {len(problem_types)} problem types...")
    print(f"Difficulties: {difficulties}")
    print()

    for problem_type in problem_types:
        generator = generators[problem_type]

        print(f"\n{'='*60}")
        print(f"Problem Type: {problem_type.upper()}")
        print(f"{'='*60}")

        for difficulty in difficulties:
            for i in range(count_per_type):
                current += 1

                # Generate problem
                problem = generator.generate(difficulty=difficulty, num_test_cases=5)

                print(f"\n[{current}/{total}] {problem.title} (difficulty={difficulty})")

                try:
                    # Generate Claude trace
                    response = client.generate_reasoning_trace(
                        problem_title=problem.title,
                        problem_description=problem.description,
                        function_signature=problem.function_signature,
                        temperature=0.3,  # Lower temp for correctness
                    )

                    # Verify the solution
                    all_passed, passed, total_tests = verify_solution(response.code, problem)

                    status = "VERIFIED" if all_passed else f"FAILED ({passed}/{total_tests})"
                    print(f"  Status: {status}")

                    if verbose and response.thinking:
                        print(f"  Thinking: {response.thinking[:200]}...")

                    # Build trace record
                    trace = {
                        "problem_type": problem_type,
                        "problem_id": problem.problem_id,
                        "problem_title": problem.title,
                        "problem_description": problem.description,
                        "function_signature": problem.function_signature,
                        "function_name": problem.function_name,
                        "test_cases": [
                            {"input_args": tc.input_args, "expected_output": tc.expected_output}
                            for tc in problem.test_cases
                        ],
                        "difficulty": difficulty,
                        "thinking": response.thinking,
                        "solution_code": response.code,
                        "raw_response": response.raw_response,
                        "teacher_model": model_name,
                        "verified": all_passed,
                        "passed_tests": passed,
                        "total_tests": total_tests,
                        "timestamp": datetime.now().isoformat(),
                    }
                    traces.append(trace)

                except Exception as e:
                    print(f"  Error: {e}")
                    continue

                # Rate limiting
                if delay > 0:
                    time.sleep(delay)

    # Save traces
    output_path.parent.mkdir(parents=True, exist_ok=True)

    verified_traces = [t for t in traces if t["verified"]]

    with open(output_path, "w") as f:
        for trace in verified_traces:
            f.write(json.dumps(trace) + "\n")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total attempts: {len(traces)}")
    print(f"Verified: {len(verified_traces)}")
    print(f"Success rate: {100*len(verified_traces)/max(len(traces),1):.1f}%")
    print(f"Saved to: {output_path}")

    # Per-problem breakdown
    print("\nPer-problem breakdown:")
    for ptype in problem_types:
        ptype_traces = [t for t in traces if t["problem_type"] == ptype]
        ptype_verified = [t for t in ptype_traces if t["verified"]]
        print(f"  {ptype}: {len(ptype_verified)}/{len(ptype_traces)} verified")

    return verified_traces


def main():
    parser = argparse.ArgumentParser(description="Generate teacher traces for hard problems")
    parser.add_argument(
        "--teacher",
        choices=["gemini", "claude"],
        default="gemini",
        help="Teacher model to use (default: gemini)",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=WEAK_PROBLEMS,
        help=f"Problem types (default: {WEAK_PROBLEMS})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all hard problems",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Traces per problem per difficulty (default: 3)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        type=int,
        default=[3, 5, 7],
        help="Difficulty levels (default: 3 5 7)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/coldstart_v2/hard_traces.jsonl"),
        help="Output file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    problem_types = ALL_HARD_PROBLEMS if args.all else args.problems

    print("="*70)
    print("TEACHER DISTILLATION FOR HARD PROBLEMS")
    print("="*70)
    print(f"Teacher: {args.teacher}")
    print(f"Problems: {problem_types}")
    print(f"Count per type per difficulty: {args.count}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Output: {args.output}")
    print()

    traces = generate_traces(
        problem_types=problem_types,
        count_per_type=args.count,
        difficulties=args.difficulties,
        output_path=args.output,
        teacher=args.teacher,
        seed=args.seed,
        delay=args.delay,
        verbose=args.verbose,
    )

    return 0 if traces else 1


if __name__ == "__main__":
    sys.exit(main())
