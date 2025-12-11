#!/usr/bin/env python3
"""
Generate teacher dataset for V2 problems using Claude or Gemini.

This script generates high-quality reasoning traces for the V2 procedural
problems, which have multiple test cases and require true algorithmic
solutions (not memorization).

Usage:
    # Generate with Claude (default)
    python scripts/generate_teacher_data_v2.py --teacher claude

    # Generate with Gemini
    python scripts/generate_teacher_data_v2.py --teacher gemini

    # Generate for specific problem types
    python scripts/generate_teacher_data_v2.py --types rpn arithmetic parentheses

    # Compare both teachers
    python scripts/generate_teacher_data_v2.py --teacher both
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.coldstart import GeminiClient, ClaudeClient
from axiom.procedural.generators_v2 import GENERATORS_V2, get_generator_v2
from axiom.procedural.base_v2 import AlgorithmicProblem
from axiom.verifier.harness import TestHarness
from axiom.verifier.result import VerificationStatus


@dataclass
class V2ReasoningTrace:
    """A reasoning trace for V2 problems with multiple test cases."""

    problem_type: str
    problem_id: str
    problem_title: str
    problem_description: str
    function_signature: str
    test_cases: list  # List of {input_args, expected_output}
    difficulty: int
    thinking: str
    solution_code: str
    teacher_model: str
    verified: bool
    passed_tests: int
    total_tests: int
    timestamp: str


def generate_v2_problem_prompt(problem: AlgorithmicProblem) -> tuple[str, str, str]:
    """
    Generate prompt components for a V2 problem.

    Returns:
        (title, description, function_signature)
    """
    # Build description with test case examples
    description = problem.description + "\n\n## Examples\n"

    # Show first 2-3 test cases as examples
    for i, tc in enumerate(problem.test_cases[:3]):
        args_str = ", ".join(repr(arg) for arg in tc.input_args)
        description += f"- {problem.function_signature.split('(')[0].replace('def ', '')}({args_str}) -> {repr(tc.expected_output)}\n"

    if len(problem.test_cases) > 3:
        description += f"\n(Plus {len(problem.test_cases) - 3} more test cases...)"

    return problem.title, description, problem.function_signature


def verify_solution(
    solution_code: str,
    problem: AlgorithmicProblem,
    harness: TestHarness,
) -> tuple[bool, int, int]:
    """
    Verify a solution against all test cases.

    Returns:
        (all_passed, passed_count, total_count)
    """
    passed = 0
    total = len(problem.test_cases)

    for tc in problem.test_cases:
        # Build test code that calls the function with specific args
        args_str = ", ".join(repr(arg) for arg in tc.input_args)
        func_name = problem.function_signature.split("(")[0].replace("def ", "")
        test_code = f"""
{solution_code}

# Test execution
result = {func_name}({args_str})
expected = {repr(tc.expected_output)}

# Verify
assert result == expected, f"Expected {{expected}}, got {{result}}"
"""
        try:
            exec(test_code, {})
            passed += 1
        except Exception:
            pass

    return passed == total, passed, total


def generate_traces_for_problem_type(
    problem_type: str,
    client,
    num_problems: int = 10,
    difficulty_range: tuple[int, int] = (1, 7),
    delay: float = 1.0,
    seed: int = 42,
) -> list[V2ReasoningTrace]:
    """Generate traces for a specific problem type."""

    generator = get_generator_v2(problem_type, seed=seed)
    harness = TestHarness()
    traces = []

    print(f"\n  Generating {num_problems} {problem_type} problems...")

    for i in range(num_problems):
        # Generate a problem with difficulty in range
        difficulty = (i % (difficulty_range[1] - difficulty_range[0] + 1)) + difficulty_range[0]
        problem = generator.generate(difficulty=difficulty, num_test_cases=5)

        # Build prompt
        title, description, signature = generate_v2_problem_prompt(problem)

        try:
            # Generate reasoning trace
            response = client.generate_reasoning_trace(
                problem_title=title,
                problem_description=description,
                function_signature=signature,
                temperature=0.7,
            )

            # Verify the solution
            all_passed, passed, total = verify_solution(
                response.code, problem, harness
            )

            trace = V2ReasoningTrace(
                problem_type=problem_type,
                problem_id=problem.problem_id,
                problem_title=title,
                problem_description=description,
                function_signature=signature,
                test_cases=[
                    {"input_args": tc.input_args, "expected_output": tc.expected_output}
                    for tc in problem.test_cases
                ],
                difficulty=difficulty,
                thinking=response.thinking,
                solution_code=response.code,
                teacher_model=client.model_name,
                verified=all_passed,
                passed_tests=passed,
                total_tests=total,
                timestamp=datetime.now().isoformat(),
            )

            traces.append(trace)

            status = "VERIFIED" if all_passed else f"FAILED ({passed}/{total})"
            print(f"    [{i+1}/{num_problems}] difficulty={difficulty}: {status}")

            # Rate limiting
            if i < num_problems - 1:
                time.sleep(delay)

        except Exception as e:
            print(f"    [{i+1}/{num_problems}] ERROR: {e}")
            continue

    return traces


def save_traces(traces: list[V2ReasoningTrace], output_path: Path) -> None:
    """Save traces to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(asdict(trace)) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher data for V2 problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_teacher_data_v2.py --teacher claude
  python scripts/generate_teacher_data_v2.py --teacher gemini --types rpn arithmetic
  python scripts/generate_teacher_data_v2.py --teacher both --problems-per-type 20
""",
    )

    parser.add_argument(
        "--teacher",
        choices=["claude", "gemini", "both"],
        default="claude",
        help="Which teacher model to use (default: claude)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(GENERATORS_V2.keys()),
        default=None,
        help=f"Problem types to generate. Default: all. Available: {list(GENERATORS_V2.keys())}",
    )
    parser.add_argument(
        "--problems-per-type",
        type=int,
        default=10,
        help="Number of problems per type (default: 10)",
    )
    parser.add_argument(
        "--min-difficulty",
        type=int,
        default=1,
        help="Minimum difficulty (default: 1)",
    )
    parser.add_argument(
        "--max-difficulty",
        type=int,
        default=7,
        help="Maximum difficulty (default: 7)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/coldstart_v2"),
        help="Output directory (default: data/coldstart_v2)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List available problem types and exit",
    )

    args = parser.parse_args()

    # List types if requested
    if args.list_types:
        print("Available V2 problem types:")
        for name, cls in GENERATORS_V2.items():
            gen = cls()
            print(f"\n  {name}:")
            print(f"    Title: {gen.title}")
            print(f"    Signature: {gen.function_signature}")
        return 0

    # Get problem types
    problem_types = args.types or list(GENERATORS_V2.keys())

    # Initialize clients
    clients = []
    if args.teacher in ["claude", "both"]:
        try:
            clients.append(("claude", ClaudeClient()))
            print("Initialized Claude client")
        except ValueError as e:
            print(f"Warning: Could not initialize Claude: {e}")

    if args.teacher in ["gemini", "both"]:
        try:
            clients.append(("gemini", GeminiClient()))
            print("Initialized Gemini client")
        except ValueError as e:
            print(f"Warning: Could not initialize Gemini: {e}")

    if not clients:
        print("Error: No teacher models available. Check API keys in .env")
        return 1

    # Generate dataset
    print("\n" + "=" * 60)
    print("V2 TEACHER DATA GENERATION")
    print("=" * 60)
    print(f"Teachers: {[name for name, _ in clients]}")
    print(f"Problem types: {problem_types}")
    print(f"Problems per type: {args.problems_per_type}")
    print(f"Difficulty range: {args.min_difficulty}-{args.max_difficulty}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    total_traces = 0
    verified_traces = 0

    for teacher_name, client in clients:
        print(f"\n>>> Using {teacher_name.upper()} as teacher")

        output_file = args.output_dir / f"{teacher_name}_traces_v2.jsonl"

        for problem_type in problem_types:
            traces = generate_traces_for_problem_type(
                problem_type=problem_type,
                client=client,
                num_problems=args.problems_per_type,
                difficulty_range=(args.min_difficulty, args.max_difficulty),
                delay=args.delay,
                seed=args.seed,
            )

            # Save traces
            verified_only = [t for t in traces if t.verified]
            save_traces(verified_only, output_file)

            total_traces += len(traces)
            verified_traces += len(verified_only)

            print(f"  >> Saved {len(verified_only)}/{len(traces)} verified traces")

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total traces generated: {total_traces}")
    print(f"Verified traces saved: {verified_traces}")
    print(f"Verification rate: {verified_traces/total_traces*100:.1f}%")
    print(f"\nOutput files:")
    for teacher_name, _ in clients:
        print(f"  - {args.output_dir / f'{teacher_name}_traces_v2.jsonl'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
