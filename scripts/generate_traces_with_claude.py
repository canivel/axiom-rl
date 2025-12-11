#!/usr/bin/env python3
"""
Generate teacher traces by having Claude (this script's user) solve problems.

This script generates V2 problems and outputs them in a format that Claude
can solve directly in conversation. The user (Claude) provides solutions,
which are then verified and saved as training data.

Usage:
    # Generate problems for Claude to solve
    python scripts/generate_traces_with_claude.py --generate --count 10

    # After Claude provides solutions, verify and save them
    python scripts/generate_traces_with_claude.py --verify solutions.jsonl
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.procedural.generators_v2 import GENERATORS_V2, get_generator_v2
from axiom.procedural.base_v2 import AlgorithmicProblem


def generate_problems_for_claude(
    problem_types: list[str],
    count_per_type: int,
    difficulty_range: tuple[int, int],
    seed: int,
    output_path: Path,
):
    """Generate problems and save them for Claude to solve."""

    problems = []

    for problem_type in problem_types:
        generator = get_generator_v2(problem_type, seed=seed)

        for i in range(count_per_type):
            difficulty = (i % (difficulty_range[1] - difficulty_range[0] + 1)) + difficulty_range[0]
            problem = generator.generate(difficulty=difficulty, num_test_cases=5)

            # Build the problem prompt
            examples = "\n".join([
                f"  - {problem.function_signature.split('(')[0].replace('def ', '')}({', '.join(repr(a) for a in tc.input_args)}) -> {repr(tc.expected_output)}"
                for tc in problem.test_cases[:3]
            ])

            problem_data = {
                "problem_type": problem_type,
                "problem_id": problem.problem_id,
                "difficulty": difficulty,
                "title": problem.title,
                "description": problem.description,
                "function_signature": problem.function_signature,
                "examples": examples,
                "test_cases": [
                    {"input_args": tc.input_args, "expected_output": tc.expected_output}
                    for tc in problem.test_cases
                ],
            }
            problems.append(problem_data)

    # Save problems
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(problems, f, indent=2)

    print(f"Generated {len(problems)} problems to {output_path}")
    print("\nCopy the problems to Claude and ask it to solve them with <think> reasoning.")

    return problems


def print_problems_for_claude(problems: list[dict], batch_size: int = 5):
    """Print problems in a format Claude can solve."""

    print("\n" + "="*70)
    print("PROBLEMS FOR CLAUDE TO SOLVE")
    print("="*70)
    print("""
Please solve each problem below. For each problem:
1. Show your reasoning in <think> tags
2. Provide the Python code in ```python``` blocks

Format your response as a JSON array with objects containing:
- problem_id: the problem ID
- thinking: your reasoning (from <think> tags)
- solution_code: your Python code
""")

    for i, p in enumerate(problems[:batch_size]):
        print(f"\n--- Problem {i+1}: {p['title']} ({p['problem_type']}) ---")
        print(f"Difficulty: {p['difficulty']}")
        print(f"\n{p['description']}")
        print(f"\nFunction signature: {p['function_signature']}")
        print(f"\nExamples:\n{p['examples']}")
        print()


def verify_solutions(solutions_path: Path, problems_path: Path, output_path: Path):
    """Verify Claude's solutions and save verified traces."""

    with open(problems_path) as f:
        problems = {p["problem_id"]: p for p in json.load(f)}

    with open(solutions_path) as f:
        solutions = json.load(f)

    traces = []
    verified = 0

    for sol in solutions:
        problem = problems.get(sol["problem_id"])
        if not problem:
            print(f"Warning: Problem {sol['problem_id']} not found")
            continue

        # Verify the solution
        passed = 0
        total = len(problem["test_cases"])

        for tc in problem["test_cases"]:
            args_str = ", ".join(repr(a) for a in tc["input_args"])
            func_name = problem["function_signature"].split("(")[0].replace("def ", "")

            test_code = f"""
{sol['solution_code']}

result = {func_name}({args_str})
expected = {repr(tc['expected_output'])}
assert result == expected, f"Expected {{expected}}, got {{result}}"
"""
            try:
                exec(test_code, {})
                passed += 1
            except Exception as e:
                pass

        all_passed = passed == total
        if all_passed:
            verified += 1

        trace = {
            "problem_type": problem["problem_type"],
            "problem_id": problem["problem_id"],
            "problem_title": problem["title"],
            "problem_description": problem["description"],
            "function_signature": problem["function_signature"],
            "test_cases": problem["test_cases"],
            "difficulty": problem["difficulty"],
            "thinking": sol.get("thinking", ""),
            "solution_code": sol["solution_code"],
            "teacher_model": "claude-direct",
            "verified": all_passed,
            "passed_tests": passed,
            "total_tests": total,
            "timestamp": datetime.now().isoformat(),
        }
        traces.append(trace)

        status = "VERIFIED" if all_passed else f"FAILED ({passed}/{total})"
        print(f"  {problem['problem_id']}: {status}")

    # Save traces
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for trace in traces:
            if trace["verified"]:  # Only save verified
                f.write(json.dumps(trace) + "\n")

    print(f"\nVerified: {verified}/{len(solutions)}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate traces with Claude directly")
    parser.add_argument("--generate", action="store_true", help="Generate problems")
    parser.add_argument("--verify", type=Path, help="Verify solutions from file")
    parser.add_argument("--problems", type=Path, default=Path("data/claude_direct/problems.json"))
    parser.add_argument("--output", type=Path, default=Path("data/coldstart_v2/claude_direct_traces.jsonl"))
    parser.add_argument("--types", nargs="+", default=list(GENERATORS_V2.keys()))
    parser.add_argument("--count", type=int, default=5, help="Problems per type")
    parser.add_argument("--min-difficulty", type=int, default=1)
    parser.add_argument("--max-difficulty", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print", action="store_true", help="Print problems for Claude")

    args = parser.parse_args()

    if args.generate:
        problems = generate_problems_for_claude(
            problem_types=args.types,
            count_per_type=args.count,
            difficulty_range=(args.min_difficulty, args.max_difficulty),
            seed=args.seed,
            output_path=args.problems,
        )
        if args.print:
            print_problems_for_claude(problems)

    elif args.verify:
        verify_solutions(args.verify, args.problems, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
