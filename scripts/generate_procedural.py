#!/usr/bin/env python3
"""Generate procedural training data with infinite unique problems.

This script generates problems that have perfect ground truth verification.
The generated data can be used for:
1. Training data augmentation
2. Evaluation benchmarks
3. Testing model generalization

Usage:
    # Generate 100 problems across all types
    python scripts/generate_procedural.py --count 100

    # Generate specific problem types
    python scripts/generate_procedural.py --types arithmetic rpn --count 50

    # Generate with specific difficulty range
    python scripts/generate_procedural.py --min-difficulty 3 --max-difficulty 7

    # Generate and save to file
    python scripts/generate_procedural.py --output data/procedural/problems.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axiom.procedural import (
    GENERATORS,
    get_generator,
    get_all_generators,
    ProceduralProblem,
)


def generate_problems(
    problem_types: list[str] | None,
    count: int,
    min_difficulty: int,
    max_difficulty: int,
    seed: int | None,
) -> list[ProceduralProblem]:
    """Generate procedural problems."""
    problems = []

    # Get generators
    if problem_types:
        generators = {t: get_generator(t, seed=seed) for t in problem_types}
    else:
        generators = get_all_generators(seed=seed)

    # Calculate problems per generator
    gen_names = list(generators.keys())
    per_generator = count // len(gen_names)
    remainder = count % len(gen_names)

    print(f"Generating {count} problems across {len(gen_names)} types...")
    print(f"  Types: {', '.join(gen_names)}")
    print(f"  Difficulty range: {min_difficulty}-{max_difficulty}")
    print()

    for i, (name, gen) in enumerate(generators.items()):
        # Distribute remainder across first generators
        gen_count = per_generator + (1 if i < remainder else 0)

        print(f"  Generating {gen_count} {name} problems...")

        batch = gen.generate_batch(
            count=gen_count,
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
        )
        problems.extend(batch)

    return problems


def problem_to_dict(problem: ProceduralProblem) -> dict:
    """Convert problem to dictionary for JSON serialization."""
    return {
        "problem_type": problem.problem_type,
        "problem_id": problem.problem_id,
        "title": problem.title,
        "description": problem.description,
        "function_signature": problem.function_signature,
        "input_data": problem.input_data,
        "expected_output": problem.expected_output,
        "difficulty": problem.difficulty,
        "complexity": problem.complexity,
    }


def save_problems(problems: list[ProceduralProblem], output_path: Path) -> None:
    """Save problems to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for problem in problems:
            f.write(json.dumps(problem_to_dict(problem)) + "\n")

    print(f"Saved {len(problems)} problems to {output_path}")


def print_sample_problems(problems: list[ProceduralProblem], count: int = 3) -> None:
    """Print sample problems for verification."""
    print("\n" + "=" * 60)
    print("SAMPLE PROBLEMS")
    print("=" * 60)

    # Get samples from different types
    by_type = {}
    for p in problems:
        if p.problem_type not in by_type:
            by_type[p.problem_type] = []
        by_type[p.problem_type].append(p)

    for prob_type, type_problems in by_type.items():
        sample = type_problems[0]
        print(f"\n--- {prob_type.upper()} ---")
        print(f"Difficulty: {sample.difficulty} ({sample.complexity})")
        print(f"Input: {sample.input_data}")
        print(f"Expected: {sample.expected_output}")
        print()


def print_statistics(problems: list[ProceduralProblem]) -> None:
    """Print statistics about generated problems."""
    print("\n" + "=" * 60)
    print("GENERATION STATISTICS")
    print("=" * 60)

    # Count by type
    by_type = {}
    for p in problems:
        by_type[p.problem_type] = by_type.get(p.problem_type, 0) + 1

    print("\nBy Problem Type:")
    for prob_type, count in sorted(by_type.items()):
        print(f"  {prob_type}: {count}")

    # Count by difficulty
    by_difficulty = {}
    for p in problems:
        by_difficulty[p.difficulty] = by_difficulty.get(p.difficulty, 0) + 1

    print("\nBy Difficulty:")
    for diff in sorted(by_difficulty.keys()):
        count = by_difficulty[diff]
        bar = "#" * (count // 2)
        print(f"  {diff:2d}: {bar} ({count})")

    # Count by complexity
    by_complexity = {}
    for p in problems:
        by_complexity[p.complexity] = by_complexity.get(p.complexity, 0) + 1

    print("\nBy Complexity:")
    for complexity in ["easy", "medium", "hard"]:
        count = by_complexity.get(complexity, 0)
        print(f"  {complexity}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural training problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_procedural.py --count 100
  python scripts/generate_procedural.py --types arithmetic rpn --count 50
  python scripts/generate_procedural.py --min-difficulty 5 --max-difficulty 10
  python scripts/generate_procedural.py --output data/procedural/problems.jsonl
        """,
    )

    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(GENERATORS.keys()),
        help=f"Problem types to generate. Available: {list(GENERATORS.keys())}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Total number of problems to generate (default: 100)",
    )
    parser.add_argument(
        "--min-difficulty",
        type=int,
        default=1,
        choices=range(1, 11),
        help="Minimum difficulty level 1-10 (default: 1)",
    )
    parser.add_argument(
        "--max-difficulty",
        type=int,
        default=10,
        choices=range(1, 11),
        help="Maximum difficulty level 1-10 (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except errors",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List available problem types and exit",
    )

    args = parser.parse_args()

    # List types and exit
    if args.list_types:
        print("Available problem types:")
        for name, cls in GENERATORS.items():
            gen = cls()
            print(f"\n  {name}:")
            print(f"    Title: {gen.title}")
            print(f"    Signature: {gen.function_signature}")
        return

    # Validate difficulty range
    if args.min_difficulty > args.max_difficulty:
        print("Error: min-difficulty must be <= max-difficulty")
        sys.exit(1)

    # Generate problems
    problems = generate_problems(
        problem_types=args.types,
        count=args.count,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        seed=args.seed,
    )

    # Save to file if specified
    if args.output:
        save_problems(problems, args.output)

    # Print statistics unless quiet
    if not args.quiet:
        print_statistics(problems)
        print_sample_problems(problems)

    print(f"\nDone! Generated {len(problems)} procedural problems")


if __name__ == "__main__":
    main()
