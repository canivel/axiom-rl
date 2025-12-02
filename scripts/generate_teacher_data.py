#!/usr/bin/env python3
"""
Generate teacher dataset using Gemini for Cold Start phase.

This script uses a strong model (Gemini 2.5) to generate
high-quality reasoning traces that will be used to prime
the smaller model before self-improvement.

Usage:
    python scripts/generate_teacher_data.py
    python scripts/generate_teacher_data.py --traces-per-problem 20
    python scripts/generate_teacher_data.py --problems two_sum fizzbuzz
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.coldstart import GeminiClient, ReasoningTraceGenerator
from axiom.problems.dataset import ProblemDataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher dataset using Gemini for Cold Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with defaults (10 traces per problem)
  python scripts/generate_teacher_data.py

  # Generate more traces per problem
  python scripts/generate_teacher_data.py --traces-per-problem 20

  # Generate for specific problems only
  python scripts/generate_teacher_data.py --problems two_sum fizzbuzz reverse_string

  # Use a different Gemini model
  python scripts/generate_teacher_data.py --model gemini-2.5-pro-preview-05-06

  # Custom output file
  python scripts/generate_teacher_data.py --output teacher_v2.jsonl
""",
    )

    parser.add_argument(
        "--traces-per-problem",
        type=int,
        default=10,
        help="Number of traces to generate per problem (default: 10)",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=None,
        help="Specific problem IDs to generate for (default: all)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/coldstart",
        help="Output directory (default: data/coldstart)",
    )
    parser.add_argument(
        "--output",
        default="teacher_traces.jsonl",
        help="Output filename (default: teacher_traces.jsonl)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include traces that failed verification (default: only verified)",
    )
    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="List available problems and exit",
    )

    args = parser.parse_args()

    # Load problem dataset
    dataset = ProblemDataset()

    # List problems if requested
    if args.list_problems:
        print("Available problems:")
        for p in dataset.problems:
            print(f"  - {p.id}: {p.title}")
        return 0

    # Get problems to generate for
    if args.problems:
        problems = [dataset.get_problem(pid) for pid in args.problems]
        problems = [p for p in problems if p is not None]
        if not problems:
            print(f"Error: No valid problems found from: {args.problems}")
            return 1
    else:
        problems = dataset.problems

    # Initialize client and generator
    try:
        client = GeminiClient(model_name=args.model)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nMake sure GEMINI_API_KEY is set in .env file")
        return 1

    generator = ReasoningTraceGenerator(
        client=client,
        output_dir=Path(args.output_dir),
        output_file=args.output,
    )

    # Generate dataset
    print("=" * 50)
    print("COLD START: Teacher Data Generation")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Problems: {len(problems)}")
    print(f"Traces per problem: {args.traces_per_problem}")
    print(f"Expected total: {len(problems) * args.traces_per_problem}")
    print(f"Output: {args.output_dir}/{args.output}")
    print("=" * 50)

    traces = generator.generate_dataset(
        problems=problems,
        traces_per_problem=args.traces_per_problem,
        delay=args.delay,
        only_verified=not args.include_failed,
    )

    verified_count = len([t for t in traces if t.verified])

    print(f"\nTeacher dataset generated!")
    print(f"Total: {len(traces)} traces ({verified_count} verified)")
    print(f"\nNext step: Train on this data to create the 'cold start' model:")
    print(f"  python scripts/run_training.py --solutions {args.output_dir}/{args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
