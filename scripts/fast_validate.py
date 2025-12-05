#!/usr/bin/env python3
"""
Fast Hypothesis Validation Script

Quick validation mode for testing hypotheses with ~30 min runtime (24x faster than full experiments).

Optimizations:
- Smaller model: Qwen2.5-Coder-0.5B (vs 1.5B)
- Fewer samples: Best-of-2 (vs Best-of-8)
- Single iteration (vs 3)
- Fewer training problems: 5 per type (vs 25)
- Easier problem types: fizzbuzz, reverse_string (vs fibonacci, remove_duplicates)

Usage:
    # Default fast mode (~30 min)
    uv run python scripts/fast_validate.py

    # Custom problem types
    uv run python scripts/fast_validate.py --problems fizzbuzz is_palindrome

    # Use 1.5B model for more accurate results (~1-2 hours)
    uv run python scripts/fast_validate.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct

    # Medium mode (~2 hours) - 1.5B model with Best-of-4
    uv run python scripts/fast_validate.py --medium
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_focused_improvement import FocusedExperimentConfig, FocusedImprovementExperiment


def main():
    parser = argparse.ArgumentParser(
        description="Fast hypothesis validation (~30 min)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation (30 min)
  uv run python scripts/fast_validate.py

  # Medium validation (2 hours)
  uv run python scripts/fast_validate.py --medium

  # Custom experiment
  uv run python scripts/fast_validate.py --problems arithmetic rpn --samples 4
        """
    )

    # Mode presets
    parser.add_argument("--medium", action="store_true",
                        help="Medium mode: 1.5B model, Best-of-4, 1 iteration (~2 hours)")

    # Overrides
    parser.add_argument("--experiment", default="fast_validate",
                        help="Experiment name (default: fast_validate)")
    parser.add_argument("--model", default=None,
                        help="Model to use (default: 0.5B for fast, 1.5B for medium)")
    parser.add_argument("--problems", nargs="+", default=None,
                        help="Problem types (default: fizzbuzz, reverse_string)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Samples per problem (default: 2 for fast, 4 for medium)")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of iterations (default: 1)")
    parser.add_argument("--train-per-type", type=int, default=None,
                        help="Training problems per type (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set defaults based on mode
    if args.medium:
        # Medium mode: ~2 hours
        model = args.model or "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        samples = args.samples or 4
        problems = args.problems or ["fizzbuzz", "reverse_string"]
        train_per_type = args.train_per_type or 10
    else:
        # Fast mode: ~30 min
        model = args.model or "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        samples = args.samples or 2
        problems = args.problems or ["fizzbuzz", "reverse_string"]
        train_per_type = args.train_per_type or 5

    print("=" * 70)
    print("FAST HYPOTHESIS VALIDATION")
    print("=" * 70)
    print(f"Mode: {'MEDIUM (~2h)' if args.medium else 'FAST (~30min)'}")
    print(f"Model: {model}")
    print(f"Problems: {problems}")
    print(f"Samples: Best-of-{samples}")
    print(f"Iterations: {args.iterations}")
    print(f"Train per type: {train_per_type}")
    print("=" * 70)

    config = FocusedExperimentConfig(
        experiment_name=args.experiment,
        base_model=model,
        problem_types=problems,
        train_problems_per_type=train_per_type,
        val_problems_per_type=5,  # Reduced for speed
        test_problems_per_type=5,  # Reduced for speed
        num_iterations=args.iterations,
        samples_per_problem=samples,
        seed=args.seed,
        output_dir=f"experiments/{args.experiment}"
    )

    experiment = FocusedImprovementExperiment(config)
    summary = experiment.run()

    # Print quick summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)

    if summary.get("metrics"):
        first = summary["metrics"][0]
        last = summary["metrics"][-1]

        print(f"Initial val accuracy: {first['val']['overall']['accuracy']:.1f}%")
        print(f"Final val accuracy:   {last['val']['overall']['accuracy']:.1f}%")

        if len(summary["metrics"]) > 1:
            change = last['val']['overall']['accuracy'] - first['val']['overall']['accuracy']
            emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"Change: {change:+.1f}% {emoji}")

    return summary


if __name__ == "__main__":
    main()
