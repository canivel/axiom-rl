#!/usr/bin/env python3
"""Run a grokking experiment to observe generalization.

This script runs a controlled experiment that:
1. Generates fixed train/val/test splits of procedural problems
2. Evaluates model performance iteratively
3. Tracks metrics to detect "grokking" (sudden generalization)

Usage:
    # Run with defaults
    python scripts/run_grokking_experiment.py

    # Custom configuration
    python scripts/run_grokking_experiment.py \
        --train-size 1000 \
        --val-size 200 \
        --iterations 100 \
        --experiment my_experiment

    # Run specific problem types
    python scripts/run_grokking_experiment.py \
        --problem-types arithmetic rpn
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axiom.experiments import GrokkingExperiment, GrokkingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run a grokking experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run
    python scripts/run_grokking_experiment.py --experiment grok_v1

    # More iterations
    python scripts/run_grokking_experiment.py --iterations 100 --eval-every 5

    # Specific problem types
    python scripts/run_grokking_experiment.py --problem-types arithmetic rpn parentheses

    # Harder problems
    python scripts/run_grokking_experiment.py --min-difficulty 5 --max-difficulty 9
        """,
    )

    # Experiment identification
    parser.add_argument(
        "--experiment",
        type=str,
        default="grokking_v1",
        help="Experiment name (default: grokking_v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Output directory (default: experiments/)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model to use",
    )

    # Data configuration
    parser.add_argument(
        "--train-size",
        type=int,
        default=30,
        help="Number of training problems (default: 30)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=10,
        help="Number of validation problems (default: 10)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10,
        help="Number of test problems (default: 10)",
    )
    parser.add_argument(
        "--problem-types",
        nargs="+",
        default=["arithmetic", "rpn", "parentheses"],
        help="Problem types to include (default: arithmetic rpn parentheses)",
    )
    parser.add_argument(
        "--min-difficulty",
        type=int,
        default=3,
        help="Minimum difficulty (1-10, default: 3)",
    )
    parser.add_argument(
        "--max-difficulty",
        type=int,
        default=7,
        help="Maximum difficulty (1-10, default: 7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Training/evaluation configuration
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations (default: 20)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Evaluate every N iterations (default: 5)",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=10,
        help="Run test evaluation every N iterations (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N iterations (default: 10)",
    )

    # Resume
    parser.add_argument(
        "--resume-from",
        type=int,
        help="Iteration to resume from",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Create config
    config = GrokkingConfig(
        experiment_name=args.experiment,
        output_dir=args.output_dir,
        base_model=args.model,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        problem_types=args.problem_types,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        seed=args.seed,
        num_iterations=args.iterations,
        eval_every=args.eval_every,
        test_every=args.test_every,
        checkpoint_every=args.checkpoint_every,
    )

    print("\n" + "=" * 60)
    print("GROKKING EXPERIMENT")
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.base_model}")
    print(f"Problems: {config.train_size} train, {config.val_size} val, {config.test_size} test")
    print(f"Types: {', '.join(config.problem_types)}")
    print(f"Difficulty: {config.min_difficulty}-{config.max_difficulty}")
    print(f"Iterations: {config.num_iterations}")
    print("=" * 60 + "\n")

    # Create and run experiment
    experiment = GrokkingExperiment(config)
    experiment.run(resume_from=args.resume_from)

    print("\nExperiment complete!")
    print(f"Results saved to: {config.output_dir / config.experiment_name}")


if __name__ == "__main__":
    main()
