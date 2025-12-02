#!/usr/bin/env python3
"""Run a self-improvement experiment with integrated training.

This script implements the full Expert Iteration cycle:
1. Evaluate model on procedural problems
2. Collect correct solutions
3. Train on correct solutions (LoRA)
4. Repeat and observe improvement

Usage:
    # Run with defaults (small scale for testing)
    python scripts/run_self_improve.py --experiment test_v1

    # Larger scale
    python scripts/run_self_improve.py \
        --experiment full_v1 \
        --train-size 100 \
        --iterations 10
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axiom.experiments import SelfImprovementExperiment, SelfImproveConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run self-improvement experiment",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="self_improve_v1",
        help="Experiment name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=30,
        help="Number of training problems",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=10,
        help="Number of validation problems",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10,
        help="Number of test problems",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of self-improvement iterations",
    )
    parser.add_argument(
        "--problem-types",
        nargs="+",
        default=["arithmetic", "rpn", "parentheses"],
        help="Problem types to include",
    )
    parser.add_argument(
        "--min-difficulty",
        type=int,
        default=3,
        help="Minimum difficulty (1-10)",
    )
    parser.add_argument(
        "--max-difficulty",
        type=int,
        default=7,
        help="Maximum difficulty (1-10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )

    args = parser.parse_args()

    config = SelfImproveConfig(
        experiment_name=args.experiment,
        output_dir=args.output_dir,
        base_model=args.model,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        num_iterations=args.iterations,
        problem_types=args.problem_types,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        seed=args.seed,
        learning_rate=args.lr,
    )

    print("\n" + "=" * 60)
    print("SELF-IMPROVEMENT EXPERIMENT")
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.base_model}")
    print(f"Problems: {config.train_size} train, {config.val_size} val, {config.test_size} test")
    print(f"Types: {', '.join(config.problem_types)}")
    print(f"Difficulty: {config.min_difficulty}-{config.max_difficulty}")
    print(f"Iterations: {config.num_iterations}")
    print("=" * 60 + "\n")

    experiment = SelfImprovementExperiment(config)
    results = experiment.run()

    print("\nExperiment complete!")
    print(f"Results saved to: {config.output_dir / config.experiment_name}")


if __name__ == "__main__":
    main()
