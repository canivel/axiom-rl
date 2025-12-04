#!/usr/bin/env python
"""Run V2 Self-Improvement Experiment.

This uses the CORRECT problem design with:
1. Multiple test cases per problem
2. Functions that take input
3. Requires actual algorithm learning

Usage:
    uv run python scripts/run_self_improve_v2.py --experiment v2_test
"""

import argparse
import sys
sys.path.insert(0, ".")

from axiom.experiments.self_improve_v2 import SelfImproveExperimentV2, SelfImproveConfigV2


def main():
    parser = argparse.ArgumentParser(description="Run V2 Self-Improvement Experiment")

    parser.add_argument(
        "--experiment",
        type=str,
        default="v2_experiment",
        help="Experiment name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model",
    )
    parser.add_argument(
        "--problem-types",
        type=str,
        nargs="+",
        default=["rpn", "parentheses"],
        help="Problem types to use",
    )
    parser.add_argument(
        "--train-per-type",
        type=int,
        default=10,
        help="Training problems per type",
    )
    parser.add_argument(
        "--val-per-type",
        type=int,
        default=5,
        help="Validation problems per type",
    )
    parser.add_argument(
        "--test-per-type",
        type=int,
        default=5,
        help="Test problems per type",
    )
    parser.add_argument(
        "--test-cases",
        type=int,
        default=5,
        help="Test cases per problem",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of self-improvement iterations",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create config
    config = SelfImproveConfigV2(
        experiment_name=args.experiment,
        base_model=args.model,
        problem_types=args.problem_types,
        train_problems_per_type=args.train_per_type,
        val_problems_per_type=args.val_per_type,
        test_problems_per_type=args.test_per_type,
        test_cases_per_problem=args.test_cases,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        seed=args.seed,
    )

    # Print config
    print("="*60)
    print("V2 SELF-IMPROVEMENT EXPERIMENT")
    print("="*60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.base_model}")
    print(f"Problem types: {config.problem_types}")
    print(f"Train problems: {config.train_problems_per_type} per type")
    print(f"Test cases per problem: {config.test_cases_per_problem}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Learning rate: {config.learning_rate}")
    print("="*60)

    # Run experiment
    experiment = SelfImproveExperimentV2(config)
    results = experiment.run()

    print("\nExperiment complete!")
    print(f"Results saved to: experiments/{config.experiment_name}/")


if __name__ == "__main__":
    main()
