#!/usr/bin/env python3
"""
Run the axiom-rl MVP pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --problems two_sum fizzbuzz
    python scripts/run_pipeline.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 4
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.config import GeneratorConfig, PipelineConfig, VerifierConfig
from axiom.pipeline.orchestrator import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run axiom-rl MVP pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all problems with default settings
  python scripts/run_pipeline.py

  # Run on specific problems
  python scripts/run_pipeline.py --problems two_sum fizzbuzz reverse_string

  # Use smaller model for faster iteration
  python scripts/run_pipeline.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct

  # Generate more samples per problem
  python scripts/run_pipeline.py --samples 16

  # Skip already-solved problems
  python scripts/run_pipeline.py --skip-existing
""",
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of samples per problem - Best-of-N (default: 8)",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        help="Specific problem IDs to run (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Execution timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/synthetic",
        help="Output directory for solutions (default: data/synthetic)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max retry attempts per problem (default: 3)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip problems that already have solutions",
    )
    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="List available problems and exit",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Experiment name - solutions saved to solutions_{name}.jsonl",
    )

    args = parser.parse_args()

    # List problems mode
    if args.list_problems:
        from axiom.problems.dataset import ProblemDataset

        dataset = ProblemDataset()
        print("Available problems:")
        print("-" * 40)
        for p in dataset:
            print(f"  {p.id:25} [{p.difficulty}] {p.title}")
        return 0

    # Determine output filename
    if args.experiment:
        output_file = f"solutions_{args.experiment}.jsonl"
    else:
        output_file = "solutions.jsonl"

    # Build config
    config = PipelineConfig(
        generator=GeneratorConfig(
            model_name=args.model,
            num_samples=args.samples,
            temperature=args.temperature,
        ),
        verifier=VerifierConfig(
            timeout=args.timeout,
        ),
        output_dir=Path(args.output_dir),
        output_file=output_file,
        max_attempts_per_problem=args.max_attempts,
    )

    # Run pipeline
    pipeline = Pipeline(config)
    stats = pipeline.run(
        problem_ids=args.problems,
        skip_existing=args.skip_existing,
    )

    # Return success if at least one problem was solved
    return 0 if stats["problems_solved"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
