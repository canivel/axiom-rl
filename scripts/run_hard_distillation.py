#!/usr/bin/env python3
"""
Full pipeline for hard problem distillation.

This script runs the complete teacher distillation pipeline for hard problems:
1. Generate Claude traces for weak problems (Coin Change, Knapsack, N-Queens)
2. SFT train on the teacher traces
3. GRPO refinement on the SFT model

This approach combines:
- Teacher distillation: Learn from Claude's correct solutions
- Reinforcement learning: Refine with execution-based rewards

Usage:
    # Full pipeline
    uv run python scripts/run_hard_distillation.py

    # Skip trace generation (use existing traces)
    uv run python scripts/run_hard_distillation.py --skip-traces

    # Custom problems
    uv run python scripts/run_hard_distillation.py --problems coin_change knapsack
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Default weak problems (from Experiment 10)
WEAK_PROBLEMS = ["coin_change", "knapsack", "n_queens"]

# Default configuration
DEFAULT_CONFIG = {
    "trace_count": 3,           # Traces per problem per difficulty
    "difficulties": [3, 5, 7],  # Difficulty levels
    "sft_epochs": 3,            # SFT training epochs
    "grpo_steps": 10,           # GRPO steps per problem
    "base_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
}


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Full hard problem distillation pipeline"
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=WEAK_PROBLEMS,
        help=f"Problem types (default: {WEAK_PROBLEMS})",
    )
    parser.add_argument(
        "--skip-traces",
        action="store_true",
        help="Skip trace generation (use existing traces)",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT training",
    )
    parser.add_argument(
        "--skip-grpo",
        action="store_true",
        help="Skip GRPO refinement",
    )
    parser.add_argument(
        "--traces-path",
        type=Path,
        default=Path("data/coldstart_v2/hard_traces.jsonl"),
        help="Path to traces file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/hard-distill"),
        help="Output directory for models",
    )
    parser.add_argument(
        "--trace-count",
        type=int,
        default=DEFAULT_CONFIG["trace_count"],
        help="Traces per problem per difficulty",
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=DEFAULT_CONFIG["sft_epochs"],
    )
    parser.add_argument(
        "--grpo-steps",
        type=int,
        default=DEFAULT_CONFIG["grpo_steps"],
    )
    parser.add_argument(
        "--experiment",
        default="hard-distill-v1",
        help="Experiment name",
    )

    args = parser.parse_args()

    print("="*70)
    print("HARD PROBLEM DISTILLATION PIPELINE")
    print("="*70)
    print(f"Problems: {args.problems}")
    print(f"Traces path: {args.traces_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Experiment: {args.experiment}")
    print()

    # Track results
    results = {
        "start_time": datetime.now().isoformat(),
        "config": vars(args),
        "steps": [],
    }

    # Step 1: Generate Claude traces
    if not args.skip_traces:
        difficulties_str = " ".join(str(d) for d in DEFAULT_CONFIG["difficulties"])
        problems_str = " ".join(args.problems)

        cmd = [
            "uv", "run", "python", "scripts/generate_hard_traces.py",
            "--problems", *args.problems,
            "--count", str(args.trace_count),
            "--difficulties", *[str(d) for d in DEFAULT_CONFIG["difficulties"]],
            "--output", str(args.traces_path),
        ]

        exit_code = run_command(cmd, "Generate Claude Traces")

        results["steps"].append({
            "name": "generate_traces",
            "exit_code": exit_code,
        })

        if exit_code != 0:
            print(f"Error: Trace generation failed with code {exit_code}")
            return 1

    # Check traces exist
    if not args.traces_path.exists():
        print(f"Error: Traces file not found: {args.traces_path}")
        print("Run with --skip-traces=false or provide traces manually")
        return 1

    # Count traces
    with open(args.traces_path) as f:
        traces = [json.loads(line) for line in f]
    print(f"\nLoaded {len(traces)} verified traces from {args.traces_path}")

    if len(traces) == 0:
        print("Error: No verified traces found")
        return 1

    # Step 2: SFT training on traces
    if not args.skip_sft:
        sft_output = args.output_dir / "sft"

        cmd = [
            "uv", "run", "python", "scripts/run_training.py",
            "--solutions", str(args.traces_path),
            "--model", DEFAULT_CONFIG["base_model"],
            "--epochs", str(args.sft_epochs),
            "--output-dir", str(sft_output),
        ]

        exit_code = run_command(cmd, "SFT Training on Teacher Traces")

        results["steps"].append({
            "name": "sft_training",
            "exit_code": exit_code,
            "output_dir": str(sft_output),
        })

        if exit_code != 0:
            print(f"Error: SFT training failed with code {exit_code}")
            return 1

    # Step 3: GRPO refinement
    if not args.skip_grpo:
        sft_model = args.output_dir / "sft"

        # Check if SFT model exists
        if not sft_model.exists():
            print(f"Warning: SFT model not found at {sft_model}")
            print("Using base model for GRPO")
            sft_model = Path(DEFAULT_CONFIG["base_model"])

        grpo_output = args.output_dir / "grpo"

        cmd = [
            "uv", "run", "python", "scripts/run_grpo_hard.py",
            "--model", str(sft_model),
            "--problems", *args.problems,
            "--steps", str(args.grpo_steps),
            "--output-dir", str(grpo_output),
        ]

        exit_code = run_command(cmd, "GRPO Refinement")

        results["steps"].append({
            "name": "grpo_refinement",
            "exit_code": exit_code,
            "output_dir": str(grpo_output),
        })

        if exit_code != 0:
            print(f"Warning: GRPO refinement failed with code {exit_code}")

    # Step 4: Final evaluation
    final_model = args.output_dir / "grpo" / "final_model"
    if not final_model.exists():
        final_model = args.output_dir / "sft"

    if final_model.exists():
        cmd = [
            "uv", "run", "python", "scripts/test_hard_problems.py",
            "--model", str(final_model),
            "--problems", *args.problems,
            "--difficulty", "5",
        ]

        exit_code = run_command(cmd, "Final Evaluation")

        results["steps"].append({
            "name": "final_evaluation",
            "exit_code": exit_code,
        })

    # Save results
    results["end_time"] = datetime.now().isoformat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Results saved to: {results_path}")
    print(f"Models saved to: {args.output_dir}")
    print()
    print("Next steps:")
    print("1. Evaluate the model on hard problems:")
    print(f"   uv run python scripts/test_hard_problems.py --model {final_model}")
    print("2. Compare with baseline:")
    print(f"   uv run python scripts/test_hard_problems.py --problems {' '.join(args.problems)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
