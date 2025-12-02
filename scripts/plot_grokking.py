#!/usr/bin/env python3
"""Plot results from a grokking experiment.

Usage:
    # Plot from experiment directory
    python scripts/plot_grokking.py --experiment grokking_v1

    # Save plots to file
    python scripts/plot_grokking.py --experiment grokking_v1 --save

    # ASCII mode (no matplotlib required)
    python scripts/plot_grokking.py --experiment grokking_v1 --ascii
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from axiom.experiments.visualization import (
    plot_grokking_curve,
    plot_generalization_gap,
    print_ascii_grokking_curve,
    generate_experiment_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot grokking experiment results",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name or path to experiment directory",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Base directory for experiments (default: experiments/)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots to files instead of displaying",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Use ASCII art instead of matplotlib",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate text report",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for plots (default: experiment directory)",
    )

    args = parser.parse_args()

    # Determine experiment directory
    if Path(args.experiment).exists():
        experiment_dir = Path(args.experiment)
    else:
        experiment_dir = args.experiments_dir / args.experiment

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    metrics_file = experiment_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        sys.exit(1)

    output_dir = args.output_dir or experiment_dir / "plots"
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting results from: {experiment_dir}")
    print()

    if args.ascii:
        # ASCII mode
        print_ascii_grokking_curve(metrics_file)
    else:
        # Matplotlib mode
        try:
            # Main grokking curve
            print("Generating grokking curve...")
            plot_grokking_curve(
                metrics_file,
                output_file=output_dir / "grokking_curve.png" if args.save else None,
                title=f"Grokking Experiment: {args.experiment}",
                show=not args.save,
            )

            # Generalization gap
            print("Generating generalization gap plot...")
            plot_generalization_gap(
                metrics_file,
                output_file=output_dir / "generalization_gap.png" if args.save else None,
                title=f"Generalization Gap: {args.experiment}",
                show=not args.save,
            )

        except Exception as e:
            print(f"Error generating plots: {e}")
            print("Falling back to ASCII mode...")
            print_ascii_grokking_curve(metrics_file)

    if args.report:
        print("\nGenerating report...")
        report = generate_experiment_report(
            experiment_dir,
            output_file=output_dir / "report.txt" if args.save else None,
        )
        if not args.save:
            print(report)

    if args.save:
        print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
