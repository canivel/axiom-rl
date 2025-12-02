"""Visualization utilities for experiment results."""

import json
from pathlib import Path
from typing import Optional

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_grokking_curve(
    metrics_file: Path,
    output_file: Optional[Path] = None,
    title: str = "Grokking Experiment",
    show: bool = True,
):
    """
    Plot train vs validation accuracy over iterations.

    Args:
        metrics_file: Path to metrics.jsonl file
        output_file: Path to save the plot (optional)
        title: Plot title
        show: Whether to display the plot
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    # Load metrics
    iterations = []
    train_acc = []
    val_acc = []
    test_acc = []

    with open(metrics_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                iterations.append(data["iteration"])
                train_acc.append(data["train_accuracy"] * 100)
                val_acc.append(data["val_accuracy"] * 100)
                if data.get("test_accuracy"):
                    test_acc.append((data["iteration"], data["test_accuracy"] * 100))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot curves
    ax.plot(iterations, train_acc, "b-", linewidth=2, label="Training Accuracy")
    ax.plot(iterations, val_acc, "r-", linewidth=2, label="Validation Accuracy")

    # Plot test points if available
    if test_acc:
        test_iters, test_vals = zip(*test_acc)
        ax.scatter(test_iters, test_vals, c="green", s=100, marker="^", label="Test Accuracy", zorder=5)

    # Styling
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Add grokking annotation if detected
    if len(val_acc) > 20:
        # Simple grokking detection for visualization
        for i in range(10, len(val_acc)):
            old_avg = sum(val_acc[i-10:i-5]) / 5
            new_avg = sum(val_acc[i-5:i]) / 5
            if new_avg - old_avg > 20:  # 20% jump
                ax.axvline(x=iterations[i], color="purple", linestyle="--", alpha=0.7)
                ax.annotate(
                    "Grokking!",
                    xy=(iterations[i], val_acc[i]),
                    xytext=(iterations[i] + 2, val_acc[i] - 15),
                    fontsize=10,
                    color="purple",
                    arrowprops=dict(arrowstyle="->", color="purple"),
                )
                break

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")

    if show:
        plt.show()

    return fig


def plot_generalization_gap(
    metrics_file: Path,
    output_file: Optional[Path] = None,
    title: str = "Generalization Gap Over Time",
    show: bool = True,
):
    """
    Plot the generalization gap (train - val accuracy) over iterations.

    Args:
        metrics_file: Path to metrics.jsonl file
        output_file: Path to save the plot (optional)
        title: Plot title
        show: Whether to display the plot
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    # Load metrics
    iterations = []
    gaps = []

    with open(metrics_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                iterations.append(data["iteration"])
                gap = (data["train_accuracy"] - data["val_accuracy"]) * 100
                gaps.append(gap)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Fill area under curve
    ax.fill_between(iterations, 0, gaps, alpha=0.3, color="orange")
    ax.plot(iterations, gaps, "orange", linewidth=2, label="Generalization Gap")

    # Add zero line
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Styling
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Gap (Train - Val) %", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")

    if show:
        plt.show()

    return fig


def plot_accuracy_by_problem_type(
    results: list[dict],
    output_file: Optional[Path] = None,
    title: str = "Accuracy by Problem Type",
    show: bool = True,
):
    """
    Plot accuracy breakdown by problem type.

    Args:
        results: List of evaluation result dicts with problem_type
        output_file: Path to save the plot (optional)
        title: Plot title
        show: Whether to display the plot
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    # Aggregate by problem type
    by_type = {}
    for r in results:
        ptype = r.get("problem_type", "unknown")
        if ptype not in by_type:
            by_type[ptype] = {"correct": 0, "total": 0}
        by_type[ptype]["total"] += 1
        if r.get("correct"):
            by_type[ptype]["correct"] += 1

    # Calculate accuracies
    types = list(by_type.keys())
    accuracies = [by_type[t]["correct"] / by_type[t]["total"] * 100 for t in types]
    counts = [by_type[t]["total"] for t in types]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.bar(types, accuracies, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"])

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(
            f"n={count}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Styling
    ax.set_xlabel("Problem Type", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")

    if show:
        plt.show()

    return fig


def print_ascii_grokking_curve(
    metrics_file: Path,
    width: int = 60,
    height: int = 20,
):
    """
    Print an ASCII art representation of the grokking curve.

    For use when matplotlib is not available.
    """
    # Load metrics
    iterations = []
    train_acc = []
    val_acc = []

    with open(metrics_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                iterations.append(data["iteration"])
                train_acc.append(data["train_accuracy"])
                val_acc.append(data["val_accuracy"])

    if not iterations:
        print("No data to plot")
        return

    # Normalize to grid
    max_iter = max(iterations)
    min_iter = min(iterations)

    def to_x(iter_val):
        return int((iter_val - min_iter) / (max_iter - min_iter + 1) * (width - 1))

    def to_y(acc_val):
        return int((1 - acc_val) * (height - 1))

    # Create grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Plot train accuracy
    for i, (it, acc) in enumerate(zip(iterations, train_acc)):
        x = to_x(it)
        y = to_y(acc)
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = "T"

    # Plot val accuracy
    for i, (it, acc) in enumerate(zip(iterations, val_acc)):
        x = to_x(it)
        y = to_y(acc)
        if 0 <= x < width and 0 <= y < height:
            if grid[y][x] == "T":
                grid[y][x] = "*"
            else:
                grid[y][x] = "V"

    # Print grid
    print("\nGrokking Curve (T=Train, V=Val, *=Both)")
    print("=" * (width + 6))
    print(f"100% |{''.join(grid[0])}")
    for i, row in enumerate(grid[1:-1], 1):
        if i == height // 2:
            print(f" 50% |{''.join(row)}")
        else:
            print(f"     |{''.join(row)}")
    print(f"  0% |{''.join(grid[-1])}")
    print(f"     +{'-' * width}")
    print(f"      0{' ' * (width - 8)}{max_iter}")
    print("                  Iterations")
    print(f"\nFinal Train: {train_acc[-1]:.1%}  Final Val: {val_acc[-1]:.1%}")


def generate_experiment_report(
    experiment_dir: Path,
    output_file: Optional[Path] = None,
) -> str:
    """
    Generate a text report of experiment results.

    Args:
        experiment_dir: Directory containing experiment files
        output_file: Path to save report (optional)

    Returns:
        Report text
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("GROKKING EXPERIMENT REPORT")
    report_lines.append("=" * 60)

    # Load config
    config_file = experiment_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        report_lines.append("\nCONFIGURATION:")
        report_lines.append(f"  Experiment: {config.get('experiment_name', 'N/A')}")
        report_lines.append(f"  Base Model: {config.get('base_model', 'N/A')}")
        report_lines.append(f"  Train Size: {config.get('train_size', 'N/A')}")
        report_lines.append(f"  Val Size: {config.get('val_size', 'N/A')}")
        report_lines.append(f"  Problem Types: {config.get('problem_types', 'N/A')}")
        report_lines.append(f"  Difficulty: {config.get('min_difficulty', 'N/A')}-{config.get('max_difficulty', 'N/A')}")

    # Load metrics
    metrics_file = experiment_dir / "metrics.jsonl"
    if metrics_file.exists():
        metrics = []
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))

        if metrics:
            report_lines.append("\nRESULTS:")
            report_lines.append(f"  Total Iterations: {len(metrics)}")
            report_lines.append(f"  Final Train Accuracy: {metrics[-1]['train_accuracy']:.1%}")
            report_lines.append(f"  Final Val Accuracy: {metrics[-1]['val_accuracy']:.1%}")

            best_val = max(metrics, key=lambda m: m["val_accuracy"])
            report_lines.append(f"  Best Val Accuracy: {best_val['val_accuracy']:.1%} (iter {best_val['iteration']})")

            # Grokking detection
            report_lines.append("\nGROKKING ANALYSIS:")
            for i in range(10, len(metrics)):
                old_avg = sum(m["val_accuracy"] for m in metrics[i-10:i-5]) / 5
                new_avg = sum(m["val_accuracy"] for m in metrics[i-5:i]) / 5
                if new_avg - old_avg > 0.2:
                    report_lines.append(f"  Grokking detected at iteration {metrics[i]['iteration']}!")
                    report_lines.append(f"  Val accuracy jumped from {old_avg:.1%} to {new_avg:.1%}")
                    break
            else:
                report_lines.append("  No grokking detected (threshold: 20% val accuracy jump)")

    report_lines.append("\n" + "=" * 60)

    report = "\n".join(report_lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_file}")

    return report
