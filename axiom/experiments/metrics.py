"""Metrics tracking and logging for experiments."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentMetrics:
    """Metrics for a single evaluation point."""

    iteration: int
    timestamp: str

    # Accuracy metrics
    train_accuracy: float
    val_accuracy: float
    test_accuracy: Optional[float] = None

    # Training metrics
    train_loss: Optional[float] = None
    learning_rate: Optional[float] = None

    # Problem-level breakdown
    train_correct: int = 0
    train_total: int = 0
    val_correct: int = 0
    val_total: int = 0
    test_correct: int = 0
    test_total: int = 0

    # Derived metrics
    generalization_gap: float = field(init=False)

    def __post_init__(self):
        self.generalization_gap = self.train_accuracy - self.val_accuracy

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class MetricsLogger:
    """Logs experiment metrics to file and tracks history."""

    def __init__(self, experiment_dir: Path, experiment_name: str):
        """
        Initialize the metrics logger.

        Args:
            experiment_dir: Directory to save metrics
            experiment_name: Name of the experiment
        """
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.metrics_file = experiment_dir / "metrics.jsonl"
        self.history: list[ExperimentMetrics] = []

        # Create directory if needed
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metrics if resuming
        if self.metrics_file.exists():
            self._load_history()

    def _load_history(self):
        """Load existing metrics from file."""
        with open(self.metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Reconstruct ExperimentMetrics
                    metrics = ExperimentMetrics(
                        iteration=data["iteration"],
                        timestamp=data["timestamp"],
                        train_accuracy=data["train_accuracy"],
                        val_accuracy=data["val_accuracy"],
                        test_accuracy=data.get("test_accuracy"),
                        train_loss=data.get("train_loss"),
                        learning_rate=data.get("learning_rate"),
                        train_correct=data.get("train_correct", 0),
                        train_total=data.get("train_total", 0),
                        val_correct=data.get("val_correct", 0),
                        val_total=data.get("val_total", 0),
                        test_correct=data.get("test_correct", 0),
                        test_total=data.get("test_total", 0),
                    )
                    self.history.append(metrics)

    def log(self, metrics: ExperimentMetrics):
        """Log metrics to file and history."""
        self.history.append(metrics)

        # Append to file
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")

    def log_values(
        self,
        iteration: int,
        train_accuracy: float,
        val_accuracy: float,
        test_accuracy: Optional[float] = None,
        train_loss: Optional[float] = None,
        **kwargs,
    ):
        """Convenience method to log metrics by values."""
        metrics = ExperimentMetrics(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            test_accuracy=test_accuracy,
            train_loss=train_loss,
            **kwargs,
        )
        self.log(metrics)

    def get_latest(self) -> Optional[ExperimentMetrics]:
        """Get the most recent metrics."""
        return self.history[-1] if self.history else None

    def get_best_val_accuracy(self) -> tuple[int, float]:
        """Get iteration and value of best validation accuracy."""
        if not self.history:
            return 0, 0.0

        best = max(self.history, key=lambda m: m.val_accuracy)
        return best.iteration, best.val_accuracy

    def detect_grokking(
        self,
        threshold: float = 0.2,
        window: int = 10,
    ) -> Optional[int]:
        """
        Detect if grokking has occurred.

        Returns the iteration where grokking was detected, or None.

        Grokking signature:
        1. Training accuracy plateaued at high value
        2. Validation accuracy suddenly jumps
        3. Gap between train and val closes rapidly
        """
        if len(self.history) < 2 * window:
            return None

        for i in range(window, len(self.history)):
            recent = self.history[i - window : i]
            older = self.history[max(0, i - 2 * window) : i - window]

            if not older:
                continue

            # Check for sudden val_accuracy improvement
            old_val = sum(m.val_accuracy for m in older) / len(older)
            new_val = sum(m.val_accuracy for m in recent) / len(recent)

            if (new_val - old_val) > threshold:
                return self.history[i].iteration

        return None

    def get_summary(self) -> dict:
        """Get summary statistics of the experiment."""
        if not self.history:
            return {}

        return {
            "experiment_name": self.experiment_name,
            "total_iterations": len(self.history),
            "final_train_accuracy": self.history[-1].train_accuracy,
            "final_val_accuracy": self.history[-1].val_accuracy,
            "best_val_accuracy": max(m.val_accuracy for m in self.history),
            "best_val_iteration": max(
                self.history, key=lambda m: m.val_accuracy
            ).iteration,
            "grokking_detected": self.detect_grokking() is not None,
            "grokking_iteration": self.detect_grokking(),
        }

    def export_for_plotting(self) -> dict:
        """Export data in format suitable for plotting."""
        return {
            "iterations": [m.iteration for m in self.history],
            "train_accuracy": [m.train_accuracy for m in self.history],
            "val_accuracy": [m.val_accuracy for m in self.history],
            "test_accuracy": [m.test_accuracy for m in self.history if m.test_accuracy],
            "generalization_gap": [m.generalization_gap for m in self.history],
            "train_loss": [m.train_loss for m in self.history if m.train_loss],
        }
