"""
Base classes for the axiom-rl benchmark framework.

This module provides abstract base classes for:
- BenchmarkLoader: Load benchmark datasets
- BenchmarkEvaluator: Evaluate model outputs
- Data classes for problems, results, and reports
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path
import json
from datetime import datetime


class BenchmarkType(Enum):
    """Types of benchmarks supported."""
    MATH = "math"
    CODE = "code"
    MCQ = "multiple_choice"
    REASONING = "reasoning"


@dataclass
class BenchmarkProblem:
    """
    A single benchmark problem.

    Attributes:
        id: Unique problem identifier
        question: The problem statement/prompt
        answer: Ground truth answer (format depends on benchmark type)
        metadata: Additional info (difficulty, subject, options for MCQ, etc.)
    """
    id: str
    question: str
    answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """
    Result for a single problem evaluation.

    Attributes:
        problem_id: ID of the problem
        correct: Whether the answer was correct
        model_answer: The answer extracted from model output
        expected_answer: The ground truth answer
        time_seconds: Time taken for this problem
        error: Any error message if evaluation failed
        raw_output: Full model output
    """
    problem_id: str
    correct: bool
    model_answer: Any
    expected_answer: Any
    time_seconds: float
    error: Optional[str] = None
    raw_output: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """
    Aggregate results for a complete benchmark run.

    Attributes:
        benchmark_name: Name of the benchmark (e.g., "math500")
        model_name: Model identifier or path
        timestamp: When the benchmark was run
        total: Total number of problems
        correct: Number of correct answers
        accuracy: Accuracy as a decimal (0.0-1.0)
        results: Per-problem results
        metadata: Additional benchmark-specific info
    """
    benchmark_name: str
    model_name: str
    timestamp: str
    total: int
    correct: int
    accuracy: float
    results: List[BenchmarkResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkReport":
        """Load report from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = [
            BenchmarkResult(**r) for r in data.pop("results", [])
        ]
        return cls(**data, results=results)

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"{self.benchmark_name}: {self.accuracy:.1%} "
            f"({self.correct}/{self.total})"
        )


class BenchmarkLoader(ABC):
    """
    Abstract base class for loading benchmark datasets.

    Subclasses must implement:
    - name: Benchmark identifier
    - benchmark_type: Type of benchmark (math, code, mcq)
    - load(): Load problems from the dataset
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this benchmark (e.g., 'math500')."""
        pass

    @property
    @abstractmethod
    def benchmark_type(self) -> BenchmarkType:
        """Type of benchmark (determines which evaluator to use)."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the benchmark."""
        return ""

    @abstractmethod
    def load(self, split: str = "test") -> List[BenchmarkProblem]:
        """
        Load benchmark problems.

        Args:
            split: Dataset split to load (usually "test")

        Returns:
            List of BenchmarkProblem instances
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class BenchmarkEvaluator(ABC):
    """
    Abstract base class for evaluating model outputs.

    Subclasses must implement:
    - extract_answer(): Extract answer from model output
    - evaluate(): Evaluate a single problem
    """

    @abstractmethod
    def extract_answer(self, output: str, problem: Optional[BenchmarkProblem] = None) -> Any:
        """
        Extract the answer from model output.

        Args:
            output: Raw model output string
            problem: Optional problem for context (e.g., MCQ options)

        Returns:
            Extracted answer (format depends on benchmark type)
        """
        pass

    @abstractmethod
    def check_answer(self, extracted: Any, expected: Any) -> bool:
        """
        Check if extracted answer matches expected.

        Args:
            extracted: Answer extracted from model output
            expected: Ground truth answer

        Returns:
            True if answers match
        """
        pass

    def evaluate(
        self,
        problem: BenchmarkProblem,
        model_output: str,
        time_seconds: float = 0.0
    ) -> BenchmarkResult:
        """
        Evaluate a single problem.

        Args:
            problem: The benchmark problem
            model_output: Raw model output
            time_seconds: Time taken for generation

        Returns:
            BenchmarkResult with evaluation outcome
        """
        try:
            extracted = self.extract_answer(model_output, problem)
            correct = self.check_answer(extracted, problem.answer)
            error = None
        except Exception as e:
            extracted = None
            correct = False
            error = str(e)

        return BenchmarkResult(
            problem_id=problem.id,
            correct=correct,
            model_answer=extracted,
            expected_answer=problem.answer,
            time_seconds=time_seconds,
            error=error,
            raw_output=model_output,
        )

    def evaluate_batch(
        self,
        problems: List[BenchmarkProblem],
        outputs: List[str],
        times: Optional[List[float]] = None,
    ) -> List[BenchmarkResult]:
        """
        Evaluate a batch of problems.

        Args:
            problems: List of benchmark problems
            outputs: List of model outputs (same order as problems)
            times: Optional list of generation times

        Returns:
            List of BenchmarkResult instances
        """
        if times is None:
            times = [0.0] * len(problems)

        return [
            self.evaluate(prob, out, t)
            for prob, out, t in zip(problems, outputs, times)
        ]


def create_report(
    benchmark_name: str,
    model_name: str,
    results: List[BenchmarkResult],
    metadata: Optional[Dict[str, Any]] = None,
) -> BenchmarkReport:
    """
    Create a BenchmarkReport from results.

    Args:
        benchmark_name: Name of the benchmark
        model_name: Model identifier
        results: List of per-problem results
        metadata: Optional additional metadata

    Returns:
        BenchmarkReport instance
    """
    correct = sum(1 for r in results if r.correct)
    total = len(results)

    return BenchmarkReport(
        benchmark_name=benchmark_name,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        total=total,
        correct=correct,
        accuracy=correct / total if total > 0 else 0.0,
        results=results,
        metadata=metadata or {},
    )
