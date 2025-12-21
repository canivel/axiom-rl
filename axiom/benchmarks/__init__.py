"""
Axiom-RL Benchmark Framework.

A reusable benchmark framework for evaluating LLMs on standard benchmarks:

**Math Benchmarks:**
- MATH500: 500 competition math problems (levels 1-5)
- AIME24: 30 problems from AIME 2024
- AIME25: 30 problems from AIME 2025

**Reasoning Benchmarks:**
- GPQA: 448 graduate-level science MCQ
- GPQA Diamond: 198 hardest GPQA questions

**Code Benchmarks:**
- LiveCodeBench: Competitive programming problems

Usage:
    # Run from command line
    python -m axiom.benchmarks.run --model <path> --benchmarks all

    # Use programmatically
    from axiom.benchmarks import run_benchmark, list_benchmarks

    # List available benchmarks
    print(list_benchmarks())

    # Run a benchmark
    report = run_benchmark(model, tokenizer, "math500", output_dir)
    print(f"Accuracy: {report.accuracy:.1%}")
"""

# Import base classes
from .base import (
    BenchmarkType,
    BenchmarkProblem,
    BenchmarkResult,
    BenchmarkReport,
    BenchmarkLoader,
    BenchmarkEvaluator,
    create_report,
)

# Import registry functions
from .registry import (
    register_loader,
    register_evaluator,
    get_loader,
    get_evaluator,
    list_benchmarks,
    get_benchmark_info,
)

# Import loaders (auto-registers them)
from . import loaders

# Import evaluators (auto-registers them)
from . import evaluators

# Import CLI functions
from .run import run_benchmark, run_all_benchmarks

__all__ = [
    # Base classes
    "BenchmarkType",
    "BenchmarkProblem",
    "BenchmarkResult",
    "BenchmarkReport",
    "BenchmarkLoader",
    "BenchmarkEvaluator",
    "create_report",
    # Registry
    "register_loader",
    "register_evaluator",
    "get_loader",
    "get_evaluator",
    "list_benchmarks",
    "get_benchmark_info",
    # CLI
    "run_benchmark",
    "run_all_benchmarks",
]
