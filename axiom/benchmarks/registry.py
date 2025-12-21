"""
Benchmark registry for axiom-rl.

Provides decorators and functions for registering and retrieving
benchmark loaders and evaluators.
"""

from typing import Dict, Type, List, Optional

from .base import BenchmarkLoader, BenchmarkEvaluator, BenchmarkType


# Global registries
_LOADERS: Dict[str, Type[BenchmarkLoader]] = {}
_EVALUATORS: Dict[BenchmarkType, Type[BenchmarkEvaluator]] = {}


def register_loader(name: str):
    """
    Decorator to register a benchmark loader.

    Usage:
        @register_loader("math500")
        class MATH500Loader(BenchmarkLoader):
            ...

    Args:
        name: Unique identifier for the benchmark
    """
    def decorator(cls: Type[BenchmarkLoader]):
        if name in _LOADERS:
            raise ValueError(f"Loader '{name}' already registered")
        _LOADERS[name] = cls
        return cls
    return decorator


def register_evaluator(benchmark_type: BenchmarkType):
    """
    Decorator to register a benchmark evaluator.

    Usage:
        @register_evaluator(BenchmarkType.MATH)
        class MathEvaluator(BenchmarkEvaluator):
            ...

    Args:
        benchmark_type: Type of benchmark this evaluator handles
    """
    def decorator(cls: Type[BenchmarkEvaluator]):
        if benchmark_type in _EVALUATORS:
            raise ValueError(f"Evaluator for '{benchmark_type}' already registered")
        _EVALUATORS[benchmark_type] = cls
        return cls
    return decorator


def get_loader(name: str) -> BenchmarkLoader:
    """
    Get a benchmark loader by name.

    Args:
        name: Benchmark identifier (e.g., "math500", "gpqa_diamond")

    Returns:
        Instance of the loader class

    Raises:
        ValueError: If benchmark not found
    """
    if name not in _LOADERS:
        available = list_benchmarks()
        raise ValueError(
            f"Unknown benchmark: '{name}'. "
            f"Available: {available}"
        )
    return _LOADERS[name]()


def get_evaluator(benchmark_type: BenchmarkType) -> BenchmarkEvaluator:
    """
    Get an evaluator for a benchmark type.

    Args:
        benchmark_type: Type of benchmark (MATH, CODE, MCQ)

    Returns:
        Instance of the evaluator class

    Raises:
        ValueError: If no evaluator for this type
    """
    if benchmark_type not in _EVALUATORS:
        available = list(_EVALUATORS.keys())
        raise ValueError(
            f"No evaluator for type: '{benchmark_type}'. "
            f"Available: {available}"
        )
    return _EVALUATORS[benchmark_type]()


def list_benchmarks() -> List[str]:
    """
    List all registered benchmark names.

    Returns:
        List of benchmark identifiers
    """
    return sorted(_LOADERS.keys())


def get_benchmark_info() -> Dict[str, dict]:
    """
    Get information about all registered benchmarks.

    Returns:
        Dict mapping benchmark names to their info
    """
    info = {}
    for name, loader_cls in _LOADERS.items():
        loader = loader_cls()
        info[name] = {
            "name": loader.name,
            "type": loader.benchmark_type.value,
            "description": loader.description,
        }
    return info


def has_evaluator(benchmark_type: BenchmarkType) -> bool:
    """Check if an evaluator exists for a benchmark type."""
    return benchmark_type in _EVALUATORS


def clear_registry():
    """Clear all registrations (useful for testing)."""
    _LOADERS.clear()
    _EVALUATORS.clear()
