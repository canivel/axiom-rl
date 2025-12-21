"""
Benchmark evaluators for different problem types.

Each evaluator is automatically registered when imported.
"""

from .math_evaluator import MathEvaluator
from .mcq_evaluator import MCQEvaluator
from .code_evaluator import CodeEvaluator

__all__ = [
    "MathEvaluator",
    "MCQEvaluator",
    "CodeEvaluator",
]
