"""Procedural problem generation for infinite training data."""

from .base import ProceduralProblem, ProceduralGenerator
from .arithmetic import ArithmeticGenerator
from .rpn import RPNGenerator
from .parentheses import ParenthesesGenerator
from .list_ops import ListSortGenerator, ListFilterGenerator, ListAggregateGenerator

# Registry of all available generators
GENERATORS = {
    "arithmetic": ArithmeticGenerator,
    "rpn": RPNGenerator,
    "parentheses": ParenthesesGenerator,
    "list_sort": ListSortGenerator,
    "list_filter": ListFilterGenerator,
    "list_aggregate": ListAggregateGenerator,
}


def get_generator(problem_type: str, seed: int = None) -> ProceduralGenerator:
    """Get a generator by problem type name."""
    if problem_type not in GENERATORS:
        raise ValueError(f"Unknown problem type: {problem_type}. Available: {list(GENERATORS.keys())}")
    return GENERATORS[problem_type](seed=seed)


def get_all_generators(seed: int = None) -> dict[str, ProceduralGenerator]:
    """Get instances of all generators."""
    return {name: cls(seed=seed) for name, cls in GENERATORS.items()}


# Import integration utilities (after GENERATORS is defined)
from .trainer_integration import (
    ProceduralTrainingSample,
    ProceduralDataStream,
    ProceduralVerifier,
    procedural_to_training_sample,
    generate_training_file,
)


__all__ = [
    # Base classes
    "ProceduralProblem",
    "ProceduralGenerator",
    # Generators
    "ArithmeticGenerator",
    "RPNGenerator",
    "ParenthesesGenerator",
    "ListSortGenerator",
    "ListFilterGenerator",
    "ListAggregateGenerator",
    # Registry and helpers
    "GENERATORS",
    "get_generator",
    "get_all_generators",
    # Training integration
    "ProceduralTrainingSample",
    "ProceduralDataStream",
    "ProceduralVerifier",
    "procedural_to_training_sample",
    "generate_training_file",
]
