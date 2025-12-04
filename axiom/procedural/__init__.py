"""Procedural problem generation for infinite training data.

V1 API (Legacy):
    - ProceduralProblem, ProceduralGenerator: Single test case per problem
    - GENERATORS: Registry of V1 generators
    - get_generator(), get_all_generators(): V1 factory functions

V2 API (Recommended):
    - AlgorithmicProblem, AlgorithmicGenerator, TestCase: Multiple test cases
    - GENERATORS_V2: Registry of V2 generators
    - get_generator_v2(), get_all_generators_v2(): V2 factory functions

V2 is recommended because it prevents models from memorizing answers.
See docs/phase7-v2-problem-design.md for details.
"""

# V1 API (Legacy - kept for backwards compatibility)
from .base import ProceduralProblem, ProceduralGenerator
from .arithmetic import ArithmeticGenerator
from .rpn import RPNGenerator
from .parentheses import ParenthesesGenerator
from .list_ops import ListSortGenerator, ListFilterGenerator, ListAggregateGenerator

# V2 API (Recommended)
from .base_v2 import AlgorithmicProblem, AlgorithmicGenerator, TestCase
from .generators_v2 import (
    RPNEvaluatorGenerator,
    ArithmeticEvaluatorGenerator,
    ParenthesesValidatorGenerator,
    GENERATORS_V2,
    get_generator_v2,
    get_all_generators_v2,
)

# V1 Registry (Legacy)
GENERATORS = {
    "arithmetic": ArithmeticGenerator,
    "rpn": RPNGenerator,
    "parentheses": ParenthesesGenerator,
    "list_sort": ListSortGenerator,
    "list_filter": ListFilterGenerator,
    "list_aggregate": ListAggregateGenerator,
}


def get_generator(problem_type: str, seed: int = None) -> ProceduralGenerator:
    """Get a V1 generator by problem type name (legacy)."""
    if problem_type not in GENERATORS:
        raise ValueError(f"Unknown problem type: {problem_type}. Available: {list(GENERATORS.keys())}")
    return GENERATORS[problem_type](seed=seed)


def get_all_generators(seed: int = None) -> dict[str, ProceduralGenerator]:
    """Get instances of all V1 generators (legacy)."""
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
    # V2 API (Recommended)
    "AlgorithmicProblem",
    "AlgorithmicGenerator",
    "TestCase",
    "RPNEvaluatorGenerator",
    "ArithmeticEvaluatorGenerator",
    "ParenthesesValidatorGenerator",
    "GENERATORS_V2",
    "get_generator_v2",
    "get_all_generators_v2",
    # V1 API (Legacy)
    "ProceduralProblem",
    "ProceduralGenerator",
    "ArithmeticGenerator",
    "RPNGenerator",
    "ParenthesesGenerator",
    "ListSortGenerator",
    "ListFilterGenerator",
    "ListAggregateGenerator",
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
