"""
Benchmark loaders for various datasets.

Each loader is automatically registered when imported.
"""

from .math500 import MATH500Loader
from .aime import AIME24Loader, AIME25Loader
from .gpqa import GPQALoader, GPQADiamondLoader
from .livecodebench import LiveCodeBenchLoader

__all__ = [
    "MATH500Loader",
    "AIME24Loader",
    "AIME25Loader",
    "GPQALoader",
    "GPQADiamondLoader",
    "LiveCodeBenchLoader",
]
