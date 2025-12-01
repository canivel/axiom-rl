"""Configuration dataclasses for axiom-rl."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GeneratorConfig:
    """Configuration for the code generator."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    device: str = "auto"  # "cuda", "cpu", "mps", or "auto"
    torch_dtype: str = "float16"  # "float16", "bfloat16", "float32", or "auto"

    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    num_samples: int = 8  # Best-of-N: generate N solutions per problem


@dataclass
class VerifierConfig:
    """Configuration for the verifier."""

    timeout: float = 5.0  # Execution timeout in seconds
    python_executable: str = "python"


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/synthetic"))
    output_file: str = "solutions.jsonl"

    # Pipeline settings
    problems_path: Optional[Path] = None  # None = use default bundled problems
    max_attempts_per_problem: int = 3  # Retry if no solution passes
    save_failed: bool = False  # Whether to save failed attempts

    def __post_init__(self):
        """Ensure output_dir is a Path."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
