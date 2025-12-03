"""Configuration for GRPO training."""

from dataclasses import dataclass
from .lora_config import LoRASFTConfig

@dataclass
class GRPOConfig(LoRASFTConfig):
    """Configuration for Group Relative Policy Optimization."""
    
    # GRPO specific settings
    num_generations: int = 16  # G: Number of samples per prompt
    beta: float = 0.04  # KL coefficient
    epsilon: float = 0.2  # Clip range
    
    # Overrides for RL
    learning_rate: float = 1e-6  # Typically lower for RL
    output_dir: str = "models/grpo-v1"
