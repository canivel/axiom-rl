"""Trainer module for fine-tuning models on verified solutions."""

from .data import SFTDataset, load_training_data
from .lora_config import get_lora_config, get_training_args
from .trainer import LoRASFTTrainer
from .grpo_config import GRPOConfig
from .grpo_trainer import GRPOTrainer
from .mgrpo_config import MGRPOConfig
from .mgrpo_trainer import MGRPOTrainer
from .entropy_utils import (
    EntropyTracker,
    EntropyMetrics,
    compute_trajectory_entropy,
    compute_batch_entropy,
    iqr_filter,
    compute_entropy_metrics,
)
from .curriculum import (
    CurriculumScheduler,
    CurriculumConfig,
    CurriculumStrategy,
    DifficultyLevel,
    create_default_curriculum,
)

__all__ = [
    "SFTDataset",
    "load_training_data",
    "get_lora_config",
    "get_training_args",
    "LoRASFTTrainer",
    "GRPOConfig",
    "GRPOTrainer",
    # M-GRPO
    "MGRPOConfig",
    "MGRPOTrainer",
    "EntropyTracker",
    "EntropyMetrics",
    "compute_trajectory_entropy",
    "compute_batch_entropy",
    "iqr_filter",
    "compute_entropy_metrics",
    # Curriculum
    "CurriculumScheduler",
    "CurriculumConfig",
    "CurriculumStrategy",
    "DifficultyLevel",
    "create_default_curriculum",
]
