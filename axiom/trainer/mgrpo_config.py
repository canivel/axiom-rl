"""
Configuration for M-GRPO (Momentum-Anchored GRPO) training.

M-GRPO extends GRPO with:
1. Momentum model for stable pseudo-ground truth estimation
2. Combined sampling from policy and momentum models
3. IQR-based entropy filtering to prevent mode collapse
4. Optional Clip-Cov/KL-Cov entropy control mechanisms
"""

from dataclasses import dataclass, field
from typing import List

from .grpo_config import GRPOConfig


@dataclass
class MGRPOConfig(GRPOConfig):
    """
    Configuration for Momentum-Anchored GRPO.

    Inherits from GRPOConfig and adds M-GRPO specific settings.
    """

    # =========================================================
    # Momentum Model Settings
    # =========================================================

    # Momentum coefficient for EMA update: theta_k <- m * theta_k + (1-m) * theta_q
    # Higher values (0.99-0.999) = slower momentum updates = more stable
    momentum: float = 0.99

    # Number of samples from policy model (M)
    num_policy_samples: int = 4

    # Number of samples from momentum model (N)
    num_momentum_samples: int = 4

    # Whether to use majority voting for pseudo-ground truth
    use_majority_voting: bool = True

    # =========================================================
    # Entropy Filtering (IQR-based)
    # =========================================================

    # Enable IQR-based entropy filtering
    use_iqr_filter: bool = True

    # IQR multiplier: T_IQR = Q1 - k * (Q3 - Q1)
    # Lower k = more aggressive filtering (removes more low-entropy samples)
    iqr_k: float = 0.75

    # Minimum entropy threshold (absolute floor)
    min_entropy_threshold: float = 0.1

    # =========================================================
    # Clip-Cov: Gradient Detachment for High-Covariance Tokens
    # =========================================================

    # Enable Clip-Cov mechanism
    use_clip_cov: bool = False

    # Covariance threshold above which to detach gradients
    clip_cov_threshold: float = 0.5

    # Fraction of high-covariance tokens to detach (random selection)
    clip_cov_ratio: float = 0.1

    # =========================================================
    # KL-Cov: Targeted KL Penalty for High-Covariance Tokens
    # =========================================================

    # Enable KL-Cov mechanism
    use_kl_cov: bool = False

    # Weight for additional KL penalty on high-covariance tokens
    kl_cov_weight: float = 0.1

    # Threshold for identifying high-covariance tokens
    kl_cov_threshold: float = 0.5

    # =========================================================
    # Entropy Tracking and Logging
    # =========================================================

    # Track per-step entropy for analysis
    track_entropy: bool = True

    # Log entropy metrics every N steps
    entropy_log_steps: int = 10

    # Early stopping if entropy drops below this threshold
    entropy_collapse_threshold: float = 0.01

    # =========================================================
    # Training Overrides
    # =========================================================

    # Lower learning rate for stability
    learning_rate: float = 1e-5

    # Output directory
    output_dir: str = "models/mgrpo"

    # Maximum new tokens for generation
    max_new_tokens: int = 512

    # Generation temperature
    temperature: float = 0.7

    def __post_init__(self):
        """Validate configuration."""
        if self.momentum < 0 or self.momentum > 1:
            raise ValueError(f"momentum must be in [0, 1], got {self.momentum}")

        if self.num_policy_samples < 1:
            raise ValueError(f"num_policy_samples must be >= 1")

        if self.num_momentum_samples < 0:
            raise ValueError(f"num_momentum_samples must be >= 0")

        if self.iqr_k < 0:
            raise ValueError(f"iqr_k must be >= 0")

    @property
    def total_samples_per_prompt(self) -> int:
        """Total samples per prompt (M + N)."""
        return self.num_policy_samples + self.num_momentum_samples

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            # Model
            "model_name": self.model_name,
            "torch_dtype": self.torch_dtype,
            # LoRA
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            # GRPO base
            "num_generations": self.num_generations,
            "beta": self.beta,
            "epsilon": self.epsilon,
            # M-GRPO specific
            "momentum": self.momentum,
            "num_policy_samples": self.num_policy_samples,
            "num_momentum_samples": self.num_momentum_samples,
            "use_majority_voting": self.use_majority_voting,
            # Entropy filtering
            "use_iqr_filter": self.use_iqr_filter,
            "iqr_k": self.iqr_k,
            "min_entropy_threshold": self.min_entropy_threshold,
            # Clip-Cov
            "use_clip_cov": self.use_clip_cov,
            "clip_cov_threshold": self.clip_cov_threshold,
            "clip_cov_ratio": self.clip_cov_ratio,
            # KL-Cov
            "use_kl_cov": self.use_kl_cov,
            "kl_cov_weight": self.kl_cov_weight,
            "kl_cov_threshold": self.kl_cov_threshold,
            # Training
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "output_dir": self.output_dir,
        }
