"""
Entropy utilities for M-GRPO training.

Implements:
1. Trajectory entropy computation
2. IQR-based entropy filtering
3. Covariance computation for Clip-Cov/KL-Cov
4. Entropy tracking and analysis

Based on:
- M-GRPO paper: IQR filtering to preserve high-entropy samples
- Entropy Mechanism paper: Covariance-based entropy control
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EntropyMetrics:
    """Metrics for entropy tracking."""
    mean_entropy: float
    std_entropy: float
    min_entropy: float
    max_entropy: float
    num_filtered: int = 0
    iqr_threshold: Optional[float] = None
    covariances: Optional[List[float]] = None


class EntropyTracker:
    """Track entropy over training for analysis."""

    def __init__(self):
        self.history: List[EntropyMetrics] = []
        self.step_entropies: List[List[float]] = []

    def log(self, metrics: EntropyMetrics, step: int):
        """Log entropy metrics for a step."""
        self.history.append(metrics)

    def get_entropy_trajectory(self) -> List[float]:
        """Get mean entropy over training."""
        return [m.mean_entropy for m in self.history]

    def detect_collapse(self, threshold: float = 0.01) -> Optional[int]:
        """
        Detect entropy collapse.

        Returns step index where entropy first drops below threshold,
        or None if no collapse detected.
        """
        for i, m in enumerate(self.history):
            if m.mean_entropy < threshold:
                return i
        return None

    def summary(self) -> Dict:
        """Get summary statistics."""
        if not self.history:
            return {}

        entropies = [m.mean_entropy for m in self.history]
        return {
            "initial_entropy": entropies[0] if entropies else 0,
            "final_entropy": entropies[-1] if entropies else 0,
            "min_entropy": min(entropies) if entropies else 0,
            "max_entropy": max(entropies) if entropies else 0,
            "entropy_change": entropies[-1] - entropies[0] if len(entropies) > 1 else 0,
            "total_filtered": sum(m.num_filtered for m in self.history),
        }


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy from logits.

    Args:
        logits: Shape (batch, seq_len, vocab_size)

    Returns:
        Entropy per token, shape (batch, seq_len)
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy


def compute_trajectory_entropy(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device: Optional[str] = None,
) -> float:
    """
    Compute average entropy for a generated trajectory.

    Args:
        model: The policy model
        tokenizer: Tokenizer
        prompt: Input prompt
        completion: Generated completion
        device: Device to use (auto-detected if None)

    Returns:
        Average entropy across completion tokens
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize full sequence
    full_text = prompt + completion
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get logits for completion tokens only
    completion_logits = logits[:, prompt_len - 1:-1, :]  # Shifted for next-token prediction

    # Compute entropy
    entropy = compute_token_entropy(completion_logits)

    # Return mean entropy
    return entropy.mean().item()


def compute_batch_entropy(
    model,
    tokenizer,
    prompts: List[str],
    completions: List[str],
) -> List[float]:
    """
    Compute entropy for a batch of generations.

    Args:
        model: The policy model
        tokenizer: Tokenizer
        prompts: List of prompts
        completions: List of completions

    Returns:
        List of entropy values
    """
    return [
        compute_trajectory_entropy(model, tokenizer, p, c)
        for p, c in zip(prompts, completions)
    ]


def iqr_filter(
    generations: List[str],
    entropies: List[float],
    k: float = 0.75,
    min_threshold: float = 0.1,
) -> Tuple[List[str], List[float], int]:
    """
    IQR-based filtering to remove low-entropy outliers.

    From M-GRPO paper: Remove samples with entropy below T_IQR = Q1 - k*(Q3-Q1)

    Args:
        generations: List of generated texts
        entropies: Corresponding entropy values
        k: IQR multiplier (default 0.75)
        min_threshold: Absolute minimum entropy threshold

    Returns:
        (filtered_generations, filtered_entropies, num_removed)
    """
    if not entropies:
        return generations, entropies, 0

    # Compute IQR threshold
    entropies_arr = np.array(entropies)
    Q1 = np.percentile(entropies_arr, 25)
    Q3 = np.percentile(entropies_arr, 75)
    T_IQR = max(Q1 - k * (Q3 - Q1), min_threshold)

    # Filter
    mask = entropies_arr >= T_IQR
    filtered_gens = [g for g, m in zip(generations, mask) if m]
    filtered_ents = [e for e, m in zip(entropies, mask) if m]
    num_removed = len(generations) - len(filtered_gens)

    return filtered_gens, filtered_ents, num_removed


def compute_logit_covariance(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device: Optional[str] = None,
) -> List[float]:
    """
    Compute covariance between log-prob and logit change for each token.

    From Entropy Mechanism paper: High covariance indicates tokens that
    drive entropy collapse.

    Args:
        model: The policy model
        tokenizer: Tokenizer
        prompt: Input prompt
        completion: Generated completion
        device: Device to use

    Returns:
        List of covariance values per token
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize
    full_text = prompt + completion
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get completion logits
    completion_logits = logits[:, prompt_len - 1:-1, :]
    seq_len = completion_logits.shape[1]

    # Compute log probs
    log_probs = F.log_softmax(completion_logits, dim=-1)

    # Get the actual next token indices
    next_tokens = inputs.input_ids[:, prompt_len:]  # Shape: (1, seq_len)

    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

    # For covariance, we need logit changes which requires computing gradients
    # This is a simplified version that uses the magnitude of logits as proxy
    token_logits = completion_logits.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

    # Compute covariance per token position
    # Cov(log_prob, logit) for each token
    covariances = []
    for i in range(seq_len):
        lp = token_log_probs[0, i].item()
        lg = token_logits[0, i].item()
        # Simple product as covariance proxy
        cov = lp * lg
        covariances.append(abs(cov))

    return covariances


def identify_high_covariance_tokens(
    covariances: List[float],
    threshold: float = 0.5,
    ratio: float = 0.1,
) -> List[int]:
    """
    Identify token indices with high covariance.

    Args:
        covariances: List of covariance values
        threshold: Absolute threshold for high covariance
        ratio: Fraction of top tokens to select

    Returns:
        List of token indices to apply entropy control
    """
    if not covariances:
        return []

    # Get indices of tokens above threshold
    high_cov_indices = [
        i for i, c in enumerate(covariances) if c > threshold
    ]

    # Also include top ratio% by covariance magnitude
    sorted_indices = sorted(range(len(covariances)), key=lambda i: covariances[i], reverse=True)
    top_k = max(1, int(len(covariances) * ratio))
    top_indices = sorted_indices[:top_k]

    # Combine and deduplicate
    all_indices = set(high_cov_indices) | set(top_indices)

    return sorted(all_indices)


def compute_entropy_metrics(
    entropies: List[float],
    filtered_count: int = 0,
    iqr_threshold: Optional[float] = None,
    covariances: Optional[List[List[float]]] = None,
) -> EntropyMetrics:
    """
    Compute entropy metrics for logging.

    Args:
        entropies: List of entropy values
        filtered_count: Number of samples filtered out
        iqr_threshold: The IQR threshold used
        covariances: Optional covariance values

    Returns:
        EntropyMetrics object
    """
    if not entropies:
        return EntropyMetrics(
            mean_entropy=0.0,
            std_entropy=0.0,
            min_entropy=0.0,
            max_entropy=0.0,
            num_filtered=filtered_count,
        )

    entropies_arr = np.array(entropies)

    # Flatten covariances if provided
    flat_cov = None
    if covariances:
        flat_cov = [c for cov_list in covariances for c in cov_list]

    return EntropyMetrics(
        mean_entropy=float(entropies_arr.mean()),
        std_entropy=float(entropies_arr.std()),
        min_entropy=float(entropies_arr.min()),
        max_entropy=float(entropies_arr.max()),
        num_filtered=filtered_count,
        iqr_threshold=iqr_threshold,
        covariances=flat_cov,
    )
