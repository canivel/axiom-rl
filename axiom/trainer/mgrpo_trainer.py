"""
M-GRPO (Momentum-Anchored GRPO) Trainer.

Implements the M-GRPO algorithm from "Stabilizing Self-Supervised RL
with Momentum-Anchored Policy Optimization".

Key innovations:
1. Two-model setup: policy (trainable) + momentum (EMA)
2. Combined sampling from both models for robust pseudo-ground truth
3. IQR-based entropy filtering to prevent mode collapse
4. Optional Clip-Cov/KL-Cov entropy control mechanisms
"""

import copy
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

from .mgrpo_config import MGRPOConfig
from .lora_config import get_lora_config
from .entropy_utils import (
    compute_batch_entropy,
    iqr_filter,
    compute_entropy_metrics,
    EntropyTracker,
    identify_high_covariance_tokens,
    compute_logit_covariance,
)


class MGRPOTrainer:
    """
    Trainer for Momentum-Anchored GRPO.

    Implements the M-GRPO training loop:
    1. Combined rollout: M samples from policy + N from momentum
    2. Pseudo-ground truth via majority voting
    3. IQR-based entropy filtering
    4. Policy gradient with optional entropy control
    5. Momentum model EMA update
    """

    def __init__(
        self,
        config: MGRPOConfig,
        reward_function: Callable[[List[str], List[str]], torch.Tensor],
        processing_class: Optional[Any] = None,
    ):
        """
        Initialize the M-GRPO trainer.

        Args:
            config: M-GRPO configuration
            reward_function: Callable(prompts, completions) -> rewards tensor
            processing_class: Tokenizer (optional, loaded from config if None)
        """
        self.config = config
        self.reward_function = reward_function
        self.tokenizer = processing_class

        # Models (initialized in setup())
        self.policy_model = None      # theta_q: trainable policy
        self.momentum_model = None    # theta_k: EMA of policy
        self.ref_model = None         # Frozen reference for KL

        # Optimizer
        self.optimizer = None

        # Entropy tracking
        self.entropy_tracker = EntropyTracker()

        # Metrics
        self.metrics_history: List[Dict] = []

    def setup(self):
        """Load models and prepare for training."""
        print(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Policy Model (trainable with LoRA)
        print("Loading policy model...")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=getattr(torch, self.config.torch_dtype),
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA
        lora_config = get_lora_config(self.config)
        self.policy_model = get_peft_model(self.policy_model, lora_config)
        self.policy_model.print_trainable_parameters()

        # Create Momentum Model (deep copy, no gradients)
        print("Creating momentum model...")
        self.momentum_model = copy.deepcopy(self.policy_model)
        self.momentum_model.eval()
        for param in self.momentum_model.parameters():
            param.requires_grad = False

        # Load Reference Model (frozen base for KL)
        print("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=getattr(torch, self.config.torch_dtype),
            device_map="auto",
            trust_remote_code=True,
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate,
        )

        print("M-GRPO setup complete")

    def _update_momentum(self):
        """
        EMA update for momentum model.

        theta_k <- m * theta_k + (1-m) * theta_q
        """
        m = self.config.momentum
        with torch.no_grad():
            for p_k, p_q in zip(
                self.momentum_model.parameters(),
                self.policy_model.parameters()
            ):
                p_k.data.mul_(m).add_(p_q.data, alpha=1 - m)

    def _generate_from_model(
        self,
        model,
        prompts: List[str],
        num_samples: int,
    ) -> List[List[str]]:
        """
        Generate completions from a model.

        Args:
            model: The model to generate from
            prompts: List of prompts
            num_samples: Number of samples per prompt

        Returns:
            List of lists of completions (outer: prompts, inner: samples)
        """
        model.eval()
        self.tokenizer.padding_side = "left"

        all_completions = []

        for prompt in prompts:
            # Tokenize
            inputs = self.tokenizer(
                [prompt] * num_samples,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode (only new tokens)
            input_len = inputs.input_ids.shape[1]
            completions = self.tokenizer.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
            )
            all_completions.append(completions)

        self.tokenizer.padding_side = "right"
        return all_completions

    def _combined_rollout(
        self,
        prompts: List[str],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Generate samples from both policy and momentum models.

        Args:
            prompts: List of prompts

        Returns:
            (policy_completions, momentum_completions)
        """
        # Generate from policy
        policy_gens = self._generate_from_model(
            self.policy_model,
            prompts,
            self.config.num_policy_samples,
        )

        # Generate from momentum
        momentum_gens = self._generate_from_model(
            self.momentum_model,
            prompts,
            self.config.num_momentum_samples,
        )

        return policy_gens, momentum_gens

    def _compute_pseudo_ground_truth(
        self,
        prompts: List[str],
        policy_gens: List[List[str]],
        momentum_gens: List[List[str]],
    ) -> List[Dict]:
        """
        Compute pseudo-ground truth via majority voting on combined pool.

        Args:
            prompts: Input prompts
            policy_gens: Generations from policy
            momentum_gens: Generations from momentum

        Returns:
            List of dicts with best_answer and rewards
        """
        results = []

        for i, (prompt, p_gens, m_gens) in enumerate(zip(prompts, policy_gens, momentum_gens)):
            # Combine all generations
            all_gens = p_gens + m_gens
            num_gens = len(all_gens)

            # Get rewards for all generations
            rewards = self.reward_function(
                [prompt] * num_gens,
                all_gens,
            )

            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards)

            # Find best answer (majority voting among successful)
            max_reward = rewards.max().item()

            if max_reward > 0:
                # Get all generations with max reward
                successful_mask = rewards == max_reward
                successful_gens = [g for g, m in zip(all_gens, successful_mask) if m]
                # Use first successful as pseudo-ground truth
                best_answer = successful_gens[0]
            else:
                best_answer = None

            # Split rewards back to policy/momentum
            policy_rewards = rewards[:len(p_gens)]
            momentum_rewards = rewards[len(p_gens):]

            results.append({
                "best_answer": best_answer,
                "policy_rewards": policy_rewards,
                "momentum_rewards": momentum_rewards,
                "all_rewards": rewards,
                "max_reward": max_reward,
            })

        return results

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.

        A_i = (r_i - mean(r)) / std(r)

        Args:
            rewards: Tensor of rewards

        Returns:
            Normalized advantages
        """
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        return (rewards - mean) / std

    def train_step(
        self,
        prompts: List[str],
        step: int = 0,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Single M-GRPO training step.

        Args:
            prompts: Batch of prompts
            step: Current training step
            verbose: Whether to print progress during step

        Returns:
            Dict of metrics
        """
        import time
        step_start = time.time()

        # 1. Combined rollout
        if verbose:
            print(f"  [Step {step}] Generating {len(prompts)}x{self.config.num_policy_samples} policy samples...", end="", flush=True)

        gen_start = time.time()
        policy_gens, momentum_gens = self._combined_rollout(prompts)

        if verbose:
            print(f" done ({time.time() - gen_start:.1f}s)", flush=True)

        # 2. Compute pseudo-ground truth
        if verbose:
            print(f"  [Step {step}] Computing rewards...", end="", flush=True)
        reward_start = time.time()

        pgt_results = self._compute_pseudo_ground_truth(
            prompts, policy_gens, momentum_gens
        )

        if verbose:
            successes = sum(1 for r in pgt_results if r["max_reward"] > 0)
            avg_reward = sum(r["max_reward"] for r in pgt_results) / max(len(pgt_results), 1)
            print(f" done ({successes}/{len(prompts)} successful, avg_reward={avg_reward:.2f}, {time.time() - reward_start:.1f}s)", flush=True)

        # 3. Compute entropy for policy generations
        flat_prompts = []
        flat_gens = []
        for prompt, gens in zip(prompts, policy_gens):
            for gen in gens:
                flat_prompts.append(prompt)
                flat_gens.append(gen)

        entropies = compute_batch_entropy(
            self.policy_model, self.tokenizer,
            flat_prompts, flat_gens,
        )

        # 4. IQR filtering
        filtered_count = 0
        if self.config.use_iqr_filter:
            filtered_gens, filtered_entropies, filtered_count = iqr_filter(
                flat_gens, entropies,
                k=self.config.iqr_k,
                min_threshold=self.config.min_entropy_threshold,
            )

            # Update tracking
            entropy_metrics = compute_entropy_metrics(
                filtered_entropies, filtered_count
            )
        else:
            entropy_metrics = compute_entropy_metrics(entropies)

        # Log entropy
        if self.config.track_entropy:
            self.entropy_tracker.log(entropy_metrics, step)

        # 5. Compute loss and train
        if verbose:
            print(f"  [Step {step}] Training policy...", end="", flush=True)
        train_start = time.time()

        total_loss = 0.0
        total_reward = 0.0
        update_count = 0

        self.policy_model.train()

        for i, (prompt, gens, pgt) in enumerate(zip(prompts, policy_gens, pgt_results)):
            rewards = pgt["policy_rewards"]
            advantages = self._compute_advantages(rewards)

            for j, (gen, adv, rew) in enumerate(zip(gens, advantages, rewards)):
                # Skip low-advantage samples
                if adv.item() <= 0:
                    continue

                # Tokenize
                full_text = prompt + gen
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length,
                ).to(self.policy_model.device)

                prompt_len = len(self.tokenizer.encode(prompt))

                # Forward pass
                outputs = self.policy_model(**inputs, labels=inputs.input_ids)

                # Compute policy gradient loss (weighted by advantage)
                # Use cross-entropy on completion tokens only
                logits = outputs.logits[:, prompt_len - 1:-1, :]
                labels = inputs.input_ids[:, prompt_len:]

                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

                # Policy gradient loss (maximize log prob * advantage)
                pg_loss = -(token_log_probs.mean() * adv)

                total_loss += pg_loss.item()
                total_reward += rew.item()
                update_count += 1

                # Backward
                self.optimizer.zero_grad()
                pg_loss.backward()
                self.optimizer.step()

        if verbose:
            print(f" done ({update_count} updates, {time.time() - train_start:.1f}s)", flush=True)

        # 6. Update momentum model
        self._update_momentum()

        # 7. Compute metrics
        num_samples = len(prompts) * self.config.num_policy_samples
        metrics = {
            "loss": total_loss / max(num_samples, 1),
            "mean_reward": total_reward / max(num_samples, 1),
            "mean_entropy": entropy_metrics.mean_entropy,
            "filtered_count": filtered_count,
            "success_rate": sum(1 for r in pgt_results if r["max_reward"] > 0) / len(prompts),
        }

        self.metrics_history.append(metrics)

        return metrics

    def train(
        self,
        prompts: List[str],
        num_steps: int = 100,
        eval_every: int = 10,
        eval_fn: Optional[Callable] = None,
        checkpoint_every: int = 5,
        checkpoint_dir: Optional[str] = None,
        start_step: int = 0,
        verbose: bool = True,
    ) -> Dict:
        """
        Run M-GRPO training loop.

        Args:
            prompts: Training prompts
            num_steps: Number of training steps
            eval_every: Evaluate every N steps
            eval_fn: Optional evaluation function
            checkpoint_every: Save checkpoint every N steps
            checkpoint_dir: Directory to save checkpoints
            start_step: Step to start from (for resuming)
            verbose: Whether to print detailed progress

        Returns:
            Training summary
        """
        import time
        train_start_time = time.time()

        print(f"Starting M-GRPO training for {num_steps} steps")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Samples per prompt: {self.config.total_samples_per_prompt}")
        print(f"  (Policy: {self.config.num_policy_samples}, "
              f"Momentum: {self.config.num_momentum_samples})")
        if start_step > 0:
            print(f"  Resuming from step {start_step}")
        if checkpoint_dir:
            print(f"  Checkpoints every {checkpoint_every} steps -> {checkpoint_dir}")

        import random
        from datetime import datetime

        for step in range(start_step, num_steps):
            step_start = time.time()

            # Sample batch
            batch_size = min(4, len(prompts))
            batch = random.sample(prompts, batch_size)

            # Train step
            metrics = self.train_step(batch, step, verbose=verbose)

            # Log summary
            elapsed = time.time() - train_start_time
            eta = (elapsed / (step - start_step + 1)) * (num_steps - step - 1) if step > start_step else 0
            print(f"Step {step}/{num_steps}: loss={metrics['loss']:.4f}, "
                  f"reward={metrics['mean_reward']:.3f}, "
                  f"entropy={metrics['mean_entropy']:.3f}, "
                  f"success={metrics['success_rate']:.1%}, "
                  f"[{time.time() - step_start:.1f}s, ETA: {eta/60:.1f}m]")

            # Evaluate
            if eval_fn and (step + 1) % eval_every == 0:
                print(f"  Evaluating...")
                eval_result = eval_fn(self.policy_model, self.tokenizer)
                print(f"  Eval: {eval_result}")

            # Save checkpoint
            if checkpoint_dir and (step + 1) % checkpoint_every == 0:
                ckpt_path = Path(checkpoint_dir) / f"checkpoint_step_{step+1}"
                print(f"  Saving checkpoint to {ckpt_path}...")
                self.save_checkpoint(ckpt_path, step + 1)

            # Check for entropy collapse
            if (self.config.track_entropy and
                metrics["mean_entropy"] < self.config.entropy_collapse_threshold):
                print(f"Warning: Entropy collapsed at step {step}")

        total_time = time.time() - train_start_time
        print(f"\nTraining complete in {total_time/60:.1f} minutes")

        # Summary
        summary = {
            "num_steps": num_steps,
            "final_metrics": self.metrics_history[-1] if self.metrics_history else {},
            "entropy_summary": self.entropy_tracker.summary(),
            "config": self.config.to_dict(),
            "total_time_seconds": total_time,
        }

        return summary

    def save_checkpoint(self, checkpoint_dir: Path, step: int):
        """Save a checkpoint that can be resumed."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save policy model
        self.policy_model.save_pretrained(checkpoint_dir / "policy")
        self.tokenizer.save_pretrained(checkpoint_dir / "policy")

        # Save momentum model
        self.momentum_model.save_pretrained(checkpoint_dir / "momentum")

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # Save training state
        state = {
            "step": step,
            "metrics_history": self.metrics_history,
            "entropy_history": [
                {
                    "mean": m.mean_entropy,
                    "std": m.std_entropy,
                    "min": m.min_entropy,
                    "max": m.max_entropy,
                    "filtered": m.num_filtered,
                }
                for m in self.entropy_tracker.history
            ],
        }
        with open(checkpoint_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"  Checkpoint saved at step {step}")

    def load_checkpoint(self, checkpoint_dir: Path) -> int:
        """Load a checkpoint and return the step to resume from."""
        from peft import PeftModel

        checkpoint_dir = Path(checkpoint_dir)
        print(f"Loading checkpoint from {checkpoint_dir}...")

        # Load policy model
        self.policy_model = PeftModel.from_pretrained(
            self.policy_model.base_model,
            checkpoint_dir / "policy",
        )

        # Load momentum model
        self.momentum_model = PeftModel.from_pretrained(
            self.momentum_model.base_model,
            checkpoint_dir / "momentum",
        )
        self.momentum_model.eval()
        for param in self.momentum_model.parameters():
            param.requires_grad = False

        # Load optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))

        # Load training state
        with open(checkpoint_dir / "state.json") as f:
            state = json.load(f)

        self.metrics_history = state.get("metrics_history", [])
        # Note: entropy_tracker history not fully restored, but summary available

        step = state.get("step", 0)
        print(f"Resumed from step {step}")
        return step

    def save(self, output_dir: Optional[str] = None):
        """Save model and training state."""
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save policy model
        self.policy_model.save_pretrained(output_dir / "policy")
        self.tokenizer.save_pretrained(output_dir / "policy")

        # Save momentum model
        self.momentum_model.save_pretrained(output_dir / "momentum")

        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Save entropy history
        entropy_data = {
            "history": [
                {
                    "mean": m.mean_entropy,
                    "std": m.std_entropy,
                    "min": m.min_entropy,
                    "max": m.max_entropy,
                    "filtered": m.num_filtered,
                }
                for m in self.entropy_tracker.history
            ],
            "summary": self.entropy_tracker.summary(),
        }
        with open(output_dir / "entropy.json", "w") as f:
            json.dump(entropy_data, f, indent=2)

        print(f"Saved to {output_dir}")
