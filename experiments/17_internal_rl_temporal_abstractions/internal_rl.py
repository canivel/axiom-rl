"""
Internal Reinforcement Learning Script (Phase 3)

This script performs RL in the abstract action space discovered by the metacontroller.

Key Innovation:
==============
Instead of doing RL on individual tokens (50K+ action space, 100+ timesteps),
we do RL on abstract actions (16D latent space, ~5 timesteps).

This dramatically reduces:
- Action space complexity: |Z| << |Vocabulary|
- Effective horizon: M << T
- Policy gradient variance: scales with M*|Z| not T*|V|

Architecture During Internal RL:
===============================

  Observation (residual e_t)
         │
         ▼
  [Abstract Action Policy π(z|e)] ← THIS GETS TRAINED
         │
         ▼
  Abstract Action z_t
         │
         ▼
  [Switching Unit] → β_t (determines when to get new z)
         │
         ▼
  [Controller Decoder] → U_t
         │
         ▼
  Modified Residual ê_t = e_t + U_t @ e_t
         │
         ▼
  [Remaining Base Model Layers] (FROZEN)
         │
         ▼
  Token a_t
         │
         ▼
  [Environment] → Observation, Reward

The "environment" from the RL perspective includes:
- The frozen base model
- The switching unit
- The controller decoder
- The actual code execution environment

Usage:
    python internal_rl.py --metacontroller_path models/metacontroller_best.pt
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metacontroller import (
    Metacontroller,
    AbstractActionPolicy
)


@dataclass
class InternalRLConfig:
    """Configuration for Internal RL training."""
    # Model paths
    base_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    metacontroller_path: str = "experiments/17_internal_rl_temporal_abstractions/models/metacontroller_best.pt"

    # Architecture
    controller_layer: int = 12
    latent_dim: int = 16
    policy_hidden_dim: int = 256

    # RL parameters
    batch_size: int = 16
    num_episodes: int = 100000
    learning_rate: float = 3e-5
    gamma: float = 1.0  # No discounting for sparse rewards
    clip_epsilon: float = 0.2  # PPO clip
    entropy_coef: float = 0.01

    # Switching
    beta_threshold: float = 0.5  # Threshold for discrete switching

    # Generation
    max_tokens: int = 256
    temperature: float = 0.7

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "experiments/17_internal_rl_temporal_abstractions/models"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class InternalRLEnvironment:
    """
    Internal RL Environment.

    From the RL perspective, we have:
    - State: Residual stream activation e_t
    - Action: Abstract action z (latent code)
    - Step: Run until switching gate β > threshold
    - Reward: Binary success (code passes tests)

    The key insight is that one abstract action z generates MULTIPLE tokens,
    until the switching unit decides it's time for a new abstract action.
    """

    def __init__(
        self,
        base_model: nn.Module,
        metacontroller: Metacontroller,
        tokenizer,
        problems: List[Dict],
        config: InternalRLConfig
    ):
        self.base_model = base_model
        self.metacontroller = metacontroller
        self.tokenizer = tokenizer
        self.problems = problems
        self.config = config

        self.device = torch.device(config.device)

        # Current episode state
        self.current_problem = None
        self.current_prompt = None
        self.generated_tokens = []
        self.h_state = None  # GRU state
        self.current_z = None  # Current abstract action
        self.current_residual = None

    def reset(self, problem_idx: Optional[int] = None) -> torch.Tensor:
        """
        Reset environment with a new problem.

        Returns:
            Initial residual stream observation
        """
        # Select problem
        if problem_idx is None:
            problem_idx = torch.randint(len(self.problems), (1,)).item()

        self.current_problem = self.problems[problem_idx]

        # Format prompt
        self.current_prompt = self._format_prompt(self.current_problem['description'])

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            self.current_prompt,
            return_tensors='pt'
        )['input_ids'].to(self.device)

        self.generated_tokens = prompt_tokens[0].tolist()

        # Get initial residual
        self.current_residual = self._get_residual(prompt_tokens)

        # Initialize metacontroller state
        batch_size = 1
        self.h_state, self.current_z = self.metacontroller.init_state(
            batch_size, self.device
        )

        return self.current_residual

    def step(self, z: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take an abstract action.

        The abstract action z is applied until the switching unit decides
        to switch (β > threshold). This generates multiple tokens.

        Args:
            z: Abstract action [batch, latent_dim]

        Returns:
            next_obs: Next residual observation
            reward: 0 until episode ends, then binary success
            done: Whether episode is finished
            info: Additional information
        """
        accumulated_reward = 0.0
        tokens_generated = 0
        done = False

        # Run until switch or max tokens or EOS
        while not done:
            # Get switching probability
            beta = self.metacontroller.switching_unit(
                self.current_residual,
                self.h_state.squeeze(0) if self.h_state.dim() == 3 else self.h_state,
                self.current_z
            )

            # Check for switch
            if beta.item() > self.config.beta_threshold and tokens_generated > 0:
                # Time to switch to new abstract action
                break

            # Apply controller
            A, B = self.metacontroller.decoder(z)
            controlled_residual = self.metacontroller.decoder.apply_controller(
                self.current_residual, A, B
            )

            # Generate next token
            next_token, next_residual = self._generate_token(controlled_residual)

            # Update state
            self.generated_tokens.append(next_token)
            self.current_residual = next_residual
            self.current_z = z

            # Update GRU state
            self.h_state = self.metacontroller.gru(
                self.current_residual,
                self.h_state.squeeze(0) if self.h_state.dim() == 3 else self.h_state
            )
            if self.h_state.dim() == 2:
                self.h_state = self.h_state.unsqueeze(0)

            tokens_generated += 1

            # Check termination conditions
            if next_token == self.tokenizer.eos_token_id:
                done = True
            elif len(self.generated_tokens) >= self.config.max_tokens:
                done = True

        # Compute reward only at episode end
        if done:
            reward = self._evaluate_solution()
        else:
            reward = 0.0

        info = {
            'tokens_generated': tokens_generated,
            'total_tokens': len(self.generated_tokens),
            'beta': beta.item() if 'beta' in dir() else 0.0
        }

        return self.current_residual, reward, done, info

    def _format_prompt(self, problem: str) -> str:
        """Format problem description as prompt."""
        return f"""Write a Python function to solve this problem.
Do NOT use a class wrapper. Write a standalone function.

Problem: {problem}

Solution:
```python
"""

    def _get_residual(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get residual stream activation at controller layer."""
        residual = None

        def hook(module, input, output):
            nonlocal residual
            if isinstance(output, tuple):
                residual = output[0][:, -1, :]  # Last position
            else:
                residual = output[:, -1, :]

        layer = self.base_model.model.layers[self.config.controller_layer]
        handle = layer.register_forward_hook(hook)

        try:
            with torch.no_grad():
                _ = self.base_model(input_ids)
        finally:
            handle.remove()

        return residual

    def _generate_token(self, controlled_residual: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Generate next token using controlled residual.

        This is simplified - full implementation would inject the controlled
        residual back into the model's forward pass.
        """
        # Get current sequence
        input_ids = torch.tensor([self.generated_tokens], device=self.device)

        # For now, standard generation (full implementation would use controlled residual)
        with torch.no_grad():
            outputs = self.base_model(
                input_ids,
                output_hidden_states=True
            )
            logits = outputs.logits[:, -1, :]

            # Sample
            if self.config.temperature > 0:
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = logits.argmax(dim=-1).item()

            # Get next residual
            next_residual = outputs.hidden_states[self.config.controller_layer + 1][:, -1, :]

        return next_token, next_residual

    def _evaluate_solution(self) -> float:
        """
        Evaluate the generated solution.

        Returns 1.0 if solution passes tests, 0.0 otherwise.
        """
        # Decode generated tokens
        generated_text = self.tokenizer.decode(
            self.generated_tokens,
            skip_special_tokens=True
        )

        # Extract code
        code = self._extract_code(generated_text)

        # Run tests
        try:
            test_cases = self.current_problem.get('test_cases', [])
            if not test_cases:
                return 0.0

            # Execute code
            namespace = {}
            exec(code, namespace)

            # Run each test
            entry_point = self.current_problem.get('entry_point', 'solution')
            func = namespace.get(entry_point)

            if func is None:
                return 0.0

            passed = 0
            for test in test_cases:
                try:
                    result = func(*test['input'])
                    if result == test['expected']:
                        passed += 1
                except:
                    pass

            return passed / len(test_cases)

        except Exception as e:
            return 0.0

    def _extract_code(self, text: str) -> str:
        """Extract Python code from generated text."""
        # Look for code between ```python and ```
        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Otherwise take everything after "Solution:"
        if "Solution:" in text:
            return text.split("Solution:")[-1].strip()

        return text


class InternalRLTrainer:
    """
    Trainer for Internal RL.

    Uses GRPO-style training but on abstract actions instead of tokens.
    """

    def __init__(
        self,
        env: InternalRLEnvironment,
        policy: AbstractActionPolicy,
        config: InternalRLConfig
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.device = torch.device(config.device)

        # Optimizer
        self.optimizer = AdamW(
            policy.parameters(),
            lr=config.learning_rate
        )

        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def collect_trajectory(self) -> Dict[str, Any]:
        """
        Collect a single trajectory using the current policy.

        Returns:
            Dictionary containing:
            - observations: List of residual observations
            - actions: List of abstract actions taken
            - log_probs: Log probabilities of actions
            - rewards: List of rewards
            - total_reward: Episode return
        """
        observations = []
        actions = []
        log_probs = []
        rewards = []

        # Reset environment
        obs = self.env.reset()
        h = self.policy.init_hidden(1, self.device)
        done = False

        while not done:
            observations.append(obs)

            # Get action from policy
            with torch.no_grad():
                mu, logvar, h = self.policy(obs, h)
                z = self.policy.sample(mu, logvar)
                log_prob = self.policy.log_prob(z, mu, logvar)

            actions.append(z)
            log_probs.append(log_prob)

            # Take action
            obs, reward, done, info = self.env.step(z)
            rewards.append(reward)

        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(len(actions))

        return {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'total_reward': total_reward
        }

    def compute_advantages(self, trajectories: List[Dict]) -> torch.Tensor:
        """
        Compute advantages using batch normalization (GRPO-style).

        For sparse rewards, advantage = (R - mean(R)) / std(R)
        """
        returns = torch.tensor([t['total_reward'] for t in trajectories])

        if returns.std() > 1e-8:
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            advantages = returns - returns.mean()

        return advantages

    def train_step(self, batch_size: int = 16) -> Dict[str, float]:
        """
        Single training step: collect batch and update policy.
        """
        # Collect trajectories
        trajectories = []
        for _ in range(batch_size):
            traj = self.collect_trajectory()
            trajectories.append(traj)

        # Compute advantages
        advantages = self.compute_advantages(trajectories)

        # Flatten for batch processing
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []

        for i, traj in enumerate(trajectories):
            for j, (obs, action, log_prob) in enumerate(zip(
                traj['observations'],
                traj['actions'],
                traj['log_probs']
            )):
                all_obs.append(obs)
                all_actions.append(action)
                all_old_log_probs.append(log_prob)
                all_advantages.append(advantages[i])

        if len(all_obs) == 0:
            return {'loss': 0.0, 'mean_reward': 0.0}

        # Stack tensors
        obs_batch = torch.cat(all_obs, dim=0)
        action_batch = torch.cat(all_actions, dim=0)
        old_log_prob_batch = torch.cat(all_old_log_probs, dim=0)
        advantage_batch = torch.tensor(all_advantages, device=self.device)

        # Forward pass
        mu, logvar, _ = self.policy(obs_batch)
        new_log_probs = self.policy.log_prob(action_batch, mu, logvar)

        # PPO objective
        ratio = torch.exp(new_log_probs - old_log_prob_batch)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)

        policy_loss = -torch.min(
            ratio * advantage_batch,
            clipped_ratio * advantage_batch
        ).mean()

        # Entropy bonus
        entropy = 0.5 * (1 + math.log(2 * math.pi) + logvar).sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_coef * entropy

        # Total loss
        loss = policy_loss + entropy_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        mean_reward = sum(t['total_reward'] for t in trajectories) / len(trajectories)

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'mean_reward': mean_reward,
            'mean_episode_length': sum(len(t['actions']) for t in trajectories) / len(trajectories)
        }

    def train(self, num_steps: int):
        """Main training loop."""
        print("=" * 60)
        print("INTERNAL RL TRAINING (Phase 3)")
        print("=" * 60)
        print(f"Device: {self.config.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print()

        for step in range(num_steps):
            metrics = self.train_step(self.config.batch_size)

            if step % self.config.log_interval == 0:
                avg_reward = sum(self.episode_rewards) / max(len(self.episode_rewards), 1)
                avg_length = sum(self.episode_lengths) / max(len(self.episode_lengths), 1)

                print(f"Step {step} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Reward: {metrics['mean_reward']:.3f} | "
                      f"Avg Reward (100): {avg_reward:.3f} | "
                      f"Avg Length: {avg_length:.1f}")

            if step % self.config.save_interval == 0 and step > 0:
                self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(num_steps, final=True)

    def save_checkpoint(self, step: int, final: bool = False):
        """Save policy checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        if final:
            path = os.path.join(self.config.output_dir, "policy_final.pt")
        else:
            path = os.path.join(self.config.output_dir, f"policy_step{step}.pt")

        torch.save({
            'step': step,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'config': self.config.__dict__
        }, path)

        print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metacontroller_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    args = parser.parse_args()

    config = InternalRLConfig(
        metacontroller_path=args.metacontroller_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    device = torch.device(config.device)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        device_map="auto" if config.device == "cuda" else None
    )
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    embed_dim = base_model.config.hidden_size

    # Load metacontroller
    print(f"Loading metacontroller from {config.metacontroller_path}...")
    mc_checkpoint = torch.load(config.metacontroller_path, map_location=device)
    mc_config = mc_checkpoint.get('config', {})

    metacontroller = Metacontroller(
        embed_dim=embed_dim,
        latent_dim=mc_config.get('latent_dim', config.latent_dim),
        gru_dim=mc_config.get('gru_dim', 64),
        seq_embed_dim=mc_config.get('seq_embed_dim', 64)
    ).to(device)
    metacontroller.load_state_dict(mc_checkpoint['metacontroller_state_dict'])

    # Freeze metacontroller (except we'll train the policy separately)
    for param in metacontroller.parameters():
        param.requires_grad = False
    metacontroller.eval()

    # Load problems
    print("Loading problems...")
    from axiom.problems import PROBLEMS
    problems = []
    for problem_type, prob_list in PROBLEMS.items():
        for prob in prob_list:
            prob['problem_type'] = problem_type
            problems.append(prob)
    print(f"Loaded {len(problems)} problems")

    # Create environment
    env = InternalRLEnvironment(
        base_model=base_model,
        metacontroller=metacontroller,
        tokenizer=tokenizer,
        problems=problems,
        config=config
    )

    # Create policy
    policy = AbstractActionPolicy(
        embed_dim=embed_dim,
        hidden_dim=config.policy_hidden_dim,
        latent_dim=config.latent_dim
    ).to(device)

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create trainer and train
    trainer = InternalRLTrainer(env, policy, config)
    trainer.train(args.num_steps)


if __name__ == "__main__":
    main()
