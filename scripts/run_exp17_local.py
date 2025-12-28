#!/usr/bin/env python3
"""
Experiment 17: Internal RL with Temporal Abstractions - Local Training Script

Optimized for RTX 3080 (10GB VRAM). Runs all three phases:
1. Base model (already pretrained - Qwen 0.5B)
2. Metacontroller training (self-supervised)
3. Internal RL (policy training in abstract action space)

Usage:
    uv run python scripts/run_exp17_local.py --phase 2  # Train metacontroller only
    uv run python scripts/run_exp17_local.py --phase 3  # Train RL policy (requires metacontroller)
    uv run python scripts/run_exp17_local.py --phase all  # Run all phases
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.problems import ProblemDataset


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Configuration optimized for RTX 3080."""
    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    controller_layer: int = 12  # Mid-depth for Qwen 0.5B (24 layers)

    # Metacontroller architecture
    latent_dim: int = 16
    gru_dim: int = 64
    seq_embed_dim: int = 64
    controller_rank: int = 32

    # Phase 2: Metacontroller training
    mc_batch_size: int = 4  # Small batch for VRAM efficiency
    mc_epochs: int = 10
    mc_lr: float = 1e-4
    kl_weight: float = 0.01

    # Phase 3: Internal RL
    rl_batch_size: int = 8  # Episodes per batch
    rl_steps: int = 500  # Training steps
    rl_lr: float = 3e-5
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01

    # Generation
    max_tokens: int = 256
    temperature: float = 0.7

    # Paths
    output_dir: str = "experiments/17_internal_rl_temporal_abstractions/models"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# METACONTROLLER ARCHITECTURE
# ==============================================================================

class SequenceEmbedder(nn.Module):
    """Embeds sequence for acausal context."""
    def __init__(self, embed_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return self.proj(attn_out.mean(dim=1))


class ControllerEncoder(nn.Module):
    """VAE encoder: maps to mean and variance."""
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class SwitchingUnit(nn.Module):
    """Predicts switching probability β."""
    def __init__(self, embed_dim: int, gru_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + gru_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, e: torch.Tensor, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([e, h, z], dim=-1))


class ControllerDecoder(nn.Module):
    """Hypernetwork: generates low-rank controller matrices."""
    def __init__(self, latent_dim: int, embed_dim: int, rank: int = 32):
        super().__init__()
        self.rank = rank
        self.embed_dim = embed_dim
        self.A_net = nn.Linear(latent_dim, embed_dim * rank)
        self.B_net = nn.Linear(latent_dim, rank * embed_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = z.shape[0]
        A = self.A_net(z).view(batch, self.embed_dim, self.rank)
        B = self.B_net(z).view(batch, self.rank, self.embed_dim)
        return A, B

    def apply_controller(self, e: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Apply controller: ê = e + A @ B @ e"""
        if e.dim() == 2:
            e = e.unsqueeze(-1)
            result = e + torch.bmm(A, torch.bmm(B, e))
            return result.squeeze(-1)
        return e + torch.bmm(A, torch.bmm(B, e.unsqueeze(-1))).squeeze(-1)


class Metacontroller(nn.Module):
    """Full metacontroller combining all components."""
    def __init__(self, embed_dim: int, latent_dim: int = 16, gru_dim: int = 64,
                 seq_embed_dim: int = 64, controller_rank: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.gru_dim = gru_dim

        # Components
        self.gru = nn.GRUCell(embed_dim, gru_dim)
        self.seq_embedder = SequenceEmbedder(embed_dim, seq_embed_dim)
        self.encoder = ControllerEncoder(gru_dim + seq_embed_dim, latent_dim)
        self.switching_unit = SwitchingUnit(embed_dim, gru_dim, latent_dim)
        self.decoder = ControllerDecoder(latent_dim, embed_dim, controller_rank)

    def init_state(self, batch_size: int, device: torch.device):
        """Initialize hidden state and z."""
        h = torch.zeros(batch_size, self.gru_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def forward(self, residuals: torch.Tensor, h: torch.Tensor, z_prev: torch.Tensor):
        """
        Forward pass for single timestep.

        Args:
            residuals: [batch, seq_len, embed_dim] or [batch, embed_dim]
            h: [batch, gru_dim] GRU hidden state
            z_prev: [batch, latent_dim] previous abstract action

        Returns:
            z_new, h_new, beta, A, B, mu, logvar
        """
        if residuals.dim() == 3:
            e_t = residuals[:, -1, :]  # Last position
            seq_embed = self.seq_embedder(residuals)
        else:
            e_t = residuals
            seq_embed = torch.zeros(e_t.shape[0], self.seq_embedder.proj.out_features,
                                   device=e_t.device)

        # Update GRU
        h_new = self.gru(e_t, h)

        # Encode to latent
        encoder_input = torch.cat([h_new, seq_embed], dim=-1)
        mu, logvar = self.encoder(encoder_input)

        # Sample z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_tilde = mu + eps * std

        # Switching probability
        beta = self.switching_unit(e_t, h_new, z_prev)

        # Temporal integration
        z_new = beta * z_tilde + (1 - beta) * z_prev

        # Decode to controller
        A, B = self.decoder(z_new)

        return z_new, h_new, beta, A, B, mu, logvar


class AbstractActionPolicy(nn.Module):
    """Policy network for Internal RL (Phase 3)."""
    def __init__(self, embed_dim: int, hidden_dim: int = 256, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs: torch.Tensor, h: Optional[torch.Tensor] = None):
        if h is None:
            h = self.init_hidden(obs.shape[0], obs.device)
        h_new = self.gru(obs, h)
        mu = self.mu_head(h_new)
        logvar = self.logvar_head(h_new)
        return mu, logvar, h_new

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        var = torch.exp(logvar)
        return -0.5 * ((z - mu)**2 / var + logvar + math.log(2 * math.pi)).sum(dim=-1)


# ==============================================================================
# PROBLEM DEFINITIONS
# ==============================================================================

def get_training_problems() -> List[Dict]:
    """Get problems formatted for training."""
    dataset = ProblemDataset()
    problems = []
    for prob in dataset:
        # Convert test cases to expected format
        test_cases = []
        for tc in prob.test_cases:
            # Handle input format - could be single value or list
            inp = tc.input if isinstance(tc.input, (list, tuple)) else [tc.input]
            test_cases.append({
                'input': inp,
                'expected': tc.expected_output
            })
        problems.append({
            'description': prob.description,
            'test_cases': test_cases,
            'entry_point': prob.function_name,
            'problem_type': prob.difficulty,
            'title': prob.title
        })
    return problems


# ==============================================================================
# PHASE 2: METACONTROLLER TRAINING
# ==============================================================================

def get_residuals_at_layer(model, tokenizer, text: str, layer: int, device: torch.device) -> torch.Tensor:
    """Get residual stream activations at specified layer."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings


def train_metacontroller(config: Config):
    """Phase 2: Train metacontroller on expert solutions."""
    print("=" * 60)
    print("PHASE 2: METACONTROLLER TRAINING")
    print("=" * 60)

    device = torch.device(config.device)

    # Load model and tokenizer
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    embed_dim = model.config.hidden_size
    print(f"Embed dim: {embed_dim}")

    # Create metacontroller
    metacontroller = Metacontroller(
        embed_dim=embed_dim,
        latent_dim=config.latent_dim,
        gru_dim=config.gru_dim,
        seq_embed_dim=config.seq_embed_dim,
        controller_rank=config.controller_rank
    ).to(device)

    print(f"Metacontroller parameters: {sum(p.numel() for p in metacontroller.parameters()):,}")

    optimizer = AdamW(metacontroller.parameters(), lr=config.mc_lr)

    # Get problems and generate expert solutions
    problems = get_training_problems()
    print(f"Training on {len(problems)} problems")

    # Training loop
    best_loss = float('inf')

    for epoch in range(config.mc_epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for i, problem in enumerate(problems):
            # Create prompt with problem
            prompt = f"""Write a Python function to solve this problem.

Problem: {problem['description']}

Solution:
```python
def solution"""

            # Get residuals from base model
            try:
                residuals = get_residuals_at_layer(
                    model, tokenizer, prompt, config.controller_layer, device
                )
            except Exception as e:
                print(f"Skipping problem {i}: {e}")
                continue

            if residuals.shape[1] < 2:
                continue

            # Forward through metacontroller
            batch_size = 1
            h, z = metacontroller.init_state(batch_size, device)

            total_recon_loss = 0.0
            total_kl_loss = 0.0
            seq_len = residuals.shape[1]

            for t in range(seq_len):
                e_t = residuals[:, t, :].float()

                z_new, h, beta, A, B, mu, logvar = metacontroller(
                    e_t, h, z
                )

                # Reconstruction: controlled residual should reconstruct next token's residual
                controlled_e = metacontroller.decoder.apply_controller(e_t, A, B)

                if t < seq_len - 1:
                    target_e = residuals[:, t + 1, :].float()
                    recon_loss = F.mse_loss(controlled_e, target_e)
                    total_recon_loss += recon_loss

                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                total_kl_loss += kl_loss

                z = z_new

            # Average losses
            avg_recon = total_recon_loss / max(seq_len - 1, 1)
            avg_kl = total_kl_loss / seq_len

            loss = avg_recon + config.kl_weight * avg_kl

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(metacontroller.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += avg_recon.item()
            epoch_kl += avg_kl.item()
            num_batches += 1

            if i % 10 == 0:
                print(f"  Problem {i}/{len(problems)} | Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_epoch_recon = epoch_recon / max(num_batches, 1)
        avg_epoch_kl = epoch_kl / max(num_batches, 1)

        print(f"\nEpoch {epoch + 1}/{config.mc_epochs}")
        print(f"  Loss: {avg_epoch_loss:.4f} | Recon: {avg_epoch_recon:.4f} | KL: {avg_epoch_kl:.4f}")

        # Save best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            os.makedirs(config.output_dir, exist_ok=True)
            save_path = os.path.join(config.output_dir, "metacontroller_best.pt")
            torch.save({
                'epoch': epoch,
                'metacontroller_state_dict': metacontroller.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'embed_dim': embed_dim,
                    'latent_dim': config.latent_dim,
                    'gru_dim': config.gru_dim,
                    'seq_embed_dim': config.seq_embed_dim,
                    'controller_rank': config.controller_rank
                }
            }, save_path)
            print(f"  Saved best model to {save_path}")

    print("\nPhase 2 complete!")
    return metacontroller


# ==============================================================================
# PHASE 3: INTERNAL RL TRAINING
# ==============================================================================

class InternalRLEnvironment:
    """Environment for Internal RL."""

    def __init__(self, model, metacontroller, tokenizer, problems, config):
        self.model = model
        self.metacontroller = metacontroller
        self.tokenizer = tokenizer
        self.problems = problems
        self.config = config
        self.device = torch.device(config.device)

        self.current_problem = None
        self.generated_tokens = []
        self.h_state = None
        self.current_z = None
        self.current_residual = None

    def reset(self, problem_idx: Optional[int] = None) -> torch.Tensor:
        """Reset with new problem."""
        if problem_idx is None:
            problem_idx = torch.randint(len(self.problems), (1,)).item()

        self.current_problem = self.problems[problem_idx]

        prompt = f"""Write a Python function to solve this problem.
Do NOT use a class wrapper. Write a standalone function.

Problem: {self.current_problem['description']}

Solution:
```python
"""

        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.generated_tokens = inputs['input_ids'][0].tolist()

        # Get initial residual
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            self.current_residual = outputs.hidden_states[self.config.controller_layer + 1][:, -1, :].float()

        # Init metacontroller state
        self.h_state, self.current_z = self.metacontroller.init_state(1, self.device)

        return self.current_residual

    def step(self, z: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take abstract action."""
        tokens_generated = 0
        done = False
        beta_threshold = 0.5

        while not done and tokens_generated < 20:  # Max tokens per abstract action
            # Get switching probability
            beta = self.metacontroller.switching_unit(
                self.current_residual, self.h_state, self.current_z
            )

            if beta.item() > beta_threshold and tokens_generated > 0:
                break

            # Generate next token
            input_ids = torch.tensor([self.generated_tokens], device=self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                logits = outputs.logits[:, -1, :].float()

                if self.config.temperature > 0:
                    probs = F.softmax(logits / self.config.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = logits.argmax(dim=-1).item()

                self.current_residual = outputs.hidden_states[self.config.controller_layer + 1][:, -1, :].float()

            self.generated_tokens.append(next_token)
            tokens_generated += 1

            # Update state
            self.h_state = self.metacontroller.gru(self.current_residual, self.h_state)
            self.current_z = z

            # Check termination
            if next_token == self.tokenizer.eos_token_id:
                done = True
            elif len(self.generated_tokens) >= self.config.max_tokens:
                done = True

        reward = self._evaluate() if done else 0.0

        return self.current_residual, reward, done, {'tokens': tokens_generated}

    def _evaluate(self) -> float:
        """Evaluate generated solution."""
        text = self.tokenizer.decode(self.generated_tokens, skip_special_tokens=True)

        # Extract code
        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            code = text[start:end].strip() if end > start else text
        else:
            code = text.split("Solution:")[-1].strip() if "Solution:" in text else text

        try:
            test_cases = self.current_problem.get('test_cases', [])
            if not test_cases:
                return 0.0

            namespace = {}
            exec(code, namespace)

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
        except:
            return 0.0


def train_internal_rl(config: Config, metacontroller_path: Optional[str] = None):
    """Phase 3: Train policy with Internal RL."""
    print("=" * 60)
    print("PHASE 3: INTERNAL RL TRAINING")
    print("=" * 60)

    device = torch.device(config.device)

    # Load model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    embed_dim = model.config.hidden_size

    # Load metacontroller
    mc_path = metacontroller_path or os.path.join(config.output_dir, "metacontroller_best.pt")
    print(f"Loading metacontroller from {mc_path}...")

    checkpoint = torch.load(mc_path, map_location=device)
    mc_config = checkpoint.get('config', {})

    metacontroller = Metacontroller(
        embed_dim=mc_config.get('embed_dim', embed_dim),
        latent_dim=mc_config.get('latent_dim', config.latent_dim),
        gru_dim=mc_config.get('gru_dim', config.gru_dim),
        seq_embed_dim=mc_config.get('seq_embed_dim', config.seq_embed_dim),
        controller_rank=mc_config.get('controller_rank', config.controller_rank)
    ).to(device)
    metacontroller.load_state_dict(checkpoint['metacontroller_state_dict'])
    metacontroller.eval()
    for param in metacontroller.parameters():
        param.requires_grad = False

    # Create policy
    policy = AbstractActionPolicy(
        embed_dim=embed_dim,
        hidden_dim=256,
        latent_dim=config.latent_dim
    ).to(device)

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    optimizer = AdamW(policy.parameters(), lr=config.rl_lr)

    # Create environment
    problems = get_training_problems()
    env = InternalRLEnvironment(model, metacontroller, tokenizer, problems, config)

    # Training metrics
    episode_rewards = deque(maxlen=100)

    print(f"Training for {config.rl_steps} steps with batch size {config.rl_batch_size}")
    print()

    for step in range(config.rl_steps):
        # Collect batch of trajectories
        trajectories = []

        for _ in range(config.rl_batch_size):
            obs = env.reset()
            h = policy.init_hidden(1, device)

            traj_obs = []
            traj_actions = []
            traj_log_probs = []
            done = False

            while not done:
                traj_obs.append(obs)

                with torch.no_grad():
                    mu, logvar, h = policy(obs, h)
                    z = policy.sample(mu, logvar)
                    log_prob = policy.log_prob(z, mu, logvar)

                traj_actions.append(z)
                traj_log_probs.append(log_prob)

                obs, reward, done, _ = env.step(z)

            total_reward = reward  # Sparse reward at end
            episode_rewards.append(total_reward)

            trajectories.append({
                'obs': traj_obs,
                'actions': traj_actions,
                'log_probs': traj_log_probs,
                'reward': total_reward
            })

        # Compute advantages (GRPO-style normalization)
        rewards = torch.tensor([t['reward'] for t in trajectories])
        if rewards.std() > 1e-8:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards - rewards.mean()

        # Policy update
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []

        for i, traj in enumerate(trajectories):
            for obs, action, log_prob in zip(traj['obs'], traj['actions'], traj['log_probs']):
                all_obs.append(obs)
                all_actions.append(action)
                all_old_log_probs.append(log_prob)
                all_advantages.append(advantages[i])

        if len(all_obs) == 0:
            continue

        obs_batch = torch.cat(all_obs, dim=0)
        action_batch = torch.cat(all_actions, dim=0)
        old_log_prob_batch = torch.cat(all_old_log_probs, dim=0)
        advantage_batch = torch.tensor(all_advantages, device=device)

        # Forward pass
        mu, logvar, _ = policy(obs_batch)
        new_log_probs = policy.log_prob(action_batch, mu, logvar)

        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_prob_batch)
        clipped = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
        policy_loss = -torch.min(ratio * advantage_batch, clipped * advantage_batch).mean()

        # Entropy bonus
        entropy = 0.5 * (1 + math.log(2 * math.pi) + logvar).sum(dim=-1).mean()

        loss = policy_loss - config.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % 10 == 0:
            avg_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
                  f"Batch Reward: {rewards.mean():.3f} | Avg Reward (100): {avg_reward:.3f}")

        # Save checkpoint
        if step > 0 and step % 100 == 0:
            save_path = os.path.join(config.output_dir, f"policy_step{step}.pt")
            torch.save({
                'step': step,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_reward': sum(episode_rewards) / max(len(episode_rewards), 1)
            }, save_path)
            print(f"  Saved checkpoint to {save_path}")

    # Final save
    save_path = os.path.join(config.output_dir, "policy_final.pt")
    torch.save({
        'step': config.rl_steps,
        'policy_state_dict': policy.state_dict(),
        'avg_reward': sum(episode_rewards) / max(len(episode_rewards), 1)
    }, save_path)
    print(f"\nSaved final policy to {save_path}")
    print(f"Final average reward: {sum(episode_rewards) / max(len(episode_rewards), 1):.3f}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 17: Internal RL")
    parser.add_argument("--phase", type=str, default="all", choices=["2", "3", "all"],
                       help="Which phase to run: 2 (metacontroller), 3 (RL), or all")
    parser.add_argument("--metacontroller_path", type=str, default=None,
                       help="Path to pre-trained metacontroller (for phase 3)")
    parser.add_argument("--mc_epochs", type=int, default=10,
                       help="Epochs for metacontroller training")
    parser.add_argument("--rl_steps", type=int, default=500,
                       help="Steps for RL training")
    parser.add_argument("--rl_batch_size", type=int, default=8,
                       help="Batch size for RL training")
    args = parser.parse_args()

    config = Config(
        mc_epochs=args.mc_epochs,
        rl_steps=args.rl_steps,
        rl_batch_size=args.rl_batch_size
    )

    print("=" * 60)
    print("EXPERIMENT 17: INTERNAL RL WITH TEMPORAL ABSTRACTIONS")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Base model: {config.base_model}")
    print(f"Output dir: {config.output_dir}")
    print()

    if args.phase in ["2", "all"]:
        train_metacontroller(config)

    if args.phase in ["3", "all"]:
        train_internal_rl(config, args.metacontroller_path)

    print("\nExperiment 17 complete!")


if __name__ == "__main__":
    main()
