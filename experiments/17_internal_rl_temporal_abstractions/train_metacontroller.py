"""
Metacontroller Training Script (Phase 2)

This script trains the metacontroller on expert code solutions using
self-supervised learning (ELBO maximization).

The metacontroller learns to:
1. Discover temporally-abstract actions from unlabeled code
2. Generate controller matrices that steer the base model
3. Learn when to switch between abstract actions

Training Process:
================
1. Load frozen Qwen 0.5B base model
2. For each batch of expert solutions:
   a. Extract residual activations at mid-layer
   b. Run metacontroller to get controlled residuals
   c. Pass controlled residuals through remaining layers
   d. Compute action prediction loss + KL regularization
   e. Update metacontroller parameters

Key Design Choices:
==================
- Base model is FROZEN (critical for abstract action discovery)
- Metacontroller has acausal access to full sequence (training only)
- KL weight controls rate-distortion trade-off

Usage:
    python train_metacontroller.py --config config.yaml
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metacontroller import (
    Metacontroller,
    compute_elbo_loss,
    AbstractActionPolicy
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    controller_layer: int = 12  # Mid-depth for Qwen 0.5B (24 layers)

    # Metacontroller architecture
    latent_dim: int = 16
    gru_dim: int = 64
    seq_embed_dim: int = 64
    encoder_hidden: int = 64
    decoder_hidden: int = 64
    switch_hidden: int = 64
    controller_rank: int = 16

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.03
    num_epochs: int = 50
    kl_weight: float = 0.1
    temperature: float = 1.0

    # Data
    max_seq_len: int = 512
    data_path: str = "data/expert_solutions.json"

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "experiments/17_internal_rl_temporal_abstractions/models"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ExpertSolutionDataset(Dataset):
    """
    Dataset of expert code solutions.

    Each sample contains:
    - problem: The problem description
    - solution: The expert solution code
    - problem_type: Category (fibonacci, binary_search, etc.)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} expert solutions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format as prompt + completion
        prompt = self._format_prompt(item['problem'])
        completion = item['solution']
        full_text = prompt + completion

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Create labels (mask prompt tokens with -100)
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors='pt'
        )
        prompt_len = prompt_encoding['input_ids'].shape[1]

        labels = encoding['input_ids'].clone()
        labels[0, :prompt_len] = -100  # Mask prompt

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'problem_type': item.get('problem_type', 'unknown')
        }

    def _format_prompt(self, problem: str) -> str:
        """Format problem as instruction prompt."""
        return f"""Write a Python function to solve this problem.
Do NOT use a class wrapper. Write a standalone function.

Problem: {problem}

Solution:
```python
"""


def create_expert_data_from_problems(output_path: str):
    """
    Create expert solution dataset from our problem definitions.
    Uses the solutions from our existing problem set.
    """
    from axiom.problems import PROBLEMS

    data = []
    for problem_type, problems in PROBLEMS.items():
        for problem in problems:
            # Get canonical solution
            solution = problem.get('canonical_solution', '')
            if not solution:
                continue

            data.append({
                'problem': problem['description'],
                'solution': solution,
                'problem_type': problem_type,
                'function_name': problem.get('entry_point', 'solution')
            })

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Created {len(data)} expert solutions at {output_path}")
    return data


class ControlledModel(nn.Module):
    """
    Wrapper that applies metacontroller to base model.

    This inserts the metacontroller at a specific layer and
    modifies the residual stream during forward pass.
    """

    def __init__(
        self,
        base_model: nn.Module,
        metacontroller: Metacontroller,
        controller_layer: int,
        freeze_base: bool = True
    ):
        super().__init__()

        self.base_model = base_model
        self.metacontroller = metacontroller
        self.controller_layer = controller_layer

        # Freeze base model
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()

        # Get embedding dimension
        self.embed_dim = base_model.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with metacontroller intervention.

        1. Run first half of model to get residuals
        2. Apply metacontroller to residuals
        3. Run second half of model
        4. Compute loss
        """
        batch_size, seq_len = input_ids.shape

        # Storage for intermediate activations
        residuals_before = []
        residuals_after = []

        # Hook to capture residuals before control layer
        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                residuals_before.append(output[0])
            else:
                residuals_before.append(output)

        # Hook to inject controlled residuals
        def inject_hook(module, input, output):
            if len(residuals_after) > 0:
                controlled = residuals_after[0]
                if isinstance(output, tuple):
                    return (controlled,) + output[1:]
                return controlled
            return output

        # Register hooks
        target_layer = self.base_model.model.layers[self.controller_layer]
        capture_handle = target_layer.register_forward_hook(capture_hook)

        try:
            # First pass: get residuals at control layer
            with torch.no_grad():
                _ = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

            # Get residuals
            if residuals_before:
                residuals = residuals_before[0]
            else:
                raise RuntimeError("Failed to capture residuals")

            # Apply metacontroller
            mc_outputs = self.metacontroller.forward_training(
                residuals,
                attention_mask
            )
            controlled_residuals = mc_outputs['controlled_sequence']

        finally:
            capture_handle.remove()

        # Store controlled residuals for injection
        residuals_after.append(controlled_residuals)

        # Register injection hook
        inject_handle = target_layer.register_forward_hook(inject_hook)

        try:
            # Second pass: run with controlled residuals
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False
            )
        finally:
            inject_handle.remove()

        # Add metacontroller outputs
        return {
            'loss': outputs.loss if hasattr(outputs, 'loss') else None,
            'logits': outputs.logits,
            'z_sequence': mc_outputs['z_sequence'],
            'mu_sequence': mc_outputs['mu_sequence'],
            'logvar_sequence': mc_outputs['logvar_sequence'],
            'beta_sequence': mc_outputs['beta_sequence']
        }


def train_metacontroller(config: TrainingConfig):
    """
    Main training loop for metacontroller.
    """
    print("=" * 60)
    print("METACONTROLLER TRAINING (Phase 2)")
    print("=" * 60)
    print(f"Config: {config}")
    print()

    # Setup
    device = torch.device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (frozen)
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        device_map="auto" if config.device == "cuda" else None
    )

    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    embed_dim = base_model.config.hidden_size
    print(f"Base model hidden size: {embed_dim}")
    print(f"Base model layers: {base_model.config.num_hidden_layers}")

    # Create metacontroller
    print("Creating metacontroller...")
    metacontroller = Metacontroller(
        embed_dim=embed_dim,
        latent_dim=config.latent_dim,
        gru_dim=config.gru_dim,
        seq_embed_dim=config.seq_embed_dim,
        encoder_hidden=config.encoder_hidden,
        decoder_hidden=config.decoder_hidden,
        switch_hidden=config.switch_hidden,
        controller_rank=config.controller_rank
    ).to(device)

    # Count parameters
    mc_params = sum(p.numel() for p in metacontroller.parameters() if p.requires_grad)
    print(f"Metacontroller parameters: {mc_params:,}")

    # Create controlled model
    controlled_model = ControlledModel(
        base_model=base_model,
        metacontroller=metacontroller,
        controller_layer=config.controller_layer
    )

    # Create dataset
    print("Loading dataset...")
    if not os.path.exists(config.data_path):
        print(f"Creating expert data at {config.data_path}...")
        create_expert_data_from_problems(config.data_path)

    dataset = ExpertSolutionDataset(
        config.data_path,
        tokenizer,
        config.max_seq_len
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Optimizer (only metacontroller parameters)
    optimizer = AdamW(
        metacontroller.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Scheduler
    total_steps = len(dataloader) * config.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    print("\nStarting training...")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {config.batch_size}")
    print()

    global_step = 0
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        epoch_nll = 0
        epoch_kl = 0
        epoch_beta_mean = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            try:
                outputs = controlled_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue

            # Compute ELBO loss
            loss_dict = compute_elbo_loss(
                action_logits=outputs['logits'],
                target_actions=labels,
                mu_sequence=outputs['mu_sequence'],
                logvar_sequence=outputs['logvar_sequence'],
                kl_weight=config.kl_weight,
                attention_mask=attention_mask
            )

            loss = loss_dict['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(metacontroller.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_nll += loss_dict['nll'].item()
            epoch_kl += loss_dict['kl'].item()
            epoch_beta_mean += outputs['beta_sequence'].mean().item()
            num_batches += 1
            global_step += 1

            # Log
            if global_step % config.log_interval == 0:
                avg_beta = outputs['beta_sequence'].mean().item()
                print(f"Step {global_step} | Loss: {loss.item():.4f} | "
                      f"NLL: {loss_dict['nll'].item():.4f} | "
                      f"KL: {loss_dict['kl'].item():.4f} | "
                      f"Beta: {avg_beta:.3f}")

            # Save checkpoint
            if global_step % config.save_interval == 0:
                checkpoint_path = os.path.join(
                    config.output_dir,
                    f"metacontroller_step{global_step}.pt"
                )
                torch.save({
                    'step': global_step,
                    'metacontroller_state_dict': metacontroller.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.__dict__
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_nll = epoch_nll / max(num_batches, 1)
        avg_kl = epoch_kl / max(num_batches, 1)
        avg_beta = epoch_beta_mean / max(num_batches, 1)

        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg NLL: {avg_nll:.4f}")
        print(f"  Avg KL: {avg_kl:.4f}")
        print(f"  Avg Beta (switching rate): {avg_beta:.3f}")
        print()

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(config.output_dir, "metacontroller_best.pt")
            torch.save({
                'epoch': epoch,
                'metacontroller_state_dict': metacontroller.state_dict(),
                'loss': best_loss,
                'config': config.__dict__
            }, best_path)
            print(f"Saved best model (loss={best_loss:.4f})")

    # Final save
    final_path = os.path.join(config.output_dir, "metacontroller_final.pt")
    torch.save({
        'metacontroller_state_dict': metacontroller.state_dict(),
        'config': config.__dict__
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    return metacontroller


def analyze_switching_patterns(
    metacontroller: Metacontroller,
    dataset: ExpertSolutionDataset,
    device: torch.device,
    num_samples: int = 10
):
    """
    Analyze the switching patterns learned by the metacontroller.

    We expect switching (β ≈ 1) to occur at logical code boundaries
    like function signatures, if-statements, loops, etc.
    """
    print("\n" + "=" * 60)
    print("SWITCHING PATTERN ANALYSIS")
    print("=" * 60)

    metacontroller.eval()

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

        # This would require the base model to get residuals
        # For now, just print info
        print(f"\nSample {i+1}: {sample['problem_type']}")
        print(f"Sequence length: {attention_mask.sum().item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="experiments/17_internal_rl_temporal_abstractions/models")
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        kl_weight=args.kl_weight,
        output_dir=args.output_dir
    )

    train_metacontroller(config)
