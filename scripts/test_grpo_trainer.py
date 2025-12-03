"""Test script for GRPOTrainer."""

import sys
from pathlib import Path
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.trainer import GRPOTrainer, GRPOConfig

def dummy_reward_function(prompts, generations):
    """Random rewards."""
    # Shape: (B * G,)
    return torch.randn(len(generations))

class DummyDataset:
    def __init__(self):
        self.data = [
            {"prompt": "def add(a, b):"},
            {"prompt": "def sub(a, b):"},
        ]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    print("Testing GRPOTrainer...")
    
    config = GRPOConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        num_generations=2,
        batch_size=1,
        num_epochs=1,
        max_seq_length=64,
        logging_steps=1,
        learning_rate=1e-5,
        output_dir="models/test_grpo"
    )
    
    trainer = GRPOTrainer(
        config=config,
        reward_function=dummy_reward_function
    )
    
    dataset = DummyDataset()
    
    print("Starting training loop...")
    trainer.train(dataset)
    
    print("Test passed!")

if __name__ == "__main__":
    main()
