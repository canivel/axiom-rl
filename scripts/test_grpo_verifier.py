"""Test script for GRPOTrainer with Real Verifier."""

import sys
from pathlib import Path
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.trainer import GRPOTrainer, GRPOConfig
from axiom.verifier import RewardEngine
from axiom.problems.dataset import ProblemDataset

def main():
    print("Testing GRPOTrainer with REAL Verifier...")
    
    # 1. Load Dataset (Real Problems)
    dataset = ProblemDataset()
    two_sum = dataset.get_problem("two_sum")
    
    if not two_sum:
        print("Error: 'two_sum' problem not found!")
        return
        
    print(f"Loaded problem: {two_sum.title}")
    
    # 2. Initialize Reward Engine
    reward_engine = RewardEngine(dataset)
    
    # 3. Configure Trainer
    config = GRPOConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        num_generations=4, # Small G for testing
        batch_size=1,
        num_epochs=1,
        max_seq_length=512, # Enough for code
        logging_steps=1,
        learning_rate=1e-6,
        output_dir="models/test_grpo_verifier"
    )
    
    trainer = GRPOTrainer(
        config=config,
        reward_function=reward_engine.compute_rewards
    )
    
    # 4. Create a Training Dataset (Prompt Only)
    # We construct the prompt exactly as the model expects
    prompt = f"""You are an expert Python programmer.
## Problem: {two_sum.title}

{two_sum.description}

## Function Signature
```python
{two_sum.function_signature}
```

## Required Format
<think>
...
</think>
```python
...
```"""

    train_data = [{"prompt": prompt}]
    
    print("Starting training loop...")
    trainer.train(train_data)
    
    print("Test passed!")

if __name__ == "__main__":
    main()
