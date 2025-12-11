"""
Validate GRPO Learning Dynamics with a Real Problem.

Task: "Valid Parentheses"
This requires stack logic, which is harder than "Output 42" but solvable.
We use the Real Verifier to check correctness.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.trainer import GRPOTrainer, GRPOConfig
from axiom.verifier import RewardEngine
from axiom.problems.dataset import ProblemDataset

def main():
    print("Validating GRPO Learning Dynamics (Real Problem)...")
    print("Task: Valid Parentheses")
    
    # 1. Load Problem
    dataset = ProblemDataset()
    problem = dataset.get_problem("valid_parentheses")
    
    if not problem:
        print("Error: Problem 'valid_parentheses' not found!")
        return
        
    print(f"Problem: {problem.title}")
    
    # 2. Initialize Reward Engine
    reward_engine = RewardEngine(dataset)
    
    # 3. Configure Trainer
    # We use a higher learning rate because we want to see movement quickly
    config = GRPOConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        num_generations=8, 
        batch_size=1,
        num_epochs=1,
        max_seq_length=512,
        logging_steps=1,
        learning_rate=2e-5, # Increased from 1e-5
        output_dir="models/test_grpo_parentheses",
        beta=0.04 # Standard KL penalty
    )
    
    trainer = GRPOTrainer(
        config=config,
        reward_function=reward_engine.compute_rewards
    )
    
    # 4. Create Training Data
    # We repeat the SAME problem 30 times to force overfitting/learning
    prompt = f"""You are an expert Python programmer.
## Problem: {problem.title}

{problem.description}

## Function Signature
```python
{problem.function_signature}
```

## Required Format
<think>
...
</think>
```python
...
```"""

    train_data = [{"prompt": prompt} for _ in range(30)]
    
    print("\nStarting Training (30 Steps)...")
    print("Expectation: Avg Reward should trend upwards.")
    
    trainer.train(train_data)
    
    print("\nValidation Complete.")

if __name__ == "__main__":
    main()
