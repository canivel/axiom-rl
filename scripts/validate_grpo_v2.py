#!/usr/bin/env python3
"""
Validate GRPO Learning Dynamics with V2 Algorithmic Problems.

This script tests the GRPO training loop using V2 problems (multiple test cases).
Uses the 0.5B model for memory efficiency on RTX 3080.

Usage:
    uv run python scripts/validate_grpo_v2.py
"""

import sys
import re
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from axiom.trainer import GRPOTrainer, GRPOConfig
from axiom.procedural.generators_v2 import get_all_generators_v2


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

    # If no code block, try to find function definition
    if "def " in response:
        lines = response.split("\n")
        code_lines = []
        in_function = False

        for line in lines:
            if line.strip().startswith("def "):
                in_function = True
            if in_function:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

    return response


def verify_solution(code: str, problem) -> tuple[bool, float]:
    """
    Verify solution against all test cases.

    Returns:
        (success, partial_score) where partial_score is in [0, 1]
    """
    func_name = problem.function_name

    # Create execution namespace
    namespace = {}

    try:
        exec(code, namespace)
    except Exception as e:
        return False, 0.0

    if func_name not in namespace:
        # Try to find any defined function
        funcs = [k for k, v in namespace.items() if callable(v) and not k.startswith("_")]
        if funcs:
            func_name = funcs[0]
        else:
            return False, 0.0

    func = namespace[func_name]

    # Run all test cases
    passed = 0
    total = len(problem.test_cases)

    for tc in problem.test_cases:
        try:
            result = func(*tc.input_args)
            if result == tc.expected_output:
                passed += 1
        except Exception:
            pass

    success = passed == total
    partial = passed / total if total > 0 else 0.0

    return success, partial


def create_v2_reward_function(problem):
    """
    Create a reward function for V2 problems.

    Returns a function that computes rewards for (prompts, completions).
    """
    def reward_function(prompts: list, completions: list) -> torch.Tensor:
        """Compute rewards based on test case passing."""
        rewards = []

        for completion in completions:
            code = extract_code(completion)
            success, partial = verify_solution(code, problem)

            # Reward scheme:
            # - 1.0 for passing all test cases
            # - partial score (0-1) for passing some
            # - 0.0 for syntax errors or no tests passed
            if success:
                reward = 1.0
            else:
                reward = partial * 0.5  # Partial credit, but capped

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    return reward_function


def main():
    print("=" * 60)
    print("GRPO Validation with V2 Problems")
    print("=" * 60)

    # Use 0.5B model for RTX 3080
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    # Get a simple problem generator
    generators = get_all_generators_v2(seed=42)

    # Use FizzBuzz - it's simple but requires logic
    generator = generators["fizzbuzz"]
    problem = generator.generate(difficulty=3, num_test_cases=5)

    print(f"\nProblem: {problem.title}")
    print(f"Function: {problem.function_signature}")
    print(f"Test cases: {len(problem.test_cases)}")
    print("\nExample test cases:")
    for tc in problem.test_cases[:3]:
        print(f"  {problem.function_name}({tc.input_args[0]}) -> {repr(tc.expected_output)}")

    # Create reward function for this problem
    reward_fn = create_v2_reward_function(problem)

    # Configure GRPO (minimal settings for 0.5B model on RTX 3080)
    config = GRPOConfig(
        model_name=model_name,
        num_generations=2,  # Minimal for memory - RTX 3080 can't handle more
        batch_size=1,
        num_epochs=1,
        max_seq_length=512,
        logging_steps=1,
        learning_rate=2e-5,
        output_dir="models/test_grpo_v2",
        beta=0.04,
        torch_dtype="float16",  # Use fp16 for memory
    )

    print(f"\nConfig:")
    print(f"  Model: {config.model_name}")
    print(f"  Generations per prompt: {config.num_generations}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Beta (KL penalty): {config.beta}")

    # Create trainer
    trainer = GRPOTrainer(
        config=config,
        reward_function=reward_fn
    )

    # Create training data - repeat the same prompt
    prompt = f"""You are an expert Python programmer. Implement the following function.

## Problem: {problem.title}

{problem.description}

## Function Signature
```python
{problem.function_signature}
    # Your implementation here
```

## Examples
"""

    # Add examples
    for tc in problem.test_cases[:3]:
        prompt += f"  {problem.function_name}({repr(tc.input_args[0])}) -> {repr(tc.expected_output)}\n"

    prompt += "\nWrite ONLY the function implementation. No explanations."

    # Create training dataset (list of dicts with "prompt" key)
    train_data = [{"prompt": prompt} for _ in range(5)]  # 5 training steps (minimal)

    print("\n" + "=" * 60)
    print("Starting GRPO Training")
    print("=" * 60)
    print("Expectation: Average reward should trend upwards")
    print()

    try:
        trainer.train(train_data)
        print("\n" + "=" * 60)
        print("GRPO Validation Complete!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
