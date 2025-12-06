#!/usr/bin/env python3
"""
GRPO Training on Hard Problems.

This script uses GRPO (Group Relative Policy Optimization) to train
the model on LeetCode-hard style problems it currently fails on.

Target problems (from Experiment 09 baseline):
- Edit Distance: 0/5 (similar to LCS but fails)
- Knapsack: 0/5 (index error - buggy implementation)
- Coin Change: 0/5 (wrong algorithm)
- N-Queens: 2/5 (partial - backtracking issues)

Usage:
    uv run python scripts/run_grpo_hard.py
    uv run python scripts/run_grpo_hard.py --problems coin_change edit_distance --steps 20
    uv run python scripts/run_grpo_hard.py --all --steps 50
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from axiom.trainer import GRPOTrainer, GRPOConfig
from axiom.procedural.generators_hard import get_all_hard_generators, GENERATORS_HARD

# Target problems the model fails on (from Experiment 09)
WEAK_PROBLEMS = ["edit_distance", "knapsack", "coin_change", "n_queens"]

# All hard problems for comparison
ALL_PROBLEMS = list(GENERATORS_HARD.keys())


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
    except Exception:
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


def create_reward_function(problem):
    """Create a reward function for the current problem."""
    def reward_function(prompts: list, completions: list) -> torch.Tensor:
        """Compute rewards based on test case passing."""
        rewards = []

        for completion in completions:
            code = extract_code(completion)
            success, partial = verify_solution(code, problem)

            # Reward scheme:
            # - 1.0 for passing all test cases
            # - partial * 0.5 for passing some (capped at 0.5)
            # - 0.0 for syntax errors or no tests passed
            if success:
                reward = 1.0
            else:
                reward = partial * 0.5

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    return reward_function


def create_prompt(problem) -> str:
    """Create training prompt from problem."""
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
    # Add example test cases
    for tc in problem.test_cases[:3]:
        if len(tc.input_args) == 1:
            prompt += f"  {problem.function_name}({repr(tc.input_args[0])}) -> {repr(tc.expected_output)}\n"
        else:
            args_str = ", ".join(repr(a) for a in tc.input_args)
            prompt += f"  {problem.function_name}({args_str}) -> {repr(tc.expected_output)}\n"

    prompt += "\nWrite ONLY the function implementation. No explanations."

    return prompt


def evaluate_problem(model, tokenizer, generator, difficulty: int = 5, verbose: bool = False):
    """Evaluate model on a single problem type."""
    problem = generator.generate(difficulty=difficulty, num_test_cases=5)

    # Generate solution
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Write ONLY the function implementation - no explanations."},
        {"role": "user", "content": problem.to_prompt()},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    code = extract_code(response)

    success, partial = verify_solution(code, problem)

    if verbose:
        print(f"  Code: {code[:200]}...")

    return success, partial


def main():
    parser = argparse.ArgumentParser(description="GRPO Training on Hard Problems")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Base model to train",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=WEAK_PROBLEMS,
        help=f"Problem types to train on (default: {WEAK_PROBLEMS})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train on all hard problems (not just weak ones)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Training steps per problem type",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=5,
        help="Difficulty level 1-10 (default: 5)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=2,
        help="Generations per prompt (G)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        default="models/grpo-hard",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Determine problem types
    if args.all:
        problem_types = ALL_PROBLEMS
    else:
        problem_types = args.problems

    print("=" * 70)
    print("GRPO TRAINING ON HARD PROBLEMS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Problems: {problem_types}")
    print(f"Steps per problem: {args.steps}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Total steps: {len(problem_types) * args.steps}")
    print()

    # Initialize generators
    generators = get_all_hard_generators(seed=args.seed)

    # Filter to selected problems
    generators = {k: v for k, v in generators.items() if k in problem_types}

    if not generators:
        print(f"Error: No valid problem types from {problem_types}")
        print(f"Available: {ALL_PROBLEMS}")
        return 1

    # Configure GRPO
    grpo_config = GRPOConfig(
        model_name=args.model,
        num_generations=args.generations,
        batch_size=1,
        num_epochs=1,
        max_seq_length=768,  # Longer for hard problems
        logging_steps=1,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        beta=0.04,
        torch_dtype="float16",
    )

    print(f"GRPO Config:")
    print(f"  Generations: {grpo_config.num_generations}")
    print(f"  Learning Rate: {grpo_config.learning_rate}")
    print(f"  Max Seq Length: {grpo_config.max_seq_length}")
    print()

    # Training metrics
    metrics = {
        "start_time": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "problems": problem_types,
            "steps_per_problem": args.steps,
            "difficulty": args.difficulty,
        },
        "training": [],
        "evaluation": {
            "before": {},
            "after": {},
        },
    }

    # Initialize trainer
    trainer = None

    print("=" * 70)
    print("Training Loop")
    print("=" * 70)

    step = 0
    for problem_type in problem_types:
        generator = generators[problem_type]

        print(f"\n{'='*60}")
        print(f"Problem Type: {problem_type.upper()}")
        print(f"{'='*60}")

        for i in range(args.steps):
            # Generate new problem instance
            problem = generator.generate(difficulty=args.difficulty, num_test_cases=5)

            print(f"\n--- Step {step + 1} ({problem_type} {i+1}/{args.steps}) ---")
            print(f"Title: {problem.title}")

            # Create reward function for this problem
            reward_fn = create_reward_function(problem)

            # Create prompt
            prompt = create_prompt(problem)

            # Initialize trainer on first step
            if trainer is None:
                trainer = GRPOTrainer(
                    config=grpo_config,
                    reward_function=reward_fn
                )
                trainer.setup()
            else:
                trainer.reward_function = reward_fn

            # Create training data
            train_data = [{"prompt": prompt}]

            try:
                # Run GRPO step
                trainer.train(train_data)

                # Compute reward for metrics
                generations = trainer._rollout([prompt])
                rewards = reward_fn([prompt], generations[0])
                avg_reward = rewards.mean().item()
                max_reward = rewards.max().item()

                print(f"Avg Reward: {avg_reward:.4f} | Max: {max_reward:.4f}")

                # Record metrics
                metrics["training"].append({
                    "step": step,
                    "problem_type": problem_type,
                    "avg_reward": avg_reward,
                    "max_reward": max_reward,
                })

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                metrics["training"].append({
                    "step": step,
                    "problem_type": problem_type,
                    "error": str(e),
                })

            step += 1

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    if trainer is not None:
        for problem_type in problem_types:
            generator = generators[problem_type]
            success, partial = evaluate_problem(
                trainer.policy_model,
                trainer.tokenizer,
                generator,
                difficulty=args.difficulty,
                verbose=args.verbose
            )

            status = "PASS" if success else "FAIL"
            print(f"{problem_type:<25} {status:<10} {partial*100:.0f}%")

            metrics["evaluation"]["after"][problem_type] = {
                "success": success,
                "partial_score": partial,
            }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_steps = len(metrics["training"])
    successful_steps = sum(1 for m in metrics["training"] if m.get("max_reward", 0) == 1.0)
    avg_reward = sum(m.get("avg_reward", 0) for m in metrics["training"]) / max(total_steps, 1)

    print(f"Total Steps: {total_steps}")
    print(f"Perfect Solutions: {successful_steps}/{total_steps}")
    print(f"Average Reward: {avg_reward:.4f}")

    # Save metrics
    metrics["end_time"] = datetime.now().isoformat()
    metrics["summary"] = {
        "total_steps": total_steps,
        "successful_steps": successful_steps,
        "average_reward": avg_reward,
    }

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "hard_training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save model
    if trainer is not None:
        model_path = output_path / "final_model"
        trainer.policy_model.save_pretrained(str(model_path))
        trainer.tokenizer.save_pretrained(str(model_path))
        print(f"Model saved to: {model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
