#!/usr/bin/env python3
"""
GRPO Training with Curriculum Learning.

This script combines GRPO (RL fine-tuning) with curriculum learning
(progressive difficulty) to efficiently train on algorithmic problems.

Usage:
    uv run python scripts/run_grpo_curriculum.py
    uv run python scripts/run_grpo_curriculum.py --strategy adaptive --steps 30
    uv run python scripts/run_grpo_curriculum.py --problems parentheses two_sum --steps 50
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
from axiom.trainer.curriculum import (
    CurriculumScheduler,
    CurriculumConfig,
    CurriculumStrategy,
    DifficultyLevel,
)
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


def main():
    parser = argparse.ArgumentParser(description="GRPO Training with Curriculum Learning")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Base model to train",
    )
    parser.add_argument(
        "--strategy",
        choices=["linear", "adaptive", "self_paced"],
        default="adaptive",
        help="Curriculum strategy",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Total training steps",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=None,
        help="Problem types to train on (default: all)",
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
        default="models/grpo-curriculum",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training with Curriculum Learning")
    print("=" * 60)

    # Initialize generators
    generators = get_all_generators_v2(seed=args.seed)

    # Filter by problem types if specified
    if args.problems:
        generators = {k: v for k, v in generators.items() if k in args.problems}
        if not generators:
            print(f"Error: No valid problem types from {args.problems}")
            return 1
        print(f"\nProblem types: {list(generators.keys())}")
    else:
        print(f"\nProblem types: ALL ({len(generators)} types)")

    # Create curriculum config
    curriculum_config = CurriculumConfig(
        strategy=CurriculumStrategy(args.strategy),
        problem_types=args.problems or [],
        max_total_steps=args.steps,
        levels=[
            DifficultyLevel("easy", (1, 3), target_success_rate=0.8, min_samples=3),
            DifficultyLevel("medium", (4, 6), target_success_rate=0.7, min_samples=5),
            DifficultyLevel("hard", (7, 10), target_success_rate=0.6, min_samples=7),
        ],
    )

    # Create curriculum scheduler
    curriculum = CurriculumScheduler(curriculum_config, generators, seed=args.seed)

    print(f"\nCurriculum Strategy: {args.strategy}")
    print(f"Starting Level: {curriculum.current_level.name}")
    print(f"Total Steps: {args.steps}")

    # Configure GRPO
    grpo_config = GRPOConfig(
        model_name=args.model,
        num_generations=args.generations,
        batch_size=1,
        num_epochs=1,
        max_seq_length=512,
        logging_steps=1,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        beta=0.04,
        torch_dtype="float16",
    )

    print(f"\nGRPO Config:")
    print(f"  Model: {grpo_config.model_name}")
    print(f"  Generations: {grpo_config.num_generations}")
    print(f"  Learning Rate: {grpo_config.learning_rate}")

    # Initialize trainer (will load models on first train)
    trainer = None

    # Training metrics
    metrics = {
        "start_time": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "strategy": args.strategy,
            "steps": args.steps,
            "problems": args.problems,
        },
        "steps": [],
    }

    print("\n" + "=" * 60)
    print("Starting Curriculum Training")
    print("=" * 60)

    step = 0
    while not curriculum.is_complete and step < args.steps:
        # Get problem from curriculum
        problem, problem_type, difficulty = curriculum.get_problem()

        print(f"\n--- Step {step + 1}/{args.steps} ---")
        print(f"Level: {curriculum.current_level.name} | Type: {problem_type} | Difficulty: {difficulty}")

        # Create reward function for this problem
        reward_fn = create_reward_function(problem)

        # Create prompt
        prompt = create_prompt(problem)

        # Initialize trainer on first step (deferred loading)
        if trainer is None:
            trainer = GRPOTrainer(
                config=grpo_config,
                reward_function=reward_fn
            )
            trainer.setup()
        else:
            # Update reward function for new problem
            trainer.reward_function = reward_fn

        # Create training data (single problem)
        train_data = [{"prompt": prompt}]

        # Run single GRPO step
        try:
            # We need to manually run one step
            # The trainer.train() runs through all data, so we use batch_size=1
            trainer.train(train_data)

            # Get average reward from last step
            # For now, we'll compute it manually
            generations = trainer._rollout([prompt])
            rewards = reward_fn([prompt], generations[0])
            avg_reward = rewards.mean().item()

            print(f"Avg Reward: {avg_reward:.4f}")

            # Record result in curriculum
            curriculum.record_result(avg_reward)

            # Record metrics
            step_metrics = {
                "step": step,
                "level": curriculum.current_level.name,
                "problem_type": problem_type,
                "difficulty": difficulty,
                "avg_reward": avg_reward,
            }
            metrics["steps"].append(step_metrics)

        except Exception as e:
            print(f"Error in step {step}: {e}")
            import traceback
            traceback.print_exc()
            # Record failure
            curriculum.record_result(0.0)
            metrics["steps"].append({
                "step": step,
                "error": str(e),
            })

        step += 1

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    stats = curriculum.get_stats()
    dist = curriculum.get_difficulty_distribution()

    print(f"\nFinal Level: {stats['current_level']}")
    print(f"Total Steps: {stats['total_steps']}")
    print(f"Avg Success (last 10): {stats['avg_success']:.2%}")
    print(f"\nDifficulty Distribution:")
    for level, count in dist.items():
        print(f"  {level}: {count} steps")

    # Save metrics
    metrics["end_time"] = datetime.now().isoformat()
    metrics["summary"] = {
        "final_level": stats["current_level"],
        "total_steps": stats["total_steps"],
        "avg_success": stats["avg_success"],
        "difficulty_distribution": dist,
    }

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "curriculum_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save model if training was successful
    if trainer is not None and stats["total_steps"] > 0:
        model_path = output_path / "final_model"
        trainer.policy_model.save_pretrained(str(model_path))
        trainer.tokenizer.save_pretrained(str(model_path))
        print(f"Model saved to: {model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
