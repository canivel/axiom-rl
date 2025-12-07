#!/usr/bin/env python3
"""
GRPO Curriculum Training for N-Queens.

This script implements curriculum learning for N-Queens:
1. Start with easiest difficulty (n=1,2,3,4)
2. Gradually increase as model improves
3. Track progress and adapt difficulty

The key insight: n=4 has only 2 solutions, so the model can find
correct solutions via exploration, providing learning signal.

Usage:
    uv run python scripts/run_grpo_nqueens_curriculum.py
    uv run python scripts/run_grpo_nqueens_curriculum.py --steps-per-level 10 --max-difficulty 5
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from axiom.trainer import GRPOTrainer, GRPOConfig
from axiom.procedural.generators_hard import NQueensCountGenerator


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    patterns = [r"```python\n(.*?)```", r"```\n(.*?)```"]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
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
    """Verify solution against test cases."""
    func_name = problem.function_name
    namespace = {}
    try:
        exec(code, namespace)
    except Exception:
        return False, 0.0
    if func_name not in namespace:
        funcs = [k for k, v in namespace.items() if callable(v) and not k.startswith("_")]
        if funcs:
            func_name = funcs[0]
        else:
            return False, 0.0
    func = namespace[func_name]
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
    """Create reward function for the current problem."""
    def reward_function(prompts: list, completions: list) -> torch.Tensor:
        rewards = []
        for completion in completions:
            code = extract_code(completion)
            success, partial = verify_solution(code, problem)
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
    for tc in problem.test_cases[:3]:
        if len(tc.input_args) == 1:
            prompt += f"  {problem.function_name}({repr(tc.input_args[0])}) -> {repr(tc.expected_output)}\n"
        else:
            args_str = ", ".join(repr(a) for a in tc.input_args)
            prompt += f"  {problem.function_name}({args_str}) -> {repr(tc.expected_output)}\n"
    prompt += "\nWrite ONLY the function implementation. No explanations."
    return prompt


def main():
    parser = argparse.ArgumentParser(description="GRPO Curriculum Training for N-Queens")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Base model")
    parser.add_argument("--sft-model", default=None, help="Start from SFT-pretrained model")
    parser.add_argument("--steps-per-level", type=int, default=10, help="Training steps per difficulty level")
    parser.add_argument("--start-difficulty", type=int, default=1, help="Starting difficulty (1-10)")
    parser.add_argument("--max-difficulty", type=int, default=5, help="Maximum difficulty to reach")
    parser.add_argument("--promotion-threshold", type=float, default=0.6, help="Avg reward to move up difficulty")
    parser.add_argument("--generations", type=int, default=4, help="Generations per prompt")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="models/grpo-nqueens-curriculum", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--low-memory", action="store_true", help="Use less memory (2 generations, shorter sequences)")
    args = parser.parse_args()

    # Override for low memory mode
    if args.low_memory:
        args.generations = 2
        print("[LOW MEMORY MODE] Using 2 generations instead of 4")

    print("=" * 70)
    print("GRPO CURRICULUM TRAINING FOR N-QUEENS")
    print("=" * 70)
    print(f"Model: {args.sft_model or args.model}")
    print(f"Difficulty: {args.start_difficulty} -> {args.max_difficulty}")
    print(f"Steps per level: {args.steps_per_level}")
    print(f"Promotion threshold: {args.promotion_threshold}")
    print()

    # Difficulty to board sizes mapping
    print("Curriculum Stages:")
    print("  Level 1-3: n in {1,2,3,4} - Trivial (n=4 has only 2 solutions)")
    print("  Level 4-6: n in {1,4,5,6} - Medium")
    print("  Level 7+:  n in {4,5,6,7,8} - Hard")
    print()

    generator = NQueensCountGenerator(seed=args.seed)

    # Configure GRPO
    model_name = args.sft_model if args.sft_model else args.model
    max_seq = 512 if args.low_memory else 768
    grpo_config = GRPOConfig(
        model_name=model_name,
        num_generations=args.generations,
        batch_size=1,
        num_epochs=1,
        max_seq_length=max_seq,
        logging_steps=1,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        beta=0.04,
        torch_dtype="float16",
    )

    # Metrics tracking
    metrics = {
        "start_time": datetime.now().isoformat(),
        "config": vars(args),
        "levels": [],
    }

    trainer = None
    current_difficulty = args.start_difficulty

    print("=" * 70)
    print("Training Loop")
    print("=" * 70)

    total_step = 0
    while current_difficulty <= args.max_difficulty:
        print(f"\n{'='*60}")
        print(f"DIFFICULTY LEVEL {current_difficulty}")
        print(f"{'='*60}")

        level_rewards = []
        level_metrics = {
            "difficulty": current_difficulty,
            "steps": [],
        }

        for step in range(args.steps_per_level):
            # Generate problem at current difficulty
            problem = generator.generate(difficulty=current_difficulty, num_test_cases=5)

            print(f"\n--- Step {total_step + 1} (Level {current_difficulty}, Step {step + 1}/{args.steps_per_level}) ---")
            print(f"Test cases: {[tc.input_args[0] for tc in problem.test_cases]}")  # Show n values

            reward_fn = create_reward_function(problem)
            prompt = create_prompt(problem)

            # Initialize trainer on first step
            if trainer is None:
                trainer = GRPOTrainer(config=grpo_config, reward_function=reward_fn)
                trainer.setup()
            else:
                trainer.reward_function = reward_fn

            train_data = [{"prompt": prompt}]

            try:
                trainer.train(train_data)

                # Compute rewards
                generations = trainer._rollout([prompt])
                rewards = reward_fn([prompt], generations[0])
                avg_reward = rewards.mean().item()
                max_reward = rewards.max().item()

                print(f"Avg Reward: {avg_reward:.4f} | Max: {max_reward:.4f}")
                level_rewards.append(avg_reward)

                level_metrics["steps"].append({
                    "step": total_step,
                    "avg_reward": avg_reward,
                    "max_reward": max_reward,
                    "test_ns": [tc.input_args[0] for tc in problem.test_cases],
                })

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                level_metrics["steps"].append({
                    "step": total_step,
                    "error": str(e),
                })

            total_step += 1

        # Check for promotion
        avg_level_reward = sum(level_rewards) / len(level_rewards) if level_rewards else 0
        level_metrics["average_reward"] = avg_level_reward

        print(f"\n--- Level {current_difficulty} Summary ---")
        print(f"Average Reward: {avg_level_reward:.4f}")

        if avg_level_reward >= args.promotion_threshold:
            print(f"✅ Promoting to level {current_difficulty + 1}")
            level_metrics["promoted"] = True
            current_difficulty += 1
        else:
            print(f"❌ Not ready for promotion (need {args.promotion_threshold})")
            level_metrics["promoted"] = False
            # Try one more round at same difficulty before giving up
            if len([l for l in metrics["levels"] if l["difficulty"] == current_difficulty]) >= 2:
                print(f"⚠️  Stuck at level {current_difficulty}, moving on anyway")
                level_metrics["forced_promotion"] = True
                current_difficulty += 1

        metrics["levels"].append(level_metrics)

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    if trainer is not None:
        for test_difficulty in [3, 5, 7]:
            problem = generator.generate(difficulty=test_difficulty, num_test_cases=5)

            messages = [
                {"role": "system", "content": "You are an expert Python programmer."},
                {"role": "user", "content": problem.to_prompt()},
            ]
            text = trainer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = trainer.tokenizer(text, return_tensors="pt").to(trainer.policy_model.device)

            with torch.no_grad():
                outputs = trainer.policy_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                    pad_token_id=trainer.tokenizer.pad_token_id,
                )
            response = trainer.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            code = extract_code(response)
            success, partial = verify_solution(code, problem)

            status = "PASS" if success else "FAIL"
            ns = [tc.input_args[0] for tc in problem.test_cases]
            print(f"Difficulty {test_difficulty} (n={ns}): {status} ({partial*100:.0f}%)")

            metrics[f"eval_difficulty_{test_difficulty}"] = {
                "success": success,
                "partial": partial,
                "test_ns": ns,
            }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total steps: {total_step}")
    print(f"Levels completed: {len(metrics['levels'])}")
    max_level = max(l['difficulty'] for l in metrics['levels']) if metrics['levels'] else 0
    print(f"Max difficulty reached: {max_level}")

    # Save
    metrics["end_time"] = datetime.now().isoformat()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "curriculum_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    if trainer is not None:
        model_path = output_path / "final_model"
        trainer.policy_model.save_pretrained(str(model_path))
        trainer.tokenizer.save_pretrained(str(model_path))
        print(f"Model saved to: {model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
