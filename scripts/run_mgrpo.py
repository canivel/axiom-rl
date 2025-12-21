#!/usr/bin/env python
"""
M-GRPO Training Script.

Runs M-GRPO (Momentum-Anchored GRPO) training with entropy control.

Usage:
    python scripts/run_mgrpo.py --experiment 15_mgrpo_entropy
    python scripts/run_mgrpo.py --experiment 15_mgrpo_entropy --config mgrpo_full
    python scripts/run_mgrpo.py --experiment 15_mgrpo_entropy --config vanilla_grpo
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from axiom.trainer import MGRPOConfig, MGRPOTrainer
from axiom.procedural import (
    RPNEvaluatorGenerator,
    ParenthesesValidatorGenerator,
    FibonacciGenerator,
    BinarySearchGenerator,
    FizzBuzzGenerator,
    MaxSubarrayGenerator,
    GENERATORS_V2,
)
from axiom.procedural.generators_hard import (
    EditDistanceGenerator,
    CoinChangeGenerator,
)
from axiom.verifier import TestHarness


# Problem generators registry - combine V2 and hard generators
GENERATORS = {
    "rpn": RPNEvaluatorGenerator,
    "parentheses": ParenthesesValidatorGenerator,
    "fibonacci": FibonacciGenerator,
    "binary_search": BinarySearchGenerator,
    "fizzbuzz": FizzBuzzGenerator,
    "max_subarray": MaxSubarrayGenerator,
    "edit_distance": EditDistanceGenerator,
    "coin_change": CoinChangeGenerator,
}


def load_config(experiment_dir: Path, ablation: str = None) -> dict:
    """Load experiment configuration with optional ablation overrides."""
    config_path = experiment_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Apply ablation overrides if specified
    if ablation and ablation in config.get("ablations", {}):
        ablation_config = config["ablations"][ablation]
        # Merge ablation settings into mgrpo config
        for key, value in ablation_config.items():
            if key != "description":
                config["mgrpo"][key] = value
        config["current_ablation"] = ablation
    else:
        config["current_ablation"] = "default"

    return config


def create_mgrpo_config(config: dict) -> MGRPOConfig:
    """Create MGRPOConfig from experiment config dict."""
    model_cfg = config["model"]
    mgrpo_cfg = config["mgrpo"]
    train_cfg = config["training"]

    return MGRPOConfig(
        # Model
        model_name=model_cfg["name"],
        torch_dtype=model_cfg["torch_dtype"],
        lora_r=model_cfg["lora_r"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        target_modules=model_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        # M-GRPO
        num_policy_samples=mgrpo_cfg["num_policy_samples"],
        num_momentum_samples=mgrpo_cfg["num_momentum_samples"],
        momentum=mgrpo_cfg["momentum"],
        beta=mgrpo_cfg["beta"],
        use_iqr_filter=mgrpo_cfg["use_iqr_filter"],
        iqr_k=mgrpo_cfg["iqr_k"],
        min_entropy_threshold=mgrpo_cfg.get("min_entropy_threshold", 0.1),
        use_clip_cov=mgrpo_cfg.get("use_clip_cov", False),
        use_kl_cov=mgrpo_cfg.get("use_kl_cov", False),
        # Training
        learning_rate=train_cfg["learning_rate"],
        max_seq_length=train_cfg["max_seq_length"],
        max_new_tokens=train_cfg["max_new_tokens"],
        temperature=train_cfg["temperature"],
        output_dir=str(Path(config["outputs"]["models_dir"]) / config["current_ablation"]),
    )


def generate_problems(config: dict, seed: int = 42):
    """Generate train/val/test problem sets."""
    import random
    rng = random.Random(seed)

    prob_cfg = config["problems"]
    train_problems, val_problems, test_problems = [], [], []

    for prob_type in prob_cfg["types"]:
        if prob_type not in GENERATORS:
            print(f"Warning: Unknown problem type '{prob_type}', skipping")
            continue

        gen = GENERATORS[prob_type](seed=rng.randint(0, 1000000))
        diff_min, diff_max = prob_cfg["difficulty_range"]

        # Generate train
        for _ in range(prob_cfg["train_per_type"]):
            diff = rng.randint(diff_min, diff_max)
            problem = gen.generate(
                difficulty=diff,
                num_test_cases=prob_cfg["test_cases_per_problem"]
            )
            train_problems.append(problem)

        # Generate val
        for _ in range(prob_cfg["val_per_type"]):
            diff = rng.randint(diff_min, diff_max)
            problem = gen.generate(
                difficulty=diff,
                num_test_cases=prob_cfg["test_cases_per_problem"]
            )
            val_problems.append(problem)

        # Generate test
        for _ in range(prob_cfg["test_per_type"]):
            diff = rng.randint(diff_min, diff_max)
            problem = gen.generate(
                difficulty=diff,
                num_test_cases=prob_cfg["test_cases_per_problem"]
            )
            test_problems.append(problem)

    rng.shuffle(train_problems)
    rng.shuffle(val_problems)
    rng.shuffle(test_problems)

    return train_problems, val_problems, test_problems


def extract_code(completion: str) -> str:
    """
    Extract Python code from model completion.

    Handles:
    1. Raw code (no markdown)
    2. ```python ... ``` blocks
    3. ``` ... ``` blocks
    4. Multiple code blocks (takes the longest one containing 'def')
    """
    import re

    # Try to find ```python ... ``` blocks
    python_blocks = re.findall(r'```python\s*(.*?)```', completion, re.DOTALL)
    if python_blocks:
        # Return the longest block that contains a function definition
        for block in sorted(python_blocks, key=len, reverse=True):
            if 'def ' in block:
                return block.strip()
        return python_blocks[0].strip()

    # Try to find ``` ... ``` blocks
    code_blocks = re.findall(r'```\s*(.*?)```', completion, re.DOTALL)
    if code_blocks:
        for block in sorted(code_blocks, key=len, reverse=True):
            if 'def ' in block:
                return block.strip()
        return code_blocks[0].strip()

    # If no code blocks, check if it starts with def or contains def
    # and try to extract just the function
    if 'def ' in completion:
        # Find the start of the function definition
        lines = completion.split('\n')
        code_lines = []
        in_function = False
        indent_level = None

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('def '):
                in_function = True
                indent_level = len(line) - len(stripped)
                code_lines = [line]
            elif in_function:
                if stripped and not stripped.startswith('#'):
                    current_indent = len(line) - len(stripped)
                    if current_indent <= indent_level and stripped:
                        # End of function
                        break
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines).strip()

    # Return as-is if no extraction possible
    return completion.strip()


def create_reward_function(problems: list, partial_reward: bool = True):
    """
    Create reward function that verifies solutions against test cases.

    Args:
        problems: List of problems to verify against
        partial_reward: If True, give proportional credit for passing some test cases.
                       If False, require all test cases to pass (binary 0/1).

    Returns reward function: (prompts, completions) -> rewards
    """
    harness = TestHarness()

    # Create prompt -> problem mapping
    prompt_to_problem = {p.to_prompt(): p for p in problems}

    def reward_fn(prompts: list, completions: list) -> torch.Tensor:
        rewards = []

        for prompt, completion in zip(prompts, completions):
            problem = prompt_to_problem.get(prompt)

            if problem is None:
                rewards.append(0.0)
                continue

            # Extract code from completion (handles markdown, etc.)
            code = extract_code(completion)

            # Verify solution
            try:
                result = harness.verify(code, problem)
                if partial_reward:
                    # Proportional reward: passed_count / total_count
                    reward = result.passed_count / max(result.total_count, 1)
                else:
                    # Binary reward: 1.0 only if all tests pass
                    reward = 1.0 if result.passed else 0.0
            except Exception:
                reward = 0.0

            rewards.append(reward)

        return torch.tensor(rewards)

    return reward_fn


def run_experiment(
    experiment_dir: Path,
    ablation: str = None,
    num_steps: int = None,
    eval_every: int = None,
    **kwargs,
):
    """Run the M-GRPO experiment."""
    print("=" * 60)
    print("M-GRPO EXPERIMENT")
    print("=" * 60)

    # Load config
    config = load_config(experiment_dir, ablation)
    print(f"\nExperiment: {config['experiment_name']}")
    print(f"Ablation: {config['current_ablation']}")

    # Create M-GRPO config
    mgrpo_config = create_mgrpo_config(config)
    print(f"\nM-GRPO Configuration:")
    print(f"  Momentum: {mgrpo_config.momentum}")
    print(f"  Policy samples: {mgrpo_config.num_policy_samples}")
    print(f"  Momentum samples: {mgrpo_config.num_momentum_samples}")
    print(f"  IQR filter: {mgrpo_config.use_iqr_filter}")
    print(f"  Clip-Cov: {mgrpo_config.use_clip_cov}")
    print(f"  KL-Cov: {mgrpo_config.use_kl_cov}")

    # Generate problems
    print("\nGenerating problem sets...")
    seed = config["reproducibility"]["seed"]
    train_problems, val_problems, test_problems = generate_problems(config, seed)
    print(f"  Train: {len(train_problems)} problems")
    print(f"  Val: {len(val_problems)} problems")
    print(f"  Test: {len(test_problems)} problems")

    # Create reward function with partial rewards enabled
    reward_fn = create_reward_function(train_problems, partial_reward=True)
    print(f"  Partial reward: enabled (proportional credit for passing tests)")

    # Create prompts for training
    train_prompts = [p.to_prompt() for p in train_problems]

    # Create trainer
    print("\nInitializing M-GRPO trainer...")
    trainer = MGRPOTrainer(
        config=mgrpo_config,
        reward_function=reward_fn,
    )
    trainer.setup()

    # Create evaluation function
    def evaluate(model, tokenizer):
        """Evaluate on validation set."""
        correct = 0
        harness = TestHarness()

        for problem in val_problems[:10]:  # Quick eval on subset
            prompt = problem.to_prompt()

            # Generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            completion = tokenizer.decode(
                outputs[0, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Verify
            try:
                result = harness.verify(completion, problem)
                if result.passed:
                    correct += 1
            except Exception:
                pass

        return f"{correct}/10 = {correct * 10}%"

    # Training parameters
    train_cfg = config["training"]
    steps = num_steps or (train_cfg["num_iterations"] * train_cfg["steps_per_iteration"])
    eval_interval = eval_every or train_cfg["eval_every"]

    # Checkpoint directory
    checkpoint_dir = experiment_dir / "checkpoints" / config["current_ablation"]

    # Check for resume
    start_step = 0
    resume_path = kwargs.get("resume")
    if resume_path:
        start_step = trainer.load_checkpoint(Path(resume_path))

    # Run training
    print(f"\nStarting training for {steps} steps...")
    summary = trainer.train(
        prompts=train_prompts,
        num_steps=steps,
        eval_every=eval_interval,
        eval_fn=evaluate,
        checkpoint_every=kwargs.get("checkpoint_every", 5),
        checkpoint_dir=str(checkpoint_dir),
        start_step=start_step,
        verbose=kwargs.get("verbose", True),
    )

    # Save model
    output_dir = experiment_dir / config["outputs"]["models_dir"] / config["current_ablation"]
    print(f"\nSaving model to {output_dir}...")
    trainer.save(output_dir)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Final metrics: {summary.get('final_metrics', {})}")
    print(f"Entropy summary: {summary.get('entropy_summary', {})}")

    # Run benchmarks if configured
    if config["benchmarks"]["run_after_training"]:
        print("\nRunning benchmarks...")
        try:
            from axiom.benchmarks import run_all_benchmarks

            benchmark_dir = experiment_dir / config["benchmarks"]["output_dir"]
            run_all_benchmarks(
                model_path=str(output_dir / "policy"),
                benchmark_names=config["benchmarks"]["benchmarks"],
                output_dir=benchmark_dir,
                max_samples=config["benchmarks"].get("max_samples_per_benchmark"),
            )
        except Exception as e:
            print(f"Benchmark error: {e}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run M-GRPO experiment")

    parser.add_argument(
        "--experiment",
        type=str,
        default="15_mgrpo_entropy",
        help="Experiment directory name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ablation config to use (e.g., vanilla_grpo, mgrpo_full)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of training steps",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Override evaluation frequency",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N steps (default: 5)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Find experiment directory
    experiment_dir = project_root / "experiments" / args.experiment
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # Run experiment
    run_experiment(
        experiment_dir=experiment_dir,
        ablation=args.config,
        num_steps=args.steps,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
