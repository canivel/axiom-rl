#!/usr/bin/env python3
"""
Evaluate SFT model on V2 algorithmic problems.

This script tests the fine-tuned model on NEW problems from each generator
to measure generalization (not just memorization).

Usage:
    uv run python scripts/evaluate_sft_v2.py
    uv run python scripts/evaluate_sft_v2.py --model models/lora-sft-enhanced-distill-v1
"""

import argparse
import re
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from axiom.procedural.generators_v2 import GENERATORS_V2, get_all_generators_v2


def load_model(model_path: str, base_model: str = None):
    """Load fine-tuned LoRA model."""
    print(f"Loading model from: {model_path}")

    # Check if it's a LoRA adapter
    adapter_config = Path(model_path) / "adapter_config.json"

    if adapter_config.exists():
        # Load base model first
        import json
        with open(adapter_config) as f:
            config = json.load(f)
        base_model_name = base_model or config.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        # Direct model load
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    # Try to find code in markdown blocks
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


def generate_response(model, tokenizer, prompt: str, max_length: int = 1024) -> str:
    """Generate model response for a prompt."""
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Write ONLY the function implementation - no explanations, no test code."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.2,  # Low temp for more deterministic output
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def execute_and_verify(code: str, problem) -> tuple:
    """Execute code and verify against all test cases."""
    func_name = problem.function_name

    # Create execution namespace
    namespace = {}

    try:
        exec(code, namespace)
    except Exception as e:
        return False, 0, len(problem.test_cases), f"Syntax/Import error: {e}"

    if func_name not in namespace:
        # Try to find any defined function
        funcs = [k for k, v in namespace.items() if callable(v) and not k.startswith("_")]
        if funcs:
            func_name = funcs[0]
        else:
            return False, 0, len(problem.test_cases), f"Function {problem.function_name} not found"

    func = namespace[func_name]

    # Run all test cases
    passed = 0
    total = len(problem.test_cases)
    last_error = None

    for tc in problem.test_cases:
        try:
            result = func(*tc.input_args)
            if result == tc.expected_output:
                passed += 1
            else:
                last_error = f"Expected {tc.expected_output}, got {result}"
        except Exception as e:
            last_error = str(e)

    success = passed == total
    return success, passed, total, last_error if not success else None


def evaluate_on_v2_problems(model, tokenizer, problems_per_type: int = 3, difficulties: list = None):
    """Evaluate model on V2 algorithmic problems."""
    if difficulties is None:
        difficulties = [1, 3, 5, 7, 10]  # Range of difficulties

    generators = get_all_generators_v2(seed=999)  # Different seed from training

    results = {
        "by_type": {},
        "by_difficulty": {},
        "overall": {"correct": 0, "total": 0, "partial_credit": 0},
    }

    print("\n" + "=" * 60)
    print("V2 Problem Evaluation")
    print("=" * 60)

    for problem_type, generator in generators.items():
        print(f"\n--- {problem_type.upper()} ---")
        results["by_type"][problem_type] = {"correct": 0, "total": 0, "partial": 0}

        for difficulty in difficulties:
            problem = generator.generate(difficulty=difficulty, num_test_cases=5)

            print(f"  Difficulty {difficulty}: {problem.problem_id}...", end=" ", flush=True)

            # Generate response
            prompt = problem.to_prompt()
            response = generate_response(model, tokenizer, prompt)
            code = extract_code(response)

            # Verify
            success, passed, total, error = execute_and_verify(code, problem)

            results["by_type"][problem_type]["total"] += 1
            results["overall"]["total"] += 1

            if difficulty not in results["by_difficulty"]:
                results["by_difficulty"][difficulty] = {"correct": 0, "total": 0}
            results["by_difficulty"][difficulty]["total"] += 1

            if success:
                results["by_type"][problem_type]["correct"] += 1
                results["overall"]["correct"] += 1
                results["by_difficulty"][difficulty]["correct"] += 1
                print(f"PASS ({passed}/{total})")
            else:
                partial = passed / total if total > 0 else 0
                results["by_type"][problem_type]["partial"] += partial
                results["overall"]["partial_credit"] += partial

                print(f"FAIL ({passed}/{total})")
                if error:
                    print(f"       Error: {error[:80]}...")

    return results


def print_summary(results):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # By type
    print("\nBy Problem Type:")
    print("-" * 40)
    for ptype, stats in results["by_type"].items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        partial = stats["partial"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {ptype:20} {stats['correct']:2}/{stats['total']:2} ({acc:5.1f}%) partial: {partial:.1f}%")

    # By difficulty
    print("\nBy Difficulty:")
    print("-" * 40)
    for diff in sorted(results["by_difficulty"].keys()):
        stats = results["by_difficulty"][diff]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  Difficulty {diff:2}: {stats['correct']:2}/{stats['total']:2} ({acc:5.1f}%)")

    # Overall
    overall = results["overall"]
    acc = overall["correct"] / overall["total"] * 100 if overall["total"] > 0 else 0
    partial_avg = overall["partial_credit"] / overall["total"] * 100 if overall["total"] > 0 else 0

    print("\n" + "=" * 60)
    print(f"OVERALL: {overall['correct']}/{overall['total']} ({acc:.1f}%)")
    print(f"Average partial credit: {partial_avg:.1f}%")
    print("=" * 60)

    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model on V2 problems")
    parser.add_argument(
        "--model",
        default="models/lora-sft-enhanced-distill-v1",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model (auto-detected from adapter config)",
    )
    parser.add_argument(
        "--problems-per-type",
        type=int,
        default=5,
        help="Number of problems per type (default: 5 - one per difficulty)",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model, args.base_model)

    # Evaluate
    results = evaluate_on_v2_problems(
        model, tokenizer,
        problems_per_type=args.problems_per_type,
    )

    # Print summary
    accuracy = print_summary(results)

    # Return success if >50% accuracy (basic sanity check)
    return 0 if accuracy >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
