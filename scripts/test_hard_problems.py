#!/usr/bin/env python3
"""
Test Model on Hard LeetCode-Style Problems.

This script evaluates how well the model performs on challenging algorithmic problems:
- Dynamic Programming (LCS, Edit Distance, Knapsack, LIS, Coin Change)
- Backtracking (N-Queens)
- Interval Problems (Merge Intervals, Trapping Rain Water)
- Divide and Conquer (Median of Two Sorted Arrays)

Usage:
    uv run python scripts/test_hard_problems.py
    uv run python scripts/test_hard_problems.py --model models/lora-sft-enhanced-distill-v1
    uv run python scripts/test_hard_problems.py --problems lcs knapsack coin_change
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from axiom.procedural.generators_hard import get_all_hard_generators, GENERATORS_HARD


def load_model(model_path: str, base_model: str = None):
    """Load model (base or fine-tuned)."""
    print(f"Loading model: {model_path}")

    adapter_config = Path(model_path) / "adapter_config.json"

    if adapter_config.exists():
        # LoRA adapter
        import json
        with open(adapter_config) as f:
            config = json.load(f)
        base_model_name = base_model or config.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

        print(f"Loading base: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        # Direct model
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
    """Extract Python code from response."""
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


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 1024) -> str:
    """Generate model response."""
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Write ONLY the function implementation - no explanations."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def execute_and_verify(code: str, problem, verbose: bool = False) -> tuple:
    """Execute code and verify against test cases."""
    func_name = problem.function_name
    namespace = {}

    try:
        exec(code, namespace)
    except Exception as e:
        return False, 0, len(problem.test_cases), f"Exec error: {e}"

    if func_name not in namespace:
        funcs = [k for k, v in namespace.items() if callable(v) and not k.startswith("_")]
        if funcs:
            func_name = funcs[0]
        else:
            return False, 0, len(problem.test_cases), f"Function {problem.function_name} not found"

    func = namespace[func_name]
    passed = 0
    total = len(problem.test_cases)
    last_error = None

    for i, tc in enumerate(problem.test_cases):
        try:
            result = func(*tc.input_args)
            if result == tc.expected_output:
                passed += 1
                if verbose:
                    print(f"    Test {i+1}: PASS")
            else:
                last_error = f"Expected {tc.expected_output}, got {result}"
                if verbose:
                    print(f"    Test {i+1}: FAIL - {last_error[:50]}")
        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"    Test {i+1}: ERROR - {e}")

    success = passed == total
    return success, passed, total, last_error if not success else None


def main():
    parser = argparse.ArgumentParser(description="Test model on hard problems")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=None,
        help=f"Problem types to test (default: all). Options: {list(GENERATORS_HARD.keys())}",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=5,
        help="Difficulty level 1-10 (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed test results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("HARD PROBLEM EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Difficulty: {args.difficulty}")
    print()

    # Load model
    model, tokenizer = load_model(args.model)

    # Get generators
    generators = get_all_hard_generators(seed=42)
    if args.problems:
        generators = {k: v for k, v in generators.items() if k in args.problems}

    print(f"Testing {len(generators)} problem types: {list(generators.keys())}")
    print()

    # Results
    results = {}
    total_passed = 0
    total_problems = 0

    for problem_type, generator in generators.items():
        print(f"\n{'='*60}")
        print(f"Problem Type: {problem_type.upper()}")
        print("=" * 60)

        problem = generator.generate(difficulty=args.difficulty, num_test_cases=5)

        print(f"Title: {problem.title}")
        print(f"Signature: {problem.function_signature}")
        print(f"Test cases: {len(problem.test_cases)}")

        # Show example test cases
        print("\nExample inputs/outputs:")
        for tc in problem.test_cases[:2]:
            args_str = ", ".join(repr(a)[:30] for a in tc.input_args)
            print(f"  {problem.function_name}({args_str}) -> {repr(tc.expected_output)[:30]}")

        # Generate solution
        prompt = problem.to_prompt()
        print("\nGenerating solution...")
        response = generate_response(model, tokenizer, prompt)
        code = extract_code(response)

        if args.verbose:
            print(f"\nGenerated code:\n{code[:500]}...")

        # Verify
        success, passed, total, error = execute_and_verify(code, problem, verbose=args.verbose)

        if success:
            print(f"\n[PASS] {passed}/{total} test cases")
            total_passed += 1
        else:
            print(f"\n[FAIL] {passed}/{total} test cases")
            if error:
                print(f"  Error: {error[:100]}")

        results[problem_type] = {
            "success": success,
            "passed": passed,
            "total": total,
            "error": error,
        }
        total_problems += 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Problem Type':<25} {'Result':<10} {'Score'}")
    print("-" * 50)
    for ptype, res in results.items():
        status = "PASS" if res["success"] else "FAIL"
        score = f"{res['passed']}/{res['total']}"
        print(f"{ptype:<25} {status:<10} {score}")

    accuracy = total_passed / total_problems * 100 if total_problems > 0 else 0
    print("-" * 50)
    print(f"\nOVERALL: {total_passed}/{total_problems} ({accuracy:.1f}%)")

    # Assessment
    print("\n" + "=" * 70)
    if accuracy >= 50:
        print("Assessment: Model shows reasonable algorithmic capability")
    elif accuracy >= 25:
        print("Assessment: Model struggles with hard problems - needs more training")
    else:
        print("Assessment: Model cannot solve hard problems - expected for 0.5B")
    print("=" * 70)

    return 0 if accuracy >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
