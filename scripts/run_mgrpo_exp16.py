#!/usr/bin/env python
"""
Experiment 16: M-GRPO with Class Wrapper Fix

Key changes from Experiment 15:
1. extract_code() now handles 'class Solution' wrappers
2. Improved prompts explicitly request standalone functions
3. Added greedy evaluation during training

Run:
    uv run python scripts/run_mgrpo_exp16.py --steps 20
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
import os
import time
import copy
import textwrap
from dataclasses import dataclass, field
from typing import List, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW


# ============================================================================
# PROBLEM DEFINITIONS
# ============================================================================

@dataclass
class TestCase:
    """A single test case with input and expected output."""
    input_args: Any
    expected_output: Any


@dataclass
class Problem:
    """A coding problem with description, signature, and test cases."""
    problem_id: str
    title: str
    description: str
    function_name: str
    function_signature: str
    test_cases: List[TestCase]
    difficulty: int = 5
    problem_type: str = "general"
    examples: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert problem to a prompt - IMPROVED VERSION with anti-class instructions."""
        examples_str = "\n".join(f"  {ex}" for ex in self.examples[:3])

        prompt = f"""Write a Python function to solve the following problem.

## Problem: {self.title}

{self.description}

## Function Signature
```python
{self.function_signature}
```

## Requirements
- Implement the function exactly as specified
- Handle edge cases appropriately
- Return the correct type

## Examples
{examples_str}

## IMPORTANT
Write ONLY a standalone Python function.
Do NOT wrap it in a class.
Do NOT use 'class Solution'.
The function must be directly callable.
"""
        return prompt.strip()


class ProblemGenerator(ABC):
    """Base class for procedural problem generators."""

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    @abstractmethod
    def generate(self, difficulty: int = 5, num_test_cases: int = 5) -> Problem:
        pass


# ============================================================================
# PROBLEM GENERATORS
# ============================================================================

class RPNEvaluatorGenerator(ProblemGenerator):
    """Generates RPN evaluation problems."""

    def generate(self, difficulty: int = 5, num_test_cases: int = 5) -> Problem:
        test_cases = []
        for _ in range(num_test_cases):
            tokens, result = self._generate_rpn_expression(difficulty)
            test_cases.append(TestCase(input_args=[tokens], expected_output=result))

        return Problem(
            problem_id=f"rpn_{self.rng.randint(1000, 9999)}",
            title="RPN Expression Evaluator",
            description="""Evaluate a Reverse Polish Notation (RPN) expression.

RPN is a mathematical notation where operators follow their operands.
For example: ["2", "3", "+"] = 5, ["4", "2", "*", "3", "+"] = 11

Supported operators: +, -, *
All operands are integers (can be negative).""",
            function_name="evaluate_rpn",
            function_signature="def evaluate_rpn(tokens: List[str]) -> int:",
            test_cases=test_cases,
            difficulty=difficulty,
            problem_type="rpn",
            examples=[
                f'evaluate_rpn({test_cases[0].input_args[0]}) -> {test_cases[0].expected_output}',
                f'evaluate_rpn({test_cases[1].input_args[0]}) -> {test_cases[1].expected_output}' if len(test_cases) > 1 else "",
            ]
        )

    def _generate_rpn_expression(self, difficulty: int):
        num_ops = min(difficulty, 5)
        operators = ['+', '-', '*']
        tokens = []
        stack = []

        for _ in range(num_ops + 1):
            if self.rng.random() < 0.2:
                val = self.rng.randint(-10, -1)
            else:
                val = self.rng.randint(1, 10)
            tokens.append(str(val))
            stack.append(val)

            if len(stack) >= 2 and self.rng.random() < 0.7:
                op = self.rng.choice(operators)
                tokens.append(op)
                b, a = stack.pop(), stack.pop()
                if op == '+': stack.append(a + b)
                elif op == '-': stack.append(a - b)
                elif op == '*': stack.append(a * b)

        while len(stack) > 1:
            op = self.rng.choice(operators)
            tokens.append(op)
            b, a = stack.pop(), stack.pop()
            if op == '+': stack.append(a + b)
            elif op == '-': stack.append(a - b)
            elif op == '*': stack.append(a * b)

        return tokens, stack[0]


class ParenthesesValidatorGenerator(ProblemGenerator):
    """Generates balanced parentheses validation problems."""

    def generate(self, difficulty: int = 5, num_test_cases: int = 5) -> Problem:
        test_cases = []
        for i in range(num_test_cases):
            if i < num_test_cases // 2:
                s = self._generate_valid(difficulty)
                test_cases.append(TestCase(input_args=[s], expected_output=True))
            else:
                s = self._generate_invalid(difficulty)
                test_cases.append(TestCase(input_args=[s], expected_output=False))

        self.rng.shuffle(test_cases)

        return Problem(
            problem_id=f"paren_{self.rng.randint(1000, 9999)}",
            title="Valid Parentheses",
            description="""Determine if a string of parentheses is valid.

A string is valid if:
- Open brackets are closed by the same type of brackets
- Open brackets are closed in the correct order
- Every close bracket has a corresponding open bracket

Valid brackets: (), [], {}""",
            function_name="is_valid_parentheses",
            function_signature="def is_valid_parentheses(s: str) -> bool:",
            test_cases=test_cases,
            difficulty=difficulty,
            problem_type="parentheses",
            examples=[
                f'is_valid_parentheses("{test_cases[0].input_args[0]}") -> {test_cases[0].expected_output}',
            ]
        )

    def _generate_valid(self, difficulty: int) -> str:
        pairs = [('(', ')'), ('[', ']'), ('{', '}')]
        length = min(difficulty * 2, 12)
        result = []

        for _ in range(length // 2):
            pair = self.rng.choice(pairs)
            pos = self.rng.randint(0, len(result))
            result.insert(pos, pair[0])
            result.insert(pos + 1, pair[1])

        return ''.join(result)

    def _generate_invalid(self, difficulty: int) -> str:
        valid = self._generate_valid(difficulty)
        if not valid:
            return "("

        chars = list(valid)
        mutation = self.rng.choice(['swap', 'remove', 'add'])

        if mutation == 'swap' and len(chars) >= 2:
            i, j = self.rng.sample(range(len(chars)), 2)
            chars[i], chars[j] = chars[j], chars[i]
        elif mutation == 'remove' and chars:
            chars.pop(self.rng.randint(0, len(chars) - 1))
        else:
            chars.insert(self.rng.randint(0, len(chars)), self.rng.choice('([{'))

        return ''.join(chars)


class FibonacciGenerator(ProblemGenerator):
    """Generates Fibonacci number computation problems."""

    def generate(self, difficulty: int = 5, num_test_cases: int = 5) -> Problem:
        max_n = min(difficulty * 5, 30)
        test_cases = []

        test_cases.append(TestCase(input_args=[0], expected_output=0))
        test_cases.append(TestCase(input_args=[1], expected_output=1))

        for _ in range(num_test_cases - 2):
            n = self.rng.randint(2, max_n)
            test_cases.append(TestCase(input_args=[n], expected_output=self._fib(n)))

        self.rng.shuffle(test_cases)

        return Problem(
            problem_id=f"fib_{self.rng.randint(1000, 9999)}",
            title="Fibonacci Number",
            description="""Compute the nth Fibonacci number.

The Fibonacci sequence is: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1""",
            function_name="fibonacci",
            function_signature="def fibonacci(n: int) -> int:",
            test_cases=test_cases,
            difficulty=difficulty,
            problem_type="fibonacci",
            examples=[
                f'fibonacci({test_cases[0].input_args[0]}) -> {test_cases[0].expected_output}',
            ]
        )

    def _fib(self, n: int) -> int:
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class BinarySearchGenerator(ProblemGenerator):
    """Generates binary search problems."""

    def generate(self, difficulty: int = 5, num_test_cases: int = 5) -> Problem:
        array_size = min(difficulty * 3, 20)
        test_cases = []

        for i in range(num_test_cases):
            arr = sorted(self.rng.sample(range(1, 100), array_size))

            if i < num_test_cases // 2:
                target = self.rng.choice(arr)
                expected = arr.index(target)
            else:
                target = self.rng.randint(1, 100)
                while target in arr:
                    target = self.rng.randint(1, 100)
                expected = -1

            test_cases.append(TestCase(input_args=[arr, target], expected_output=expected))

        return Problem(
            problem_id=f"bsearch_{self.rng.randint(1000, 9999)}",
            title="Binary Search",
            description="""Find the index of a target value in a sorted array.

Return the index if found, -1 if not found.
The array is sorted in ascending order.
Use binary search for O(log n) time complexity.""",
            function_name="binary_search",
            function_signature="def binary_search(arr: List[int], target: int) -> int:",
            test_cases=test_cases,
            difficulty=difficulty,
            problem_type="binary_search",
            examples=[
                f'binary_search({test_cases[0].input_args[0]}, {test_cases[0].input_args[1]}) -> {test_cases[0].expected_output}',
            ]
        )


class EditDistanceGenerator(ProblemGenerator):
    """Generates edit distance problems."""

    def generate(self, difficulty: int = 5, num_test_cases: int = 5) -> Problem:
        max_len = min(difficulty + 2, 8)
        test_cases = []

        for _ in range(num_test_cases):
            s1 = self._random_word(self.rng.randint(2, max_len))
            s2 = self._random_word(self.rng.randint(2, max_len))
            dist = self._edit_distance(s1, s2)
            test_cases.append(TestCase(input_args=[s1, s2], expected_output=dist))

        return Problem(
            problem_id=f"edit_{self.rng.randint(1000, 9999)}",
            title="Edit Distance",
            description="""Compute the edit distance (Levenshtein distance) between two strings.

The edit distance is the minimum number of single-character edits
(insertions, deletions, or substitutions) needed to transform one string into another.""",
            function_name="edit_distance",
            function_signature="def edit_distance(s1: str, s2: str) -> int:",
            test_cases=test_cases,
            difficulty=difficulty,
            problem_type="edit_distance",
            examples=[
                f'edit_distance("{test_cases[0].input_args[0]}", "{test_cases[0].input_args[1]}") -> {test_cases[0].expected_output}',
            ]
        )

    def _random_word(self, length: int) -> str:
        return ''.join(self.rng.choice('abcdefghij') for _ in range(length))

    def _edit_distance(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]


class CoinChangeGenerator(ProblemGenerator):
    """Generates coin change problems."""

    def generate(self, difficulty: int = 5, num_test_cases: int = 5) -> Problem:
        coins = sorted(self.rng.sample([1, 2, 5, 10, 20, 25, 50], min(difficulty, 5)))
        if 1 not in coins:
            coins = [1] + coins

        test_cases = []
        max_amount = difficulty * 10

        for _ in range(num_test_cases):
            amount = self.rng.randint(1, max_amount)
            result = self._min_coins(coins, amount)
            test_cases.append(TestCase(input_args=[coins, amount], expected_output=result))

        return Problem(
            problem_id=f"coin_{self.rng.randint(1000, 9999)}",
            title="Coin Change",
            description=f"""Find the minimum number of coins needed to make up a given amount.

You have an infinite supply of each coin denomination.
Return -1 if the amount cannot be made up by any combination of the coins.

Coin denominations for this problem: {coins}""",
            function_name="coin_change",
            function_signature="def coin_change(coins: List[int], amount: int) -> int:",
            test_cases=test_cases,
            difficulty=difficulty,
            problem_type="coin_change",
            examples=[
                f'coin_change({coins}, {test_cases[0].input_args[1]}) -> {test_cases[0].expected_output}',
            ]
        )

    def _min_coins(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)

        return dp[amount] if dp[amount] != float('inf') else -1


GENERATORS = {
    "rpn": RPNEvaluatorGenerator,
    "parentheses": ParenthesesValidatorGenerator,
    "fibonacci": FibonacciGenerator,
    "binary_search": BinarySearchGenerator,
    "edit_distance": EditDistanceGenerator,
    "coin_change": CoinChangeGenerator,
}


# ============================================================================
# CODE EXTRACTION (FIXED VERSION)
# ============================================================================

def extract_method_from_class(code: str, func_name: str) -> str:
    """
    Extract a method from a class and convert to standalone function.
    Handles 'class Solution' wrappers.
    """
    pattern = rf'def\s+{func_name}\s*\(\s*self\s*,?\s*([^)]*)?(\))\s*(?:->\s*[^:]+)?\s*:'
    match = re.search(pattern, code)

    if not match:
        return code

    method_start = match.start()
    lines = code[method_start:].split('\n')
    method_lines = [lines[0]]

    def_indent = len(lines[0]) - len(lines[0].lstrip())

    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:
            method_lines.append(line)
            continue

        current_indent = len(line) - len(stripped)
        if current_indent <= def_indent and stripped and not stripped.startswith('#'):
            break

        method_lines.append(line)

    method_code = '\n'.join(method_lines)

    # Remove 'self' parameter
    method_code = re.sub(
        rf'(def\s+{func_name}\s*\()self\s*,?\s*',
        r'\1',
        method_code
    )

    method_code = textwrap.dedent(method_code)
    return method_code.strip()


def extract_code(completion: str, func_name: str = None) -> str:
    """
    Extract Python code from model completion.
    FIXED: Handles class Solution wrappers.
    """
    code = None

    # Try ```python blocks
    python_blocks = re.findall(r'```python\s*(.*?)```', completion, re.DOTALL)
    if python_blocks:
        for block in sorted(python_blocks, key=len, reverse=True):
            if 'def ' in block:
                code = block.strip()
                break
        if not code:
            code = python_blocks[0].strip()

    if not code:
        # Try ``` blocks
        code_blocks = re.findall(r'```\s*(.*?)```', completion, re.DOTALL)
        if code_blocks:
            for block in sorted(code_blocks, key=len, reverse=True):
                if 'def ' in block:
                    code = block.strip()
                    break
            if not code:
                code = code_blocks[0].strip()

    if not code:
        if 'def ' in completion:
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
                            break
                    code_lines.append(line)

            if code_lines:
                code = '\n'.join(code_lines).strip()

    if not code:
        code = completion.strip()

    # FIXED: Handle class Solution wrappers
    if 'class Solution' in code and func_name:
        extracted = extract_method_from_class(code, func_name)
        if 'def ' in extracted:
            code = extracted

    return code


# ============================================================================
# VERIFICATION
# ============================================================================

@dataclass
class VerificationResult:
    passed: bool
    passed_count: int
    total_count: int
    error: str = None


def verify_solution(code: str, problem: Problem, timeout: float = 5.0) -> VerificationResult:
    """Verify a solution against test cases."""
    func_name = problem.function_name

    test_cases_data = [
        {"input": tc.input_args, "expected": tc.expected_output}
        for tc in problem.test_cases
    ]
    test_cases_json = json.dumps(test_cases_data)

    test_script = f'''# -*- coding: utf-8 -*-
import json
from typing import List, Optional, Tuple, Dict, Any, Set

# === SOLUTION CODE ===
{code}
# === END SOLUTION ===

def run_tests():
    test_cases = json.loads('{test_cases_json}')
    results = []

    for i, tc in enumerate(test_cases):
        inp = tc["input"]
        expected = tc["expected"]

        try:
            if isinstance(inp, list):
                actual = {func_name}(*inp)
            else:
                actual = {func_name}(inp)

            passed = actual == expected
            results.append({{"index": i, "passed": passed}})
        except Exception as e:
            results.append({{"index": i, "passed": False, "error": str(e)}})

    print(json.dumps({{"results": results, "success": True}}))

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(json.dumps({{"results": [], "success": False, "error": str(e)}}))
'''

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(test_script)
            temp_path = f.name

        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        os.unlink(temp_path)

        if result.returncode != 0 and not result.stdout.strip():
            return VerificationResult(
                passed=False,
                passed_count=0,
                total_count=len(problem.test_cases),
                error=result.stderr or "Execution error"
            )

        data = json.loads(result.stdout)

        if not data.get("success", False):
            return VerificationResult(
                passed=False,
                passed_count=0,
                total_count=len(problem.test_cases),
                error=data.get("error", "Unknown error")
            )

        passed_count = sum(1 for r in data["results"] if r["passed"])
        total_count = len(data["results"])

        return VerificationResult(
            passed=(passed_count == total_count),
            passed_count=passed_count,
            total_count=total_count
        )

    except subprocess.TimeoutExpired:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return VerificationResult(
            passed=False,
            passed_count=0,
            total_count=len(problem.test_cases),
            error="Timeout"
        )
    except Exception as e:
        return VerificationResult(
            passed=False,
            passed_count=0,
            total_count=len(problem.test_cases),
            error=str(e)
        )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def generate_samples(model, tokenizer, prompts: List[str], n_samples: int = 4,
                     max_new_tokens: int = 512, temperature: float = 0.7) -> List[List[str]]:
    """Generate multiple samples for each prompt."""
    all_samples = []

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            samples = []
            for _ in range(n_samples):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                completion = tokenizer.decode(
                    outputs[0, inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                samples.append(completion)

            all_samples.append(samples)

    return all_samples


def compute_rewards(prompts: List[str], samples: List[List[str]],
                    problems: List[Problem]) -> torch.Tensor:
    """Compute rewards for all samples using partial credit."""
    rewards = []

    for prompt, prompt_samples, problem in zip(prompts, samples, problems):
        sample_rewards = []
        for sample in prompt_samples:
            code = extract_code(sample, problem.function_name)
            result = verify_solution(code, problem)
            reward = result.passed_count / max(result.total_count, 1)
            sample_rewards.append(reward)

        rewards.append(sample_rewards)

    return torch.tensor(rewards)


def compute_log_probs(model, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    """Compute log probabilities for a completion given a prompt."""
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    # NOTE: No torch.no_grad() here - we need gradients for training!
    outputs = model(**inputs)
    logits = outputs.logits

    shift_logits = logits[:, prompt_tokens-1:-1, :]
    shift_labels = inputs["input_ids"][:, prompt_tokens:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum()


def update_momentum_model(policy_model, momentum_model, momentum: float = 0.99):
    """Update momentum model with EMA of policy weights."""
    with torch.no_grad():
        for (name_p, param_p), (name_m, param_m) in zip(
            policy_model.named_parameters(),
            momentum_model.named_parameters()
        ):
            if param_p.requires_grad:
                param_m.data = momentum * param_m.data + (1 - momentum) * param_p.data


def compute_entropy(model, tokenizer, prompt: str, completion: str) -> float:
    """Compute entropy of the completion."""
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, prompt_tokens-1:-1, :]

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()

    return entropy


def evaluate_greedy(model, tokenizer, problems: List[Problem], max_problems: int = 10) -> dict:
    """Evaluate model with sampling (for validation)."""
    model.eval()
    results_by_type = {}

    for problem in problems[:max_problems]:
        prompt = problem.to_prompt()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion = tokenizer.decode(
            outputs[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        code = extract_code(completion, problem.function_name)
        result = verify_solution(code, problem)

        prob_type = problem.problem_type
        if prob_type not in results_by_type:
            results_by_type[prob_type] = {"passed": 0, "total": 0}

        results_by_type[prob_type]["total"] += 1
        if result.passed:
            results_by_type[prob_type]["passed"] += 1

    total_passed = sum(r["passed"] for r in results_by_type.values())
    total_problems = sum(r["total"] for r in results_by_type.values())

    return {
        "by_type": results_by_type,
        "overall": total_passed / max(total_problems, 1),
        "passed": total_passed,
        "total": total_problems
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def generate_problem_sets(seed: int = 42):
    """Generate train/val/test problem sets."""
    rng = random.Random(seed)

    problem_types = ["rpn", "parentheses", "fibonacci", "binary_search", "edit_distance", "coin_change"]
    train_per_type = 10
    val_per_type = 5

    train_problems, val_problems = [], []

    for prob_type in problem_types:
        gen = GENERATORS[prob_type](seed=rng.randint(0, 1000000))

        for _ in range(train_per_type):
            diff = rng.randint(4, 7)
            train_problems.append(gen.generate(difficulty=diff, num_test_cases=5))

        for _ in range(val_per_type):
            diff = rng.randint(4, 7)
            val_problems.append(gen.generate(difficulty=diff, num_test_cases=5))

    rng.shuffle(train_problems)
    rng.shuffle(val_problems)

    return train_problems, val_problems


def main():
    parser = argparse.ArgumentParser(description="Run M-GRPO Experiment 16")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--eval-every", type=int, default=2, help="Evaluate every N steps")
    parser.add_argument("--greedy-eval-every", type=int, default=5, help="Greedy eval every N steps")
    args = parser.parse_args()

    print("=" * 60)
    print("EXPERIMENT 16: M-GRPO WITH CLASS WRAPPER FIX")
    print("=" * 60)

    # Load model
    MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    print(f"\nLoading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    policy_model = get_peft_model(model, lora_config)
    policy_model.print_trainable_parameters()

    momentum_model = copy.deepcopy(policy_model)
    for param in momentum_model.parameters():
        param.requires_grad = False

    print(f"Device: {next(policy_model.parameters()).device}")

    # Generate problems
    print("\nGenerating problems...")
    train_problems, val_problems = generate_problem_sets(seed=42)
    print(f"  Train: {len(train_problems)}, Val: {len(val_problems)}")

    # Training config
    NUM_STEPS = args.steps
    BATCH_SIZE = args.batch_size
    NUM_POLICY_SAMPLES = 4
    NUM_MOMENTUM_SAMPLES = 4
    LEARNING_RATE = 1e-5
    MOMENTUM = 0.99

    optimizer = AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )

    metrics_history = []

    print("\n" + "=" * 60)
    print(f"Training for {NUM_STEPS} steps...")
    print("=" * 60)

    # Initial evaluation
    print("\nInitial evaluation...")
    initial_eval = evaluate_greedy(policy_model, tokenizer, val_problems, max_problems=10)
    print(f"Initial accuracy: {initial_eval['passed']}/{initial_eval['total']} = {initial_eval['overall']*100:.1f}%")

    for step in range(NUM_STEPS):
        step_start = time.time()

        print(f"\n{'='*60}")
        print(f"Step {step}/{NUM_STEPS}")
        print(f"{'='*60}")

        # Sample batch
        batch_problems = random.sample(train_problems, BATCH_SIZE)
        prompts = [p.to_prompt() for p in batch_problems]

        # Generate from policy
        print("Generating from policy model...")
        policy_samples = generate_samples(policy_model, tokenizer, prompts, n_samples=NUM_POLICY_SAMPLES)

        # Generate from momentum
        print("Generating from momentum model...")
        momentum_samples = generate_samples(momentum_model, tokenizer, prompts, n_samples=NUM_MOMENTUM_SAMPLES)

        # Compute rewards
        print("Computing rewards...")
        policy_rewards = compute_rewards(prompts, policy_samples, batch_problems)
        momentum_rewards = compute_rewards(prompts, momentum_samples, batch_problems)

        all_rewards = torch.cat([policy_rewards, momentum_rewards], dim=1)

        # Log per-problem results
        for i, problem in enumerate(batch_problems):
            best_reward = all_rewards[i].max().item()
            print(f"  [{i+1}/{BATCH_SIZE}] {problem.title}... reward={best_reward:.2f}")

        # Compute advantages
        advantages = torch.zeros_like(policy_rewards)
        for i in range(BATCH_SIZE):
            prompt_rewards = all_rewards[i]
            mean_r = prompt_rewards.mean()
            std_r = prompt_rewards.std() + 1e-8
            advantages[i] = (policy_rewards[i] - mean_r) / std_r

        # Training step
        policy_model.train()
        total_loss = 0
        num_updates = 0

        for i, (prompt, samples, advs, problem) in enumerate(
            zip(prompts, policy_samples, advantages, batch_problems)
        ):
            for sample, adv in zip(samples, advs):
                if adv.item() <= 0:
                    continue

                log_prob = compute_log_probs(policy_model, tokenizer, prompt, sample)
                loss = -log_prob * adv

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        # Update momentum
        update_momentum_model(policy_model, momentum_model, MOMENTUM)

        # Compute entropy
        entropy = compute_entropy(policy_model, tokenizer, prompts[0], policy_samples[0][0])

        # Metrics
        avg_loss = total_loss / max(num_updates, 1)
        avg_reward = policy_rewards.max(dim=1).values.mean().item()
        success_rate = (policy_rewards.max(dim=1).values > 0).float().mean().item() * 100

        step_time = time.time() - step_start
        eta = (NUM_STEPS - step - 1) * step_time / 60

        # Quick validation
        val_correct = 0
        val_subset = random.sample(val_problems, min(5, len(val_problems)))
        for prob in val_subset:
            val_samples = generate_samples(policy_model, tokenizer, [prob.to_prompt()], n_samples=1)
            code = extract_code(val_samples[0][0], prob.function_name)
            result = verify_solution(code, prob)
            if result.passed:
                val_correct += 1
        val_acc = val_correct / len(val_subset) * 100

        print(f"\nStep {step} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Reward: {avg_reward:.3f}")
        print(f"  Entropy: {entropy:.3f}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Val Accuracy: {val_acc:.1f}%")
        print(f"  Updates: {num_updates}")
        print(f"  Time: {step_time:.1f}s, ETA: {eta:.1f}m")

        if entropy < 0.1:
            print("\n⚠️ WARNING: Entropy below 0.1 - potential collapse!")

        # Greedy evaluation
        if step > 0 and step % args.greedy_eval_every == 0:
            print("\n--- Greedy Evaluation ---")
            greedy_eval = evaluate_greedy(policy_model, tokenizer, val_problems, max_problems=10)
            print(f"Greedy accuracy: {greedy_eval['passed']}/{greedy_eval['total']} = {greedy_eval['overall']*100:.1f}%")
            for ptype, results in greedy_eval['by_type'].items():
                print(f"  {ptype}: {results['passed']}/{results['total']}")

        metrics_history.append({
            "step": step,
            "loss": avg_loss,
            "reward": avg_reward,
            "entropy": entropy,
            "success_rate": success_rate,
            "val_accuracy": val_acc,
            "num_updates": num_updates,
            "time": step_time
        })

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    final_eval = evaluate_greedy(policy_model, tokenizer, val_problems, max_problems=len(val_problems))

    print("\nResults by Problem Type:")
    print("-" * 40)
    for ptype, results in final_eval['by_type'].items():
        acc = results['passed'] / max(results['total'], 1) * 100
        print(f"  {ptype:20s}: {results['passed']}/{results['total']} = {acc:.1f}%")
    print("-" * 40)
    print(f"  {'OVERALL':20s}: {final_eval['passed']}/{final_eval['total']} = {final_eval['overall']*100:.1f}%")

    # Save model
    output_dir = Path("experiments/16_mgrpo_class_fix/models/default")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(output_dir / "policy")
    tokenizer.save_pretrained(output_dir / "policy")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\nModel saved to {output_dir}")

    total_time = sum(m['time'] for m in metrics_history)
    print(f"\nTotal training time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
