"""Self-Improvement Experiment V2 - Correct Design.

Key differences from V1:
1. Uses AlgorithmicProblems with MULTIPLE test cases
2. Model must pass ALL test cases to collect a solution
3. Functions take INPUT - can't just hardcode
4. This forces actual algorithm learning

The Expert Iteration cycle:
1. Generate solutions for training problems
2. Verify which solutions pass ALL test cases
3. Train on correct solutions (LoRA SFT)
4. Evaluate on held-out problems
5. Repeat
"""

import json
import re
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

from axiom.procedural.base_v2 import AlgorithmicProblem, TestCase
from axiom.procedural.generators_v2 import get_all_generators_v2


@dataclass
class SelfImproveConfigV2:
    """Configuration for V2 self-improvement experiment."""

    # Experiment
    experiment_name: str = "self_improve_v2"
    output_dir: Path = field(default_factory=lambda: Path("experiments"))

    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32

    # Problems
    problem_types: List[str] = field(default_factory=lambda: ["rpn", "parentheses"])
    train_problems_per_type: int = 10
    val_problems_per_type: int = 5
    test_problems_per_type: int = 5
    test_cases_per_problem: int = 5
    min_difficulty: int = 3
    max_difficulty: int = 7
    seed: int = 42

    # Training
    num_iterations: int = 5
    learning_rate: float = 5e-5  # Lower LR for stability
    epochs_per_iteration: int = 2
    batch_size: int = 1
    gradient_accumulation: int = 4

    # Generation
    temperature: float = 0.7
    max_new_tokens: int = 512


class SelfImproveExperimentV2:
    """V2 Self-improvement with algorithmic problems."""

    def __init__(self, config: SelfImproveConfigV2):
        self.config = config
        self.experiment_dir = config.output_dir / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "solutions").mkdir(exist_ok=True)

        # Save config
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(self._config_to_dict(), f, indent=2)

        # Generate problem sets
        self.generators = get_all_generators_v2(seed=config.seed)
        self._generate_problem_sets()

        # Model
        self.model = None
        self.tokenizer = None
        self.peft_model = None

        # Metrics
        self.metrics_history = []

    def _config_to_dict(self) -> dict:
        d = {}
        for k, v in self.config.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d

    def _generate_problem_sets(self):
        """Generate fixed train/val/test problem sets."""
        print("Generating problem sets...")

        self.train_problems = []
        self.val_problems = []
        self.test_problems = []

        for ptype in self.config.problem_types:
            if ptype not in self.generators:
                print(f"  Warning: Unknown type {ptype}, skipping")
                continue

            gen = self.generators[ptype]

            # Train problems
            for _ in range(self.config.train_problems_per_type):
                diff = gen.rng.randint(self.config.min_difficulty, self.config.max_difficulty)
                problem = gen.generate(diff, self.config.test_cases_per_problem)
                self.train_problems.append(problem)

            # Val problems (different seed offset)
            gen.rng.seed(self.config.seed + 1000)
            for _ in range(self.config.val_problems_per_type):
                diff = gen.rng.randint(self.config.min_difficulty, self.config.max_difficulty)
                problem = gen.generate(diff, self.config.test_cases_per_problem)
                self.val_problems.append(problem)

            # Test problems (different seed offset)
            gen.rng.seed(self.config.seed + 2000)
            for _ in range(self.config.test_problems_per_type):
                diff = gen.rng.randint(self.config.min_difficulty, self.config.max_difficulty)
                problem = gen.generate(diff, self.config.test_cases_per_problem)
                self.test_problems.append(problem)

        print(f"  Train: {len(self.train_problems)} problems")
        print(f"  Val: {len(self.val_problems)} problems")
        print(f"  Test: {len(self.test_problems)} problems")
        print(f"  Test cases per problem: {self.config.test_cases_per_problem}")

        # Save problems
        self._save_problems(self.train_problems, "train.jsonl")
        self._save_problems(self.val_problems, "val.jsonl")
        self._save_problems(self.test_problems, "test.jsonl")

    def _save_problems(self, problems: List[AlgorithmicProblem], filename: str):
        filepath = self.experiment_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for p in problems:
                data = {
                    "problem_type": p.problem_type,
                    "problem_id": p.problem_id,
                    "title": p.title,
                    "function_signature": p.function_signature,
                    "difficulty": p.difficulty,
                    "num_test_cases": len(p.test_cases),
                    "test_cases": [tc.to_dict() for tc in p.test_cases],
                }
                f.write(json.dumps(data) + "\n")

    def _load_model(self):
        """Load the base model."""
        if self.model is not None:
            return

        print(f"Loading model: {self.config.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"  Loaded on: {next(self.model.parameters()).device}")

    def _generate_solution(self, problem: AlgorithmicProblem) -> str:
        """Generate a solution for a problem."""
        model = self.peft_model if self.peft_model else self.model
        model.eval()

        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Write ONLY the function implementation - no explanations."},
            {"role": "user", "content": problem.to_prompt()},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return self._extract_code(response)

    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        # Try ```python blocks
        matches = re.findall(r"```python\s*(.*?)```", response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Try generic ``` blocks
        matches = re.findall(r"```\s*(.*?)```", response, re.DOTALL)
        if matches:
            return matches[0].strip()

        return response.strip()

    def _verify_solution(self, code: str, problem: AlgorithmicProblem) -> tuple:
        """
        Verify solution against ALL test cases.

        Returns: (passed_all, passed_count, total_count)
        """
        func_name = problem.function_name
        total = len(problem.test_cases)
        passed = 0

        # Execute code to define function
        local_ns = {}
        try:
            exec(code, {"__builtins__": __builtins__}, local_ns)
        except Exception as e:
            return False, 0, total

        if func_name not in local_ns:
            return False, 0, total

        func = local_ns[func_name]

        # Test each case
        for tc in problem.test_cases:
            try:
                result = func(*tc.input_args)
                if result == tc.expected_output:
                    passed += 1
            except Exception:
                pass

        return passed == total, passed, total

    def evaluate(self, problems: List[AlgorithmicProblem], name: str) -> dict:
        """Evaluate on a set of problems."""
        print(f"  Evaluating {name} ({len(problems)} problems)...")

        correct = 0
        total_passed = 0
        total_tests = 0

        for i, problem in enumerate(problems):
            print(f"\r    Progress: {i+1}/{len(problems)}", end="", flush=True)

            code = self._generate_solution(problem)
            passed_all, passed, total = self._verify_solution(code, problem)

            if passed_all:
                correct += 1
            total_passed += passed
            total_tests += total

        print()

        accuracy = correct / len(problems) if problems else 0
        test_pass_rate = total_passed / total_tests if total_tests else 0

        print(f"    {name}: {accuracy:.1%} ({correct}/{len(problems)}) | Test pass rate: {test_pass_rate:.1%}")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(problems),
            "test_pass_rate": test_pass_rate,
            "tests_passed": total_passed,
            "tests_total": total_tests,
        }

    def collect_solutions(self, problems: List[AlgorithmicProblem]) -> List[dict]:
        """Collect solutions that pass ALL test cases."""
        print(f"  Collecting solutions from {len(problems)} problems...")

        solutions = []

        for i, problem in enumerate(problems):
            print(f"\r    Generating: {i+1}/{len(problems)}", end="", flush=True)

            code = self._generate_solution(problem)
            passed_all, passed, total = self._verify_solution(code, problem)

            if passed_all:
                solutions.append({
                    "problem_id": problem.problem_id,
                    "problem_type": problem.problem_type,
                    "title": problem.title,
                    "description": problem.description,
                    "function_signature": problem.function_signature,
                    "solution_code": code,
                    "test_cases_passed": passed,
                })

        print(f"\n    Collected {len(solutions)} correct solutions")
        return solutions

    def train_on_solutions(self, solutions: List[dict], iteration: int):
        """Train on collected solutions."""
        if not solutions:
            print("  No solutions to train on, skipping")
            return

        print(f"  Training on {len(solutions)} solutions...")

        # Save solutions
        solutions_file = self.experiment_dir / "solutions" / f"iter_{iteration}.jsonl"
        with open(solutions_file, "w", encoding="utf-8") as f:
            for sol in solutions:
                f.write(json.dumps(sol) + "\n")

        # Create training dataset
        from torch.utils.data import Dataset

        class SFTDataset(Dataset):
            def __init__(self, samples, tokenizer, max_length=1024):
                self.samples = samples
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.processed = [self._process(s) for s in samples]

            def _process(self, sample):
                messages = [
                    {"role": "system", "content": "You are an expert Python programmer."},
                    {"role": "user", "content": f"Implement: {sample['function_signature']}\n\n{sample['description']}"},
                    {"role": "assistant", "content": f"```python\n{sample['solution_code']}\n```"},
                ]

                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")

                return {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": enc["input_ids"].squeeze(0).clone(),
                }

            def __len__(self):
                return len(self.processed)

            def __getitem__(self, idx):
                return self.processed[idx]

        dataset = SFTDataset(solutions, self.tokenizer)

        # Setup LoRA
        if self.peft_model is None:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.peft_model = get_peft_model(self.model, lora_config)

        self.peft_model.train()

        # Train
        training_args = TrainingArguments(
            output_dir=str(self.experiment_dir / "checkpoints" / f"iter_{iteration}"),
            num_train_epochs=self.config.epochs_per_iteration,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            fp16=True,
        )

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # Merge for fast inference
        print("  Merging LoRA weights...")
        self.model = self.peft_model.merge_and_unload()
        self.peft_model = None
        self.model.eval()

        print("  Training complete!")

    def run(self):
        """Run the experiment."""
        print("\n" + "="*60)
        print("SELF-IMPROVEMENT V2 EXPERIMENT")
        print("="*60)
        print(f"Model: {self.config.base_model}")
        print(f"Problem types: {self.config.problem_types}")
        print(f"Train: {len(self.train_problems)}, Val: {len(self.val_problems)}, Test: {len(self.test_problems)}")
        print(f"Test cases per problem: {self.config.test_cases_per_problem}")
        print(f"Iterations: {self.config.num_iterations}")
        print("="*60)

        self._load_model()

        for iteration in range(self.config.num_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print("="*60)

            # Evaluate
            train_result = self.evaluate(self.train_problems, "train")
            val_result = self.evaluate(self.val_problems, "val")
            test_result = self.evaluate(self.test_problems, "test")

            # Log metrics
            metrics = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "train_accuracy": train_result["accuracy"],
                "val_accuracy": val_result["accuracy"],
                "test_accuracy": test_result["accuracy"],
                "train_test_pass_rate": train_result["test_pass_rate"],
                "val_test_pass_rate": val_result["test_pass_rate"],
                "test_test_pass_rate": test_result["test_pass_rate"],
            }
            self.metrics_history.append(metrics)

            # Save metrics
            with open(self.experiment_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")

            # Collect and train (except last iteration)
            if iteration < self.config.num_iterations - 1:
                solutions = self.collect_solutions(self.train_problems)
                self.train_on_solutions(solutions, iteration)

        # Print summary
        self._print_summary()

        return self.metrics_history

    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)

        print("\nAccuracy over iterations:")
        print("-"*50)
        print(f"{'Iter':<6} {'Train':<12} {'Val':<12} {'Test':<12}")
        print("-"*50)

        for m in self.metrics_history:
            print(f"{m['iteration']:<6} {m['train_accuracy']:.1%}{'':<6} {m['val_accuracy']:.1%}{'':<6} {m['test_accuracy']:.1%}")

        if len(self.metrics_history) > 1:
            first = self.metrics_history[0]
            last = self.metrics_history[-1]
            print("-"*50)
            print(f"Change:  {last['train_accuracy']-first['train_accuracy']:+.1%}       "
                  f"{last['val_accuracy']-first['val_accuracy']:+.1%}       "
                  f"{last['test_accuracy']-first['test_accuracy']:+.1%}")

        print("="*60)
