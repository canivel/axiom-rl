#!/usr/bin/env python3
"""
Experiment 02: Focused Improvement on Weak Problems

This experiment targets specific problems where the base model shows weakness:
- remove_duplicates: 12.5% baseline accuracy (1/8)
- fibonacci: 62.5% baseline accuracy (5/8)

We generate 5x more training data for these problems and run self-improvement
to measure if targeted training improves performance.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
from datasets import Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.procedural.base_v2 import AlgorithmicProblem, TestCase
from axiom.procedural.generators_v2 import GENERATORS_V2


@dataclass
class FocusedExperimentConfig:
    """Configuration for focused improvement experiment."""
    experiment_name: str = "02_focused_improvement"
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # Target weak problems
    problem_types: List[str] = field(default_factory=lambda: ["remove_duplicates", "fibonacci"])

    # 5x more data than baseline
    train_problems_per_type: int = 25  # 5x the normal 5
    val_problems_per_type: int = 10
    test_problems_per_type: int = 10
    test_cases_per_problem: int = 5

    # Evaluation
    samples_per_problem: int = 8  # Best-of-N evaluation

    # Training
    num_iterations: int = 3
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 4

    seed: int = 42
    output_dir: str = "experiments/02_focused_improvement"


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    # Try markdown code blocks first
    pattern = r'```(?:python)?\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find function definition
    pattern2 = r'(def \w+\([^)]*\)[^:]*:.*?)(?=\n\n[A-Z]|\n\n#|\Z)'
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        return match2.group(1).strip()

    return response.strip()


def verify_solution(code: str, problem: AlgorithmicProblem) -> tuple:
    """Verify solution against all test cases."""
    passed = 0
    total = len(problem.test_cases)

    try:
        local_ns = {'List': list}
        exec(code, {'__builtins__': __builtins__, 'List': list}, local_ns)

        # Extract function name
        func_name = problem.function_signature.split('(')[0].replace('def ', '').strip()
        func = local_ns.get(func_name)

        if func is None:
            return False, 0, total

        for tc in problem.test_cases:
            try:
                result = func(*tc.input_args)
                if result == tc.expected_output:
                    passed += 1
            except Exception:
                pass

        return passed == total, passed, total
    except Exception:
        return False, 0, total


class FocusedImprovementExperiment:
    """Run focused self-improvement on weak problems."""

    def __init__(self, config: FocusedExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize generators
        self.generators = {
            pt: GENERATORS_V2[pt](seed=config.seed)
            for pt in config.problem_types
        }

        # Metrics tracking
        self.metrics = []

    def generate_problems(self, split: str, count_per_type: int) -> List[AlgorithmicProblem]:
        """Generate problems for a split."""
        problems = []
        for pt, gen in self.generators.items():
            for i in range(count_per_type):
                problem = gen.generate(
                    difficulty=5,
                    num_test_cases=self.config.test_cases_per_problem
                )
                # Override problem_id with split-specific naming
                problem.problem_id = f"{pt}_{split}_{i}"
                problems.append(problem)
        return problems

    def load_model(self, model_path: Optional[str] = None):
        """Load model and tokenizer."""
        path = model_path or self.config.base_model
        print(f"Loading model: {path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"  Loaded on: {self.model.device}")

    def generate_solution(self, problem: AlgorithmicProblem) -> str:
        """Generate a solution for a problem."""
        prompt = f"""Implement the following function:

{problem.function_signature}

{problem.description}

Return only the Python function implementation."""

        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Implement the requested function."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response

    def evaluate_best_of_n(self, problems: List[AlgorithmicProblem], n_samples: int = 8) -> Dict[str, Any]:
        """Evaluate using Best-of-N methodology."""
        results = {}

        for problem in problems:
            passed_count = 0
            for _ in range(n_samples):
                response = self.generate_solution(problem)
                code = extract_code(response)
                passed_all, _, _ = verify_solution(code, problem)
                if passed_all:
                    passed_count += 1

            results[problem.problem_id] = {
                "passed": passed_count,
                "total": n_samples,
                "accuracy": passed_count / n_samples * 100
            }
            print(f"  {problem.problem_id}: {passed_count}/{n_samples}")

        # Aggregate by problem type
        by_type = {}
        for pid, res in results.items():
            ptype = pid.rsplit('_', 2)[0]  # Extract problem type
            if ptype not in by_type:
                by_type[ptype] = {"passed": 0, "total": 0}
            by_type[ptype]["passed"] += res["passed"]
            by_type[ptype]["total"] += res["total"]

        for ptype in by_type:
            by_type[ptype]["accuracy"] = by_type[ptype]["passed"] / by_type[ptype]["total"] * 100

        overall_passed = sum(r["passed"] for r in results.values())
        overall_total = sum(r["total"] for r in results.values())

        return {
            "by_problem": results,
            "by_type": by_type,
            "overall": {
                "passed": overall_passed,
                "total": overall_total,
                "accuracy": overall_passed / overall_total * 100
            }
        }

    def collect_solutions(self, problems: List[AlgorithmicProblem]) -> List[Dict]:
        """Collect verified solutions for training."""
        solutions = []

        for problem in problems:
            response = self.generate_solution(problem)
            code = extract_code(response)
            passed_all, passed, total = verify_solution(code, problem)

            if passed_all:
                solutions.append({
                    "problem_id": problem.problem_id,
                    "problem_type": problem.problem_type,
                    "prompt": problem.to_prompt(),
                    "code": code,
                    "response": response
                })

        return solutions

    def train_on_solutions(self, solutions: List[Dict], iteration: int):
        """Fine-tune model on verified solutions."""
        if not solutions:
            print("  No solutions to train on!")
            return

        print(f"  Training on {len(solutions)} solutions...")

        # Prepare training data
        def format_example(sol):
            messages = [
                {"role": "system", "content": "You are an expert Python programmer."},
                {"role": "user", "content": sol["prompt"]},
                {"role": "assistant", "content": f"```python\n{sol['code']}\n```"}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)

        texts = [format_example(sol) for sol in solutions]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt"
        )

        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone()
        })

        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        peft_model = get_peft_model(self.model, lora_config)

        # Training args
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"checkpoints/iter_{iteration}"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            fp16=True,
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # Merge and save
        print("  Merging LoRA weights...")
        self.model = peft_model.merge_and_unload()

        model_path = self.output_dir / f"models/iter_{iteration}"
        model_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        print("  Training complete!")

    def run(self):
        """Run the full experiment."""
        print("=" * 70)
        print("EXPERIMENT 02: FOCUSED IMPROVEMENT ON WEAK PROBLEMS")
        print("=" * 70)
        print(f"Target problems: {self.config.problem_types}")
        print(f"Training problems per type: {self.config.train_problems_per_type}")
        print(f"Iterations: {self.config.num_iterations}")
        print("=" * 70)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Generate problem sets
        print("\nGenerating problem sets...")
        train_problems = self.generate_problems("train", self.config.train_problems_per_type)
        val_problems = self.generate_problems("val", self.config.val_problems_per_type)
        test_problems = self.generate_problems("test", self.config.test_problems_per_type)

        print(f"  Train: {len(train_problems)} problems")
        print(f"  Val: {len(val_problems)} problems")
        print(f"  Test: {len(test_problems)} problems")

        # Save problems
        def save_problems(problems, name):
            data = []
            for p in problems:
                data.append({
                    "problem_id": p.problem_id,
                    "problem_type": p.problem_type,
                    "title": p.title,
                    "description": p.description,
                    "function_signature": p.function_signature,
                    "test_cases": [{"input": tc.input_args, "output": tc.expected_output} for tc in p.test_cases]
                })
            with open(self.output_dir / f"{name}.json", "w") as f:
                json.dump(data, f, indent=2)

        save_problems(train_problems, "train")
        save_problems(val_problems, "val")
        save_problems(test_problems, "test")

        # Load model
        self.load_model()

        # Run iterations
        for iteration in range(self.config.num_iterations):
            print(f"\n{'=' * 70}")
            print(f"ITERATION {iteration}")
            print("=" * 70)

            # Evaluate
            print(f"\nEvaluating on validation set ({len(val_problems)} problems)...")
            val_results = self.evaluate_best_of_n(val_problems, self.config.samples_per_problem)

            print(f"\nEvaluating on test set ({len(test_problems)} problems)...")
            test_results = self.evaluate_best_of_n(test_problems, self.config.samples_per_problem)

            # Record metrics
            metric = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "val": val_results,
                "test": test_results
            }
            self.metrics.append(metric)

            # Save metrics
            with open(self.output_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metric) + "\n")

            print(f"\n  Val accuracy by type:")
            for ptype, res in val_results["by_type"].items():
                print(f"    {ptype}: {res['accuracy']:.1f}%")
            print(f"  Val overall: {val_results['overall']['accuracy']:.1f}%")

            print(f"\n  Test accuracy by type:")
            for ptype, res in test_results["by_type"].items():
                print(f"    {ptype}: {res['accuracy']:.1f}%")
            print(f"  Test overall: {test_results['overall']['accuracy']:.1f}%")

            # Collect and train (except last iteration)
            if iteration < self.config.num_iterations - 1:
                print(f"\nCollecting solutions from training set...")
                solutions = self.collect_solutions(train_problems)
                print(f"  Collected {len(solutions)} verified solutions")

                # Save solutions
                sol_dir = self.output_dir / "solutions"
                sol_dir.mkdir(exist_ok=True)
                with open(sol_dir / f"iter_{iteration}.jsonl", "w") as f:
                    for sol in solutions:
                        f.write(json.dumps(sol) + "\n")

                self.train_on_solutions(solutions, iteration)

        # Final summary
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)

        print("\nAccuracy over iterations:")
        print("-" * 70)
        print(f"{'Iter':<6} {'Val Overall':<15} {'Test Overall':<15}")
        print("-" * 70)

        for m in self.metrics:
            print(f"{m['iteration']:<6} {m['val']['overall']['accuracy']:<15.1f} {m['test']['overall']['accuracy']:<15.1f}")

        if len(self.metrics) >= 2:
            first = self.metrics[0]
            last = self.metrics[-1]
            val_change = last['val']['overall']['accuracy'] - first['val']['overall']['accuracy']
            test_change = last['test']['overall']['accuracy'] - first['test']['overall']['accuracy']
            print("-" * 70)
            print(f"{'Change':<6} {val_change:+.1f}%{'':<9} {test_change:+.1f}%")

        print("=" * 70)

        # Save final summary
        summary = {
            "experiment": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "metrics": self.metrics,
            "final_results": {
                "val": self.metrics[-1]["val"] if self.metrics else None,
                "test": self.metrics[-1]["test"] if self.metrics else None
            }
        }

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {self.output_dir}/")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run focused improvement experiment")
    parser.add_argument("--experiment", default="02_focused_improvement", help="Experiment name")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base model")
    parser.add_argument("--problems", nargs="+", default=["remove_duplicates", "fibonacci"],
                        help="Problem types to focus on")
    parser.add_argument("--train-per-type", type=int, default=25, help="Training problems per type")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument("--samples", type=int, default=8, help="Samples per problem for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = FocusedExperimentConfig(
        experiment_name=args.experiment,
        base_model=args.model,
        problem_types=args.problems,
        train_problems_per_type=args.train_per_type,
        num_iterations=args.iterations,
        samples_per_problem=args.samples,
        seed=args.seed,
        output_dir=f"experiments/{args.experiment}"
    )

    experiment = FocusedImprovementExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
