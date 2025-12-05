#!/usr/bin/env python3
"""
Experiment 04: Self-Improvement with Replay Buffer

This experiment tests whether a replay buffer can prevent catastrophic forgetting.
Key insight from Experiment 03: Training on narrow data causes -30% degradation.

Strategy:
- Mix new solutions with replay from diverse problem types
- Include solutions from ALL available problem types during training
- Prevent overfitting to narrow problem set
"""

import argparse
import json
import random
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
class ReplayExperimentConfig:
    """Configuration for replay buffer experiment."""
    experiment_name: str = "04_replay_buffer"
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # Focus problems (what we're trying to improve)
    focus_problem_types: List[str] = field(default_factory=lambda: ["fibonacci", "remove_duplicates"])

    # Replay problems (diverse set to prevent forgetting)
    replay_problem_types: List[str] = field(default_factory=lambda: [
        "fizzbuzz", "reverse_string", "is_palindrome", "parentheses", "arithmetic"
    ])

    # Data generation
    train_problems_per_type: int = 5
    val_problems_per_type: int = 5
    test_problems_per_type: int = 5
    test_cases_per_problem: int = 5

    # Replay buffer settings
    replay_ratio: float = 0.5  # 50% of training data is replay
    replay_samples_per_type: int = 3  # How many replay samples per type

    # Evaluation
    samples_per_problem: int = 2  # Best-of-N evaluation

    # Training
    num_iterations: int = 2
    learning_rate: float = 2e-5  # Lower LR to reduce forgetting
    num_epochs: int = 2
    batch_size: int = 4

    seed: int = 42
    output_dir: str = "experiments/04_replay_buffer"


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    pattern = r'```(?:python)?\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

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


class ReplayBuffer:
    """Manages replay solutions from diverse problem types."""

    def __init__(self, seed: int = 42):
        self.solutions: Dict[str, List[Dict]] = {}  # problem_type -> solutions
        self.rng = random.Random(seed)

    def add_solutions(self, solutions: List[Dict]):
        """Add verified solutions to the replay buffer."""
        for sol in solutions:
            ptype = sol.get("problem_type", "unknown")
            if ptype not in self.solutions:
                self.solutions[ptype] = []
            self.solutions[ptype].append(sol)

    def sample_diverse(self, n_per_type: int) -> List[Dict]:
        """Sample solutions ensuring diversity across problem types."""
        sampled = []
        for ptype, sols in self.solutions.items():
            n = min(n_per_type, len(sols))
            sampled.extend(self.rng.sample(sols, n))
        return sampled

    def sample_balanced(self, total_samples: int) -> List[Dict]:
        """Sample balanced number of solutions across types."""
        if not self.solutions:
            return []

        n_types = len(self.solutions)
        per_type = max(1, total_samples // n_types)
        return self.sample_diverse(per_type)

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about replay buffer contents."""
        return {ptype: len(sols) for ptype, sols in self.solutions.items()}

    def save(self, path: Path):
        """Save replay buffer to file."""
        all_solutions = []
        for sols in self.solutions.values():
            all_solutions.extend(sols)

        with open(path, "w") as f:
            for sol in all_solutions:
                f.write(json.dumps(sol) + "\n")

    def load(self, path: Path):
        """Load replay buffer from file."""
        if not path.exists():
            return

        with open(path) as f:
            for line in f:
                sol = json.loads(line.strip())
                self.add_solutions([sol])


class ReplayExperiment:
    """Run self-improvement with replay buffer to prevent forgetting."""

    def __init__(self, config: ReplayExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # All problem types
        self.all_problem_types = config.focus_problem_types + config.replay_problem_types

        # Initialize generators for all types
        self.generators = {
            pt: GENERATORS_V2[pt](seed=config.seed)
            for pt in self.all_problem_types
        }

        # Replay buffer
        self.replay_buffer = ReplayBuffer(seed=config.seed)

        # Metrics tracking
        self.metrics = []

    def generate_problems(self, split: str, count_per_type: int, problem_types: List[str]) -> List[AlgorithmicProblem]:
        """Generate problems for a split."""
        problems = []
        for pt in problem_types:
            gen = self.generators[pt]
            for i in range(count_per_type):
                problem = gen.generate(
                    difficulty=5,
                    num_test_cases=self.config.test_cases_per_problem
                )
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
            ptype = pid.rsplit('_', 2)[0]
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

    def train_with_replay(self, new_solutions: List[Dict], iteration: int):
        """Fine-tune model on new solutions + replay buffer."""
        if not new_solutions and not self.replay_buffer.solutions:
            print("  No solutions to train on!")
            return

        # Calculate how many replay samples to use
        n_new = len(new_solutions)
        n_replay = int(n_new * self.config.replay_ratio / (1 - self.config.replay_ratio))
        n_replay = max(n_replay, len(self.config.replay_problem_types) * self.config.replay_samples_per_type)

        # Sample from replay buffer
        replay_solutions = self.replay_buffer.sample_balanced(n_replay)

        # Combine new + replay
        all_solutions = new_solutions + replay_solutions
        random.shuffle(all_solutions)

        print(f"  Training on {len(all_solutions)} solutions:")
        print(f"    - New (focus): {n_new}")
        print(f"    - Replay (diverse): {len(replay_solutions)}")

        # Show replay buffer stats
        stats = self.replay_buffer.get_stats()
        if stats:
            print(f"  Replay buffer contents:")
            for ptype, count in sorted(stats.items()):
                print(f"    - {ptype}: {count}")

        # Prepare training data
        def format_example(sol):
            messages = [
                {"role": "system", "content": "You are an expert Python programmer."},
                {"role": "user", "content": sol["prompt"]},
                {"role": "assistant", "content": f"```python\n{sol['code']}\n```"}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)

        texts = [format_example(sol) for sol in all_solutions]

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

        # Training args - lower LR for stability
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
        print("EXPERIMENT 04: SELF-IMPROVEMENT WITH REPLAY BUFFER")
        print("=" * 70)
        print(f"Focus problems: {self.config.focus_problem_types}")
        print(f"Replay problems: {self.config.replay_problem_types}")
        print(f"Replay ratio: {self.config.replay_ratio:.0%}")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 70)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Generate problem sets for ALL types (focus + replay)
        print("\nGenerating problem sets...")

        # Focus problems - for evaluation
        focus_train = self.generate_problems("train", self.config.train_problems_per_type, self.config.focus_problem_types)
        focus_val = self.generate_problems("val", self.config.val_problems_per_type, self.config.focus_problem_types)
        focus_test = self.generate_problems("test", self.config.test_problems_per_type, self.config.focus_problem_types)

        # Replay problems - for diversity training
        replay_train = self.generate_problems("train", self.config.train_problems_per_type, self.config.replay_problem_types)

        print(f"  Focus train: {len(focus_train)} problems ({self.config.focus_problem_types})")
        print(f"  Focus val: {len(focus_val)} problems")
        print(f"  Focus test: {len(focus_test)} problems")
        print(f"  Replay train: {len(replay_train)} problems ({self.config.replay_problem_types})")

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

        save_problems(focus_train, "focus_train")
        save_problems(focus_val, "focus_val")
        save_problems(focus_test, "focus_test")
        save_problems(replay_train, "replay_train")

        # Load model
        self.load_model()

        # Pre-populate replay buffer with solutions from diverse problems
        print("\nPre-populating replay buffer...")
        replay_solutions = self.collect_solutions(replay_train)
        self.replay_buffer.add_solutions(replay_solutions)
        print(f"  Collected {len(replay_solutions)} diverse solutions")

        # Save replay buffer
        self.replay_buffer.save(self.output_dir / "replay_buffer.jsonl")

        # Run iterations
        for iteration in range(self.config.num_iterations):
            print(f"\n{'=' * 70}")
            print(f"ITERATION {iteration}")
            print("=" * 70)

            # Evaluate on focus problems
            print(f"\nEvaluating on focus validation set ({len(focus_val)} problems)...")
            val_results = self.evaluate_best_of_n(focus_val, self.config.samples_per_problem)

            print(f"\nEvaluating on focus test set ({len(focus_test)} problems)...")
            test_results = self.evaluate_best_of_n(focus_test, self.config.samples_per_problem)

            # Record metrics
            metric = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "val": val_results,
                "test": test_results,
                "replay_buffer_stats": self.replay_buffer.get_stats()
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
                print(f"\nCollecting solutions from focus training set...")
                focus_solutions = self.collect_solutions(focus_train)
                print(f"  Collected {len(focus_solutions)} focus solutions")

                # Save solutions
                sol_dir = self.output_dir / "solutions"
                sol_dir.mkdir(exist_ok=True)
                with open(sol_dir / f"iter_{iteration}.jsonl", "w") as f:
                    for sol in focus_solutions:
                        f.write(json.dumps(sol) + "\n")

                # Train with replay buffer
                self.train_with_replay(focus_solutions, iteration)

                # Add new solutions to replay buffer for future iterations
                self.replay_buffer.add_solutions(focus_solutions)

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
    parser = argparse.ArgumentParser(description="Run self-improvement with replay buffer")
    parser.add_argument("--experiment", default="04_replay_buffer", help="Experiment name")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Base model")
    parser.add_argument("--focus-problems", nargs="+", default=["fibonacci", "remove_duplicates"],
                        help="Focus problem types to improve")
    parser.add_argument("--replay-problems", nargs="+",
                        default=["fizzbuzz", "reverse_string", "is_palindrome", "parentheses", "arithmetic"],
                        help="Replay problem types for diversity")
    parser.add_argument("--train-per-type", type=int, default=5, help="Training problems per type")
    parser.add_argument("--iterations", type=int, default=2, help="Number of iterations")
    parser.add_argument("--samples", type=int, default=2, help="Samples per problem for evaluation")
    parser.add_argument("--replay-ratio", type=float, default=0.5, help="Ratio of replay to new samples")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = ReplayExperimentConfig(
        experiment_name=args.experiment,
        base_model=args.model,
        focus_problem_types=args.focus_problems,
        replay_problem_types=args.replay_problems,
        train_problems_per_type=args.train_per_type,
        num_iterations=args.iterations,
        samples_per_problem=args.samples,
        replay_ratio=args.replay_ratio,
        learning_rate=args.lr,
        seed=args.seed,
        output_dir=f"experiments/{args.experiment}"
    )

    experiment = ReplayExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
