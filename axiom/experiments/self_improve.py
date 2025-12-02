"""Self-improvement experiment with integrated training loop.

This implements the full Expert Iteration cycle:
1. Generate solutions for training problems
2. Verify which solutions are correct
3. Train on correct solutions (LoRA SFT)
4. Evaluate on val/test sets
5. Repeat
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel

from axiom.procedural import ProceduralProblem, get_all_generators
from axiom.trainer.lora_config import get_lora_config, get_training_args, LoRASFTConfig
from axiom.trainer.data import TrainingSample, SFTDataset
from .metrics import MetricsLogger
from .evaluation import ProceduralEvaluator, EvaluationResult


@dataclass
class SelfImproveConfig:
    """Configuration for self-improvement experiment."""

    # Experiment identification
    experiment_name: str = "self_improve_v1"
    output_dir: Path = field(default_factory=lambda: Path("experiments"))

    # Model configuration
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32

    # Data configuration
    train_size: int = 30
    val_size: int = 10
    test_size: int = 10
    problem_types: list[str] = field(default_factory=lambda: ["arithmetic", "rpn", "parentheses"])
    min_difficulty: int = 3
    max_difficulty: int = 7
    seed: int = 42

    # Training configuration
    num_iterations: int = 5
    learning_rate: float = 2e-4
    epochs_per_iteration: int = 1
    batch_size: int = 1
    gradient_accumulation: int = 4

    # Evaluation configuration
    eval_every: int = 1
    test_every: int = 1

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d


class SelfImprovementExperiment:
    """
    Full self-improvement experiment with training.

    Each iteration:
    1. Evaluate current model on all problems
    2. Collect correct solutions from training problems
    3. Train on correct solutions
    4. Move to next iteration with improved model
    """

    def __init__(self, config: SelfImproveConfig):
        self.config = config
        self.experiment_dir = config.output_dir / config.experiment_name

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "solutions").mkdir(exist_ok=True)

        # Save config
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Initialize components
        self.metrics_logger = MetricsLogger(self.experiment_dir, config.experiment_name)
        self.evaluator = ProceduralEvaluator()

        # Generate problem sets
        self._generate_problem_sets()

        # Model components
        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def _generate_problem_sets(self):
        """Generate fixed problem sets."""
        print("Generating problem sets...")

        generators = get_all_generators(seed=self.config.seed)
        selected_gens = {k: v for k, v in generators.items() if k in self.config.problem_types}

        self.train_problems = []
        self.val_problems = []
        self.test_problems = []

        per_type_train = self.config.train_size // len(selected_gens)
        per_type_val = self.config.val_size // len(selected_gens)
        per_type_test = self.config.test_size // len(selected_gens)

        for name, gen in selected_gens.items():
            gen.rng.seed(self.config.seed)
            for _ in range(per_type_train):
                diff = gen.rng.randint(self.config.min_difficulty, self.config.max_difficulty)
                self.train_problems.append(gen.generate(diff))

            gen.rng.seed(self.config.seed + 1000)
            for _ in range(per_type_val):
                diff = gen.rng.randint(self.config.min_difficulty, self.config.max_difficulty)
                self.val_problems.append(gen.generate(diff))

            gen.rng.seed(self.config.seed + 2000)
            for _ in range(per_type_test):
                diff = gen.rng.randint(self.config.min_difficulty, self.config.max_difficulty)
                self.test_problems.append(gen.generate(diff))

        print(f"  Train: {len(self.train_problems)} problems")
        print(f"  Val:   {len(self.val_problems)} problems")
        print(f"  Test:  {len(self.test_problems)} problems")

        # Save problems
        self._save_problems(self.train_problems, "train.jsonl")
        self._save_problems(self.val_problems, "val.jsonl")
        self._save_problems(self.test_problems, "test.jsonl")

    def _save_problems(self, problems: list[ProceduralProblem], filename: str):
        filepath = self.experiment_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for p in problems:
                data = {
                    "problem_type": p.problem_type,
                    "problem_id": p.problem_id,
                    "title": p.title,
                    "description": p.description,
                    "function_signature": p.function_signature,
                    "input_data": p.input_data,
                    "expected_output": p.expected_output,
                    "difficulty": p.difficulty,
                    "complexity": p.complexity,
                }
                f.write(json.dumps(data) + "\n")

    def _load_model(self):
        """Load base model and tokenizer."""
        if self.model is not None:
            return

        print(f"Loading model: {self.config.base_model}...")

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

        print(f"  Model loaded on {self.model.device}")

    def _generate_response(self, prompt: str) -> str:
        """Generate response from current model."""
        model_to_use = self.peft_model if self.peft_model else self.model

        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Write ONLY the function implementation."},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(model_to_use.device)

        with torch.no_grad():
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return response

    def evaluate(self, problems: list[ProceduralProblem], name: str) -> EvaluationResult:
        """Evaluate on a set of problems."""
        print(f"  Evaluating on {name} ({len(problems)} problems)...")

        result = self.evaluator.evaluate_batch(
            problems,
            generate_fn=self._generate_response,
            verbose=False,
        )

        print(f"    {name}: {result}")
        return result

    def collect_solutions(self, problems: list[ProceduralProblem]) -> list[dict]:
        """Generate solutions and collect correct ones."""
        print(f"  Collecting solutions from {len(problems)} problems...")

        correct_solutions = []

        for i, problem in enumerate(problems):
            print(f"\r    Generating: {i+1}/{len(problems)}", end="", flush=True)

            prompt = problem.to_prompt()
            response = self._generate_response(prompt)

            # Evaluate
            result = self.evaluator.evaluate_response(problem, response)

            if result["correct"]:
                # Extract just the code
                code = self.evaluator.extract_code(response)
                if code:
                    correct_solutions.append({
                        "problem_id": problem.problem_id,
                        "problem_title": problem.title,
                        "problem_description": problem.description,
                        "function_signature": problem.function_signature,
                        "solution_code": code,
                        "model_name": self.config.base_model,
                    })

        print(f"\n    Collected {len(correct_solutions)} correct solutions")
        return correct_solutions

    def train_on_solutions(self, solutions: list[dict], iteration: int):
        """Train on collected solutions using LoRA."""
        if not solutions:
            print("  No solutions to train on, skipping training")
            return

        print(f"  Training on {len(solutions)} solutions...")

        # Save solutions to file
        solutions_file = self.experiment_dir / "solutions" / f"iter_{iteration}.jsonl"
        with open(solutions_file, "w", encoding="utf-8") as f:
            for sol in solutions:
                f.write(json.dumps(sol) + "\n")

        # Create training samples
        samples = []
        for sol in solutions:
            samples.append(TrainingSample(
                problem_id=sol["problem_id"],
                problem_title=sol["problem_title"],
                problem_description=sol["problem_description"],
                function_signature=sol["function_signature"],
                solution_code=sol["solution_code"],
                model_name=sol["model_name"],
            ))

        # Create dataset
        dataset = SFTDataset(
            samples,
            self.tokenizer,
            max_length=1024,
        )

        # Prepare model for training
        if self.peft_model is None:
            # First iteration: create new PEFT model
            lora_config = LoRASFTConfig(
                model_name=self.config.base_model,
                lora_r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                learning_rate=self.config.learning_rate,
                num_epochs=self.config.epochs_per_iteration,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation,
            )
            peft_config = get_lora_config(lora_config)
            self.peft_model = get_peft_model(self.model, peft_config)
        else:
            # Subsequent iterations: reuse existing PEFT model
            pass

        self.peft_model.train()

        # Simple training loop
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        training_args = TrainingArguments(
            output_dir=str(self.experiment_dir / "checkpoints" / f"iter_{iteration}"),
            num_train_epochs=self.config.epochs_per_iteration,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        trainer.train()

        self.peft_model.eval()
        print("  Training complete!")

    def run(self):
        """Run the full self-improvement experiment."""
        print(f"\n{'#'*60}")
        print(f"# SELF-IMPROVEMENT EXPERIMENT: {self.config.experiment_name}")
        print(f"{'#'*60}")

        self._load_model()

        all_results = []

        for iteration in range(self.config.num_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print(f"{'='*60}")

            results = {"iteration": iteration}

            # Evaluate on all sets
            train_result = self.evaluate(self.train_problems, "train")
            results["train_accuracy"] = train_result.accuracy
            results["train_correct"] = train_result.correct

            val_result = self.evaluate(self.val_problems, "val")
            results["val_accuracy"] = val_result.accuracy
            results["val_correct"] = val_result.correct

            test_result = self.evaluate(self.test_problems, "test")
            results["test_accuracy"] = test_result.accuracy
            results["test_correct"] = test_result.correct

            # Log metrics
            self.metrics_logger.log_values(
                iteration=iteration,
                train_accuracy=results["train_accuracy"],
                val_accuracy=results["val_accuracy"],
                test_accuracy=results["test_accuracy"],
                train_correct=results["train_correct"],
                train_total=len(self.train_problems),
                val_correct=results["val_correct"],
                val_total=len(self.val_problems),
                test_correct=results["test_correct"],
                test_total=len(self.test_problems),
            )

            all_results.append(results)

            # Collect and train (except last iteration)
            if iteration < self.config.num_iterations - 1:
                solutions = self.collect_solutions(self.train_problems)
                self.train_on_solutions(solutions, iteration)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results: list[dict]):
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")

        print("\nAccuracy over iterations:")
        print("-" * 50)
        print(f"{'Iter':<6} {'Train':<12} {'Val':<12} {'Test':<12}")
        print("-" * 50)

        for r in results:
            print(f"{r['iteration']:<6} {r['train_accuracy']:.1%}{'':<6} {r['val_accuracy']:.1%}{'':<6} {r['test_accuracy']:.1%}")

        print("-" * 50)

        # Calculate improvement
        if len(results) > 1:
            train_improve = results[-1]["train_accuracy"] - results[0]["train_accuracy"]
            val_improve = results[-1]["val_accuracy"] - results[0]["val_accuracy"]
            test_improve = results[-1]["test_accuracy"] - results[0]["test_accuracy"]

            print(f"\nImprovement (first -> last):")
            print(f"  Train: {train_improve:+.1%}")
            print(f"  Val:   {val_improve:+.1%}")
            print(f"  Test:  {test_improve:+.1%}")

        print(f"{'='*60}")
