"""Grokking experiment runner for observing generalization."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from axiom.procedural import (
    ProceduralProblem,
    ProceduralDataStream,
    get_all_generators,
)
from .metrics import MetricsLogger, ExperimentMetrics
from .evaluation import ProceduralEvaluator, EvaluationResult


@dataclass
class GrokkingConfig:
    """Configuration for a grokking experiment."""

    # Experiment identification
    experiment_name: str = "grokking_v1"
    output_dir: Path = field(default_factory=lambda: Path("experiments"))

    # Model configuration
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # Data configuration
    train_size: int = 500
    val_size: int = 100
    test_size: int = 100
    problem_types: list[str] = field(default_factory=lambda: ["arithmetic", "rpn", "parentheses"])
    min_difficulty: int = 3
    max_difficulty: int = 7
    seed: int = 42

    # Training configuration
    num_iterations: int = 50
    samples_per_problem: int = 4
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    epochs_per_iteration: int = 1

    # Evaluation configuration
    eval_every: int = 5
    test_every: int = 10
    checkpoint_every: int = 10

    # Grokking detection
    grokking_threshold: float = 0.2
    grokking_window: int = 10
    early_stopping_patience: int = 20

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GrokkingConfig":
        """Create from dictionary."""
        if "output_dir" in d:
            d["output_dir"] = Path(d["output_dir"])
        return cls(**d)


class GrokkingExperiment:
    """
    Runs a grokking experiment to observe generalization.

    The experiment:
    1. Generates fixed train/val/test splits of procedural problems
    2. Iteratively trains the model on its own verified solutions
    3. Tracks train/val/test accuracy over iterations
    4. Detects when "grokking" occurs (sudden generalization)
    """

    def __init__(self, config: GrokkingConfig):
        """
        Initialize the experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.experiment_dir = config.output_dir / config.experiment_name

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)

        # Save config
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Initialize components
        self.metrics_logger = MetricsLogger(self.experiment_dir, config.experiment_name)
        self.evaluator = ProceduralEvaluator()

        # Generate fixed problem sets
        self._generate_problem_sets()

        # Model and tokenizer (loaded lazily)
        self.model = None
        self.tokenizer = None

    def _generate_problem_sets(self):
        """Generate fixed train/val/test problem sets."""
        print("Generating problem sets...")

        # Use different seeds for each split to ensure no overlap
        train_stream = ProceduralDataStream(
            problem_types=self.config.problem_types,
            min_difficulty=self.config.min_difficulty,
            max_difficulty=self.config.max_difficulty,
            seed=self.config.seed,
        )
        val_stream = ProceduralDataStream(
            problem_types=self.config.problem_types,
            min_difficulty=self.config.min_difficulty,
            max_difficulty=self.config.max_difficulty,
            seed=self.config.seed + 1000,
        )
        test_stream = ProceduralDataStream(
            problem_types=self.config.problem_types,
            min_difficulty=self.config.min_difficulty,
            max_difficulty=self.config.max_difficulty,
            seed=self.config.seed + 2000,
        )

        # Generate problems using the generators directly
        generators = get_all_generators(seed=self.config.seed)
        selected_gens = {k: v for k, v in generators.items() if k in self.config.problem_types}

        self.train_problems = []
        self.val_problems = []
        self.test_problems = []

        # Distribute problems evenly across types
        per_type_train = self.config.train_size // len(selected_gens)
        per_type_val = self.config.val_size // len(selected_gens)
        per_type_test = self.config.test_size // len(selected_gens)

        for name, gen in selected_gens.items():
            # Training problems
            gen.rng.seed(self.config.seed)
            for _ in range(per_type_train):
                diff = gen.rng.randint(self.config.min_difficulty, self.config.max_difficulty)
                self.train_problems.append(gen.generate(diff))

            # Validation problems (different seed)
            gen.rng.seed(self.config.seed + 1000)
            for _ in range(per_type_val):
                diff = gen.rng.randint(self.config.min_difficulty, self.config.max_difficulty)
                self.val_problems.append(gen.generate(diff))

            # Test problems (different seed)
            gen.rng.seed(self.config.seed + 2000)
            for _ in range(per_type_test):
                diff = gen.rng.randint(self.config.min_difficulty, self.config.max_difficulty)
                self.test_problems.append(gen.generate(diff))

        print(f"  Train: {len(self.train_problems)} problems")
        print(f"  Val:   {len(self.val_problems)} problems")
        print(f"  Test:  {len(self.test_problems)} problems")

        # Save problem sets for reproducibility
        self._save_problem_set(self.train_problems, "train.jsonl")
        self._save_problem_set(self.val_problems, "val.jsonl")
        self._save_problem_set(self.test_problems, "test.jsonl")

    def _save_problem_set(self, problems: list[ProceduralProblem], filename: str):
        """Save problem set to file."""
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
        """Load model and tokenizer."""
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
        """Generate a response from the model."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert Python programmer. Write ONLY the function implementation.",
            },
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,  # Reduced for speed
                do_sample=False,  # Greedy decoding is faster
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response

    def evaluate(
        self,
        problems: list[ProceduralProblem],
        name: str = "eval",
        verbose: bool = False,
    ) -> EvaluationResult:
        """Evaluate model on a set of problems."""
        print(f"  Evaluating on {name} ({len(problems)} problems)...")

        result = self.evaluator.evaluate_batch(
            problems,
            generate_fn=self._generate_response,
            verbose=verbose,
        )

        print(f"    {name}: {result}")
        return result

    def run_iteration(self, iteration: int) -> dict:
        """
        Run a single iteration of the experiment.

        Args:
            iteration: Current iteration number

        Returns:
            Dictionary with iteration results
        """
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")

        results = {"iteration": iteration}

        # Evaluate on training set
        train_result = self.evaluate(self.train_problems, "train")
        results["train_accuracy"] = train_result.accuracy
        results["train_correct"] = train_result.correct
        results["train_total"] = train_result.total

        # Evaluate on validation set
        val_result = self.evaluate(self.val_problems, "val")
        results["val_accuracy"] = val_result.accuracy
        results["val_correct"] = val_result.correct
        results["val_total"] = val_result.total

        # Evaluate on test set (less frequently)
        if iteration % self.config.test_every == 0:
            test_result = self.evaluate(self.test_problems, "test")
            results["test_accuracy"] = test_result.accuracy
            results["test_correct"] = test_result.correct
            results["test_total"] = test_result.total

        return results

    def run(self, resume_from: Optional[int] = None):
        """
        Run the full grokking experiment.

        Args:
            resume_from: Iteration to resume from (optional)
        """
        print(f"\n{'#'*60}")
        print(f"# GROKKING EXPERIMENT: {self.config.experiment_name}")
        print(f"{'#'*60}")

        # Load model
        self._load_model()

        start_iteration = resume_from or 0

        # Main loop
        for iteration in range(start_iteration, self.config.num_iterations):
            # Run iteration
            if iteration % self.config.eval_every == 0:
                results = self.run_iteration(iteration)

                # Log metrics
                self.metrics_logger.log_values(
                    iteration=iteration,
                    train_accuracy=results["train_accuracy"],
                    val_accuracy=results["val_accuracy"],
                    test_accuracy=results.get("test_accuracy"),
                    train_correct=results["train_correct"],
                    train_total=results["train_total"],
                    val_correct=results["val_correct"],
                    val_total=results["val_total"],
                    test_correct=results.get("test_correct", 0),
                    test_total=results.get("test_total", 0),
                )

                # Check for grokking
                grokking_iter = self.metrics_logger.detect_grokking(
                    threshold=self.config.grokking_threshold,
                    window=self.config.grokking_window,
                )
                if grokking_iter:
                    print(f"\n*** GROKKING DETECTED at iteration {grokking_iter}! ***\n")

            # TODO: Add training step here
            # For now, this is an evaluation-only experiment
            # Full training would require integrating with the trainer module

            # Checkpoint
            if iteration % self.config.checkpoint_every == 0:
                self._save_checkpoint(iteration)

        # Final summary
        self._print_summary()

    def _save_checkpoint(self, iteration: int):
        """Save experiment checkpoint."""
        checkpoint_dir = self.experiment_dir / "checkpoints" / f"iter_{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics summary
        summary = self.metrics_logger.get_summary()
        with open(checkpoint_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Checkpoint saved: {checkpoint_dir}")

    def _print_summary(self):
        """Print experiment summary."""
        summary = self.metrics_logger.get_summary()

        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Experiment: {summary.get('experiment_name', 'N/A')}")
        print(f"Iterations: {summary.get('total_iterations', 0)}")
        print(f"Final Train Accuracy: {summary.get('final_train_accuracy', 0):.1%}")
        print(f"Final Val Accuracy: {summary.get('final_val_accuracy', 0):.1%}")
        print(f"Best Val Accuracy: {summary.get('best_val_accuracy', 0):.1%} (iter {summary.get('best_val_iteration', 0)})")
        print(f"Grokking Detected: {summary.get('grokking_detected', False)}")
        if summary.get("grokking_iteration"):
            print(f"Grokking Iteration: {summary.get('grokking_iteration')}")
        print(f"{'='*60}")
