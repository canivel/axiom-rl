"""Data loading and preprocessing for SFT training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from torch.utils.data import Dataset


@dataclass
class TrainingSample:
    """A single training sample for SFT."""

    problem_id: str
    problem_title: str
    problem_description: str
    function_signature: str
    solution_code: str
    model_name: str
    thinking: Optional[str] = None  # Reasoning trace (for cold start data)

    def to_prompt_completion(self) -> dict:
        """Convert to prompt-completion format for SFT."""
        # Build the user prompt (same format as generation)
        user_prompt = f"""Solve the following problem by implementing the function.

## Problem: {self.problem_title}

{self.problem_description}

## Function Signature
```python
{self.function_signature}
    # Your implementation here
```

Write ONLY the complete function implementation. Do not include any explanations, examples, or test code."""

        # The completion includes thinking (if present) + code
        if self.thinking:
            # Cold start format: include reasoning trace
            completion = f"<think>\n{self.thinking}\n</think>\n\n```python\n{self.solution_code}\n```"
        else:
            # Standard format: just code
            completion = f"```python\n{self.solution_code}\n```"

        return {
            "prompt": user_prompt,
            "completion": completion,
            "problem_id": self.problem_id,
        }


def load_training_data(
    solutions_path: Path,
    max_samples: Optional[int] = None,
) -> list[TrainingSample]:
    """
    Load verified solutions from JSONL file.

    Args:
        solutions_path: Path to solutions.jsonl file
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of TrainingSample objects
    """
    samples = []

    with open(solutions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            sample = TrainingSample(
                problem_id=data["problem_id"],
                problem_title=data["problem_title"],
                problem_description=data["problem_description"],
                function_signature=data["function_signature"],
                solution_code=data["solution_code"],
                model_name=data.get("model_name", data.get("teacher_model", "unknown")),
                thinking=data.get("thinking"),  # Optional reasoning trace
            )
            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                break

    return samples


class SFTDataset(Dataset):
    """PyTorch Dataset for SFT training on verified solutions."""

    def __init__(
        self,
        samples: list[TrainingSample],
        tokenizer,
        max_length: int = 2048,
    ):
        """
        Initialize the dataset.

        Args:
            samples: List of TrainingSample objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Preprocess all samples to chat format
        self.processed_samples = [self._process_sample(s) for s in samples]

    def _process_sample(self, sample: TrainingSample) -> dict:
        """Convert sample to tokenized chat format."""
        pc = sample.to_prompt_completion()

        # Determine if this is a cold start sample (has thinking)
        has_thinking = sample.thinking is not None

        # Build messages in chat format
        if has_thinking:
            system_content = """You are an expert Python programmer. Your task is to solve algorithmic problems by writing clean, efficient, and correct Python code.

When solving problems:
1. First, reason through the problem inside <think> tags
2. Then, provide the complete Python code inside ```python``` blocks

Your reasoning should include:
- Understanding the problem inputs/outputs
- Considering edge cases
- Planning the approach and algorithm
- Analyzing time/space complexity

Rules for code:
- Use standard Python libraries only (no external packages)
- Write clear, readable code
- Handle edge cases appropriately"""
        else:
            system_content = """You are an expert Python programmer. Your task is to solve algorithmic problems by writing clean, efficient, and correct Python code.

Rules:
1. Write ONLY the function implementation - no explanations, no test code
2. The function signature is provided - implement the function body
3. Use standard Python libraries only (no external packages)
4. Write clear, readable code
5. Handle edge cases appropriately
6. Return the result as specified"""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": pc["prompt"]},
            {"role": "assistant", "content": pc["completion"]},
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0).clone(),
        }

    def __len__(self) -> int:
        return len(self.processed_samples)

    def __getitem__(self, idx: int) -> dict:
        return self.processed_samples[idx]


def create_train_val_split(
    samples: list[TrainingSample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    """
    Split samples into training and validation sets.

    Args:
        samples: All training samples
        val_ratio: Fraction for validation (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        (train_samples, val_samples)
    """
    import random

    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]
