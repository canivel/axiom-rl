"""LoRA SFT Trainer implementation."""

from pathlib import Path
from typing import Optional

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
)

from .data import SFTDataset, TrainingSample, create_train_val_split, load_training_data
from .lora_config import LoRASFTConfig, get_lora_config, get_training_args


class LoRASFTTrainer:
    """
    Trainer for LoRA-based Supervised Fine-Tuning.

    Implements the "Trainer" component of Expert Iteration:
    - Takes verified solutions as training data
    - Fine-tunes the model using LoRA (parameter-efficient)
    - Produces Model N+1 from Model N
    """

    def __init__(
        self,
        config: Optional[LoRASFTConfig] = None,
        solutions_path: Optional[Path] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: LoRASFTConfig instance (uses defaults if None)
            solutions_path: Path to solutions.jsonl (default: data/synthetic/solutions.jsonl)
        """
        self.config = config or LoRASFTConfig()
        self.solutions_path = solutions_path or Path("data/synthetic/solutions_baseline.jsonl")

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self) -> None:
        """Load model, tokenizer, and prepare for training."""
        print(f"Loading model: {self.config.model_name}")

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for training with gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Apply LoRA
        lora_config = get_lora_config(self.config)
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        print("Model setup complete!")

    def prepare_data(self) -> tuple[SFTDataset, SFTDataset]:
        """
        Load and prepare training data.

        Returns:
            (train_dataset, val_dataset)
        """
        print(f"Loading training data from: {self.solutions_path}")

        # Load samples
        samples = load_training_data(self.solutions_path)
        print(f"Loaded {len(samples)} verified solutions")

        # Split into train/val
        train_samples, val_samples = create_train_val_split(samples, val_ratio=0.1)
        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

        # Create datasets
        train_dataset = SFTDataset(
            train_samples,
            self.tokenizer,
            max_length=self.config.max_seq_length,
        )
        val_dataset = SFTDataset(
            val_samples,
            self.tokenizer,
            max_length=self.config.max_seq_length,
        )

        return train_dataset, val_dataset

    def train(self) -> dict:
        """
        Run the training loop.

        Returns:
            Training metrics dict
        """
        # Setup model if not already done
        if self.model is None:
            self.setup()

        # Prepare data
        train_dataset, val_dataset = self.prepare_data()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked
        )

        # Get training arguments
        training_args = get_training_args(self.config)

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        # Train!
        print("\n" + "=" * 50)
        print("Starting LoRA SFT Training")
        print("=" * 50)
        print(f"Model: {self.config.model_name}")
        print(f"LoRA rank: {self.config.lora_r}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size} x {self.config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 50 + "\n")

        train_result = self.trainer.train()

        # Save final model
        self.save()

        return train_result.metrics

    def save(self, output_dir: Optional[str] = None) -> None:
        """
        Save the LoRA adapter.

        Args:
            output_dir: Directory to save to (uses config default if None)
        """
        save_dir = output_dir or self.config.output_dir
        print(f"\nSaving LoRA adapter to: {save_dir}")

        # Save LoRA adapter
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save config for reference
        import json

        config_path = Path(save_dir) / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "base_model": self.config.model_name,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "num_epochs": self.config.num_epochs,
                    "learning_rate": self.config.learning_rate,
                    "solutions_path": str(self.solutions_path),
                },
                f,
                indent=2,
            )

        print("Model saved successfully!")

    def evaluate(self) -> dict:
        """
        Evaluate the model on validation set.

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Must call train() before evaluate()")

        return self.trainer.evaluate()
