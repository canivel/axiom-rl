"""LoRA and training configuration for SFT."""

from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig, TaskType
from transformers import TrainingArguments


@dataclass
class LoRASFTConfig:
    """Configuration for LoRA SFT training."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    torch_dtype: str = "float16"

    # LoRA settings
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training settings
    output_dir: str = "models/lora-sft"
    num_epochs: int = 3
    batch_size: int = 1  # Small for 12GB VRAM
    gradient_accumulation_steps: int = 8  # Effective batch = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50

    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False  # Set True if your GPU supports bf16


def get_lora_config(config: Optional[LoRASFTConfig] = None) -> LoraConfig:
    """
    Create LoRA configuration for PEFT.

    Args:
        config: LoRASFTConfig instance (uses defaults if None)

    Returns:
        LoraConfig for peft
    """
    if config is None:
        config = LoRASFTConfig()

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def get_training_args(config: Optional[LoRASFTConfig] = None) -> TrainingArguments:
    """
    Create TrainingArguments for the Trainer.

    Args:
        config: LoRASFTConfig instance (uses defaults if None)

    Returns:
        TrainingArguments for transformers Trainer
    """
    if config is None:
        config = LoRASFTConfig()

    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",  # Disable wandb/tensorboard for MVP
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
