#!/usr/bin/env python3
"""
Run LoRA SFT training on verified solutions.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --solutions data/synthetic/solutions_baseline.jsonl
    python scripts/run_training.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --epochs 5
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.trainer import LoRASFTTrainer
from axiom.trainer.lora_config import LoRASFTConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run LoRA SFT training on verified solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults (uses solutions_baseline.jsonl)
  python scripts/run_training.py

  # Train on a specific solutions file
  python scripts/run_training.py --solutions data/synthetic/solutions_exp1.jsonl

  # Use a different base model
  python scripts/run_training.py --model Qwen/Qwen2.5-Coder-3B-Instruct

  # Adjust training parameters
  python scripts/run_training.py --epochs 5 --lr 1e-4 --lora-r 32

  # Name the output experiment
  python scripts/run_training.py --experiment v1
""",
    )

    # Model settings
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model to fine-tune (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)",
    )

    # Data settings
    parser.add_argument(
        "--solutions",
        default="data/synthetic/solutions_baseline.jsonl",
        help="Path to solutions JSONL file (default: data/synthetic/solutions_baseline.jsonl)",
    )

    # LoRA settings
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha/scaling (default: 32)",
    )

    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size (default: 1)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        default="models/lora-sft",
        help="Output directory for model (default: models/lora-sft)",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Experiment name - saves to models/lora-sft-{name}",
    )

    # Memory optimization
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more VRAM)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 instead of float16 (requires Ampere+ GPU)",
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output_dir
    if args.experiment:
        output_dir = f"models/lora-sft-{args.experiment}"

    # Build config
    config = LoRASFTConfig(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_length,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.bf16,
        bf16=args.bf16,
    )

    # Check solutions file exists
    solutions_path = Path(args.solutions)
    if not solutions_path.exists():
        print(f"Error: Solutions file not found: {solutions_path}")
        print("\nHint: Run the pipeline first to generate solutions:")
        print("  python scripts/run_pipeline.py")
        return 1

    # Create trainer and run
    trainer = LoRASFTTrainer(config=config, solutions_path=solutions_path)

    try:
        metrics = trainer.train()

        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Model saved to: {output_dir}")
        print("\nTo test the fine-tuned model, run:")
        print(f"  python scripts/run_pipeline.py --model {output_dir} --experiment finetuned")
        print("=" * 50)

        return 0

    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
