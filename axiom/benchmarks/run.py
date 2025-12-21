"""
CLI for running axiom-rl benchmarks.

Usage:
    python -m axiom.benchmarks.run --model <path> --benchmarks all
    python -m axiom.benchmarks.run --model <path> --benchmarks math500 gpqa_diamond
    python -m axiom.benchmarks.run --list

Examples:
    # Run all benchmarks on a model
    python -m axiom.benchmarks.run --model Qwen/Qwen2.5-Coder-1.5B-Instruct --benchmarks all

    # Run specific benchmarks with sample limit
    python -m axiom.benchmarks.run --model ./models/my_model --benchmarks math500 aime24 --max-samples 50

    # List available benchmarks
    python -m axiom.benchmarks.run --list
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import BenchmarkReport, BenchmarkProblem, create_report
from .registry import get_loader, get_evaluator, list_benchmarks, get_benchmark_info


def format_prompt_for_benchmark(
    problem: BenchmarkProblem,
    benchmark_type: str
) -> str:
    """
    Format a problem as a prompt for the model.

    Args:
        problem: The benchmark problem
        benchmark_type: Type of benchmark (math, mcq, code)

    Returns:
        Formatted prompt string
    """
    if benchmark_type == "math":
        return (
            f"Solve the following math problem. "
            f"Show your work and put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {problem.question}"
        )
    elif benchmark_type == "multiple_choice":
        return (
            f"Answer the following multiple choice question. "
            f"Explain your reasoning, then state your final answer as a single letter (A, B, C, or D).\n\n"
            f"{problem.question}"
        )
    elif benchmark_type == "code":
        return (
            f"Write a Python function to solve the following problem. "
            f"Include only the code in a ```python block.\n\n"
            f"{problem.question}"
        )
    else:
        return problem.question


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """
    Generate a response from the model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature (0 for greedy)

    Returns:
        Generated text
    """
    import torch

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only new tokens
    input_len = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

    return response


def run_benchmark(
    model,
    tokenizer,
    benchmark_name: str,
    output_dir: Path,
    max_samples: Optional[int] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    verbose: bool = True,
) -> BenchmarkReport:
    """
    Run a single benchmark.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        benchmark_name: Name of benchmark to run
        output_dir: Directory for saving results
        max_samples: Optional limit on number of problems
        temperature: Sampling temperature
        max_new_tokens: Max tokens per response
        verbose: Print progress

    Returns:
        BenchmarkReport with results
    """
    # Load benchmark
    loader = get_loader(benchmark_name)
    problems = loader.load()

    if max_samples and len(problems) > max_samples:
        problems = problems[:max_samples]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running: {benchmark_name}")
        print(f"Problems: {len(problems)}")
        print(f"{'=' * 60}")

    # Get evaluator
    evaluator = get_evaluator(loader.benchmark_type)

    # Run evaluation
    results = []
    start_time = time.time()

    for i, problem in enumerate(problems):
        if verbose:
            print(f"\r  Evaluating: {i + 1}/{len(problems)}", end="", flush=True)

        # Generate prompt
        prompt = format_prompt_for_benchmark(problem, loader.benchmark_type.value)

        # Generate response
        gen_start = time.time()
        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        gen_time = time.time() - gen_start

        # Evaluate
        result = evaluator.evaluate(problem, response, gen_time)
        results.append(result)

    if verbose:
        print()

    # Create report
    total_time = time.time() - start_time
    correct = sum(1 for r in results if r.correct)

    report = create_report(
        benchmark_name=benchmark_name,
        model_name=str(getattr(model, "name_or_path", "unknown")),
        results=results,
        metadata={
            "total_time_seconds": total_time,
            "avg_time_per_problem": total_time / len(problems) if problems else 0,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }
    )

    # Save report
    output_file = output_dir / f"{benchmark_name}_results.json"
    report.save(output_file)

    if verbose:
        print(f"  Result: {report.accuracy:.1%} ({correct}/{report.total})")
        print(f"  Time: {total_time:.1f}s")
        print(f"  Saved: {output_file}")

    return report


def run_all_benchmarks(
    model_path: str,
    benchmark_names: List[str],
    output_dir: Path,
    max_samples: Optional[int] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    verbose: bool = True,
) -> Dict[str, BenchmarkReport]:
    """
    Run multiple benchmarks.

    Args:
        model_path: Path or HuggingFace ID of model
        benchmark_names: List of benchmarks to run
        output_dir: Output directory
        max_samples: Optional sample limit per benchmark
        temperature: Sampling temperature
        max_new_tokens: Max tokens per response
        verbose: Print progress

    Returns:
        Dict mapping benchmark names to reports
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    if verbose:
        print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if verbose:
        print(f"Model loaded on {model.device}")

    # Run benchmarks
    reports = {}
    for name in benchmark_names:
        try:
            report = run_benchmark(
                model, tokenizer, name, output_dir,
                max_samples=max_samples,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            reports[name] = report
        except Exception as e:
            print(f"  Error running {name}: {e}")

    # Save summary
    summary = {
        "model": model_path,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {
            name: {"accuracy": r.accuracy, "correct": r.correct, "total": r.total}
            for name, r in reports.items()
        }
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 60}")
        for name, report in reports.items():
            print(f"  {name}: {report.accuracy:.1%} ({report.correct}/{report.total})")
        print(f"\nSummary saved: {summary_file}")

    return reports


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run axiom-rl benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="Benchmarks to run (or 'all')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Max samples per benchmark (for quick testing)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per problem",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks and exit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # List benchmarks
    if args.list:
        print("Available benchmarks:")
        print("-" * 60)
        info = get_benchmark_info()
        for name, data in info.items():
            print(f"\n  {name}")
            print(f"    Type: {data['type']}")
            print(f"    {data['description']}")
        return

    # Validate args
    if not args.model:
        parser.error("--model is required (or use --list)")

    # Determine which benchmarks to run
    if "all" in args.benchmarks:
        benchmark_names = list_benchmarks()
    else:
        benchmark_names = args.benchmarks

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    run_all_benchmarks(
        model_path=args.model,
        benchmark_names=benchmark_names,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
