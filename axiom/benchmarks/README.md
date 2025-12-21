# Axiom-RL Benchmark Framework

A reusable benchmark framework for evaluating LLMs on standard academic benchmarks.
Run the same benchmarks after each experiment to track progress.

## Quick Start

```bash
# List available benchmarks
python -m axiom.benchmarks.run --list

# Run all benchmarks on a model
python -m axiom.benchmarks.run --model Qwen/Qwen2.5-Coder-1.5B-Instruct --benchmarks all

# Run specific benchmarks
python -m axiom.benchmarks.run --model ./models/my_model --benchmarks math500 gpqa_diamond

# Quick test with limited samples
python -m axiom.benchmarks.run --model <path> --benchmarks math500 --max-samples 50
```

## Available Benchmarks

### Math Benchmarks

#### MATH500
**What it tests:** Mathematical reasoning across 7 subjects
- Algebra, Counting & Probability, Geometry
- Intermediate Algebra, Number Theory
- Prealgebra, Precalculus

**Format:** Open-ended problems with LaTeX `\boxed{}` answers

**Difficulty:** Levels 1-5 (AMC to Olympiad level)

**Size:** 500 problems sampled from MATH dataset

**Scoring:** Exact match after normalization

**Dataset:** `lighteval/MATH` or `hendrycks/competition_math`

---

#### AIME24 / AIME25
**What it tests:** Advanced high school competition mathematics

**Format:** Problems with integer answers (0-999)

**Difficulty:** Very high - only top 2.5% of AMC test-takers qualify for AIME

**Size:** 30 problems per year (15 from AIME I + 15 from AIME II)

**Scoring:** Exact integer match

**Dataset:** `AI-MO/aimo-validation-aime`

**Why it matters:**
- AIME is a prestigious math competition
- Tests deep mathematical reasoning
- Answers are always integers, making evaluation unambiguous

---

### Reasoning Benchmarks

#### GPQA (Graduate-Level Google-Proof Q&A)
**What it tests:** Graduate-level science reasoning
- Physics, Chemistry, Biology
- Written by domain experts (PhDs)
- Designed to be resistant to simple web searches

**Format:** 4-choice multiple choice questions

**Difficulty:** Graduate-level, expert-validated

**Size:** 448 questions

**Scoring:** Accuracy on correct option selection

**Dataset:** `Idavidrein/gpqa` (gpqa_main split)

---

#### GPQA Diamond
**What it tests:** Hardest subset of GPQA

**Format:** Same as GPQA (4-choice MCQ)

**Difficulty:** Highest - additional expert validation

**Size:** 198 questions

**Scoring:** Accuracy on correct option selection

**Dataset:** `Idavidrein/gpqa` (gpqa_diamond split)

**Why Diamond matters:**
- Questions have been validated by multiple experts
- Experts often disagree, indicating genuine difficulty
- Better discriminates between frontier models

---

### Code Benchmarks

#### LiveCodeBench
**What it tests:** Real-world code generation ability
- Problems from actual competitive programming contests
- LeetCode, Codeforces, AtCoder, etc.

**Format:** Problem description + test cases

**Difficulty:** Real competitive programming level

**Scoring:** Pass all test cases (both public and hidden)

**Dataset:** `livecodebench/code_generation`

**Why it matters:**
- Uses real, recent problems (less contamination)
- Requires both understanding and implementation
- Hidden test cases prevent overfitting to examples

---

## Programmatic Usage

```python
from axiom.benchmarks import (
    list_benchmarks,
    get_loader,
    get_evaluator,
    run_benchmark,
    run_all_benchmarks,
)

# List available benchmarks
print(list_benchmarks())
# ['aime24', 'aime25', 'gpqa', 'gpqa_diamond', 'livecodebench', 'math500']

# Load a specific benchmark
loader = get_loader("math500")
problems = loader.load()
print(f"Loaded {len(problems)} problems")

# Get the appropriate evaluator
evaluator = get_evaluator(loader.benchmark_type)

# Evaluate a model output
result = evaluator.evaluate(problems[0], model_output)
print(f"Correct: {result.correct}")

# Run full benchmark programmatically
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("...")
tokenizer = AutoTokenizer.from_pretrained("...")

report = run_benchmark(
    model, tokenizer,
    "math500",
    output_dir=Path("results"),
    max_samples=100,  # Optional limit
)
print(f"Accuracy: {report.accuracy:.1%}")
```

## Output Format

Results are saved as JSON files:

```
benchmark_results/
├── math500_results.json      # Per-problem results
├── gpqa_diamond_results.json
├── ...
└── summary.json              # Combined summary
```

### Result Structure

```json
{
  "benchmark_name": "math500",
  "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "timestamp": "2024-12-20T10:30:00",
  "total": 500,
  "correct": 312,
  "accuracy": 0.624,
  "results": [
    {
      "problem_id": "math500_0",
      "correct": true,
      "model_answer": "42",
      "expected_answer": "42",
      "time_seconds": 2.3
    }
  ],
  "metadata": {
    "total_time_seconds": 1234.5,
    "temperature": 0.0
  }
}
```

## Benchmark Comparison Table

From M-GRPO paper on Qwen3-4B-Base:

| Benchmark | Original | SRT-Best | M-GRPO+IQR |
|-----------|----------|----------|------------|
| MATH500 | 61.50% | 79.20% | **79.75%** |
| AIME24 | 0.83% | 12.50% | **14.58%** |
| AIME25 | 5.00% | 11.67% | **14.17%** |
| GPQA Diamond | 34.41% | 38.26% | **39.65%** |
| GPQA | 29.91% | 35.04% | **35.49%** |
| LiveCode | 9.61% | 19.69% | **27.12%** |

## Adding New Benchmarks

1. Create a loader in `loaders/`:

```python
from ..base import BenchmarkLoader, BenchmarkProblem, BenchmarkType
from ..registry import register_loader

@register_loader("my_benchmark")
class MyBenchmarkLoader(BenchmarkLoader):
    @property
    def name(self) -> str:
        return "my_benchmark"

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.MATH  # or MCQ, CODE

    def load(self, split: str = "test") -> List[BenchmarkProblem]:
        # Load your data
        return problems
```

2. Import it in `loaders/__init__.py`

3. If needed, create a custom evaluator in `evaluators/`

## Tips

- Use `--max-samples 50` for quick validation runs
- Set `--temperature 0` for reproducible greedy decoding
- Results are cached - delete output files to re-run
- Check `summary.json` for quick comparison across benchmarks
