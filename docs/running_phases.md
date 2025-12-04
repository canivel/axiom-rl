# Running Axiom-RL: Step-by-Step Guide

This guide walks you through running the complete V2 self-improvement pipeline.

## Prerequisites

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU (12GB+ VRAM recommended)

## Quick Start

```bash
# 1. Install dependencies
uv venv && uv pip install -e .

# 2. Test generators work
uv run python scripts/test_v2_generators.py

# 3. Run V2 experiment
uv run python scripts/run_self_improve_v2.py --experiment my_test --iterations 2
```

---

## Step 1: Verify Installation

```bash
# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Test imports
uv run python -c "from axiom.procedural import get_generator_v2; print('OK')"
```

---

## Step 2: Test Problem Generators

```bash
uv run python scripts/test_v2_generators.py
```

**What this validates:**
- Problems have multiple test cases (5+)
- Functions take input arguments
- Hardcoding cannot pass all tests
- Correct algorithms pass all tests

---

## Step 3: Run V2 Experiment

### Quick Test (2 iterations)

```bash
uv run python scripts/run_self_improve_v2.py \
    --experiment v2_quick \
    --train-per-type 5 \
    --iterations 2
```

### Full Experiment (10 iterations)

```bash
uv run python scripts/run_self_improve_v2.py \
    --experiment v2_full \
    --train-per-type 20 \
    --iterations 10
```

---

## Step 4: Analyze Results

```bash
# View metrics
type experiments\v2_quick\metrics.jsonl

# View collected solutions
type experiments\v2_quick\solutions\iter_0.jsonl
```

### Expected Results

```
Iteration   Train    Val      Test
-----------------------------------------
0           70.0%    83.3%    100.0%
1           90.0%    100.0%   83.3%
-----------------------------------------
Change:     +20.0%   +16.7%   -16.7%
```

**Key:** Training accuracy should IMPROVE over iterations (not degrade like V1).

---

## CLI Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--experiment` | `v2_experiment` | Experiment name |
| `--model` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Base model |
| `--problem-types` | `rpn parentheses` | Problem types |
| `--train-per-type` | `10` | Training problems per type |
| `--test-cases` | `5` | Test cases per problem |
| `--iterations` | `5` | Self-improvement iterations |
| `--lr` | `5e-5` | Learning rate |

---

## Interactive Notebook

For hands-on analysis, use the Colab notebook:

**[notebooks/axiom_v2_step_by_step.ipynb](../notebooks/axiom_v2_step_by_step.ipynb)**

11 parts with isolated cells and detailed outputs.

---

## Troubleshooting

### "CUDA out of memory"
Use smaller model: `--model Qwen/Qwen2.5-Coder-0.5B-Instruct`

### "0 solutions collected"
Reduce difficulty: `--test-cases 3`

### "CONFIG not defined" (notebook)
Run cell 1.3 first

---

## Documentation

- [Phase 7: V2 Problem Design](phase7-v2-problem-design.md) - Full technical details
- [Critical Analysis](critical-analysis-problem-design.md) - Why V1 failed
