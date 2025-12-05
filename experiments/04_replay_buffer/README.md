# Experiment 04: Self-Improvement with Replay Buffer

**Date:** 2024-12-05
**Status:** Ready to Run

## Executive Summary

This experiment tests whether a **replay buffer** can prevent the catastrophic forgetting discovered in Experiments 02 and 03. The key insight: training on narrow problem sets causes -30% validation degradation. The hypothesis is that mixing new solutions with diverse replay examples will maintain model capability.

---

## Objective

Test if replay buffer prevents catastrophic forgetting during self-improvement training.

## Background

### The Problem (from Experiment 03)

| Setting | Without Replay | Expected with Replay |
|---------|----------------|---------------------|
| Baseline Val | 85% | 85% |
| After Training Val | 55% (-30%) | ~80% (stable) |
| Root Cause | Overfitting to narrow data | Diverse training data |

### Hypothesis

Training with a replay buffer containing solutions from diverse problem types will:
1. Prevent catastrophic forgetting
2. Maintain performance on non-focus problem types
3. Still allow improvement on focus problem types

---

## Methodology

### Replay Buffer Strategy

```
Training Data Composition:
├── 50% New solutions (focus problems: fibonacci, remove_duplicates)
└── 50% Replay solutions (diverse: fizzbuzz, reverse_string, is_palindrome, parentheses, arithmetic)
```

### Key Differences from Experiment 02/03

| Parameter | Exp 02/03 (No Replay) | Exp 04 (With Replay) |
|-----------|----------------------|---------------------|
| Training data | Focus problems only | Focus + Replay mix |
| Problem diversity | 2 types | 7 types |
| Learning rate | 5e-5 | **2e-5** (lower) |
| Replay ratio | 0% | **50%** |

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-Coder-0.5B-Instruct |
| Focus Problems | fibonacci, remove_duplicates |
| Replay Problems | fizzbuzz, reverse_string, is_palindrome, parentheses, arithmetic |
| Samples | Best-of-2 |
| Iterations | 2 |
| Replay Ratio | 50% |
| Learning Rate | 2e-5 |
| Train per type | 5 |

---

## Commands

### Quick Test (Fast Mode)
```bash
uv run python scripts/run_with_replay.py
```

### Custom Configuration
```bash
uv run python scripts/run_with_replay.py \
    --experiment 04_replay_buffer \
    --focus-problems fibonacci remove_duplicates \
    --replay-problems fizzbuzz reverse_string is_palindrome parentheses arithmetic \
    --replay-ratio 0.5 \
    --lr 2e-5 \
    --iterations 2
```

### Compare: With vs Without Replay
```bash
# Without replay (control)
uv run python scripts/fast_validate.py --problems fibonacci remove_duplicates --iterations 2

# With replay (experiment)
uv run python scripts/run_with_replay.py --iterations 2
```

---

## Expected Results

### Success Criteria

| Metric | Without Replay | With Replay (Success) |
|--------|----------------|----------------------|
| Val Change | -30% | **> -5%** |
| Test Change | 0% | **> 0%** |
| Focus Improvement | None | **> 0%** |

### Possible Outcomes

| Outcome | Interpretation | Next Steps |
|---------|---------------|------------|
| Val stable, Focus improves | **SUCCESS** - Replay works | Scale up to 1.5B, more iterations |
| Val stable, Focus same | Partial - Prevents forgetting but no learning | Increase focus training |
| Val still drops | Replay ratio too low | Increase to 70% replay |
| Focus degrades | Replay too dominant | Decrease to 30% replay |

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This documentation |
| `config.json` | Experiment configuration |
| `focus_train.json` | Focus training problems |
| `focus_val.json` | Focus validation problems |
| `focus_test.json` | Focus test problems |
| `replay_train.json` | Replay training problems |
| `replay_buffer.jsonl` | Pre-populated replay buffer |
| `metrics.jsonl` | Per-iteration metrics |
| `summary.json` | Final experiment summary |

---

## Implementation Details

### ReplayBuffer Class

```python
class ReplayBuffer:
    """Manages replay solutions from diverse problem types."""

    def add_solutions(self, solutions: List[Dict])
        """Add verified solutions to the replay buffer."""

    def sample_diverse(self, n_per_type: int) -> List[Dict]
        """Sample solutions ensuring diversity across problem types."""

    def sample_balanced(self, total_samples: int) -> List[Dict]
        """Sample balanced number of solutions across types."""
```

### Training Flow

```
Iteration N:
1. Evaluate on focus val/test sets
2. Collect solutions from focus training problems
3. Sample replay solutions from buffer (50% of training data)
4. Combine: focus_solutions + replay_solutions
5. Shuffle combined dataset
6. Train with LoRA (lower LR: 2e-5)
7. Add focus_solutions to replay buffer for future iterations
8. Repeat
```

---

## Test Log

### Test 1: Initial Run

**Status:** PENDING

**Command:**
```bash
uv run python scripts/run_with_replay.py
```

**Results:**

| Iteration | Val Overall | Test Overall | Change |
|-----------|-------------|--------------|--------|
| 0 (baseline) | TBD | TBD | - |
| 1 (after training) | TBD | TBD | TBD |

---

## References

- [Experiment 02: Focused Improvement](../02_focused_improvement/README.md) - Original forgetting discovery
- [Experiment 03: Fast Validation](../03_fast_validation/README.md) - Confirmed forgetting is fundamental
- [Phase 8 Plan](../../docs/phase8-replay-buffer.md) - Replay buffer design (if exists)
