# Axiom-RL Experiments

This directory contains all experimental results for the axiom-rl self-improvement research.

## Experiment Index

| ID | Name | Status | Key Finding |
|----|------|--------|-------------|
| 01 | [Baseline Evaluation](01_baseline_evaluation/) | Completed | 85% overall accuracy, 2 weak problems identified |
| 02 | [Focused Improvement](02_focused_improvement/) | Completed | **Catastrophic forgetting**: accuracy degraded -19% due to overfitting |
| 03 | [Fast Validation](03_fast_validation/) | Completed | Confirmed forgetting is fundamental (-30%), Best-of-4 optimal |
| 04 | [Replay Buffer](04_replay_buffer/) | ✅ Completed | **SUCCESS**: Replay buffer prevents forgetting (+10% val vs -30% without) |
| 05 | [Enhanced Distillation](05_enhanced_distillation/) | ✅ Completed | Claude 100% vs Gemini 75.8% verification, 60 traces generated |
| 06 | [SFT Baseline V2](06_sft_baseline_v2/) | ✅ Completed | **76.4% accuracy** on V2 problems, 6 problem types at 100% |
| 07 | [GRPO Validation](07_grpo_validation/) | ✅ Completed | Training loop functional, ~5 min/step on RTX 3080, memory-stable |
| 08 | [Curriculum Learning](08_curriculum_learning/) | ✅ Completed | Framework validated, adaptive strategy working |
| 09 | [Hard Problems Baseline](09_hard_problems_baseline/) | ✅ Completed | **60% accuracy** on LeetCode-hard, 4 weak problem types identified |
| 10 | [GRPO Hard Problems](10_grpo_hard_problems/) | ✅ Completed | **Edit Distance: 0% → 100%** via GRPO, Coin Change needs more work |
| 11 | Teacher Distillation Hard | 🔄 Next | Generate Claude traces for Coin Change, Knapsack |

## Methodology

### Evaluation Protocol: Best-of-N

All experiments use **Best-of-N** (N=8) evaluation:
- Generate 8 independent solutions per problem (temperature=0.7)
- Count how many pass ALL test cases
- Score = passed / total samples

This methodology:
1. Accounts for sampling variance
2. Measures model reliability
3. Identifies problems the model struggles with

### Self-Improvement Loop (Expert Iteration)

Each iteration:
1. **Evaluate** - Test current model on train/val/test sets
2. **Collect** - Gather verified correct solutions
3. **Train** - Fine-tune with LoRA on correct solutions
4. **Repeat** - Use improved model for next iteration

### Success Criteria

- **Accuracy improvement** - Training accuracy should increase over iterations
- **Generalization** - Validation accuracy should improve, not just training
- **No degradation** - V1 showed degradation due to memorization; V2 fixes this

## Model Information

| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Training Method | LoRA (rank=16, alpha=32) |
| Trainable Params | ~18M (1.18% of total) |
| Hardware | Single NVIDIA GPU (12GB+ VRAM) |

## Key Discoveries

### V1 vs V2 Problem Design

**V1 (Flawed):**
- Single test case per problem
- Functions with no input arguments
- Allowed memorization instead of learning
- Training accuracy **degraded** over iterations

**V2 (Corrected):**
- Multiple test cases (5+) per problem
- Functions take input arguments
- Requires implementing actual algorithms
- Training accuracy **improves** over iterations

See [Phase 7 Documentation](../docs/phase7-v2-problem-design.md) for details.

## Running Experiments

### Fast Hypothesis Validation (~30 min)

For rapid iteration on hypotheses, use the fast validation script:

```bash
# Quick validation (30 min) - 0.5B model, Best-of-2, 1 iteration
uv run python scripts/fast_validate.py

# Medium validation (2 hours) - 1.5B model, Best-of-4, 1 iteration
uv run python scripts/fast_validate.py --medium

# Custom fast validation
uv run python scripts/fast_validate.py --problems arithmetic rpn --samples 4
```

| Mode | Model | Samples | Iterations | Time |
|------|-------|---------|------------|------|
| Fast | 0.5B | 2 | 1 | ~30 min |
| Medium | 1.5B | 4 | 1 | ~2 hours |
| Full | 1.5B | 8 | 3 | ~12 hours |

### Replay Buffer Experiment

```bash
# Experiment 04: Self-improvement with replay buffer
uv run python scripts/run_with_replay.py

# Custom replay configuration
uv run python scripts/run_with_replay.py \
    --focus-problems fibonacci remove_duplicates \
    --replay-problems fizzbuzz reverse_string is_palindrome \
    --replay-ratio 0.5 \
    --lr 2e-5 \
    --iterations 2
```

### GRPO Hard Problems Training

```bash
# Train on weak problem types (Edit Distance, Coin Change)
uv run python scripts/run_grpo_hard.py \
    --problems edit_distance coin_change \
    --steps 5 \
    --difficulty 5

# Train on all weak problems
uv run python scripts/run_grpo_hard.py \
    --problems edit_distance knapsack coin_change n_queens \
    --steps 10

# Test trained model on hard problems
uv run python scripts/test_hard_problems.py \
    --model models/grpo-hard/final_model \
    --difficulty 5
```

### Full Experiments

```bash
# Experiment 01: Baseline evaluation
# (Already completed, see results)

# Experiment 02: Focused improvement (full mode)
uv run python scripts/run_focused_improvement.py \
    --experiment 02_focused_improvement \
    --train-per-type 25 \
    --iterations 3

# General V2 experiments
uv run python scripts/run_self_improve_v2.py \
    --experiment my_experiment \
    --problem-types rpn parentheses \
    --train-per-type 20 \
    --iterations 5
```

## Directory Structure

Each experiment follows this structure:

```
experiments/{experiment_name}/
├── README.md           # Experiment documentation
├── config.json         # Configuration used
├── train.json          # Training problems
├── val.json            # Validation problems
├── test.json           # Test problems
├── metrics.jsonl       # Per-iteration metrics
├── summary.json        # Final summary
├── solutions/          # Verified solutions per iteration
│   ├── iter_0.jsonl
│   ├── iter_1.jsonl
│   └── ...
└── models/             # Model checkpoints (if saved)
    ├── iter_0/
    └── ...
```
