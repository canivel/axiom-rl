# Experiment 11: Teacher Distillation for Hard Problems

**Status**: ✅ Completed
**Date**: 2025-12-06
**Duration**: ~30 minutes

## Results Summary

| Problem | Baseline | After SFT | Target | Status |
|---------|----------|-----------|--------|--------|
| **Coin Change** | 0% | **100%** | >80% | ✅ EXCEEDED |
| **Knapsack** | 0% | **100%** | >80% | ✅ EXCEEDED |
| **N-Queens** | 40% | 40% | >80% | ❌ Needs more data |

**Key Achievement**: Teacher distillation (Gemini → SFT) solved 2 of 3 hard problems that GRPO alone couldn't solve.

## Objective

Use Gemini as a teacher model to generate high-quality reasoning traces for hard problems that GRPO alone couldn't solve:

| Problem | GRPO-Only Result | Reason for Failure |
|---------|------------------|-------------------|
| Coin Change | 0% | Different DP structure (1D with coin iteration, no transfer from 2D LCS) |
| Knapsack | 0% | Similar to Coin Change (1D DP with item iteration) |
| N-Queens | 40% | Backtracking logic - needs clearer constraint checking |

## Hypothesis

The key insight from Experiment 10 was that **Edit Distance succeeded because it could transfer from LCS**. The model already "knew" 2D DP patterns, and GRPO helped it generalize to the slightly different recurrence.

For problems without transfer targets, we need to **bootstrap the model with correct solutions first** via teacher distillation, then refine with GRPO.

## Methodology

### Three-Stage Pipeline

```
Stage 1: Claude Distillation     Stage 2: SFT Training    Stage 3: GRPO Refinement
┌─────────────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│ Generate Claude traces  │ ──> │ Train on traces  │ ──> │ Refine with execution │
│ with <think> reasoning  │     │ (LoRA SFT)       │     │ rewards               │
└─────────────────────────┘     └──────────────────┘     └───────────────────────┘
```

### Stage 1: Claude Trace Generation

Claude generates solutions with detailed reasoning:

```python
# Prompt format
<think>
Step 1: Understand the problem - Coin Change is finding minimum coins
Step 2: This is 1D DP where dp[i] = min coins to make amount i
Step 3: For each amount, try each coin and take minimum
Step 4: Time: O(amount * len(coins)), Space: O(amount)
Step 5: Edge case: return -1 if amount can't be made
</think>

```python
def coin_change(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### Stage 2: SFT on Verified Traces

Train the small model to mimic Claude's solutions:

```python
LoRASFTConfig(
    model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    learning_rate=2e-4,
)
```

Only verified traces (100% test pass) are used for training.

### Stage 3: GRPO Refinement

Fine-tune the SFT model with execution-based RL:

```python
GRPOConfig(
    model_name="models/hard-distill/sft",  # SFT checkpoint
    num_generations=2,
    learning_rate=2e-5,
    beta=0.04,
)
```

This stage corrects any minor errors and improves robustness.

## Configuration

### Trace Generation

```bash
uv run python scripts/generate_hard_traces.py \
    --problems coin_change knapsack n_queens \
    --count 3 \
    --difficulties 3 5 7 \
    --output data/coldstart_v2/hard_traces.jsonl
```

Expected output: ~27 traces (3 problems × 3 difficulties × 3 per difficulty)

### SFT Training

```bash
uv run python scripts/run_training.py \
    --solutions data/coldstart_v2/hard_traces.jsonl \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --epochs 3 \
    --output-dir models/hard-distill/sft
```

### GRPO Refinement

```bash
uv run python scripts/run_grpo_hard.py \
    --model models/hard-distill/sft \
    --problems coin_change knapsack n_queens \
    --steps 10 \
    --output-dir models/hard-distill/grpo
```

### Full Pipeline

```bash
# Run entire pipeline
uv run python scripts/run_hard_distillation.py

# With custom options
uv run python scripts/run_hard_distillation.py \
    --problems coin_change knapsack \
    --trace-count 5 \
    --grpo-steps 15
```

## Expected Results

### Success Criteria

| Problem | Before (GRPO-only) | Target |
|---------|-------------------|--------|
| Coin Change | 0% | >80% |
| Knapsack | 0% | >80% |
| N-Queens | 40% | >80% |

### Comparison with GRPO-Only

The hypothesis is that teacher distillation + GRPO will succeed where GRPO-only failed because:

1. **Bootstrap effect**: Claude provides correct algorithmic patterns
2. **Verified training**: Only 100% correct solutions used for SFT
3. **Refinement**: GRPO corrects edge cases and improves robustness

## Files Created

```
scripts/generate_hard_traces.py      # Claude trace generation
scripts/run_hard_distillation.py     # Full pipeline script
experiments/11_teacher_distillation_hard/
├── README.md                        # This documentation
├── config.json                      # Configuration used
├── traces/                          # Generated traces
│   └── hard_traces.jsonl
└── results/                         # Evaluation results
    └── metrics.json
```

## Reproduction

### Prerequisites

1. Set `ANTHROPIC_API_KEY` in `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. Ensure base model downloaded:
   ```bash
   uv run python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B-Instruct')"
   ```

### Quick Start

```bash
# Generate traces only (requires API key)
uv run python scripts/generate_hard_traces.py --problems coin_change --count 2

# Full pipeline
uv run python scripts/run_hard_distillation.py

# Skip trace generation (use existing)
uv run python scripts/run_hard_distillation.py --skip-traces
```

### Evaluation

```bash
# Test final model
uv run python scripts/test_hard_problems.py \
    --model models/hard-distill/grpo/final_model \
    --problems coin_change knapsack n_queens

# Compare with baseline
uv run python scripts/test_hard_problems.py \
    --problems coin_change knapsack n_queens
```

## Relationship to Other Experiments

```
Exp 09: Hard Problems Baseline
         │
         v
Exp 10: GRPO Hard Problems ──────> SUCCESS: Edit Distance 0%→100%
         │                         FAIL: Coin Change, Knapsack, N-Queens
         v
Exp 11: Teacher Distillation ────> This experiment
         │                         Target: Solve remaining problems
         v
Exp 12: Full Evaluation (future)
```

## Theoretical Background

### Why Teacher Distillation Works

1. **Knowledge Transfer**: Claude's reasoning traces contain algorithmic patterns the small model hasn't seen during pretraining

2. **Structured Learning**: The `<think>` tags provide step-by-step reasoning that the model can learn to mimic

3. **Verified Signal**: Unlike preference-based RLHF, our traces are verified by execution - no reward hacking possible

### Why GRPO Alone Failed on Coin Change

The key insight is about **representation distance**:

```
Edit Distance (success):
  Model knows: LCS (2D DP with similar structure)
  Delta to learn: Replace max() with min(), add +1

Coin Change (failure):
  Model knows: ???
  Delta to learn: Entire 1D DP with nested coin loop

The representation distance is too large for GRPO exploration.
```

Teacher distillation bridges this gap by providing the target representation directly.

## Actual Results

### Stage 1: Trace Generation

Used Gemini 2.5 Flash (due to rate limits on Claude API):

```bash
uv run python scripts/generate_hard_traces.py \
    --teacher gemini \
    --problems coin_change knapsack n_queens \
    --count 3 \
    --difficulties 3 5 7 \
    --output data/coldstart_v2/hard_traces_full.jsonl
```

**Results:**
- Total attempts: 18 (rate-limited from 27)
- Verified: **18 (100%)**
- Coin Change: 9/9 verified
- Knapsack: 8/8 verified
- N-Queens: 1/1 verified (limited by rate limits)

### Stage 2: SFT Training

```bash
uv run python scripts/run_training.py \
    --solutions data/coldstart_v2/hard_traces_full.jsonl \
    --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --epochs 3 \
    --output-dir models/hard-distill/sft
```

**Results:**
- Training samples: 16 (train) + 2 (val)
- Trainable params: 8.8M (1.75% of 502M)
- Final train loss: 1.1953
- Training time: ~29 seconds

### Stage 3: Evaluation

**Baseline Model (Qwen 0.5B):**
```
knapsack      FAIL   0/5  (Exec error: name 'N' is not defined)
coin_change   FAIL   0/5  (Expected 5, got 3)
n_queens      FAIL   2/5  (Expected 2, got 1)
OVERALL: 0/3 (0.0%)
```

**After SFT Distillation:**
```
knapsack      PASS   5/5  ✅
coin_change   PASS   5/5  ✅
n_queens      FAIL   2/5
OVERALL: 2/3 (66.7%)
```

### Analysis

| Problem | Baseline | After SFT | Improvement | Analysis |
|---------|----------|-----------|-------------|----------|
| Coin Change | 0% | 100% | **+100%** | Teacher traces bootstrapped 1D DP pattern |
| Knapsack | 0% | 100% | **+100%** | Similar to Coin Change - 1D DP learned |
| N-Queens | 40% | 40% | 0% | Only 1 trace generated due to rate limits |

**Key Insight**: The success of Coin Change and Knapsack validates the hypothesis. N-Queens didn't improve because we only had 1 training example (9 for the others).

### Why N-Queens Didn't Improve

The rate-limiting meant we only generated 1 N-Queens trace vs 8-9 for the other problems:
- Coin Change: 9 traces → 100% accuracy
- Knapsack: 8 traces → 100% accuracy
- N-Queens: 1 trace → 40% accuracy (unchanged)

**Prediction**: With 8+ N-Queens traces, we would see similar improvement.

## Next Steps

Based on these results:

1. **Generate more N-Queens traces** - Need 8+ to match other problems
2. **Try GRPO refinement** - May help with edge cases
3. **Apply to graph/tree problems** - Same technique should work
4. **Build continuous improvement pipeline** - Automate the trace → SFT → GRPO loop
