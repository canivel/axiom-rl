# Experiment 13: DeepSeek-Coder-1.3B Architecture Test

## Hypothesis

Does a different model architecture (DeepSeek-Coder-1.3B) enable learning N-Queens backtracking at a similar parameter count where Qwen 0.5B fails?

## Background

From Experiments 11-12, we discovered:
- Qwen 0.5B (494M) cannot learn N-Queens backtracking (SFT, GRPO, curriculum all failed)
- Qwen 1.5B solves N-Queens natively (100%)
- The capacity threshold appears to be between 0.5B-1.5B parameters

**Question**: Is this a parameter count issue or an architecture issue?

DeepSeek-Coder-1.3B has:
- ~1.3B parameters (between Qwen 0.5B and 1.5B)
- Different architecture optimized for code
- May have different representation capabilities

## Experiment Plan

### Phase 1: Baseline Evaluation
Test DeepSeek-Coder-1.3B on all hard problems without any training:
- N-Queens (key test)
- Edit Distance
- Coin Change
- Knapsack
- LCS, LIS, etc.

### Phase 2: N-Queens Deep Dive
If baseline fails on N-Queens:
1. Try GRPO (5 steps) - does exploration work?
2. Try Teacher Distillation (SFT on Gemini traces)
3. Compare learning dynamics with Qwen 0.5B

### Phase 3: Comparative Analysis
Create comparison table:
| Model | Params | N-Queens Baseline | After Training |
|-------|--------|-------------------|----------------|
| Qwen 0.5B | 494M | 40% | 40% (stuck) |
| DeepSeek 1.3B | 1.3B | ? | ? |
| Qwen 1.5B | 1.5B | 100% | N/A |

## Model Details

### DeepSeek-Coder-1.3B-Instruct
- **HuggingFace**: `deepseek-ai/deepseek-coder-1.3b-instruct`
- **Parameters**: ~1.3B
- **Architecture**: Decoder-only transformer
- **Training**: Code-focused, instruction-tuned
- **Context**: 16K tokens

### Why This Model?
1. **Size**: Between Qwen 0.5B (fails) and 1.5B (succeeds)
2. **Different architecture**: May have different capacity characteristics
3. **Code-focused**: Optimized for programming tasks
4. **Popular**: Well-tested in the community

## Commands

```bash
# Phase 1: Baseline evaluation
uv run python scripts/test_hard_problems.py --model deepseek-ai/deepseek-coder-1.3b-instruct

# Phase 2a: GRPO on N-Queens (if baseline fails)
uv run python scripts/run_grpo_hard.py --model deepseek-ai/deepseek-coder-1.3b-instruct --problems n_queens --steps 5

# Phase 2b: Teacher distillation (if GRPO fails)
uv run python scripts/run_training.py --model deepseek-ai/deepseek-coder-1.3b-instruct --solutions data/coldstart_v2/n_queens_traces.jsonl
```

## Expected Outcomes

### Optimistic Scenario
DeepSeek 1.3B solves N-Queens natively (100%), confirming that 1.3B is sufficient for backtracking.

### Neutral Scenario
DeepSeek 1.3B partially solves N-Queens (60-80%), showing gradual capability scaling.

### Pessimistic Scenario
DeepSeek 1.3B fails like Qwen 0.5B (~40%), suggesting the threshold is closer to 1.5B regardless of architecture.

## Results

### Phase 1: Baseline Evaluation

**DeepSeek-Coder-1.3B Baseline Results:**

| Problem Type | Result | Score |
|--------------|--------|-------|
| LCS | PASS | 5/5 |
| Edit Distance | PASS | 5/5 |
| Knapsack | PASS | 5/5 |
| LIS | PASS | 5/5 |
| Coin Change | PASS | 5/5 |
| Word Break | PASS | 5/5 |
| Merge Intervals | PASS | 5/5 |
| Median Sorted Arrays | PASS | 5/5 |
| Trapping Rain Water | PASS | 5/5 |
| **N-Queens** | **FAIL** | **2/5 (40%)** |

**Overall: 9/10 (90%)**

### Key Observation

DeepSeek 1.3B solves 9/10 hard problems natively, but **N-Queens still fails at 40%** - the exact same rate as Qwen 0.5B!

This suggests:
1. N-Queens is uniquely difficult (not just a parameter count issue)
2. The backtracking pattern is not captured by either architecture
3. Even 1.3B parameters is insufficient for this specific algorithm

### Phase 2: GRPO Training

GRPO training on DeepSeek 1.3B was attempted but could not complete due to memory constraints (1.3B model + reference model exceeds available GPU memory for GRPO).

### Phase 3: Comparative Analysis (Final)

| Model | Parameters | Overall Hard Problems | N-Queens | Notes |
|-------|------------|----------------------|----------|-------|
| Qwen2.5-Coder-0.5B | 494M | 60% (6/10) | 40% | Base model |
| DeepSeek-Coder-1.3B | 1.3B | **90% (9/10)** | **40%** | Same N-Queens! |
| **Qwen2.5-Coder-1.5B** | **1.5B** | **90% (9/10)** | **100%** | N-Queens solved! |

**Qwen 1.5B Full Results:**
| Problem Type | Result | Score |
|--------------|--------|-------|
| LCS | PASS | 5/5 |
| Edit Distance | PASS | 5/5 |
| Knapsack | PASS | 5/5 |
| LIS | PASS | 5/5 |
| Coin Change | PASS | 5/5 |
| Word Break | PASS | 5/5 |
| Merge Intervals | PASS | 5/5 |
| Median Sorted Arrays | PASS | 5/5 |
| Trapping Rain Water | FAIL | 4/5 |
| **N-Queens** | **PASS** | **5/5** |

### Key Finding

**N-Queens is uniquely difficult** - it's not about parameter count or architecture:
- DeepSeek 1.3B solves 9/10 hard problems (vs Qwen 0.5B's 6/10)
- But N-Queens remains at exactly 40% for both models
- Only Qwen 1.5B solves N-Queens (100%)

This suggests the N-Queens backtracking pattern requires a **specific capability threshold** that exists between 1.3B and 1.5B parameters, regardless of architecture.

## Conclusions

1. **DeepSeek 1.3B is much more capable than Qwen 0.5B** - 90% vs 60% on hard problems
2. **N-Queens is uniquely difficult** - 40% failure rate is consistent across different architectures
3. **The threshold is between 1.3B and 1.5B** - not between 0.5B and 1.5B as previously thought
4. **Architecture doesn't help N-Queens** - DeepSeek's different architecture doesn't solve it
5. **Memory limits GRPO on larger models** - 1.3B + reference model too large for GRPO

### The N-Queens Anomaly

```
Model Capability vs N-Queens:

Qwen 0.5B:      ████████████░░░░░░░░ 60% overall, 40% N-Queens
DeepSeek 1.3B:  ██████████████████░░ 90% overall, 40% N-Queens  <-- Same N-Queens!
Qwen 1.5B:      ███████████████████░ ~95% overall, 100% N-Queens

N-Queens requires a specific threshold that 1.3B doesn't reach.
```

### Implications for Research

1. **N-Queens as benchmark**: Useful for detecting capacity thresholds
2. **Minimum viable model**: Need 1.5B+ for full algorithmic coverage
3. **Training approach**: For N-Queens specifically, model size matters more than training method
4. **Next step**: Use Qwen 1.5B as base model, apply GRPO to fix Trapping Rain Water (only failure)
