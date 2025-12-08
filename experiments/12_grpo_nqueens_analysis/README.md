# Experiment 12: GRPO Analysis for N-Queens

## Hypothesis

Can GRPO (Group Relative Policy Optimization) help the 0.5B model learn N-Queens through reinforcement learning, even though supervised fine-tuning failed?

## Background

From Experiment 11, we discovered:
- Teacher distillation works for Coin Change (0% -> 100%) and Knapsack (0% -> 100%)
- N-Queens remains stuck at 40% despite 16 high-quality synthetic traces
- Qwen 1.5B solves N-Queens natively (100%), confirming model capacity limit

The question: Can GRPO provide learning signal where SFT failed?

## Experiment A: Standard GRPO (5 Steps)

### Configuration
```
Model: Qwen/Qwen2.5-Coder-0.5B-Instruct
Problems: n_queens
Steps: 5
Generations per step: 4
Learning rate: 5e-5
Difficulty: 4 (n in {1, 4, 5, 6})
```

### Results

| Step | Avg Reward | Max Reward | Model Outputs |
|------|------------|------------|---------------|
| 1 | 0.05 | 0.20 | 0, 36, 2258753, 1, 7, 2113 |
| 2 | 0.00 | 0.00 | 24, 1, 40320 |
| 3 | 0.225 | 0.30 | 1, 7, 2113, 1, 0, 0 |
| 4 | 0.00 | 0.00 | (no correct) |
| 5 | 0.025 | 0.10 | 1, 24, 60 |

**Final Evaluation: 40%** (unchanged from baseline)

### Analysis

The model outputs reveal critical failure patterns:

1. **Wildly incorrect values**: 2258753, 40320 (8!), 2113
   - Correct answers for n=4,5,6 are 2, 10, 4
   - Model is not learning backtracking algorithm

2. **No exploration success**:
   - Max reward never exceeds 0.30 (partial credit)
   - Model never generates a fully correct solution

3. **Random guessing patterns**:
   - Some outputs look like factorials (40320 = 8!)
   - Others seem random (2258753)
   - Model doesn't understand constraint satisfaction

### Why GRPO Fails for N-Queens

GRPO requires the model to occasionally generate correct solutions through exploration, which it can then reinforce. For N-Queens:

1. **Solution space is sparse**: Correct N-Queens solutions require precise backtracking
2. **No nearby transfer target**: Unlike Edit Distance (which has LCS), N-Queens has no related problem the model already knows
3. **Random exploration doesn't work**: The probability of randomly generating correct backtracking code is effectively zero
4. **Partial rewards insufficient**: Getting 1-2 test cases right doesn't teach the algorithm

## Experiment B: Curriculum Learning

### Hypothesis

Starting with trivial cases (n=1,2,3,4) where answers are simple (1,0,0,2) might allow the model to discover correct solutions.

### Configuration
```
uv run python scripts/run_grpo_nqueens_curriculum.py \
    --steps-per-level 5 \
    --start-difficulty 1 \
    --max-difficulty 3 \
    --output-dir models/grpo-curriculum
```

Difficulty mapping:
- Level 1-3: n in {1,2,3,4} - Trivial (n=4 has only 2 solutions)
- Level 4-6: n in {1,4,5,6} - Medium
- Level 7+:  n in {4,5,6,7,8} - Hard

### Results (Full 27-Step Run)

**Level 1 (n=1,2,3,4) - Three Attempts:**

| Attempt | Steps | Avg Reward | Max Reward | Result |
|---------|-------|------------|------------|--------|
| 1 | 1-3 | 0.217 | 1.0 | Failed promotion |
| 2 | 4-6 | 0.017 | 0.1 | Failed promotion |
| 3 | 7-9 | 0.050 | 0.3 | Forced promotion |

**Level 2 (n=1,2,3,4) - Three Attempts:**

| Attempt | Steps | Avg Reward | Max Reward | Result |
|---------|-------|------------|------------|--------|
| 1 | 10-12 | 0.167 | 1.0 | Failed promotion |
| 2 | 13-15 | 0.000 | 0.0 | Failed promotion |
| 3 | 16-18 | 0.000 | 0.0 | Forced promotion |

**Level 3 (n=1,2,3,4) - Three Attempts:**

| Attempt | Steps | Avg Reward | Max Reward | Result |
|---------|-------|------------|------------|--------|
| 1 | 19-21 | 0.000 | 0.0 | Failed promotion |
| 2 | 22-24 | 0.000 | 0.0 | Failed promotion |
| 3 | 25-27 | 0.000 | 0.0 | Forced promotion |

**Final Evaluation:**
```
Difficulty 3 (n=[2,2,4,2,3]): FAIL (0%)
Difficulty 5 (n=[6,4,4,5,6]): FAIL (0%)
Difficulty 7 (n=[6,7,8,7,4]): FAIL (0%)
```

### Analysis

The curriculum experiment reveals a **catastrophic degradation pattern**:

1. **Initial lucky hits disappear**: Early steps occasionally hit Max=1.0, but this stops completely
2. **Rewards collapse to zero**: By Level 2-3, average reward is consistently 0.0
3. **Loss explodes**: Loss increases from -0.12 to -6.57 (model confidence in wrong answers)
4. **Final evaluation: 0% across all difficulties**

The model is **actively getting worse** - GRPO is reinforcing incorrect patterns because:
- Random exploration never finds correct backtracking solutions
- Partial rewards for wrong answers teach wrong patterns
- Without positive signal, the model drifts further from correct behavior

**Conclusion**: Curriculum learning does not help. The model fundamentally cannot represent N-Queens, and extended training makes it worse.

## Model Capacity Analysis

### The Specific Model Tested

| Property | Value |
|----------|-------|
| **Model** | Qwen/Qwen2.5-Coder-0.5B-Instruct |
| **Parameters** | 494M total |
| **Architecture** | Transformer decoder-only |
| **Context Length** | 32K tokens |
| **Vocabulary** | 151,936 tokens |
| **Hidden Size** | 896 |
| **Layers** | 24 |
| **Attention Heads** | 14 |
| **Training** | Code-focused instruction tuning |

### What 0.5B Cannot Do

The model fails at N-Queens because backtracking requires:

1. **Recursive state management**: Tracking which columns/diagonals are occupied
2. **Constraint propagation**: Understanding that placing a queen affects future placements
3. **Search tree exploration**: Mentally simulating different placement orders

The 0.5B model:
- Generates syntactically valid Python
- Understands function signatures
- Can do simple DP (Coin Change, Knapsack after SFT)
- **Cannot represent recursive constraint satisfaction**

### Comparison: What Works vs What Doesn't

| Algorithm Type | 0.5B Performance | Why |
|----------------|------------------|-----|
| Dynamic Programming | Learnable via SFT | Tabular, no recursion needed |
| Edit Distance | Learnable via GRPO | Transfer from LCS |
| Simple Recursion | Works | Fibonacci, factorial |
| Backtracking | **FAILS** | Requires constraint tracking |
| Graph Algorithms | Unknown | Needs testing |

### Alternative Small Models to Research

| Model | Size | Architecture | Notes |
|-------|------|--------------|-------|
| **Qwen2.5-Coder-1.5B** | 1.5B | Same family | Solves N-Queens natively (100%) |
| **DeepSeek-Coder-1.3B** | 1.3B | Decoder | Strong code model, may work |
| **StarCoder2-3B** | 3B | Decoder | Code-focused, larger |
| **Phi-2** | 2.7B | Decoder | Microsoft, efficient |
| **CodeLlama-7B** | 7B | Llama | Meta, proven code ability |

### The Capacity Threshold

```
         N-Queens Capability
              |
    100% ─────┼─────────────────────────● Qwen 1.5B
              |                        /
              |                       /
              |                      /
     40% ─────┼─────● Qwen 0.5B ────/
              |     (stuck)
              |
      0% ─────┼────────────────────────────────
              |
              0.5B    1.0B    1.5B    2.0B
                    Model Size
```

The threshold appears to be between **0.5B and 1.5B parameters** for backtracking algorithms.

## Key Insights

### Why Some Problems Respond to GRPO and Others Don't

| Problem | GRPO Works? | Reason |
|---------|-------------|--------|
| Edit Distance | Yes (0% -> 100%) | LCS is nearby transfer target |
| Coin Change | No (used SFT) | DP pattern learnable from examples |
| Knapsack | No (used SFT) | DP pattern learnable from examples |
| N-Queens | **No** | No exploration success, no transfer target |

### The Exploration Problem

GRPO learning requires: `P(correct solution | random exploration) > 0`

For N-Queens with the 0.5B model:
- Model generates syntactically valid Python
- But semantically, it doesn't understand backtracking
- Random code generation never produces correct constraint satisfaction

## Conclusions

1. **GRPO cannot help 0.5B learn N-Queens** - confirmed (5 steps, 40% unchanged)
2. **Curriculum learning fails catastrophically** - 27 steps, final evaluation 0% (worse than baseline!)
3. **Extended training is harmful**: Loss explodes (-0.12 → -6.57), model gets more confident in wrong answers
4. **The failure is fundamental**: Model architecture cannot represent backtracking
5. **Model capacity is the bottleneck**: Need 1.5B+ parameters for N-Queens
6. **Key insight**: Without exploration success, GRPO reinforces wrong patterns

### The Degradation Problem

```
Performance over curriculum training:
Step  1-3:   Max=1.0 (lucky hits)
Step  4-9:   Max=0.3 (declining)
Step 10-18:  Max=0.0-1.0 (unstable)
Step 19-27:  Max=0.0 (complete collapse)
Final:       0% on ALL difficulties (worse than 40% baseline!)
```

This demonstrates that **GRPO without exploration success is actively harmful** - it teaches the model to be confident in incorrect solutions.

## Research Recommendations

To solve N-Queens with a small model, options are:

1. **Use 1.5B model**: Qwen2.5-Coder-1.5B solves it natively
2. **Try different architectures**: DeepSeek, Phi-2 may have different capabilities
3. **Problem decomposition**: Break N-Queens into sub-problems the 0.5B can solve
4. **Symbolic scaffolding**: Provide helper functions that handle constraint tracking

## Files

- `scripts/run_grpo_hard.py` - Standard GRPO training script
- `scripts/run_grpo_nqueens_curriculum.py` - Curriculum learning script
- `models/grpo-nqueens-test/` - Model from Experiment A
