# Experiment 15: M-GRPO with Entropy Control

**Status:** In Progress (First Run)
**Date:** 2024-12-20
**Branch:** v2-problem-design
**Hardware:** RTX 3080 10GB (Local)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Core Problem We're Solving](#the-core-problem-were-solving)
3. [What is GRPO and Why Does It Fail?](#what-is-grpo-and-why-does-it-fail)
4. [M-GRPO: The Solution](#m-grpo-the-solution)
5. [What We're Teaching the Model](#what-were-teaching-the-model)
6. [The Training Pipeline](#the-training-pipeline)
7. [Bugs We Found and Fixed](#bugs-we-found-and-fixed)
8. [Live Training Results](#live-training-results)
9. [How to Interpret the Metrics](#how-to-interpret-the-metrics)
10. [Running the Experiment](#running-the-experiment)
11. [Technical Details](#technical-details)
12. [References](#references)

---

## Executive Summary

This experiment implements **M-GRPO (Momentum-Anchored GRPO)** - a reinforcement learning technique for training language models to solve coding problems. We're training a small 0.5B parameter model to write Python functions that pass test cases.

**Key Innovation:** Instead of training one model, we maintain two:
- A **policy model** that learns and improves
- A **momentum model** that evolves slowly and provides stability

This prevents the common failure mode where the model becomes overconfident and "forgets" how to generate diverse solutions.

---

## The Core Problem We're Solving

### The Challenge: Teaching AI to Code via Self-Improvement

We want to train a language model to write correct Python code **without** human-labeled examples. The model should:

1. Read a problem description (e.g., "Write a function to compute Fibonacci numbers")
2. Generate Python code that solves the problem
3. Have that code **verified** by running it against test cases
4. Learn from its successes and failures

This is called **self-supervised reinforcement learning** because:
- **Self-supervised**: No human labels - we verify correctness by running the code
- **Reinforcement learning**: The model gets "rewards" for correct solutions and learns to maximize them

### Why Is This Hard?

Traditional supervised learning requires thousands of (problem, solution) pairs curated by humans. But:
- Creating high-quality coding examples is expensive
- Models can memorize solutions instead of learning to reason
- Doesn't scale to novel problem types

RL lets the model discover solutions on its own, but it's notoriously unstable for language models.

---

## What is GRPO and Why Does It Fail?

### GRPO: Group Relative Policy Optimization

GRPO (from DeepSeekMath) is a technique for training LLMs with RL:

1. **Generate multiple solutions** for each problem (e.g., 8 samples)
2. **Verify each solution** against test cases
3. **Compute advantages**: Solutions better than the group average get positive advantage
4. **Update the policy**: Increase probability of high-advantage solutions

```
Advantage(solution) = (reward - mean_reward) / std_reward
```

### The Failure Mode: Policy Collapse

In practice, GRPO often fails catastrophically:

```
Step 1:  reward=0.3, entropy=0.8  ✓ Learning
Step 5:  reward=0.6, entropy=0.5  ✓ Improving
Step 10: reward=0.8, entropy=0.2  ⚠️ Getting confident
Step 15: reward=0.4, entropy=0.05 ❌ COLLAPSED - model stuck on one solution
Step 20: reward=0.2, entropy=0.01 ❌ Completely broken
```

**What happened?**

1. Model finds a solution that works for some problems
2. Aggressively increases probability of that solution pattern
3. Entropy (diversity) collapses - model only generates one type of answer
4. When that pattern fails on new problems, model can't recover
5. Performance crashes

This is called **mode collapse** or **policy collapse**.

---

## M-GRPO: The Solution

### Key Idea: Use a Slow-Moving "Anchor" Model

M-GRPO introduces a **momentum model** that acts as an anchor:

```
┌─────────────────┐     ┌─────────────────┐
│  Policy Model   │     │ Momentum Model  │
│  (θ_q - fast)   │     │ (θ_k - slow)    │
│                 │     │                 │
│ Gets gradients  │     │ EMA of policy   │
│ Changes quickly │     │ Changes slowly  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
    Generate 4              Generate 4
    samples each            samples each
         │                       │
         └───────────┬───────────┘
                     ▼
              8 total samples
                     │
                     ▼
         ┌───────────────────────┐
         │   Verify & Compute    │
         │   Pseudo-Ground Truth │
         └───────────────────────┘
```

### The Algorithm

```python
for each training step:
    # 1. Generate from both models
    policy_samples = policy_model.generate(prompts, n=4)
    momentum_samples = momentum_model.generate(prompts, n=4)

    # 2. Combine and verify all samples
    all_samples = policy_samples + momentum_samples  # 8 total
    rewards = verify_all(all_samples)  # Run against test cases

    # 3. Find best answer (pseudo-ground truth)
    best_answer = majority_vote(successful_samples)

    # 4. Train policy on its own samples
    for sample, reward in zip(policy_samples, policy_rewards):
        advantage = (reward - mean) / std
        loss = -log_prob(sample) * advantage
        policy_model.update(loss)

    # 5. Slowly update momentum model (EMA)
    momentum_model = 0.99 * momentum_model + 0.01 * policy_model
```

### Why This Works

1. **Momentum provides stability**: Even if policy starts failing, momentum model still generates good samples
2. **Pseudo-ground truth**: Best answer from combined pool is more reliable
3. **Slow evolution**: Momentum can't collapse quickly because it changes slowly
4. **Recovery**: If policy degrades, momentum pulls it back toward working solutions

### IQR Filtering for Entropy

We also filter out low-entropy (overconfident) samples:

```python
# Compute entropy of each trajectory
entropies = [compute_entropy(sample) for sample in all_samples]

# IQR-based threshold
Q1, Q3 = percentile(entropies, [25, 75])
threshold = Q1 - 0.75 * (Q3 - Q1)

# Remove low-entropy samples
filtered_samples = [s for s, e in zip(samples, entropies) if e > threshold]
```

This prevents the model from training on overconfident outputs.

---

## What We're Teaching the Model

### The Task: Algorithmic Problem Solving

We generate **procedural coding problems** across 6 categories:

| Problem Type | Description | Example |
|-------------|-------------|---------|
| **RPN** | Evaluate Reverse Polish Notation | `["3", "4", "+"]` → `7` |
| **Parentheses** | Validate balanced brackets | `"(())"` → `True` |
| **Fibonacci** | Compute nth Fibonacci number | `fib(10)` → `55` |
| **Binary Search** | Find element in sorted array | `search([1,3,5], 3)` → `1` |
| **Edit Distance** | Levenshtein distance between strings | `dist("cat", "car")` → `1` |
| **Coin Change** | Minimum coins for amount | `coins([1,5,10], 11)` → `2` |

### Problem Format

Each problem includes:
- **Description**: What the function should do
- **Function signature**: `def fibonacci(n: int) -> int:`
- **Examples**: Input/output pairs
- **Test cases**: 5 hidden tests for verification

Example prompt:
```
## Problem: Fibonacci Number

Implement a function to compute the nth Fibonacci number.

The Fibonacci sequence is: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

## Function Signature
def fibonacci(n: int) -> int:
    # Your implementation here

## Examples
  fibonacci(1) -> 1
  fibonacci(3) -> 2
  fibonacci(0) -> 0

Write ONLY the complete function implementation.
```

### What the Model Outputs

The model generates a response like:
```markdown
## Solution

To compute Fibonacci numbers efficiently, we use iteration:

```python
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```
```

### The Reward Signal

We extract the code and run it against test cases:

| Test Case | Input | Expected | Actual | Pass? |
|-----------|-------|----------|--------|-------|
| 1 | `n=0` | 0 | 0 | ✓ |
| 2 | `n=1` | 1 | 1 | ✓ |
| 3 | `n=5` | 5 | 5 | ✓ |
| 4 | `n=10` | 55 | 55 | ✓ |
| 5 | `n=20` | 6765 | 6765 | ✓ |

**Reward = passed / total = 5/5 = 1.0**

---

## The Training Pipeline

### Step-by-Step Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING STEP                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. SAMPLE BATCH                                              │
│     Select 4 random problems from 60 training problems        │
│                                                               │
│  2. GENERATE SOLUTIONS                                        │
│     Policy model: 4 samples × 4 prompts = 16 samples         │
│     Momentum model: 4 samples × 4 prompts = 16 samples       │
│     Total: 32 code samples                                    │
│     Time: ~300 seconds (bottleneck - GPU generation)          │
│                                                               │
│  3. EXTRACT CODE                                              │
│     Parse markdown output to get Python function              │
│     Handle ```python ... ``` blocks                           │
│                                                               │
│  4. VERIFY SOLUTIONS                                          │
│     Run each solution against 5 test cases                    │
│     Compute partial reward: passed_count / 5                  │
│                                                               │
│  5. COMPUTE ADVANTAGES                                        │
│     For each prompt's samples:                                │
│       advantage = (reward - mean) / std                       │
│                                                               │
│  6. UPDATE POLICY                                             │
│     loss = -log_prob(tokens) × advantage                      │
│     Only update for samples with non-zero advantage           │
│                                                               │
│  7. UPDATE MOMENTUM (EMA)                                     │
│     θ_k = 0.99 × θ_k + 0.01 × θ_q                            │
│                                                               │
│  8. LOG METRICS                                               │
│     Record loss, reward, entropy, success rate                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### The Partial Reward Innovation

Originally, we used **binary rewards**: 1.0 if ALL tests pass, 0.0 otherwise.

**Problem**: The 0.5B model couldn't solve any problems perfectly, so:
- All rewards = 0
- No gradient signal
- No learning

**Solution**: Partial rewards!
```python
# Old: Binary (all-or-nothing)
reward = 1.0 if result.passed else 0.0

# New: Proportional (partial credit)
reward = result.passed_count / result.total_count
# Pass 3/5 tests → reward = 0.6
```

This gives the model **gradient signal even for imperfect solutions**, allowing it to improve incrementally.

---

## Bugs We Found and Fixed

### Bug 1: TestCase Attribute Mismatch

**Symptom**: `'TestCase' object has no attribute 'input'`

**Root Cause**: Two different TestCase classes in the codebase:
- V1: `tc.input` (for old problems)
- V2: `tc.input_args` (for new procedural problems)

The test harness only supported V1.

**Fix** (harness.py):
```python
# Before
{"input": tc.input, "expected": tc.expected_output}

# After
{"input": getattr(tc, 'input', None) or getattr(tc, 'input_args', None),
 "expected": tc.expected_output}
```

### Bug 2: No Code Extraction

**Symptom**: All rewards = 0 even when model generates correct code

**Root Cause**: Model outputs markdown with explanatory text:
```markdown
## Solution
Here's how to solve this...

```python
def fibonacci(n):
    ...
```

### Explanation
This works because...
```

But we were feeding the **entire output** (including markdown) to the Python interpreter!

**Fix** (run_mgrpo.py):
```python
def extract_code(completion: str) -> str:
    """Extract Python code from markdown output."""
    # Try ```python ... ``` blocks
    python_blocks = re.findall(r'```python\s*(.*?)```', completion, re.DOTALL)
    if python_blocks:
        for block in sorted(python_blocks, key=len, reverse=True):
            if 'def ' in block:
                return block.strip()
    # ... fallback logic
```

### Bug 3: Zero Gradient Signal

**Symptom**: `0 updates` every step, model never learns

**Root Cause**: With binary rewards, small model couldn't pass ALL 5 tests on ANY problem → all rewards = 0 → no learning.

**Fix**: Implemented partial rewards (see above).

---

## Live Training Results

### Current Run (Step 0-7)

| Step | Success Rate | Avg Reward | Updates | Entropy | Notes |
|------|-------------|------------|---------|---------|-------|
| 0 | 50% | 0.50 | 2 | 0.763 | First step - model works! |
| 1 | 50% | 0.50 | 2 | 0.750 | Stable |
| 2 | **75%** | **0.75** | 6 | 0.451 | Improvement! |
| 3 | 50% | 0.45 | 1 | 0.672 | Variance in batch |
| 4 | **75%** | **0.75** | 3 | 0.458 | Good again |
| 5 | **100%** | **0.95** | 3 | 0.928 | Excellent step! |
| 6 | 75% | 0.75 | 4 | 0.390 | Consistent |
| 7 | 50% | 0.40 | 1 | 0.647 | Harder batch |

### Interpretation

**Positive Signs:**
- ✅ Non-zero rewards (model generating working code)
- ✅ Updates happening (gradient signal flowing)
- ✅ Step 5 achieved 100% success with 0.95 avg reward
- ✅ Entropy staying healthy (0.39-0.93), not collapsing
- ✅ Multiple steps at 75%+ success rate

**Expected Variance:**
- Each step samples 4 random problems from 60
- Some batches are harder than others
- Success rate fluctuates based on problem difficulty

**What We're Watching For:**
- Entropy should stay above 0.1 (no collapse)
- Average reward should trend upward over 20 steps
- More "updates" per step = more learning signal

---

## How to Interpret the Metrics

### Success Rate (X/4 successful)

How many of the 4 prompts in the batch had at least one working solution:
- `4/4 = 100%`: All prompts got at least one correct answer
- `2/4 = 50%`: Half the prompts were solved
- `0/4 = 0%`: No correct solutions (bad!)

### Average Reward (avg_reward)

Average of the best reward per prompt:
- `1.0`: All prompts solved perfectly (5/5 tests)
- `0.75`: On average, best solutions pass 3.75/5 tests
- `0.0`: No tests passing (very bad)

### Updates

Number of gradient updates performed:
- `0 updates`: No learning this step (all rewards equal or zero)
- `4 updates`: Learning on 4 samples
- More updates = more diverse reward signal

### Entropy

Diversity of model outputs (higher = more diverse):
- `> 0.5`: Healthy, model exploring different solutions
- `0.1-0.5`: Getting focused, but still okay
- `< 0.1`: **Danger zone** - model becoming overconfident

### Loss

Policy gradient loss (not directly interpretable):
- Non-zero = gradients flowing
- Fluctuations are normal

### ETA

Estimated time remaining based on average step time.

---

## Running the Experiment

### Prerequisites

```bash
# Install dependencies
cd c:/Research/axiom-rl
uv sync

# Requires NVIDIA GPU with ~10GB VRAM
# Tested on RTX 3080 10GB
```

### Basic Usage

```bash
# Full training (20 steps)
uv run python -u scripts/run_mgrpo.py --experiment 15_mgrpo_entropy --steps 20

# Quick test (2 steps)
uv run python -u scripts/run_mgrpo.py --experiment 15_mgrpo_entropy --steps 2 --eval-every 1

# With checkpoints every 3 steps
uv run python -u scripts/run_mgrpo.py --experiment 15_mgrpo_entropy --steps 20 --checkpoint-every 3
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--experiment` | `15_mgrpo_entropy` | Experiment directory |
| `--config` | `None` | Ablation config |
| `--steps` | 200 | Training steps |
| `--eval-every` | 2 | Evaluate every N steps |
| `--checkpoint-every` | 5 | Save every N steps |
| `--resume` | None | Resume from checkpoint |
| `--quiet` | False | Less verbose output |

### Resume from Checkpoint

```bash
uv run python -u scripts/run_mgrpo.py --experiment 15_mgrpo_entropy --steps 20 \
    --resume experiments/15_mgrpo_entropy/checkpoints/default/checkpoint_step_10
```

### Output Files

```
experiments/15_mgrpo_entropy/
├── README.md              # This documentation
├── config.json            # Experiment configuration
├── checkpoints/
│   └── default/
│       └── checkpoint_step_5/
│           ├── policy/    # LoRA weights
│           ├── momentum/  # Momentum model weights
│           ├── optimizer.pt
│           └── state.json # Metrics history
└── models/
    └── default/
        ├── policy/        # Final trained model
        └── entropy.json   # Entropy tracking
```

---

## Technical Details

### Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Fits in 10GB VRAM |
| LoRA Rank | 16 | Good tradeoff |
| LoRA Alpha | 32 | 2× rank |
| Learning Rate | 1e-5 | Low for RL stability |
| Trainable Params | 2.16M (0.44%) | Efficient fine-tuning |

### M-GRPO Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Momentum (m) | 0.99 | Slow evolution |
| Policy Samples | 4 | Per prompt |
| Momentum Samples | 4 | Per prompt |
| IQR k | 0.75 | Entropy filtering |
| Beta (KL) | 0.04 | GRPO default |

### Problem Configuration

| Setting | Value |
|---------|-------|
| Problem Types | 6 (rpn, parentheses, fibonacci, binary_search, edit_distance, coin_change) |
| Difficulty | 4-7 |
| Train Problems | 60 (10 per type) |
| Val Problems | 30 |
| Test Cases per Problem | 5 |
| Reward Type | Partial (proportional) |

### Timing (RTX 3080)

| Phase | Time |
|-------|------|
| Generation (32 samples) | ~300s |
| Verification | ~3s |
| Training | ~1s |
| **Total per step** | **~305s** |
| **20 steps** | **~100 min** |

---

## Hypotheses Being Tested

| ID | Hypothesis | Success Criteria | Status |
|----|------------|------------------|--------|
| **H1** | Momentum stabilizes training | Lower reward variance than vanilla GRPO | 🔄 Testing |
| **H2** | IQR filtering prevents collapse | Entropy > 0.1 throughout training | ✅ So far (0.39-0.93) |
| **H3** | Extended learning beyond early phase | Gains in steps 10-20 | 🔄 Waiting |

---

## References

1. **M-GRPO Paper:** Bai et al., "Stabilizing Self-Supervised RL with Momentum-Anchored Policy Optimization", arXiv:2512.13070, Dec 2025

2. **Entropy Mechanism:** Cui et al., "The Entropy Mechanism of RL for LLM Reasoning", arXiv:2505.22617, May 2025

3. **GRPO:** Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning", 2024

4. **Previous Experiments:** Experiments 7 (GRPO validation), 9-14 (V2 problem design)

---

## Changelog

### 2024-12-20: Initial Run

- Fixed TestCase attribute mismatch (V1 vs V2)
- Added code extraction from markdown
- Implemented partial rewards
- First successful training run with 50-100% success rates
- Model generating working Python code!
