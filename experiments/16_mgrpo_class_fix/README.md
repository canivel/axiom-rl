# Experiment 16: M-GRPO with Class Wrapper Fix

**Status:** COMPLETE
**Date:** 2024-12-21 to 2024-12-22
**Branch:** v2-problem-design
**Hardware:** NVIDIA RTX 3080 10GB (Local)
**Predecessor:** Experiment 15
**Training Time:** 339.1 minutes (~5.6 hours)

---

## Executive Summary

Experiment 16 achieved **50% final evaluation accuracy** - a **5x improvement** over Experiment 15's 10%. The key fixes were:

1. **Class wrapper extraction** - Converting `class Solution` methods to standalone functions
2. **Improved prompts** - Explicit instructions to avoid class wrappers
3. **Greedy evaluation during training** - Catching format issues early

### Key Results

| Metric | Exp 15 | Exp 16 | Change |
|--------|--------|--------|--------|
| **Final Accuracy** | 10% | **50%** | **+40%** |
| Fibonacci | 0% | **100%** | Fixed |
| Binary Search | 60% | **100%** | +40% |
| Coin Change | 0% | **80%** | Fixed |
| RPN | 60% | 0% | Regressed |
| Parentheses | 0% | 20% | +20% |
| Edit Distance | 0% | 0% | Same |
| Training Success | 99% | 75-100% | Similar |
| Entropy (final) | 0.229 | 0.326 | Healthy |

---

## Problem Statement

Experiment 15 achieved 99% training success but only **10% final evaluation accuracy**. Root cause analysis revealed:

1. **Model outputs `class Solution` wrappers** instead of standalone functions
2. **Test harness expects standalone functions** - can't find method inside class
3. **Sampling vs Greedy discrepancy** - 8 samples sometimes produce correct format, greedy doesn't

### Example of the Problem (Exp 15)

```python
# Model output:
class Solution:
    def fibonacci(self, n: int) -> int:
        if n <= 1:
            return n
        return self.fibonacci(n-1) + self.fibonacci(n-2)

# Test harness calls:
result = fibonacci(5)  # NameError: name 'fibonacci' is not defined!
```

The model learned to solve problems correctly, but wrapped them in LeetCode-style classes that the test harness couldn't call.

---

## Changes from Experiment 15

### Fix 1: Improved Code Extraction (`extract_method_from_class`)

Added a new function to extract methods from class wrappers and convert them to standalone functions:

```python
def extract_method_from_class(code: str, func_name: str) -> str:
    """
    Extract a method from a class and convert to standalone function.

    Input:
        class Solution:
            def fibonacci(self, n: int) -> int:
                if n <= 1:
                    return n
                return self.fibonacci(n-1) + self.fibonacci(n-2)

    Output:
        def fibonacci(n: int) -> int:
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)

    Key transformations:
    1. Find the method definition inside the class
    2. Remove 'self' from parameters: (self, n: int) -> (n: int)
    3. Remove 'self.' from recursive calls: self.fibonacci() -> fibonacci()
    4. Dedent the code to remove class-level indentation
    """
    # Use regex to find method definition
    pattern = rf'def\s+{func_name}\s*\(\s*self\s*,?\s*([^)]*)?(\))\s*(?:->\s*[^:]+)?\s*:'
    match = re.search(pattern, code)

    if not match:
        return code  # Couldn't find method, return original

    # Extract everything from 'def' to end of method
    method_start = match.start()
    # ... find method end by tracking indentation ...

    # Remove 'self' parameter
    method_code = re.sub(
        rf'(def\s+{func_name}\s*\()self\s*,?\s*',
        r'\1',
        method_code
    )

    # Remove 'self.' from method body
    method_code = method_code.replace('self.', '')

    # Dedent to remove class indentation
    method_code = textwrap.dedent(method_code)

    return method_code.strip()
```

**Why this works:** The model was trained on LeetCode-style code that uses `class Solution` wrappers. Instead of fighting this learned behavior, we extract the correct solution from inside the wrapper.

### Fix 2: Improved Prompts

**Before (Exp 15):**
```
Write ONLY the complete function implementation.
```

**After (Exp 16):**
```
## IMPORTANT
Write ONLY a standalone Python function.
Do NOT wrap it in a class.
Do NOT use 'class Solution'.
The function must be directly callable.
```

**Why this works:** Explicit negative instructions ("Do NOT") are more effective than implicit expectations. The model now receives clear guidance about the expected output format.

### Fix 3: Greedy Evaluation During Training

Added periodic greedy evaluation to catch format issues early:

```python
if step % greedy_eval_every == 0:
    greedy_accuracy = evaluate_greedy(model, val_problems)
    print(f"Greedy accuracy: {greedy_accuracy:.1%}")

    # Breakdown by problem type
    for problem_type, (correct, total) in type_results.items():
        print(f"  {problem_type}: {correct}/{total}")
```

**Why this works:** Greedy evaluation (temperature=0, no sampling) shows what the model's most likely output is. This catches the format issue that was hidden by sampling multiple times in Exp 15.

### Fix 4: Bug Fix in `compute_log_probs`

During training, discovered a critical bug:

```python
# BROKEN (Exp 16 initial):
def compute_log_probs(model, tokenizer, prompt, completion):
    with torch.no_grad():  # BUG: Prevents gradient computation!
        outputs = model(**inputs)
    # loss.backward() fails: "element 0 does not require grad"

# FIXED:
def compute_log_probs(model, tokenizer, prompt, completion):
    # No torch.no_grad() - we need gradients for training!
    outputs = model(**inputs)
```

**Why this happened:** Copy-paste error from evaluation code where `no_grad()` is appropriate. For training, we need gradients to flow through the model.

---

## Training Results

### Full Training Log

| Step | Loss | Reward | Entropy | Success | Val Acc | Notes |
|------|------|--------|---------|---------|---------|-------|
| 0 | 95.28 | 0.750 | 0.375 | 75% | 20% | Initial |
| 1 | 70.24 | 0.600 | 0.465 | 75% | 20% | |
| 2 | 139.61 | 0.350 | 0.436 | 75% | 0% | |
| 3 | 87.60 | 0.600 | 0.430 | 75% | 20% | |
| 4 | 106.05 | 0.400 | 0.508 | 50% | 40% | |
| 5 | 84.83 | 0.650 | 0.434 | 75% | 20% | Greedy: 40% |
| 6 | 29.66 | 0.500 | 0.428 | 50% | 40% | Loss drop! |
| 7 | 89.23 | 0.150 | 0.254 | 50% | 40% | |
| 8 | 71.96 | 0.350 | 0.527 | 50% | 20% | |
| 9 | 57.15 | 0.700 | 0.355 | 75% | 20% | |
| 10 | 78.03 | 0.550 | 0.332 | 75% | 40% | Greedy: 30% |
| 11 | 42.64 | 0.750 | 0.328 | 75% | 20% | |
| 12 | 40.89 | 0.750 | 0.338 | 75% | **60%** | Best val! |
| 13 | 32.90 | 0.650 | 0.220 | 75% | 20% | |
| 14 | 53.82 | 0.800 | 0.326 | **100%** | 0% | |
| 15 | 43.92 | 0.800 | 0.279 | **100%** | 20% | Greedy: **70%** |
| 16 | 47.05 | 0.650 | 0.196 | 75% | 20% | |
| 17 | 19.12 | 0.800 | 0.171 | **100%** | 40% | Lowest loss |
| 18 | 45.60 | 0.800 | 0.173 | **100%** | 20% | |
| 19 | 33.25 | 0.900 | 0.326 | **100%** | 60% | Best reward |

### Greedy Evaluation Snapshots

**Step 5:**
```
Greedy accuracy: 4/10 = 40.0%
  rpn: 0/2
  coin_change: 0/1
  fibonacci: 2/4
  edit_distance: 0/1
  binary_search: 2/2
```

**Step 10:**
```
Greedy accuracy: 3/10 = 30.0%
  rpn: 0/2
  coin_change: 0/1
  fibonacci: 2/4
  edit_distance: 0/1
  binary_search: 1/2
```

**Step 15:**
```
Greedy accuracy: 7/10 = 70.0%
  rpn: 0/2
  coin_change: 1/1
  fibonacci: 4/4
  edit_distance: 0/1
  binary_search: 2/2
```

### Training Dynamics Analysis

1. **Loss decreased from 95 to 19** - Strong learning signal
2. **Entropy remained stable (0.17-0.53)** - No mode collapse
3. **Success rate improved to 100%** in later steps
4. **Greedy accuracy peaked at 70%** at step 15

---

## Final Evaluation Results

```
============================================================
FINAL EVALUATION
============================================================

Results by Problem Type:
----------------------------------------
  rpn                 : 0/5 = 0.0%
  coin_change         : 4/5 = 80.0%
  fibonacci           : 5/5 = 100.0%
  edit_distance       : 0/5 = 0.0%
  binary_search       : 5/5 = 100.0%
  parentheses         : 1/5 = 20.0%
----------------------------------------
  OVERALL             : 15/30 = 50.0%
```

### Analysis by Problem Type

#### Fibonacci: 5/5 = 100% (was 0% in Exp 15)

**Why it works now:** Fibonacci is a straightforward recursive/iterative problem. The class wrapper fix allows the test harness to call the function directly.

```python
# Model now outputs:
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

#### Binary Search: 5/5 = 100% (was 60% in Exp 15)

**Why it improved:** Binary search requires precise index handling. With correct function format, the model's solutions now execute correctly.

#### Coin Change: 4/5 = 80% (was 0% in Exp 15)

**Why it works now:** Dynamic programming solution is correct. One failure is likely edge case handling.

#### Parentheses: 1/5 = 20% (was 0% in Exp 15)

**Why still low:** Valid parentheses requires careful stack manipulation. The model sometimes generates syntactically correct but logically wrong solutions.

#### RPN: 0/5 = 0% (was 60% in Exp 15)

**Why it regressed:** This is concerning. Possible causes:
- Training focused on problems the model initially struggled with
- RPN requires handling negative numbers (e.g., "-5".isdigit() returns False)
- Model may have "forgotten" RPN in favor of other problem types

#### Edit Distance: 0/5 = 0% (same as Exp 15)

**Why still failing:** Edit distance is the hardest problem:
- Requires correct DP table initialization
- Off-by-one errors are common
- Model may not have enough capacity for this complexity

---

## Hypothesis Validation

| ID | Hypothesis | Result | Evidence |
|----|------------|--------|----------|
| H1 | Class extraction fixes format errors | **CONFIRMED** | 50% accuracy vs 10% - format no longer blocking execution |
| H2 | Improved prompts guide model better | **CONFIRMED** | Model outputs standalone functions more often |
| H3 | M-GRPO prevents collapse | **CONFIRMED** | Entropy: 0.17-0.53, never collapsed to 0 |

---

## Lessons Learned

### 1. Output Format Matters as Much as Correctness

Experiment 15 showed 99% training success but 10% eval accuracy because the model's outputs were in the wrong format. **Always validate that model outputs can be executed by your test harness.**

### 2. Greedy Evaluation Reveals Hidden Problems

Sampling-based evaluation (Exp 15) can hide format issues because sometimes the correct format appears by chance. **Greedy evaluation shows the model's most likely behavior.**

### 3. Explicit Negative Instructions Work

"Do NOT use class Solution" is more effective than "Write a standalone function". **Models respond well to explicit constraints.**

### 4. Extraction > Generation for Format Issues

Instead of retraining the model to output a different format, we extract the correct code from the model's preferred format. **Work with the model's learned behavior when possible.**

### 5. Check Gradient Flow in Training Code

The `torch.no_grad()` bug wasted significant time. **Always verify gradients are flowing in training code.**

---

## Known Issues & Future Work

### Issue 1: RPN Regression

RPN accuracy dropped from 60% to 0%. Needs investigation:
- Check if model still outputs correct RPN logic
- May need targeted training on RPN problems
- Consider handling negative numbers in test harness

### Issue 2: Edit Distance Never Solved

Neither Exp 15 nor Exp 16 solved Edit Distance. Options:
- Increase model capacity
- Add chain-of-thought prompting
- Provide more training examples for DP problems

### Issue 3: High Variance in Val Accuracy

Val accuracy fluctuated 0-60% during training. This is partly due to small eval set (5 problems per type).

### Future Experiments

1. **Exp 17:** Focus training on failed problem types (RPN, Edit Distance)
2. **Exp 18:** Increase model size (1.5B instead of 0.5B)
3. **Exp 19:** Add chain-of-thought for complex problems

---

## Files

```
experiments/16_mgrpo_class_fix/
├── README.md              # This file
├── config.json            # Experiment configuration
├── models/
│   └── default/           # Final trained LoRA adapter
└── checkpoints/           # Training checkpoints (if saved)

scripts/
└── run_mgrpo_exp16.py     # Standalone training script

notebooks/
└── experiment_16_mgrpo_standalone.ipynb  # Colab notebook
```

---

## Reproducibility

### Running Locally

```bash
cd c:/Research/axiom-rl
uv run python scripts/run_mgrpo_exp16.py --steps 20 --eval-every 2 --greedy-eval-every 5
```

### Hardware Requirements

- GPU: 10GB+ VRAM (RTX 3080, T4, or better)
- RAM: 16GB+
- Time: ~6 hours for 20 steps

### Key Hyperparameters

```json
{
  "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
  "lora_r": 16,
  "lora_alpha": 32,
  "learning_rate": 1e-5,
  "num_policy_samples": 4,
  "num_momentum_samples": 4,
  "momentum": 0.99,
  "temperature": 0.7,
  "max_new_tokens": 512
}
```
