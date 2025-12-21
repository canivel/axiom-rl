# Experiment 16: M-GRPO with Class Wrapper Fix

**Status:** In Progress
**Date:** 2024-12-21
**Branch:** v2-problem-design
**Hardware:** NVIDIA T4 16GB (Google Colab)
**Predecessor:** Experiment 15

---

## Problem Statement

Experiment 15 achieved 99% training success but only **10% final evaluation accuracy**. Root cause analysis revealed:

1. **Model outputs `class Solution` wrappers** instead of standalone functions
2. **Test harness expects standalone functions** - can't find method inside class
3. **Sampling vs Greedy discrepancy** - 8 samples sometimes produce correct format, greedy doesn't

## Changes from Experiment 15

### Fix 1: Improved Code Extraction

```python
def extract_code(completion: str) -> str:
    """
    Extract Python code from model completion.

    NEW: Handles class Solution wrappers by extracting the method
    and converting to standalone function.
    """
    # ... existing extraction logic ...

    # NEW: Handle class Solution wrappers
    if "class Solution" in code:
        code = extract_method_from_class(code, func_name)

    return code

def extract_method_from_class(code: str, func_name: str) -> str:
    """Convert class method to standalone function."""
    # Find the method definition
    # Remove 'self' parameter
    # Adjust indentation
    # Return standalone function
```

### Fix 2: Improved Prompts

**Before (Exp 15):**
```
Write ONLY the complete function implementation.
```

**After (Exp 16):**
```
Write ONLY a standalone Python function.
Do NOT wrap it in a class.
Do NOT use 'class Solution'.
The function should be directly callable.
```

### Fix 3: Greedy Evaluation During Training

Added periodic greedy evaluation to catch format issues early:
```python
if step % greedy_eval_every == 0:
    greedy_accuracy = evaluate_greedy(model, val_problems)
    if greedy_accuracy < 0.2:
        print("WARNING: Greedy accuracy low - check output format!")
```

## Expected Improvements

| Metric | Exp 15 | Expected Exp 16 |
|--------|--------|-----------------|
| Training Success | 99% | 99% |
| Final Eval Accuracy | 10% | **60%+** |
| Format Errors | ~90% | **<10%** |

## Running the Experiment

### Google Colab (Recommended)

Use the standalone notebook: `notebooks/experiment_16_mgrpo_standalone.ipynb`

1. Open in Colab
2. Select T4 GPU runtime
3. Run all cells
4. Training takes ~2.5 hours for 20 steps

### Local (if you have GPU)

```bash
cd c:/Research/axiom-rl
uv run python scripts/run_mgrpo.py --experiment 16_mgrpo_class_fix --steps 20 --eval-every 2
```

## Hypotheses

| ID | Hypothesis | Success Criteria |
|----|------------|------------------|
| H1 | Class extraction fixes format errors | <10% format errors in eval |
| H2 | Improved prompts guide model better | Greedy accuracy > 50% |
| H3 | M-GRPO still prevents collapse | Entropy > 0.1 throughout |

## Files

```
experiments/16_mgrpo_class_fix/
├── README.md              # This file
├── config.json            # Experiment configuration
├── checkpoints/           # Training checkpoints
└── models/                # Final trained model

notebooks/
└── experiment_16_mgrpo_standalone.ipynb  # Colab notebook
```
