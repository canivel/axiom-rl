# Experiment 02: Focused Improvement on Weak Problems

**Date:** 2024-12-05
**Status:** Completed

## Objective

Test whether focused training on weak problems (`remove_duplicates` and `fibonacci`) can improve model performance using expert iteration with expanded training data (5x more problems per type).

## Configuration

| Parameter | Value |
|-----------|-------|
| Target Problems | `remove_duplicates`, `fibonacci` |
| Training per type | 25 problems |
| Validation per type | 10 problems |
| Test per type | 10 problems |
| Iterations | 3 |
| Samples per problem | 8 (Best-of-8) |
| Test cases per problem | 5 |

## Results

### Accuracy Over Iterations

| Iteration | Val Overall | Test Overall | remove_duplicates (Val) | fibonacci (Val) |
|-----------|-------------|--------------|-------------------------|-----------------|
| 0 (Baseline) | 51.2% | 41.9% | 30.0% | 72.5% |
| 1 | 34.4% | 23.8% | 5.0% | 63.7% |
| 2 | 31.9% | 31.2% | 0.0% | 63.7% |
| **Change** | **-19.4%** | **-10.6%** | **-30.0%** | **-8.8%** |

### Training Metrics

| Iteration | Solutions Collected | Training Loss (Start → End) |
|-----------|--------------------|-----------------------------|
| 0 → 1 | 38 | 2.50 → 0.80 |
| 1 → 2 | 25 | 0.75 → 0.65 |

## Analysis

### Key Finding: Catastrophic Forgetting

**The model degraded over training iterations instead of improving.** This is a critical negative result that reveals fundamental challenges with expert iteration on small, focused problem sets.

### What Went Wrong

1. **Complete collapse on `remove_duplicates`**
   - Baseline: 30% validation accuracy
   - After training: 0% accuracy
   - The model completely forgot how to solve this problem type

2. **Slight degradation on `fibonacci`**
   - Baseline: 72.5% validation accuracy
   - After training: 63.7% accuracy
   - Some forgetting, but less severe

3. **Overfitting to specific solutions**
   - Training loss decreased (2.5 → 0.65), showing the model was learning
   - But it memorized specific solutions rather than generalizing
   - Too few training examples (38 solutions) led to overfitting

4. **Solution diversity problem**
   - Collecting only verified solutions creates a narrow distribution
   - The model sees the same patterns repeatedly
   - No exposure to failed attempts or alternative approaches

### Why This Happened

1. **Small training set** - Only 38 verified solutions in iteration 0
2. **Narrow problem distribution** - Only 2 problem types (vs. 10 in baseline)
3. **Cumulative LoRA training** - Each iteration builds on previous, compounding errors
4. **No regularization** - No mechanism to preserve original capabilities

## Conclusions

1. **Expert iteration alone is insufficient** for focused improvement
2. **Need more problem diversity** during training to prevent catastrophic forgetting
3. **Procedural generation is critical** - Need much larger, more diverse problem sets
4. **May need different training strategy**:
   - Mixing focused and general problems
   - Lower learning rates for fine-tuning
   - Regularization to preserve base capabilities
   - RLHF/DPO instead of pure SFT

## Lessons for Future Experiments

1. **Scale up problem diversity** - Use procedural generation for 100s of problem types
2. **Include negative examples** - Show failed attempts, not just successes
3. **Mixed training** - Combine focused improvement with general capability preservation
4. **Early stopping** - Monitor validation loss and stop if degradation detected
5. **Smaller learning rate** - Prevent aggressive weight updates

## Files

- `train.json` - Training problems (50 total)
- `val.json` - Validation problems (20 total)
- `test.json` - Test problems (20 total)
- `metrics.jsonl` - Per-iteration metrics
- `models/iter_0/` - Model checkpoint after iteration 0

## Next Steps

See **Experiment 03: Procedural Generation** for testing with much larger, more diverse problem sets generated procedurally.
