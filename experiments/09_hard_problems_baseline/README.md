# Experiment 09: Hard Problems Baseline

**Status**: Completed
**Date**: 2025-12-06
**Duration**: ~10 minutes

## Objective

Evaluate the base Qwen 0.5B model on LeetCode-hard style algorithmic problems to:
1. Establish a baseline for hard problem performance
2. Identify which problem types the model struggles with
3. Create targets for GRPO reinforcement learning

## Methodology

### Problem Categories

We created 10 hard problem generators covering major algorithmic paradigms:

| Category | Problems | Algorithms Required |
|----------|----------|---------------------|
| Dynamic Programming | LCS, Edit Distance, Knapsack, LIS, Coin Change | Memoization, 2D DP, State transitions |
| Backtracking | N-Queens | Recursive search, constraint checking |
| Interval Problems | Merge Intervals | Sorting, greedy merging |
| Two Pointers | Trapping Rain Water | Left-right scan |
| Binary Search | Median of Sorted Arrays | O(log(m+n)) search |
| String DP | Word Break | DP with dictionary lookup |

### Evaluation Protocol

- **Model**: Qwen/Qwen2.5-Coder-0.5B-Instruct (base, no fine-tuning)
- **Test Cases**: 5 per problem
- **Difficulty**: Level 5 (medium-hard)
- **Temperature**: 0.2 (deterministic)
- **Success Criteria**: Pass ALL test cases

## Results

### Per-Problem Performance

| Problem Type | Result | Score | Category | Analysis |
|--------------|--------|-------|----------|----------|
| LCS | PASS | 5/5 | DP | Model knows classic DP pattern |
| Edit Distance | FAIL | 0/5 | DP | Similar to LCS but fails - interesting |
| Knapsack | FAIL | 0/5 | DP | Index error - buggy implementation |
| LIS | PASS | 5/5 | DP | Binary search optimization works |
| Coin Change | FAIL | 0/5 | DP | Wrong algorithm applied |
| Word Break | PASS | 5/5 | DP | Memoization correct |
| Merge Intervals | PASS | 5/5 | Greedy | Standard pattern recognized |
| Median Sorted Arrays | PASS | 5/5 | Binary Search | Complex but model handles it |
| Trapping Rain Water | PASS | 5/5 | Two Pointer | Classic technique works |
| N-Queens | FAIL | 2/5 | Backtracking | Partial - some test cases pass |

### Summary Statistics

```
Overall Accuracy: 60% (6/10 problems passed)

By Category:
- Dynamic Programming: 50% (3/6 passed)
- Backtracking: 0% (0/1 passed, partial credit 2/5)
- Greedy/Intervals: 100% (1/1 passed)
- Two Pointer: 100% (1/1 passed)
- Binary Search: 100% (1/1 passed)
```

## Detailed Analysis

### Strong Performance Areas

1. **LCS (Longest Common Subsequence)**
   - Model correctly implements 2D DP
   - Handles variable-length strings
   - Returns correct subsequence length

2. **LIS (Longest Increasing Subsequence)**
   - Uses O(n log n) binary search approach
   - Handles edge cases correctly
   - Efficient implementation

3. **Word Break**
   - Correct DP formulation
   - Proper dictionary lookup
   - Handles all test cases

4. **Merge Intervals**
   - Sorts by start time
   - Correctly merges overlapping intervals
   - Returns proper format

5. **Median of Two Sorted Arrays**
   - Surprisingly, model handles this O(log(m+n)) problem
   - Correct binary search implementation
   - Handles odd/even total lengths

6. **Trapping Rain Water**
   - Classic two-pointer or prefix max approach
   - Model knows this pattern well

### Weak Performance Areas

1. **Edit Distance (0/5)**
   - Very similar to LCS but model fails
   - Likely wrong recurrence relation
   - Good candidate for targeted training

2. **Knapsack (0/5)**
   - Index out of range error
   - Implementation bug, not algorithmic
   - Model has partial understanding

3. **Coin Change (0/5)**
   - Returns wrong values
   - Confuses with different DP formulation
   - Classic interview problem - should improve

4. **N-Queens (2/5)**
   - Partial success suggests understanding
   - Backtracking logic issues
   - Constraint checking may be wrong

## Key Insights

### Surprising Findings

1. **Median of Sorted Arrays works** - This is often considered one of the hardest LeetCode problems, yet the 0.5B model solves it correctly.

2. **Edit Distance fails but LCS passes** - These are nearly identical problems (both 2D DP), showing the model may have memorized patterns rather than truly understanding DP.

3. **60% baseline is respectable** - A 0.5B model achieving 60% on hard problems without any fine-tuning shows strong base capabilities.

### Training Targets

The 4 failing problems are excellent GRPO training targets:
- **Edit Distance**: Close to LCS, small delta to learn
- **Knapsack**: Classic problem, high value
- **Coin Change**: Common interview problem
- **N-Queens**: Already partial success, could improve

## Files Created

```
axiom/procedural/generators_hard.py  - 10 hard problem generators
scripts/test_hard_problems.py        - Evaluation script
```

## Next Steps

1. **GRPO Training**: Use curriculum learning on the 4 failing problem types
2. **Teacher Distillation**: Generate Claude/Gemini traces for hard problems
3. **Difficulty Scaling**: Test at higher difficulty levels (7-10)
4. **Compare Models**: Test SFT-fine-tuned model on same problems

## Reproduction

```bash
# Run evaluation
uv run python scripts/test_hard_problems.py --difficulty 5

# Test specific problems
uv run python scripts/test_hard_problems.py --problems coin_change edit_distance --difficulty 5

# Verbose output
uv run python scripts/test_hard_problems.py --verbose
```

## Conclusion

The base Qwen 0.5B model shows surprising capability on hard algorithmic problems (60% accuracy), but has clear weaknesses in certain DP formulations and backtracking. These weaknesses represent excellent training targets for GRPO, as:

1. The problems have clear reward signals (test case pass/fail)
2. The model has partial understanding (not starting from zero)
3. The failures are systematic (same problem types fail consistently)

This experiment establishes the baseline for the next phase: targeted GRPO training on hard problems.
