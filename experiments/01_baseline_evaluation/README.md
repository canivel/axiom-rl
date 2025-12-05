# Experiment 01: Baseline Evaluation

**Date:** 2024-12-04
**Status:** Completed

## Objective

Establish baseline performance of `Qwen/Qwen2.5-Coder-1.5B-Instruct` on classic programming problems using Best-of-N evaluation methodology.

## Methodology

### Evaluation Protocol: Best-of-N

For each problem, we generate N=8 independent solutions using temperature sampling (T=0.7). We count how many of these 8 samples pass ALL test cases. This methodology:

1. **Accounts for variance** - Single-sample evaluation is noisy
2. **Measures reliability** - A score of 8/8 means the model reliably solves the problem
3. **Identifies weaknesses** - Low scores indicate problems the model struggles with

### Problems Evaluated

10 classic programming problems from `axiom/problems/problems.json`:

| Problem | Test Cases | Algorithm Required |
|---------|------------|-------------------|
| two_sum | 4 | Hash map lookup |
| fizzbuzz | 3 | Modulo conditionals |
| reverse_string | 5 | String slicing |
| is_palindrome | 5 | String comparison |
| max_subarray | 5 | Kadane's algorithm |
| fibonacci | 6 | Dynamic programming |
| binary_search | 5 | Divide and conquer |
| valid_parentheses | 6 | Stack-based matching |
| merge_sorted_arrays | 5 | Two-pointer merge |
| remove_duplicates | 5 | In-place modification |

## Results

### Per-Problem Accuracy

| Problem | Score | Accuracy | Status |
|---------|-------|----------|--------|
| binary_search | 8/8 | 100% | Strong |
| fizzbuzz | 8/8 | 100% | Strong |
| is_palindrome | 8/8 | 100% | Strong |
| max_subarray | 8/8 | 100% | Strong |
| reverse_string | 8/8 | 100% | Strong |
| two_sum | 8/8 | 100% | Strong |
| merge_sorted_arrays | 7/8 | 87.5% | Minor issues |
| valid_parentheses | 7/8 | 87.5% | Minor issues |
| **fibonacci** | **5/8** | **62.5%** | **Needs improvement** |
| **remove_duplicates** | **1/8** | **12.5%** | **Needs improvement** |

### Overall

- **Total:** 68/80 samples passed
- **Accuracy:** 85.0%

## Analysis

### Strong Performance (100%)

The model excels at:
- **Simple algorithms**: fizzbuzz, reverse_string, is_palindrome
- **Well-known patterns**: binary_search, max_subarray (Kadane's), two_sum

### Moderate Performance (87.5%)

- **merge_sorted_arrays**: Occasional off-by-one errors in pointer management
- **valid_parentheses**: Some edge cases with nested brackets

### Weak Performance (<70%)

1. **fibonacci (62.5%)**:
   - Often fails on edge cases (n=0, n=1)
   - Sometimes uses inefficient recursive approach that times out

2. **remove_duplicates (12.5%)**:
   - Struggles with in-place modification requirement
   - Often returns new list instead of modifying in place
   - Misunderstands the problem specification

## Conclusions

1. The model has strong baseline capability on standard algorithms
2. **Two problems identified for focused improvement:**
   - `remove_duplicates` (12.5%) - Primary target
   - `fibonacci` (62.5%) - Secondary target
3. These weak spots provide opportunity to demonstrate self-improvement

## Files

- `baseline_best_of_8.json` - Raw results data
- `config.json` - Evaluation configuration

## Next Steps

See **Experiment 02: Focused Improvement** for targeted training on weak problems.
