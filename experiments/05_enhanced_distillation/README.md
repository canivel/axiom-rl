# Experiment 05: Enhanced Distillation

## Hypothesis

Distilling from multiple teacher models (Claude + Gemini) produces more robust reasoning patterns than single-teacher distillation. The combination provides:
1. More diverse reasoning styles
2. Higher verification rates
3. Better downstream SFT performance

## Configuration

### Teacher Models
- **Claude**: claude-sonnet-4-20250514 (Anthropic)
- **Gemini**: gemini-2.5-flash (Google)

### Problem Types
All 11 V2 algorithmic generators:
- rpn, arithmetic, parentheses
- fizzbuzz, reverse_string, is_palindrome
- fibonacci, binary_search, two_sum
- max_subarray, remove_duplicates

### Generation Parameters
- Problems per type: 10 (110 total per teacher)
- Difficulty range: 1-7
- Temperature: 0.7
- Test cases per problem: 5
- Random seed: 42

### Commands
```bash
# Generate with Claude
python scripts/generate_teacher_data_v2.py --teacher claude --problems-per-type 10

# Generate with Gemini
python scripts/generate_teacher_data_v2.py --teacher gemini --problems-per-type 10

# Generate with both (comparison)
python scripts/generate_teacher_data_v2.py --teacher both --problems-per-type 10
```

## Results

| Teacher | Total Traces | Verified | Verification Rate | Notes |
|---------|--------------|----------|-------------------|-------|
| **Claude Direct** | 33 | 33 | **100%** | Solved in conversation |
| Gemini API | 33 | 25 | 75.8% | Via API calls |
| **Combined** | 60 | 58 | 96.7% | Best of both |

### Per-Problem-Type Breakdown

| Problem Type | Claude Pass% | Gemini Pass% | Notes |
|--------------|--------------|--------------|-------|
| rpn          | 100% (3/3)   | 67% (2/3)    |       |
| arithmetic   | 100% (3/3)   | 33% (1/3)    | Gemini struggles |
| parentheses  | 100% (3/3)   | 67% (2/3)    |       |
| fizzbuzz     | 100% (3/3)   | 100% (3/3)   | Both perfect |
| reverse_string | 100% (3/3) | 100% (3/3)   | Both perfect |
| is_palindrome | 100% (3/3)  | 67% (2/3)    |       |
| **fibonacci** | 100% (3/3)  | **0% (0/3)** | Gemini failed all |
| binary_search | 100% (3/3)  | 100% (3/3)   | Both perfect |
| two_sum      | 100% (3/3)   | 100% (3/3)   | Both perfect |
| max_subarray | 100% (3/3)   | 100% (3/3)   | Both perfect |
| remove_duplicates | 100% (3/3) | 100% (3/3) | Both perfect |

## Analysis

1. **Claude Direct is more reliable**: 100% vs 75.8% verification rate
2. **Fibonacci is problematic for Gemini**: 0/3 passed (likely off-by-one or indexing issue)
3. **Both excel at simpler problems**: fizzbuzz, reverse_string, binary_search, two_sum, max_subarray, remove_duplicates
4. **Gemini struggles with**: arithmetic expressions, some RPN/parentheses edge cases

### Key Insight: Claude Direct Approach
Using Claude in conversation to solve problems directly (without API calls):
- Zero cost
- Zero latency
- 100% verification rate
- Perfect for bootstrapping training data

## Decision

**Use Claude Direct as primary teacher**, supplement with Gemini for diversity:
- Claude traces: Higher quality, more reliable
- Gemini traces: Add diversity in reasoning style
- Combined dataset: 60 verified traces ready for training

## Next Step

After completing this experiment, proceed to Phase 1A: SFT Baseline training on the enhanced dataset.

## Files

- `config.json` - Experiment configuration
- `claude_traces_v2.jsonl` - Claude-generated traces (data/coldstart_v2/)
- `gemini_traces_v2.jsonl` - Gemini-generated traces (data/coldstart_v2/)
- `summary.json` - Results summary
