# Critical Analysis: What's Wrong With Our Approach

## Executive Summary

**The core problem**: Our procedural problems are TOO EASY and don't require actual algorithmic reasoning. The model can solve them by memorizing patterns, not by learning algorithms.

---

## The Fundamental Misunderstanding

### What the README Promises

From the README:
> "The model must **Explore** - Generate diverse candidate solutions... **Discover** - Find solutions that satisfy the environment (pass tests)... **Improve** - Train on its own discoveries to get better at exploring"

The AlphaGo analogy:
> "Learn from code *the model generates itself*, where the only signal is 'did it pass the tests?'"

### What We're Actually Doing

**Problem**: "Evaluate `47 + 17 + 15`"
**Expected Output**: `79`
**Model's "Solution"**:
```python
def evaluate_expression(expr: str) -> int:
    return 79  # Hardcoded!
```

This is NOT learning an algorithm. This is **memorizing answers**.

---

## The Three Critical Flaws

### Flaw #1: Each Problem Has ONE Test Case

Our procedural problems generate:
```python
ProceduralProblem(
    input_data="47 + 17 + 15",
    expected_output=79,
    test_cases=[{"input": [], "output": 79}]  # ONE test case!
)
```

**The Problem**: With only ONE test case, the model can pass by returning a hardcoded constant. There's no way to verify it learned the algorithm.

**The Fix**: Each problem should have MULTIPLE test cases with the SAME function signature but DIFFERENT inputs.

### Flaw #2: Our Function Signatures Take NO Arguments

Current:
```python
def solve() -> int:  # No arguments!
    return 79
```

Expected (correct):
```python
def evaluate_expression(expr: str) -> int:  # Takes input!
    # Must actually parse and evaluate
```

**The Problem**: With `def solve()`, the model cannot be tested on different inputs. The function is meaningless.

**The Fix**: Functions must take the problem input as an argument.

### Flaw #3: We're Not Testing Generalization

Our "evaluation" on val/test sets uses the SAME function signature:
- Train problem: "Evaluate `3 + 5`" → `def solve(): return 8`
- Test problem: "Evaluate `7 + 2`" → `def solve(): return 9`

These are DIFFERENT FUNCTIONS. Knowing the first doesn't help with the second.

**The Problem**: We measure "accuracy" by counting how many problems the model can solve, but each problem is independent. The model doesn't need to generalize - it just needs to recognize patterns and compute.

**The Fix**: Test generalization by training on examples of ONE function and testing with NEW inputs to the SAME function.

---

## How Cold Start Data is Different

Look at the cold start data from Gemini:

```python
def two_sum(nums: List[int], target: int) -> List[int]:
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
```

This is a REAL algorithm that:
1. Takes INPUT parameters
2. Works for ANY valid input
3. Requires actual reasoning to implement

Our procedural "solve()" functions are NOT algorithms - they're just constant returns.

---

## The Correct Design

### What Procedural Problems Should Look Like

**Problem Type**: Arithmetic Expression Evaluation

**Training Instance**:
```python
{
    "function_signature": "def evaluate(expr: str) -> int:",
    "description": "Evaluate an arithmetic expression...",
    "test_cases": [
        {"input": ["3 + 5"], "expected": 8},
        {"input": ["10 - 2"], "expected": 8},
        {"input": ["2 * 4"], "expected": 8},
        {"input": ["(1 + 1) * 4"], "expected": 8},
    ]
}
```

**Correct Solution** (must pass ALL test cases):
```python
def evaluate(expr: str) -> int:
    return eval(expr)  # Or proper parser
```

**Wrong Solution** (passes only ONE test case):
```python
def evaluate(expr: str) -> int:
    return 8  # Fails on any other input
```

### The Key Insight

With MULTIPLE test cases per problem, the model CANNOT memorize - it MUST learn the algorithm.

---

## Comparison Table

| Aspect | Current (Broken) | Correct Design |
|--------|------------------|----------------|
| Test cases per problem | 1 | 5-10 |
| Function takes input | No (`def solve()`) | Yes (`def f(x)`) |
| Can hardcode answer | Yes | No |
| Forces algorithm learning | No | Yes |
| Measures generalization | No | Yes |

---

## Action Plan

### Phase 1: Fix Problem Generation

1. **Redesign ProceduralProblem** to have:
   - A function that takes INPUT
   - MULTIPLE test cases per problem instance
   - Test cases that share the SAME function but DIFFERENT inputs

2. **Update Generators** to:
   - Generate a function TYPE (e.g., "RPN evaluator")
   - Generate MULTIPLE input/output pairs for that function
   - Ensure test cases require algorithmic solution

### Phase 2: Fix Verification

1. **Test with ALL test cases**, not just one
2. **Reject solutions that hardcode** by testing on unseen inputs
3. **Track partial credit** (e.g., 4/5 test cases passed)

### Phase 3: Fix Training

1. **Only train on FULL solutions** - pass ALL test cases
2. **Cold start integration** - include Teacher traces that show reasoning
3. **Curriculum** - start with easy functions, progress to hard ones

### Phase 4: Correct Evaluation Metrics

1. **In-distribution**: Same function, different inputs
2. **Out-of-distribution**: New function type entirely
3. **Transfer**: Does learning RPN help with arithmetic?

---

## Expected Behavior After Fix

**Before** (current broken design):
```
Iteration 0: 54% train, 44% val, 44% test
Iteration 1: 41% train, 33% val, 22% test  # WORSE!
```

**After** (correct design):
```
Iteration 0: 20% train, 15% val, 15% test  # Harder problems!
Iteration 1: 35% train, 30% val, 28% test  # Improving!
Iteration 5: 70% train, 60% val, 55% test  # Learning algorithms!
```

The accuracy starts LOWER (because problems are harder) but IMPROVES over iterations (because the model is actually learning).

---

## Conclusion

Our current approach is fundamentally broken because:

1. **Problems are trivial** - one test case, no input arguments
2. **No generalization required** - each problem is independent
3. **Memorization works** - model can pass by returning constants

The fix requires redesigning procedural problems to:

1. **Require algorithms** - multiple test cases, function takes input
2. **Test generalization** - same function, different inputs
3. **Prevent memorization** - varied test inputs

This is not a bug in the training loop - it's a flaw in the problem design itself.
