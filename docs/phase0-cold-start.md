# Phase 0: Cold Start - Implementation Details

## Overview

Phase 0 addresses a critical stability issue in self-improvement loops: **models need to learn the format of reasoning before being rewarded for reasoning correctly.**

Without a "cold start" phase, models trained via RL/self-play can:
- Collapse into gibberish
- Get stuck in local minima
- Produce inconsistent output formats

## Research Background

This approach is inspired by:
- **DeepSeek-R1**: Demonstrated that cold-starting with supervised data stabilizes RL training
- **STaR (Self-Taught Reasoner)**: Showed that bootstrapping reasoning improves performance
- **Expert Iteration**: The general framework of using a "teacher" to guide learning

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COLD START PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Problem (from problems.json)                                   │
│       │                                                         │
│       ▼                                                         │
│  Gemini 2.5 Flash (Teacher Model)                              │
│       │                                                         │
│       ▼                                                         │
│  Response: <think>reasoning</think> + ```python```code```       │
│       │                                                         │
│       ▼                                                         │
│  Our Verifier (Python Sandbox)                                  │
│       │                                                         │
│       ├──► PASS: Save to teacher_traces.jsonl                  │
│       └──► FAIL: Discard                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### New Module: `axiom/coldstart/`

```
axiom/coldstart/
├── __init__.py           # Module exports
├── gemini_client.py      # Gemini API wrapper
└── trace_generator.py    # Generates and verifies reasoning traces
```

### Key Components

#### 1. GeminiClient (`gemini_client.py`)

Wrapper for Gemini 2.5 API that:
- Sends problems to Gemini with a structured prompt
- Requests reasoning inside `<think>` tags
- Parses the response to extract thinking and code separately

```python
class GeminiClient:
    def generate_reasoning_trace(
        self,
        problem_title: str,
        problem_description: str,
        function_signature: str,
    ) -> GeminiResponse:
        # Returns: thinking (str), code (str), raw_response (str)
```

**Prompt Template:**
```
You are an expert Python programmer solving algorithmic problems.

Your task is to solve the problem below. You MUST:
1. First, show your reasoning process inside <think> tags
2. Then, provide the complete Python code inside ```python``` blocks

## Required Format

<think>
Step 1: [Understand the problem - what are the inputs/outputs?]
Step 2: [Consider edge cases]
Step 3: [Plan the approach - what algorithm/data structure?]
Step 4: [Analyze time/space complexity]
Step 5: [Implement the solution]
</think>

```python
# Your complete implementation here
```
```

#### 2. ReasoningTraceGenerator (`trace_generator.py`)

Orchestrates the generation process:
- Iterates through problems
- Calls Gemini for each problem (multiple times for diversity)
- Verifies generated code using our existing test harness
- Only saves traces where code passes all tests

```python
@dataclass
class ReasoningTrace:
    problem_id: str
    problem_title: str
    problem_description: str
    function_signature: str
    thinking: str          # The reasoning process
    solution_code: str     # The verified code
    teacher_model: str     # "gemini-2.5-flash"
    verified: bool         # Always True (we only save verified)
    passed_tests: int
    total_tests: int
    timestamp: str
```

#### 3. CLI Script (`scripts/generate_teacher_data.py`)

```bash
# Generate with defaults (10 traces per problem)
python scripts/generate_teacher_data.py

# Generate for specific problems
python scripts/generate_teacher_data.py --problems two_sum fizzbuzz

# Generate more traces per problem
python scripts/generate_teacher_data.py --traces-per-problem 50

# Use a different output file
python scripts/generate_teacher_data.py --output teacher_v2.jsonl
```

### Training Data Format

The cold start data includes a `thinking` field that standard solution data doesn't have:

**Cold Start Format (`teacher_traces.jsonl`):**
```json
{
  "problem_id": "two_sum",
  "problem_title": "Two Sum",
  "thinking": "Step 1: I need to find two numbers...\nStep 2: A hash map gives O(n)...",
  "solution_code": "def two_sum(nums, target):\n    seen = {}\n    ...",
  "teacher_model": "gemini-2.5-flash",
  "verified": true
}
```

**Standard Format (`solutions.jsonl`):**
```json
{
  "problem_id": "two_sum",
  "problem_title": "Two Sum",
  "solution_code": "def two_sum(nums, target):\n    seen = {}\n    ...",
  "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct"
}
```

### Trainer Modifications

The trainer (`axiom/trainer/data.py`) was updated to handle both formats:

1. **TrainingSample** now has optional `thinking` field
2. **to_prompt_completion()** formats output differently based on whether thinking exists
3. **System prompt** changes to teach `<think>` tag usage when training on cold start data

**Without thinking (standard):**
```
Assistant: ```python
def two_sum(nums, target):
    ...
```

**With thinking (cold start):**
```
Assistant: <think>
Step 1: I need to find two numbers that sum to target
Step 2: A hash map gives O(n) lookup
...
</think>

```python
def two_sum(nums, target):
    ...
```

## Usage

### Step 1: Generate Teacher Data

```bash
# Ensure GEMINI_API_KEY is set in .env file
python scripts/generate_teacher_data.py --traces-per-problem 5
```

Output: `data/coldstart/teacher_traces.jsonl`

### Step 2: Train Cold Start Model

```bash
python scripts/run_training.py \
    --solutions data/coldstart/teacher_traces.jsonl \
    --experiment coldstart_v1
```

Output: `models/lora-sft-coldstart_v1/`

### Step 3: Use Cold-Started Model for Self-Improvement

```bash
python scripts/run_pipeline.py \
    --model models/lora-sft-coldstart_v1 \
    --experiment self_improve_v1
```

## Why This Matters

### The Format Learning Problem

When we train a model to self-improve, we're rewarding it for producing correct code. But if the model doesn't know HOW to reason, it's just pattern matching.

**Without Cold Start:**
- Model sees: problem → code
- Model learns: "Given this problem shape, output this code shape"
- No transferable reasoning skills

**With Cold Start:**
- Model sees: problem → thinking → code
- Model learns: "First analyze, then plan, then implement"
- Reasoning process is explicit and learnable

### The Stability Problem

RL training is notoriously unstable. Starting from a model that:
- Knows the output format (`<think>` + code)
- Has seen examples of good reasoning
- Produces consistent, parseable outputs

...is much more stable than starting from a generic model.

## Experiment Results

### Initial Generation (2024-12-01)

| Metric | Value |
|--------|-------|
| Teacher Model | Gemini 2.5 Flash |
| Problems | 10 |
| Traces per Problem | ~4 |
| Total Traces | 39 |
| Verification Rate | 100% |

### Quality Observations

Gemini's reasoning traces include:
- Clear step-by-step problem analysis
- Edge case consideration
- Algorithm selection with justification
- Time/space complexity analysis
- Implementation walkthrough

Example thinking trace (truncated):
```
Step 1: Understand the problem - what are the inputs/outputs?
The problem asks us to find two distinct elements in a given list
of integers `nums` that sum up to a specific `target` integer.
We need to return the *indices* of these two numbers.

Step 2: Consider edge cases
- Numbers can be positive, negative, or zero
- Duplicate numbers are possible
- The problem guarantees exactly one solution

Step 3: Plan the approach
A hash map gives O(n) lookup. For each number, we check if
(target - num) exists in our map...

Step 4: Analyze time/space complexity
Time: O(n) - single pass through array
Space: O(n) - hash map storage
```

## Dependencies Added

```toml
# pyproject.toml
dependencies = [
    ...
    "google-generativeai>=0.8.0",
    "python-dotenv>=1.0.0",
]
```

## Files Created/Modified

### New Files
- `axiom/coldstart/__init__.py`
- `axiom/coldstart/gemini_client.py`
- `axiom/coldstart/trace_generator.py`
- `scripts/generate_teacher_data.py`
- `docs/phase0-cold-start.md`

### Modified Files
- `axiom/trainer/data.py` - Added `thinking` field support
- `pyproject.toml` - Added dependencies

## Next Steps

After Cold Start training:

1. **Run self-improvement loop** with the cold-started model
2. **Verify the model outputs `<think>` tags** in its responses
3. **Compare performance** against non-cold-started baseline
4. **Proceed to Phase 5** (Procedural Generation) for infinite training data
