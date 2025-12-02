# Phase 5: Procedural Generation - The "Infinite" Dataset

## Overview

Phase 5 implements **procedural problem generation** - the ability to create infinite unique training problems with perfect ground truth verification. Unlike hand-crafted problems with limited test cases, procedural problems are algorithmically generated with mathematically provable correct answers.

This is a critical component for self-improvement at scale. When models get better at solving problems, they "exhaust" the training data (memorization). Procedural generation provides an inexhaustible supply of fresh challenges.

## Why Procedural Generation?

### The Data Exhaustion Problem

Traditional ML training has a fixed dataset. For self-improvement loops:

1. **Model improves** → Solves more problems
2. **Data becomes stale** → Model has seen all problems multiple times
3. **Memorization** → Model memorizes patterns instead of learning algorithms
4. **Plateau** → No more improvement

### The Solution: Infinite Fresh Problems

Procedural generation solves this by:

1. **Algorithmic generation** → Infinite unique instances
2. **Perfect verification** → No annotation needed, ground truth is computed
3. **Difficulty scaling** → Generate problems at any difficulty level
4. **Diversity** → Multiple problem types prevent overfitting

### Research Foundation

This approach is inspired by:
- **AlphaGo/AlphaZero**: Self-play generates infinite training games
- **Curriculum Learning**: Start easy, increase difficulty as model improves
- **Synthetic Data**: Used in math reasoning (GSM8K augmentation), code generation, etc.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 PROCEDURAL GENERATION SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │  Arithmetic  │   │    RPN      │   │ Parentheses │           │
│  │  Generator   │   │  Generator  │   │  Generator  │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                    │
│         └────────────┬────┴─────────────────┘                   │
│                      │                                           │
│                      ▼                                           │
│            ┌─────────────────┐                                  │
│            │ ProceduralData  │                                  │
│            │     Stream      │                                  │
│            └────────┬────────┘                                  │
│                     │                                           │
│         ┌───────────┼───────────┐                              │
│         ▼           ▼           ▼                              │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│    │Training │ │Evaluation│ │  Live   │                        │
│    │  Data   │ │Benchmark │ │Training │                        │
│    └─────────┘ └─────────┘ └─────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### Module Structure

```
axiom/procedural/
├── __init__.py              # Module exports & registry
├── base.py                  # Base classes
├── arithmetic.py            # Arithmetic expression evaluator
├── rpn.py                   # Reverse Polish Notation evaluator
├── parentheses.py           # Bracket matching validator
├── list_ops.py              # List operations (sort, filter, aggregate)
└── trainer_integration.py   # Training pipeline integration
```

### Base Classes

#### ProceduralProblem

A single generated problem instance:

```python
@dataclass
class ProceduralProblem:
    problem_type: str       # "arithmetic", "rpn", etc.
    problem_id: str         # Unique ID (e.g., "arithmetic_42")
    title: str              # Human-readable title
    description: str        # Problem description
    function_signature: str # The function to implement
    input_data: Any         # The specific input
    expected_output: Any    # The correct output (ground truth)
    difficulty: int         # 1-10 scale
    complexity: str         # "easy", "medium", "hard"
```

#### ProceduralGenerator

Abstract base class for generators:

```python
class ProceduralGenerator(ABC):
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    @abstractmethod
    def generate_instance(self, difficulty: int) -> tuple[Any, Any]:
        """Generate (input, expected_output) pair."""
        pass

    @abstractmethod
    def verify(self, input_data, output, expected) -> bool:
        """Check if output matches expected."""
        pass

    def generate(self, difficulty: int) -> ProceduralProblem:
        """Generate complete problem instance."""
        pass

    def generate_batch(self, count: int, ...) -> list[ProceduralProblem]:
        """Generate multiple problems."""
        pass

    def generate_infinite(self, ...) -> Iterator[ProceduralProblem]:
        """Generate problems forever."""
        pass
```

### Available Generators

#### 1. Arithmetic Expression Evaluator

**Problem**: Given a string like `"2 + 3 * 4"`, return `14`.

**Skills tested**:
- Operator precedence
- Parentheses handling
- Integer division

**Difficulty scaling**:
| Difficulty | Operands | Operators | Parentheses | Max Number |
|------------|----------|-----------|-------------|------------|
| 1-2        | 2        | +, -      | No          | 10         |
| 3-4        | 3-4      | +, -, *   | No          | 50         |
| 5-6        | 4-5      | All       | Maybe       | 100        |
| 7-8        | 5-7      | All       | Yes         | 100        |
| 9-10       | 7-10     | All       | Nested      | 200        |

**Example**:
```python
gen = ArithmeticGenerator(seed=42)
problem = gen.generate(difficulty=5)
# Input: "23 + 45 * 2 - 10"
# Expected: 103
```

#### 2. RPN (Reverse Polish Notation) Evaluator

**Problem**: Given `"3 4 + 2 *"`, return `14`.

**Skills tested**:
- Stack-based evaluation
- Token parsing
- Order of operations in RPN

**Example**:
```python
gen = RPNGenerator(seed=42)
problem = gen.generate(difficulty=6)
# Input: "5 3 + 8 2 - * 4 //"
# Expected: 12
```

#### 3. Parentheses Validator

**Problem**: Given `"({[]})"`, return `True`. Given `"({[}])"`, return `False`.

**Skills tested**:
- Stack-based matching
- Multiple bracket types
- Edge case handling

**Difficulty scaling**:
- Easy: Single bracket type, short strings
- Medium: Two bracket types, medium length
- Hard: All bracket types, long strings, tricky patterns

**Example**:
```python
gen = ParenthesesGenerator(seed=42)
problem = gen.generate(difficulty=7)
# Input: "{[()]}{[]}"
# Expected: True
```

#### 4. List Operations (Sort, Filter, Aggregate)

**Problem types**:

**List Sort**:
```python
# Input: {"nums": [3, -1, 4, -1, 5], "criterion": "absolute"}
# Expected: [-1, -1, 3, 4, 5]
```

**List Filter**:
```python
# Input: {"nums": [1, 2, 3, 4, 5, 6], "condition": "even", "param": 0}
# Expected: [2, 4, 6]
```

**List Aggregate**:
```python
# Input: {"nums": [1, 2, 3, 4, 5], "operation": "second_max", "param": 2}
# Expected: 4
```

### Training Integration

#### ProceduralDataStream

Infinite iterator for online training:

```python
from axiom.procedural import ProceduralDataStream

# Create stream for all problem types
stream = ProceduralDataStream(
    problem_types=None,  # All types
    min_difficulty=1,
    max_difficulty=10,
    seed=42,
)

# Generate problems on-the-fly
for sample in stream:
    prompt = sample.to_prompt()
    # ... train on sample
```

#### Batch Generation

```python
from axiom.procedural import generate_training_file

# Generate 10,000 problems to JSONL file
generate_training_file(
    output_path=Path("data/procedural/train.jsonl"),
    count=10000,
    min_difficulty=3,
    max_difficulty=8,
    seed=42,
)
```

## CLI Usage

### Generate Problems

```bash
# Generate 100 problems across all types
python scripts/generate_procedural.py --count 100

# Generate specific types
python scripts/generate_procedural.py --types arithmetic rpn --count 50

# Control difficulty
python scripts/generate_procedural.py --min-difficulty 5 --max-difficulty 10

# Save to file
python scripts/generate_procedural.py --output data/procedural/train.jsonl --count 1000

# Reproducible generation
python scripts/generate_procedural.py --seed 42 --count 100
```

### List Available Types

```bash
python scripts/generate_procedural.py --list-types
```

Output:
```
Available problem types:

  arithmetic:
    Title: Evaluate Arithmetic Expression
    Signature: def evaluate_expression(expr: str) -> int:

  rpn:
    Title: Evaluate RPN Expression
    Signature: def evaluate_rpn(expression: str) -> int:

  parentheses:
    Title: Valid Parentheses
    Signature: def is_valid_parentheses(s: str) -> bool:

  list_sort:
    Title: Custom List Sort
    Signature: def custom_sort(nums: list[int], criterion: str) -> list[int]:

  list_filter:
    Title: Filter List
    Signature: def filter_list(nums: list[int], condition: str, param: int) -> list[int]:

  list_aggregate:
    Title: List Aggregation
    Signature: def aggregate(nums: list[int], operation: str, param: int) -> int:
```

## Integration with Expert Iteration Loop

### How It Fits

```
┌───────────────────────────────────────────────────────────────┐
│                    EXPERT ITERATION LOOP                       │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Generate Problems                                          │
│     ├─ Hand-crafted (problems.json) - for benchmarking        │
│     └─ Procedural (infinite) - for training                   │
│                                                                │
│  2. Model Generates Solutions                                  │
│     └─ Current model attempts all problems                    │
│                                                                │
│  3. Verify Solutions                                           │
│     ├─ Hand-crafted: TestHarness runs test cases              │
│     └─ Procedural: Direct comparison with ground truth        │
│                                                                │
│  4. Collect Successful Solutions                               │
│     └─ Only verified-correct solutions go to training         │
│                                                                │
│  5. Train on Successes                                         │
│     └─ LoRA fine-tuning on verified solutions                 │
│                                                                │
│  6. Repeat with improved model                                 │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Curriculum Learning Strategy

As the model improves:

1. **Phase 1**: Easy problems (difficulty 1-3)
   - Build foundational understanding
   - High success rate → more training data

2. **Phase 2**: Medium problems (difficulty 4-6)
   - Challenge the model moderately
   - Filter for truly learned solutions

3. **Phase 3**: Hard problems (difficulty 7-10)
   - Push model limits
   - Identify capability boundaries

### Preventing Memorization

Key properties that prevent memorization:

1. **Infinite unique instances**: Same problem type, different numbers
2. **Varied difficulty**: Different structural complexity
3. **Multiple problem types**: Can't specialize too narrowly
4. **Random seeds**: Different generation each time

## Properties of Good Procedural Problems

### What Makes a Good Procedural Problem?

1. **Perfect verification**: Output can be computed algorithmically
2. **Scalable difficulty**: Parameters control complexity
3. **Clear algorithm**: Well-defined solution strategy
4. **Generalizable skills**: Transferable to other problems

### Why These Specific Problems?

| Problem Type | Key Skill | Why It Matters |
|--------------|-----------|----------------|
| Arithmetic | Parsing, Precedence | Foundation of expression handling |
| RPN | Stack operations | Core data structure understanding |
| Parentheses | Matching, Validation | Pattern recognition, stack mastery |
| List Ops | Iteration, Filtering | Data manipulation fundamentals |

## Extending with New Generators

### Adding a New Problem Type

1. Create a new file in `axiom/procedural/`:

```python
# axiom/procedural/my_generator.py
from .base import ProceduralGenerator

class MyGenerator(ProceduralGenerator):
    @property
    def problem_type(self) -> str:
        return "my_problem"

    @property
    def title(self) -> str:
        return "My Problem Type"

    @property
    def description_template(self) -> str:
        return """Description of the problem..."""

    @property
    def function_signature(self) -> str:
        return "def solve_my_problem(input: str) -> int:"

    def generate_instance(self, difficulty: int) -> tuple[str, int]:
        # Generate input and expected output
        input_data = ...
        expected = ...
        return input_data, expected

    def verify(self, input_data, output, expected) -> bool:
        return output == expected
```

2. Register in `__init__.py`:

```python
from .my_generator import MyGenerator

GENERATORS["my_problem"] = MyGenerator
```

## Files Created

| File | Description |
|------|-------------|
| `axiom/procedural/__init__.py` | Module exports and registry |
| `axiom/procedural/base.py` | Base classes |
| `axiom/procedural/arithmetic.py` | Arithmetic expression generator |
| `axiom/procedural/rpn.py` | RPN expression generator |
| `axiom/procedural/parentheses.py` | Bracket validation generator |
| `axiom/procedural/list_ops.py` | List operation generators |
| `axiom/procedural/trainer_integration.py` | Training pipeline integration |
| `scripts/generate_procedural.py` | CLI for problem generation |

## Example Generated Problems

### Arithmetic (Difficulty 7)
```
Input: "(45 + 23) * 3 - 100 // 4"
Expected: 179
```

### RPN (Difficulty 5)
```
Input: "10 5 3 + * 2 //"
Expected: 40
```

### Parentheses (Difficulty 6)
```
Input: "{[()()]}{}"
Expected: True
```

### List Sort (Difficulty 4)
```
Input: {"nums": [15, -8, 23, -3, 12], "criterion": "absolute"}
Expected: [-3, -8, 12, 15, 23]
```

## Next Steps

After implementing procedural generation:

1. **Phase 6: Reinforcement Learning**
   - Use procedural problems as environment
   - Reward based on verification
   - Policy gradient optimization

2. **Phase 7: Multi-Task Learning**
   - Train on all problem types simultaneously
   - Investigate transfer between problem types
   - Measure generalization

3. **Future Enhancements**
   - More problem types (graph algorithms, dynamic programming)
   - Adaptive difficulty based on model performance
   - Problem difficulty estimation from model behavior
