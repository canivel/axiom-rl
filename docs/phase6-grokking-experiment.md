# Phase 6: The Grokking Experiment - Observing Generalization

## Overview

Phase 6 is the **scientific core** of axiom-rl. We use procedurally generated problems to run controlled experiments that observe the "Grokking" phenomenon - the sudden emergence of generalization after extended training.

**Key Question**: Does self-improvement via verified solutions cause faster/better generalization than standard supervised fine-tuning?

## What is Grokking?

### The Phenomenon

"Grokking" (named after Robert Heinlein's term for deep understanding) was discovered by researchers at OpenAI who observed:

1. **Phase 1 - Memorization**: Model achieves ~100% training accuracy
2. **Phase 2 - Plateau**: Validation accuracy stays low (seemingly stuck)
3. **Phase 3 - Grokking**: Suddenly, validation accuracy jumps to ~100%

```
Accuracy
100% │                      ┌────────── Training
     │    ┌─────────────────┤
     │   /                  │
     │  /                   │     ┌──── Validation (Grokking!)
 50% │ /                    │    /
     │/                     │   /
     │                      │  /
  0% └──────────────────────┴──────────────────────
     0                    Time/Steps
           Memorization      Generalization
           Phase             Phase
```

### Why Grokking Matters for Self-Improvement

For our Expert Iteration loop:
- **Memorization** = Model remembers specific solutions it generated
- **Generalization** = Model learns the *algorithm* behind the solutions

If we can observe grokking, it proves the model is learning *reasoning patterns*, not just copying.

## Experimental Design

### Hypothesis

> **H1**: Models trained via verified self-improvement (Expert Iteration) will grok faster than models trained on equivalent amounts of static supervised data.

> **H2**: Procedurally generated problems with perfect verification provide a cleaner signal for grokking than hand-crafted problems with test cases.

### Variables

**Independent Variables**:
- Training method (Expert Iteration vs. Standard SFT)
- Number of training iterations
- Problem difficulty distribution
- Dataset size

**Dependent Variables**:
- Training accuracy (on seen problems)
- Test accuracy (on held-out problems)
- Time/steps to grokking
- Final generalization performance

**Controlled Variables**:
- Base model (Qwen2.5-Coder-1.5B)
- LoRA configuration (rank=16, alpha=32)
- Problem types (arithmetic, rpn, parentheses, etc.)
- Verification method (exact match for procedural)

### Dataset Structure

```
data/
├── procedural/
│   ├── train.jsonl           # Training problems (seen during training)
│   ├── val.jsonl             # Validation (same distribution, unseen)
│   └── test.jsonl            # Test (different seeds, truly held-out)
└── experiments/
    └── grokking_v1/
        ├── config.json       # Experiment configuration
        ├── metrics.jsonl     # Per-step metrics
        ├── checkpoints/      # Model snapshots
        └── plots/            # Visualization outputs
```

### Metrics Tracked

| Metric | Description | When Measured |
|--------|-------------|---------------|
| `train_accuracy` | % of training problems solved | Every epoch |
| `val_accuracy` | % of validation problems solved | Every epoch |
| `test_accuracy` | % of test problems solved | Every N epochs |
| `loss` | Training loss | Every step |
| `generalization_gap` | train_accuracy - val_accuracy | Every epoch |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GROKKING EXPERIMENT LOOP                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐         ┌─────────────┐                       │
│   │  Procedural │         │    Model    │                       │
│   │  Generator  │────────▶│   (LoRA)    │                       │
│   └─────────────┘         └──────┬──────┘                       │
│         │                        │                               │
│         │                        ▼                               │
│         │                 ┌─────────────┐                       │
│         │                 │  Generate   │                       │
│         │                 │  Solutions  │                       │
│         │                 └──────┬──────┘                       │
│         │                        │                               │
│         │                        ▼                               │
│         │                 ┌─────────────┐                       │
│         │                 │   Verify    │                       │
│         │                 │  (Direct)   │                       │
│         │                 └──────┬──────┘                       │
│         │                        │                               │
│         │         ┌──────────────┼──────────────┐               │
│         │         ▼              ▼              ▼               │
│         │    ┌─────────┐   ┌─────────┐   ┌─────────┐           │
│         │    │ Training│   │  Val    │   │  Test   │           │
│         │    │ Metrics │   │ Metrics │   │ Metrics │           │
│         │    └────┬────┘   └────┬────┘   └────┬────┘           │
│         │         │             │             │                 │
│         │         └─────────────┼─────────────┘                 │
│         │                       ▼                               │
│         │                ┌─────────────┐                       │
│         │                │   Logger    │                       │
│         │                │  & Plots    │                       │
│         │                └─────────────┘                       │
│         │                                                       │
│         │    Only if val_accuracy improving:                   │
│         │         ┌─────────────┐                               │
│         └────────▶│    Train    │                               │
│                   │   on Gold   │                               │
│                   └─────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### Module Structure

```
axiom/experiments/
├── __init__.py
├── grokking.py           # Main experiment runner
├── metrics.py            # Metrics collection and tracking
├── evaluation.py         # Model evaluation on procedural problems
└── visualization.py      # Plotting utilities
```

### Key Components

#### 1. GrokkingExperiment

Main class orchestrating the experiment:

```python
class GrokkingExperiment:
    def __init__(
        self,
        base_model: str,
        train_problems: list[ProceduralProblem],
        val_problems: list[ProceduralProblem],
        test_problems: list[ProceduralProblem],
        experiment_name: str,
    ):
        ...

    def run(
        self,
        num_iterations: int = 100,
        eval_every: int = 5,
        checkpoint_every: int = 10,
    ):
        """Run the full grokking experiment."""
        for iteration in range(num_iterations):
            # 1. Generate solutions for training problems
            solutions = self.generate_solutions(self.train_problems)

            # 2. Verify and collect gold data
            gold_data = self.verify_solutions(solutions)

            # 3. Train on gold data
            self.train_iteration(gold_data)

            # 4. Evaluate
            if iteration % eval_every == 0:
                train_acc = self.evaluate(self.train_problems)
                val_acc = self.evaluate(self.val_problems)
                self.log_metrics(iteration, train_acc, val_acc)

            # 5. Checkpoint
            if iteration % checkpoint_every == 0:
                self.save_checkpoint(iteration)

            # 6. Check for grokking
            if self.detect_grokking():
                self.log_grokking_event(iteration)
```

#### 2. Evaluation on Procedural Problems

```python
def evaluate_model(
    model,
    tokenizer,
    problems: list[ProceduralProblem],
    num_samples: int = 1,
) -> EvaluationResult:
    """
    Evaluate model on procedural problems.

    Unlike test-case evaluation, we directly compare
    the model's output to the known ground truth.
    """
    correct = 0
    total = len(problems)

    for problem in problems:
        # Generate solution
        prompt = problem.to_prompt()
        response = generate(model, tokenizer, prompt)

        # Extract function and run it
        code = extract_code(response)
        try:
            output = execute_function(code, problem.input_data)
            if output == problem.expected_output:
                correct += 1
        except Exception:
            pass  # Execution error = incorrect

    return EvaluationResult(
        accuracy=correct / total,
        correct=correct,
        total=total,
    )
```

#### 3. Grokking Detection

```python
def detect_grokking(
    metrics_history: list[dict],
    threshold: float = 0.2,
    window: int = 10,
) -> bool:
    """
    Detect if grokking has occurred.

    Grokking signature:
    1. Training accuracy plateaued at high value
    2. Validation accuracy suddenly jumps
    3. Gap between train and val closes rapidly
    """
    if len(metrics_history) < window:
        return False

    recent = metrics_history[-window:]
    older = metrics_history[-2*window:-window] if len(metrics_history) >= 2*window else []

    if not older:
        return False

    # Check for sudden val_accuracy improvement
    old_val = np.mean([m['val_accuracy'] for m in older])
    new_val = np.mean([m['val_accuracy'] for m in recent])

    return (new_val - old_val) > threshold
```

### CLI Script

```bash
# Run grokking experiment
python scripts/run_grokking_experiment.py \
    --train-size 1000 \
    --val-size 200 \
    --test-size 200 \
    --iterations 100 \
    --eval-every 5 \
    --experiment grokking_v1

# Resume from checkpoint
python scripts/run_grokking_experiment.py \
    --resume experiments/grokking_v1/checkpoints/iter_50

# Visualize results
python scripts/plot_grokking.py --experiment grokking_v1
```

## Expected Results

### What We Hope to Observe

1. **Early Phase (Iterations 0-20)**:
   - Training accuracy rises quickly
   - Validation accuracy rises slowly or plateaus
   - Model is "memorizing"

2. **Middle Phase (Iterations 20-60)**:
   - Training accuracy ~95-100%
   - Validation accuracy stuck at 40-60%
   - Large generalization gap

3. **Grokking Phase (Iterations 60-100)**:
   - Training accuracy stable
   - Validation accuracy suddenly jumps
   - Generalization gap closes
   - Model has "grokked" the underlying algorithm

### Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                 EXPECTED GROKKING CURVE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  100% ─┬─────────────────────────────────────────────────────── │
│        │              Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│   80% ─┤         ╭━━━━━╯                                         │
│        │        ╱                                                │
│   60% ─┤      ╱                     Validation                   │
│        │     ╱                           ╭━━━━━━━━━━━━━━━━━━━━   │
│   40% ─┤   ╱                            ╱      Grokking!         │
│        │  ╱              ━━━━━━━━━━━━━╱                          │
│   20% ─┤╱                                                        │
│        │                                                         │
│    0% ─┼─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────── │
│        0    10    20    30    40    50    60    70    80   100   │
│                           Iterations                             │
│                                                                  │
│           ◄─ Memorization ─►    ◄─ Generalization ─►            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Scientific Significance

### What a Positive Result Would Mean

If we observe grokking:

1. **Validates Self-Improvement**: The model learned *algorithms*, not just *patterns*
2. **Scalable Intelligence**: More compute → better reasoning (not just more memorization)
3. **No Human Ceiling**: The system can surpass its training data

### Comparison to Prior Work

| Study | Domain | Grokking Observed? | Iterations |
|-------|--------|-------------------|------------|
| Power et al. (OpenAI) | Modular arithmetic | Yes | ~100k steps |
| This work | Code generation | TBD | ~100 iterations |

Our key innovation: Using **code verification** as the reward signal instead of just loss.

## Configuration

### Default Experiment Parameters

```python
GROKKING_CONFIG = {
    # Data
    "train_size": 1000,
    "val_size": 200,
    "test_size": 200,
    "problem_types": ["arithmetic", "rpn", "parentheses"],
    "difficulty_range": (3, 7),

    # Model
    "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "lora_r": 16,
    "lora_alpha": 32,

    # Training
    "iterations": 100,
    "samples_per_problem": 4,
    "batch_size": 1,
    "gradient_accumulation": 8,
    "learning_rate": 2e-4,

    # Evaluation
    "eval_every": 5,
    "checkpoint_every": 10,
    "early_stopping_patience": 20,

    # Grokking detection
    "grokking_threshold": 0.2,
    "grokking_window": 10,
}
```

## Next Steps After Phase 6

If grokking is observed:

1. **Phase 7: Replay Buffer** - Prevent catastrophic forgetting during extended training
2. **Phase 8: Curriculum Learning** - Progressively increase difficulty
3. **Phase 9: Multi-Task Grokking** - Does grokking on one problem type transfer to others?

If grokking is NOT observed:

1. **Increase scale** - More iterations, more data
2. **Adjust hyperparameters** - Learning rate, LoRA rank
3. **Try different problem types** - Some may grok easier than others
4. **Add regularization** - Weight decay, dropout

## Files to Create

| File | Purpose |
|------|---------|
| `axiom/experiments/__init__.py` | Module exports |
| `axiom/experiments/grokking.py` | Main experiment class |
| `axiom/experiments/metrics.py` | Metrics tracking |
| `axiom/experiments/evaluation.py` | Model evaluation |
| `axiom/experiments/visualization.py` | Plotting utilities |
| `scripts/run_grokking_experiment.py` | CLI entry point |
| `scripts/plot_grokking.py` | Visualization script |
