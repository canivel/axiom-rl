# Experiment 08: Curriculum Learning for GRPO

## Overview

This experiment implements **curriculum learning** for GRPO training - a technique where the model progressively trains on harder problems as it masters easier ones. This is inspired by human learning and has been shown to improve RL training efficiency.

## Hypothesis

Curriculum learning will:
1. **Faster convergence**: Starting easy gives quick rewards, building momentum
2. **Better generalization**: Gradual difficulty prevents overfitting to hard patterns
3. **Stable training**: Avoids the "reward desert" of starting with too-hard problems
4. **Efficient exploration**: Time spent in "optimal learning zone" (50-80% success)

## Background: Why Curriculum Learning?

### The Problem with Random Sampling

In standard RL training:
- Random difficulty sampling often gives 0 reward on hard problems
- Model learns nothing from complete failures
- Training signal is sparse and noisy

### The Curriculum Solution

```
Traditional RL:         Curriculum RL:

Reward                  Reward
  ^                       ^
  |   ~~~                 |        /~~~
  |  /   \                |      /
  | /     ~~~/\           |    /
  |/          \ /         |  /
  +------------>          +------------>
       Steps                   Steps

  Sparse, noisy           Smooth, progressive
```

## Curriculum Strategies Implemented

### 1. LINEAR Strategy
Fixed schedule through difficulty levels:
```
Steps  1-20:  Easy (difficulty 1-3)
Steps 21-40:  Medium (difficulty 4-6)
Steps 41-60:  Hard (difficulty 7-10)
```

**Pros**: Simple, predictable
**Cons**: Doesn't adapt to model capability

### 2. ADAPTIVE Strategy (Default)
Advances/regresses based on performance:
```python
if avg_success(last_10_steps) >= 0.7:
    advance_level()
elif avg_success(last_10_steps) < 0.3:
    regress_level()
```

**Pros**: Adapts to model capability
**Cons**: More hyperparameters to tune

### 3. SELF_PACED Strategy
Immediate difficulty adjustment:
```python
if step_success > 0.85:  # Too easy
    advance_level()
elif step_success < 0.3:  # Too hard
    regress_level()
```

**Pros**: Very responsive
**Cons**: Can oscillate between levels

## Configuration

### Difficulty Levels

| Level | Difficulty Range | Target Success | Min Samples |
|-------|------------------|----------------|-------------|
| Easy | 1-3 | 80% | 3 |
| Medium | 4-6 | 70% | 5 |
| Hard | 7-10 | 60% | 7 |

### GRPO Settings
| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen/Qwen2.5-Coder-0.5B-Instruct | Memory-efficient |
| Generations (G) | 2 | RTX 3080 constraint |
| Learning Rate | 2e-5 | Conservative |
| KL Beta | 0.04 | Standard penalty |
| Max Seq Length | 512 | Token limit |

### Commands

```bash
# Quick test (5 steps)
uv run python scripts/run_grpo_curriculum.py --steps 5

# Adaptive curriculum on specific problems
uv run python scripts/run_grpo_curriculum.py \
    --strategy adaptive \
    --problems parentheses two_sum \
    --steps 30

# Linear curriculum, full training
uv run python scripts/run_grpo_curriculum.py \
    --strategy linear \
    --steps 60

# Self-paced with more generations
uv run python scripts/run_grpo_curriculum.py \
    --strategy self_paced \
    --generations 4 \
    --steps 40
```

## Architecture

### Curriculum Scheduler

```
┌─────────────────────────────────────────────────────────┐
│                  CurriculumScheduler                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  get_problem()                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Select   │───►│ Select   │───►│ Generate │──► Problem│
│  │ Generator│    │Difficulty│    │ Problem  │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│       ▲               ▲                                  │
│       │               │                                  │
│       └───────────────┴── Based on current_level        │
│                                                          │
│  record_result(success_rate)                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Update   │───►│ Check    │───►│ Advance/ │          │
│  │ History  │    │ Strategy │    │ Regress  │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Integration with GRPO

```
for step in range(max_steps):
    # 1. Curriculum selects problem
    problem, type, difficulty = curriculum.get_problem()

    # 2. Create reward function for this problem
    reward_fn = create_reward_function(problem)

    # 3. Run GRPO step
    loss, avg_reward = grpo_trainer.step(problem)

    # 4. Update curriculum based on result
    curriculum.record_result(avg_reward)

    # 5. Curriculum may advance/regress level
```

## Expected Results

### Success Metrics

| Metric | Target | Reasoning |
|--------|--------|-----------|
| Easy Level Mastery | >80% | Should quickly pass |
| Medium Level Progress | >60% | Steady improvement |
| Hard Level Attempts | >30% | Some success expected |
| Level Progression | Reach Hard | Full curriculum traversal |
| No Collapse | Loss stable | Training doesn't diverge |

### Comparison: With vs Without Curriculum

| Metric | No Curriculum | With Curriculum |
|--------|---------------|-----------------|
| Steps to 50% success | ~40 | ~15 (expected) |
| Final success rate | ~40% | ~60% (expected) |
| Training stability | Variable | Smooth |
| Hard problem success | ~10% | ~30% (expected) |

## Implementation Details

### Problem Selection

```python
def get_problem(self):
    # Random generator from active set
    gen_name = random.choice(list(self.active_generators.keys()))
    generator = self.active_generators[gen_name]

    # Difficulty from current level range
    min_diff, max_diff = self.current_level.difficulty_range
    difficulty = random.randint(min_diff, max_diff)

    # Generate fresh problem instance
    return generator.generate(difficulty=difficulty, num_test_cases=5)
```

### Level Advancement Logic

```python
def _update_adaptive(self):
    if len(self.success_history) < self.success_window:
        return  # Not enough data

    recent = self.success_history[-self.success_window:]
    avg_success = sum(recent) / len(recent)

    if avg_success >= self.advancement_threshold:
        if self.level_step_count >= self.current_level.min_samples:
            self._advance_level()

    elif avg_success < self.regression_threshold:
        if self.current_level_idx > 0:
            self._regress_level()
```

### Reward Shaping

```python
def reward_function(completions):
    for completion in completions:
        code = extract_code(completion)
        success, partial = verify_solution(code, problem)

        if success:
            reward = 1.0          # Full credit
        else:
            reward = partial * 0.5  # Partial credit, capped

    return rewards
```

## Files

- `axiom/trainer/curriculum.py` - Curriculum scheduler implementation
- `scripts/run_grpo_curriculum.py` - Training script
- `curriculum_metrics.json` - Training metrics (after run)
- `final_model/` - Saved LoRA adapter (after run)

## Related Experiments

- **Experiment 06**: SFT Baseline (starting point for GRPO)
- **Experiment 07**: GRPO Validation (training loop verification)
- **Experiment 09**: Extended Training (longer curriculum runs)

## References

1. Bengio et al. "Curriculum Learning" (2009) - Original curriculum learning paper
2. DeepMind's AlphaStar - Progressive training league
3. OpenAI's RLHF - Stage-wise difficulty in human feedback
