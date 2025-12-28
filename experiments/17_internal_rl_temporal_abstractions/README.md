# Experiment 17: Internal RL with Temporal Abstractions

## Executive Summary

This experiment implements the key insights from **"Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning"** (Kobayashi et al., Google, Dec 2025) to improve code generation with Qwen 0.5B.

**Core Insight**: Instead of exploring token-by-token (which is inefficient for sparse rewards), we can discover and leverage *temporally-abstract actions* - reusable subroutines that span multiple tokens and correspond to meaningful programming concepts.

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [Key Concepts Explained](#2-key-concepts-explained)
3. [The Paper's Solution](#3-the-papers-solution)
4. [Mathematical Foundation](#4-mathematical-foundation)
5. [Our Adaptation for Code Generation](#5-our-adaptation-for-code-generation)
6. [Implementation Plan](#6-implementation-plan)
7. [Expected Results](#7-expected-results)

---

## 1. The Problem We're Solving

### Why Token-by-Token RL is Inefficient

In our previous experiments (15-16), we used GRPO/M-GRPO which works like this:

```
Token 1 → Token 2 → Token 3 → ... → Token N → REWARD (pass/fail)
   ↓         ↓         ↓              ↓
 Sample   Sample   Sample          Sample
```

**The problem**: If generating correct code requires 100 tokens, and we only get a reward at the end:
- Each token is a random variable
- The chance of randomly hitting a correct sequence is astronomically low (~1 in millions)
- Credit assignment is nearly impossible (which token caused failure?)

### Real Example from Our Experiments

In Experiment 15, we saw:
- 99% training success on simple patterns
- 10% final evaluation accuracy
- Why? The model learned superficial patterns but couldn't explore the space of *meaningful* code structures

### The Hierarchical Nature of Code

Code is naturally hierarchical:

```python
# Level 3: Function (abstract goal)
def fibonacci(n):
    # Level 2: Logic block (subroutine)
    if n <= 1:
        return n
    # Level 2: Another logic block
    return fibonacci(n-1) + fibonacci(n-2)
    # Level 1: Individual tokens
```

**Key insight**: We should explore at the level of "logic blocks" (Level 2), not individual tokens (Level 1).

---

## 2. Key Concepts Explained

### 2.1 Temporal Abstraction

**Definition**: A temporally-abstract action is a subroutine that:
1. Spans multiple time steps (tokens)
2. Achieves a meaningful sub-goal
3. Has a learned termination condition

**Analogy**: Think of typing. You don't think "press k, press e, press y, press w, press o, press r, press d". You think "type keyword". The letters are low-level actions; "type keyword" is a temporally-abstract action.

### 2.2 The Residual Stream

In transformers, information flows through a "residual stream" - the hidden states that get modified by each layer:

```
Input Embedding
      ↓
   [Layer 1] ──┐
      ↓       │
   e_{1,l}  ←─┘  (residual stream after layer l)
      ↓
   [Layer 2] ──┐
      ↓       │
   e_{2,l}  ←─┘
      ...
      ↓
   Output
```

**Key finding**: The residual stream at mid-depth contains linearly-decodable representations of abstract goals!

### 2.3 Internal Controllers

Instead of modifying the *output* (tokens), we modify the *internal state* (residual stream):

```
Standard RL:     Observation → Model → [Sample Token] → Environment
                                          ↑
                                    Noise here (high variance)

Internal RL:     Observation → [Controller] → Model → Token → Environment
                                   ↑
                             Noise here (low variance, abstract level)
```

The controller applies a linear transformation to the residual stream:
```
e_{t,l} ← e_{t,l} + U_t * e_{t,l}
```

Where `U_t` is a low-rank matrix that "steers" the model toward a specific abstract goal.

### 2.4 The Metacontroller

The metacontroller is a neural network that:
1. **Reads** the residual stream activations
2. **Generates** controller codes `z_t` (latent abstract actions)
3. **Decides** when to switch to a new abstract action (via switching gate `β_t`)
4. **Produces** the controller matrix `U_t` from the code

```
┌─────────────────────────────────────────────────────────────┐
│                     METACONTROLLER                          │
│                                                             │
│  e_{1:T} → [Sequence Embedder] → s(e_{1:T})                │
│                                      ↓                      │
│  e_t, h_{t-1} → [Controller Encoder] → μ_t, Σ_t            │
│                        ↓                                    │
│                   z̃_t ~ N(μ_t, Σ_t)  (latent proposal)     │
│                        ↓                                    │
│  e_t, h_{t-1}, z_{t-1} → [Switching Unit] → β_t ∈ [0,1]    │
│                        ↓                                    │
│         z_t = β_t * z̃_t + (1-β_t) * z_{t-1}                │
│                        ↓                                    │
│              [Controller Decoder] → U_t                     │
│                        ↓                                    │
│              e_{t,l} ← e_{t,l} + U_t * e_{t,l}              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. The Paper's Solution

### 3.1 Three-Phase Training

**Phase 1: Pretrain the Base Model**
- Train autoregressive model on expert trajectories
- Model learns to predict next actions given observations
- Internally, it develops representations of abstract goals

**Phase 2: Train the Metacontroller (Self-Supervised)**
- Freeze the base model
- Train metacontroller to generate controllers that improve action prediction
- The metacontroller discovers temporally-abstract actions *without labels*
- Key: Use future-conditioning (acausal) during training

**Phase 3: Internal RL (Post-Training)**
- Freeze base model and most of metacontroller
- Train only the causal abstract action policy
- RL happens in the *abstract action space* (z), not token space

### 3.2 Why This Works

1. **Reduced Action Space**: Instead of exploring over vocabulary (50K+ tokens), explore over abstract actions (8-16 dimensions)

2. **Temporal Compression**: One abstract action spans many tokens, reducing effective horizon from 100s to ~4-10

3. **Better Credit Assignment**: Reward is attributed to abstract actions, not individual tokens

4. **Compositional Generalization**: Abstract actions can be recombined in new ways

### 3.3 The Critical Design Choices

1. **Frozen Base Model**: The paper shows that co-training the base model with the metacontroller leads to degenerate solutions. The pretrained model provides the structure.

2. **Future-Conditioning**: During metacontroller training, it sees the whole sequence. This allows it to learn what abstract actions are needed *before* they're executed.

3. **Soft Switching**: The switching gate β_t is learned, not hard-coded. It naturally learns to switch when abstract goals change.

4. **Low-Rank Controllers**: U_t is a low-rank matrix (like LoRA), making it efficient and preventing overfitting.

---

## 4. Mathematical Foundation

### 4.1 The Evidence Lower Bound (ELBO)

The metacontroller is trained to maximize the ELBO:

```
log p(a_{1:T} | e_{1:T}) ≥ ELBO

ELBO = Σ_t [ log p(a_t | z_t, e_{1:t})           # Reconstruction term
           - α * D_KL(q(z_t|...) || p(z_t|...))  # Regularization term
           ]
```

**In plain English**:
- **Reconstruction term**: The controller should help predict the correct action
- **Regularization term**: The latent codes should be close to a simple prior (standard normal)

The hyperparameter α controls the rate-distortion trade-off:
- Low α → Better reconstruction, but codes may be unstructured
- High α → Structured codes, but may lose information

### 4.2 Temporal Integration

The key equation for temporal abstraction:

```
z_t = β_t ⊙ z̃_t + (1 - β_t) ⊙ z_{t-1}
```

Where:
- `z̃_t` is the new latent proposal (sampled from encoder)
- `z_{t-1}` is the previous latent code
- `β_t ∈ [0, 1]` is the switching gate
- `⊙` is element-wise multiplication

**When β_t ≈ 0**: Keep the same abstract action (z_t ≈ z_{t-1})
**When β_t ≈ 1**: Switch to a new abstract action (z_t ≈ z̃_t)

### 4.3 The Controller Equation

The residual stream modification:

```
ê_{t,l} = e_{t,l} + U_t * e_{t,l}
        = (I + U_t) * e_{t,l}
```

Where U_t is produced by a hypernetwork:
```
U_t = f_hyp(z_t)
```

This is like a dynamic, input-dependent LoRA adapter!

### 4.4 Why Internal RL Has Lower Variance

**Standard RL policy gradient** (token-level):
```
∇ = E[ r_T * Σ_t ∇_θ log π(a_t | s_t) ]

Variance ∝ T * |A|  (T = sequence length, |A| = action space size)
```

**Internal RL policy gradient** (abstract-level):
```
∇ = E[ r_T * Σ_m ∇_φ log P(z_{t_m} | s_{t_m}) ]

Variance ∝ M * |Z|  (M = number of abstract actions, |Z| = latent dim)
```

Since M << T and |Z| << |A|, internal RL has dramatically lower variance!

---

## 5. Our Adaptation for Code Generation

### 5.1 Mapping Concepts to Code

| Paper Concept | Code Generation Equivalent |
|--------------|---------------------------|
| Observation (o_t) | Problem description + code so far |
| Action (a_t) | Next token |
| Abstract action (z_t) | Code pattern (e.g., "write loop", "define base case") |
| Subgoal | Complete a logical unit (function signature, if-block, etc.) |
| Episode | Generate complete solution |
| Reward | Binary: passes all tests or not |

### 5.2 Expected Abstract Actions for Our Problems

For our 6 problem types, we expect the metacontroller to discover patterns like:

**Fibonacci**:
- z1: "write base case check"
- z2: "write recursive return"

**Binary Search**:
- z1: "initialize pointers"
- z2: "write while loop header"
- z3: "write comparison and update"
- z4: "write return"

**Coin Change (DP)**:
- z1: "initialize DP array"
- z2: "write nested loops"
- z3: "write transition"
- z4: "write return"

### 5.3 Architecture Adaptation for Qwen 0.5B

```
Qwen 0.5B has:
- 24 layers
- Hidden dim: 896
- We'll insert controller at layer 12 (mid-depth)

Metacontroller dimensions:
- Latent code dim (n_z): 16
- GRU hidden dim (n_h): 64
- Sequence embedding dim (n_s): 64
- Controller rank: 16 (low-rank U)
```

### 5.4 Training Data

We'll use our existing expert solutions as the pretraining data:
- Each solution is a trajectory
- No explicit abstract action labels needed
- The metacontroller will discover the structure

---

## 6. Implementation Plan

### Phase 1: Data Preparation
```python
# Collect expert trajectories
# Format: (problem, solution_tokens, test_results)
# We already have this from experiments 15-16
```

### Phase 2: Metacontroller Training
```python
# 1. Load frozen Qwen 0.5B
# 2. Initialize metacontroller components:
#    - GRU for history
#    - SSM for sequence embedding
#    - MLP for encoder (μ, Σ)
#    - MLP for decoder (z → U)
#    - MLP for switching unit (→ β)
# 3. Train on ELBO objective
```

### Phase 3: Internal RL
```python
# 1. Load trained metacontroller
# 2. Freeze everything except abstract action policy
# 3. Define internal environment:
#    - State: residual stream e_t
#    - Action: latent code z
#    - Step: run until β > threshold
# 4. Train with GRPO on abstract actions
```

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Controller layer | 12 | Mid-depth for best controllability |
| Latent dim (n_z) | 16 | Small but expressive |
| Controller rank | 16 | Match LoRA rank from exp 16 |
| KL weight (α) | 0.1 | Balance reconstruction vs structure |
| β threshold | 0.5 | Binary switching during RL |
| Learning rate (meta) | 1e-3 | Standard for small networks |
| Learning rate (RL) | 3e-5 | Conservative for stability |

---

## 7. Expected Results

### Comparison with Previous Experiments

| Metric | Exp 15 (GRPO) | Exp 16 (M-GRPO) | Exp 17 (Internal RL) |
|--------|---------------|-----------------|----------------------|
| Exploration | Token-level | Token-level | Abstract-level |
| Effective horizon | ~100 tokens | ~100 tokens | ~5 abstract actions |
| Credit assignment | Very hard | Hard | Tractable |
| Expected accuracy | 10% | 50% | 70%+ |

### Success Criteria

1. **Metacontroller Training**: Switching patterns (β) should align with logical code boundaries
2. **Abstract Action Quality**: Latent codes should cluster by code pattern type
3. **RL Performance**: Higher accuracy than Exp 16, especially on hard problems (Edit Distance, RPN)

### Potential Challenges

1. **Code is more variable than navigation**: Unlike grid worlds, code has many valid solutions
2. **Longer sequences**: Our solutions are 50-200 tokens vs 100-160 in the paper
3. **Discrete actions**: The paper used continuous actions for the ant; we have discrete tokens

### Mitigation Strategies

1. Train on diverse expert solutions (not just one per problem)
2. Use larger latent dimension if needed
3. May need to adjust switching prior for longer sequences

---

## Files

- `README.md` - This documentation
- `train_metacontroller.py` - Phase 2 training script
- `internal_rl.py` - Phase 3 RL training script
- `metacontroller.py` - Metacontroller architecture
- `models/` - Saved checkpoints

---

## References

1. Kobayashi et al. (2025). "Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning." arXiv:2512.20605v2

2. Related work:
   - Options framework (Sutton et al., 1999)
   - CompILE (Kipf et al., 2019)
   - LoRA (Hu et al., 2022)
   - GRPO (DeepSeek-R1, 2025)

---

## Changelog

- 2024-12-28: Initial experiment design based on paper analysis
