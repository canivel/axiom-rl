# Experiment 01: Trainer Mechanics Analysis

## 1. Introduction: The Optimization Engine

The Trainer is responsible for updating the model's weights to minimize a loss function. In the context of DeepSeek-R1 replication, we have two distinct training phases:
1.  **Phase 1 (SFT):** Supervised Fine-Tuning on "Golden Data" (Teacher Traces).
2.  **Phase 2 (RL):** Reinforcement Learning via Group Relative Policy Optimization (GRPO).

This document analyzes the current implementation in `axiom/trainer` and identifies the architectural gap required to reach Phase 2.

## 2. Current Implementation: The SFT Trainer

The existing `LoRASFTTrainer` in `axiom/trainer/trainer.py` is a robust implementation of **Phase 1**.

### 2.1 Standard SFT Architecture
*   **Base Class:** Uses Hugging Face's `Trainer`.
*   **Technique:** LoRA (Low-Rank Adaptation) via `peft`. This allows us to fine-tune large models (7B+) on consumer hardware by freezing the main weights and training small adapter matrices.
*   **Objective:** Standard Causal Language Modeling (CLM) loss.
    $$ \mathcal{L}_{SFT} = -\sum \log P(token_t | context) $$

### 2.2 Cold Start Integration
Crucially, the `SFTDataset` in `data.py` has been adapted to support the **Reasoning Prior**.
*   **Data Format:** It accepts JSONL records with a `thinking` field.
*   **Prompt Construction:** It formats the input as:
    ```
    User: Problem...
    Assistant: <think>Reasoning...</think>
    ```
    This ensures the model learns to *always* output the thinking block before the code.

## 3. The Missing Link: GRPO Trainer

The current codebase **does not yet implement GRPO**. To achieve the goals of the Technical Audit, we must build a new trainer class, likely inheriting from `TRL` (Transformer Reinforcement Learning) or built from scratch.

### 3.1 Why SFT is Not Enough
SFT is "Behavioral Cloning." It teaches the model to *mimic* the teacher.
*   If the teacher is wrong, the student learns to be wrong.
*   The student cannot surpass the teacher.

RL (GRPO) allows the model to **explore**.
*   The model generates 64 different thoughts.
*   If one thought leads to a correct answer (verified by code execution), that specific thought is reinforced, *even if it's different from what the teacher would have done*.
*   This allows for **Self-Improvement**.

### 3.2 Required Architecture for GRPO

We need to implement a `GRPOTrainer` that performs the following loop:

1.  **Rollout (Experience Collection):**
    *   Take a batch of problems $Q$.
    *   For each $q \in Q$, sample $G$ outputs $\{o_1, ..., o_G\}$ from the current policy $\pi_\theta$.
    *   *Critical:* This requires high-throughput inference (vLLM integration recommended).

2.  **Evaluation:**
    *   Pass all $G \times |Q|$ outputs to the **Verifier**.
    *   Get rewards $r_i$.

3.  **Advantage Estimation:**
    *   Compute group mean and std: $\mu = \text{mean}(r_{1..G})$, $\sigma = \text{std}(r_{1..G})$.
    *   Compute advantage: $A_i = \frac{r_i - \mu}{\sigma}$.

4.  **Optimization:**
    *   Compute the GRPO loss:
        $$ \mathcal{L} = \mathbb{E} \left[ \min \left( \frac{\pi(o|q)}{\pi_{old}(o|q)} A, \text{clip}(...) A \right) - \beta D_{KL} \right] $$
    *   Update weights.

## 4. Replay Buffer Dynamics

The Technical Audit mentions a **Prioritized Replay Buffer**. This is currently missing.

*   **Current State:** The `SFTTrainer` reads from a static file (`solutions.jsonl`).
*   **Desired State:** The `GRPOTrainer` should maintain a dynamic buffer of "Best Found Solutions."
    *   When the model stumbles upon a correct answer for a hard problem, save it.
    *   Mix these "Gold Samples" into the training batch (Off-Policy SFT) to prevent catastrophic forgetting.

## 5. Conclusion

We have a solid **Phase 1 (SFT)** engine. The immediate research priority is to build the **Phase 2 (GRPO)** engine. This will require significant engineering, particularly in the **Rollout** phase, as generating 64 samples per problem is computationally expensive and requires efficient batching.
