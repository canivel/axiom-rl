# Experiment 02: GRPO Implementation Architecture

**Date:** 2025-12-03
**Status:** Planning

## 1. Overview

This experiment aims to implement **Group Relative Policy Optimization (GRPO)**, the core reinforcement learning algorithm used in DeepSeek-R1. This replaces the standard PPO algorithm, eliminating the need for a Value Network (Critic) and reducing memory usage by ~50%.

## 2. Mathematical Formulation

### 2.1 The Objective
$$ \mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(o|q)} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right) \right] $$

### 2.2 Advantage Estimation
Instead of $A(s,a) = Q(s,a) - V(s)$, we use the group statistics:
$$ A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\}) + \epsilon} $$

## 3. Class Architecture

We will implement a new trainer class `GRPOTrainer` in `axiom/trainer/grpo_trainer.py`.

### 3.1 `GRPOTrainer`
*   **Inheritance:** `transformers.Trainer` (or custom loop if TRL is not flexible enough).
*   **Components:**
    *   `policy_model`: The model being trained (LoRA).
    *   `ref_model`: The frozen reference model (SFT checkpoint).
    *   `reward_function`: A callable that takes `(prompts, completions)` and returns scalar rewards.

### 3.2 The Training Loop (Step-by-Step)

1.  **Rollout (Experience Collection):**
    *   Sample a batch of prompts $Q$ (size $B$).
    *   For each prompt, generate $G$ completions using `model.generate()`.
    *   Total batch size for forward pass: $B \times G$.
    *   *Optimization:* Use `vLLM` for generation if possible, otherwise standard HF `generate`.

2.  **Reward Calculation:**
    *   Decode all $B \times G$ completions.
    *   Pass them to the `Verifier` (via `reward_function`).
    *   Get rewards tensor $R$ of shape $(B, G)$.

3.  **Advantage Computation:**
    *   Compute mean and std along dimension 1 (Group dimension).
    *   Normalize rewards to get Advantages $A$.

4.  **Loss Computation:**
    *   Forward pass on `policy_model` to get log-probs of the generated tokens.
    *   Forward pass on `ref_model` to get ref log-probs (for KL).
    *   Compute GRPO loss using the formula above.
    *   Backpropagate.

## 4. Integration Plan

### 4.1 Verifier Integration
We need to update `axiom/verifier` to expose a `batch_verify` method that accepts a list of `(prompt, completion)` tuples and returns a list of scores.

### 4.2 Configuration
New config class `GRPOConfig`:
*   `num_generations`: $G$ (default 16 or 64).
*   `beta`: KL coefficient (default 0.04).
*   `epsilon`: Clip range (default 0.2).

## 5. Implementation Steps

1.  **Step 1:** Create `GRPOConfig` and `GRPOTrainer` skeleton.
2.  **Step 2:** Implement the `rollout` method (generation).
3.  **Step 3:** Implement the `compute_loss` method (GRPO math).
4.  **Step 4:** Integrate with `TestHarness` for rewards.
5.  **Step 5:** Run a dummy training loop on a single batch.
