# Experiment 01: Systemic Emergence & Failure Modes

## 1. Introduction: The Ghost in the Machine

When training reasoning models with Reinforcement Learning, we often observe "emergent behaviors"—capabilities or failure modes that were not explicitly programmed. This document analyzes how `axiom-rl` is designed to handle these phenomena, specifically focusing on the transition from Phase 1 (SFT) to Phase 2 (GRPO).

## 2. The "Aha Moment" (Length Extension)

**Phenomenon:**
DeepSeek-R1 researchers observed that as the model is reinforced for correctness, it naturally learns to "think longer." The length of the `<think>` block increases, and the model starts double-checking its work.

**Current Support in Axiom-RL:**
*   **Phase 0 (Seeding):** We explicitly seed this behavior using the "Long CoT" prompts in `trace_generator.py`. By forcing the teacher (Gemini) to output verbose steps, we initialize the student model with a bias towards length.
*   **Phase 2 (Reinforcement):** The missing GRPO trainer is expected to amplify this. If "thinking longer" correlates with "higher accuracy" (which it does for math/code), the RL agent will naturally exploit this.

**Risk:**
If the model is *only* rewarded for length (e.g., if we naively add a `length_reward`), it will produce gibberish. We must rely on **Accuracy Rewards** as the primary driver. Length should be a side effect, not a target.

## 3. Failure Mode: Reward Hacking

**Phenomenon:**
The RL agent finds a loophole in the Verifier to get high rewards without actually solving the problem.

**Potential Hacks & Defenses:**

| Hack | Description | Defense Strategy |
| :--- | :--- | :--- |
| **The "Hardcode" Hack** | Model memorizes the answer (e.g., "42") and outputs it without reasoning. | **Format Penalty:** The Verifier must strictly enforce the presence of `<think>` tags. If the thinking is empty or too short, reward = 0. |
| **The "Language Mixing" Hack** | Model thinks in a different language (e.g., Chinese) but answers in English. | **Language Penalty:** Use `langdetect` on the `<think>` block. If language != target, apply a penalty. |
| **The "Infinite Loop" Hack** | Model generates endless thinking to maximize a (hypothetical) length reward. | **Time/Token Limit:** The `GenerationConfig` must set a hard `max_new_tokens` limit. |

## 4. Failure Mode: Distribution Shift

**Phenomenon:**
As the model optimizes for the specific problems in the training set (e.g., LeetCode), it loses its ability to chat normally or solve other types of problems (Catastrophic Forgetting).

**Current Defense (Phase 1):**
*   The `SFTTrainer` uses a static dataset, so the distribution is fixed.

**Required Defense (Phase 2):**
*   **KL Divergence Penalty:** The GRPO loss function includes a term $\beta D_{KL}(\pi || \pi_{ref})$. This forces the RL model to stay "close" to the SFT model (the Reference Model), preserving its general linguistic capabilities.
*   **Replay Buffer Mixing:** We must mix "General QA" data into the RL batches, even if we aren't optimizing them, just to maintain the weights.

## 5. Conclusion

The "Intelligence" of the system emerges from the tension between **Exploration** (trying new reasoning paths) and **Constraints** (Verifier rules).

*   **Too few constraints:** The model hacks the reward (gibberish).
*   **Too many constraints:** The model collapses to the SFT baseline (no improvement).

The art of tuning `axiom-rl` lies in balancing the **Temperature** (Exploration) against the **Verifier Penalties** (Constraints).
