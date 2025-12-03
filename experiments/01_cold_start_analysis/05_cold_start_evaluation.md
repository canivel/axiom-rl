# Experiment 01: Cold Start Evaluation & GRPO Readiness

**Date:** 2025-12-03
**Status:** Analysis Complete

## 1. The Core Question: "Do we really need a Cold Start?"

**Verdict: YES.**

Based on the "Technical Audit & Research Strategy" and the DeepSeek-R1 methodology, the Cold Start (Phase 0) is **critical**.

### Why?
1.  **The Bootloader Problem:** A standard base model (e.g., Qwen-Coder) has a near-zero probability of spontaneously generating the specific XML structure (`<think>...</think>`) and long-chain reasoning required by our Verifier.
2.  **Sparse Rewards:** Without this "Reasoning Prior," the RL agent (GRPO) would explore for millions of steps without ever hitting a positive reward (passing the test *and* having the correct format).
3.  **Collapse Prevention:** Naive RL without a behavioral clone anchor often collapses into gibberish or "reward hacking" (e.g., outputting the answer immediately without thinking).

## 2. Configuration Evaluation: Is it GRPO-Ready?

We analyzed `axiom/coldstart/trace_generator.py` against the requirements for a stable GRPO loop.

### 2.1 Format Consistency
*   **Requirement:** The SFT model must output `<think>` tags >95% of the time.
*   **Current State:** The `trace_generator.py` uses a rigid prompt template that enforces this structure.
*   **Verdict:** **READY**.

### 2.2 Diversity (The "Homogeneity Risk")
*   **Requirement:** The dataset must contain diverse reasoning paths (e.g., solving a problem via iteration vs. recursion) to prevent the model from overfitting to a single "style" of reasoning.
*   **Current State:** The `trace_generator.py` currently generates `num_traces` per problem, but it uses the *default temperature* of the Gemini client (likely 0.0 or low).
*   **Risk:** This will lead to identical or near-identical traces for the same problem.
*   **Fix Required:** We must update `GeminiClient` to accept a `temperature` parameter and set it to **0.7** during generation.

### 2.3 Data Source (The "Memorization Trap")
*   **Requirement:** The model must learn *algorithms*, not *answers*.
*   **Current State:** The system uses static problems (likely LeetCode-style).
*   **Risk:** The model might memorize "Two Sum" instead of learning "Hash Map Logic."
*   **Fix Required:** We must implement **Procedural Data Generation** (Phase 5 in the Roadmap) to create infinite variations of problems (e.g., RPN expressions, Logic Puzzles) *before* running the full Cold Start generation.

## 3. Updated Action Plan

To ensure the Cold Start is properly configured for GRPO, we will execute the following steps:

### Step 1: Implement Procedural Generation (New Priority)
*   Create `axiom/procedural/` module.
*   Implement generators for:
    *   **Arithmetic:** `(a + b) * c` variations.
    *   **RPN:** Reverse Polish Notation calculators.
    *   **Logic:** Simple boolean logic puzzles.
*   *Why:* This guarantees that our Cold Start data is impossible to memorize.

### Step 2: Update Trace Generator
*   Modify `GeminiClient` to support `temperature`.
*   Update `trace_generator.py` to use `temperature=0.7`.
*   Integrate with `axiom/procedural` to generate traces for *procedural* problems, not just static ones.

### Step 3: Generate the "Golden Dataset"
*   Run the generator to create ~1000 procedural reasoning traces.
*   Verify that they pass the `TestHarness`.

### Step 4: Train Phase 0 Model
*   Run SFT on this dataset.
*   Validate that the resulting model outputs `<think>` tags on unseen problems.

## 4. Conclusion

The Cold Start is **necessary**, but the current implementation needs two upgrades to be **GRPO-Ready**:
1.  **High-Temperature Sampling** (for diversity).
2.  **Procedural Data Sources** (for generalization).

We will proceed to implement these upgrades before starting the GRPO implementation.
