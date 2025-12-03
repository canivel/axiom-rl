# Experiment 01: Cold Start Mechanics Analysis

## 1. Introduction: The "Reasoning Prior"

In the DeepSeek-R1 paradigm, the "Cold Start" phase is not merely about data augmentation; it is about **Behavioral Cloning (BC)** of a specific cognitive process. Before we can use Reinforcement Learning (RL) to optimize *how well* a model thinks, we must first teach it *how* to think.

This document analyzes the implementation of this phase in `axiom-rl`, specifically focusing on `axiom/coldstart/trace_generator.py` and the associated data recipes.

## 2. The Synthetic Data Engine (`trace_generator.py`)

The core of the Cold Start phase is the `ReasoningTraceGenerator` class. Its purpose is to bridge the gap between a generic instruction-following model and a specialized reasoning agent.

### 2.1 The "Long CoT" Prompt Strategy

The generator uses a specific prompt structure (found in `GeminiClient._build_prompt`) to force the teacher model (Gemini 2.5 Flash) into a "System 2" thinking mode.

**Code Analysis (`gemini_client.py`):**
```python
    def _build_prompt(self, ...):
        return f"""...
You MUST:
1. First, show your reasoning process inside <think> tags
2. Then, provide the complete Python code inside ```python``` blocks

## Required Format
<think>
Step 1: [Understand the problem...]
Step 2: [Consider edge cases]
Step 3: [Plan the approach...]
...
</think>
..."""
```

**Why this matters:**
-   **Explicit Structure:** It doesn't just ask for the answer; it mandates a specific XML-like schema (`<think>...</think>`). This is crucial because the subsequent RL phase (GRPO) will use the presence of these tags as a hard constraint.
-   **Step-by-Step Scaffolding:** The prompt explicitly lists steps (Understand, Edge Cases, Plan, Complexity). This acts as a "cognitive scaffold," ensuring the teacher model generates high-quality, dense reasoning traces rather than skipping straight to the code.

### 2.2 Rejection Sampling Implementation

The generator implements a **Rejection Sampling** loop to ensure data quality.

**Code Analysis (`trace_generator.py`):**
```python
        for i in range(num_traces):
            try:
                # 1. Generate
                response = self.client.generate_reasoning_trace(...)
                
                # 2. Verify
                verification = self.harness.verify(
                    solution_code=response.code,
                    problem=problem,
                )
                
                # 3. Filter
                is_verified = verification.status == VerificationStatus.PASSED
                
                # ... (Only verified traces are eventually saved/used)
```

**Why this matters:**
-   **Ground Truth Guarantee:** Unlike standard text generation where "quality" is subjective, here we have an objective oracle (the `TestHarness`). We only train on reasoning paths that *actually lead to the correct solution*.
-   **Diversity:** By looping `num_traces` times (default 5-10), we capture multiple valid ways to solve the same problem. This prevents the model from overfitting to a single "correct" reasoning path.

### 2.3 Missing Features & Improvements

Based on the audit, there are a few areas where the current implementation could be aligned closer to the DeepSeek-R1 specification:

1.  **Temperature Control:** The current `GeminiClient` uses the default temperature. DeepSeek-R1 suggests using a temperature between **0.5 - 0.7** to encourage diversity in reasoning paths.
    *   *Action Item:* Add `temperature` parameter to `GeminiClient` initialization.
2.  **Language Consistency:** There is no explicit check to ensure the `<think>` block is in the same language as the problem.
    *   *Action Item:* Add a lightweight language detection check in the verification loop.

## 3. The Data Recipe (`phase0-cold-start.md`)

The documentation reveals the composition of the Cold Start dataset.

### 3.1 Data Composition
The current recipe focuses heavily on **Algorithmic Reasoning** (Python coding problems).

*   **Input:** `problems.json` (likely LeetCode-style problems).
*   **Output:** `teacher_traces.jsonl` containing `(Problem, Thinking_Trace, Solution_Code)`.

**Critique:**
To fully replicate DeepSeek-R1, we need to expand beyond just coding. The "Reasoning Prior" should also include:
*   **Mathematics:** LaTeX-heavy derivations.
*   **Logic Puzzles:** Pure logic tasks.

### 3.2 The "Thinking" Field
The data loader (`axiom/trainer/data.py`) has been modified to handle the `thinking` field. This is a critical architectural change.

*   **Standard SFT:** `Input -> Output`
*   **Reasoning SFT:** `Input -> <think>Reasoning</think> -> Output`

This effectively changes the training objective from "predict the code" to "predict the thought process *and* the code".

## 4. Conclusion

The `axiom-rl` Cold Start implementation is a solid foundation. It correctly implements the **Generate -> Verify -> Filter** loop required for reasoning models.

**Key Takeaways for Research:**
1.  **Quality over Quantity:** The Rejection Sampling step ensures we don't train on hallucinations.
2.  **Structure is Key:** The strict enforcement of `<think>` tags is the most important feature, as it defines the action space for the future RL agent.
3.  **Expansion Needed:** To achieve general reasoning, we must expand the problem set beyond Python coding tasks.
