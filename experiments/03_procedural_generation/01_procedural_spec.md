# Experiment 03: Procedural Data Generation

**Date:** 2025-12-03
**Status:** Planning

## 1. The Goal
To create infinite, verifiable training data that forces the model to learn *algorithms* rather than memorizing answers.

## 2. Why Procedural?
If we train on `two_sum`, the model learns "If I see 'Two Sum', output this code."
If we train on `(a + b) * c`, where a, b, c are random integers, the model *must* learn the logic of arithmetic and operator precedence. It cannot memorize the answer to `(342 + 12) * 9` because it has likely never seen it.

## 3. Problem Types
We will implement generators for the following domains:

### A. Arithmetic (The "Hello World" of Reasoning)
*   **Task:** Evaluate an arithmetic expression.
*   **Format:** `Calculate: (12 + 4) * 3 - 5`
*   **Verification:** `eval()` in Python.
*   **Reasoning:** Requires understanding precedence (PEMDAS).

### B. Reverse Polish Notation (RPN)
*   **Task:** Evaluate a postfix expression.
*   **Format:** `Evaluate RPN: 3 4 + 5 *`
*   **Verification:** Stack-based calculation.
*   **Reasoning:** Requires stack simulation (perfect for Chain-of-Thought).

### C. Logic Puzzles (Boolean Algebra)
*   **Task:** Evaluate a boolean expression.
*   **Format:** `True AND (False OR True)`
*   **Verification:** `eval()` in Python.

## 4. Implementation Plan

### Step 1: Create `axiom/procedural` module
*   `base.py`: Abstract base class for generators.
*   `arithmetic.py`: Generator for arithmetic expressions.
*   `rpn.py`: Generator for RPN.

### Step 2: Update `ProblemDataset`
*   Allow it to accept a "Generator" instead of just a static JSON file.
*   Implement `dataset.sample(n)` to get $n$ random problems.

### Step 3: Integration
*   Update `generate_teacher_data.py` to use procedural problems.
*   Update `GRPOTrainer` to sample fresh problems every epoch.

## 5. Success Criteria
*   We can generate 1000 unique Arithmetic problems.
*   The Verifier correctly validates the solutions.
*   The Teacher (Gemini) can solve them (to create the Cold Start data).
