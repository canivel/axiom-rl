# How to Run and Validate Axiom-RL Phases

This guide provides step-by-step instructions to run and validate each component of the Axiom-RL pipeline in isolation. It explains **what** happens, **why** it matters, and **how** to interpret the results.

**Prerequisites:**
- Ensure `uv` is installed.
- Ensure your `.env` file has a valid `GEMINI_API_KEY`.

---

## Phase 1: The Verifier (Ground Truth)
**Goal:** Ensure the sandbox correctly judges Python code against test cases.

### The Concept
The Verifier is the "Environment" in our Reinforcement Learning setup. It acts as the impartial judge. If the Verifier is buggy (e.g., accepts wrong answers), the model will learn to cheat. If it rejects right answers, the model will never learn. We must prove it is strict and accurate.

### How to Validate
We run a script that feeds two known solutions to the Verifier:
1.  A **Correct** solution (Hash Map implementation of Two Sum).
2.  An **Incorrect** solution (Returns `[0, 0]` blindly).

**Command:**
```powershell
uv run python scripts/test_verifier_isolation.py
```

**Expected Output & Analysis:**
```text
Testing Verifier on problem: Two Sum
Correct Solution: PASSED
Incorrect Solution: FAILED
```
*   **Why this makes sense:**
    *   `PASSED` for the correct code means the sandbox successfully executed the Python code, ran the test cases, and verified the output matched the expected result.
    *   `FAILED` for the incorrect code means the sandbox correctly identified that `[0, 0]` is not the answer for every input.
    *   **Conclusion:** The "Judge" is fair and functional.

---

## Phase 2: Cold Start (Data Generation)
**Goal:** Ensure the system can generate synthetic reasoning traces using the Teacher Model (Gemini).

### The Concept
A random model cannot "reason" from scratch. We need to "seed" it with examples of good reasoning. We use a smart model (Gemini) to generate "Thought Process" + "Code" pairs. This data will be the textbook our model studies before it tries to solve problems itself.

### How to Validate
We generate a single trace for the `two_sum` problem. This tests the API connection, the Prompt Engineering (forcing `<think>` tags), and the saving logic.

**Command:**
```powershell
uv run python scripts/generate_teacher_data.py --problems two_sum --traces-per-problem 1
```

**Expected Output & Analysis:**
```text
COLD START: Teacher Data Generation
...
[1/1] Two Sum (two_sum)
    Trace 1/1: VERIFIED
Teacher dataset generated!
Output: data/coldstart/teacher_traces.jsonl
```
*   **Why this makes sense:**
    *   `VERIFIED` means the Teacher (Gemini) didn't just hallucinate; it wrote code that actually passed the Verifier (Phase 1).
    *   The output file `teacher_traces.jsonl` contains the "Gold Standard" data: a problem, a step-by-step thought process, and a correct solution.
    *   **Conclusion:** We have a working "Data Factory" to create training material.

---

## Phase 3: SFT Training (Behavioral Cloning)
**Goal:** Train the model to mimic the format and reasoning style of the Cold Start data.

### The Concept
Before the model can *discover* new solutions (RL), it must learn *how to speak the language of reasoning*. SFT (Supervised Fine-Tuning) forces the model to predict the Teacher's tokens. It learns: "When I see a problem, I should open a `<think>` tag and break it down."

### How to Validate
We run a very short training run (1 epoch) on the data we just generated. We aren't trying to make it smart yet, just checking if the "School" (Trainer) is open.

**Command:**
```powershell
uv run python scripts/run_training.py --solutions data/coldstart/teacher_traces.jsonl --epochs 1 --output-dir models/test_sft
```

**Expected Output & Analysis:**
```text
Loading model...
Starting SFT Training...
Epoch 1 | Step 10 | Loss: 0.5432
Training complete!
```
*   **Why this makes sense:**
    *   `Loss: 0.5432` (or similar number) means the model is surprised by the data but learning. If it were `0.0`, it memorized it instantly (bad). If `NaN`, it crashed.
    *   **Conclusion:** The training pipeline works. The model can ingest our data and update its weights.

---

## Phase 4: GRPO Training (Self-Improvement)
**Goal:** Run the full Reinforcement Learning loop where the model generates solutions, the Verifier scores them, and the model updates.

### The Concept
This is the "Straight Shot" engine. The model is no longer copying a teacher.
1.  **Rollout:** It tries 16 different ways to solve a problem.
2.  **Verify:** The Verifier scores them (1.0 or 0.0).
3.  **Update:** The math (GRPO) makes the winning thoughts more likely and the losing thoughts less likely.

### How to Validate
We run the integration test script. This uses the **Real Verifier** (Phase 1) inside the **GRPO Loop**.

**Command:**
```powershell
uv run python scripts/test_grpo_verifier.py
```

**Expected Output & Analysis:**
```text
Testing GRPOTrainer with REAL Verifier...
Loaded problem: Two Sum
Starting GRPO Training...
Epoch 0 | Step 0 | Loss: 0.0000 | Avg Reward: 0.0000 (or 1.0000)
Test passed!
```
*   **Why this makes sense:**
    *   `Avg Reward: 0.0000` usually means the untrained model failed all 16 attempts (normal for a hard problem).
    *   `Avg Reward: 0.2500` would mean 4 out of 16 attempts passed.
    *   The fact that it runs without crashing proves that the **Model** is talking to the **Verifier** and learning from the result.
    *   **Conclusion:** The Self-Improvement Engine is operational.

---

## Appendix A: `scripts/test_verifier_isolation.py`

(Script content as previously defined)
