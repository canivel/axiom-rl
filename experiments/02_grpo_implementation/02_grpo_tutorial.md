# Tutorial: Understanding and Running GRPO in Axiom-RL

**Date:** 2025-12-03
**Experiment:** 02_grpo_implementation

Welcome to the deep dive into **Group Relative Policy Optimization (GRPO)**. This guide is designed to help you understand *exactly* what is happening under the hood of the new trainer we just built, and how to validate it yourself.

---

## 1. The Concept: Why GRPO?

In standard Reinforcement Learning (like PPO), we usually have two models:
1.  **The Actor:** The model trying to solve the problem.
2.  **The Critic:** A second model that judges how good the Actor's current state is.

**The Problem:** The Critic is huge (same size as the Actor). Training a 7B model with PPO requires enough VRAM for *two* 7B models. This is expensive and slow.

**The Solution (GRPO):**
Instead of a Critic, we use **Peer Pressure**.
1.  We ask the Actor to solve the *same problem* **16 times** (Group Size $G=16$).
2.  We score all 16 solutions using our Verifier (Python Interpreter).
3.  We calculate the **average score** of the group.
4.  Solutions *above average* get a positive boost. Solutions *below average* get suppressed.

This mathematically eliminates the need for a Critic, saving ~50% VRAM and making training much more stable for reasoning tasks.

---

## 2. The Code Structure

We implemented three files to make this happen. Here is what each one does:

### A. `axiom/trainer/grpo_config.py` (The Settings)
This defines the rules of the game.
*   `num_generations`: How many times to try each problem (The Group Size).
*   `beta`: The "Safety Anchor". It prevents the model from changing too drastically from its original training, ensuring it doesn't unlearn English while learning to code.

### B. `axiom/trainer/grpo_trainer.py` (The Engine)
This is the heart of the operation. It runs a 4-step loop:
1.  **Rollout:** "Try to solve this." (Generates 16 solutions).
2.  **Reward:** "Did it work?" (Runs the Verifier/Dummy Reward).
3.  **Advantage:** "How good was this solution compared to the others?" (Calculates Group Mean/Std).
4.  **Update:** "Learn from this." (Updates model weights).

### C. `scripts/test_grpo_trainer.py` (The Smoke Test)
This is a simplified test script.
*   **Data:** Instead of real problems, it uses fake prompts (`def add...`).
*   **Reward:** Instead of running Python code, it assigns random scores.
*   **Goal:** To prove that the *plumbing* works (Memory, Tensors, Math) without worrying about the *complexity* of the real Verifier yet.

---

## 3. How to Run It (Step-by-Step)

Now, you will run the engine yourself.

### Step 1: Prepare the Environment
Ensure you are in the root directory `f:\Research\axiom-rl` and your environment is active (using `uv`).

### Step 2: Execute the Test Script
Run the following command in your terminal:

```powershell
uv run python scripts/test_grpo_trainer.py
```

### Step 3: Watch the Output
You will see a sequence of events. Here is what they mean:

**1. Loading Model:**
```text
Loading model: Qwen/Qwen2.5-Coder-1.5B-Instruct
```
*   *What's happening:* The script is loading the base model into VRAM. Since it's a 1.5B model, it should take ~3-4GB of VRAM.

**2. LoRA Initialization:**
```text
trainable params: 18,448,384 || all params: 1,562,282,072 || trainable%: 1.18
```
*   *What's happening:* We are not retraining the whole brain (1.5B params). We are adding small "adapter" layers (18M params) to learn the new skills. This is **Low-Rank Adaptation (LoRA)**.

**3. The Training Loop:**
```text
Starting GRPO Training...
Epoch 0 | Step 1 | Loss: -0.0010 | Avg Reward: 0.2825
```
*   **Step 1:** The model just generated a batch of solutions.
*   **Avg Reward:** The average score of those solutions (random in this test).
*   **Loss:** The calculated gradient update.
    *   *Note:* In GRPO, the loss can look weird (negative or oscillating) because it depends on the *relative* advantage. The important thing is that it is **not NaN** (Not a Number) and not exploding to Infinity.

**4. Success:**
```text
Test passed!
```
*   *What's happening:* The script finished 1 epoch without crashing. The plumbing is solid.

---

## 4. Verification Checklist

When you run this, ask yourself:
1.  **Did it crash?** (OOM - Out of Memory is the most common risk).
2.  **Is the Loss a number?** (If it says `nan`, our math is wrong).
3.  **Did it take a few seconds?** (Generation is slow; if it finished instantly, it didn't generate anything).

## 5. Next Steps

Once you have validated this "Smoke Test", we are ready to swap out the `dummy_reward_function` for the **Real Verifier**. This will turn the random number game into actual Intelligence Reinforcement.
