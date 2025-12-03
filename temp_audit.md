# **Axiom-RL: Technical Audit & Research Strategy**

Project: axiom-rl  
Goal: Investigate the "Straight Shot" to AGI via verifiable inference-time compute.  
Current Status: Phase 0 (Cold Start) & Phase 1 (Verifier) Implemented.

## ---

**1\. Strategic Alignment: The "Straight Shot" Hypothesis**

Your project is correctly positioned at the intersection of **System 2 Reasoning** and **Verifiable Reinforcement Learning**. By implementing a closed loop where the model learns from *verifiable* interaction (Python execution) rather than human labels, you are building the engine for the "Straight Shot" described by Ilya Sutskever.

**The Core Thesis:**

*If a model can verify its own solutions (Truth), and we scale the time it spends generating those solutions (Inference Compute), it can self-improve indefinitely without new human data.*

**Verdict:** Your architecture (Generator \-\> Verifier \-\> Trainer) is the correct minimal viable loop to test this. The inclusion of **Phase 0 (Cold Start)** is the critical enabler that prevents the "collapse" seen in naive RL attempts.

## ---

**2\. Repo & Implementation Audit**

I have reviewed the structure and components of axiom-rl based on the provided documentation and standard research practices.

### **✅ Phase 0: Cold Start (trace\_generator.py)**

Status: Correct & Critical.  
You are generating synthetic "reasoning traces" to prime the model. This is the "bootloader" for intelligence.

* **Why it works:** A random model cannot reason. It needs a "template" of what reasoning looks like (e.g., \<think\>... code...\</think\>). Your trace\_generator.py likely uses a stronger model (or heuristics) to create this initial distribution.  
* **Research Risk:** If your traces are too homogeneous (e.g., always solve problems the same way), the model will overfit to the *style* of the reasoning rather than the *substance*.  
* **Fix:** Ensure your trace\_generator varies the **length** and **method** of reasoning.

### **✅ Phase 1: The Verifier (verifier.py)**

Status: The Foundation of Truth.  
Using a Python interpreter as the Reward Function ($R=1$ if passes tests, $R=0$ if fails) is the scientifically correct choice for this research. It provides a noise-free gradient signal.

* **Critique (Safety):** In a "Straight Shot" scenario, the model explores millions of paths. It *will* accidentally (or intentionally) write malicious code (infinite loops, fork bombs).  
* **Requirement:** Ensure verifier.py uses multiprocessing with strict **timeouts** and memory limits. For the next level, you *must* use a containerized sandbox (like nsjail or gVisor) or the model will eventually crash your training node.

### **⚖️ Phase 2: The Trainer (GRPO & Replay)**

Status: The Engine of Improvement.  
You are moving to the training loop.

* **Algorithm Choice:** You must avoid PPO (Proximal Policy Optimization) if possible. It requires a "Critic" model which doubles memory usage.  
* **Recommendation:** Implement **GRPO (Group Relative Policy Optimization)**.  
  * *How:* Sample $G=16$ outputs for one problem.  
  * *Advantage:* $(Reward\_i \- Mean(Rewards\_{group})) / StdDev(Rewards\_{group})$.  
  * *Why:* This is mathematically equivalent to a Critic but uses the "group average" as the baseline. It saves 50% VRAM and is more stable for reasoning.

## ---

**3\. "What Am I Doing Wrong?" (Constructive Critique)**

Based on the "Straight Shot" goal, here are the likely pitfalls in the current setup:

### **1\. The "Memorization" Trap**

Issue: If you train on 50 LeetCode problems, the model will just memorize the code for those 50 problems. It won't learn reasoning; it will learn retrieval.  
Fix: You need Procedural Data.  
Instead of static datasets, write a generator that creates infinite variations of a problem type.

* *Example:* Don't train on "Calculate 2+2". Train on a script that generates random RPN expressions (3 4 \+, 5 2 /, etc.). The model can never memorize the answer; it *must* learn the algorithm.

### **2\. Missing "Aha\!" Metrics**

Issue: Standard "Loss" curves don't show reasoning.  
Fix: You need to track Pass@1 (correctness) vs. Reasoning Length (tokens).

* *Hypothesis:* As the model improves, does it "think" longer? Or does it find shortcuts? The "Straight Shot" hypothesis implies that *harder* problems should induce *longer* thinking times.

### **3\. Catastrophic Forgetting**

Issue: When the model learns to solve Hard Problem B, it forgets Easy Problem A.  
Fix: Your training loop needs a Prioritized Replay Buffer.

* Mix 20% "Cold Start" data (perfect formatting) \+ 40% "Old Successes" \+ 40% "New Discoveries" in every batch. This anchors the model's behavior.

## ---

**4\. The "Straight Shot" Research Roadmap**

To publish this or make it a serious contribution, follow this execution order:

### **🧪 Step 1: The "Grokking" Experiment (The Proof)**

Prove that your loop creates intelligence, not just memorization.

1. **Task:** Modular Arithmetic or Reverse Polish Notation (RPN).  
2. **Setup:** Train a small model (1B or 0.5B) using your axiom-rl loop.  
3. **The Signal:** Watch the **Test Accuracy**. It will stay flat (0%) for a long time, then suddenly spike (100%). This "Phase Transition" is the model *grokking* the underlying logic.  
4. **Why:** This proves the "Straight Shot" works on a toy model.

### **🔄 Step 2: Self-Correction Loops**

Prove that "Thinking" helps.

1. **Modify verifier.py:** If the code fails, return the *stderr* (error message) to the model.  
2. **New Prompt:** "You failed with error X. Think about why and fix it."  
3. **Measure:** Does specific "Correction" training improve the Pass@1 rate more than just generating more samples?

### **🚀 Step 3: The Curriculum (Scaling)**

1. Start with Easy problems (e.g., 2+2).  
2. Once Pass@1 \> 90%, unlock Medium (e.g., 2\*x \+ 5 \= 15).  
3. Once Pass@1 \> 90%, unlock Hard.  
* *Goal:* Show that the model *cannot* solve Hard problems without first mastering Easy ones (Dependency Learning).

## ---

**5\. Suggested Git Repository Updates**

To align with this research plan, I recommend adding these folders to your repo:

axiom-rl/  
├── axiom/  
│ ├── envs/ \# Procedural Environments (RPN, Arithmetic, Logic)  
│ │ ├── rpn\_calc.py \# Generates infinite RPN problems  
│ │ └── logic\_puz.py \# Generates logic puzzles  
│ ├── trainer/  
│ │ └── grpo.py \# GRPO implementation (replaces standard PPO)  
│ └── verifier.py \# (Keep this, but add 'stderr' feedback)  
├── experiments/ \# reproducible research runs  
│ ├── 01\_grokking/ \# Configs to reproduce the Grokking phase transition  
│ └── 02\_self\_correct/ \# Configs for error-correction loops  
└── README.md \# Update to reflect the "Straight Shot" hypothesis

### **Summary for the User**

You are building a **Reasoning Engine**, not just a model trainer.

1. **Keep** Phase 0 (Cold Start). It is essential.  
2. **Switch** your Trainer to **GRPO** (Group Relative Policy Optimization) for stability and efficiency.  
3. **Focus** on **Procedural Data** (infinite generation) to prove the model is learning algorithms, not memorizing answers.

This path validates the Ilya Sutskever hypothesis: creating a system that turns *compute* (generation \+ verification) into *intelligence* (correct reasoning).