# Experiment 01: Cold Start & Architecture Analysis

**Date:** 2025-12-03
**Status:** Complete

## Overview
This experiment analyzes the current state of the `axiom-rl` repository against the "Technical Audit" and the DeepSeek-R1 paradigm. The goal was to understand the existing "Cold Start" mechanics and identify the gaps required to reach full Reasoning Reinforcement Learning.

## Documents

1.  **[01_cold_start_mechanics.md](./01_cold_start_mechanics.md)**
    *   **Topic:** Phase 0 (Data Generation).
    *   **Key Finding:** The `trace_generator.py` correctly implements "Long CoT" seeding via Teacher Distillation (Gemini).
    *   **Gap:** Needs temperature control and broader problem diversity.

2.  **[02_verifier_mechanics.md](./02_verifier_mechanics.md)**
    *   **Topic:** The Judge System.
    *   **Key Finding:** The current Verifier (`harness.py`) is a binary (Pass/Fail) system designed for SFT.
    *   **Gap:** Needs to be upgraded to a **Scalar Reward Engine** (Format + Accuracy + Safety) for GRPO.

3.  **[03_trainer_mechanics.md](./03_trainer_mechanics.md)**
    *   **Topic:** The Optimization Engine.
    *   **Key Finding:** The repository contains a robust **SFT Trainer** (Phase 1) but **NO GRPO Trainer** (Phase 2).
    *   **Gap:** The entire GRPO loop (Rollout -> Advantage -> Update) needs to be built.

4.  **[04_systemic_emergence.md](./04_systemic_emergence.md)**
    *   **Topic:** Safety & Dynamics.
    *   **Key Finding:** "Length Extension" is seeded in Phase 0.
    *   **Gap:** We need KL penalties and Format Rewards to prevent "Reward Hacking" in Phase 2.

5.  **[05_cold_start_evaluation.md](./05_cold_start_evaluation.md)**
    *   **Topic:** Necessity & Readiness Check.
    *   **Verdict:** Cold Start is **MANDATORY**.
    *   **Action Items:**
        1.  Implement **Procedural Data Generation** (to prevent memorization).
        2.  Update `trace_generator.py` to use **Temperature=0.7** (to prevent mode collapse).

## Conclusion
The repository is currently at **Phase 1 (SFT)** maturity. The "Technical Audit" describes a future state (Phase 2) that we need to build.

**Next Steps:**
1.  Implement Procedural Data Generators (`axiom/procedural`).
2.  Update Cold Start Generator with Temperature control.
3.  Build the GRPO Trainer.
